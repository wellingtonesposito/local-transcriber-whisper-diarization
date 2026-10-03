"""FastAPI application: pages (Jinja + HTMX), JSON/API routes, SSE, uploads and exports."""
from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import zipfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import __version__, config
from ..core import audio, device
from ..core.model import Edits, Transcript
from ..core.pipeline import JobSettings
from ..core.render import ExportOptions, review_rows
from ..core.transforms import TransformOptions
from ..exporters import get_exporters, render as render_export
from .db import DB, new_id
from .events import EventBus
from .jobs import JobManager
from .runner import ProcessRunner
from .summary import transcript_summary

HERE = Path(__file__).parent
MEDIA_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aif", ".aiff", ".caf",
              ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".wmv", ".mpg", ".mpeg", ".3gp", ".ts", ".mts"}
VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".wmv", ".mpg", ".mpeg", ".3gp", ".ts", ".mts"}
ALLOWED_HOSTS = {"localhost", "127.0.0.1", "[::1]", "testserver"}
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
LANGUAGES = [("", "Auto-detect"), ("en", "English"), ("es", "Spanish"), ("pt", "Portuguese"), ("fr", "French"),
             ("de", "German"), ("it", "Italian"), ("nl", "Dutch"), ("pl", "Polish"), ("ru", "Russian"),
             ("uk", "Ukrainian"), ("tr", "Turkish"), ("ar", "Arabic"), ("hi", "Hindi"), ("zh", "Chinese"),
             ("ja", "Japanese"), ("ko", "Korean"), ("sv", "Swedish"), ("he", "Hebrew")]


# ------------------------------------------------------------------ helpers
def clean_settings(d: dict | None) -> dict:
    """Validate/coerce user-supplied job settings; unknown keys are dropped, defaults filled in."""
    d = d or {}
    base = JobSettings()

    def opt_int(key, lo=1, hi=20):
        """A missing key means "use the default"; a key that is present but blank/null means "automatic"."""
        if key not in d:
            return getattr(base, key)
        try:
            n = int(d[key])
        except (TypeError, ValueError):
            return None
        return n if lo <= n <= hi else None

    def boolean(v, default):
        return default if v is None else (v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes"))

    s = JobSettings(
        model_size=d.get("model_size") if d.get("model_size") in MODEL_SIZES else base.model_size,
        language=(d.get("language") or None) if (d.get("language") or "") in dict(LANGUAGES) else None,
        engine=d.get("engine") if d.get("engine") in ("auto", "mlx", "faster-whisper") else "auto",
        beam_size=opt_int("beam_size", 1, 10) or base.beam_size,
        vad=boolean(d.get("vad"), base.vad),
        initial_prompt=(str(d.get("initial_prompt") or "").strip()[:500] or None),
        preserve_disfluencies=boolean(d.get("preserve_disfluencies"), base.preserve_disfluencies),
        condition_on_previous_text=boolean(d.get("condition_on_previous_text"), base.condition_on_previous_text),
        diarize=boolean(d.get("diarize"), base.diarize),
        num_speakers=opt_int("num_speakers"),
        min_speakers=opt_int("min_speakers"),
        max_speakers=opt_int("max_speakers"),
    )
    return asdict(s)


def export_options(q) -> ExportOptions:
    """Build ExportOptions from query parameters (shared by single-file and zip export)."""
    def flag(name, default=False):
        v = q.get(name)
        return default if v is None else str(v).lower() in ("1", "true", "on", "yes")

    cleanup = q.get("cleanup", "verbatim")
    if cleanup not in ("verbatim", "clean", "custom"):
        raise HTTPException(400, "cleanup must be verbatim, clean or custom")
    tf = None
    if cleanup == "custom":
        tf = TransformOptions(fillers=flag("fillers"), phrase_fillers=flag("phrase_fillers"), stutters=flag("stutters"),
                              pause_markers=flag("pause_markers"), normalize_punct=flag("normalize_punct"),
                              spell=flag("spell"))
    try:
        precision = max(0, min(3, int(q.get("precision", 1))))
    except ValueError:
        precision = 1
    style = q.get("speaker_style", "prefix")
    merge = None if q.get("merge_turns") is None else flag("merge_turns")  # None = automatic
    return ExportOptions(cleanup=cleanup, transforms=tf, timestamps=flag("timestamps", True), ts_precision=precision,
                         speaker_style=style if style in ("prefix", "voice") else "prefix", merge_turns=merge)


def safe_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name.strip()
    name = re.sub(r"[\x00-\x1f]", "", name)
    return name[:200] or "file"


def content_disposition(filename: str) -> str:
    ascii_name = re.sub(r'[^A-Za-z0-9._ -]', "_", filename)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def fmt_duration(sec) -> str:
    if not sec:
        return "—"
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_hm(sec) -> str:
    """Compact total length: '1h 12m', '24m', '45s' or an em dash."""
    sec = int(round(sec or 0))
    if sec <= 0:
        return "—"
    h, r = divmod(sec, 3600)
    m = r // 60
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m" if m else f"{sec}s"


def fmt_size(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


# ------------------------------------------------------------------ app factory
def create_app(runner=None, data_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        d = data_path or config.data_dir()
        app.state.db = DB(d / "app.db")
        app.state.bus = EventBus()
        app.state.runner = runner or ProcessRunner(str(d))
        app.state.jobs = JobManager(app.state.db, app.state.bus, app.state.runner)
        app.state.jobs.start()
        yield
        app.state.jobs.stop()

    app = FastAPI(title="Transcriber", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.filters["duration"] = fmt_duration
    templates.env.filters["filesize"] = fmt_size
    templates.env.filters["hm"] = fmt_hm
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    # --- security: local-only, same-origin ----------------------------------------------
    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        hostname = host.split("]")[0] + "]" if host.startswith("[") else host.split(":")[0]
        if hostname not in ALLOWED_HOSTS:  # blocks DNS-rebinding
            return Response("Forbidden host", status_code=400)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != host:  # blocks cross-site form posts / fetches
                return Response("Cross-origin request blocked", status_code=403)
        return await call_next(request)

    def db() -> DB:
        return app.state.db

    def get_project_or_404(pid: str) -> dict:
        p = db().get_project(pid)
        if not p:
            raise HTTPException(404, "Project not found")
        return p

    def get_media_or_404(mid: str) -> dict:
        m = db().get_media(mid)
        if not m:
            raise HTTPException(404, "File not found")
        return m

    def load_transcript(m: dict) -> Transcript | None:
        f = config.media_dir(m["project_id"], m["id"]) / "transcript.json"
        return Transcript.from_dict(json.loads(f.read_text())) if f.exists() else None

    def load_edits(m: dict) -> Edits:
        f = config.media_dir(m["project_id"], m["id"]) / "edits.json"
        return Edits.from_dict(json.loads(f.read_text())) if f.exists() else Edits()

    def save_edits(m: dict, e: Edits) -> None:
        f = config.media_dir(m["project_id"], m["id"]) / "edits.json"
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(e.to_dict(), ensure_ascii=False))
        tmp.replace(f)

    def nav_ctx(ctx: dict) -> dict:
        proj = ctx.get("project")
        return {"nav_projects": db().list_projects(), "nav_active": proj["id"] if proj else None,
                "nav_page": ctx.get("nav_page", ""), "nav_sub": ctx.get("nav_sub", "")}

    def page(request: Request, name: str, **ctx):
        return templates.TemplateResponse(request, name, {"version": __version__, **nav_ctx(ctx), **ctx})

    # --- pages -----------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return page(request, "projects.html", projects=db().list_projects(), nav_page="projects")

    @app.post("/projects")
    async def create_project(request: Request):
        form = await request.form()
        name = str(form.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "Project name is required")
        p = db().create_project(name[:120], str(form.get("description") or "")[:500])
        config.project_dir(p["id"]).mkdir(parents=True, exist_ok=True)
        return RedirectResponse(f"/projects/{p['id']}", status_code=303)

    def project_ctx(p: dict) -> dict:
        settings = clean_settings(p["settings"])
        media = db().list_media(p["id"])
        pending = sum(m["status"] in ("uploaded", "failed", "cancelled") for m in media)
        done_ids = [m["id"] for m in media if m["status"] == "done"]
        return {"project": p, "media": media, "settings": settings, "pending": pending, "video_exts": VIDEO_EXTS,
                "done_ids": done_ids,
                "model_sizes": MODEL_SIZES, "languages": LANGUAGES, "exporters": get_exporters().values(),
                "has_token": config.has_hf_token(), "extensions": sorted(MEDIA_EXTS)}

    @app.get("/projects/{pid}", response_class=HTMLResponse)
    def project_page(request: Request, pid: str):
        return page(request, "project.html", nav_sub="recordings", **project_ctx(get_project_or_404(pid)))

    @app.get("/projects/{pid}/files", response_class=HTMLResponse)
    def files_fragment(request: Request, pid: str):
        return page(request, "_files.html", **project_ctx(get_project_or_404(pid)))

    def transcript_items(pid: str) -> list[dict]:
        """One entry per transcribed recording, with the facts the Transcripts screen shows."""
        items = []
        for m in db().list_media(pid):
            if m["status"] != "done":
                continue
            summary = json.loads(m["summary"]) if m["summary"] else None
            if not summary:  # transcripts made before summaries existed: build once, then remember
                try:
                    summary = transcript_summary(config.media_dir(pid, m["id"]) / "transcript.json")
                    db().update_media(m["id"], summary=json.dumps(summary))
                except Exception:
                    summary = {}
            edits = load_edits(m)
            n_edits = len(edits.segments)
            state = "reviewed" if m["reviewed"] else ("in_review" if (n_edits or edits.speakers) else "to_review")
            items.append({**m, "info": summary, "edits": n_edits, "state": state,
                          "speaker_names": [edits.speakers.get(x) or x for x in summary.get("speakers", [])]})
        return items

    def transcripts_ctx(p: dict) -> dict:
        items = transcript_items(p["id"])
        stats = {"total": len(items), "reviewed": sum(i["state"] == "reviewed" for i in items)}
        stats["to_review"] = stats["total"] - stats["reviewed"]
        return {"project": p, "items": items, "tstats": stats, "exporters": get_exporters().values(),
                "done_ids": [i["id"] for i in items], "nav_sub": "transcripts"}

    @app.get("/projects/{pid}/transcripts", response_class=HTMLResponse)
    def transcripts_page(request: Request, pid: str):
        return page(request, "transcripts.html", **transcripts_ctx(get_project_or_404(pid)))

    @app.get("/projects/{pid}/transcripts/list", response_class=HTMLResponse)
    def transcripts_fragment(request: Request, pid: str):
        return page(request, "_transcripts.html", **transcripts_ctx(get_project_or_404(pid)))

    @app.put("/api/media/{mid}/reviewed")
    async def set_reviewed(mid: str, request: Request):
        get_media_or_404(mid)
        reviewed = 1 if (await request.json()).get("reviewed") else 0
        db().update_media(mid, reviewed=reviewed)
        return {"reviewed": bool(reviewed)}

    @app.post("/projects/{pid}/delete")
    def delete_project(pid: str):
        get_project_or_404(pid)
        for m in db().list_media(pid):
            app.state.jobs.cancel(m["id"])
        db().delete_project(pid)
        shutil.rmtree(config.project_dir(pid), ignore_errors=True)
        return RedirectResponse("/", status_code=303)

    @app.get("/media/{mid}", response_class=HTMLResponse)
    def review_page(request: Request, mid: str):
        m = get_media_or_404(mid)
        if m["status"] != "done":
            return RedirectResponse(f"/projects/{m['project_id']}", status_code=303)
        return page(request, "review.html", media=m, project=get_project_or_404(m["project_id"]),
                    exporters=get_exporters().values(), is_video=m["ext"] in VIDEO_EXTS, nav_sub="transcripts")

    @app.get("/media/{mid}/file")
    def media_file(mid: str):
        m = get_media_or_404(mid)
        path = config.media_dir(m["project_id"], mid) / f"original{m['ext']}"
        if not path.exists():
            raise HTTPException(404, "Media file missing")
        return FileResponse(path)  # Starlette serves HTTP Range requests, so seeking works

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_page(request: Request):
        return page(request, "setup.html", nav_page="setup", **(await run_in_threadpool(setup_status)))

    def setup_status() -> dict:
        info = device.detect()
        try:
            ffmpeg = audio.ffmpeg_exe()
        except audio.AudioError:
            ffmpeg = None
        cached = False
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = isinstance(try_to_load_from_cache("pyannote/speaker-diarization-3.1", "config.yaml"), str)
        except Exception:
            pass
        return {"info": info, "ffmpeg": ffmpeg, "has_token": config.has_hf_token(), "diar_cached": cached,
                "data_dir": str(config.data_dir())}

    # --- JSON API: setup ---------------------------------------------------------------------
    @app.post("/api/setup/token")
    async def set_token(request: Request):
        token = str((await request.json()).get("token") or "").strip()
        if not token:
            config.clear_hf_token()
            return {"ok": True, "has_token": False}
        if not token.startswith("hf_") or len(token) < 10:
            raise HTTPException(400, "That does not look like a Hugging Face token (it should start with hf_).")
        config.save_hf_token(token)
        return {"ok": True, "has_token": True}

    @app.post("/api/setup/test-token")
    async def test_token():
        def check():
            token = config.get_hf_token()
            if not token:
                return {"ok": False, "message": "No token saved yet."}
            try:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
            except ImportError:
                return {"ok": False, "message": "huggingface_hub is not installed."}
            for repo in ("pyannote/speaker-diarization-3.1", "pyannote/segmentation-3.0"):
                try:
                    hf_hub_download(repo, "config.yaml", token=token)
                except GatedRepoError:
                    return {"ok": False, "message": f"Token is valid, but you have not accepted the terms for {repo}. "
                                                   f"Open https://huggingface.co/{repo} and accept them."}
                except (RepositoryNotFoundError, HfHubHTTPError) as e:
                    return {"ok": False, "message": f"Hugging Face rejected the request for {repo} ({type(e).__name__}). "
                                                   "Check that the token is valid and has read access."}
                except Exception as e:
                    return {"ok": False, "message": f"Could not reach Hugging Face: {e}"}
            return {"ok": True, "message": "Token works and both model agreements are accepted."}

        return await run_in_threadpool(check)

    # --- JSON API: projects / uploads / jobs ------------------------------------------------
    @app.put("/api/projects/{pid}/settings")
    async def save_settings(pid: str, request: Request):
        get_project_or_404(pid)
        s = clean_settings(await request.json())
        db().update_project(pid, settings=s)
        return s

    @app.post("/api/projects/{pid}/upload")
    async def upload(pid: str, request: Request, filename: str):
        """Raw-body upload (XHR `send(file)`): streamed straight to disk, no multipart temp copy."""
        get_project_or_404(pid)
        filename = safe_filename(filename)
        ext = Path(filename).suffix.lower()
        if ext not in MEDIA_EXTS:
            raise HTTPException(415, f"Unsupported file type '{ext or filename}'. Upload an audio or video file.")
        try:
            declared = int(request.headers.get("content-length") or 0)
        except ValueError:
            declared = 0
        mid = new_id()
        d = config.media_dir(pid, mid)
        d.mkdir(parents=True, exist_ok=True)
        if declared and shutil.disk_usage(d).free < declared * 2.5:  # original + extracted wav + headroom
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(507, "Not enough free disk space for this file.")
        dest = d / f"original{ext}"
        size = 0
        try:
            with open(dest, "wb") as f:
                async for chunk in request.stream():
                    await run_in_threadpool(f.write, chunk)
                    size += len(chunk)
            info = await run_in_threadpool(audio.probe, dest)
            if not info["has_audio"]:
                raise audio.AudioError(f"{filename} has no audio track.")
        except audio.AudioError as e:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(422, str(e))
        except BaseException:  # client disconnected mid-upload, etc.
            shutil.rmtree(d, ignore_errors=True)
            raise
        db().create_media(mid, pid, filename, ext)
        db().update_media(mid, size=size, duration=info["duration"])
        app.state.bus.publish({"type": "media", "media_id": mid, "project_id": pid, "status": "uploaded",
                               "stage": "", "progress": 0, "message": ""})
        return {"id": mid, "filename": filename}

    @app.post("/api/projects/{pid}/transcribe")
    async def transcribe_project(pid: str, request: Request):
        p = get_project_or_404(pid)
        body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
        settings = clean_settings(body.get("settings") or p["settings"])
        ids = body.get("media_ids")
        started = []
        for m in db().list_media(pid):
            if ids is not None and m["id"] not in ids:
                continue
            if ids is None and m["status"] not in ("uploaded", "failed", "cancelled"):
                continue
            if app.state.jobs.enqueue(m["id"], settings) is not None:
                if m["status"] == "done":  # re-run replaces the transcript, so edits and review state no longer apply
                    (config.media_dir(pid, m["id"]) / "edits.json").unlink(missing_ok=True)
                    db().update_media(m["id"], reviewed=0)
                started.append(m["id"])
        return {"queued": started}

    @app.post("/api/media/{mid}/cancel")
    def cancel_media(mid: str):
        get_media_or_404(mid)
        app.state.jobs.cancel(mid)
        return {"ok": True}

    @app.delete("/api/media/{mid}")
    def delete_media(mid: str):
        m = get_media_or_404(mid)
        app.state.jobs.cancel(mid)
        db().delete_media(mid)
        shutil.rmtree(config.media_dir(m["project_id"], mid), ignore_errors=True)
        return {"ok": True}

    # --- JSON API: transcript review ---------------------------------------------------------
    @app.get("/api/media/{mid}/transcript")
    def get_transcript(mid: str, request: Request):
        m = get_media_or_404(mid)
        t = load_transcript(m)
        if t is None:
            raise HTTPException(404, "No transcript yet")
        edits = load_edits(m)
        opts = export_options(request.query_params)
        speakers = [{"id": s, "name": edits.speakers.get(s) or t.speakers.get(s, {}).get("name") or ""}
                    for s in t.speaker_ids()]
        return {"media": {"id": mid, "filename": m["filename"], "duration": m["duration"], "ext": m["ext"],
                          "language": t.language, "engine": t.engine, "model": t.model},
                "speakers": speakers, "rows": review_rows(t, edits, opts)}

    @app.put("/api/media/{mid}/speakers")
    async def rename_speakers(mid: str, request: Request):
        m = get_media_or_404(mid)
        names = await request.json()
        e = load_edits(m)
        for k, v in names.items():
            v = str(v).strip()[:80]
            if v:
                e.speakers[k] = v
            else:
                e.speakers.pop(k, None)
        save_edits(m, e)
        return {"ok": True}

    @app.put("/api/media/{mid}/segments/{sid}")
    async def edit_segment(mid: str, sid: int, request: Request):
        m = get_media_or_404(mid)
        t = load_transcript(m)
        seg = next((s for s in (t.segments if t else []) if s.id == sid), None)
        if seg is None:
            raise HTTPException(404, "Segment not found")
        body = await request.json()
        e = load_edits(m)
        entry = e.segments.get(str(sid), {})
        if "text" in body:
            text = str(body["text"]).strip()
            if text and text != seg.text:
                entry["text"] = text
            else:
                entry.pop("text", None)
        if "deleted" in body:
            if body["deleted"]:
                entry["deleted"] = True
            else:
                entry.pop("deleted", None)
        if entry:
            e.segments[str(sid)] = entry
        else:
            e.segments.pop(str(sid), None)
        save_edits(m, e)
        return {"ok": True, "edited": "text" in entry, "deleted": bool(entry.get("deleted"))}

    # --- exports -----------------------------------------------------------------------------
    @app.get("/media/{mid}/export/{fmt}")
    def export_media(mid: str, fmt: str, request: Request):
        m = get_media_or_404(mid)
        if fmt not in get_exporters():
            raise HTTPException(404, "Unknown export format")
        t = load_transcript(m)
        if t is None:
            raise HTTPException(409, "This file has not been transcribed yet")
        data, ex = render_export(fmt, t, load_edits(m), export_options(request.query_params))
        name = f"{Path(m['filename']).stem}{ex.suffix}.{ex.ext}"
        return Response(data, media_type=ex.mime, headers={"Content-Disposition": content_disposition(name)})

    @app.get("/projects/{pid}/export.zip")
    def export_project(pid: str, request: Request):
        p = get_project_or_404(pid)
        formats = [f for f in (request.query_params.get("formats") or "vtt").split(",") if f in get_exporters()]
        if not formats:
            raise HTTPException(400, "Choose at least one format")
        opts = export_options(request.query_params)
        only = {i for i in (request.query_params.get("media") or "").split(",") if i}
        buf = io.BytesIO()
        n = 0
        stems: list[str] = []
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            used: set[str] = set()
            for m in db().list_media(pid):
                if only and m["id"] not in only:
                    continue
                t = load_transcript(m) if m["status"] == "done" else None
                if t is None:
                    continue
                edits = load_edits(m)
                stems.append(Path(m["filename"]).stem)
                for fmt in formats:
                    data, ex = render_export(fmt, t, edits, opts)
                    stem = Path(m["filename"]).stem + ex.suffix
                    arc, i = f"{stem}.{ex.ext}", 1
                    while arc in used:
                        i += 1
                        arc = f"{stem} ({i}).{ex.ext}"
                    used.add(arc)
                    z.writestr(arc, data)
                    n += 1
        if n == 0:
            raise HTTPException(409, "No transcribed files to export yet")
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": content_disposition(
                            f"{stems[0]}.zip" if len(stems) == 1 else f"{safe_filename(p['name'])}_transcripts.zip")})

    # --- live progress (SSE) ------------------------------------------------------------------
    @app.get("/events")
    async def events(request: Request):
        q = app.state.bus.subscribe()

        async def gen():
            try:
                yield "retry: 2000\n\n"
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), 15)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
                        continue
                    yield f"data: {json.dumps(ev)}\n\n"
            finally:
                app.state.bus.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith(("/api/", "/events")) or "application/json" in request.headers.get("accept", ""):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        html = templates.get_template("error.html").render(request=request, status=exc.status_code,
                                                           detail=exc.detail, version=__version__,
                                                           **nav_ctx({}))
        return HTMLResponse(html, status_code=exc.status_code)

    return app
