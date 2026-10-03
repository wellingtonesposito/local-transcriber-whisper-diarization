"""`transcriber` command line: serve the web app, run one file headlessly, or check the setup."""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from . import config


def cmd_serve(a: argparse.Namespace) -> int:
    import threading

    import uvicorn

    from .server.app import create_app

    url = f"http://{a.host}:{a.port}"
    print(f"Transcriber running at {url}  (data folder: {config.data_dir()})")
    print("Press Ctrl+C to stop.")
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    """Headless single-file run (what the old whisper_diarize_clean_export.py did)."""
    from .core.device import quiet_third_party_warnings
    from .core.pipeline import JobSettings, run_pipeline
    from .core.model import Edits
    from .core.render import ExportOptions
    from .exporters import get_exporters, render

    quiet_third_party_warnings()
    src = Path(a.media)
    if not src.exists():
        print(f"File not found: {src}", file=sys.stderr)
        return 2
    out_dir = Path(a.out_dir or "Local - Transcripts to revise")
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / ".work" / src.stem
    work.mkdir(parents=True, exist_ok=True)

    s = JobSettings(model_size=a.model_size, language=a.language, engine=a.engine, beam_size=a.beam_size, vad=a.vad,
                    diarize=not a.no_diarize, num_speakers=a.num_speakers, min_speakers=a.min_speakers,
                    max_speakers=a.max_speakers, max_gap=a.max_merge_gap, preserve_disfluencies=a.keep_disfluencies)

    last = [""]

    def progress(stage, frac, msg):
        line = f"[{stage}] {int(frac * 100):3d}% {msg}"
        if line != last[0]:
            print(line, flush=True)
            last[0] = line

    from .core.audio import AudioError
    from .core.diarize import DiarizationError
    from .core.engines import EngineUnavailable

    try:
        t = run_pipeline(src, work, s, progress, hf_token=config.get_hf_token(), display_name=src.name)
    except (DiarizationError, EngineUnavailable, AudioError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        if isinstance(e, DiarizationError):
            print("The transcription itself is saved; re-running will skip it. "
                  "Use --no-diarize to export without speaker labels.", file=sys.stderr)
        return 1

    edits = Edits()
    if a.rename_map:
        import json

        edits.speakers = json.loads(Path(a.rename_map).read_text(encoding="utf-8"))
    opts = ExportOptions(cleanup="clean" if a.clean else "verbatim")
    prefix = a.out_prefix or src.stem
    for fmt in a.formats.split(","):
        if fmt not in get_exporters():
            print(f"Unknown format '{fmt}'. Choose from: {', '.join(get_exporters())}", file=sys.stderr)
            return 2
        data, ex = render(fmt, t, edits, opts)
        path = out_dir / f"{prefix}{ex.suffix}.{ex.ext}"
        path.write_bytes(data)
        print("Saved", path)
    print("Speakers:", ", ".join(t.speaker_ids()))
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    from .core import audio, device

    info = device.detect()
    ok = True

    def line(label, value, good=True):
        nonlocal ok
        ok &= good
        print(f"{'✓' if good else '✗'} {label}: {value}")

    line("Platform", info["platform"])
    try:
        line("ffmpeg", audio.ffmpeg_exe())
    except audio.AudioError as e:
        line("ffmpeg", str(e), False)
    line("Engine (recommended)", info["recommended_engine"], info["mlx_whisper"] or info["faster_whisper"])
    line("faster-whisper", "installed" if info["faster_whisper"] else "missing", info["faster_whisper"] or info["mlx_whisper"])
    line("mlx-whisper", "installed" if info["mlx_whisper"] else "not installed (Apple Silicon only)", True)
    line("pyannote.audio", "installed" if info["pyannote"] else "missing (speaker identification unavailable)", info["pyannote"])
    line("GPU", "CUDA" if info["cuda"] else "MPS" if info["mps"] else "none (CPU)", True)
    line("Hugging Face token", "saved" if config.has_hf_token() else "not set (needed for speaker identification)", config.has_hf_token())
    line("Data folder", str(config.data_dir()))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="transcriber", description="Local transcription + diarization")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--host", default=config.DEFAULT_HOST, help="keep 127.0.0.1 unless you know what you are doing")
    s.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_serve)

    r = sub.add_parser("run", help="transcribe one file without the web app")
    r.add_argument("media")
    r.add_argument("--out-dir")
    r.add_argument("--out-prefix")
    r.add_argument("--formats", default="vtt", help="comma-separated: vtt,srt,txt,csv,json,docx,docx_table,pdf,md")
    r.add_argument("--model-size", default="small")
    r.add_argument("--language")
    r.add_argument("--engine", default="auto", choices=["auto", "mlx", "faster-whisper"])
    r.add_argument("--beam-size", type=int, default=5)
    r.add_argument("--vad", action="store_true")
    r.add_argument("--no-diarize", action="store_true", help="skip speaker identification (no token needed)")
    r.add_argument("--num-speakers", type=int)
    r.add_argument("--min-speakers", type=int)
    r.add_argument("--max-speakers", type=int)
    r.add_argument("--max-merge-gap", type=float, default=1.0)
    r.add_argument("--keep-disfluencies", action="store_true")
    r.add_argument("--clean", action="store_true", help="remove um/uh and stutters in the exported text")
    r.add_argument("--rename-map", help="JSON file mapping SPEAKER_00 -> name")
    r.set_defaults(fn=cmd_run)

    d = sub.add_parser("doctor", help="check the installation")
    d.set_defaults(fn=cmd_doctor)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
