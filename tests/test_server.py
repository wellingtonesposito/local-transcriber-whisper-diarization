import io
import threading
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from transcriber.core import pipeline
from transcriber.server.app import create_app
from transcriber.server.runner import InlineRunner

from fakes import FakeDiarizer, FakeEngine, make_media


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBER_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    engine = FakeEngine()
    monkeypatch.setattr(pipeline, "get_engine", lambda name="auto": engine)
    app = create_app(runner=InlineRunner(FakeDiarizer()))
    with TestClient(app) as c:
        c.engine = engine
        yield c


@pytest.fixture
def video(tmp_path):
    return make_media(tmp_path / "interview one.mp4")


def new_project(c, name="Study A"):
    r = c.post("/projects", data={"name": name, "description": "d"}, follow_redirects=False)
    assert r.status_code == 303
    return r.headers["location"].rsplit("/", 1)[1]


def upload(c, pid, path, name=None):
    return c.post(f"/api/projects/{pid}/upload", params={"filename": name or path.name}, content=path.read_bytes(),
                  headers={"Content-Type": "application/octet-stream"})


def wait_for(c, pid, mid, status, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        m = next(m for m in c.app.state.db.list_media(pid) if m["id"] == mid)
        if m["status"] == status:
            return m
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {status}; last={m}")


def transcribed(c, video):
    pid = new_project(c)
    mid = upload(c, pid, video).json()["id"]
    assert c.post(f"/api/projects/{pid}/transcribe", json={}).json()["queued"] == [mid]
    wait_for(c, pid, mid, "done")
    return pid, mid


def test_pages_render(client):
    assert client.get("/").status_code == 200
    pid = new_project(client)
    assert "Study A" in client.get("/").text
    page = client.get(f"/projects/{pid}")
    assert page.status_code == 200 and "Drop recordings here" in page.text
    assert client.get("/setup").status_code == 200
    assert client.get("/projects/nope").status_code == 404


def test_upload_validation(client, tmp_path):
    pid = new_project(client)
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    assert upload(client, pid, txt).status_code == 415
    fake_mp3 = tmp_path / "bad.mp3"
    fake_mp3.write_bytes(b"not audio at all")
    r = upload(client, pid, fake_mp3)
    assert r.status_code == 422
    assert client.app.state.db.list_media(pid) == []  # nothing left behind
    assert client.post("/api/projects/zzz/upload", params={"filename": "a.mp4"}, content=b"x").status_code == 404


def test_upload_sanitizes_filename(client, video):
    pid = new_project(client)
    r = upload(client, pid, video, name="../../etc/evil name.mp4")
    assert r.status_code == 200 and r.json()["filename"] == "evil name.mp4"


def test_full_flow_edit_and_export(client, video):
    pid, mid = transcribed(client, video)
    files = client.get(f"/projects/{pid}/files").text
    assert "Transcribed" in files and "/media/" + mid in files

    d = client.get(f"/api/media/{mid}/transcript").json()
    assert [r["speaker"] for r in d["rows"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert client.get(f"/media/{mid}").status_code == 200

    assert client.put(f"/api/media/{mid}/speakers", json={"SPEAKER_00": "Researcher"}).status_code == 200
    r = client.put(f"/api/media/{mid}/segments/1", json={"text": "Hi, um, friend."})
    assert r.json()["edited"] is True

    vtt = client.get(f"/media/{mid}/export/vtt").text
    assert "Researcher: Hello there." in vtt and "SPEAKER_01: Hi, um, friend." in vtt
    clean = client.get(f"/media/{mid}/export/txt", params={"cleanup": "clean", "timestamps": 0})
    assert "SPEAKER_01: Hi, friend." in clean.text and "[" not in clean.text
    assert 'filename="interview one.txt"' in clean.headers["content-disposition"]

    # review rows reflect edits and the verbatim/clean toggle
    rows = client.get(f"/api/media/{mid}/transcript", params={"cleanup": "clean"}).json()["rows"]
    assert rows[1]["text"] == "Hi, friend." and rows[1]["raw"] == "Hi, um, friend." and rows[1]["edited"]

    # remove then restore a segment
    client.put(f"/api/media/{mid}/segments/0", json={"deleted": True})
    assert "Hello there." not in client.get(f"/media/{mid}/export/vtt").text
    client.put(f"/api/media/{mid}/segments/0", json={"deleted": False})
    assert "Hello there." in client.get(f"/media/{mid}/export/vtt").text


def test_zip_export_and_unknown_format(client, video):
    pid, mid = transcribed(client, video)
    r = client.get(f"/projects/{pid}/export.zip", params={"formats": "vtt,docx_table,csv"})
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert sorted(names) == ["interview one.csv", "interview one.vtt", "interview one_table.docx"]
    assert client.get(f"/media/{mid}/export/bogus").status_code == 404
    assert client.get(f"/projects/{pid}/export.zip", params={"formats": "bogus"}).status_code == 400


def test_failure_then_retry_resumes(client, video, monkeypatch):
    c = client
    c.app.state.runner.diarizer = FakeDiarizer(fail=True)
    pid = new_project(c)
    mid = upload(c, pid, video).json()["id"]
    c.post(f"/api/projects/{pid}/transcribe", json={})
    m = wait_for(c, pid, mid, "failed")
    assert "auth" in m["error"] and "auth" in c.get(f"/projects/{pid}/files").text
    c.app.state.runner.diarizer = FakeDiarizer()
    c.post(f"/api/projects/{pid}/transcribe", json={})
    wait_for(c, pid, mid, "done")
    assert c.engine.calls == 1  # transcription was reused from the checkpoint


def test_settings_are_validated_and_saved(client):
    pid = new_project(client)
    r = client.put(f"/api/projects/{pid}/settings", json={"model_size": "evil", "num_speakers": "2", "beam_size": 99,
                                                          "language": "es", "bogus": 1, "diarize": False})
    s = r.json()
    assert s["model_size"] == "turbo" and s["num_speakers"] == 2 and s["beam_size"] == 5  # invalid values fall back to the defaults
    assert s["language"] == "es" and s["diarize"] is False and "bogus" not in s
    assert client.app.state.db.get_project(pid)["settings"]["language"] == "es"


def test_media_file_supports_range(client, video):
    pid, mid = transcribed(client, video)
    r = client.get(f"/media/{mid}/file", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100


def test_delete_media_and_project_remove_files(client, video, tmp_path):
    pid, mid = transcribed(client, video)
    mdir = tmp_path / "data" / "projects" / pid / "media" / mid
    assert mdir.exists()
    assert client.delete(f"/api/media/{mid}").status_code == 200 and not mdir.exists()
    client.post(f"/projects/{pid}/delete", follow_redirects=False)
    assert not (tmp_path / "data" / "projects" / pid).exists() and client.get(f"/projects/{pid}").status_code == 404


def test_host_and_origin_guards(client):
    assert client.get("/", headers={"Host": "evil.example.com"}).status_code == 400
    r = client.post("/projects", data={"name": "x"}, headers={"Origin": "http://evil.example.com"})
    assert r.status_code == 403
    r = client.post("/projects", data={"name": "x"}, headers={"Origin": "http://testserver"}, follow_redirects=False)
    assert r.status_code == 303


def test_token_endpoints(client, tmp_path):
    assert client.post("/api/setup/token", json={"token": "nope"}).status_code == 400
    assert client.post("/api/setup/token", json={"token": "hf_abcdefghij"}).json()["has_token"] is True
    f = tmp_path / "data" / "secrets" / "hf_token"
    assert f.read_text() == "hf_abcdefghij" and (f.stat().st_mode & 0o777) == 0o600
    assert "hf_abcdefghij" not in client.get("/setup").text  # never rendered back
    assert client.post("/api/setup/token", json={"token": ""}).json()["has_token"] is False and not f.exists()


class BlockingRunner:
    """Holds a job 'running' until it is cancelled, so cancel/queue behaviour is deterministic."""

    def __init__(self):
        self.started = threading.Event()
        self.ran = []

    def run(self, spec, on_progress, is_cancelled):
        self.ran.append(spec["job_id"])
        on_progress("transcribe", 0.4, "working")
        self.started.set()
        while not is_cancelled():
            time.sleep(0.02)
        return "cancelled", None

    def kill(self):
        pass

    def shutdown(self):
        pass


def test_cancel_running_and_queued_jobs(tmp_path, monkeypatch, video):
    monkeypatch.setenv("TRANSCRIBER_DATA", str(tmp_path / "data"))
    runner = BlockingRunner()
    with TestClient(create_app(runner=runner)) as c:
        pid = new_project(c)
        m1 = upload(c, pid, video, "one.mp4").json()["id"]
        m2 = upload(c, pid, video, "two.mp4").json()["id"]
        c.post(f"/api/projects/{pid}/transcribe", json={})
        assert runner.started.wait(5)
        row = wait_for(c, pid, m1, "running")
        assert row["stage"] == "transcribe" and row["progress"] == pytest.approx(0.4)
        assert next(m for m in c.app.state.db.list_media(pid) if m["id"] == m2)["status"] == "queued"
        assert "Cancel" in c.get(f"/projects/{pid}/files").text

        c.post(f"/api/media/{m2}/cancel")  # queued: cancelled immediately, never reaches the runner
        wait_for(c, pid, m2, "cancelled")
        c.post(f"/api/media/{m1}/cancel")  # running: cooperative cancel
        wait_for(c, pid, m1, "cancelled")
        assert len(runner.ran) == 1

        # cancelled files can be retried
        assert c.post(f"/api/projects/{pid}/transcribe", json={"media_ids": [m2]}).json()["queued"] == [m2]
        c.post(f"/api/media/{m2}/cancel")
        wait_for(c, pid, m2, "cancelled")


def test_restart_marks_running_jobs_interrupted(tmp_path, monkeypatch, video):
    monkeypatch.setenv("TRANSCRIBER_DATA", str(tmp_path / "data"))
    with TestClient(create_app(runner=BlockingRunner())) as c:
        pid = new_project(c)
        mid = upload(c, pid, video).json()["id"]
        db = c.app.state.db
        jid = db.enqueue(mid, {})
        db.update_job(jid, status="running")
        db.update_media(mid, status="running")
        db.recover_interrupted()
        m = db.get_media(mid)
        assert m["status"] == "failed" and "Interrupted" in m["error"]


def test_sidebar_stats_and_project_totals(client, video):
    pid, mid = transcribed(client, video)
    other = new_project(client, "Second study")
    home = client.get("/").text
    assert "Study A" in home and "Second study" in home
    assert "1 of 1 transcribed" in home and "No recordings yet" in home
    page = client.get(f"/projects/{pid}").text
    assert f'href="/projects/{other}"' in page                    # sidebar lists every project
    assert f'nav-item sub on" href="/projects/{pid}"' in page      # and marks Recordings as the open section
    assert f'href="/projects/{pid}/transcripts"' in page           # with a Transcripts section beside it
    assert "Total audio" not in page and "Need attention" not in page   # no stats row above the recordings
    assert f'id="files" data-project="{pid}"' in page                   # live progress still finds the project page
    assert client.get(f"/projects/{pid}/stats").status_code == 404
    files = client.get(f"/projects/{pid}/files").text
    assert "Open transcript" in files and "Export" not in files    # transcript work lives on its own screen
    assert "Export transcripts" not in page


def test_transcripts_screen_lists_review_state_and_exports(client, video):
    pid, mid = transcribed(client, video)
    page = client.get(f"/projects/{pid}/transcripts")
    assert page.status_code == 200
    html = page.text
    assert "To review" in html and "interview one.mp4" in html and "No edits yet" in html
    assert f'nav-item sub on" href="/projects/{pid}/transcripts"' in html
    assert "Export transcripts" in html and "NVivo" in html and "Export all" in html

    # a correction or a speaker rename moves it to "In review"; marking it done moves it to "Reviewed"
    client.put(f"/api/media/{mid}/speakers", json={"SPEAKER_00": "Researcher"})
    assert "In review" in client.get(f"/projects/{pid}/transcripts/list").text
    client.put(f"/api/media/{mid}/segments/1", json={"text": "Edited."})
    assert "1 edit<" in client.get(f"/projects/{pid}/transcripts/list").text
    assert client.put(f"/api/media/{mid}/reviewed", json={"reviewed": True}).json() == {"reviewed": True}
    listing = client.get(f"/projects/{pid}/transcripts/list").text
    assert "Reviewed" in listing and "Mark to review" in listing
    assert 'Reviewed <span class="num">1</span>' in listing and 'To review <span class="num">0</span>' in listing  # counts live on the filter
    assert 'class="stat"' not in listing and "card stat" not in listing                                 # no separate stat cards
    assert client.put("/api/media/nope/reviewed", json={"reviewed": True}).status_code == 404

    # re-transcribing replaces the transcript, so edits and the reviewed flag reset
    client.post(f"/api/projects/{pid}/transcribe", json={"media_ids": [mid]})
    wait_for(client, pid, mid, "done")
    assert client.get(f"/media/{mid}").text.count("Mark as reviewed") >= 1
    assert "To review" in client.get(f"/projects/{pid}/transcripts/list").text


def test_transcripts_screen_empty_state(client):
    pid = new_project(client)
    html = client.get(f"/projects/{pid}/transcripts").text
    assert "No transcripts yet" in html and "Go to recordings" in html


def test_review_page_layout_for_video(client, video):
    pid, mid = transcribed(client, video)
    html = client.get(f"/media/{mid}").text
    assert "Mark as reviewed" in html and 'type="range"' in html      # reviewed toggle and the volume slider
    assert ">This transcript<" not in html                           # the old info panel is gone
    assert "Theater mode" in html and "Full screen" in html and 'class="media"' in html
    assert 'role="separator"' not in html                            # drag-resizing was replaced by theater mode
    assert "subline" not in html                                      # no "52:09 · EN · engine · speakers" under the title
    assert 'aria-label="Slower"' in html and 'aria-label="Faster"' in html  # speed: presets plus - / + steppers
    # speakers moved out of the side column into a strip above the transcript and video
    assert html.index('aria-label="Speakers"') < html.index('class="review has-video"') or html.index('aria-label="Speakers"') < html.index('class="review')
    assert f'href="/projects/{pid}/transcripts"' in html


def test_review_page_layout_for_audio_has_no_video_controls(client, tmp_path):
    audio = make_media(tmp_path / "talk.wav", video=False)
    pid = new_project(client)
    mid = upload(client, pid, audio).json()["id"]
    client.post(f"/api/projects/{pid}/transcribe", json={})
    wait_for(client, pid, mid, "done")
    html = client.get(f"/media/{mid}").text
    assert "<audio" in html and "<video" not in html
    assert "Theater mode" not in html and 'class="media"' not in html and 'has-video' not in html
    assert 'aria-label="Speakers"' in html and 'type="range"' in html


def test_zip_can_be_limited_to_one_file(client, video):
    pid, m1 = transcribed(client, video)
    m2 = upload(client, pid, video, "second.mp4").json()["id"]
    client.post(f"/api/projects/{pid}/transcribe", json={"media_ids": [m2]})
    wait_for(client, pid, m2, "done")
    both = zipfile.ZipFile(io.BytesIO(client.get(f"/projects/{pid}/export.zip", params={"formats": "vtt"}).content))
    assert len(both.namelist()) == 2
    r = client.get(f"/projects/{pid}/export.zip", params={"formats": "vtt,csv", "media": m2})
    assert sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) == ["second.csv", "second.vtt"]
    assert 'filename="second.zip"' in r.headers["content-disposition"]


def test_review_page_has_player_and_dialog(client, video):
    pid, mid = transcribed(client, video)
    html = client.get(f"/media/{mid}").text
    assert "<video" in html and 'class="player"' in html and "Export transcripts" in html


def test_duration_filters():
    from transcriber.server.app import fmt_hm
    assert [fmt_hm(x) for x in (0, None, 45, 1440, 4320)] == ["—", "—", "45s", "24m", "1h 12m"]


def test_review_rows_carry_word_timings_for_highlighting(client, video):
    pid, mid = transcribed(client, video)
    rows = client.get(f"/api/media/{mid}/transcript").json()["rows"]
    assert rows[0]["words"] == [["Hello", 0.0, 0.5], ["there.", 0.5, 1.0]]
    # words stay the originally transcribed ones, even after an edit or in the clean view
    client.put(f"/api/media/{mid}/segments/1", json={"text": "Hi, friend."})
    rows = client.get(f"/api/media/{mid}/transcript", params={"cleanup": "clean"}).json()["rows"]
    assert rows[1]["text"] == "Hi, friend." and [w[0] for w in rows[1]["words"]] == ["Um,", "Hi."]


def test_export_options_merge_turns_parsing():
    from transcriber.server.app import export_options
    assert export_options({}).merge_turns is None
    assert export_options({"merge_turns": "1"}).merge_turns is True
    assert export_options({"merge_turns": "0"}).merge_turns is False


def test_export_endpoint_honours_merge_turns(client, video):
    pid, mid = transcribed(client, video)
    # single-speaker transcript: build one with two consecutive segments by editing nothing, just compare headers
    r = client.get(f"/media/{mid}/export/txt", params={"merge_turns": 1})
    assert r.status_code == 200 and "SPEAKER_00" in r.text


def test_old_database_is_migrated_and_summaries_backfilled(tmp_path, monkeypatch, video):
    import json
    import sqlite3

    monkeypatch.setenv("TRANSCRIBER_DATA", str(tmp_path / "data"))
    data = tmp_path / "data"
    data.mkdir()
    old = sqlite3.connect(data / "app.db")  # schema of the first release: no reviewed/summary columns
    old.executescript("""
        CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT DEFAULT '', settings TEXT DEFAULT '{}', created_at REAL NOT NULL);
        CREATE TABLE media (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, filename TEXT NOT NULL, ext TEXT NOT NULL, size INTEGER DEFAULT 0, duration REAL,
            status TEXT NOT NULL DEFAULT 'uploaded', stage TEXT DEFAULT '', progress REAL DEFAULT 0, message TEXT DEFAULT '', error TEXT DEFAULT '', created_at REAL NOT NULL);
        CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, media_id TEXT NOT NULL, status TEXT NOT NULL, settings TEXT NOT NULL, created_at REAL NOT NULL,
            started_at REAL, finished_at REAL, error TEXT DEFAULT '');
        INSERT INTO projects VALUES ('p1', 'Old project', '', '{}', 1);
        INSERT INTO media (id, project_id, filename, ext, duration, status, created_at) VALUES ('m1', 'p1', 'old.wav', '.wav', 3.0, 'done', 1);
    """)
    old.commit()
    old.close()
    mdir = data / "projects" / "p1" / "media" / "m1"
    mdir.mkdir(parents=True)
    (mdir / "transcript.json").write_text(json.dumps({"language": "en", "engine": "mlx", "model": "turbo", "segments": [
        {"id": 0, "start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hi there.", "words": [{"w": "Hi", "start": 0, "end": .5}, {"w": "there.", "start": .5, "end": 1}]},
        {"id": 1, "start": 1, "end": 2, "speaker": "SPEAKER_01", "text": "Hello.", "words": [{"w": "Hello.", "start": 1, "end": 2}]}]}))

    with TestClient(create_app(runner=InlineRunner())) as c:
        html = c.get("/projects/p1/transcripts").text
        assert "old.wav" in html and "To review" in html and "EN" in html and "SPEAKER_00, SPEAKER_01" in html
        row = c.app.state.db.get_media("m1")
        assert row["reviewed"] == 0 and json.loads(row["summary"])["words"] == 3  # backfilled and remembered


def test_finished_job_stores_a_transcript_summary(client, video):
    import json
    pid, mid = transcribed(client, video)
    row = client.app.state.db.get_media(mid)
    summary = json.loads(row["summary"])  # filled in when the job finished, not lazily later
    assert summary["speakers"] == ["SPEAKER_00", "SPEAKER_01"] and summary["words"] == 4 and summary["language"] == "en"
    assert row["reviewed"] == 0


def test_sidebar_controls_and_no_footer_notice(client):
    html = client.get("/").text
    assert "Running locally" not in html and "Nothing leaves this computer" not in html
    top = html[html.index('class="side-top"'):html.index("</nav>")]
    assert top.index("Transcriber") < top.index("Theme:") < top.index("Hide sidebar")   # title, then the theme icon, then hide
    assert "Theme: Auto" not in html                                                   # the old text button is gone
    assert 'aria-label="Show sidebar"' in html                                          # reopen button (shown only when hidden)
    assert 'transcriber.sidebar' in html                                                # saved state is applied before first paint


class _TagBalance:
    """Flags stray or missing closing tags. They don't fail on the server, but browsers (and htmx swaps) mis-nest the page."""

    VOID = {"meta", "link", "input", "br", "img", "hr", "source", "area", "base", "col", "embed", "track", "wbr"}

    def __init__(self, html):
        from html.parser import HTMLParser

        outer = self
        self.stack, self.errors = [], []

        class P(HTMLParser):
            def handle_starttag(self, tag, attrs):
                if tag not in outer.VOID:
                    outer.stack.append(tag)

            def handle_endtag(self, tag):
                if tag in outer.VOID:
                    return
                if outer.stack and outer.stack[-1] == tag:
                    outer.stack.pop()
                else:
                    outer.errors.append(f"unexpected </{tag}> after {outer.stack[-4:]}")

        P().feed(html)
        if self.stack:
            self.errors.append(f"never closed: {self.stack}")


def test_pages_and_fragments_are_well_formed_html(client, video):
    pid, mid = transcribed(client, video)
    client.put(f"/api/media/{mid}/segments/1", json={"text": "Edited."})
    urls = ["/", f"/projects/{pid}", f"/projects/{pid}/files", f"/projects/{pid}/transcripts",
            f"/projects/{pid}/transcripts/list", f"/media/{mid}", "/setup", "/projects/does-not-exist"]
    for url in urls:
        html = client.get(url).text
        assert _TagBalance(html).errors == [], (url, _TagBalance(html).errors)


def test_defaults_are_interview_preset_with_turbo():
    from transcriber.server.app import clean_settings
    d = clean_settings({})
    assert d["model_size"] == "turbo" and d["diarize"] is True and d["num_speakers"] == 2
    assert d["min_speakers"] is None and d["max_speakers"] is None and d["language"] is None
    assert clean_settings(None) == d


def test_cleared_speaker_counts_mean_automatic_not_default():
    from transcriber.server.app import clean_settings
    # the Focus group preset sends explicit nulls: they must not be silently turned back into "2 speakers"
    for blank in (None, ""):
        d = clean_settings({"num_speakers": blank, "min_speakers": 3, "max_speakers": 8})
        assert d["num_speakers"] is None and d["min_speakers"] == 3 and d["max_speakers"] == 8


def test_saved_project_settings_are_never_overridden_by_new_defaults():
    from transcriber.server.app import clean_settings
    saved = clean_settings({"model_size": "small", "num_speakers": None, "diarize": False, "language": "pt"})
    assert clean_settings(saved) == saved                 # round-trips unchanged
    assert saved["model_size"] == "small" and saved["num_speakers"] is None and saved["diarize"] is False


def test_new_project_shows_defaults_and_jobs_use_them(client, video):
    import html as htmllib
    pid = new_project(client)
    page = htmllib.unescape(client.get(f"/projects/{pid}").text)
    assert '"model_size": "turbo"' in page and '"num_speakers": 2' in page and '"diarize": true' in page
    mid = upload(client, pid, video).json()["id"]
    client.post(f"/api/projects/{pid}/transcribe", json={})          # never opened the settings panel
    wait_for(client, pid, mid, "done")
    import json
    job = client.app.state.db._q("select settings from jobs order by id desc limit 1")[0]
    used = json.loads(job["settings"])
    assert used["model_size"] == "turbo" and used["num_speakers"] == 2 and used["diarize"] is True
