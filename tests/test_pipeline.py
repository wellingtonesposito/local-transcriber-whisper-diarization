import json

import pytest

from transcriber.core import audio, pipeline
from transcriber.core.engines import Cancelled
from transcriber.core.pipeline import JobSettings, run_pipeline

from fakes import FakeDiarizer, FakeEngine, make_media


@pytest.fixture
def media(tmp_path):
    return make_media(tmp_path / "in.mp4")


@pytest.fixture
def engine(monkeypatch):
    e = FakeEngine()
    monkeypatch.setattr(pipeline, "get_engine", lambda name="auto": e)
    return e


def test_probe_and_extract(media, tmp_path):
    info = audio.probe(media)
    assert info["has_audio"] and info["duration"] == pytest.approx(3.0, abs=0.2)
    wav = audio.extract_wav(media, tmp_path / "x" / "a.wav")
    assert wav.exists() and not (tmp_path / "x" / "a.part.wav").exists()
    with pytest.raises(audio.AudioError):
        audio.probe(tmp_path)  # a directory is not media


def test_full_run_writes_transcript(media, tmp_path, engine):
    events = []
    t = run_pipeline(media, tmp_path / "work", JobSettings(), lambda s, f, m: events.append((s, f)),
                     diarizer=FakeDiarizer())
    d = json.loads((tmp_path / "work" / "transcript.json").read_text())
    assert [s["speaker"] for s in d["segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert d["language"] == "en" and d["engine"] == "fake" and d["params"]["model_size"] == "turbo"
    assert t.segments[0].text == "Hello there."
    fr = [f for _, f in events]
    assert fr == sorted(fr) and fr[-1] == 1.0
    assert {s for s, _ in events} == {"prepare", "transcribe", "diarize", "align"}


def test_diarization_failure_keeps_asr_and_retry_skips_transcription(media, tmp_path, engine):
    work = tmp_path / "work"
    with pytest.raises(RuntimeError, match="auth"):
        run_pipeline(media, work, JobSettings(), lambda *a: None, diarizer=FakeDiarizer(fail=True))
    assert engine.calls == 1 and (work / "checkpoints" / "asr.json").exists()
    run_pipeline(media, work, JobSettings(), lambda *a: None, diarizer=FakeDiarizer())
    assert engine.calls == 1  # transcription reused


def test_changed_asr_settings_invalidate_checkpoint(media, tmp_path, engine):
    work = tmp_path / "work"
    run_pipeline(media, work, JobSettings(diarize=False), lambda *a: None)
    run_pipeline(media, work, JobSettings(diarize=False), lambda *a: None)
    assert engine.calls == 1
    run_pipeline(media, work, JobSettings(diarize=False, model_size="tiny"), lambda *a: None)
    assert engine.calls == 2


def test_diarize_off_gives_single_speaker(media, tmp_path, engine):
    t = run_pipeline(media, tmp_path / "w", JobSettings(diarize=False), lambda *a: None)
    assert {s.speaker for s in t.segments} == {"SPEAKER_00"}


def test_cancel(media, tmp_path, engine):
    with pytest.raises(Cancelled):
        run_pipeline(media, tmp_path / "w", JobSettings(), lambda *a: None, should_cancel=lambda: True)
