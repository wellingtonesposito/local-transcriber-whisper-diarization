"""Stage orchestration: prepare -> transcribe -> diarize -> align, with disk checkpoints.

Checkpoints mean a failure in a later stage (typically diarization auth) never throws away
a long transcription, and a retry only redoes what is missing or whose settings changed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from . import audio
from .align import assign_speakers, build_segments
from .engines import AsrOptions, Cancelled, get_engine
from .model import AsrResult, Transcript, Turn

# stage -> (start, end) share of the overall progress bar
STAGES = {"prepare": (0.0, 0.03), "transcribe": (0.03, 0.70), "diarize": (0.70, 0.97), "align": (0.97, 1.0)}

ProgressFn = Callable[[str, float, str], None]  # (stage, overall fraction 0..1, message)


@dataclass
class JobSettings:
    model_size: str = "turbo"
    language: str | None = None
    engine: str = "auto"
    beam_size: int = 5
    vad: bool = True
    initial_prompt: str | None = None
    preserve_disfluencies: bool = False
    condition_on_previous_text: bool = True
    diarize: bool = True
    num_speakers: int | None = 2  # interviews are the common case; the Focus group preset clears it
    min_speakers: int | None = None
    max_speakers: int | None = None
    max_gap: float = 1.0
    max_duration: float = 30.0
    max_chars: int = 600

    def asr_options(self) -> AsrOptions:
        return AsrOptions(self.model_size, self.language, self.beam_size, self.vad, self.initial_prompt,
                          self.preserve_disfluencies, self.condition_on_previous_text)

    def asr_key(self) -> str:
        d = {k: v for k, v in asdict(self).items()
             if k in ("model_size", "language", "engine", "beam_size", "vad", "initial_prompt",
                      "preserve_disfluencies", "condition_on_previous_text")}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def diar_key(self) -> str:
        d = {k: getattr(self, k) for k in ("num_speakers", "min_speakers", "max_speakers")}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: dict | None) -> "JobSettings":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _read_ckpt(path: Path, key: str):
    if path.exists():
        try:
            d = json.loads(path.read_text())
            if d.get("key") == key:
                return d["data"]
        except Exception:
            pass
    return None


def _write_ckpt(path: Path, key: str, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": key, "data": data}))
    tmp.replace(path)


def run_pipeline(
    media_path: Path,
    work_dir: Path,
    settings: JobSettings,
    on_progress: ProgressFn,
    should_cancel: Callable[[], bool] = lambda: False,
    hf_token: str | None = None,
    diarizer=None,
    display_name: str | None = None,
) -> Transcript:
    """Run all stages and write `<work_dir>/transcript.json`. Returns the Transcript."""

    def stage_progress(stage: str):
        lo, hi = STAGES[stage]
        return lambda frac, msg="": on_progress(stage, lo + (hi - lo) * max(0.0, min(1.0, frac)), msg)

    def check():
        if should_cancel():
            raise Cancelled()

    # 1) prepare ----------------------------------------------------------------------------
    prep = stage_progress("prepare")
    prep(0.0, "Extracting audio")
    wav = work_dir / "audio.wav"
    if not wav.exists():
        audio.extract_wav(media_path, wav)
    prep(1.0, "")
    check()

    # 2) transcribe -------------------------------------------------------------------------
    ck = work_dir / "checkpoints"
    tr = stage_progress("transcribe")
    cached = _read_ckpt(ck / "asr.json", settings.asr_key())
    engine = get_engine(settings.engine)
    if cached:
        asr = AsrResult.from_dict(cached)
        tr(1.0, "Reusing earlier transcription")
    else:
        tr(0.0, f"Transcribing ({engine.name}, {settings.model_size})")
        asr = engine.transcribe(str(wav), settings.asr_options(), tr, should_cancel)
        _write_ckpt(ck / "asr.json", settings.asr_key(), asr.to_dict())
    check()

    # 3) diarize ----------------------------------------------------------------------------
    dz = stage_progress("diarize")
    turns: list[Turn] = []
    if settings.diarize:
        cached_t = _read_ckpt(ck / "diarization.json", settings.diar_key())
        if cached_t is not None:
            turns = [Turn(**t) for t in cached_t]
            dz(1.0, "Reusing earlier diarization")
        else:
            if diarizer is None:
                from .diarize import Diarizer

                diarizer = Diarizer()
            dz(0.0, "Identifying speakers")
            turns = diarizer.run(str(wav), hf_token, dz, should_cancel, settings.num_speakers,
                                 settings.min_speakers, settings.max_speakers)
            _write_ckpt(ck / "diarization.json", settings.diar_key(), [asdict(t) for t in turns])
    else:
        dz(1.0, "Speaker identification skipped")
    check()

    # 4) align ------------------------------------------------------------------------------
    al = stage_progress("align")
    al(0.0, "Building transcript")
    labels = assign_speakers(asr.words, turns)
    segments = build_segments(asr.words, labels, settings.max_gap, settings.max_duration, settings.max_chars)
    probe = audio.probe(wav)
    transcript = Transcript(
        media={"filename": display_name or media_path.name, "duration": asr.duration or probe["duration"]},
        language=asr.language,
        engine=engine.name,
        model=settings.model_size,
        params=asdict(settings),
        segments=segments,
        speakers={s: {"name": None} for s in dict.fromkeys(labels)},
    )
    out = work_dir / "transcript.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False))
    tmp.replace(out)
    al(1.0, "Done")
    return transcript
