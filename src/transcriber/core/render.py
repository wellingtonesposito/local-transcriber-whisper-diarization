"""Turn a Transcript (+ user edits + cleanup options) into display cues for exporters."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .model import Edits, Transcript
from .transforms import TransformOptions, apply_transforms


@dataclass
class Cue:
    start: float
    end: float
    speaker: str  # display name
    text: str


@dataclass
class ExportOptions:
    cleanup: str = "verbatim"                 # "verbatim" | "clean" | "custom"
    transforms: TransformOptions | None = None  # used when cleanup == "custom"
    timestamps: bool = True
    ts_precision: int = 1                     # decimals for human-readable timestamps (txt/docx/pdf/md)
    speaker_style: str = "prefix"             # vtt: "prefix" ("Name: text") | "voice" (<v Name>text)
    merge_turns: bool | None = None           # None = auto (merge only when timestamps are off)
    extra: dict = field(default_factory=dict)

    def unmerged(self) -> "ExportOptions":
        """Same options but one cue per stored segment (captions/CSV need time-aligned cues)."""
        return replace(self, merge_turns=False)

    def resolve_transforms(self, language: str | None) -> TransformOptions:
        if self.cleanup == "custom" and self.transforms is not None:
            t = self.transforms
            t.language = t.language or language
            return t
        return TransformOptions.preset("clean" if self.cleanup == "clean" else "verbatim", language)


def speaker_display(transcript: Transcript, edits: Edits | None, spk: str) -> str:
    if edits and edits.speakers.get(spk):
        return edits.speakers[spk]
    return transcript.speakers.get(spk, {}).get("name") or spk


def build_cues(transcript: Transcript, edits: Edits | None = None, opts: ExportOptions | None = None) -> list[Cue]:
    opts = opts or ExportOptions()
    edits = edits or Edits()
    tf = opts.resolve_transforms(transcript.language)
    cues: list[Cue] = []
    for seg in transcript.segments:
        e = edits.segments.get(str(seg.id), {})
        if e.get("deleted"):
            continue
        text = apply_transforms(e.get("text", seg.text), tf)
        if not text:
            continue
        cues.append(Cue(seg.start, seg.end, speaker_display(transcript, edits, seg.speaker), text))
    if opts.merge_turns if opts.merge_turns is not None else not opts.timestamps:
        cues = merge_turns(cues)
    return cues


def review_rows(transcript: Transcript, edits: Edits | None, opts: ExportOptions | None = None) -> list[dict]:
    """Every stored segment (including deleted ones, so they can be restored) for the review page."""
    opts = opts or ExportOptions()
    edits = edits or Edits()
    tf = opts.resolve_transforms(transcript.language)
    rows = []
    for seg in transcript.segments:
        e = edits.segments.get(str(seg.id), {})
        base = e.get("text", seg.text)
        rows.append({"id": seg.id, "start": seg.start, "end": seg.end, "speaker": seg.speaker,
                     "text": apply_transforms(base, tf), "raw": base, "edited": "text" in e,
                     "deleted": bool(e.get("deleted")),
                     "words": [[w.w.strip(), round(w.start, 2), round(w.end, 2)] for w in seg.words]})
    return rows


def merge_turns(cues: list[Cue]) -> list[Cue]:
    out: list[Cue] = []
    for c in cues:
        if out and out[-1].speaker == c.speaker:
            out[-1] = Cue(out[-1].start, c.end, c.speaker, out[-1].text + " " + c.text)
        else:
            out.append(Cue(c.start, c.end, c.speaker, c.text))
    return out


def fmt_ts(seconds: float, precision: int = 3, sep: str = ".", with_hours: bool = True) -> str:
    """HH:MM:SS<sep>fff, rounded (not truncated) to `precision` decimals."""
    seconds = max(0.0, float(seconds))
    scale = 10 ** precision
    total = round(seconds * scale)
    whole, frac = divmod(total, scale)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    base = f"{h:02d}:{m:02d}:{s:02d}" if with_hours or h else f"{m:02d}:{s:02d}"
    return base + (f"{sep}{frac:0{precision}d}" if precision > 0 else "")
