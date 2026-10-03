from __future__ import annotations

from ..core.model import Edits, Transcript
from ..core.render import ExportOptions, build_cues, fmt_ts


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(transcript: Transcript, edits: Edits | None, opts: ExportOptions) -> bytes:
    # Cues, not merged turns: captions need to stay short and time-aligned.
    opts_cues = opts.unmerged()
    out = ["WEBVTT", ""]
    for c in build_cues(transcript, edits, opts_cues):
        out.append(f"{fmt_ts(c.start, 3)} --> {fmt_ts(c.end, 3)}")
        if opts.speaker_style == "voice":
            out.append(f"<v {_esc(c.speaker)}>{_esc(c.text)}")
        else:
            out.append(f"{_esc(c.speaker)}: {_esc(c.text)}")
        out.append("")
    return "\n".join(out).encode("utf-8")
