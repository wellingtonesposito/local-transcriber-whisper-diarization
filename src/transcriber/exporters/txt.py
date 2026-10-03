from __future__ import annotations

from ..core.model import Edits, Transcript
from ..core.render import ExportOptions, build_cues, fmt_ts


def render(transcript: Transcript, edits: Edits | None, opts: ExportOptions) -> bytes:
    lines = []
    for c in build_cues(transcript, edits, opts):
        prefix = f"[{fmt_ts(c.start, opts.ts_precision)}] " if opts.timestamps else ""
        lines.append(f"{prefix}{c.speaker}: {c.text}")
    return ("\n\n".join(lines) + "\n").encode("utf-8")
