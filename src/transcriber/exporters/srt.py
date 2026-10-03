from __future__ import annotations

from ..core.model import Edits, Transcript
from ..core.render import ExportOptions, build_cues, fmt_ts


def render(transcript: Transcript, edits: Edits | None, opts: ExportOptions) -> bytes:
    opts_cues = opts.unmerged()
    out = []
    for i, c in enumerate(build_cues(transcript, edits, opts_cues), 1):
        out += [str(i), f"{fmt_ts(c.start, 3, ',')} --> {fmt_ts(c.end, 3, ',')}", f"{c.speaker}: {c.text}", ""]
    return "\n".join(out).encode("utf-8")
