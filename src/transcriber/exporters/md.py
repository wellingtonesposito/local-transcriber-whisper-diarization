from __future__ import annotations

from ..core.model import Edits, Transcript
from ..core.render import ExportOptions, build_cues, fmt_ts


def render(transcript: Transcript, edits: Edits | None, opts: ExportOptions) -> bytes:
    title = transcript.media.get("filename", "Transcript")
    lines = [f"# {title}", ""]
    for c in build_cues(transcript, edits, opts):
        ts = f" `{fmt_ts(c.start, opts.ts_precision)}`" if opts.timestamps else ""
        lines += [f"**{c.speaker}**{ts}", "", c.text, ""]
    return "\n".join(lines).encode("utf-8")
