from __future__ import annotations

import csv
import io

from ..core.model import Edits, Transcript
from ..core.render import ExportOptions, build_cues, fmt_ts


def render(transcript: Transcript, edits: Edits | None, opts: ExportOptions) -> bytes:
    opts_cues = opts.unmerged()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["start", "end", "start_time", "end_time", "speaker", "text"])
    for c in build_cues(transcript, edits, opts_cues):
        w.writerow([f"{c.start:.3f}", f"{c.end:.3f}", fmt_ts(c.start, 3), fmt_ts(c.end, 3), c.speaker, c.text])
    return buf.getvalue().encode("utf-8-sig")  # BOM so Excel detects UTF-8
