from __future__ import annotations

import json

from ..core.model import Edits, Transcript
from ..core.render import ExportOptions, build_cues, speaker_display


def render(transcript: Transcript, edits: Edits | None, opts: ExportOptions) -> bytes:
    """Full export: metadata, word timings (raw), and the rendered text per segment."""
    d = transcript.to_dict()
    cues = {(c.start, c.end): c for c in build_cues(transcript, edits, opts.unmerged())}
    segs = []
    for s in d["segments"]:
        c = cues.get((s["start"], s["end"]))
        if c is None:  # deleted or emptied by cleanup
            continue
        s["speaker_name"] = speaker_display(transcript, edits, s["speaker"])
        s["rendered_text"] = c.text
        segs.append(s)
    d["segments"] = segs
    d["export"] = {"cleanup": opts.cleanup}
    return json.dumps(d, ensure_ascii=False, indent=2).encode("utf-8")
