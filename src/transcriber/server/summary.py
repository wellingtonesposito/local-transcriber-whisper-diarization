"""Small facts about a finished transcript, kept in the database so list pages never have to parse the full JSON."""
from __future__ import annotations

import json
from pathlib import Path


def transcript_summary(path: Path) -> dict:
    d = json.loads(path.read_text())
    segs = d.get("segments", [])
    speakers = list(dict.fromkeys(s["speaker"] for s in segs))
    return {"language": d.get("language"), "engine": d.get("engine"), "model": d.get("model"),
            "speakers": speakers, "segments": len(segs), "words": sum(len(s.get("words", [])) for s in segs)}
