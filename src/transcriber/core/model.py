"""Canonical transcript data model.

`transcript.json` is the immutable, verbatim source of truth. User changes live in a
separate edits overlay (see `Edits`) and cleanup is applied only when rendering.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class Word:
    w: str
    start: float
    end: float
    p: float | None = None


@dataclass
class Turn:
    """A diarization speaker turn."""

    start: float
    end: float
    spk: str


@dataclass
class Segment:
    id: int
    start: float
    end: float
    speaker: str
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class AsrResult:
    """Output of a speech-recognition engine (word level)."""

    language: str | None
    duration: float
    words: list[Word]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AsrResult":
        return cls(d.get("language"), float(d.get("duration") or 0.0), [Word(**w) for w in d["words"]])


@dataclass
class Transcript:
    media: dict[str, Any]
    language: str | None
    engine: str
    model: str
    params: dict[str, Any]
    segments: list[Segment]
    speakers: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcript":
        segs = [
            Segment(
                id=s["id"], start=s["start"], end=s["end"], speaker=s["speaker"], text=s["text"],
                words=[Word(**w) for w in s.get("words", [])],
            )
            for s in d["segments"]
        ]
        return cls(
            media=d.get("media", {}), language=d.get("language"), engine=d.get("engine", ""),
            model=d.get("model", ""), params=d.get("params", {}), segments=segs,
            speakers=d.get("speakers", {}), schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def speaker_ids(self) -> list[str]:
        """Speaker ids in order of first appearance."""
        seen: dict[str, None] = {}
        for s in self.segments:
            seen.setdefault(s.speaker, None)
        return list(seen)


@dataclass
class Edits:
    """User overlay applied on top of the raw transcript."""

    speakers: dict[str, str] = field(default_factory=dict)  # speaker id -> display name
    segments: dict[str, dict[str, Any]] = field(default_factory=dict)  # str(seg id) -> {"text":..,"deleted":..}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Edits":
        d = d or {}
        return cls(speakers=dict(d.get("speakers", {})), segments=dict(d.get("segments", {})))
