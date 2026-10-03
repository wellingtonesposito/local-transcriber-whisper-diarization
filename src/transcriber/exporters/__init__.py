"""Exporter registry: name -> Exporter. Each renders (transcript, edits, options) to bytes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..core.model import Edits, Transcript
from ..core.render import ExportOptions


@dataclass(frozen=True)
class Exporter:
    key: str
    label: str
    ext: str
    mime: str
    description: str
    render: Callable[[Transcript, Edits | None, ExportOptions], bytes]
    suffix: str = ""  # appended to the file stem so two formats sharing an extension don't collide
    group: str = "data"  # "analysis" (QDA software) | "data" (data & reading)


_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _registry() -> dict[str, Exporter]:
    from . import csv_, docx_, json_, md, pdf, srt, txt, vtt

    items = [
        Exporter("docx_table", "NVivo", "docx", _DOCX, "Word table: Timespan · Speaker · Content",
                 docx_.render_table, suffix="_table", group="analysis"),
        Exporter("vtt", "ATLAS.ti & MAXQDA: captions", "vtt", "text/vtt",
                 "Speaker-labelled captions synced to your media", vtt.render, group="analysis"),
        Exporter("docx", "ATLAS.ti & MAXQDA: document", "docx", _DOCX,
                 "[00:01:12.3] Speaker: text paragraphs", docx_.render_paragraphs, group="analysis"),
        Exporter("csv", "Spreadsheet · Dedoose", "csv", "text/csv", "start, end, speaker, text", csv_.render),
        Exporter("txt", "Plain text", "txt", "text/plain", "[timestamp] Speaker: text, or grouped by speaker", txt.render),
        Exporter("pdf", "PDF", "pdf", "application/pdf", "Readable, printable transcript", pdf.render),
        Exporter("srt", "SRT captions", "srt", "application/x-subrip", "For video players and editors", srt.render),
        Exporter("md", "Markdown", "md", "text/markdown", "For notes tools and documentation", md.render),
        Exporter("json", "Full data", "json", "application/json", "JSON with word timings and the settings used", json_.render),
    ]
    return {e.key: e for e in items}


_CACHE: dict[str, Exporter] | None = None


def get_exporters() -> dict[str, Exporter]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _registry()
    return _CACHE


def render(key: str, transcript: Transcript, edits: Edits | None, opts: ExportOptions | None = None) -> tuple[bytes, Exporter]:
    ex = get_exporters()[key]
    return ex.render(transcript, edits, opts or ExportOptions()), ex
