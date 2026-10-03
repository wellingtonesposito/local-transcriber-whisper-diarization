from __future__ import annotations

import io
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from ..core.model import Edits, Transcript
from ..core.render import ExportOptions, build_cues, fmt_ts

# Unicode-capable fonts to try; falls back to built-in Helvetica (Latin-1: covers en/es/pt/fr/de).
_FONT_CANDIDATES = [
    ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", None),
    ("/Library/Fonts/Arial Unicode.ttf", None),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
]


def _register_font() -> tuple[str, str]:
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists():
            try:
                pdfmetrics.registerFont(TTFont("TxBody", regular))
                if bold and Path(bold).exists():
                    pdfmetrics.registerFont(TTFont("TxBold", bold))
                    return "TxBody", "TxBold"
                return "TxBody", "TxBody"
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold"


def render(transcript: Transcript, edits: Edits | None, opts: ExportOptions) -> bytes:
    body_font, bold_font = _register_font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=body_font, fontSize=10.5, leading=15, spaceAfter=8)
    title = ParagraphStyle("title", parent=styles["Title"], fontName=bold_font, fontSize=16, alignment=0)

    story = [Paragraph(escape(transcript.media.get("filename", "Transcript")), title), Spacer(1, 10)]
    for c in build_cues(transcript, edits, opts):
        ts = f"[{fmt_ts(c.start, opts.ts_precision)}] " if opts.timestamps else ""
        story.append(Paragraph(f'{escape(ts)}<font name="{bold_font}">{escape(c.speaker)}:</font> {escape(c.text)}', body))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54,
                      title=transcript.media.get("filename", "Transcript")).build(story)
    return buf.getvalue()
