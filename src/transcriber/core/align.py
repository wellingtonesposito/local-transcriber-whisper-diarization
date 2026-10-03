"""Word -> speaker assignment and segment building.

Improves on the original script, which assigned a whole Whisper segment to one speaker.
Here every word gets a speaker, so a segment that spans a speaker change is split.
"""
from __future__ import annotations

import bisect

from .model import Segment, Turn, Word

SENTENCE_END = (".", "?", "!", "…")


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(words: list[Word], turns: list[Turn], smooth: bool = True) -> list[str]:
    """Return one speaker label per word (same order as `words`)."""
    if not words:
        return []
    if not turns:
        return ["SPEAKER_00"] * len(words)

    ts = sorted(turns, key=lambda t: (t.start, t.end))
    starts = [t.start for t in ts]
    max_len = max(t.end - t.start for t in ts)

    labels: list[str] = []
    for w in words:
        hi = bisect.bisect_right(starts, w.end)
        lo = bisect.bisect_left(starts, w.start - max_len)
        best, best_ov = None, 0.0
        for t in ts[lo:hi]:
            ov = overlap(w.start, w.end, t.start, t.end)
            if ov > best_ov:
                best, best_ov = t, ov
        if best is None:  # word falls in a gap (or zero-length): use the nearest turn
            cands = ts[max(0, hi - 2): hi + 1] or ts[-1:]
            mid = (w.start + w.end) / 2
            best = min(cands, key=lambda t: 0.0 if t.start <= mid <= t.end else min(abs(mid - t.start), abs(mid - t.end)))
        labels.append(best.spk)

    if smooth:  # a lone word flanked by the same other speaker is almost always boundary noise
        for i in range(1, len(labels) - 1):
            if labels[i - 1] == labels[i + 1] != labels[i]:
                labels[i] = labels[i - 1]
    return labels


def _join(words: list[Word]) -> str:
    return " ".join(w.w.strip() for w in words if w.w.strip())


def _split_long(words: list[Word], max_duration: float, max_chars: int) -> list[list[Word]]:
    """Split one run of same-speaker words into cue-sized chunks, preferring sentence ends."""
    chunks: list[list[Word]] = []
    buf: list[Word] = []
    chars = 0
    for w in words:
        buf.append(w)
        chars += len(w.w) + 1
        too_long = (buf[-1].end - buf[0].start) > max_duration or chars > max_chars
        if not too_long:
            continue
        cut = None
        for i in range(len(buf) - 2, -1, -1):  # last sentence end, but not leaving a tiny head
            if buf[i].w.strip().endswith(SENTENCE_END) and i + 1 >= max(3, len(buf) * 0.4):
                cut = i + 1
                break
        if cut is None:
            cut = len(buf) if len(buf) == 1 else len(buf) - 1
        chunks.append(buf[:cut])
        buf = buf[cut:]
        chars = sum(len(x.w) + 1 for x in buf)
    if buf:
        chunks.append(buf)
    return chunks


def build_segments(
    words: list[Word],
    labels: list[str],
    max_gap: float = 1.0,
    max_duration: float = 30.0,
    max_chars: int = 600,
) -> list[Segment]:
    """Group words into segments: split on speaker change, long pauses, and size limits."""
    runs: list[tuple[str, list[Word]]] = []
    for w, spk in zip(words, labels):
        if not w.w.strip():
            continue
        if runs and runs[-1][0] == spk and (w.start - runs[-1][1][-1].end) < max_gap:
            runs[-1][1].append(w)
        else:
            runs.append((spk, [w]))

    segments: list[Segment] = []
    for spk, run in runs:
        for chunk in _split_long(run, max_duration, max_chars):
            segments.append(
                Segment(id=len(segments), start=chunk[0].start, end=chunk[-1].end, speaker=spk,
                        text=_join(chunk), words=chunk)
            )
    return segments
