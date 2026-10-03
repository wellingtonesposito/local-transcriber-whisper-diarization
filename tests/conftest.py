import pytest

from transcriber.core.align import assign_speakers, build_segments
from transcriber.core.model import Transcript, Turn, Word


def make_words(spec):
    """spec: list of (text, start, end)."""
    return [Word(t, s, e) for t, s, e in spec]


@pytest.fixture
def interview() -> Transcript:
    """Two speakers; the first Whisper 'segment' would span a speaker change."""
    words = make_words([
        ("Hello,", 0.0, 0.5), ("um,", 0.6, 0.8), ("thanks", 0.9, 1.3), ("for", 1.3, 1.5), ("joining.", 1.5, 2.0),
        ("I", 2.4, 2.5), ("like", 2.5, 2.8), ("pizza,", 2.8, 3.3), ("you", 3.4, 3.5), ("know.", 3.5, 3.9),
        ("(inaudible)", 4.0, 4.5),
        ("Sure!", 6.0, 6.4),
    ])
    turns = [Turn(0.0, 2.2, "SPEAKER_00"), Turn(2.3, 4.6, "SPEAKER_01"), Turn(5.9, 6.5, "SPEAKER_00")]
    labels = assign_speakers(words, turns)
    segs = build_segments(words, labels)
    return Transcript(media={"filename": "interview.mp4"}, language="en", engine="test", model="tiny",
                      params={}, segments=segs)
