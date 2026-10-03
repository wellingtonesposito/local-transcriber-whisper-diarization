from transcriber.core.align import assign_speakers, build_segments
from transcriber.core.model import Turn, Word


def W(t, s, e):
    return Word(t, s, e)


def test_segment_spanning_speaker_change_is_split():
    words = [W("hi", 0, 1), W("there", 1, 2), W("hello", 2.1, 3), W("back", 3, 4)]
    turns = [Turn(0, 2.05, "A"), Turn(2.05, 4, "B")]
    labels = assign_speakers(words, turns)
    assert labels == ["A", "A", "B", "B"]
    segs = build_segments(words, labels)
    assert [(s.speaker, s.text) for s in segs] == [("A", "hi there"), ("B", "hello back")]


def test_no_turns_does_not_crash():
    words = [W("a", 0, 1)]
    assert assign_speakers(words, []) == ["SPEAKER_00"]


def test_word_in_gap_goes_to_nearest_turn():
    words = [W("x", 5.0, 5.2)]
    turns = [Turn(0, 1, "A"), Turn(5.5, 7, "B")]
    assert assign_speakers(words, turns) == ["B"]


def test_lone_flipped_word_is_smoothed():
    words = [W("a", 0, 1), W("b", 1, 2), W("c", 2, 3)]
    turns = [Turn(0, 1, "A"), Turn(1, 2, "B"), Turn(2, 3, "A")]
    assert assign_speakers(words, turns) == ["A", "A", "A"]
    assert assign_speakers(words, turns, smooth=False) == ["A", "B", "A"]


def test_long_run_is_split_at_sentence_end():
    words = [W(f"w{i}.", i, i + 0.9) if i in (9, 19) else W(f"w{i}", i, i + 0.9) for i in range(30)]
    segs = build_segments(words, ["A"] * 30, max_gap=5, max_duration=12)
    assert len(segs) >= 3
    assert all(s.end - s.start <= 12 for s in segs)
    assert segs[0].text.endswith("w9.")
    assert [s.id for s in segs] == list(range(len(segs)))


def test_pause_splits_same_speaker():
    words = [W("a", 0, 1), W("b", 5, 6)]
    segs = build_segments(words, ["A", "A"], max_gap=1.0)
    assert len(segs) == 2
