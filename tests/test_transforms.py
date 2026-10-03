import pytest

from transcriber.core.transforms import TransformOptions, apply_transforms


def test_verbatim_is_noop():
    t = "Um, I like pizza, you know (inaudible) ..."
    assert apply_transforms(t, TransformOptions.preset("verbatim")) == t


def test_clean_keeps_like_and_real_content():
    out = apply_transforms("I like pizza and what kind of cheese", TransformOptions.preset("clean"))
    assert out == "I like pizza and what kind of cheese"


def test_clean_removes_fillers_and_fixes_case_and_comma():
    assert apply_transforms("Um, I think so", TransformOptions.preset("clean")) == "I think so"
    assert apply_transforms("Well, uh, maybe", TransformOptions.preset("clean")) == "Well, maybe"
    assert apply_transforms("Hmm that is fine", TransformOptions.preset("clean")) == "Hmm that is fine"


def test_phrase_fillers_are_opt_in():
    o = TransformOptions(phrase_fillers=True)
    assert apply_transforms("It was, you know, hard", o) == "It was, hard"
    assert apply_transforms("It was, you know, hard", TransformOptions.preset("clean")) == "It was, you know, hard"


def test_pause_markers_removed_only_when_asked():
    assert apply_transforms("Yes (inaudible) no", TransformOptions.preset("clean")) == "Yes (inaudible) no"
    assert apply_transforms("Yes (inaudible) no", TransformOptions(pause_markers=True)) == "Yes no"


def test_stutters_and_punct():
    o = TransformOptions.preset("clean")
    assert apply_transforms("I-I-I think", o) == "I think"
    assert apply_transforms("Really?! Well... ok", o) == "Really? Well. ok"


def test_spell_only_for_english():
    pytest.importorskip("spellchecker")
    en = TransformOptions(spell=True, language="en")
    pt = TransformOptions(spell=True, language="pt")
    assert apply_transforms("teh quick brwn", pt) == "teh quick brwn"
    assert apply_transforms("wierd thingz", en) != "wierd thingz"
