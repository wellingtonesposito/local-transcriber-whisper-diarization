"""Export-time text cleanup. Pure functions; the stored transcript is never modified.

Ported from the original script's NVivo-friendly cleanup, with the destructive parts made
opt-in: "like" is never removed, phrase fillers ("you know", ...) are opt-in, pause markers
are kept unless asked, and spell-correction only runs for English.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FILLER_WORDS = r"(?:u+m+|u+h+|e+r+m+|uhm+)"
FILLER_PHRASES = [r"you know", r"i mean", r"sort of", r"kind of"]
PAUSE_MARKERS = [
    r"\(\s*pause\s*\)",
    r"\(\s*silence\s*\)",
    r"\(\s*\d+(?:\.\d+)?\s*s(?:ec(?:onds)?)?\s*\)",
    r"\[\s*pause\s*\]",
    r"\[\s*silence\s*\]",
    r"\(\s*inaudible\s*\)",
    r"\(\s*overlap\s*\)",
]


@dataclass
class TransformOptions:
    fillers: bool = False          # um / uh / erm
    phrase_fillers: bool = False   # you know / I mean / sort of / kind of (can remove real content)
    stutters: bool = False         # "I-I-I think" -> "I think", "sooo" -> "so"
    pause_markers: bool = False    # remove (pause) (inaudible) (overlap) ...
    normalize_punct: bool = False  # "..." -> ".", "?!" -> "?"
    spell: bool = False            # English only
    language: str | None = None

    @classmethod
    def preset(cls, name: str, language: str | None = None) -> "TransformOptions":
        if name == "verbatim":
            return cls(language=language)
        if name == "clean":
            return cls(fillers=True, stutters=True, normalize_punct=True, language=language)
        raise ValueError(f"unknown cleanup preset: {name}")

    @property
    def is_noop(self) -> bool:
        return not any((self.fillers, self.phrase_fillers, self.stutters, self.pause_markers,
                        self.normalize_punct, self.spell))


def remove_fillers(t: str, words: bool = True, phrases: bool = False) -> str:
    if words:  # also eat an attached comma/period so "Um, I think" doesn't leave ", I think"
        t = re.sub(rf"\b{FILLER_WORDS}\b[,.]?(?=\s|$)", "", t, flags=re.IGNORECASE)
    if phrases:
        for p in FILLER_PHRASES:
            t = re.sub(rf"\b{p}\b[,]?", "", t, flags=re.IGNORECASE)
    return t


def remove_pause_markers(t: str) -> str:
    for p in PAUSE_MARKERS:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    return t


def collapse_stutters(t: str) -> str:
    t = re.sub(r"\b([A-Za-z])(-\1){1,}\b", r"\1", t)
    t = re.sub(r"\b([A-Za-z])\1{2,}\b", r"\1", t)
    return t


def normalize_punct(t: str) -> str:
    t = re.sub(r"\.{2,}|…", ".", t)
    t = re.sub(r"[!?]{2,}", lambda m: m.group(0)[0], t)
    return t


_spell = None


def _get_spell():
    global _spell
    if _spell is None:
        from spellchecker import SpellChecker  # optional dependency

        _spell = SpellChecker(distance=1)
    return _spell


_TOKEN = re.compile(r"^(\W*)(\w+)(\W*)$")


def light_spell_correct(t: str) -> str:
    spell = _get_spell()
    out = []
    for tok in t.split():
        m = _TOKEN.match(tok)
        if not m:
            out.append(tok)
            continue
        pre, w, post = m.groups()
        if w[0].isupper() or w.isupper() or any(c.isdigit() for c in w) or len(w) <= 3:
            out.append(tok)
            continue
        cand = spell.correction(w.lower())
        if cand and cand != w.lower() and abs(len(cand) - len(w)) <= 2:
            out.append(pre + cand + post)
        else:
            out.append(tok)
    return " ".join(out)


def apply_transforms(text: str, opts: TransformOptions) -> str:
    if opts.is_noop:
        return text
    original = text.strip()
    t = original
    if opts.fillers or opts.phrase_fillers:
        t = remove_fillers(t, words=opts.fillers, phrases=opts.phrase_fillers)
    if opts.pause_markers:
        t = remove_pause_markers(t)
    if opts.stutters:
        t = collapse_stutters(t)
    if opts.normalize_punct:
        t = normalize_punct(t)
    if opts.spell and (opts.language or "").lower().startswith("en"):
        t = light_spell_correct(t)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[,;:]\s*", "", t)  # leading orphan punctuation left by a removed filler
    if t and original[:1].isupper() and t[0].isalpha():
        t = t[0].upper() + t[1:]
    return t
