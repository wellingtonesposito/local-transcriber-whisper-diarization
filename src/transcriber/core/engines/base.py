from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..model import AsrResult

# Whisper normally drops disfluencies; a disfluent prompt nudges it to keep them (useful for verbatim work).
DISFLUENCY_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."


class EngineUnavailable(RuntimeError):
    """The engine's Python package (or hardware) is not available."""


class Cancelled(Exception):
    """Raised when the user cancels a running job."""


@dataclass
class AsrOptions:
    model_size: str = "small"
    language: str | None = None          # None = auto-detect
    beam_size: int = 5
    vad: bool = True
    initial_prompt: str | None = None    # domain vocabulary / names
    preserve_disfluencies: bool = False
    condition_on_previous_text: bool = True

    def effective_prompt(self) -> str | None:
        parts = [p for p in (self.initial_prompt, DISFLUENCY_PROMPT if self.preserve_disfluencies else None) if p]
        return " ".join(parts) or None


ProgressFn = Callable[[float, str], None]  # (fraction 0..1, message)


class Engine(Protocol):
    name: str

    def transcribe(self, wav_path: str, opts: AsrOptions, on_progress: ProgressFn,
                   should_cancel: Callable[[], bool]) -> AsrResult: ...
