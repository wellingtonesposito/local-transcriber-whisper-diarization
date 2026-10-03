from __future__ import annotations

from ..device import recommended_engine
from .base import AsrOptions, Cancelled, Engine, EngineUnavailable

__all__ = ["AsrOptions", "Cancelled", "Engine", "EngineUnavailable", "get_engine"]

_ENGINES: dict[str, Engine] = {}


def get_engine(name: str = "auto") -> Engine:
    """Engines are cached so loaded models survive across jobs in the worker process."""
    if name == "auto":
        name = recommended_engine()
    if name not in _ENGINES:
        if name == "faster-whisper":
            from .faster_whisper_engine import FasterWhisperEngine

            _ENGINES[name] = FasterWhisperEngine()
        elif name == "mlx":
            from .mlx_engine import MlxEngine

            _ENGINES[name] = MlxEngine()
        else:
            raise ValueError(f"unknown engine: {name}")
    return _ENGINES[name]
