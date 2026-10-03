"""Hardware/engine detection. Imports are lazy so this works without the ML extras installed."""
from __future__ import annotations

import importlib.util
import platform


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def detect() -> dict:
    info = {
        "platform": f"{platform.system()} {platform.machine()}",
        "faster_whisper": _has("faster_whisper"),
        "mlx_whisper": _has("mlx_whisper"),
        "pyannote": _has("pyannote.audio") if _has("pyannote") else False,
        "torch": _has("torch"),
        "cuda": False,
        "mps": False,
    }
    if info["torch"]:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    info["recommended_engine"] = recommended_engine(info)
    return info


def recommended_engine(info: dict | None = None) -> str:
    info = info or detect()
    if is_apple_silicon() and info.get("mlx_whisper"):
        return "mlx"
    return "faster-whisper"


def torch_device() -> str:
    """Device for pyannote: cuda > mps > cpu. MPS is attempted but callers must fall back to cpu on error."""
    if not _has("torch"):
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def quiet_third_party_warnings() -> None:
    """Hide harmless deprecation noise from the ML libraries (torchaudio backend notices, pyannote's regex
    SyntaxWarnings, speechbrain moves). Warnings from our own code and everything else stay visible."""
    import warnings

    warnings.filterwarnings("ignore", category=SyntaxWarning)  # raised while pyannote.database is compiled
    warnings.filterwarnings("ignore", module=r"(pyannote|torchaudio|speechbrain|lightning|pytorch_lightning|torch)(\.|$)")
