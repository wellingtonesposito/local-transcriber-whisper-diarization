"""pyannote speaker diarization with real progress, device selection and speaker-count hints."""
from __future__ import annotations

from typing import Callable

from .device import torch_device
from .engines.base import Cancelled, EngineUnavailable
from .model import Turn

MODEL_ID = "pyannote/speaker-diarization-3.1"
TERMS_URLS = [
    "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "https://huggingface.co/pyannote/segmentation-3.0",
]

# Approximate share of wall time per pyannote pipeline step, used to build one smooth progress bar.
_STEP_WEIGHTS = {"segmentation": (0.0, 0.35), "embeddings": (0.35, 0.95)}


class DiarizationError(RuntimeError):
    pass


def _allow_pyannote_checkpoints(torch) -> None:
    """PyTorch >= 2.6 loads checkpoints with weights_only=True. pyannote 3.1's checkpoints store four of its own
    (and torch's) classes, so allowlist exactly those instead of disabling the safety check for everything."""
    add = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if add is None:  # older torch: nothing to do
        return
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from torch.torch_version import TorchVersion

    add([TorchVersion, Specifications, Problem, Resolution])


def _looks_like_access_error(e: Exception) -> bool:
    """True only for Hugging Face auth/gating problems, so we never blame the token for anything else."""
    names = {c.__name__ for c in type(e).__mro__}
    if names & {"GatedRepoError", "RepositoryNotFoundError", "HfHubHTTPError", "LocalEntryNotFoundError"}:
        return True
    text = str(e).lower()
    return any(k in text for k in ("401", "403", "gated", "access token", "invalid token", "unauthorized", "restricted"))


def _load_error(e: Exception) -> Exception:
    """Turn a model-loading failure into an error message that points at the real cause."""
    if isinstance(e, ImportError):  # e.g. a library pyannote needs but does not declare
        return EngineUnavailable(
            f"Speaker identification is missing a library ({type(e).__name__}: {e}). "
            "Reinstall with: uv pip install -e \".[asr]\""
        )
    if _looks_like_access_error(e):
        return DiarizationError(
            "Hugging Face denied access to the speaker model. Add a token in Setup and accept the model terms at "
            f"{' and '.join(TERMS_URLS)}. ({type(e).__name__}: {e})"
        )
    return DiarizationError(f"Could not load the speaker model ({type(e).__name__}: {e}).")


class Diarizer:
    def __init__(self) -> None:
        self._pipeline = None
        self._token: str | None = None
        self.device = "cpu"

    def _load(self, hf_token: str | None):
        if self._pipeline is not None and self._token == hf_token:
            return self._pipeline
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise EngineUnavailable("pyannote.audio is not installed (pip install 'transcriber[asr]').") from e
        _allow_pyannote_checkpoints(torch)
        try:
            try:
                pipe = Pipeline.from_pretrained(MODEL_ID, use_auth_token=hf_token)
            except TypeError:  # newer pyannote renamed the argument
                pipe = Pipeline.from_pretrained(MODEL_ID, token=hf_token)
        except Exception as e:
            raise _load_error(e) from e
        if pipe is None:  # pyannote returns None (instead of raising) when access is gated
            raise DiarizationError(
                "Hugging Face denied access to the pyannote model. Accept the terms at "
                f"{' and '.join(TERMS_URLS)} and check your token."
            )
        self.device = torch_device()
        try:
            pipe.to(torch.device(self.device))
        except Exception:
            self.device = "cpu"
            pipe.to(torch.device("cpu"))
        self._pipeline, self._token = pipe, hf_token
        return pipe

    def run(self, wav_path: str, hf_token: str | None, on_progress: Callable[[float, str], None],
            should_cancel: Callable[[], bool], num_speakers: int | None = None,
            min_speakers: int | None = None, max_speakers: int | None = None) -> list[Turn]:
        pipe = self._load(hf_token)

        def hook(step_name, step_artifact, file=None, total=None, completed=None):
            if should_cancel():
                raise Cancelled()
            lo_hi = _STEP_WEIGHTS.get(step_name)
            if lo_hi and total and completed is not None:
                lo, hi = lo_hi
                on_progress(lo + (hi - lo) * min(1.0, completed / total), step_name)

        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers:
                kwargs["min_speakers"] = min_speakers
            if max_speakers:
                kwargs["max_speakers"] = max_speakers

        try:
            diar = pipe(wav_path, hook=hook, **kwargs)
        except Cancelled:
            raise
        except Exception as e:
            if self.device != "cpu":  # MPS/CUDA op failure: retry once on CPU
                import torch

                self.device = "cpu"
                pipe.to(torch.device("cpu"))
                diar = pipe(wav_path, hook=hook, **kwargs)
            else:
                raise DiarizationError(f"Diarization failed: {e}") from e
        return [Turn(float(t.start), float(t.end), str(spk)) for t, _, spk in diar.itertracks(yield_label=True)]
