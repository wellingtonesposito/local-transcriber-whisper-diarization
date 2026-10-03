"""Apple Silicon GPU engine (mlx-whisper). Word timestamps via `word_timestamps=True`."""
from __future__ import annotations

import sys

from ..audio import load_wav
from ..model import AsrResult, Word
from .base import AsrOptions, Cancelled, EngineUnavailable

_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def _patch_progress(on_progress, should_cancel):
    """mlx-whisper has no callback API; it drives a tqdm bar over audio frames. Hook that bar."""
    import tqdm as tqdm_mod

    class HookedTqdm(tqdm_mod.tqdm):
        def update(self, n=1):
            r = super().update(n)
            if should_cancel():
                raise Cancelled()
            if self.total:
                on_progress(min(1.0, self.n / self.total), "")
            return r

    mod = sys.modules.get("mlx_whisper.transcribe")
    if mod is None or not hasattr(mod, "tqdm"):
        return lambda: None
    original = mod.tqdm
    mod.tqdm = type("tqdm_ns", (), {"tqdm": HookedTqdm})  # module-like namespace exposing .tqdm
    return lambda: setattr(mod, "tqdm", original)


class MlxEngine:
    name = "mlx"

    def transcribe(self, wav_path, opts: AsrOptions, on_progress, should_cancel) -> AsrResult:
        try:
            import mlx_whisper
        except ImportError as e:
            raise EngineUnavailable("mlx-whisper is not installed (pip install 'transcriber[mlx]').") from e

        repo = _REPOS.get(opts.model_size, opts.model_size)  # allow a custom HF repo id
        restore = _patch_progress(on_progress, should_cancel)
        try:
            out = mlx_whisper.transcribe(
                load_wav(wav_path),
                path_or_hf_repo=repo,
                language=opts.language,
                word_timestamps=True,
                initial_prompt=opts.effective_prompt(),
                condition_on_previous_text=opts.condition_on_previous_text,
                verbose=False,
            )
        finally:
            restore()

        words: list[Word] = []
        for seg in out.get("segments", []):
            if seg.get("words"):
                words += [Word(w["word"], float(w["start"]), float(w["end"]), w.get("probability")) for w in seg["words"]]
            elif (seg.get("text") or "").strip():
                toks = seg["text"].split()
                step = (seg["end"] - seg["start"]) / max(1, len(toks))
                words += [Word(t, seg["start"] + i * step, seg["start"] + (i + 1) * step) for i, t in enumerate(toks)]
        duration = float(out["segments"][-1]["end"]) if out.get("segments") else 0.0
        return AsrResult(language=out.get("language", opts.language), duration=duration, words=words)
