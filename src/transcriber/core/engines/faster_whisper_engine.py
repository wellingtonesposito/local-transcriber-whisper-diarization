from __future__ import annotations

from ..audio import load_wav
from ..model import AsrResult, Word
from .base import AsrOptions, Cancelled, EngineUnavailable


def _device() -> tuple[str, str]:
    """(device, compute_type). 'auto' lets CTranslate2 pick the best type the GPU supports."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "auto"
    except Exception:
        pass
    return "cpu", "int8"


class FasterWhisperEngine:
    name = "faster-whisper"

    def __init__(self) -> None:
        self._models: dict[tuple, object] = {}

    def _model(self, size: str):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise EngineUnavailable("faster-whisper is not installed (pip install 'transcriber[asr]').") from e
        device, ctype = _device()
        key = (size, device, ctype)
        if key not in self._models:
            self._models.clear()  # keep one model in memory
            self._models[key] = WhisperModel(size, device=device, compute_type=ctype)
        return self._models[key]

    def transcribe(self, wav_path, opts: AsrOptions, on_progress, should_cancel) -> AsrResult:
        model = self._model(opts.model_size)
        segments, info = model.transcribe(
            load_wav(wav_path),
            language=opts.language,
            beam_size=opts.beam_size,
            vad_filter=opts.vad,
            word_timestamps=True,
            initial_prompt=opts.effective_prompt(),
            condition_on_previous_text=opts.condition_on_previous_text,
        )
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        words: list[Word] = []
        for seg in segments:  # lazy generator: this loop is the actual transcription
            if should_cancel():
                raise Cancelled()
            if seg.words:
                words += [Word(w.word, float(w.start), float(w.end), float(w.probability)) for w in seg.words]
            elif (seg.text or "").strip():  # no word timings: spread the segment text evenly
                toks = seg.text.split()
                step = (seg.end - seg.start) / max(1, len(toks))
                words += [Word(t, seg.start + i * step, seg.start + (i + 1) * step) for i, t in enumerate(toks)]
            if duration:
                on_progress(min(1.0, seg.end / duration), f"{int(seg.end // 60)} of {int(duration // 60)} min")
        return AsrResult(language=getattr(info, "language", opts.language), duration=duration, words=words)
