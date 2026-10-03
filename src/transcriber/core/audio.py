"""ffmpeg helpers. Uses a system ffmpeg if present, else the binary bundled by imageio-ffmpeg."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class AudioError(RuntimeError):
    pass


def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover - environment dependent
        raise AudioError("ffmpeg not found. Install it (e.g. `brew install ffmpeg`) or `pip install imageio-ffmpeg`.") from e


_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe(path: str | Path) -> dict:
    """Return {"duration": seconds|None, "has_audio": bool} by parsing `ffmpeg -i` output."""
    r = subprocess.run([ffmpeg_exe(), "-nostdin", "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True, errors="replace")
    err = r.stderr
    m = _DURATION.search(err)
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None
    has_audio = bool(re.search(r"Stream #\d+:\d+.*Audio:", err))
    if duration is None and not has_audio:
        raise AudioError(f"Not a readable audio/video file: {Path(path).name}")
    return {"duration": duration, "has_audio": has_audio}


def load_wav(path: str | Path):
    """Read our 16 kHz mono 16-bit WAV as float32 samples.

    Engines get an array instead of a path so they never depend on a PATH `ffmpeg`
    (mlx-whisper) or on the PyAV build that faster-whisper would otherwise use to decode.
    """
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise AudioError("expected 16 kHz mono 16-bit WAV (use extract_wav first)")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def extract_wav(src: str | Path, dst: str | Path) -> Path:
    """Convert any media file to 16 kHz mono WAV at `dst` (a path we own, so overwriting is safe)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.wav")
    cmd = [ffmpeg_exe(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise AudioError(f"ffmpeg could not extract audio from {Path(src).name}: {r.stderr.strip()[-500:]}")
    tmp.replace(dst)
    return dst
