import subprocess

from transcriber.core import audio
from transcriber.core.model import AsrResult, Turn, Word


class FakeEngine:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def transcribe(self, wav, opts, on_progress, should_cancel):
        self.calls += 1
        on_progress(0.5, "half")
        return AsrResult("en", 3.0, [Word("Hello", 0.0, 0.5), Word("there.", 0.5, 1.0), Word("Um,", 1.4, 1.5),
                                     Word("Hi.", 1.5, 2.0)])


class FakeDiarizer:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def run(self, wav, token, on_progress, should_cancel, *a):
        self.calls += 1
        if self.fail:
            raise RuntimeError("auth")
        on_progress(1.0, "")
        return [Turn(0, 1.2, "SPEAKER_00"), Turn(1.2, 3, "SPEAKER_01")]


def make_media(path, seconds=3, video=True):
    cmd = [audio.ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    if video:
        cmd += ["-f", "lavfi", "-i", f"color=c=black:s=64x64:d={seconds}", "-shortest"]
    subprocess.run(cmd + [str(path)], check=True)
    return path
