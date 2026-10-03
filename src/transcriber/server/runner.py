"""Job runners. ProcessRunner is the real one (long-lived spawned worker that keeps models warm);
InlineRunner runs in the caller's thread and exists so tests can use fake engines."""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Callable

ProgressCb = Callable[[str, float, str], None]
# result: ("done", None) | ("cancelled", None) | ("error", "message")
Result = tuple[str, str | None]


def execute_job(spec: dict, on_progress: ProgressCb, should_cancel: Callable[[], bool], diarizer=None) -> Result:
    from pathlib import Path

    from ..config import get_hf_token
    from ..core.engines import Cancelled
    from ..core.pipeline import JobSettings, run_pipeline

    try:
        run_pipeline(Path(spec["media_path"]), Path(spec["work_dir"]), JobSettings.from_dict(spec["settings"]),
                     on_progress, should_cancel, hf_token=get_hf_token(), diarizer=diarizer,
                     display_name=spec.get("display_name"))
        return "done", None
    except Cancelled:
        return "cancelled", None
    except Exception as e:  # surfaced to the user verbatim; the checkpoints keep finished stages
        return "error", f"{type(e).__name__}: {e}"


def _worker_main(conn, cancel_event, data_dir: str) -> None:
    os.environ["TRANSCRIBER_DATA"] = data_dir
    from ..core.device import quiet_third_party_warnings

    quiet_third_party_warnings()
    diarizer = None
    last = [0.0]

    while True:
        spec = conn.recv()
        if spec is None:
            return
        cancel_event.clear()

        def on_progress(stage, frac, msg, spec=spec):
            now = time.monotonic()
            if frac >= 1.0 or now - last[0] > 0.25:  # throttle chatter from tight engine loops
                last[0] = now
                conn.send(("progress", stage, frac, msg))

        if spec["settings"].get("diarize", True) and diarizer is None:
            from ..core.diarize import Diarizer

            diarizer = Diarizer()
        status, msg = execute_job(spec, on_progress, cancel_event.is_set, diarizer)
        conn.send((status, msg))


class ProcessRunner:
    CANCEL_GRACE = 20.0  # seconds to wait for cooperative cancel before killing the worker

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None
        self._cancel = None

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        parent, child = self._ctx.Pipe()
        self._cancel = self._ctx.Event()
        self._proc = self._ctx.Process(target=_worker_main, args=(child, self._cancel, self._data_dir), daemon=True)
        self._proc.start()
        child.close()
        self._conn = parent

    def kill(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.join(5)
        self._proc = self._conn = None

    def run(self, spec: dict, on_progress: ProgressCb, is_cancelled: Callable[[], bool]) -> Result:
        self._ensure()
        self._cancel.clear()
        self._conn.send(spec)
        cancel_at: float | None = None
        while True:
            if is_cancelled() and cancel_at is None:
                self._cancel.set()
                cancel_at = time.monotonic() + self.CANCEL_GRACE
            if cancel_at and time.monotonic() > cancel_at:
                self.kill()
                return "cancelled", None
            if self._conn.poll(0.3):
                try:
                    msg = self._conn.recv()
                except (EOFError, OSError):
                    self.kill()
                    return "error", "The transcription worker crashed (possibly out of memory). Retry to resume."
                if msg[0] == "progress":
                    on_progress(msg[1], msg[2], msg[3])
                else:
                    return msg[0], msg[1]
            elif not self._proc.is_alive():
                self.kill()
                return "error", "The transcription worker crashed (possibly out of memory). Retry to resume."

    def shutdown(self) -> None:
        try:
            if self._conn:
                self._conn.send(None)
            if self._proc:
                self._proc.join(3)
        except Exception:
            pass
        self.kill()


class InlineRunner:
    """Runs the job synchronously in the dispatcher thread (tests, or `transcriber run`)."""

    def __init__(self, diarizer=None) -> None:
        self.diarizer = diarizer

    def run(self, spec: dict, on_progress: ProgressCb, is_cancelled: Callable[[], bool]) -> Result:
        return execute_job(spec, on_progress, is_cancelled, self.diarizer)

    def kill(self) -> None:
        pass

    def shutdown(self) -> None:
        pass
