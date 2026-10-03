"""Queue dispatcher: runs queued jobs one at a time and mirrors progress to the DB and event bus."""
from __future__ import annotations

import json
import logging
import threading
import time

from ..config import media_dir
from .db import DB
from .events import EventBus
from .summary import transcript_summary


log = logging.getLogger(__name__)


class JobManager:
    def __init__(self, db: DB, bus: EventBus, runner) -> None:
        self.db, self.bus, self.runner = db, bus, runner
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._cancel_media: set[str] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="job-dispatcher", daemon=True)

    # lifecycle --------------------------------------------------------------------------
    def start(self) -> None:
        self.db.recover_interrupted()
        self._thread.start()
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self.runner.shutdown()

    # public API -------------------------------------------------------------------------
    def enqueue(self, media_id: str, settings: dict) -> int | None:
        if self.db.active_job_for_media(media_id):
            return None
        jid = self.db.enqueue(media_id, settings)
        self._emit(media_id)
        self._wake.set()
        return jid

    def cancel(self, media_id: str) -> None:
        job = self.db.active_job_for_media(media_id)
        if not job:
            return
        if job["status"] == "queued":
            self.db.update_job(job["id"], status="cancelled", finished_at=time.time())
            self.db.update_media(media_id, status="cancelled", message="Cancelled", stage="", progress=0)
            self._emit(media_id)
        else:
            with self._lock:
                self._cancel_media.add(media_id)
            self.db.update_media(media_id, message="Cancelling…")
            self._emit(media_id)

    # internals --------------------------------------------------------------------------
    def _emit(self, media_id: str) -> None:
        m = self.db.get_media(media_id)
        if m:
            self.bus.publish({"type": "media", "media_id": media_id, "project_id": m["project_id"],
                              "status": m["status"], "stage": m["stage"], "progress": m["progress"],
                              "message": m["message"]})

    def _is_cancelled(self, media_id: str) -> bool:
        with self._lock:
            return media_id in self._cancel_media

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.db.next_queued()
            if job is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            try:
                self._run(job)
            except Exception as e:  # never let the dispatcher die
                self._finish(job, "error", f"{type(e).__name__}: {e}")

    def _run(self, job: dict) -> None:
        mid = job["media_id"]
        media = self.db.get_media(mid)
        if media is None:  # deleted while queued
            self.db.update_job(job["id"], status="cancelled", finished_at=time.time())
            return
        with self._lock:
            self._cancel_media.discard(mid)
        self.db.update_job(job["id"], status="running", started_at=time.time())
        self.db.update_media(mid, status="running", stage="prepare", progress=0, message="Starting", error="")
        self._emit(mid)

        wd = media_dir(media["project_id"], mid)
        spec = {"job_id": job["id"], "media_path": str(wd / f"original{media['ext']}"), "work_dir": str(wd),
                "settings": json.loads(job["settings"]), "display_name": media["filename"]}

        def on_progress(stage: str, frac: float, msg: str) -> None:
            self.db.update_media(mid, stage=stage, progress=round(frac, 4), message=msg or "")
            self._emit(mid)

        status, msg = self.runner.run(spec, on_progress, lambda: self._is_cancelled(mid))
        self._finish(job, status, msg)

    def _summary(self, job: dict) -> str:
        m = self.db.get_media(job["media_id"])
        try:
            return json.dumps(transcript_summary(media_dir(m["project_id"], m["id"]) / "transcript.json"))
        except Exception:  # not fatal: the transcripts page rebuilds a missing summary on demand
            log.exception("could not summarise transcript for %s", job["media_id"])
            return ""

    def _finish(self, job: dict, status: str, msg: str | None) -> None:
        mid = job["media_id"]
        if self.db.get_media(mid) is None:
            self.db.update_job(job["id"], status="cancelled", finished_at=time.time())
            return
        if status == "done":
            self.db.update_job(job["id"], status="done", finished_at=time.time())
            self.db.update_media(mid, status="done", stage="", progress=1.0, message="", error="",
                                 reviewed=0, summary=self._summary(job))
        elif status == "cancelled":
            self.db.update_job(job["id"], status="cancelled", finished_at=time.time())
            self.db.update_media(mid, status="cancelled", stage="", progress=0, message="Cancelled")
        else:
            self.db.update_job(job["id"], status="failed", finished_at=time.time(), error=msg or "")
            self.db.update_media(mid, status="failed", stage="", message="", error=msg or "Unknown error")
        with self._lock:
            self._cancel_media.discard(mid)
        self._emit(mid)
