"""Small SQLite store: projects, media files, jobs. One connection guarded by a lock."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT DEFAULT '',
    settings TEXT DEFAULT '{}', created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS media (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename TEXT NOT NULL, ext TEXT NOT NULL, size INTEGER DEFAULT 0, duration REAL,
    status TEXT NOT NULL DEFAULT 'uploaded', stage TEXT DEFAULT '', progress REAL DEFAULT 0,
    message TEXT DEFAULT '', error TEXT DEFAULT '', created_at REAL NOT NULL,
    reviewed INTEGER NOT NULL DEFAULT 0, summary TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, media_id TEXT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    status TEXT NOT NULL, settings TEXT NOT NULL, created_at REAL NOT NULL,
    started_at REAL, finished_at REAL, error TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_media_project ON media(project_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class DB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(path, check_same_thread=False)
        self._c.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._c.execute("PRAGMA journal_mode=WAL")
            self._c.execute("PRAGMA foreign_keys=ON")
            self._c.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the first release to databases created by older versions."""
        cols = {r["name"] for r in self._c.execute("PRAGMA table_info(media)")}
        for name, ddl in (("reviewed", "INTEGER NOT NULL DEFAULT 0"), ("summary", "TEXT NOT NULL DEFAULT ''")):
            if name not in cols:
                self._c.execute(f"ALTER TABLE media ADD COLUMN {name} {ddl}")
        self._c.commit()

    def _q(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._c.execute(sql, args).fetchall()]

    def _x(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            cur = self._c.execute(sql, args)
            self._c.commit()
            return cur.lastrowid or cur.rowcount

    # projects ---------------------------------------------------------------------------
    def create_project(self, name: str, description: str = "") -> dict:
        pid = new_id()
        self._x("INSERT INTO projects(id,name,description,created_at) VALUES (?,?,?,?)",
                (pid, name.strip(), description.strip(), time.time()))
        return self.get_project(pid)

    def get_project(self, pid: str) -> dict | None:
        rows = self._q("SELECT * FROM projects WHERE id=?", (pid,))
        return self._project(rows[0]) if rows else None

    def list_projects(self) -> list[dict]:
        rows = self._q("""SELECT p.*, (SELECT COUNT(*) FROM media m WHERE m.project_id=p.id) AS n_files,
                          (SELECT COUNT(*) FROM media m WHERE m.project_id=p.id AND m.status='done') AS n_done,
                          (SELECT COALESCE(SUM(duration), 0) FROM media m WHERE m.project_id=p.id) AS total_duration
                          FROM projects p ORDER BY created_at DESC""")
        return [self._project(r) for r in rows]

    @staticmethod
    def _project(r: dict) -> dict:
        r["settings"] = json.loads(r.get("settings") or "{}")
        return r

    def update_project(self, pid: str, **fields: Any) -> None:
        if "settings" in fields:
            fields["settings"] = json.dumps(fields["settings"])
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE projects SET {cols} WHERE id=?", (*fields.values(), pid))

    def delete_project(self, pid: str) -> None:
        self._x("DELETE FROM projects WHERE id=?", (pid,))

    # media ------------------------------------------------------------------------------
    def create_media(self, mid: str, project_id: str, filename: str, ext: str) -> None:
        self._x("INSERT INTO media(id,project_id,filename,ext,created_at) VALUES (?,?,?,?,?)",
                (mid, project_id, filename, ext, time.time()))

    def get_media(self, mid: str) -> dict | None:
        rows = self._q("SELECT * FROM media WHERE id=?", (mid,))
        return rows[0] if rows else None

    def list_media(self, project_id: str) -> list[dict]:
        return self._q("SELECT * FROM media WHERE project_id=? ORDER BY created_at, filename", (project_id,))

    def update_media(self, mid: str, **fields: Any) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE media SET {cols} WHERE id=?", (*fields.values(), mid))

    def delete_media(self, mid: str) -> None:
        self._x("DELETE FROM media WHERE id=?", (mid,))

    # jobs -------------------------------------------------------------------------------
    def enqueue(self, media_id: str, settings: dict) -> int:
        jid = self._x("INSERT INTO jobs(media_id,status,settings,created_at) VALUES (?,?,?,?)",
                      (media_id, "queued", json.dumps(settings), time.time()))
        self.update_media(media_id, status="queued", stage="", progress=0, message="Waiting in queue", error="")
        return jid

    def next_queued(self) -> dict | None:
        rows = self._q("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1")
        return rows[0] if rows else None

    def update_job(self, jid: int, **fields: Any) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), jid))

    def get_job(self, jid: int) -> dict | None:
        rows = self._q("SELECT * FROM jobs WHERE id=?", (jid,))
        return rows[0] if rows else None

    def active_job_for_media(self, mid: str) -> dict | None:
        rows = self._q("SELECT * FROM jobs WHERE media_id=? AND status IN ('queued','running') ORDER BY id DESC LIMIT 1", (mid,))
        return rows[0] if rows else None

    def recover_interrupted(self) -> None:
        """After a crash/restart: running jobs become failed(interrupted); queued ones stay queued."""
        for j in self._q("SELECT id, media_id FROM jobs WHERE status='running'"):
            self.update_job(j["id"], status="failed", error="Interrupted (app restarted). Retry to resume.")
            self.update_media(j["media_id"], status="failed", error="Interrupted (app restarted). Retry to resume.", message="")
