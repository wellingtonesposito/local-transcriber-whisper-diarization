"""Paths and secrets. Everything lives under one data directory (default ~/Transcriber)."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def data_dir() -> Path:
    d = Path(os.environ.get("TRANSCRIBER_DATA") or Path.home() / "Transcriber").expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def project_dir(project_id: str) -> Path:
    return data_dir() / "projects" / project_id


def media_dir(project_id: str, media_id: str) -> Path:
    return project_dir(project_id) / "media" / media_id


def _token_file() -> Path:
    return data_dir() / "secrets" / "hf_token"


def get_hf_token() -> str | None:
    """Env var wins; otherwise the token saved through the Setup page. Never logged or rendered."""
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if env:
        return env.strip()
    f = _token_file()
    return f.read_text().strip() or None if f.exists() else None


def has_hf_token() -> bool:
    return bool(get_hf_token())


def save_hf_token(token: str) -> None:
    f = _token_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch(mode=0o600)
    f.chmod(0o600)
    f.write_text(token.strip())


def clear_hf_token() -> None:
    f = _token_file()
    if f.exists():
        f.unlink()
