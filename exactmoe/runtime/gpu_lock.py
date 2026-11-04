"""The shared-machine GPU lock (SecurityWatch rule).

Before any GPU job or model server that uses the GPU, create the lock file exclusively; delete it after.
Never delete a lock this process does not hold: a stale lock is reported, not removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
from typing import Any

DEFAULT_LOCK = Path.home() / "SecurityWatch" / "SHARED-RESOURCES" / "locks" / "GPU.lock"


class GpuLockHeld(RuntimeError):
    """Another holder owns the GPU lock."""

    def __init__(self, path: Path, holder: dict[str, Any] | None):
        self.path = path
        self.holder = holder
        who = (holder or {}).get("holder", "unknown holder")
        super().__init__(f"GPU lock {path} is held by {who}; not taking the GPU")


def read_holder(path: Path = DEFAULT_LOCK) -> dict[str, Any] | None:
    """The current holder record, or None when the lock is free. An unreadable lock is still a lock."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return {"holder": "unreadable lock file"}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"holder": text.strip()[:200]}
    except ValueError:
        return {"holder": text.strip()[:200] or "empty lock file"}


@dataclass
class GpuLock:
    """Exclusive, holder-checked lock. Use as a context manager or call acquire()/release()."""

    holder: str
    purpose: str
    path: Path = DEFAULT_LOCK
    _token: str = ""

    def acquire(self) -> "GpuLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._token = f"{socket.gethostname()}:{os.getpid()}:{datetime.now(timezone.utc).isoformat()}"
        record = {"holder": self.holder, "purpose": self.purpose, "pid": os.getpid(), "token": self._token,
                  "acquired_utc": datetime.now(timezone.utc).isoformat()}
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            raise GpuLockHeld(self.path, read_holder(self.path)) from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh)
        return self

    def held(self) -> bool:
        current = read_holder(self.path)
        return bool(self._token) and bool(current) and current.get("token") == self._token

    def release(self) -> None:
        """Delete the lock only if it is still ours."""
        if self.held():
            self.path.unlink()
        self._token = ""

    def __enter__(self) -> "GpuLock":
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()
