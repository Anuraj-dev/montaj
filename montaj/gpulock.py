"""One GPU job at a time on this box.

ACE-Step and the CUDA renderer share 8 GB; run together and one of them OOMs. `hold()` takes an
exclusive `fcntl` lock on `~/.cache/montaj/gpu.lock` (`MONTAJ_GPU_LOCK` overrides the path for
tests), so `render` and `music gen|analyze` queue instead of colliding. The loser waits and says so
once in the log — the job is then silent, exactly like the other montaj commands.

The lock file is created if absent and never unlinked: unlinking it would hand a third process a
fresh inode to lock while the real holder still owns the old one.
"""
from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOCK_ENV = "MONTAJ_GPU_LOCK"
DEFAULT_LOCK = Path.home() / ".cache" / "montaj" / "gpu.lock"
POLL_S = 0.2


def lock_path() -> Path:
    """Where the GPU lock lives: `MONTAJ_GPU_LOCK` wins, else the standard cache path."""
    given = os.environ.get(LOCK_ENV, "").strip()
    return Path(given).expanduser() if given else DEFAULT_LOCK


def _note(log: Path | str | None, text: str) -> None:
    """One line in the caller's log, or nowhere when there is no log."""
    if log is None:
        return
    path = Path(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{text}\n")


@contextmanager
def hold(log: Path | str | None = None) -> Iterator[Path]:
    """Hold the GPU lock for the block; yield the lock path.

    Blocks (polling) until the current holder exits. On release the lock goes back so the next
    waiter in — kernel-held state, not scheduling, so a killed holder frees it too.
    """
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        waited = False
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not waited:
                    waited = True
                    _note(log, f"waiting for GPU ({path} is held by another montaj job)")
                time.sleep(POLL_S)
        _note(log, f"gpu lock: {os.getpid()} holds {path}")
        try:
            yield path
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)