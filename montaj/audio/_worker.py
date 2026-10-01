"""Subprocess seam for the external-model workers.

ACE-Step and faster_whisper live in venvs montaj neither controls nor imports (torch here,
ctranslate2 there). Each `montaj/audio/_*_worker.py` is a standalone script: montaj passes it flags
and reads JSON back over stdout, everything else the worker prints goes to `build/*.log`.

Why the workers own a private fd: model libraries print progress on stdout, which would corrupt the
JSON lines. `_ace_worker`/`_whisper_worker` point fd 1 at stderr before loading anything and emit
their results on the saved descriptor, so a chatty library cannot break the protocol.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from montaj.encode import tail_lines


class AudioError(RuntimeError):
    """A worker failed, or said something montaj cannot read.

    `tail` (the worker's last stderr lines) is part of the message, not just an attribute: the CLI
    prints `str(exc)` as the single ERR line, so a reason kept outside the message is a reason lost.
    """

    def __init__(self, what: str, tail: str = "", *, lines: list[str] | None = None) -> None:
        super().__init__(f"{what}: {tail.strip()[-800:]}" if tail else what)
        self.what = what
        self.tail = tail
        self.lines = lines or []


def append_log(log: Path | str, cmd: list[str], stdout: str, stderr: str) -> Path:
    """Record the invocation and both streams in `log`; return the log path."""
    path = Path(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n{stdout}{stderr}")
    return path


def run_worker(
    desc: str,
    cmd: list[str],
    *,
    log: Path | str,
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
    timeout: float | None = None,
) -> list[str]:
    """Run one worker to completion; return its non-empty stdout lines.

    Protocol-sized stdout (a few JSON/segment lines) is read in one go; that is safe here and would
    not be for a worker printing megabytes.
    """
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        append_log(log, cmd, "", f"\ntimed out after {timeout:g}s\n")
        raise AudioError(f"{desc}: timed out after {timeout:g}s") from None
    except OSError as exc:
        raise AudioError(f"{desc}: cannot run {cmd[0]}: {exc}") from None
    append_log(log, cmd, proc.stdout, proc.stderr)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0:
        raise AudioError(f"{desc} failed ({proc.returncode})", tail_lines(proc.stderr, 6), lines=lines)
    return lines


def parse_candidates(lines: list[str], desc: str) -> list[dict[str, Any]]:
    """The `CAND <json>` lines of a `music gen` worker, in order. Any other line is a bug."""
    out: list[dict[str, Any]] = []
    for line in lines:
        if not line.startswith("CAND "):
            raise AudioError(f"{desc}: unexpected worker output {_one_line(line)!r}")
        try:
            item = json.loads(line[5:])
        except json.JSONDecodeError as exc:
            raise AudioError(f"{desc}: unreadable candidate line ({exc})") from None
        if not isinstance(item, dict) or "seed" not in item or "path" not in item:
            raise AudioError(f"{desc}: candidate line has no seed/path: {_one_line(line)!r}")
        out.append(item)
    return out


def _one_line(text: str) -> str:
    return " ".join(text.split())