"""Persistent creative defaults at `~/.config/montaj/taste.md` (`MONTAJ_TASTE` overrides)."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

TASTE_ENV = "MONTAJ_TASTE"
DEFAULT_PATH = Path.home() / ".config" / "montaj" / "taste.md"
DEFAULTS = """\
- photos never warp/drift unless the recipe asks
- motion lives in transitions
- cuts land on the beat grid
- text is short and centred
- end on black
"""


def taste_path() -> Path:
    given = os.environ.get(TASTE_ENV, "").strip()
    return Path(given).expanduser() if given else DEFAULT_PATH


def _ensure() -> Path:
    path = taste_path()
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULTS, encoding="utf-8")
    return path


def _n_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def read() -> str:
    return _ensure().read_text(encoding="utf-8")


def add(line: str) -> int:
    path = _ensure()
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text += "\n"
    text += f"- {line} ({date.today().isoformat()})\n"
    path.write_text(text, encoding="utf-8")
    return _n_lines(text)
