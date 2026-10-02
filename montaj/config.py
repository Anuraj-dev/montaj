"""Machine config: `~/.config/montaj/config.toml` (`MONTAJ_CONFIG` overrides the path).

The keys name *other interpreters*, not libraries: ACE-Step and whisper each need a Python that
montaj's own venv does not have, and montaj must not import them (torch here, ctranslate2 there).
So montaj reads this file and runs both models as subprocesses. The file belongs to the machine:
montaj reads it, never writes it.

`ace_python` + `ace_site_packages` is the dangling-venv trick from
`claude-test/bday-video/music/py.sh` lines 3-4: ACE-Step's `.venv/bin/python` symlink is dead, so a
working CPython 3.12 runs with that venv's `site-packages` on `PYTHONPATH`.
"""
from __future__ import annotations

import os
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

CONFIG_ENV = "MONTAJ_CONFIG"
DEFAULT_PATH = Path.home() / ".config" / "montaj" / "config.toml"
DEFAULT_HINT = "~/.config/montaj/config.toml"
KEYS = ("ace_step_dir", "ace_python", "ace_site_packages", "whisper_python")
ACE_KEYS = ("ace_step_dir", "ace_python", "ace_site_packages")
WHISPER_KEYS = ("whisper_python",)


class ConfigError(RuntimeError):
    """The config file is missing, unreadable, or lacks a key a command needs.

    A RuntimeError so the CLI turns it into one `ERR <cmd>: ...` line like every other failure.
    """


@dataclass(frozen=True)
class Config:
    """A validated machine config. Frozen: one read, then shared, never mutated."""

    path: Path
    ace_step_dir: str
    ace_python: str
    ace_site_packages: str
    whisper_python: str

    def ace_argv(self, script: Path | str, *args: object) -> list[str]:
        """argv for an ACE-Step worker script."""
        return [self.ace_python, str(script), *(str(a) for a in args)]

    def whisper_argv(self, script: Path | str, *args: object) -> list[str]:
        """argv for a whisper worker script."""
        return [self.whisper_python, str(script), *(str(a) for a in args)]

    def ace_env(self) -> dict[str, str]:
        """Environment for ACE-Step: its site-packages first, the caller's PYTHONPATH kept after."""
        prev = os.environ.get("PYTHONPATH", "")
        joined = f"{self.ace_site_packages}{os.pathsep}{prev}" if prev else self.ace_site_packages
        return {**os.environ, "PYTHONPATH": joined}


def config_path() -> Path:
    """Where the config lives: `MONTAJ_CONFIG` wins, else the standard path."""
    given = os.environ.get(CONFIG_ENV, "").strip()
    return Path(given).expanduser() if given else DEFAULT_PATH


def _hint(path: Path) -> str:
    return DEFAULT_HINT if path == DEFAULT_PATH else path.as_posix()


def load(*, required: Iterable[str] = KEYS) -> Config:
    """Read and validate the keys needed by one command.

    The default keeps the old all-keys contract for callers such as `doctor`; commands pass the
    smallest set they actually use so a beat-only analysis does not require model configuration.
    """
    path = config_path()
    if not path.is_file():
        raise ConfigError(f"config: {path} missing (set it in {_hint(path)})")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"config: {path} is not readable TOML: {' '.join(str(exc).split())}") from None
    unknown = sorted(set(data) - set(KEYS))
    if unknown:
        raise ConfigError(f"config: unknown key(s) {', '.join(unknown)} in {path} (have: {', '.join(KEYS)})")
    needed = tuple(required)
    invalid = sorted(set(needed) - set(KEYS))
    if invalid:
        raise ConfigError(f"config: unknown required key(s) {', '.join(invalid)}")
    missing = [key for key in needed if not str(data.get(key, "")).strip()]
    if missing:
        raise ConfigError(f"config: {missing[0]} missing (set it in {_hint(path)})")
    return Config(
        path=path,
        ace_step_dir=str(data.get("ace_step_dir", "")),
        ace_python=str(data.get("ace_python", "")),
        ace_site_packages=str(data.get("ace_site_packages", "")),
        whisper_python=str(data.get("whisper_python", "")),
    )
