"""#62: `montaj new` puts the recipe's song at `music/compose.py`, with or without `--photos`, so the agent
Reads and Edits it instead of writing one."""
from __future__ import annotations

from pathlib import Path

from montaj.cli import RECIPES, main
from tests.test_spec import _jpeg


def _new(tmp: Path, recipe: str) -> Path:
    (tmp / "photos").mkdir(exist_ok=True)
    _jpeg(tmp / "photos", "a")
    dest = tmp / recipe
    assert main(["new", str(dest), "--recipe", recipe, "--photos", str(tmp / "photos")]) == 0
    return dest


def test_new_copies_the_recipe_song(tmp_path: Path) -> None:
    dest = _new(tmp_path, "tech-launch")
    want = (RECIPES / "music" / "tech-launch-compose.py").read_text()
    assert (dest / "music" / "compose.py").read_text() == want


def test_a_recipe_without_a_song_gets_none(tmp_path: Path) -> None:
    assert not (RECIPES / "music" / "birthday-short-compose.py").exists()
    assert not (_new(tmp_path, "birthday-short") / "music" / "compose.py").exists()
