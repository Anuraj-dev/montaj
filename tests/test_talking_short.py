"""recipes/talking-short.yaml on its synthetic demo: validates (CPU), renders and passes check (GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from montaj.cli import main

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "talking-short.yaml"


def test_recipe_loads_and_declares_pacing() -> None:
    spec = yaml.safe_load(RECIPE.read_text())
    assert spec["video"]["size"] == "1080x1920" and spec["video"]["fps"] == 30
    assert spec["check"]["pacing"]["hook_text_by"] == "1s"
    assert spec["audio"]["sfx"] == "auto" and spec["captions"]["preset"] == "punch"


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`montaj new demo --recipe talking-short`: synthetic voice/presenter/b-roll, cut, film-time words."""
    proj = tmp_path_factory.mktemp("talk") / "demo"
    rc = main(["new", str(proj), "--recipe", "talking-short"])
    log = proj / "build" / "new-demo.log"
    assert rc == 0, log.read_text() if log.is_file() else "no demo log"
    return proj


def test_demo_cuts_and_film_words(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cuts = json.loads((demo / "assets" / "voice.cuts.json").read_text())
    words = json.loads((demo / "talk.words.json").read_text())["words"]
    assert len(cuts["spans"]) == 14
    assert words[0]["w"].lower().startswith("stop") and words[0]["s"] < 0.2
    assert all(a["s"] <= b["s"] for a, b in zip(words, words[1:])), "film-time words must be in order"
    assert main(["-C", str(demo), "validate"]) == 0
    assert "OK montaj.yaml 1084f 36.13s" in capsys.readouterr().out


@pytest.mark.gpu
def test_recipe_renders_and_checks_ok(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-C", str(demo), "render"]) == 0
    assert main(["-C", str(demo), "check"]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().splitlines()[-1] == "OK check"
    assert "hook first_text=0.0s" in out
