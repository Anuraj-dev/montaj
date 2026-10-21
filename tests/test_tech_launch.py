"""recipes/tech-launch.yaml on its synthetic demo assets: validates (CPU), renders and passes check (GPU)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from montaj.cli import main

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "tech-launch.yaml"


def test_recipe_loads_and_declares_expectations() -> None:
    spec = yaml.safe_load(RECIPE.read_text())
    assert spec["video"]["size"] == "1920x1080" and spec["video"]["fps"] == 60 and spec["video"]["bpm"] == 120
    assert spec["check"]["expect"], "the intended freeze/black spans must be declared"


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`montaj new demo --recipe tech-launch`: the recipe plus demo assets from recipes/tech-launch-demo.sh."""
    proj = tmp_path_factory.mktemp("tech") / "demo"
    rc = main(["new", str(proj), "--recipe", "tech-launch"])
    log = proj / "build" / "new-demo.log"
    assert rc == 0, log.read_text() if log.is_file() else "no demo log"
    return proj


def test_recipe_validates_on_demo_assets(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-C", str(demo), "validate"]) == 0
    assert "OK montaj.yaml 840f 14.00s" in capsys.readouterr().out


@pytest.mark.gpu
def test_recipe_renders_and_checks_ok(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-C", str(demo), "render"]) == 0
    assert main(["-C", str(demo), "check"]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().splitlines()[-1] == "OK check"
    assert "ignored as intended: freeze on the drop shot" in out
    assert "ignored as intended: fade to black" in out
