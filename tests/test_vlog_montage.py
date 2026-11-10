"""recipes/vlog-montage.yaml on its synthetic demo assets: validates (CPU), renders and passes check (GPU)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from montaj.cli import main

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "vlog-montage.yaml"


def test_recipe_declares_its_contract() -> None:
    spec = yaml.safe_load(RECIPE.read_text())
    assert spec["video"]["size"] == "1920x1080" and spec["video"]["color"]["working"] == "linear"
    assert set(spec["check"]["pacing"]) == {"max_shot", "min_per10s", "hook_text_by"}
    shots = spec["shots"]
    assert all("grade" in s for s in shots), "every shot carries a grade"
    assert any(isinstance(s.get("speed"), list) for s in shots), "at least one speed ramp"
    assert any(s.get("audio_tail") for s in shots), "at least one L cut"
    first = (RECIPE.with_suffix(".md")).read_text().splitlines()[0]
    assert first.startswith("vlog-montage:") and len(first) < 120


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`montaj new demo --recipe vlog-montage`: the recipe plus demo assets from recipes/vlog-montage-demo.sh."""
    proj = tmp_path_factory.mktemp("vlog") / "demo"
    rc = main(["new", str(proj), "--recipe", "vlog-montage"])
    log = proj / "build" / "new-demo.log"
    assert rc == 0, log.read_text() if log.is_file() else "no demo log"
    return proj


def test_recipe_validates_on_demo_assets(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-C", str(demo), "validate"]) == 0
    assert "OK montaj.yaml 1800f 60.00s" in capsys.readouterr().out


@pytest.mark.gpu
def test_recipe_renders_and_checks_ok(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-C", str(demo), "render"]) == 0
    assert main(["-C", str(demo), "check"]) == 0
    assert capsys.readouterr().out.rstrip().splitlines()[-1] == "OK check"
