"""recipes/explainer.yaml on its demo (code-drawn shots + composed score): validates (CPU), renders and checks (GPU)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from montaj.cli import main

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "explainer.yaml"


def test_recipe_declares_its_contract() -> None:
    spec = yaml.safe_load(RECIPE.read_text())
    assert spec["video"]["size"] == "1920x1080"
    assert set(spec["check"]["pacing"]) == {"max_shot", "min_per10s", "hook_text_by"}
    shots = spec["shots"]
    assert all("custom" in s for s in shots), "every picture is code-drawn"
    for rel in {s["custom"] for s in shots} | {s["in"]["src"] for s in shots if s.get("in", {}).get("src")}:
        assert (RECIPE.parent / rel).is_file(), rel
    blocks = spec["text"]
    assert sum("lower_third" in b for b in blocks) >= 3, "one lower third per part"
    kinetic = [ln for b in blocks for ln in b.get("lines", []) if {"in", "idle", "emphasis"} & set(ln)]
    assert len(kinetic) >= 4, "kinetic type: animated entries, idle loops, emphasis"
    first = RECIPE.with_suffix(".md").read_text().splitlines()[0]
    assert first.startswith("explainer:") and len(first) < 120


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`montaj new demo --recipe explainer`: the recipe, its draw files and the composed score."""
    proj = tmp_path_factory.mktemp("explainer") / "demo"
    rc = main(["new", str(proj), "--recipe", "explainer"])
    log = proj / "build" / "new-demo.log"
    assert rc == 0, log.read_text() if log.is_file() else "no demo log"
    return proj


def test_recipe_validates_on_demo_assets(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-C", str(demo), "validate"]) == 0
    assert "OK montaj.yaml 1350f 45.00s" in capsys.readouterr().out


@pytest.mark.gpu
def test_recipe_renders_and_checks_ok(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-C", str(demo), "render"]) == 0
    assert main(["-C", str(demo), "check"]) == 0
    assert capsys.readouterr().out.rstrip().splitlines()[-1] == "OK check"
