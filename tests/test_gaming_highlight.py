"""recipes/gaming-highlight.yaml on its synthetic demo assets: validates (CPU), renders and passes check (GPU)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from montaj.cli import main

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "gaming-highlight.yaml"


def test_recipe_declares_its_contract() -> None:
    spec = yaml.safe_load(RECIPE.read_text())
    assert spec["video"]["size"] == "1920x1080"
    assert set(spec["check"]["pacing"]) == {"max_shot", "min_per10s", "hook_text_by"}
    assert spec["captions"]["preset"] == "comic"
    assert spec["audio"]["sfx"] == "auto"
    shots = spec["shots"]
    gameplay = [s for s in shots if "at_event" in s]
    assert len(gameplay) >= len(shots) - 1, "gameplay shots pick their in-point from the event log"
    assert any("pre" in s for s in gameplay)
    assert any("freeze" in s and "camera" in s for s in shots), "a freeze-and-zoom key moment"
    kinds = {e["type"] for s in shots for e in s.get("effects", [])}
    assert {"shake", "chroma", "bloom"} <= kinds
    first = RECIPE.with_suffix(".md").read_text().splitlines()[0]
    assert first.startswith("gaming-highlight:") and len(first) < 120


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`montaj new demo --recipe gaming-highlight`: the recipe plus assets from recipes/gaming-highlight-demo.sh."""
    proj = tmp_path_factory.mktemp("gaming") / "demo"
    rc = main(["new", str(proj), "--recipe", "gaming-highlight"])
    log = proj / "build" / "new-demo.log"
    assert rc == 0, log.read_text() if log.is_file() else "no demo log"
    return proj


def test_recipe_validates_on_demo_assets(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (demo / "assets" / "match.events.json").is_file(), "ingest-rec wrote the event log"
    assert main(["-C", str(demo), "validate"]) == 0
    assert "OK montaj.yaml 1800f 30.00s" in capsys.readouterr().out


@pytest.mark.gpu
def test_recipe_renders_and_checks_ok(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-C", str(demo), "render"]) == 0
    assert main(["-C", str(demo), "check"]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().splitlines()[-1] == "OK check"
    assert "ignored as intended: fade to black" in out
