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
    assert all("at_event" in s and "clip_in" not in s for s in shots), "every shot picks its in-point from the events"
    assert any(str(s.get("pre", "")).startswith("-") for s in shots), "a shot that starts after its event"
    hits = {k: v for k, v in spec["markers"].items() if k.startswith("hit")}
    assert len(hits) == 5 and all(v.startswith("shown:match/") for v in hits.values()), "kills on the film clock"
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


def test_kill_markers_land_where_the_hand_placed_beats_were(demo: Path) -> None:
    from montaj.spec import load_spec
    from montaj.timeline import resolve

    tl = resolve(load_spec(demo / "montaj.yaml"), demo)
    beat = 30  # 120 BPM at 60 fps
    assert [tl.markers[f"hit{n}"] for n in range(1, 6)] == [8 * beat, 12 * beat, 16 * beat, 20 * beat, 24 * beat]
    assert tl.shots[-1].spec.clip_in == "52s"  # win at 50 s, pre -2s


@pytest.mark.gpu
def test_recipe_renders_and_checks_ok(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-C", str(demo), "render"]) == 0
    assert main(["-C", str(demo), "check"]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().splitlines()[-1] == "OK check"
    assert "ignored as intended: fade to black" in out
