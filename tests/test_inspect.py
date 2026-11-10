"""`montaj inspect`: text geometry from the FramePlan, contrast from one frame, `--problems`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from montaj.cli import main
from montaj.inspector import Inspector

_CLEAN = '{text: "Hello world", y: 900, at: 0s, shadow: false}'
_OFF = '{text: "Far away", y: 1500, at: 0s, align: left, indent: 30}'


def _project(root: Path, lines: str, size: str = "270x480") -> Path:
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "montaj.yaml").write_text(
        f"video:\n  size: {size}\n  fps: 30\n  bpm: 76\n  look: golden-film\nassets: assets\nshots:\n"
        f'  - card: {{color: "#223344", text: [{lines}]}}\n    hold: 90f\n'
    )
    return root


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _project(tmp_path_factory.mktemp("clean"), _CLEAN)


@pytest.fixture(scope="module")
def off(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _project(tmp_path_factory.mktemp("off"), f"{_CLEAN}, {_OFF}")


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = main(argv)
    return code, capsys.readouterr().out


def test_at_json_reports_text_box(clean: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(["-C", str(clean), "inspect", "--at", "2s", "--fast", "--json"], capsys)
    assert code == 0
    fr = json.loads(out)
    assert (fr["t"], fr["frame"], fr["shot"], fr["transition"]) == (2.0, 60, 0, None)
    [el] = fr["text"]
    x0, y0, x1, y1 = el["box"]
    assert el["text"] == "Hello world" and 0 < x0 < x1 < 270 and 0 < y0 < y1 < 480
    assert abs((x0 + x1) / 2 - 135) < 1.5  # centred text
    assert el["off_frame_px"] == 0 and el["safe"] and el["contrast"] is None and el["over_face"] is None


def test_early_frame_shows_only_revealed_words(clean: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, late = _run(["-C", str(clean), "inspect", "--at", "2s", "--fast", "--json"], capsys)
    _, early = _run(["-C", str(clean), "inspect", "--at", "0.07s", "--fast", "--json"], capsys)
    w = lambda o: json.loads(o)["text"][0]["box"]
    assert w(early)[2] - w(early)[0] < w(late)[2] - w(late)[0]


def test_text_lines_range_and_shot(clean: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(["-C", str(clean), "inspect", "--at", "2s", "--fast"], capsys)
    lines = out.splitlines()
    assert code == 0 and lines[-1] == "OK inspect 1 frames 1 text elements" and "card#0.0" in lines[1]
    code, out = _run(["-C", str(clean), "inspect", "--range", "1s", "2s", "--every", "10f", "--fast", "--json"], capsys)
    assert [f["frame"] for f in json.loads(out)] == [30, 40, 50, 60]
    code, out = _run(["-C", str(clean), "inspect", "--shot", "0", "--fast", "--json"], capsys)
    assert json.loads(out)["frame"] == 44
    code, out = _run(["-C", str(clean), "inspect", "--shot", "5", "--fast"], capsys)
    assert code == 1 and out.startswith("ERR inspect")
    code, out = _run(["-C", str(clean), "inspect", "--fast"], capsys)
    assert code == 1 and out.startswith("ERR inspect: give exactly one")


def test_problems_clean_and_off_frame(clean: Path, off: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(["-C", str(clean), "inspect", "--problems", "--fast"], capsys)
    assert (code, out.strip()) == (0, "OK inspect 0 problems")
    code, out = _run(["-C", str(off), "inspect", "--problems", "--fast"], capsys)
    lines = out.strip().splitlines()
    assert code == 0 and lines[-1] == "WARN inspect 1 problems"
    assert lines[0].startswith("PROBLEM card#0.1 ") and "off_frame" in lines[0] and "'Far away'" in lines[0]


@pytest.mark.gpu
def test_boxes_match_rendered_pixels(tmp_path: Path) -> None:
    """Ink box from `describe` vs the bounding box of the glyph pixels the renderer draws (within 2 px)."""
    root = _project(tmp_path, _CLEAN, size="540x960")
    ins = Inspector(root, fast=True)
    f = 60
    el = ins.at(f)["text"][0]
    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer

    r = Renderer(ins.tl, root, Canvas(540, 960, device="cuda"))
    r._ensure_rasters()
    with torch.no_grad():
        img = r._eval(ins.tl.plan(f).scene, float(f)).cpu()
    flat = torch.tensor([0x22, 0x33, 0x44], dtype=torch.float32)[:, None, None] / 255.0
    ys, xs = torch.nonzero((img - flat).abs().amax(dim=0) > 0.1, as_tuple=True)
    got = [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1]
    assert all(abs(a - b) <= 2 for a, b in zip(el["box"], got)), (el["box"], got)


@pytest.mark.gpu
def test_contrast_and_problems_with_render(clean: Path, off: Path) -> None:
    ins = Inspector(clean, fast=False)
    el = ins.at(60)["text"][0]
    assert el["contrast"] is not None and el["contrast"] > 8  # cream on #223344
    assert ins.problems() == []
    bad = Inspector(off, fast=False).problems()
    assert [p["id"] for p in bad] == ["card#0.1"] and bad[0]["issue"] == "off_frame"
