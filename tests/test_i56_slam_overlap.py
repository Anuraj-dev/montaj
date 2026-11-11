"""Issue #56: a staggered `by: word` slam never lands on the word before it, and `inspect --problems`
reports ink over ink (`overlap`) at the animated pose."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from montaj import timeline
from montaj.cli import main
from montaj.inspector import Inspector, lines
from montaj.render import motion as M
from montaj.spec import load_spec
from montaj.timeline import resolve

from tests.test_spec import _write

# The Auren short's styles (#47), where "SENTENCE." slammed over "ONE" for 0.15 s.
STYLES = """styles:
  slam:
    font: anton
    case: upper
    size: 120
    color: "#ffffff"
    stroke: {width: 7, color: "#000000"}
    in: {type: slam, by: word, stagger: 0.06s}
    out: cut
  slamx:
    extends: slam
    emphasis: {color: "#ffe600", scale: 1.12, shake: 12}
"""
SENTENCE = '{text: "ONE *SENTENCE.*", style: slamx, size: 100, pos: [540, 300], at: 0.5s}'


def _proj(tmp: Path, *lines_: str) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    blocks = "".join(f"  - from: 0s\n    to: 2s\n    lines: [{ln}]\n" for ln in lines_)
    return _write(tmp, "  - {photo: a, hold: 3s}\n", ["a"], video="size: 1920x1080\n  fps: 60\n  bpm: 120\n  look: cool-tech",
                  extra=f"{STYLES}text:\n{blocks}")


def _line(tmp: Path, line: str = SENTENCE):
    p = _proj(tmp, line)
    return resolve(load_spec(p), p.parent).texts[0].lines[0]


def _posed(ln, motion, t: float, pad: float) -> dict[int, tuple[float, float, float, float]]:
    """Per visible reveal unit: its ink box at frame `t` under `motion`, grown by `pad` (the stroke)."""
    rest: dict[int, list] = {}
    for g in ln.glyphs.glyphs:
        if g.ink[2] > g.ink[0]:
            b = rest.setdefault(g.unit, [*g.ink, bool(g.emph)])
            b[:4] = min(b[0], g.ink[0]), min(b[1], g.ink[1]), max(b[2], g.ink[2]), max(b[3], g.ink[3])
    out = {}
    for u, (x0, y0, x1, y1, emph) in rest.items():
        pose = M.unit_pose(motion, u, emph, t, ln.at, ln.stagger)
        if pose.opacity <= 1e-4:
            continue
        cx, cy = (x0 + x1) / 2 + pose.dx, (y0 + y1) / 2 + pose.dy
        hw, hh = (x1 - x0) / 2 * pose.scale + pad, (y1 - y0) / 2 * pose.scale * pose.sy + pad
        out[u] = (cx - hw, cy - hh, cx + hw, cy + hh)
    return out


def _worst_gap(ln, motion, pad: float, frames: float = 120) -> float:
    """Smallest horizontal gap (design px; negative = covered) between ONE and SENTENCE. while both show,
    over the first `frames` of the line (default the whole 2 s block)."""
    worst = float("inf")
    for k in range(int(frames * 4)):  # quarter frames
        boxes = _posed(ln, motion, ln.at + k / 4, pad)
        if 0 in boxes and 1 in boxes:
            worst = min(worst, boxes[1][0] - boxes[0][2])
    return worst


def test_staggered_word_slam_never_covers_the_word_before(tmp_path: Path) -> None:
    ln = _line(tmp_path)
    m = ln.motion
    assert m.kind == "slam" and ln.stagger > 0 and m.slam_from is not None and m.slam_from < M.SLAM_FROM
    space = max(g.advance for g in ln.glyphs.glyphs if g.ink[2] <= g.ink[0])
    rest_gap = _worst_gap(ln, replace(m, slam_from=1.0, shake=0.0), 0.0)
    stroke = ln.stroke.width
    # Painted ink (stroke included) never touches, and the outline keeps a space (or its resting gap).
    assert _worst_gap(ln, m, stroke) > 0
    entry = ln.stagger + m.dur  # until SENTENCE. lands; its shake after that is not the slam's to cap
    assert _worst_gap(ln, m, 0.0, entry) >= min(space, rest_gap) - 0.5
    # The test can fail: the slam before #56 (capped only at the frame edge) covered ONE.
    assert _worst_gap(ln, replace(m, slam_from=1.731), stroke) < -50


def test_words_that_start_together_keep_the_edge_cap(tmp_path: Path) -> None:
    """With no stagger every word lands at once and only the frame edge caps the slam, as before #56."""
    together = _line(tmp_path / "a", SENTENCE.replace("at: 0.5s", "at: 0.5s, in: {type: slam, by: word, stagger: 0s}"))
    staggered = _line(tmp_path / "b")
    assert together.stagger == 0
    assert together.motion.slam_from == pytest.approx(1.731)
    assert staggered.motion.slam_from < together.motion.slam_from


def test_inspect_reports_the_uncapped_word_slam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _proj(tmp_path / "capped", SENTENCE).parent
    assert [p for p in Inspector(root, fast=True).problems() if p["issue"] == "overlap"] == []
    monkeypatch.setattr(timeline, "_cap_slam", lambda m, *a, **k: m)  # the layout #47 shipped
    root = _proj(tmp_path / "uncapped", SENTENCE).parent
    found = [p for p in Inspector(root, fast=True).problems() if p["issue"] == "overlap"]
    assert len(found) == 1, found
    p = found[0]
    assert (p["id"], p["text"], p["under"]) == ("text#0.0", "SENTENCE.", "ONE") and p["value"] > 30
    assert 30 <= p["frame"] <= 30 + 14  # during SENTENCE.'s entry (block at 0.5 s, 60 fps)
    out = lines({"problems": found})
    assert out[0] == f"PROBLEM text#0.0 f={p['frame']} overlap 'SENTENCE.' over 'ONE' {p['value']}%"


def test_inspect_reports_two_elements_on_top_of_each_other(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _proj(tmp_path, '{text: "HELLO", style: slam, pos: [540, 300], at: 0s, in: pop}',
                 '{text: "WORLD", style: slam, pos: [540, 310], at: 0s, in: pop}').parent
    assert main(["-C", str(root), "inspect", "--problems", "--fast"]) == 0
    got = capsys.readouterr().out.strip().splitlines()
    over = [ln for ln in got if " overlap " in ln]
    assert len(over) == 1 and over[0].startswith("PROBLEM text#1.0 f=") and "overlap 'WORLD' over 'HELLO' " in over[0], got
    assert got[-1].startswith("WARN inspect ")
