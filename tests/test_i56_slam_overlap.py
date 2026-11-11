"""Issue #56: a staggered `by: word` slam never lands on the word before it — each word grows away from
the words on screen, keeping its zoom — and `inspect --problems` reports ink over ink (`overlap`) at the
animated pose."""
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
# Every staggered slam of the Auren short, one per block.
AUREN = [
    '{text: YOUR PC, style: slam, size: 130, pos: [540, 230], at: 0.5s}',
    '{text: "*TALKS BACK*", style: slamx, size: 150, pos: [540, 370], at: 0.5s}',
    '{text: "*SPOTIFY* ON", style: slamx, pos: [540, 500], at: 0.5s}',
    '{text: "*NEW SPACE*", style: slamx, size: 120, pos: [540, 500], at: 0.5s}',
    '{text: "TOKYO *NIGHT*", style: slamx, size: 140, pos: [540, 300], at: 0.5s}',
    SENTENCE,
    '{text: PICK YOUR, style: slam, size: 100, pos: [330, 220], at: 0.5s}',
    '{text: "OR *VEYRA*", style: slamx, size: 150, pos: [540, 470], at: 0.5s}',
    '{text: READY WHEN, style: slam, size: 130, pos: [540, 470], at: 0.5s}',
    '{text: "*YOU* ARE", style: slamx, size: 150, pos: [540, 470], at: 0.5s}',
]


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
    assert m.kind == "slam" and ln.stagger > 0 and m.slam_units is not None
    (one, one_x, _), (sentence, sentence_x, _) = m.slam_units
    # ONE keeps the full slam growing leftward, SENTENCE. grows rightward from its left edge as far as
    # the safe area allows (1.5 would need 642 px between ONE and the safe edge; there are about 610 once both outlines are cleared).
    assert one == M.SLAM_FROM and one_x > 0
    assert sentence >= 1.42 and sentence_x < 0
    space = max(g.advance for g in ln.glyphs.glyphs if g.ink[2] <= g.ink[0])
    rest_gap = _worst_gap(ln, replace(m, slam_from=1.0, slam_units=None, shake=0.0), 0.0)
    stroke = ln.stroke.width
    # Painted ink (stroke included) never touches, and the outline keeps a space (or its resting gap).
    assert _worst_gap(ln, m, stroke) > 0
    entry = ln.stagger + m.dur  # until SENTENCE. lands; its shake after that is not the slam's to cap
    assert _worst_gap(ln, m, 0.0, entry) >= min(space, rest_gap) - 0.5
    # The test can fail: the slam before #56 (capped only at the frame edge) covered ONE.
    assert _worst_gap(ln, replace(m, slam_from=1.731, slam_units=None), stroke) < -50


def _overlaps(root: Path) -> list[dict]:
    return [p for p in Inspector(root, fast=True).problems() if p["issue"] == "overlap"]


def test_words_that_slam_together_never_overlap(tmp_path: Path) -> None:
    """With no stagger every word grows at once: each takes half the gap to its neighbour (YOUR PC covered
    49% when only the frame edge capped it)."""
    p = _proj(tmp_path, SENTENCE.replace("at: 0.5s", "at: 0.5s, in: {type: slam, by: word, stagger: 0s}"))
    ln = resolve(load_spec(p), p.parent).texts[0].lines[0]
    assert ln.stagger == 0 and ln.motion.slam_units is not None
    assert _worst_gap(ln, ln.motion, ln.stroke.width) > 0
    assert _overlaps(p.parent) == []


# Lines of one block that slam near each other: whichever enters first, neither grows over the other (#56 v2
# anchored a line only against lines before it in the block).
@pytest.mark.parametrize("lines_", [
    pytest.param(['{text: TALKS, style: slam, size: 150, pos: [540, 200], at: 0.5s}',
                  '{text: BACK, style: slam, size: 150, pos: [540, 340], at: 0.5s}'], id="same-at"),
    pytest.param(['{text: TALKS, style: slam, size: 150, pos: [540, 200], at: 0.7s}',
                  '{text: BACK, style: slam, size: 150, pos: [540, 340], at: 0.5s}'], id="later-line-first"),
    pytest.param(['{text: "ONE SENTENCE TALKS BACK TO YOU", style: slam, size: 100, pos: [540, 300], at: 0.5s,'
                  ' wrap: balance, max_width: 800, line_height: 0.9}'], id="balance-wrap"),
])
def test_slamming_lines_of_a_block_never_cover_each_other(tmp_path: Path, lines_: list[str]) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    body = "".join(f"      - {ln}\n" for ln in lines_)
    root = _write(tmp_path, "  - {photo: a, hold: 3s}\n", ["a"],
                  video="size: 1920x1080\n  fps: 60\n  bpm: 120\n  look: cool-tech",
                  extra=STYLES + "text:\n  - from: 0s\n    to: 2s\n    lines:\n" + body).parent
    assert _overlaps(root) == []
    block = resolve(load_spec(root / "montaj.yaml"), root).texts[0]
    assert len(block.lines) >= 2 and any(ln.motion.slam_units is not None for ln in block.lines)


def test_a_char_slam_keeps_the_centre_cascade(tmp_path: Path) -> None:
    """`by: char` letters land one after another about their own centres, capped only at the frame edge:
    the slam start and segment hash are those from before #56."""
    from PIL import Image

    from montaj.project import segment_hash

    (tmp_path / "assets").mkdir(parents=True)
    Image.new("RGB", (8, 8), (20, 40, 60)).save(tmp_path / "assets" / "a.jpg")
    (tmp_path / "montaj.yaml").write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\nassets: assets\nshots:\n"
        "  - {photo: a, hold: 2s}\n"
        "text:\n  - from: 0s\n    to: 2s\n    lines:\n"
        "      - {text: HELLO, font: anton, size: 330, stroke: {width: 7, color: \"#000000\"}, pos: [470, 900], at: 0.2s,"
        " in: {type: slam, by: char, stagger: 0.07s}}\n")
    tl = resolve(load_spec(tmp_path / "montaj.yaml"), tmp_path)
    m = tl.texts[0].lines[0].motion
    assert m.slam_from == pytest.approx(1.673) and m.slam_units is None
    assert [segment_hash(tl, s, {"a": "x"}, "e1") for s in tl.segments] == [
        "b83aa8b392888cf783f7ee95c86b1fca0837a05242cb5a5f82b7c5a13abd33cc"]
    # The letters' own ink boxes touch at rest: a char run is not checked against itself.
    assert _overlaps(tmp_path) == []


def test_slam_units_enter_the_segment_hash(tmp_path: Path) -> None:
    from montaj.project import segment_hash

    p = _proj(tmp_path, SENTENCE)
    tl = resolve(load_spec(p), p.parent)
    ln = tl.texts[0].lines[0]
    assert ln.motion.slam_units is not None
    before = segment_hash(tl, tl.segments[0], {"a": "x"}, "e1")
    units = tuple((s - 0.1, ox, oy) for s, ox, oy in ln.motion.slam_units)
    tl.texts[0] = replace(tl.texts[0], lines=(replace(ln, motion=replace(ln.motion, slam_units=units)),))
    assert segment_hash(tl, tl.segments[0], {"a": "x"}, "e1") != before


def test_inspect_ignores_ink_that_overlaps_at_rest(tmp_path: Path) -> None:
    """Words the font sets into each other (tight tracking, an italic overhang) are designed: only a unit
    moved onto another, past how much they share at rest, is an overlap."""
    root = _proj(tmp_path, '{text: "WAVE WAVE", style: slam, pos: [540, 300], at: 0s, tracking: -0.45,'
                 ' in: {type: pop, by: word}}').parent
    ln = resolve(load_spec(root / "montaj.yaml"), root).texts[0].lines[0]
    rest = _posed(ln, ln.motion, 100, 0.0)  # settled
    assert min(rest[0][2], rest[1][2]) > max(rest[0][0], rest[1][0])  # the words share ink at rest
    assert _overlaps(root) == []


def test_a_single_word_slam_keeps_its_centre(tmp_path: Path) -> None:
    ln = _line(tmp_path, '{text: "*PAUSED.*", style: slamx, pos: [540, 500], at: 0.5s}')
    assert ln.motion.slam_units is None and ln.motion.slam_from == pytest.approx(1.08)


def test_auren_word_slams_keep_their_zoom_without_overlap(tmp_path: Path) -> None:
    """Every Auren two-word slam: no word covers another, and each still zooms (the first #56 fix capped
    them all to ~1.0). A word's start is bounded only by the room the safe area leaves beside it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    blocks = "".join(f"  - from: {k}s\n    to: {k + 1}s\n    lines: [{ln.replace('at: 0.5s', f'at: {k + 0.2}s')}]\n"
                     for k, ln in enumerate(AUREN))
    root = _write(tmp_path, f"  - {{photo: a, hold: {len(AUREN) + 1}s}}\n", ["a"],
                  video="size: 1920x1080\n  fps: 60\n  bpm: 120\n  look: cool-tech", extra=f"{STYLES}text:\n{blocks}").parent
    tl = resolve(load_spec(root / "montaj.yaml"), root)
    starts = {b.lines[0].text: [s for s, _, _ in b.lines[0].motion.slam_units] for b in tl.texts}
    assert all(s >= 1.25 for v in starts.values() for s in v), starts
    assert sum(s >= 1.5 for v in starts.values() for s in v) >= 13, starts
    assert [p for p in Inspector(root, fast=True).problems() if p["issue"] == "overlap"] == []


def test_a_slam_under_a_line_on_screen_grows_away_from_it(tmp_path: Path) -> None:
    """TALKS BACK lands 2 px under YOUR PC: it grows downward instead of over the line above."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = _write(tmp_path, "  - {photo: a, hold: 3s}\n", ["a"],
                  video="size: 1920x1080\n  fps: 60\n  bpm: 120\n  look: cool-tech",
                  extra=STYLES + "text:\n  - from: 0s\n    to: 2s\n    lines:\n"
                  "      - {text: YOUR PC, style: slam, size: 130, pos: [540, 230], at: 0s}\n"
                  "      - {text: TALKS BACK, style: slam, size: 150, pos: [540, 370], at: 0.5s}\n").parent
    talks = resolve(load_spec(root / "montaj.yaml"), root).texts[0].lines[1]
    ink = [g.ink for g in talks.glyphs.glyphs if g.ink[2] > g.ink[0]]
    h = max(b[3] for b in ink) - min(b[1] for b in ink)
    assert all(oy == pytest.approx(-h / 2, abs=0.01) for _, _, oy in talks.motion.slam_units)
    assert [p for p in Inspector(root, fast=True).problems() if p["issue"] == "overlap"] == []


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
