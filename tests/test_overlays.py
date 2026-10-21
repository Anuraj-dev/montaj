"""Overlays: spec validation, plan/hash wiring, and the CPU rasteriser. No GPU."""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import torch
from PIL import Image

from montaj.pipeline import engine_version
from montaj.project import segment_hash
from montaj.render.canvas import Canvas
from montaj.render.frame import Renderer
from montaj.render.overlay import draw_overlays
from montaj.spec import SpecError, load_spec
from montaj.timeline import ResolvedOverlay, resolve

_LINE = re.compile(r"^[^:]+: .+ \(.+\)$")


def _write(tmp: Path, overlays: str, *, size: str = "270x480", hold: str = "90f") -> Path:
    assets = tmp / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (30, 60, 90)).save(assets / "a.jpg", "JPEG")
    p = tmp / "montaj.yaml"
    p.write_text(
        f"video:\n  size: {size}\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
        f"assets: assets\nshots:\n  - {{photo: a, hold: {hold}}}\n{overlays}"
    )
    return p


def _err(tmp: Path, overlays: str, needle: str) -> None:
    with pytest.raises(SpecError) as ei:
        load_spec(_write(tmp, overlays))
    assert all(_LINE.match(e) for e in ei.value.errors), ei.value.errors
    assert any(needle in e for e in ei.value.errors), ei.value.errors


RING = "overlays:\n  - {shape: ring, from: 0f, to: 30f, center: [90, 160], radius: [10, 50], width: 4}\n"


def test_ring_brackets_line_validate_and_resolve(tmp_path: Path) -> None:
    p = _write(tmp_path, RING + (
        "  - {shape: brackets, from: 1s, to: 2s, box: [10, 20, 170, 120], draw: 0.3s}\n"
        "  - {shape: line, from: 2s, to: 3s, points: [[0, 160], [180, 160]], draw: 10f, color: '#7FE7FF'}\n"
    ))
    tl = resolve(load_spec(p), tmp_path)
    ring, br, ln = tl.overlays
    assert ring.radius == (10.0, 50.0) and ring.sweep == (360.0, 360.0) and ring.draw is None
    assert (br.from_, br.to, br.draw, br.arm) == (30.0, 60.0, 9.0, 28.0)
    assert ln.color == "#7fe7ff" and ln.points == ((0.0, 160.0), (180.0, 160.0))
    assert tl.plan(0).overlays == (0,)
    assert tl.plan(29).overlays == (0,)
    assert tl.plan(30).overlays == (1,)
    assert tl.plan(70).overlays == (2,)


@pytest.mark.parametrize("ov, needle", [
    ("{shape: star, from: 0f, to: 9f}", "unknown 'star'"),
    ("{shape: ring, from: 0f, to: 9f, radius: 5}", "center: required for ring"),
    ("{shape: ring, from: 0f, to: 9f, center: [1, 1]}", "radius: required for ring"),
    ("{shape: ring, from: 0f, to: 9f, center: [1, 1], radius: 5, box: [0, 0, 9, 9]}", "box: not used by ring"),
    ("{shape: ring, from: 0f, to: 9f, center: [1, 1], radius: 5, sweep: [0, 400]}", "sweep"),
    ("{shape: ring, from: 9f, to: 9f, center: [1, 1], radius: 5}", "to: must be after from"),
    ("{shape: ring, from: 0f, to: 9f, center: [1, 1], radius: -5}", "radius"),
    ("{shape: ring, from: 0f, to: 9f, center: [1, 1], radius: 5, color: red}", "color"),
    ("{shape: ring, from: 0f, to: 9f, center: [1, 1], radius: 5, ease: wobble}", "ease"),
    ("{shape: ring, from: 0f, to: 9f, center: [1, 1], radius: 5, width: 0}", "width"),
    ("{shape: ring, from: 0f, to: 9f, center: [1, 1], radius: 5, opacity: 2}", "opacity"),
    ("{shape: brackets, from: 0f, to: 9f, box: [10, 10, 5, 50]}", "box"),
    ("{shape: brackets, from: 0f, to: 9f, box: [0, 0, 9, 9], draw: 0f}", "draw"),
    ("{shape: line, from: 0f, to: 9f, points: [[0, 0]]}", "at least 2 points"),
    ("{shape: line, from: 0f, to: 9f, points: [[0, 0], [5, 5]], radius: 3}", "radius: not used by line"),
    ("{shape: line, from: 0f, to: 9f, points: [[0, 0], [5, 5]], draw: 4b}", "video.bpm"),
])
def test_bad_overlay_is_a_helpful_error(tmp_path: Path, ov: str, needle: str) -> None:
    _err(tmp_path, f"overlays:\n  - {ov}\n", needle)


def test_unknown_overlay_key_is_rejected(tmp_path: Path) -> None:
    _err(tmp_path, "overlays:\n  - {shape: ring, from: 0f, to: 9f, center: [1, 1], radius: 5, blur: 3}\n", "blur")


# --- hash -------------------------------------------------------------------

def _hashes(tl) -> list[str]:
    return [segment_hash(tl, s, {"a": "x"}, engine_version()) for s in tl.segments]


def test_overlay_reaches_only_the_segments_it_is_drawn_in(tmp_path: Path) -> None:
    def film(name: str, overlays: str) -> list[str]:
        d = tmp_path / name
        d.mkdir()
        p = _write(d, overlays, hold="30f")
        p.write_text(p.read_text().replace("hold: 30f}", "hold: 30f}\n  - {photo: a, hold: 30f}", 1))
        return _hashes(resolve(load_spec(p), d))

    ring = "overlays:\n  - {shape: ring, from: 0f, to: 20f, center: [90, 160], radius: 40, width: %s}\n"
    plain, a, b = film("plain", ""), film("a", ring % 4), film("b", ring % 5)
    assert a[0] != plain[0] and a[1] == plain[1]     # second shot never sees the ring
    assert a[0] != b[0] and a[1] == b[1]             # a rendered field reaches the hash


# --- rasteriser ---------------------------------------------------------------

def _ov(**kw) -> ResolvedOverlay:
    base = dict(
        shape="ring", from_=0.0, to=10.0, center=(90.0, 90.0), radius=(40.0, 40.0), sweep=(360.0, 360.0),
        box=None, arm=28.0, points=None, draw=None, width=4.0, color="#ffffff", opacity=1.0, glow=0.0,
        ease="linear", above_finish=False,
    )
    base.update(kw)
    return ResolvedOverlay(**base)


def _canvas(w: int = 1080, h: int = 180) -> Canvas:
    """Width 1080 keeps k = 1, so design px are device px."""
    return Canvas(w, h, device="cpu")


def _black(cv: Canvas) -> torch.Tensor:
    return torch.zeros(3, cv.H, cv.W, device=cv.device)


def test_ring_pixel_coverage() -> None:
    cv = _canvas()
    # Centre on a pixel centre so pixel (90, 90 + n) is exactly n px from it.
    out = draw_overlays(cv, _black(cv), [_ov(center=(90.5, 90.5))], 5.0)[0]
    assert out[90, 130] == pytest.approx(1.0, abs=1e-5)
    assert out[90, 90] == 0.0                     # centre untouched
    assert out[90, 140] == 0.0                    # 10 px outside the band
    # The band spans radius 38..42, so the pixels exactly on those edges are half covered.
    assert float(out[90, 90 + 42]) == pytest.approx(0.5, abs=1e-5)
    assert float(out[90, 90 + 38]) == pytest.approx(0.5, abs=1e-5)
    # Total ink ~ circumference * width.
    area = float(out.sum())
    assert area == pytest.approx(2 * math.pi * 40 * 4, rel=0.02)


def test_ring_scales_with_canvas_k() -> None:
    small = Canvas(540, 90, device="cpu")                    # k = 0.5
    big = _canvas()
    ov = _ov(center=(90.0, 90.0), radius=(40.0, 40.0), width=4.0)
    a = draw_overlays(small, _black(small), [ov], 0.0)
    b = draw_overlays(big, _black(big), [ov], 0.0)
    # Same design values: half the scale means a quarter of the ink, and the same picture.
    assert float(b.sum()) == pytest.approx(4.0 * float(a.sum()), rel=0.03)
    down = torch.nn.functional.avg_pool2d(b[None], 2)[0]
    assert (down - a).abs().max() < 0.12


def test_ring_radius_and_sweep_animate() -> None:
    cv = _canvas()
    ov = _ov(radius=(10.0, 50.0), sweep=(0.0, 360.0))
    assert float(draw_overlays(cv, _black(cv), [ov], 0.0).sum()) == 0.0   # sweep 0: nothing
    mid = draw_overlays(cv, _black(cv), [ov], 5.0)[0]                      # r=30, sweep=180
    # Sweep runs clockwise from 12 o'clock: the right half is drawn, the left is empty.
    assert mid[90, 90 + 30] > 0.9 and mid[90, 90 - 30] == 0.0
    assert mid[90 - 30, 90] > 0.4 and mid[90 + 30, 90] > 0.4
    assert float(mid[:, :90].sum()) < 1.0
    end = draw_overlays(cv, _black(cv), [ov], 9.999)[0]
    assert end[90, 90 - 49] > 0.9 and end[90, 90 + 49] > 0.9


def test_line_draw_on_at_0_half_1() -> None:
    cv = _canvas()
    ov = _ov(shape="line", center=None, radius=None, points=((0.0, 90.0), (180.0, 90.0)), draw=10.0, width=2.0)
    inks = []
    for f in (0.0, 5.0, 10.0):
        out = draw_overlays(cv, _black(cv), [ov], f)[0]
        inks.append(float(out[90].sum()) + float(out[89].sum()))
    assert inks[0] == 0.0
    assert inks[1] == pytest.approx(inks[2] / 2, rel=0.1)
    full = draw_overlays(cv, _black(cv), [ov], 10.0)[0]
    assert full[89, 5] == pytest.approx(1.0, abs=1e-5) and full[90, 175] == pytest.approx(1.0, abs=1e-5)
    half = draw_overlays(cv, _black(cv), [ov], 5.0)[0]
    assert half[89, 60] > 0.9 and half[89, 120] == 0.0


def test_brackets_draw_corners_only() -> None:
    cv = _canvas()
    ov = _ov(shape="brackets", center=None, radius=None, box=(30.0, 30.0, 150.0, 150.0), arm=20.0, width=4.0)
    out = draw_overlays(cv, _black(cv), [ov], 0.0)[0]
    assert out[31, 30:50].min() > 0.9 and out[30:50, 31].min() > 0.9      # top-left arms
    assert out[148, 130:150].min() > 0.9                                  # bottom-right arm
    assert out[90, 90] == 0.0 and out[31, 90] == 0.0                      # no edges between corners
    # draw: arms grow with progress.
    grow = _ov(shape="brackets", center=None, radius=None, box=ov.box, arm=20.0, width=4.0, draw=10.0)
    half = draw_overlays(cv, _black(cv), [grow], 5.0)[0]
    assert half[31, 30:40].min() > 0.9 and half[31, 45] == 0.0


def test_glow_adds_light_around_the_shape_and_opacity_scales_it() -> None:
    cv = _canvas()
    crisp = draw_overlays(cv, _black(cv), [_ov()], 0.0)
    glow = draw_overlays(cv, _black(cv), [_ov(glow=0.8)], 0.0)
    assert float(glow[0, 90, 90 + 40 + 6]) > float(crisp[0, 90, 90 + 40 + 6]) == 0.0   # halo outside the band
    assert float(glow[0, 90, 90]) < float(glow[0, 90, 90 + 40 + 6])                    # but fades toward the centre
    assert torch.equal(glow[:, 90, 130], crisp[:, 90, 130]) or float(glow[0, 90, 130]) >= 1.0 - 1e-6
    half = draw_overlays(cv, _black(cv), [_ov(opacity=0.5)], 0.0)
    assert half[0, 90, 130] == pytest.approx(0.5, abs=1e-5)


def test_colour_and_composite_over_the_picture() -> None:
    cv = _canvas()
    img = torch.full((3, cv.H, cv.W), 0.25)
    out = draw_overlays(cv, img, [_ov(color="#ff8000")], 0.0)
    assert out[:, 90, 130].tolist() == pytest.approx([1.0, 128 / 255, 0.0], abs=1e-5)
    assert out[:, 90, 90].tolist() == pytest.approx([0.25] * 3)
    assert img[0, 90, 130] == 0.25           # input is not modified
    assert draw_overlays(cv, img, [], 0.0) is img


# --- renderer wiring ------------------------------------------------------------

def test_renderer_draws_overlay_and_leaves_plain_films_alone(tmp_path: Path) -> None:
    plain_p = _write(tmp_path / "plain", "", hold="12f")
    ov_p = _write(tmp_path / "ov", "overlays:\n  - {shape: ring, from: 0f, to: 6f, center: [90, 160], radius: 40, width: 12, color: '#00ff00'}\n", hold="12f")
    frames = []
    for p in (plain_p, ov_p):
        tl = resolve(load_spec(p), p.parent)
        frames.append([Renderer(tl, p.parent, Canvas(*tl.size, device="cpu")).frame(f) for f in (2, 8)])
    (plain2, plain8), (ov2, ov8) = frames
    assert not torch.equal(plain2, ov2)
    assert torch.equal(plain8, ov8)          # outside the window: bit-identical
    diff = (ov2 - plain2).abs().sum(0) > 1e-3
    assert diff.any() and not diff[:20].any() and not diff[60:].any()   # only around the ring


def test_above_finish_skips_the_grade(tmp_path: Path) -> None:
    body = "overlays:\n  - {shape: ring, from: 0f, to: 6f, center: [90, 160], radius: 40, width: 12, color: '#ff0000', above_finish: %s}\n"
    outs = {}
    for flag in ("false", "true"):
        p = _write(tmp_path / flag, body % flag, hold="12f")
        tl = resolve(load_spec(p), p.parent)
        outs[flag] = Renderer(tl, p.parent, Canvas(*tl.size, device="cpu")).frame(2)
    y, x = 30, 22                              # top of the ring at k = 0.25
    assert outs["true"][:, y, x].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-5)
    assert outs["false"][0, y, x] < 1.0       # graded: warm-film lifts/gains the red


def test_offscreen_stroke_glows_into_the_canvas() -> None:
    cv = _canvas()
    line = dict(shape="line", center=None, radius=None, points=((-3.0, 0.0), (-3.0, 180.0)), width=2.0)
    glow = draw_overlays(cv, _black(cv), [_ov(glow=0.8, **line)], 0.0)
    crisp = draw_overlays(cv, _black(cv), [_ov(glow=0.0, **line)], 0.0)
    assert float(crisp[0, 90, 0]) == 0.0                 # the stroke itself is offscreen
    assert float(glow[0, 90, 0]) > 0.02                  # its halo reaches the edge


@pytest.mark.gpu
def test_cuda_overlay_matches_cpu() -> None:
    cpu, gpu = _canvas(), Canvas(1080, 180, device="cuda")
    ovs = [_ov(glow=0.6, sweep=(0.0, 300.0), radius=(20.0, 60.0)),
           _ov(shape="brackets", center=None, radius=None, box=(300.0, 20.0, 900.0, 160.0), draw=8.0)]
    a = draw_overlays(cpu, _black(cpu), ovs, 5.0)
    b = draw_overlays(gpu, _black(gpu), ovs, 5.0).cpu()
    assert (a - b).abs().max() < 1e-3


@pytest.mark.parametrize("field", ["width", "glow", "opacity", "arm"])
def test_bool_numeric_fields_are_rejected(tmp_path: Path, field: str) -> None:
    shape = "brackets, box: [10, 20, 170, 120]" if field == "arm" else "ring, center: [90, 160], radius: 40"
    _err(tmp_path, f"overlays:\n  - {{shape: {shape}, from: 0f, to: 6f, {field}: true}}\n", f"overlays[0].{field}")


@pytest.mark.parametrize("shape", [
    "ring, center: [true, 160], radius: 40",
    "ring, center: [90, 160], radius: [true, 40]",
    "brackets, box: [10, 20, 170, true]",
    "line, points: [[0, 0], [true, 5]]",
])
def test_bool_coordinates_are_rejected(tmp_path: Path, shape: str) -> None:
    _err(tmp_path, f"overlays:\n  - {{shape: {shape}, from: 0f, to: 6f}}\n", "overlays[0]")


def test_above_finish_shape_sits_under_text(tmp_path: Path) -> None:
    ring = "overlays:\n  - {shape: ring, from: 0f, to: 12f, center: [540, 960], radius: 100, width: 2400, color: '#ff0000', above_finish: true}\n"
    text = "text:\n  - from: 0f\n    to: 12f\n    lines:\n      - {text: HELLO, style: caps, size: 200, y: 900, at: 0f}\n"
    outs = {}
    for name, body in (("ring", ring), ("both", ring + text)):
        d = tmp_path / name
        p = _write(d, body, size="270x480", hold="12f")
        p.write_text(p.read_text().replace("size: 270x480", "size: 1080x1920"))
        tl = resolve(load_spec(p), d)
        outs[name] = Renderer(tl, d, Canvas(*tl.size, device="cpu")).frame(3)
    changed = (outs["both"] - outs["ring"]).abs().sum(0) > 0.05
    assert changed.any()          # text shows through the opaque above_finish ring
