"""Text motion (issue #22): per-unit `in`, `out`, emphasis shake, `idle` and `text[].motion_blur` on gpu lines."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from montaj.render import motion as M
from montaj.render.motion import HIDDEN, REST, unit_pose
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

from tests.test_gpu_text import _cuda, _hashes, _no_browser, _text_spec

FPS = 30
WORD = "text: HELLO, size: 120, y: 900, at: 0s"


def _spec(tmp: Path, *lines: str, block: str = "") -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    p = _text_spec(tmp, *lines)
    if block:
        p.write_text(p.read_text().replace("    lines:", f"    {block}\n    lines:"))
    return p


def _lines(tmp: Path, *lines: str, block: str = ""):
    p = _spec(tmp, *lines, block=block)
    return resolve(load_spec(p), p.parent).texts[0]


def _pose(line, unit: int, t: float, emph: bool = False):
    return unit_pose(line.motion, unit, emph, t, line.at, line.stagger)


def _errors(tmp: Path, *lines: str) -> list[str]:
    with pytest.raises(SpecError) as e:
        load_spec(_spec(tmp, *lines))
    return e.value.errors


# --- entries: start, peak, settle (frames, 30 fps, one unit) ---------------------------------------


def test_every_in_type_has_a_start_peak_and_settle(tmp_path: Path) -> None:
    def line(kind: str, extra: str = ""):
        return _lines(tmp_path / kind, f"{WORD}, in: {{type: {kind}{extra}}}").lines[0]

    pop = line("pop")  # 0.2 s = 6 frames; 0.8 → 1.08 at 60 % → 1
    assert pop.motion.dur == 6
    assert _pose(pop, 0, 0).scale == pytest.approx(0.8) and _pose(pop, 0, 0).opacity == 0
    assert _pose(pop, 0, 6 * M.PEAK_AT).scale == pytest.approx(1.08)
    assert _pose(pop, 0, 6) == REST

    slam = line("slam")  # 1.8 → 1 accelerating over 6 frames, then a 2-frame squash
    assert _pose(slam, 0, 0).scale == pytest.approx(1.8)
    assert _pose(slam, 0, 5).scale > 1.0
    assert _pose(slam, 0, 7).scale == pytest.approx(0.94)  # mid-settle: 1 − (1.06 − 1)
    assert _pose(slam, 0, 6 + M.SLAM_SETTLE) == REST and _pose(slam, 0, 7.99).scale < 1.0

    slide = line("slide", ", from: right")  # 0.3 s = 9 frames from +160 px
    assert _pose(slide, 0, 0).dx == pytest.approx(160) and _pose(slide, 0, 0).dy == 0
    assert _pose(slide, 0, 9 * M.PEAK_AT).dx == pytest.approx(-160 * 0.04)  # past rest by the overshoot
    assert _pose(slide, 0, 9) == REST

    drop = line("drop")  # 0.5 s = 15 frames from 220 px above, bouncing
    assert _pose(drop, 0, 0).dy == pytest.approx(-220)
    assert _pose(drop, 0, 15 / 2.75).dy == pytest.approx(0.0, abs=1e-6)  # first touch
    assert _pose(drop, 0, 15 * 1.5 / 2.75).dy < -10  # up again on the first bounce
    assert _pose(drop, 0, 15) == REST

    rise = line("rise")  # the legacy reveal: 0.9 s, rise 30 px, blur to 0
    assert rise.motion.dur == 27 and rise.motion.distance == rise.rise
    assert _pose(rise, 0, 0) == M.Pose(dy=rise.rise, opacity=0.0, blur=24.0)
    mid = _pose(rise, 0, 13.5)
    assert 0 < mid.dy < rise.rise and 0 < mid.opacity < 1 and 0 < mid.blur < 24
    assert _pose(rise, 0, 27) == REST

    flip = line("flip")  # 0.3 s: squashed flat → full height
    assert _pose(flip, 0, 0).sy == 0
    assert 0.5 < _pose(flip, 0, 4.5).sy < 1
    assert _pose(flip, 0, 9) == REST

    scale_in = line("scale-in", ", overshoot: 1.1")
    assert _pose(scale_in, 0, 0).scale == 0
    assert _pose(scale_in, 0, 7.5 * M.PEAK_AT).scale == pytest.approx(1.1)
    assert _pose(scale_in, 0, 7.5) == REST

    typed = _lines(tmp_path / "type", "text: HI, y: 900, at: 0s, in: type").lines[0]
    assert typed.by == "char" and typed.stagger == pytest.approx(1.5)
    assert _pose(typed, 0, 0) == REST and _pose(typed, 1, 0) == HIDDEN
    assert _pose(typed, 1, 1.5) == REST


def test_in_shorthand_and_overrides_resolve(tmp_path: Path) -> None:
    a = _lines(tmp_path / "a", "text: one two, y: 900, at: 0s, in: pop").lines[0]
    b = _lines(tmp_path / "b", "text: one two, y: 900, at: 0s, in: {type: pop}").lines[0]
    assert a.motion == b.motion and a.by == "word"
    c = _lines(tmp_path / "c", "text: one two, y: 900, at: 0s, in: {type: pop, by: char, dur: 0.5s, stagger: 3f}").lines[0]
    assert c.by == "char" and c.dur == 15 and c.stagger == 3 and c.motion.dur == 15


def test_in_rise_and_type_resolve_like_reveal(tmp_path: Path) -> None:
    """`in: rise`/`in: type` reach the renderer with the reveal's timing; the frames match on the GPU."""
    old = _lines(tmp_path / "old", "text: one two, y: 900, at: 0s, engine: gpu").lines[0]
    new = _lines(tmp_path / "new", "text: one two, y: 900, at: 0s, in: rise").lines[0]
    assert (new.at, new.stagger, new.dur, new.rise, new.by) == (old.at, old.stagger, old.dur, old.rise, old.by)
    old = _lines(tmp_path / "ot", "text: one two, y: 900, at: 0s, engine: gpu, reveal: type").lines[0]
    new = _lines(tmp_path / "nt", "text: one two, y: 900, at: 0s, in: type").lines[0]
    assert (new.at, new.stagger, new.dur, new.rise, new.by) == (old.at, old.stagger, old.dur, old.rise, old.by)


# --- exits, emphasis, idle -------------------------------------------------------------------------


def test_exit_ends_at_block_to_across_wrapped_pieces(tmp_path: Path) -> None:
    line = ("text: alpha beta gamma delta epsilon zeta, size: 120, y: 600, at: 0s, wrap: balance, max_lines: 3, "
            "out: {type: fade, stagger: 2f}")
    block = _lines(tmp_path, line)
    assert len(block.lines) > 1  # wrapped
    assert block.fade_out == 0  # the exit replaces the block fade
    to = block.to
    units = [(ln, i) for ln in block.lines for i in range(len(ln.text.split(" ")))]
    starts = [ln.motion.out_at + i * ln.motion.out_stagger for ln, i in units]
    assert all(b - a == pytest.approx(2) for a, b in zip(starts, starts[1:]))  # continuous across pieces
    last, i = units[-1]
    assert starts[-1] + last.motion.out_dur == pytest.approx(to)
    assert _pose(last, i, to - 1e-3).opacity > 0 and _pose(last, i, to) == HIDDEN
    first, _ = units[0]
    assert _pose(first, 0, starts[0] - 1) == REST and _pose(first, 0, starts[0] + first.motion.out_dur) == HIDDEN


@pytest.mark.parametrize("kind", M.OUT_TYPES)
def test_each_out_type_is_hidden_from_its_end(tmp_path: Path, kind: str) -> None:
    ln = _lines(tmp_path, f"{WORD}, out: {kind}").lines[0]
    end = ln.motion.out_at + ln.motion.out_dur
    assert end == pytest.approx(90)
    assert _pose(ln, 0, ln.motion.out_at - 0.5) == REST
    assert _pose(ln, 0, end) == HIDDEN
    if kind == "pop-out":
        assert _pose(ln, 0, ln.motion.out_at + 0.3 * ln.motion.out_dur).scale == pytest.approx(M.POP_OUT_PEAK)
    if kind == "slide":
        assert _pose(ln, 0, ln.motion.out_at + 3).dx > 0  # default: off to the right


def test_emphasis_slam_and_shake_are_seeded(tmp_path: Path) -> None:
    text = 'text: "big *news* today", y: 900, at: 0s, in: pop, emphasis: {in: slam, shake: 6}'
    ln = _lines(tmp_path / "a", text).lines[0]
    again = _lines(tmp_path / "b", text).lines[0]
    m = ln.motion
    assert m.emph_kind == "slam" and m.shake == 6 and m.shake_dur == round(M.SHAKE_DUR * FPS)
    hit = ln.at + ln.stagger + m.emph_dur  # unit 1 lands
    assert _pose(ln, 1, ln.at + ln.stagger, emph=True).scale == pytest.approx(m.slam_from or M.SLAM_FROM)  # #24 caps it off the neighbours
    shaken = [_pose(ln, 1, hit + s, emph=True) for s in (0.5, 1.5, 3.5)]
    assert any(abs(p.dx) > 0.1 or abs(p.dy) > 0.1 for p in shaken)
    assert all(abs(p.dx) <= 6 and abs(p.dy) <= 6 for p in shaken)
    assert shaken == [_pose(again, 1, hit + s, emph=True) for s in (0.5, 1.5, 3.5)]  # deterministic
    assert _pose(ln, 1, hit + m.shake_dur + M.SLAM_SETTLE) == REST  # settled after the shake
    assert _pose(ln, 0, hit + 1) == REST  # plain words do not shake


def test_idle_float_loops_after_landing(tmp_path: Path) -> None:
    ln = _lines(tmp_path, f"{WORD}, in: pop, idle: {{type: float, amount: 5}}").lines[0]
    land = ln.motion.dur
    assert _pose(ln, 0, land) == REST
    quarter = land + ln.motion.idle_period / 4
    assert _pose(ln, 0, quarter).dy == pytest.approx(-5)
    pulse = _lines(tmp_path / "p", f"{WORD}, idle: pulse").lines[0]  # no `in`: rises, then pulses 4 %
    assert pulse.motion.kind == "rise"
    assert _pose(pulse, 0, pulse.motion.dur + pulse.motion.idle_period / 4).scale == pytest.approx(1.04)


# --- hash and validation ---------------------------------------------------------------------------


def test_motion_reaches_the_hash_and_unset_fields_do_not(tmp_path: Path) -> None:
    base = _hashes(_spec(tmp_path / "a", "text: Hi there, y: 900, at: 0s, engine: gpu"))
    pop = _hashes(_spec(tmp_path / "b", "text: Hi there, y: 900, at: 0s, in: pop"))
    slam = _hashes(_spec(tmp_path / "c", "text: Hi there, y: 900, at: 0s, in: slam"))
    ov = _hashes(_spec(tmp_path / "d", "text: Hi there, y: 900, at: 0s, in: {type: pop, overshoot: 1.2}"))
    assert len({tuple(base), tuple(pop), tuple(slam), tuple(ov)}) == 4
    blurred = _hashes(_spec(tmp_path / "e", "text: Hi there, y: 900, at: 0s, in: pop", block="motion_blur: true"))
    assert blurred != pop
    explicit_off = _hashes(_spec(tmp_path / "f", "text: Hi there, y: 900, at: 0s, engine: gpu", block="motion_blur: false"))
    assert explicit_off == base
    tl = _lines(tmp_path / "g", "text: Hi there, y: 900, at: 0s, engine: gpu")
    assert tl.lines[0].motion is None and tl.motion_blur is False


@pytest.mark.parametrize(
    ("extra", "where"),
    [
        ("in: wobble", "lines[0].in.type"),
        ("in: {type: pop, by: line}", "lines[0].in.by"),
        ("in: {type: type, dur: 1s}", "lines[0].in.dur"),
        ("in: {type: pop, from: left}", "lines[0].in.from"),
        ("in: {type: slide, from: up}", "lines[0].in.from"),
        ("in: {type: drop, overshoot: 1.1}", "lines[0].in.overshoot"),
        ("in: {type: pop, overshoot: 3}", "lines[0].in.overshoot"),
        ("in: {type: pop, distance: 40}", "lines[0].in.distance"),
        ("in: pop, reveal: type", "lines[0].reveal"),
        ("out: {type: pop-out}, reveal: pop", "lines[0].reveal"),
        ("out: vanish", "lines[0].out.type"),
        ("out: {type: cut, dur: 1s}", "lines[0].out.dur"),
        ("out: {type: fade, to: left}", "lines[0].out.to"),
        ("idle: spin", "lines[0].idle.type"),
        ("idle: {type: float, amount: 500}", "lines[0].idle.amount"),
    ],
)
def test_bad_motion_is_rejected(tmp_path: Path, extra: str, where: str) -> None:
    assert any(where in e for e in _errors(tmp_path, f"text: one two, y: 900, at: 0s, {extra}"))


def test_bad_emphasis_motion_is_rejected(tmp_path: Path) -> None:
    errs = _errors(tmp_path, 'text: "a *b*", y: 900, at: 0s, emphasis: {in: type, shake: 99}')
    assert any("emphasis.in" in e for e in errs) and any("emphasis.shake" in e for e in errs)


def test_motion_fields_make_a_chromium_line_an_error(tmp_path: Path) -> None:
    assert any("lines[0].in" in e for e in _errors(tmp_path, "text: a, y: 900, at: 0s, engine: chromium, in: pop"))


# --- GPU frames ------------------------------------------------------------------------------------


def _render(tmp: Path, line: str, frames, block: str = "") -> list[torch.Tensor]:
    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer

    p = _spec(tmp, line, block=block)
    tl = resolve(load_spec(p), p.parent)
    cv = Canvas(*tl.size, device="cuda")
    r = Renderer(tl, p.parent, cv)
    with torch.no_grad():
        return [r.frame(f).float() for f in frames]


def _box(img: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float, float] | None:
    d = (img - ref).abs().sum(0) > 0.15
    if not d.any():
        return None
    ys, xs = torch.nonzero(d, as_tuple=True)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


@pytest.fixture
def blank_ref(tmp_path: Path):
    def make(frames):
        return _render(tmp_path / "blank", "text: x, y: 3000, at: 0s, engine: gpu", frames)
    return make


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("kind", "peak", "settle"),
    [("pop", 4, 6), ("slam", 3, 8), ("slide", 2, 9), ("drop", 3, 15), ("rise", 6, 27), ("flip", 3, 9),
     ("scale-in", 4, 8)],
)
def test_in_frames_start_peak_settle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blank_ref,
                                     kind: str, peak: int, settle: int) -> None:
    """Frame 0 shows nothing; the peak frame differs from rest the way the type moves; the settle
    frame is pixel-equal to the same line at rest."""
    _cuda()
    started = _no_browser(monkeypatch)
    extra = ", from: left" if kind == "slide" else ""
    frames = [0, peak, settle, 60]
    start, mid, done, late = _render(tmp_path / kind, f"{WORD}, in: {{type: {kind}{extra}}}", frames)
    still = _render(tmp_path / "still", f"{WORD}, engine: gpu", [60])[0]
    ref = blank_ref([0])[0]
    assert _box(start, ref) is None, kind  # every entry starts transparent
    assert torch.equal(late, still), kind  # at rest: the plain gpu line, bit for bit
    assert torch.equal(done, late), kind
    rest, at = _box(still, ref), _box(mid, ref)
    assert at is not None and at != rest, kind
    w, h = rest[2] - rest[0], rest[3] - rest[1]
    aw, ah = at[2] - at[0], at[3] - at[1]
    if kind == "pop":
        assert aw > w * 1.03, (kind, at, rest)  # overshoot past full size
    elif kind == "scale-in":
        assert aw < w * 0.97, (kind, at, rest)  # still growing (no overshoot by default)
    elif kind == "slam":
        assert aw > w * 1.2, (kind, at, rest)  # still large
    elif kind == "slide":
        assert at[0] < rest[0] - 20, (kind, at, rest)  # coming from the left
    elif kind == "drop":
        assert at[1] < rest[1] - 20, (kind, at, rest)  # above its line
    elif kind == "rise":
        assert at[3] > rest[3] + 2, (kind, at, rest)  # below its line
    elif kind == "flip":
        assert ah < h * 0.9 and abs(aw - w) < 0.1 * w, (kind, at, rest)  # squashed, full width
    assert not started


@pytest.mark.gpu
@pytest.mark.parametrize(("new", "old"), [("in: rise", "engine: gpu"), ("in: type", "engine: gpu, reveal: type")])
def test_in_rise_and_type_frames_equal_the_reveal(tmp_path: Path, new: str, old: str) -> None:
    """`reveal: rise` output is unchanged: `in: rise` / `in: type` draw the same pixels at every frame."""
    _cuda()
    frames = [0, 2, 5, 9, 14, 20, 27, 40]
    line = "text: one two three, y: 900, at: 0s"
    a = _render(tmp_path / "new", f"{line}, {new}", frames)
    b = _render(tmp_path / "old", f"{line}, {old}", frames)
    for f, x, y in zip(frames, a, b):
        assert (x - y).abs().max() < 1e-3, (new, f)


@pytest.mark.gpu
def test_out_leaves_nothing_at_block_end(tmp_path: Path, blank_ref) -> None:
    _cuda()
    imgs = _render(tmp_path, f"{WORD}, in: pop, out: {{type: slide, to: right}}", [60, 87, 89])
    refs = blank_ref([60, 87, 89])
    rest, moving, last = (_box(i, r) for i, r in zip(imgs, refs))
    assert rest is not None and moving is not None and moving[0] > rest[0] + 10
    assert last is None or last[0] > rest[0] + 100


@pytest.mark.gpu
def test_text_motion_blur_smears_a_moving_unit(tmp_path: Path, blank_ref) -> None:
    _cuda()
    line = f"{WORD}, in: {{type: slide, from: left, distance: 400}}"
    sharp = _render(tmp_path / "sharp", line, [3, 60])
    soft = _render(tmp_path / "soft", line, [3, 60], block="motion_blur: true")
    ref = blank_ref([3])[0]
    assert torch.allclose(sharp[1], soft[1], atol=1e-4)  # at rest the sub-frames agree
    a, b = _box(sharp[0], ref), _box(soft[0], ref)
    assert a is not None and b is not None
    assert (b[2] - b[0]) > (a[2] - a[0]) + 10  # the smear widens the unit along its path
