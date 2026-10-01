"""M2 scene evaluation and the golden-film finish order. CPU; no Chromium."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from montaj.pipeline import engine_version
from montaj.project import segment_hash
from montaj.render.canvas import Canvas
from montaj.render.core import sine, smooth, still
from montaj.render.finish import LOOKS, finish
from montaj.render.frame import Renderer
from montaj.render.shots import focus_frac, kenburns, tone
from montaj.spec import load_spec
from montaj.timeline import Blank, FramePlan, Trans, resolve


def _edge(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (64, 80), (220, 30, 30))
    ImageDraw.Draw(im).rectangle((0, 0, 31, 79), fill=(20, 30, 220))
    im.save(path, quality=95)


def _spec(tmp: Path, shots: str, *, size: str = "180x320", background: str = "#102030") -> Path:
    _edge(tmp / "assets" / "a.jpg")
    path = tmp / "montaj.yaml"
    path.write_text(
        f"video:\n  size: {size}\n  fps: 30\n  look: warm-film\n"
        f"  background: \"{background}\"\n  motion_blur: 1\n"
        f"assets: assets\nshots:\n{shots}"
    )
    return path


def _open(path: Path) -> tuple:
    tl = resolve(load_spec(path), path.parent)
    cv = Canvas(tl.size[0], tl.size[1], device="cpu")
    return tl, Renderer(tl, path.parent, cv)


def test_blank_is_the_background(tmp_path: Path) -> None:
    tl, renderer = _open(_spec(tmp_path, "  - {blank: true, hold: 4f}\n"))
    assert isinstance(tl.plan(0).scene, Blank)
    img = renderer._eval(tl.plan(0).scene, 0.0)
    want = torch.tensor([0x10, 0x20, 0x30], dtype=torch.float32) / 255
    assert torch.allclose(img, want[:, None, None].expand_as(img))


def test_fade_from_blank_and_blur(tmp_path: Path) -> None:
    sharp = _spec(tmp_path / "sharp", "  - {photo: a, hold: 12f, in: {type: fade, dur: 8f, blur: 0}}\n")
    soft = _spec(tmp_path / "soft", "  - {photo: a, hold: 12f, in: {type: fade, dur: 8f, blur: 60}}\n")
    tl0, r0 = _open(sharp)
    tl1, r1 = _open(soft)
    assert isinstance(tl1.plan(0).scene, Trans)
    bg = torch.tensor([0x10, 0x20, 0x30], dtype=torch.float32) / 255
    at0 = r1._eval(tl1.plan(0).scene, 0.0)
    assert torch.allclose(at0, bg[:, None, None].expand_as(at0), atol=1e-5)
    # blur 0 stays on the M1 fade. A wide blur at mid-fade moves the edge.
    mid0 = r0._eval(tl0.plan(4).scene, 4.0)
    mid1 = r1._eval(tl1.plan(4).scene, 4.0)
    assert (mid0 - mid1).abs().max() > 0.02


def test_m1_still_matches_still_and_drift_moves(tmp_path: Path) -> None:
    tl, renderer = _open(_spec(
        tmp_path / "m1",
        "  - {photo: a, hold: 6f, focus: [20, 30], zoom: 1.4}\n",
        size="96x128",
    ))
    sh = tl.shots[0].spec
    photo = renderer._photo(sh.photo, sh.crop, False)
    expect = still(renderer.cv, photo, focus=sh.focus, zoom=sh.zoom)
    got = renderer._eval(tl.plan(0).scene, 0.0)
    assert (got - expect).abs().max() < 1e-5
    assert torch.equal(got, renderer._eval(tl.plan(3).scene, 3.0))

    tl, renderer = _open(_spec(
        tmp_path / "drift",
        "  - {photo: a, hold: 10f, drift: {zoom: [1, 1.8]}}\n",
        size="96x128",
    ))
    early = renderer._eval(tl.plan(0).scene, 0.0)
    late = renderer._eval(tl.plan(9).scene, 9.0)
    assert (early - late).abs().max() > 0.01


def test_clip_window_follows_clip_in(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=160x120:rate=30:duration=1", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )

    def project(name: str, clip_in: str):
        proj = tmp_path / name
        assets = proj / "assets"
        assets.mkdir(parents=True)
        (assets / "bars.mp4").write_bytes(src.read_bytes())
        path = proj / "montaj.yaml"
        path.write_text(
            "video:\n  size: 160x120\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
            "assets: assets\nshots:\n"
            f"  - {{clip: bars, hold: 10f, clip_in: {clip_in}}}\n"
        )
        tl, renderer = _open(path)
        return renderer._eval(tl.plan(0).scene, 0.0), renderer._eval(tl.plan(5).scene, 5.0)

    head, later = project("zero", "0s")
    shifted, _later = project("shift", "0.4s")
    assert float(head.std()) > 0.02
    assert (head - later).abs().max() > 0.02
    assert (head - shifted).abs().max() > 0.02


def test_clip_decodes_through_the_outgoing_fade(tmp_path: Path) -> None:
    """A 10f hold plus a 6f fade must show source frames 10–15, not freeze on frame 9."""
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=160x120:rate=30:duration=1", "-frames:v", "16",
         "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    proj = tmp_path / "long"
    assets = proj / "assets"
    assets.mkdir(parents=True)
    (assets / "bars.mp4").write_bytes(src.read_bytes())
    path = proj / "montaj.yaml"
    path.write_text(
        "video:\n  size: 160x120\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
        "assets: assets\nshots:\n"
        "  - {clip: bars, hold: 10f}\n"
        "  - {blank: true, hold: 8f, in: {type: fade, dur: 6f}}\n"
    )
    tl, renderer = _open(path)
    scene = tl.plan(12).scene
    assert isinstance(scene, Trans)
    during = renderer._eval(scene.a, 12.0)
    frozen = renderer._eval(scene.a, 9.0)
    assert (during - frozen).abs().max() > 0.02

    short = tmp_path / "short.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=160x120:rate=30:duration=1", "-frames:v", "10",
         "-pix_fmt", "yuv420p", str(short)],
        check=True,
    )
    (assets / "bars.mp4").write_bytes(short.read_bytes())
    tl, renderer = _open(path)
    scene = tl.plan(12).scene
    assert isinstance(scene, Trans)
    # The source ends at frame 9. The overlap clamps there instead of raising.
    assert torch.allclose(renderer._eval(scene.a, 12.0), renderer._eval(scene.a, 9.0))


def test_tone_keeps_explicit_zoom(tmp_path: Path) -> None:
    tl, renderer = _open(_spec(
        tmp_path,
        "  - {photo: a, hold: 4f, focus: [20, 30], zoom: 1.4, tone: {brightness: 0.5}}\n",
        size="96x128",
    ))
    sh = tl.shots[0].spec
    photo = renderer._photo(sh.photo, sh.crop, False)
    expect = tone(still(renderer.cv, photo, focus=sh.focus, zoom=sh.zoom), 0.5, 1.0, 1.0)
    got = renderer._eval(tl.plan(0).scene, 0.0)
    assert (got - expect).abs().max() < 1e-5
    fx, fy = focus_frac(photo, sh.focus)
    cover = tone(kenburns(renderer.cv, photo, fx, fy, 1.0, 0.0, 0.0), 0.5, 1.0, 1.0)
    assert (got - cover).abs().max() > 0.02


def test_m2_blurred_fade_uses_sine_and_m1_fade_stays_smooth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """film.html 411: vis() returns es(P(...)). es (line 73) is sine. Line 417 blurs by (1−a)·blur.

    sine(0.25) is about 0.146447. smoothstep(0.25) is 0.15625. M1 blur stays on smoothstep.
    """
    shots = (
        "  - {photo: a, hold: 8f}\n"
        "  - {photo: b, hold: 12f, in: {type: fade, dur: 8f, blur: 40}}\n"
    )
    soft = _spec(tmp_path / "soft", shots, size="96x128")
    Image.new("RGB", (64, 80), (20, 200, 40)).save(soft.parent / "assets" / "b.jpg")
    tl, renderer = _open(soft)
    scene = tl.plan(10).scene
    assert isinstance(scene, Trans) and scene.blur == 40
    a = renderer._eval(scene.a, 10.0)
    b = renderer._eval(scene.b, 10.0)
    sigmas: list[float] = []

    def fake_blur(img, sigma):
        sigmas.append(float(sigma))
        return img

    monkeypatch.setattr("montaj.render.frame.css_blur", fake_blur)
    mixed = renderer._eval(scene, 10.0)
    e = sine(0.25)
    assert e == pytest.approx(0.1464466, abs=1e-5)
    assert sigmas == pytest.approx([e * 40 * renderer.cv.k, (1.0 - e) * 40 * renderer.cv.k])
    assert torch.allclose(mixed, a * (1.0 - e) + b * e, atol=1e-5)
    assert not torch.allclose(mixed, a * (1.0 - smooth(0.25)) + b * smooth(0.25), atol=1e-4)

    hard = _spec(tmp_path / "hard", shots.replace("blur: 40", "blur: 0"), size="96x128")
    Image.new("RGB", (64, 80), (20, 200, 40)).save(hard.parent / "assets" / "b.jpg")
    tl0, r0 = _open(hard)
    scene0 = tl0.plan(10).scene
    assert isinstance(scene0, Trans)
    a0 = r0._eval(scene0.a, 10.0)
    b0 = r0._eval(scene0.b, 10.0)
    mixed0 = r0._eval(scene0, 10.0)
    s = smooth(0.25)
    assert s == pytest.approx(0.15625)
    assert torch.allclose(mixed0, a0 * (1.0 - s) + b0 * s, atol=1e-5)


def _plan(**kwargs) -> FramePlan:
    base = dict(f=15, scene=Blank(), flashes=(), leaks=(), intro=None, outro=None)
    base.update(kwargs)
    return FramePlan(**base)


def test_golden_finish_order_converts_frames_to_seconds(monkeypatch) -> None:
    order: list[str] = []
    seen: dict = {}

    def grab(name: str, img_at: int):
        def fn(*args, **_kwargs):
            order.append(name)
            return args[img_at]
        return fn

    def dust(cv, img, t, level, bursts):
        order.append("dust")
        seen["bursts"] = bursts
        seen["t"] = t
        return img

    def hits(cv, img, t, hit_list):
        order.append("hits")
        seen["hits"] = hit_list
        return img

    def fade(img, f, start, dur):
        order.append("fade")
        seen["fade"] = (f, start, dur)
        return img

    monkeypatch.setattr("montaj.render.finish.leaks", grab("leaks", 1))
    monkeypatch.setattr("montaj.render.finish.dust", dust)
    monkeypatch.setattr("montaj.render.finish.vignette", grab("vignette", 1))
    monkeypatch.setattr("montaj.render.finish.grain", grab("grain", 1))
    monkeypatch.setattr("montaj.render.finish.hits", hits)
    monkeypatch.setattr("montaj.render.finish.bars", grab("bars", 1))
    monkeypatch.setattr("montaj.render.finish.fade_layer", fade)

    plan = _plan(
        hits=((30.0, 0.5),),
        bursts=((2, 45.0, 0.2, 0.3, 5),),
        dust=0.4,
        glow=0.8,
        bars=20.0,
        fade_out=(36, 100),
    )

    def overlay(img):
        order.append("text")
        return img

    img = torch.zeros(3, 8, 8)
    finish(None, img, 15, {"kind": "golden-film", "fps": 30.0, "dur": 5.0}, plan, overlay)
    assert order == ["leaks", "dust", "text", "vignette", "grain", "hits", "bars", "fade"]
    assert seen["t"] == 0.5
    assert seen["bursts"] == ((2, 1.5, 0.2, 0.3, 5),)
    assert seen["hits"] == ((1.0, 0.5),)
    assert seen["fade"] == (15, 100, 36)


def test_warm_plan_matches_the_legacy_call_and_overlay_is_last() -> None:
    cv = Canvas(32, 48, device="cpu")
    img = torch.rand(3, 48, 32)
    plan = _plan(
        f=3,
        flashes=((0, 0.4),),
        leaks=((1, 0.2, 1),),
        intro=("white-lift", 10),
    )
    legacy = finish(cv, img, 3, LOOKS["warm-film"], plan.flashes, plan.leaks, plan.intro, plan.outro)
    got = finish(cv, img, 3, LOOKS["warm-film"], plan, None)
    assert torch.equal(legacy, got)
    painted = finish(cv, img, 3, LOOKS["warm-film"], plan, lambda frame: torch.full_like(frame, 0.25))
    assert torch.allclose(painted, torch.full_like(img, 0.25))


def test_warm_fade_out_darkens_after_the_overlay() -> None:
    cv = Canvas(32, 48, device="cpu")
    img = torch.full((3, 48, 32), 0.8)
    plain = finish(cv, img, 20, LOOKS["warm-film"], _plan(f=20), None)
    faded = finish(cv, img, 20, LOOKS["warm-film"], _plan(f=20, fade_out=(10, 15)), None)
    assert torch.allclose(faded, plain * 0.5, atol=1e-4)
    def white(frame):
        return torch.ones_like(frame)

    held = finish(cv, img, 20, LOOKS["warm-film"], _plan(f=20), white)
    assert torch.allclose(held, torch.ones_like(img))
    # Halfway: the overlay is painted, then the fade pulls it to half, including the white.
    mid = finish(cv, img, 20, LOOKS["warm-film"], _plan(f=20, fade_out=(10, 15)), white)
    assert torch.allclose(mid, torch.full_like(img, 0.5), atol=1e-5)
    end = finish(cv, img, 25, LOOKS["warm-film"], _plan(f=25, fade_out=(10, 15)), white)
    assert torch.allclose(end, torch.zeros_like(img))


def test_later_hold_changes_the_leak_and_the_earlier_segment_hash(tmp_path: Path) -> None:
    """glow=1, two blanks, 90f then 30f. Stretching the second hold to 90f moves frame 75."""

    def film(hold: str) -> Path:
        proj = tmp_path / hold
        proj.mkdir()
        (proj / "assets").mkdir()
        path = proj / "montaj.yaml"
        path.write_text(
            "video:\n  size: 180x320\n  fps: 30\n  look: golden-film\n  motion_blur: 1\n"
            "assets: assets\ntracks:\n  glow:\n    - {t: 0s, v: 1}\n"
            f"shots:\n  - {{blank: true, hold: 90f}}\n  - {{blank: true, hold: {hold}}}\n"
        )
        return path

    short_path = film("30f")
    long_path = film("90f")
    short = resolve(load_spec(short_path), short_path.parent)
    long = resolve(load_spec(long_path), long_path.parent)
    assert segment_hash(short, short.segments[0], {}, engine_version()) != segment_hash(
        long, long.segments[0], {}, engine_version(),
    )
    a = Renderer(short, short_path.parent, Canvas(*short.size, device="cpu")).frame(75)
    b = Renderer(long, long_path.parent, Canvas(*long.size, device="cpu")).frame(75)
    assert (a - b).abs().max() > 0.05
    # Only segments inside the last 2 s depend on the film length: 90f vs 120f leaves segment 0 alone.
    longer_path = film("120f")
    longer = resolve(load_spec(longer_path), longer_path.parent)
    assert segment_hash(long, long.segments[0], {}, engine_version()) == segment_hash(
        longer, longer.segments[0], {}, engine_version(),
    )
