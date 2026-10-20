"""Clip shots: crop, camera keys, speed and freeze. Time map, spec checks, hash, framing."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from montaj.clipmap import camera_at, source_index
from montaj.project import file_sha, segment_hash
from montaj.render.canvas import Canvas
from montaj.render.clip import ClipSource
from montaj.render.frame import Renderer
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

# ---------------------------------------------------------------- time map


def test_constant_speed_fills_the_hold_exactly() -> None:
    # speed 2 on a 3 s (90 frame) source fills a 1.5 s (45 frame) hold: frames 0, 2, ... 88.
    table = source_index(2.0, None, 45)
    assert table == tuple(range(0, 90, 2))
    assert max(table) + 1 <= 90


def test_fractional_and_slow_speed_floor_to_frames() -> None:
    assert source_index(1.5, None, 6) == (0, 1, 3, 4, 6, 7)
    assert source_index(0.5, None, 6) == (0, 0, 1, 1, 2, 2)
    assert source_index(1.0, None, 4) == (0, 1, 2, 3)


def test_freeze_holds_the_source_frame_at_its_time() -> None:
    assert source_index(None, 3, 7) == (0, 1, 2, 3, 3, 3, 3)
    assert source_index(2.0, 3, 7) == (0, 2, 4, 6, 6, 6, 6)
    assert source_index(None, None, 5) is None


def test_speed_ramp_integrates_the_curve() -> None:
    # Linear ramp from 1x to 3x over 30 frames, then 3x: position = t + t^2 / 30 for t <= 30.
    keys = [(0, 1.0, "linear"), (30, 3.0, "linear")]
    table = source_index(keys, None, 45)
    assert table[0] == 0 and table[30] == 60 and table[44] == 60 + 14 * 3
    assert all(b >= a for a, b in zip(table, table[1:]))
    assert table[15] == int(15 + 15**2 / 30)
    # An ease that starts slow is behind the linear ramp and ends at the same speed.
    slow = source_index([(0, 1.0, "inc"), (30, 3.0, "inc")], None, 45)
    assert slow[15] < table[15] and slow[44] - slow[43] == 3


def test_ramp_then_freeze_stops_at_the_freeze_frame() -> None:
    keys = [(0, 1.0, "linear"), (10, 2.0, "linear")]
    table = source_index(keys, 5, 12)
    assert table[5:] == (table[5],) * 7


def test_camera_defaults_hold_and_fractions() -> None:
    crop = (100, 50, 300, 250)
    # No keys given values: zoom 1 at the crop centre.
    assert camera_at([(0, None, None, "linear")], crop, 3.0) == (1.0, 0.5, 0.5)
    keys = [(0, 1.0, None, "linear"), (20, 4.0, (200.0, 100.0), "linear")]
    z, fx, fy = camera_at(keys, crop, 0.0)
    assert (z, fx, fy) == (1.0, 0.5, 0.5)
    z, fx, fy = camera_at(keys, crop, 10.0)
    assert z == pytest.approx(2.0)  # geometric midpoint of 1 and 4
    # Focus runs from the crop centre (200, 150) to (200, 100): x stays, y moves a quarter of the way up.
    assert (fx, fy) == pytest.approx((0.5, 0.5 - 0.125))
    z, fx, fy = camera_at(keys, crop, 99.0)
    assert z == pytest.approx(4.0) and (fx, fy) == pytest.approx((0.5, 0.25))
    # A key with no zoom holds the previous one.
    held = [(0, 2.0, None, "linear"), (10, None, (250.0, 150.0), "linear")]
    assert camera_at(held, crop, 10.0)[0] == pytest.approx(2.0)


# ---------------------------------------------------------------- fixtures


def _ffmpeg(*args: str) -> None:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-400:]


def _numbered_clip(path: Path, seconds: int = 3) -> None:
    """320x240 @ 30. Red is the frame number, green the x gradient, blue the y gradient."""
    _ffmpeg(
        "-f", "lavfi", "-i", f"color=size=320x240:rate=30:duration={seconds},format=gbrp,"
        "geq=r='N':g='255*X/W':b='255*Y/H'",
        "-c:v", "ffv1", "-pix_fmt", "rgb24", str(path),
    )


def _project(tmp: Path, shots: str, *, size: str = "108x192") -> Path:
    assets = tmp / "assets"
    assets.mkdir(exist_ok=True)
    if not (assets / "rec.mkv").exists():
        _numbered_clip(assets / "rec.mkv")
    path = tmp / "montaj.yaml"
    path.write_text(
        f"video:\n  size: {size}\n  fps: 30\n  bpm: 120\n  look: warm-film\nassets: assets\nshots:\n{shots}\n"
    )
    return path


def _errors(tmp: Path, shots: str) -> list[str]:
    with pytest.raises(SpecError) as ei:
        load_spec(_project(tmp, shots))
    return ei.value.errors


# ---------------------------------------------------------------- decode


def test_decode_crop_is_the_source_region(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    _numbered_clip(assets / "rec.mkv")
    clip = ClipSource(assets / "rec.mkv", 30)
    full = clip.frames(0.0, 320, 240, nframes=4)
    boxed = clip.frames(0.0, 120, 80, nframes=4, crop=(20, 10, 140, 90))
    assert boxed[0].shape == (3, 80, 120)
    for i in range(4):
        assert torch.equal(boxed[i], full[i][:, 10:90, 20:140])
    # A crop is its own window.
    assert len(clip._cache) == 2


def test_src_table_picks_source_frames(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    _numbered_clip(assets / "rec.mkv")
    clip = ClipSource(assets / "rec.mkv", 30)
    src = tuple(range(0, 90, 2))  # speed 2 over a 45 frame hold
    for i in (0, 1, 20, 44):
        got = clip.frame(100 + i + 0.4, 100, 0.0, 320, 240, src=src)
        assert round(float(got[0, 0, 0]) * 255) == 2 * i
    # Past the hold the last mapped frame stays; the 89-frame window is all that was decoded.
    assert round(float(clip.frame(500, 100, 0.0, 320, 240, src=src)[0, 0, 0]) * 255) == 88
    assert list(clip._cache)[0][3] == 89


def test_big_frames_shrink_the_frame_budget(tmp_path: Path) -> None:
    from montaj.render import clip as clip_mod

    assets = tmp_path / "assets"
    assets.mkdir()
    _numbered_clip(assets / "rec.mkv")
    clip = ClipSource(assets / "rec.mkv", 30)
    assert clip._cap == clip_mod._CACHE_FRAMES
    clip.frames(0.0, 320, 240, nframes=2)
    assert clip._cap == clip_mod._CACHE_FRAMES
    clip.frames(0.0, 3840, 2160, nframes=1, crop=(0, 0, 320, 240))
    assert clip._cap == clip_mod._CACHE_BYTES // (3840 * 2160 * 3)


# ---------------------------------------------------------------- spec


def test_keys_become_frames_and_reach_the_hash(tmp_path: Path) -> None:
    shots = """  - clip: rec
    hold: 60f
    crop: [20, 10, 300, 230]
    camera:
      - {t: 0s, zoom: 1.0}
      - {t: 1s, zoom: 1.5, focus: [200, 100], ease: outc}
    speed:
      - {t: 0s, v: 1}
      - {t: 0.5s, v: 1.5, ease: inc}
    freeze: 1.5s
"""
    tl = resolve(load_spec(_project(tmp_path, shots)), tmp_path)
    sh = tl.shots[0].spec
    assert [k.t for k in sh.camera] == [0, 30]
    assert [k.t for k in sh.speed] == [0, 15]
    assert sh.freeze == 45
    shas = {"clip:rec": file_sha(tmp_path / "assets" / "rec.mkv")}
    base = segment_hash(tl, tl.segments[0], shas, "e1")

    def edited(old: str, new: str) -> str:
        other = tmp_path / "other"
        other.mkdir(exist_ok=True)
        (other / "assets").mkdir(exist_ok=True)
        (other / "assets" / "rec.mkv").write_bytes((tmp_path / "assets" / "rec.mkv").read_bytes())
        tl2 = resolve(load_spec(_project(other, shots.replace(old, new))), other)
        return segment_hash(tl2, tl2.segments[0], shas, "e1")

    assert edited("freeze: 1.5s", "freeze: 45f") == base  # same frames, same bytes to the renderer
    for old, new in (("zoom: 1.5", "zoom: 1.6"), ("v: 1.5", "v: 1.4"), ("freeze: 1.5s", "freeze: 1.4s"),
                     ("crop: [20, 10, 300, 230]", "crop: [20, 10, 300, 220]")):
        assert edited(old, new) != base, old


@pytest.mark.parametrize(
    ("shots", "needle"),
    [
        ("  - {clip: rec, hold: 30f, speed: 0}\n", "shots[0].speed: must be finite and > 0"),
        ("  - {clip: rec, hold: 30f, speed: -1}\n", "reverse playback"),
        ("  - {clip: rec, hold: 30f, speed: [{t: 0f, v: 1}, {t: 0f, v: 2}]}\n", "increasing time"),
        ("  - {clip: rec, hold: 30f, speed: [{t: 0f, v: 1, ease: bounce}]}\n", "unknown 'bounce'"),
        ("  - {clip: rec, hold: 30f, freeze: 30f}\n", "at or after the end"),
        ("  - {clip: rec, hold: 30f, camera: [{t: 0f, zoom: 0.5}]}\n", "multiple of cover"),
        ("  - {clip: rec, hold: 30f, camera: [{t: 9f}, {t: 3f}]}\n", "increasing time"),
        ("  - {clip: rec, hold: 30f, camera: []}\n", "need at least one key"),
        ("  - {clip: rec, hold: 30f, camera: [{t: 0f}], zoom: 1.2}\n", "mutually exclusive"),
        ("  - {clip: rec, hold: 30f, zoom: 0.4}\n", "multiple of cover"),
        ("  - {clip: rec, hold: 30f, crop: [10, 10, 10, 50]}\n", "not a box"),
        ("  - {clip: rec, hold: 30f, crop: [0, 0, 400, 100]}\n", "outside the 320x240 source"),
        ("  - {clip: rec, hold: 30f, crop: [0, 0, 100, 100], focus: [200, 50]}\n", "outside the crop"),
        ("  - {clip: rec, hold: 30f, camera: [{t: 0f, focus: [999, 5]}]}\n", "camera[0].focus"),
        ("  - {clip: rec, hold: 60f, speed: 2}\n", "clip ends at 3.0s of source, needs 4.0s"),
        ("  - {clip: rec, hold: 30f, clip_in: 2s, speed: 1.5}\n", "needs"),
    ],
)
def test_clip_shot_errors_are_helpful(tmp_path: Path, shots: str, needle: str) -> None:
    errors = _errors(tmp_path, shots)
    assert any(needle in e for e in errors), errors
    assert all(e.endswith(")") for e in errors), errors


def test_camera_speed_freeze_need_a_clip(tmp_path: Path) -> None:
    import re

    shots = "  - {photo: a, hold: 30f, speed: 2, freeze: 10f, camera: [{t: 0f}]}\n"
    (tmp_path / "assets").mkdir()
    from PIL import Image

    Image.new("RGB", (8, 8)).save(tmp_path / "assets" / "a.jpg")
    path = tmp_path / "montaj.yaml"
    path.write_text(f"video:\n  size: 108x192\n  fps: 30\n  look: warm-film\nassets: assets\nshots:\n{shots}")
    with pytest.raises(SpecError) as ei:
        load_spec(path)
    for name in ("camera", "speed", "freeze"):
        assert any(re.search(rf"shots\[0\]\.{name}: {name} needs clip", e) for e in ei.value.errors), ei.value.errors


def test_speed_that_fits_is_accepted(tmp_path: Path) -> None:
    # 3 s source, speed 2, 1.5 s hold: the last frame read is source frame 88 of 90.
    tl = resolve(load_spec(_project(tmp_path, "  - {clip: rec, hold: 45f, speed: 2}\n")), tmp_path)
    assert tl.shots[0].spec.speed == 2.0


# ---------------------------------------------------------------- framing


def _renderer(tmp: Path, shots: str) -> tuple[Renderer, object]:
    tl = resolve(load_spec(_project(tmp, shots)), tmp)
    return Renderer(tl, tmp, Canvas(tl.size[0], tl.size[1], device="cpu")), tl


def _centre(img: torch.Tensor) -> tuple[float, float, float]:
    _c, h, w = img.shape
    px = img[:, h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1].mean(dim=(1, 2)) * 255
    return float(px[0]), float(px[1]), float(px[2])


def test_crop_and_focus_land_on_the_centre_pixel(tmp_path: Path) -> None:
    shots = """  - clip: rec
    hold: 30f
    crop: [80, 60, 240, 180]
    camera:
      - {t: 0f, zoom: 1.0}
      - {t: 20f, zoom: 2.0, focus: [200, 100], ease: linear}
"""
    rend, tl = _renderer(tmp_path, shots)
    scene = tl.plan(0).scene
    # Zoom 1 shows the crop centre (160, 120): green = 255 x / 320, blue = 255 y / 240.
    r, g, b = _centre(rend._clip_frame(scene, 0.0))
    assert (g, b) == pytest.approx((255 * 160 / 320, 255 * 120 / 240), abs=3)
    # At the end the focus point is centred (cover-clamped: at zoom 2 the crop still reaches it).
    r, g, b = _centre(rend._clip_frame(scene, 25.0))
    assert (g, b) == pytest.approx((255 * 200 / 320, 255 * 100 / 240), abs=3)
    # Frame 25 of the shot is source frame 25.
    assert r == pytest.approx(25, abs=0.6)


def test_camera_moves_every_frame_without_steps(tmp_path: Path) -> None:
    shots = """  - clip: rec
    hold: 30f
    camera:
      - {t: 0f, zoom: 1.0, focus: [160, 120]}
      - {t: 30f, zoom: 2.0, focus: [230, 120], ease: linear}
"""
    rend, tl = _renderer(tmp_path, shots)
    scene = tl.plan(0).scene
    greens = [_centre(rend._clip_frame(scene, float(t)))[1] for t in range(0, 30)]
    steps = [b - a for a, b in zip(greens, greens[1:])]
    # The focus glides every frame (the 8-bit source gradient adds about one level of jitter).
    assert min(steps) > 0
    mean = (greens[-1] - greens[0]) / len(steps)
    assert max(steps) < 2.5 * mean


def test_speed_and_freeze_pick_the_mapped_source_frame(tmp_path: Path) -> None:
    shots = "  - {clip: rec, hold: 40f, speed: 2, freeze: 30f}\n"
    rend, tl = _renderer(tmp_path, shots)
    scene = tl.plan(0).scene
    reds = [round(_centre(rend._clip_frame(scene, float(f)))[0]) for f in (0, 1, 10, 29, 30, 39)]
    assert reds == [0, 2, 20, 58, 60, 60]
    # The window holds only what the table reaches.
    assert rend._clip_plan(0).count == 61


def test_plain_clip_plan_is_unchanged(tmp_path: Path) -> None:
    rend, _tl = _renderer(tmp_path, "  - {clip: rec, hold: 40f}\n")
    plan = rend._clip_plan(0)
    assert (plan.crop, plan.src, plan.count, plan.w, plan.h) == (None, None, 40, 108, 192)
    assert rend._clip_request(0) == ("rec", 0.0, 40)


@pytest.mark.gpu
def test_gpu_clip_shot_renders_a_finished_frame(tmp_path: Path) -> None:
    shots = """  - clip: rec
    hold: 30f
    crop: [80, 60, 240, 180]
    camera:
      - {t: 0f, zoom: 1.0}
      - {t: 20f, zoom: 1.5, focus: [200, 100], ease: outc}
    speed: 1.5
  - {clip: rec, hold: 20f, in: {type: fade, dur: 8f}, freeze: 5f}
"""
    path = _project(tmp_path, shots, size="360x640")
    tl = resolve(load_spec(path), tmp_path)
    rend = Renderer(tl, tmp_path, Canvas(tl.size[0], tl.size[1], device="cuda"))
    for f in (0, 10, 29, 30, 40):
        frame = rend.frame(f)
        assert frame.shape == (3, 640, 360) and bool(torch.isfinite(frame).all())


# ---------------------------------------------------------------- review regressions


def _subsampled_clip(path: Path) -> None:
    """200x100 yuv420p, neutral chroma, luma a position-dependent pattern (no wrap-free neighbours)."""
    _ffmpeg(
        "-f", "lavfi", "-i", "color=size=200x100:rate=30:duration=1,format=yuv420p,"
        "geq=lum='mod(X*5+Y*3,256)':cb=128:cr=128",
        "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(path),
    )


@pytest.mark.parametrize("box", [(21, 11, 140, 90), (21, 11, 22, 12), (3, 5, 4, 100)])
def test_crop_is_exact_on_chroma_subsampled_video(tmp_path: Path, box: tuple[int, int, int, int]) -> None:
    """Odd offsets and sizes must not be rounded to the 2x2 chroma grid."""
    _subsampled_clip(tmp_path / "sub.mkv")
    clip = ClipSource(tmp_path / "sub.mkv", 30)
    x0, y0, x1, y1 = box
    full = clip.frames(0.0, 200, 100, nframes=1)[0]
    boxed = clip.frames(0.0, x1 - x0, y1 - y0, nframes=1, crop=box)[0]
    assert boxed.shape == (3, y1 - y0, x1 - x0)
    assert torch.allclose(boxed.float(), full[:, y0:y1, x0:x1].float(), atol=1.0)


def test_zero_fps_is_an_error_not_a_crash(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    _numbered_clip(assets / "rec.mkv")
    path = tmp_path / "montaj.yaml"
    path.write_text(
        "video:\n  size: 108x192\n  fps: 0\n  look: warm-film\nassets: assets\n"
        "shots:\n  - {clip: rec, hold: 30f, speed: 1}\n"
    )
    with pytest.raises(SpecError) as ei:
        load_spec(path)
    assert any("fps" in e for e in ei.value.errors)


def test_unused_clip_fields_keep_legacy_segment_hashes(tmp_path: Path) -> None:
    """Pinned from the pre-clip-fields release: camera/speed/freeze must not enter an old film's hash."""
    from PIL import Image

    assets = tmp_path / "assets"
    assets.mkdir()
    for stem in "ab":
        Image.new("RGB", (8, 8), (20, 40, 60)).save(assets / f"{stem}.jpg")
    path = tmp_path / "montaj.yaml"
    path.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\nassets: assets\nshots:\n"
        "  - {photo: a, hold: 10f, zoom: 1.2, focus: [4, 4]}\n"
        "  - {photo: b, hold: 12f, crop: [1, 1, 5, 5]}\n"
        "  - {blank: true, hold: 8f}\n"
    )
    tl = resolve(load_spec(path), tmp_path)
    got = [segment_hash(tl, s, {"a": "x", "b": "y"}, "e1") for s in tl.segments]
    assert got == [
        "ca1fb13b58722cdcfa8a76f7c60b77959bd77ae8d0a5d5cfead00d4e6ae95ec9",
        "db968c8a68cc62e55dc111961096a139e4844847fdea7ff453372b1b41fa4ba1",
        "5cb54da7acd1c9af1f192dfc2a83a8902fec03924d962cf464b6a43f3e76e99f",
    ]
