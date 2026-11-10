"""Layers (#34): several sources per shot with placement, transform, opacity and blend."""
from __future__ import annotations

import io
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from montaj.project import segment_hash
from montaj.render.canvas import Canvas
from montaj.render.frame import Renderer
from montaj.render.layers import LayerState, LayerTexture, blend, composite, decode_png, layer_state
from montaj.spec import SpecError, layer_warnings, load_spec
from montaj.timeline import Layers, resolve


def _ffmpeg(*args: str) -> None:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-400:]


def _numbered_clip(path: Path) -> None:
    """320x240 @ 30, 3 s. Red is the frame number, green the x gradient, blue the y gradient."""
    _ffmpeg("-f", "lavfi", "-i", "color=size=320x240:rate=30:duration=3,format=gbrp,"
            "geq=r='N':g='255*X/W':b='255*Y/H'", "-c:v", "ffv1", "-pix_fmt", "rgb24", str(path))


def _sticker(path: Path) -> None:
    """40x20 RGBA: left half opaque (200, 100, 50), right half transparent."""
    arr = np.zeros((20, 40, 4), np.uint8)
    arr[:, :20] = (200, 100, 50, 255)
    Image.fromarray(arr, "RGBA").save(path)


def _project(tmp: Path, shots: str, *, size: str = "108x192") -> Path:
    assets = tmp / "assets"
    assets.mkdir(exist_ok=True)
    if not (assets / "rec.mkv").exists():
        _numbered_clip(assets / "rec.mkv")
        _sticker(assets / "tag.png")
    path = tmp / "montaj.yaml"
    path.write_text(f"video:\n  size: {size}\n  fps: 30\n  look: warm-film\nassets: assets\nshots:\n{shots}\n")
    return path


def _errors(tmp: Path, shots: str) -> list[str]:
    with pytest.raises(SpecError) as ei:
        load_spec(_project(tmp, shots))
    return ei.value.errors


def _renderer(tmp: Path, shots: str) -> tuple[Renderer, object]:
    tl = resolve(load_spec(_project(tmp, shots)), tmp)
    return Renderer(tl, tmp, Canvas(tl.size[0], tl.size[1], device="cpu")), tl


# ---------------------------------------------------------------- blend and state


def test_blend_modes_match_the_w3c_formulas() -> None:
    cb = torch.tensor([0.2, 0.6, 0.9])[:, None, None]
    cs = torch.tensor([0.5, 0.3, 0.1])[:, None, None]
    one = torch.ones(1, 1, 1)
    want = {
        "normal": cs,
        "screen": cb + cs - cb * cs,
        "multiply": cb * cs,
        "difference": (cb - cs).abs(),
        "add": (cb + cs).clamp(max=1),
        "overlay": torch.tensor([2 * 0.2 * 0.5, 1 - 2 * 0.4 * 0.7, 1 - 2 * 0.1 * 0.9])[:, None, None],
    }
    for mode, expect in want.items():
        assert torch.allclose(blend(cb, cs, one, mode), expect, atol=1e-6), mode
    # Soft light: cs <= .5 darkens by cb(1 - cb)(1 - 2cs); above it lightens towards D(cb).
    soft = blend(cb, cs, one, "soft-light")
    assert float(soft[0]) == pytest.approx(0.2)  # cs = .5 is neutral
    assert float(soft[2]) == pytest.approx(0.9 - 0.8 * 0.9 * 0.1)
    # Half alpha is a straight mix of backdrop and blend result (premultiplied source).
    half = blend(cb, cs * 0.5, one * 0.5, "multiply")
    assert torch.allclose(half, 0.5 * cb + 0.5 * cb * cs, atol=1e-6)
    # Zero alpha leaves the backdrop exactly.
    assert torch.equal(blend(cb, cs * 0, one * 0, "screen"), cb)


def _layer(**kw):
    from montaj.spec import Layer, LayerAnim

    for name in ("in_", "out"):
        if name in kw:
            kw[name] = LayerAnim(**kw[name])
    return Layer(**kw)


def test_layer_state_window_and_anims() -> None:
    plain = _layer(clip="a")
    assert layer_state(plain, -0.5, 30) == LayerState(1.0, 0.0, 1.0)  # unset from: motion-blur samples see it
    assert layer_state(plain, 45.0, 30) == LayerState(1.0, 0.0, 1.0)  # unset to: the outgoing transition too
    timed = _layer(image="a", from_=10, to=20, opacity=0.5)
    assert layer_state(timed, 9.9, 30) is None and layer_state(timed, 20.0, 30) is None
    assert layer_state(timed, 10.0, 30).opacity == 0.5
    fade = _layer(image="a", from_=10, in_={"type": "fade", "dur": 4}, out={"type": "fade", "dur": 4})
    assert layer_state(fade, 10.0, 30) is None  # opacity 0 is off
    assert layer_state(fade, 12.0, 30).opacity == pytest.approx(0.5)
    assert layer_state(fade, 20.0, 30).opacity == 1.0
    assert layer_state(fade, 28.0, 30).opacity == pytest.approx(0.5)  # out ends at the hold
    pop = _layer(image="a", in_={"type": "pop", "dur": 10})
    scales = [layer_state(pop, float(t), 30).scale for t in range(1, 11)]
    assert scales[0] < 0.75 and max(scales) > 1.0 and scales[-1] == pytest.approx(1.0)


def test_composite_box_radius_and_opacity() -> None:
    cv = Canvas(100, 100, device="cpu")
    dst = torch.zeros(3, 100, 100)
    tex = LayerTexture(torch.ones(3, 10, 10))
    composite(cv, dst, tex, (0, 0, 10, 10), (20.0, 30.0, 60.0, 50.0), LayerState(1.0, 0.0, 0.5))
    assert float(dst[0, 40, 40]) == pytest.approx(0.5) and float(dst[0, 25, 40]) == 0.0
    assert float(dst[0, 30, 20]) == pytest.approx(0.5) and float(dst[0, 29, 20]) == 0.0  # 1 px exact edge
    rounded = torch.zeros(3, 100, 100)
    composite(cv, rounded, tex, (0, 0, 10, 10), (20.0, 30.0, 60.0, 50.0), LayerState(1.0, 0.0, 1.0), radius=8)
    assert float(rounded[0, 30, 20]) == 0.0 and float(rounded[0, 40, 40]) == 1.0
    # Rotation by 90 degrees about the centre swaps the box's extent.
    turned = torch.zeros(3, 100, 100)
    composite(cv, turned, tex, (0, 0, 10, 10), (20.0, 30.0, 60.0, 50.0), LayerState(1.0, 90.0, 1.0))
    assert float(turned[0, 22, 40]) == 1.0 and float(turned[0, 40, 22]) == 0.0
    # The shadow darkens below the box, not above it.
    bg = torch.full((3, 100, 100), 0.8)
    composite(cv, bg, tex, (0, 0, 10, 10), (20.0, 30.0, 60.0, 50.0), LayerState(1.0, 0.0, 1.0), shadow=0.6)
    assert float(bg[0, 52, 40]) < 0.7 and float(bg[0, 28, 40]) > float(bg[0, 52, 40])


# ---------------------------------------------------------------- PNG on the device


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def _png(arr: np.ndarray, ctype: int, depth: int = 8, plte: bytes | None = None, trns: bytes | None = None) -> bytes:
    """Encode with every row filter in turn (row y uses filter y % 5), so each predictor is exercised."""
    h, w = arr.shape[:2]
    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    raw = (arr.astype(">u2").view(np.uint8) if depth == 16 else arr.astype(np.uint8)).reshape(h, -1)
    bpp = ch * depth // 8
    out = bytearray()
    prev = np.zeros(raw.shape[1], np.int32)
    for y in range(h):
        row = raw[y].astype(np.int32)
        left = np.concatenate([np.zeros(bpp, np.int32), row[:-bpp]])
        ul = np.concatenate([np.zeros(bpp, np.int32), prev[:-bpp]])
        ft = y % 5
        if ft == 0:
            f = row
        elif ft == 1:
            f = row - left
        elif ft == 2:
            f = row - prev
        elif ft == 3:
            f = row - ((left + prev) >> 1)
        else:
            p = left + prev - ul
            pa, pb, pc = abs(p - left), abs(p - prev), abs(p - ul)
            f = row - np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, ul))
        out.append(ft)
        out += (f & 255).astype(np.uint8).tobytes()
        prev = row
    data = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 0))
    if plte is not None:
        data += _chunk(b"PLTE", plte)
    if trns is not None:
        data += _chunk(b"tRNS", trns)
    return data + _chunk(b"IDAT", zlib.compress(bytes(out))) + _chunk(b"IEND", b"")


def _pil_premul(data: bytes) -> np.ndarray:
    a = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"), np.float32) / 255
    return np.concatenate([a[..., :3] * a[..., 3:], a[..., 3:]], -1)


_rng = np.random.default_rng(7)
_PNGS = {
    "rgba": lambda: _png(_rng.integers(0, 256, (23, 31, 4)), 6),
    "rgb": lambda: _png(_rng.integers(0, 256, (17, 29, 3)), 2),
    "gray": lambda: _png(_rng.integers(0, 256, (13, 11)), 0),
    "gray-alpha": lambda: _png(_rng.integers(0, 256, (13, 11, 2)), 4),
    "palette-trns": lambda: _png(_rng.integers(0, 7, (19, 21)), 3, plte=bytes(range(0, 252, 12)),
                                 trns=bytes([0, 128, 255])),
    "rgb-key": lambda: _png(np.where(_rng.random((9, 8, 1)) < 0.3, [10, 20, 30], _rng.integers(0, 256, (9, 8, 3))),
                            2, trns=struct.pack(">3H", 10, 20, 30)),
}


@pytest.mark.parametrize("name", sorted(_PNGS))
def test_device_png_decoder_matches_pil(name: str) -> None:
    data = _PNGS[name]()
    got = decode_png(data, "cpu").permute(1, 2, 0).numpy()
    assert np.array_equal(got, _pil_premul(data))


def test_device_png_decoder_reads_16_bit() -> None:
    arr = _rng.integers(0, 65536, (7, 9, 4))
    got = decode_png(_png(arr, 6, depth=16), "cpu").permute(1, 2, 0).numpy()
    a = arr / 65535.0
    want = np.concatenate([a[..., :3] * a[..., 3:], a[..., 3:]], -1)
    assert np.allclose(got, want, atol=1e-6)


# ---------------------------------------------------------------- spec


@pytest.mark.parametrize(
    ("shots", "needle"),
    [
        ("  - {photo: x, hold: 1s, layers: [{clip: rec}]}\n", "layers replace the shot's single source"),
        ("  - {hold: 1s, layers: []}\n", "need at least one layer"),
        ("  - {hold: 1s, layers: [{clip: rec, image: tag}]}\n", "need exactly one of clip or image"),
        ("  - {hold: 1s, layers: [{image: tag, clip_in: 1s}]}\n", "clip_in needs clip"),
        ("  - {hold: 1s, layers: [{image: tag, box: [0, 0, 10, 10], pos: [5, 5]}]}\n", "mutually exclusive"),
        ("  - {hold: 1s, layers: [{image: tag, blend: lighten}]}\n", "use normal or screen"),
        ("  - {hold: 1s, layers: [{image: tag, opacity: 1.5}]}\n", "must be 0 to 1"),
        ("  - {hold: 1s, layers: [{image: tag, box: [10, 0, 5, 10]}]}\n", "is not a box"),
        ("  - {hold: 1s, layers: [{image: tag, from: 10f, to: 5f}]}\n", "is not after from"),
        ("  - {hold: 1s, layers: [{image: tag, to: 2s}]}\n", "after the end of the 30f hold"),
        ("  - {hold: 1s, layers: [{image: tag, in: {type: spin, dur: 4f}}]}\n", "use fade or pop"),
        ("  - {hold: 1s, focus: [1, 1], layers: [{clip: rec}]}\n", "set it on the layer"),
        ("  - {hold: 1s, layers: [{image: tag, box: [1200, 0, 1300, 100]}]}\n", "entirely off-frame"),
        ("  - {hold: 1s, layers: [{image: nope}]}\n", "nope"),
        ("  - {hold: 1s, layers: [{clip: rec, clip_in: 2.5s}]}\n", "clip ends at 3.0s of source"),
        ("  - {hold: 1s, layers: [{clip: rec, focus: [900, 10], box: [0, 0, 500, 500]}]}\n", "outside the 320x240"),
    ],
)
def test_layer_errors_are_helpful(tmp_path: Path, shots: str, needle: str) -> None:
    errs = _errors(tmp_path, shots)
    assert any(needle in e for e in errs), errs


def test_partly_off_frame_is_a_warning(tmp_path: Path) -> None:
    shots = ("  - hold: 1s\n    layers:\n      - {clip: rec}\n"
             "      - {image: tag, pos: [1070, 100]}\n      - {image: tag, pos: [500, 100], rotate: 30}\n")
    spec = load_spec(_project(tmp_path, shots))
    warns = layer_warnings(spec, tmp_path)
    assert len(warns) == 1 and warns[0].startswith("WARN shots[0].layers[1]: partly off-frame")


def test_layer_times_become_frames(tmp_path: Path) -> None:
    shots = ("  - hold: 1s\n    layers:\n      - {clip: rec}\n"
             "      - {image: tag, from: 0.25s, to: 0.75s, in: {type: pop, dur: 6f}, out: {type: fade, dur: 0.1s}}\n")
    tl = resolve(load_spec(_project(tmp_path, shots)), tmp_path)
    layer = tl.shots[0].spec.layers[1]
    assert (layer.from_, layer.to, layer.in_.dur, layer.out.dur) == (8, 23, 6, 3)
    assert isinstance(tl.plan(0).scene, Layers)


def test_bare_number_times_are_rejected(tmp_path: Path) -> None:
    errs = _errors(tmp_path, "  - {hold: 1s, layers: [{image: tag, from: 4}]}\n")
    assert any("layers[0].from" in e for e in errs), errs


# ---------------------------------------------------------------- hash


def test_unset_layer_fields_stay_out_of_the_hash(tmp_path: Path) -> None:
    from montaj.project import _shot_payload

    tl = resolve(load_spec(_project(tmp_path, "  - {hold: 1s, layers: [{clip: rec}, {image: tag, opacity: 0.5}]}\n")),
                 tmp_path)
    payload = _shot_payload(tl.shots[0].spec)
    assert payload["layers"] == [{"clip": "rec"}, {"image": "tag", "opacity": 0.5}]
    shas = {"clip:rec": "c1", "photo:tag": "p1"}
    base = segment_hash(tl, tl.segments[0], shas, "e1")
    # The sticker bytes and the clip bytes both decide the hash.
    assert segment_hash(tl, tl.segments[0], {**shas, "photo:tag": "p2"}, "e1") != base
    assert segment_hash(tl, tl.segments[0], {**shas, "clip:rec": "c2"}, "e1") != base
    other = resolve(load_spec(_project(tmp_path, "  - {hold: 1s, layers: [{clip: rec}, {image: tag, opacity: 0.6}]}\n")),
                    tmp_path)
    assert segment_hash(other, other.segments[0], shas, "e1") != base


def test_pipeline_hashes_layer_assets(tmp_path: Path) -> None:
    from montaj.pipeline import _asset_shas

    spec = load_spec(_project(tmp_path, "  - {hold: 1s, layers: [{clip: rec}, {image: tag}]}\n"))
    assert set(_asset_shas(spec, tmp_path)) == {"clip:rec", "photo:tag"}


# ---------------------------------------------------------------- render (CPU)


def test_one_full_frame_clip_layer_is_the_clip_shot(tmp_path: Path) -> None:
    clip_r, clip_tl = _renderer(tmp_path, "  - {clip: rec, hold: 20f}\n")
    lay_r, lay_tl = _renderer(tmp_path, "  - {hold: 20f, layers: [{clip: rec}]}\n")
    for f in (0, 7, 19):
        assert torch.equal(clip_r.frame(f), lay_r.frame(f)), f
    assert lay_r._layer_plans(0)[0].clip == clip_r._clip_plan(0)


def test_inset_and_sticker_land_where_placed(tmp_path: Path) -> None:
    shots = """  - hold: 20f
    layers:
      - {clip: rec}
      - {clip: rec, clip_in: 10f, box: [100, 200, 500, 500], fit: contain}
      - {image: tag, pos: [700, 1500], scale: 4, blend: screen}
"""
    rend, tl = _renderer(tmp_path, shots)
    k = rend.cv.k  # 0.1 at 108 wide
    img = rend._layers_frame(tl.plan(0).scene, 0.0) * 255
    # Inset (contain: 4:3 source in a 400x300 box) shows source frame 10 at its own centre.
    cy, cx = round(350 * k), round(300 * k)
    assert float(img[0, cy, cx]) == pytest.approx(10, abs=1.5)
    assert float(img[1, cy, cx]) == pytest.approx(127, abs=12)
    # Outside the inset the base clip shows frame 0.
    assert float(img[0, round(800 * k), round(900 * k)]) == pytest.approx(0, abs=1.5)
    # Sticker: 40x20 scaled 4x = 160x80 design px centred at (700, 1500); its opaque left half screens.
    sx, sy = round((700 - 40) * k), round(1500 * k)
    base = rend._layer_clip(rend._layer_plans(0)[0], 0.0)[:, sy, sx] * 255
    cs = torch.tensor([200.0, 100.0, 50.0])
    want = base + cs - base * cs / 255
    assert torch.allclose(img[:, sy, sx], want, atol=1.0)
    # Its transparent right half leaves the base untouched.
    tx = round((700 + 40) * k)
    assert torch.allclose(img[:, sy, tx], rend._layer_clip(rend._layer_plans(0)[0], 0.0)[:, sy, tx] * 255, atol=1e-4)


def test_layer_clip_windows_stay_open_across_a_segment(tmp_path: Path) -> None:
    shots = "  - hold: 10f\n    layers:\n      - {clip: rec}\n      - {clip: rec, clip_in: 1s, box: [0, 0, 540, 540]}\n"
    rend, tl = _renderer(tmp_path, shots)
    rend.begin_segment([0])
    for f in range(3):
        rend.frame(f)
    windows = rend._clips.get("rec")._cache
    assert len(windows) == 2  # one reader per layer, both kept


# ---------------------------------------------------------------- GPU


def _gpu_film(tmp: Path) -> Path:
    assets = tmp / "assets"
    assets.mkdir(exist_ok=True)
    for name, src in (("talk", "testsrc2"), ("broll", "mandelbrot")):
        _ffmpeg("-f", "lavfi", "-i", f"{src}=size=320x240:rate=30", "-t", "1", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(assets / f"{name}.mp4"))
    _sticker(assets / "tag.png")
    (tmp / "montaj.yaml").write_text(
        "video: {size: 360x640, fps: 30, look: warm-film}\nassets: assets\nshots:\n"
        "  - hold: 20f\n    layers:\n      - {clip: talk}\n"
        "      - {clip: broll, box: [80, 900, 1000, 1600], radius: 32, shadow: 0.5, in: {type: pop, dur: 6f}}\n"
        "      - {image: tag, pos: [540, 400], scale: 6, rotate: -8, blend: screen, opacity: 0.9}\n"
    )
    return tmp / "montaj.yaml"


@pytest.mark.gpu
def test_strict_layer_film_is_all_gpu(tmp_path: Path) -> None:
    from montaj.gpu_policy import GpuPolicy
    from montaj.pipeline import render

    pol = GpuPolicy(strict=True)
    result = render(_gpu_film(tmp_path), "final", pol)
    assert result.gpu == "all" and result.rendered > 0
    assert pol.seams.n("photo", "gpu") > 0 and pol.seams.n("decode", "gpu") > 0


@pytest.mark.gpu
def test_gpu_one_layer_matches_the_clip_shot(tmp_path: Path) -> None:
    _gpu_film(tmp_path)
    frames = []
    for shot in ("  - {clip: talk, hold: 12f}\n", "  - {hold: 12f, layers: [{clip: talk}]}\n"):
        (tmp_path / "montaj.yaml").write_text(
            "video: {size: 360x640, fps: 30, look: warm-film}\nassets: assets\nshots:\n" + shot)
        tl = resolve(load_spec(tmp_path / "montaj.yaml"), tmp_path)
        rend = Renderer(tl, tmp_path, Canvas(tl.size[0], tl.size[1], device="cuda"))
        frames.append([rend.frame(f) for f in (0, 5, 11)])
    for a, b in zip(*frames):
        assert torch.equal(a, b)
