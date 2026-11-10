"""Built-in effects (#38): shake, blur, bloom, chroma, trails on shots and layers.

Contract: amount 0 is the identity; every effect is a pure function of the frame (no RNG, no frame
memory), so a segment's first frame matches the whole-film render; unset fields stay out of the hash.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from montaj.gpu_policy import GpuPolicy, GpuStrictError
from montaj.project import segment_hash
from montaj.render import effects as E
from montaj.render.canvas import Canvas
from montaj.render.frame import Renderer
from montaj.spec import Effect, SpecError, load_spec
from montaj.timeline import resolve

# A moving picture: a bright square sliding right on a gradient, so blur/bloom/chroma/trails all show.
MOVER = (
    "import torch\n"
    "def draw(ctx):\n"
    "    u, v = ctx.uv[..., 0], ctx.uv[..., 1]\n"
    "    x = -0.5 + ctx.t\n"
    "    sq = ((u - x).abs() < 0.2) & (v.abs() < 0.2)\n"
    "    base = torch.stack((u * 0.2 + 0.4, v * 0.2 + 0.4, torch.full_like(u, 0.3)))\n"
    "    return torch.where(sq[None], torch.ones_like(base), base)\n"
)
FILES = {"fx/mover.py": MOVER}


def _project(tmp: Path, shots: str, *, size: str = "108x192", fps: int = 10, extra: str = "") -> Path:
    (tmp / "assets").mkdir(exist_ok=True)
    for rel, body in FILES.items():
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_text(body)
    p = tmp / "montaj.yaml"
    p.write_text(f"video:\n  size: {size}\n  fps: {fps}\n  look: warm-film\n{extra}assets: assets\nshots:\n{shots}")
    return p


def _errors(tmp: Path, shots: str, extra: str = "") -> list[str]:
    p = _project(tmp, shots, extra=extra)
    with pytest.raises(SpecError) as ei:
        resolve(load_spec(p), tmp)
    return ei.value.errors


def _renderer(tmp: Path, shots: str, *, device: str = "cpu", policy=None, size: str = "108x192") -> Renderer:
    tl = resolve(load_spec(_project(tmp, shots, size=size)), tmp)
    return Renderer(tl, tmp, Canvas(tl.size[0], tl.size[1], device=device), policy=policy)


def _raw(r: Renderer, f: float) -> torch.Tensor:
    """The scene before the look and finish."""
    return r._eval(r.tl.plan(int(f)).scene, float(f))


def _shot(fx: str = "", hold: str = "10f") -> str:
    return f"  - {{custom: fx/mover.py, hold: {hold}{', effects: [' + fx + ']' if fx else ''}}}\n"


# --- spec ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fx", "needle"),
    [
        ("{type: wobble}", "shots[0].effects[0].type: unknown 'wobble'"),
        ("{amount: 2}", "need exactly one of custom or type"),
        ("{type: blur, custom: fx/mover.py}", "need exactly one of custom or type"),
        ("{custom: fx/mover.py, amount: 1}", "shots[0].effects[0].amount: is for a built-in type"),
        ("{type: blur, params: {a: 1}}", "params is for custom"),
        ("{type: blur, seed: 3}", "shots[0].effects[0].seed: not used by blur"),
        ("{type: bloom, frames: 3}", "not used by bloom"),
        ("{type: shake, amount: 500}", "must be 0 to 200, got 500"),
        ("{type: chroma, amount: -1}", "must be 0 to 4"),
        ("{type: blur, amount: []}", "empty keyframe list"),
        ("{type: blur, amount: [{t: 4f, v: 1}, {t: 2f, v: 3}]}", "keyframe times must increase"),
        ("{type: shake, dur: 4f}", "dur needs at"),
        ("{type: shake, at: 2f, dur: 0f}", "must be > 0"),
        ("{type: shake, at: boom}", "shots[0].effects[0].at"),
        ("{type: bloom, threshold: 1}", "threshold: must be 0 to below 1"),
        ("{type: bloom, radius: 0}", "radius: must be > 0"),
        ("{type: trails, frames: 40}", "frames: must be 1 to 16"),
        ("{type: trails, decay: 0}", "decay: must be above 0"),
    ],
)
def test_bad_effects_are_one_line_errors(tmp_path: Path, fx: str, needle: str) -> None:
    errs = _errors(tmp_path, _shot(fx))
    assert any(needle in e for e in errs), errs


def test_layer_effect_errors(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir(exist_ok=True)
    _sticker(tmp_path / "assets" / "tag.png")
    errs = _errors(tmp_path, "  - hold: 4f\n    layers:\n      - {image: tag, effects: [{custom: fx/mover.py}]}\n")
    assert any("layers[0].effects[0].custom: custom effects run on a shot's picture" in e for e in errs), errs
    errs = _errors(tmp_path, "  - hold: 4f\n    layers:\n      - {image: tag, effects: []}\n")
    assert any("layers[0].effects: empty list" in e for e in errs), errs


def test_effect_times_become_shot_clock_frames(tmp_path: Path) -> None:
    shots = (_shot(hold="10f") + "  - custom: fx/mover.py\n    hold: 20f\n    effects:\n"
             "      - {type: shake, at: 1.5s, dur: 0.5s}\n"
             "      - {type: blur, amount: [{t: 0f, v: 0}, {t: 0.4s, v: 8}]}\n"
             "      - {type: chroma, at: hit}\n")
    tl = resolve(load_spec(_project(tmp_path, shots, extra="markers: {hit: 2s}\n")), tmp_path)
    shake, blur, chroma = tl.shots[1].spec.effects
    assert (shake.at, shake.dur) == (5, 5)  # film 15f, shot starts at 10f
    assert [(k.t, k.v) for k in blur.amount] == [(0, 0), (4, 8)]
    assert chroma.at == 10 and chroma.dur is None


def test_unset_effect_fields_stay_out_of_the_hash(tmp_path: Path) -> None:
    assert Effect(type="blur").model_dump() == {"type": "blur"}
    assert Effect(custom="fx/a.py").model_dump() == {"custom": "fx/a.py"}

    def hashes(shots: str) -> list[str]:
        p = _project(tmp_path, shots)
        tl = resolve(load_spec(p), tmp_path)
        from montaj.pipeline import _asset_shas

        shas = _asset_shas(load_spec(p), tmp_path)
        return [segment_hash(tl, s, shas, "e1") for s in tl.segments]

    plain = hashes(_shot() + _shot())
    blurred = hashes(_shot() + _shot("{type: blur}"))
    assert plain[0] == blurred[0] and plain[1] != blurred[1]
    assert hashes(_shot() + _shot("{type: blur, amount: 9}"))[1] != blurred[1]


# --- pure functions ------------------------------------------------------------------------------


def test_level_keyframes_and_hit_envelope() -> None:
    fx = Effect.model_validate({"type": "blur", "amount": [{"t": 0, "v": 0}, {"t": 10, "v": 8}]})
    assert [E.level(fx, t) for t in (-3, 0, 5, 10, 20)] == [0.0, 0.0, 4.0, 8.0, 8.0]
    hit = Effect(type="shake", amount=10, at=4, dur=4)
    assert [E.level(hit, t) for t in (3, 4, 6, 7.5, 8)] == [0.0, 10.0, 5.0, 1.25, 0.0]
    assert E.level(Effect(type="chroma", at=0), E.HIT_FRAMES) == 0.0
    assert E.level(Effect(type="bloom"), 3) == E.BLOOM_AMOUNT


def test_shake_offset_is_a_pure_seeded_function_of_time() -> None:
    torch.manual_seed(0)
    a = [E.shake_offset(7, t, 30, 10.0) for t in (0, 0.25, 1, 13)]
    torch.manual_seed(123)
    np.random.seed(5)
    assert [E.shake_offset(7, t, 30, 10.0) for t in (0, 0.25, 1, 13)] == a
    assert E.shake_offset(8, 13, 30, 10.0) != a[3]
    assert all(abs(x) <= 10 and abs(y) <= 10 for x, y in a)
    # Pinned: the hash is plain integer math, so this never moves between machines.
    assert E.shake_offset(0, 0, 30, 1.0) == pytest.approx((0.61539, 0.11159), abs=1e-5)


# --- renderer (CPU canvas) -----------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["shake", "blur", "bloom", "chroma", "trails"])
def test_amount_zero_is_the_identity(tmp_path: Path, kind: str) -> None:
    plain = _renderer(tmp_path, _shot())
    off = _renderer(tmp_path, _shot(f"{{type: {kind}, amount: 0}}"))
    on = _renderer(tmp_path, _shot(f"{{type: {kind}}}"))
    for f in (0, 4.25, 7):
        assert torch.equal(_raw(plain, f), _raw(off, f))
    assert not torch.equal(_raw(plain, 5), _raw(on, 5))


def test_hit_is_off_outside_its_window(tmp_path: Path) -> None:
    plain = _renderer(tmp_path, _shot())
    hit = _renderer(tmp_path, _shot("{type: shake, at: 3f, dur: 2f, amount: 40}"))
    assert torch.equal(_raw(plain, 2), _raw(hit, 2)) and torch.equal(_raw(plain, 5), _raw(hit, 5))
    assert not torch.equal(_raw(plain, 3), _raw(hit, 3))


def test_pinned_effect_values(tmp_path: Path) -> None:
    plain = _raw(_renderer(tmp_path, _shot()), 5)
    sq = plain[0] > 0.99  # the white square
    blur = _raw(_renderer(tmp_path, _shot("{type: blur, amount: 40}")), 5)
    assert blur.max() < 1.0 and torch.allclose(blur.mean(), plain.mean(), atol=0.01)
    bloom = _raw(_renderer(tmp_path, _shot("{type: bloom, amount: 2, radius: 60}")), 5)
    assert (bloom >= plain - 1e-6).all() and bloom[:, ~sq].mean() > plain[:, ~sq].mean() + 0.01
    chroma = _raw(_renderer(tmp_path, _shot("{type: chroma, amount: 2}")), 5)
    assert torch.equal(chroma[1], plain[1]) and not torch.equal(chroma[0], plain[0])
    # Pinned values on this 108x192 film (k = 0.1): chroma moves red outward, blue inward.
    assert float((chroma - plain).abs().sum()) == pytest.approx(PIN_CHROMA, rel=1e-3)
    assert float((blur - plain).abs().sum()) == pytest.approx(PIN_BLUR, rel=1e-3)


PIN_CHROMA = 192.63593
PIN_BLUR = 500.41495


def test_shake_moves_the_picture_and_never_shows_an_edge(tmp_path: Path) -> None:
    plain = _raw(_renderer(tmp_path, _shot()), 5)
    shook = [_raw(_renderer(tmp_path, _shot("{type: shake, amount: 200, seed: 3}")), 5) for _ in range(2)]
    assert torch.equal(shook[0], shook[1])
    assert not torch.equal(shook[0], plain)
    assert shook[0].min() > 0.05  # no black border: the overscan covers the move


def test_trails_at_a_segment_start_match_the_whole_film(tmp_path: Path) -> None:
    shots = _shot() + _shot("{type: trails, frames: 3, decay: 0.8}") + _shot()
    whole = _renderer(tmp_path, shots)
    whole.begin_segment([0, 1, 2])
    seq = {f: whole.frame(f) for f in range(0, 20)}
    fresh = _renderer(tmp_path, shots)
    fresh.begin_segment([1])
    assert torch.equal(fresh.frame(10), seq[10]) and torch.equal(fresh.frame(14), seq[14])
    plain = _renderer(tmp_path, _shot() + _shot() + _shot())
    assert not torch.equal(plain.frame(14), seq[14])


def test_effects_run_in_list_order_with_custom_ones(tmp_path: Path) -> None:
    FILES["fx/inv.py"] = "def draw(ctx):\n    return 1 - ctx.inputs[0]\n"
    a = _raw(_renderer(tmp_path, _shot("{type: bloom, amount: 2}, {custom: fx/inv.py}")), 5)
    b = _raw(_renderer(tmp_path, _shot("{custom: fx/inv.py}, {type: bloom, amount: 2}")), 5)
    assert not torch.equal(a, b)
    inv = _raw(_renderer(tmp_path, _shot("{custom: fx/inv.py}")), 5)
    bloom = _raw(_renderer(tmp_path, _shot("{type: bloom, amount: 2}")), 5)
    assert torch.allclose(a, 1 - bloom, atol=1e-6) and (b >= inv - 1e-6).all()


def _sticker(path: Path) -> None:
    arr = np.zeros((20, 40, 4), np.uint8)
    arr[:, :20] = (250, 250, 250, 255)
    Image.fromarray(arr, "RGBA").save(path)


LAYERS = "  - hold: 10f\n    layers:\n      - {{image: tag, pos: [540, 960], scale: 10{fx}}}\n"


def test_layer_blur_spreads_past_its_box_and_shake_moves_it(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir(exist_ok=True)
    _sticker(tmp_path / "assets" / "tag.png")
    plain = _raw(_renderer(tmp_path, LAYERS.format(fx="")), 3)
    blur = _raw(_renderer(tmp_path, LAYERS.format(fx=", effects: [{type: blur, amount: 30}]")), 3)
    above = slice(0, 60)  # rows above the sticker (it spans y 86..106 at k = 0.1)
    assert torch.equal(plain[:, above], torch.full_like(plain[:, above], plain[0, 0, 0].item()))
    assert blur[:, 80:86].mean() > plain[:, 80:86].mean() + 0.01
    off = _raw(_renderer(tmp_path, LAYERS.format(fx=", effects: [{type: blur, amount: 0}, {type: shake, amount: 0}]")), 3)
    assert torch.equal(off, plain)
    shook = _raw(_renderer(tmp_path, LAYERS.format(fx=", effects: [{type: shake, amount: 150}]")), 3)
    assert not torch.equal(shook, plain)
    assert torch.allclose(shook.mean(), plain.mean(), atol=1e-3)  # moved, not resized


def test_layer_trails_draw_older_copies(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir(exist_ok=True)
    _sticker(tmp_path / "assets" / "tag.png")
    slide = ", in: {type: pop, dur: 8f}"
    plain = _raw(_renderer(tmp_path, LAYERS.format(fx=slide)), 4)
    trail = _raw(_renderer(tmp_path, LAYERS.format(fx=slide + ", effects: [{type: trails, frames: 3}]")), 4)
    assert trail.sum() > plain.sum()


def test_policy_sees_an_effect_seam(tmp_path: Path) -> None:
    pol = GpuPolicy(strict=False)
    r = _renderer(tmp_path, _shot("{type: blur}"), policy=pol)
    _raw(r, 3)
    assert pol.seams.n("effect", "cpu") == 1
    strict = _renderer(tmp_path, _shot("{type: blur}"), policy=GpuPolicy(strict=True))
    with pytest.raises(GpuStrictError, match="blur effect"):
        _raw(strict, 3)


# --- GPU -----------------------------------------------------------------------------------------

GPU_KINDS = ["shake", "blur", "bloom", "chroma", "trails"]


@pytest.mark.gpu
@pytest.mark.parametrize("kind", GPU_KINDS)
def test_gpu_identity_at_zero_and_matches_cpu_at_a_pinned_value(tmp_path: Path, kind: str) -> None:
    pinned = {"shake": 60, "blur": 20, "bloom": 1.5, "chroma": 1.5, "trails": 1}[kind]
    plain = _renderer(tmp_path, _shot(), device="cuda", size="360x640")
    off = _renderer(tmp_path, _shot(f"{{type: {kind}, amount: 0}}"), device="cuda", size="360x640")
    assert torch.equal(_raw(plain, 5), _raw(off, 5))
    gpu = _raw(_renderer(tmp_path, _shot(f"{{type: {kind}, amount: {pinned}}}"), device="cuda", size="360x640"), 5)
    cpu = _raw(_renderer(tmp_path, _shot(f"{{type: {kind}, amount: {pinned}}}"), size="360x640"), 5)
    assert gpu.is_cuda and not torch.equal(gpu, _raw(plain, 5))
    assert torch.allclose(gpu.cpu(), cpu, atol=2e-3)


@pytest.mark.gpu
def test_strict_effects_film_is_all_gpu_and_cached(tmp_path: Path) -> None:
    from montaj.pipeline import render

    (tmp_path / "assets").mkdir(exist_ok=True)
    _sticker(tmp_path / "assets" / "tag.png")
    shots = (_shot("{type: shake, at: 2f}, {type: bloom}, {type: chroma, amount: 0.5}", hold="8f")
             + _shot("{type: trails}, {type: blur, amount: [{t: 0f, v: 0}, {t: 7f, v: 12}]}", hold="8f")
             + LAYERS.format(fx=", effects: [{type: shake}, {type: bloom}, {type: chroma}, {type: trails}]")
             .replace("hold: 10f", "hold: 8f"))
    p = _project(tmp_path, shots, size="360x640")
    pol = GpuPolicy(strict=True)
    result = render(p, "final", pol)
    assert result.gpu == "all" and result.rendered == 3
    assert pol.seams.n("effect", "gpu") > 0 and pol.seams.n("effect", "cpu") == 0
    assert render(p, "final", GpuPolicy(strict=True)).cached == 3


def test_clip_layer_trails_at_a_segment_start_match_the_whole_film(tmp_path: Path) -> None:
    from tests.test_layers import _numbered_clip

    (tmp_path / "assets").mkdir(exist_ok=True)
    _numbered_clip(tmp_path / "assets" / "rec.mkv")
    shots = ("  - {clip: rec, hold: 6f}\n  - hold: 6f\n    layers:\n"
             "      - {clip: rec, clip_in: 1s, box: [100, 100, 900, 900], effects: [{type: trails, frames: 4}]}\n")
    whole = _renderer(tmp_path, shots)
    seq = {f: whole.frame(f) for f in range(12)}
    fresh = _renderer(tmp_path, shots)
    fresh.begin_segment([1])
    assert torch.equal(fresh.frame(6), seq[6]) and torch.equal(fresh.frame(8), seq[8])
