"""glitch / zoom / dip / push transitions and the fx rgb_split hit (CPU tensors, no GPU)."""
from pathlib import Path

import pytest
import torch
from PIL import Image

from montaj.project import segment_hash
from montaj.render import transitions as mt
from montaj.render.canvas import Canvas
from montaj.render.finish import LOOKS, finish
from montaj.spec import SpecError, load_spec
from montaj.timeline import Trans, resolve

W, H = 108, 192


@pytest.fixture(scope="module")
def cv():
    return Canvas(W, H, device="cpu")


@pytest.fixture(scope="module")
def ab():
    g = torch.Generator().manual_seed(1)
    return torch.rand(3, H, W, generator=g), torch.rand(3, H, W, generator=g)


def _call(name, cv, a, b, x):
    return {
        "glitch": lambda: mt.glitch(cv, a, b, x, 1.0, 3, 10),
        "zoom": lambda: mt.zoom(cv, a, b, x, 1.0),
        "dip": lambda: mt.dip(cv, a, b, x, "#ffffff"),
        "push": lambda: mt.push(cv, a, b, x, "x"),
    }[name]()


@pytest.mark.parametrize("name", ["glitch", "zoom", "dip", "push"])
def test_endpoints_and_shape(cv, ab, name):
    a, b = ab
    assert torch.allclose(_call(name, cv, a, b, 0.0), a, atol=1e-5)
    assert torch.allclose(_call(name, cv, a, b, 1.0), b, atol=1e-5)
    mid = _call(name, cv, a, b, 0.5)
    assert mid.shape == a.shape and mid.dtype == a.dtype


def test_push_matches_whip_geometry(cv, ab):
    a, b = ab
    # Same slide as whip, just a different ease: at x=0.5 both are half-way.
    assert torch.allclose(mt.push(cv, a, b, 0.5), mt.whip(cv, a, b, 0.5), atol=1e-5)


def test_dip_midpoint_is_colour(cv, ab):
    a, b = ab
    out = mt.dip(cv, a, b, 0.5, "#ff0000")
    assert torch.allclose(out[0], torch.ones(H, W)) and out[1:].abs().max() == 0


def test_glitch_deterministic_and_seeded(cv, ab):
    a, b = ab
    one = mt.glitch(cv, a, b, 0.5, 1.0, 3, 10)
    assert torch.equal(one, mt.glitch(cv, a, b, 0.5, 1.0, 3, 10))
    assert not torch.equal(one, mt.glitch(cv, a, b, 0.5, 1.0, 4, 10))  # seed
    assert not torch.equal(one, mt.glitch(cv, a, b, 0.5, 1.0, 3, 11))  # frame


def test_glitch_does_not_touch_global_rng(cv, ab):
    a, b = ab
    torch.manual_seed(5)
    want = torch.rand(2)
    torch.manual_seed(5)
    mt.glitch(cv, a, b, 0.5, 1.0, 3, 10)
    assert torch.equal(torch.rand(2), want)


def test_zoom_strength_changes_output(cv, ab):
    a, b = ab
    assert not torch.allclose(mt.zoom(cv, a, b, 0.3, 0.5), mt.zoom(cv, a, b, 0.3, 2.0))


def test_rgb_split_decays_and_keeps_green(cv, ab):
    img, _ = ab
    hit = mt.rgb_split(cv, img, 1.0, 0.0)
    assert hit.shape == img.shape and torch.equal(hit[1], img[1]) and not torch.equal(hit[0], img[0])
    assert torch.equal(mt.rgb_split(cv, img, 1.0, 0.999), img)


def test_finish_applies_split_only_inside_its_window(cv, ab):
    from montaj.timeline import FramePlan

    img, _ = ab
    look = {**LOOKS["warm-film"], "sharpen": 0.0, "scurve": 0.0, "lift": 0.0, "gain": 1.0, "vignette": 0.0}
    base = dict(scene=None, flashes=(), leaks=(), intro=None, outro=None)
    on = finish(cv, img, 20, look, FramePlan(f=20, splits=((20, 1.0, 4),), **base))
    off = finish(cv, img, 20, look, FramePlan(f=20, **base))
    assert not torch.equal(on, off)


# ---- spec / timeline ----------------------------------------------------------------


def _write(tmp: Path, body: str, fx: str = "") -> Path:
    assets = tmp / "assets"
    assets.mkdir(exist_ok=True)
    for s in ("a", "b", "c"):
        Image.new("RGB", (8, 8), (10, 20, 30)).save(assets / f"{s}.jpg", "JPEG")
    p = tmp / "montaj.yaml"
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 120\n  look: warm-film\nassets: assets\n"
        f"shots:\n  - {{photo: a, hold: 30f}}\n{body}{fx}"
    )
    return p


def _errors(tmp: Path, body: str, fx: str = "") -> list[str]:
    with pytest.raises(SpecError) as ei:
        load_spec(_write(tmp, body, fx))
    return ei.value.errors


@pytest.mark.parametrize(
    "inline,kind",
    [
        ("{type: glitch, dur: 6f, strength: 1.5, seed: 3}", "glitch"),
        ("{type: zoom, dur: 8f}", "zoom"),
        ("{type: dip, dur: 10f, color: '#FFFFFF'}", "dip"),
        ("{type: push, dur: 8f, axis: y}", "push"),
    ],
)
def test_window_starts_at_cut_and_carries_params(tmp_path, inline, kind):
    tl = resolve(load_spec(_write(tmp_path, f"  - {{photo: b, hold: 30f, in: {inline}}}\n")), tmp_path)
    assert not isinstance(tl.plan(29).scene, Trans)
    sc = tl.plan(30).scene
    assert isinstance(sc, Trans) and sc.kind == kind and sc.t0 == 30.0
    assert not isinstance(tl.plan(30 + int(sc.dur)).scene, Trans)
    if kind == "glitch":
        assert (sc.strength, sc.seed) == (1.5, 3)
    if kind == "dip":
        assert sc.color == "#ffffff"
    if kind == "push":
        assert sc.axis == "y"


@pytest.mark.parametrize(
    "inline,needle",
    [
        ("{type: glitch, dur: 6f, strength: 0}", "in.strength"),
        ("{type: glitch, dur: 6f, strength: 9}", "in.strength"),
        ("{type: zoom, dur: 6f, strength: true}", "strength"),
        ("{type: glitch, dur: 6f, seed: -1}", "in.seed"),
        ("{type: glitch, dur: 6f, seed: 1.5}", "seed"),
        ("{type: glitch, dur: 6f, seed: true}", "seed"),
        ("{type: zoom, dur: 6f, seed: 1}", "in.seed"),
        ("{type: dip, dur: 6f, color: red}", "in.color"),
        ("{type: glitch, dur: 6f, color: '#ffffff'}", "in.color"),
        ("{type: dip, dur: 6f, strength: 1}", "in.strength"),
        ("{type: push, dur: 6f, axis: z}", "in.axis"),
        ("{type: glitch}", "in.dur"),
        ("{type: glitch, dur: 6}", "in.dur"),
        ("{type: dip, dur: 40f}", "longer than hold"),
        ("{type: zoom, dur: 0f}", "in.dur"),
    ],
)
def test_bad_params_rejected(tmp_path, inline, needle):
    errs = _errors(tmp_path, f"  - {{photo: b, hold: 30f, in: {inline}}}\n")
    assert any(needle in e for e in errs), errs


def test_new_types_overlap_checked(tmp_path):
    errs = _errors(
        tmp_path,
        "  - {photo: b, hold: 30f, in: {type: glitch, dur: 20f}}\n"
        "  - {photo: c, hold: 30f, in: {type: whip, dur: 30f}}\n",
    )
    assert any("overlaps" in e for e in errs), errs


def test_first_shot_cannot_glitch(tmp_path):
    p = tmp_path / "montaj.yaml"
    (tmp_path / "assets").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "assets" / "a.jpg", "JPEG")
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\nassets: assets\n"
        "shots:\n  - {photo: a, hold: 30f, in: {type: glitch, dur: 6f}}\n"
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert any("previous shot" in e for e in ei.value.errors)


def _hashes(tl):
    return [segment_hash(tl, s, {"photo:a": "x", "photo:b": "y", "photo:c": "z"}, "v") for s in tl.segments]


def test_params_reach_segment_hash(tmp_path):
    def build(sub, inline):
        d = tmp_path / sub
        d.mkdir()
        return resolve(load_spec(_write(d, f"  - {{photo: b, hold: 30f, in: {inline}}}\n")), d)

    base = _hashes(build("a", "{type: glitch, dur: 6f, seed: 1}"))
    assert base == _hashes(build("a2", "{type: glitch, dur: 6f, seed: 1}"))
    assert base[1] != _hashes(build("b", "{type: glitch, dur: 6f, seed: 2}"))[1]
    assert base[1] != _hashes(build("c", "{type: glitch, dur: 6f, strength: 2}"))[1]
    dip_a = _hashes(build("d", "{type: dip, dur: 6f, color: '#ffffff'}"))
    assert dip_a[1] != _hashes(build("e", "{type: dip, dur: 6f}"))[1]
    assert base[0] == _hashes(build("f", "{type: dip, dur: 6f}"))[0]


def test_fx_rgb_split_plan_and_validation(tmp_path):
    tl = resolve(
        load_spec(_write(tmp_path, "  - {photo: b, hold: 60f}\n", "fx:\n  - {rgb_split: 0.8, at: 40f, dur: 4f}\n")),
        tmp_path,
    )
    assert tl.plan(39).splits == ()
    assert tl.plan(40).splits == ((40, 0.8, 4),)
    assert tl.plan(43).splits == ((40, 0.8, 4),)
    assert tl.plan(44).splits == ()
    d = tmp_path / "x"
    d.mkdir()
    default = resolve(load_spec(_write(d, "  - {photo: b, hold: 60f}\n", "fx:\n  - {rgb_split: 1, at: 10f}\n")), d)
    assert default.plan(15).splits == ((10, 1.0, 6),) and default.plan(16).splits == ()


@pytest.mark.parametrize(
    "fx,needle",
    [
        ("  - {rgb_split: 0, at: 5f}\n", "rgb_split"),
        ("  - {rgb_split: 9, at: 5f}\n", "rgb_split"),
        ("  - {rgb_split: 1, flash: 1, at: 5f}\n", "need leak"),
        ("  - {flash: 1, at: 5f, dur: 4f}\n", "fx[0].dur"),
        ("  - {rgb_split: 1, at: 5f, dur: 4}\n", "fx[0].dur"),
    ],
)
def test_fx_rgb_split_bad(tmp_path, fx, needle):
    errs = _errors(tmp_path, "  - {photo: b, hold: 60f}\n", "fx:\n" + fx)
    assert any(needle in e for e in errs), errs


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["glitch", "zoom", "dip", "push"])
def test_cuda_matches_cpu(cv, ab, name):
    """The same inputs give the same pixels on CUDA (glitch draws on CPU, so its tears match exactly)."""
    torch.cuda.is_available() or pytest.skip("needs CUDA")
    gcv = Canvas(W, H, device="cuda")
    a, b = ab
    want = _call(name, cv, a, b, 0.5)
    got = _call(name, gcv, a.cuda(), b.cuda(), 0.5).cpu()
    assert (got - want).abs().max() < 2 / 255
