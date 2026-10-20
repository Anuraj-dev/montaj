"""Looks `clean`, `cool-tech` and `look: {lut: …}`: grade maths, .cube parsing, spec validation, cache hash."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from montaj.cube import MAX_3D, parse_cube
from montaj.pipeline import _asset_shas
from montaj.project import file_sha, segment_hash
from montaj.render.canvas import Canvas
from montaj.render.finish import LOOKS, finish, lut_texture
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve


def _cube_text(n: int, fn=lambda r, g, b: (r, g, b), header: str = "") -> str:
    rows = [f"{x:.6f} {y:.6f} {z:.6f}"
            for b in range(n) for g in range(n) for r in range(n)
            for x, y, z in [fn(r / (n - 1), g / (n - 1), b / (n - 1))]]
    return f"TITLE \"t\"\n# comment\nLUT_3D_SIZE {n}\n{header}" + "\n".join(rows) + "\n"


def _plan(flashes=(), fade_out=None):
    return SimpleNamespace(flashes=tuple(flashes), dust=(), fade_out=fade_out, leaks=(), intro=None, outro=None)


def _gradient(h=48, w=64) -> torch.Tensor:
    ramp = torch.linspace(0, 1, w)[None, :].expand(h, w)
    return torch.stack([ramp, ramp.flip(1), ramp * 0.5 + 0.25])


def _cv(w=64, h=48) -> Canvas:
    return Canvas(w, h, device="cpu")


def _lit(text: str):
    return lut_texture(parse_cube(text), "cpu")


# --- grades -----------------------------------------------------------------

def test_clean_is_near_identity_and_neutral() -> None:
    cv = _cv()
    img = _gradient()
    out = finish(cv, img, 3, LOOKS["clean"], _plan())
    mse = ((out - img) ** 2).mean().item()
    psnr = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()
    assert psnr > 28, psnr
    black = finish(cv, torch.zeros(3, 48, 64), 3, LOOKS["clean"], _plan())
    assert black.max().item() < 0.02
    grey = finish(cv, torch.full((3, 48, 64), 0.5), 3, LOOKS["clean"], _plan())
    centre = grey[:, 24, 32]
    assert (centre.max() - centre.min()).item() < 1e-3  # no colour cast


def test_cool_tech_teal_shadows_cool_whites() -> None:
    cv = _cv()
    dark = finish(cv, torch.zeros(3, 48, 64), 3, LOOKS["cool-tech"], _plan())[:, 24, 32]
    assert dark[2] > dark[0] and dark[1] > dark[0]
    assert dark.max().item() < 0.03
    white = finish(cv, torch.full((3, 48, 64), 0.9), 3, LOOKS["cool-tech"], _plan())[:, 24, 32]
    assert white[2] > white[1] > white[0]


def test_graded_looks_honour_flash_and_fade_ignore_leaks() -> None:
    cv = _cv()
    img = torch.full((3, 48, 64), 0.2)
    base = finish(cv, img, 10, LOOKS["clean"], _plan())
    flash = finish(cv, img, 10, LOOKS["clean"], _plan(flashes=[(10, 1.0)]))
    assert flash.min().item() > base.max().item()
    leaky = _plan()
    leaky.leaks = ((10, 1.0, 1),)
    assert torch.equal(finish(cv, img, 10, LOOKS["clean"], leaky), base)
    faded = finish(cv, img, 20, LOOKS["clean"], _plan(fade_out=(10, 10)))
    assert faded.max().item() < 1e-6  # fully faded at the end of the span


def test_existing_looks_unchanged_shape() -> None:
    """warm-film and golden-film keep their dicts: no `kind` on warm, golden dispatches by kind."""
    assert "kind" not in LOOKS["warm-film"]
    assert LOOKS["golden-film"] == {"kind": "golden-film"}


# --- LUT --------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 17])
def test_identity_lut_reproduces_input(n: int) -> None:
    cv = _cv()
    img = torch.rand(3, 48, 64)
    look = {"kind": "graded", "lut": _lit(_cube_text(n))}
    out = finish(cv, img, 3, look, _plan())
    assert (out - img).abs().max().item() < 1 / 255


def test_lut_strength_lerps_and_maps_channels() -> None:
    cv = _cv()
    img = torch.full((3, 48, 64), 0.25)
    swap = _lit(_cube_text(9, lambda r, g, b: (b, g, r)))  # swaps red and blue
    img[0] = 0.8
    full = finish(cv, img, 3, {"kind": "graded", "lut": swap}, _plan())
    assert torch.allclose(full[2], torch.full((48, 64), 0.8), atol=1e-4)
    assert torch.allclose(full[0], torch.full((48, 64), 0.25), atol=1e-4)
    half = finish(cv, img, 3, {"kind": "graded", "lut": swap, "strength": 0.5}, _plan())
    assert torch.allclose(half[0], torch.full((48, 64), 0.525), atol=1e-4)
    none = finish(cv, img, 3, {"kind": "graded", "lut": swap, "strength": 0.0}, _plan())
    assert torch.allclose(none, img, atol=1e-6)


def test_1d_lut_and_domain() -> None:
    cv = _cv()
    invert = parse_cube("LUT_1D_SIZE 2\n1 1 1\n0 0 0\n")
    img = torch.full((3, 48, 64), 0.25)
    out = finish(cv, img, 3, {"kind": "graded", "lut": lut_texture(invert, "cpu")}, _plan())
    assert torch.allclose(out, torch.full_like(img, 0.75), atol=1e-5)
    # DOMAIN 0..2: input 0.5 reads the middle of the table.
    dom = parse_cube("LUT_1D_SIZE 3\nDOMAIN_MAX 2 2 2\n0 0 0\n0.2 0.2 0.2\n1 1 1\n")
    out = finish(cv, torch.full((3, 48, 64), 1.0), 3, {"kind": "graded", "lut": lut_texture(dom, "cpu")}, _plan())
    assert torch.allclose(out, torch.full_like(img, 0.2), atol=1e-5)


def test_lut_look_vignette_and_grain_options() -> None:
    cv = _cv()
    img = torch.full((3, 48, 64), 0.5)
    lut = _lit(_cube_text(2))
    plain = finish(cv, img, 3, {"kind": "graded", "lut": lut}, _plan())
    vig = finish(cv, img, 3, {"kind": "graded", "lut": lut, "vignette": 0.3}, _plan())
    assert vig[:, 0, 0].max() < plain[:, 0, 0].min()
    assert torch.allclose(vig[:, 24, 32], plain[:, 24, 32], atol=0.02)
    grained = finish(cv, img, 3, {"kind": "graded", "lut": lut, "grain": 1.0}, _plan())
    assert not torch.allclose(grained, plain)


# --- .cube parsing ----------------------------------------------------------

def test_cube_parse_errors() -> None:
    for text, needle in [
        ("", "exactly one"),
        ("LUT_3D_SIZE 2\n0 0 0\n", "needs 8 rows"),
        (f"LUT_3D_SIZE {MAX_3D + 1}\n", "outside 2.."),
        ("LUT_3D_SIZE 1\n0 0 0\n", "outside 2.."),
        ("LUT_1D_SIZE 2\nLUT_3D_SIZE 2\n", "exactly one"),
        ("LUT_3D_SIZE 2\nBOGUS 1\n", "unsupported keyword"),
        ("LUT_1D_SIZE 2\n0 0\n1 1 1\n", "three finite"),
        ("LUT_1D_SIZE 2\n0 0 nan\n1 1 1\n", "three finite"),
        ("LUT_1D_SIZE 2\nDOMAIN_MAX 0 0 0\n0 0 0\n1 1 1\n", "exceed"),
    ]:
        with pytest.raises(ValueError, match=needle):
            parse_cube(text)


# --- spec, validation and hash ---------------------------------------------

def _spec(tmp: Path, look: str) -> Path:
    assets = tmp / "assets"
    assets.mkdir(exist_ok=True)
    Image.new("RGB", (8, 8), (20, 40, 60)).save(assets / "a.jpg", "JPEG")
    p = tmp / "montaj.yaml"
    p.write_text(f"video:\n  size: 1080x1920\n  fps: 30\n  look: {look}\nassets: assets\nshots:\n  - {{photo: a, hold: 10f}}\n")
    return p


@pytest.mark.parametrize("name", ["clean", "cool-tech"])
def test_named_looks_validate(tmp_path: Path, name: str) -> None:
    assert load_spec(_spec(tmp_path, name)).video.look == name


def test_unknown_look_lists_choices(tmp_path: Path) -> None:
    with pytest.raises(SpecError) as ei:
        load_spec(_spec(tmp_path, "sepia"))
    msg = "\n".join(ei.value.errors)
    assert "video.look" in msg and "clean" in msg and "cool-tech" in msg and "lut" in msg


def test_lut_look_validates_and_rejects_bad(tmp_path: Path) -> None:
    (tmp_path / "looks").mkdir()
    (tmp_path / "looks" / "brand.cube").write_text(_cube_text(2))
    ok = load_spec(_spec(tmp_path, "{lut: looks/brand.cube, strength: 0.8, vignette: 0.1, grain: 0}"))
    assert ok.video.look.lut == "looks/brand.cube" and ok.video.look.strength == 0.8

    def bad(look: str, needle: str) -> None:
        with pytest.raises(SpecError) as ei:
            load_spec(_spec(tmp_path, look))
        assert any(needle in e and e.endswith(")") for e in ei.value.errors), ei.value.errors

    bad("{lut: looks/missing.cube}", "not found")
    bad("{lut: looks/brand.txt}", "not a .cube")
    bad("{lut: looks/brand.cube, strength: 1.5}", "video.look.strength")
    bad("{lut: looks/brand.cube, grain: -1}", "video.look.grain")
    bad("{lut: looks/brand.cube, sharpen: 1}", "sharpen")
    bad("{strength: 1}", "lut")
    (tmp_path / "looks" / "big.cube").write_text(f"LUT_3D_SIZE {MAX_3D + 1}\n")
    bad("{lut: looks/big.cube}", "outside 2..")
    (tmp_path / "looks" / "short.cube").write_text("LUT_3D_SIZE 2\n0 0 0\n")
    bad("{lut: looks/short.cube}", "needs 8 rows")


def test_lut_bytes_reach_segment_hash(tmp_path: Path) -> None:
    (tmp_path / "looks").mkdir()
    cube = tmp_path / "looks" / "brand.cube"
    cube.write_text(_cube_text(2))
    p = _spec(tmp_path, "{lut: looks/brand.cube}")
    spec = load_spec(p)
    tl = resolve(spec, tmp_path)
    shas = _asset_shas(spec, tmp_path)
    assert shas["lut:looks/brand.cube"] == file_sha(cube)
    h0 = segment_hash(tl, tl.segments[0], shas, "e1")
    assert h0 == segment_hash(tl, tl.segments[0], _asset_shas(spec, tmp_path), "e1")
    cube.write_text(_cube_text(2, lambda r, g, b: (b, g, r)))  # same path, new bytes
    h1 = segment_hash(tl, tl.segments[0], _asset_shas(spec, tmp_path), "e1")
    assert h1 != h0
    # A plain look's hash does not need a lut entry.
    plain = load_spec(_spec(tmp_path, "clean"))
    tlp = resolve(plain, tmp_path)
    segment_hash(tlp, tlp.segments[0], _asset_shas(plain, tmp_path), "e1")


def test_renderer_loads_lut_once_and_renders(tmp_path: Path) -> None:
    from montaj.render.frame import Renderer

    (tmp_path / "looks").mkdir()
    (tmp_path / "looks" / "b.cube").write_text(_cube_text(2))
    spec = load_spec(_spec(tmp_path, "{lut: looks/b.cube, strength: 0.5}"))
    r = Renderer(resolve(spec, tmp_path), tmp_path, Canvas(54, 96, device="cpu"))
    a, b = r._look(), r._look()
    assert a["lut"] is b["lut"] and a["strength"] == 0.5
    assert r.frame(0).shape == (3, 96, 54)


@pytest.mark.gpu
def test_identity_lut_on_cuda() -> None:
    cv = Canvas(64, 48, device="cuda")
    img = torch.rand(3, 48, 64, device="cuda")
    look = {"kind": "graded", "lut": lut_texture(parse_cube(_cube_text(33)), "cuda")}
    out = finish(cv, img, 3, look, _plan())
    assert (out - img).abs().max().item() < 1 / 255


def test_cube_domain_collapsing_in_float32_is_rejected():
    text = "LUT_1D_SIZE 2\nDOMAIN_MIN 1 1 1\nDOMAIN_MAX 1.00000001 1.00000001 1.00000001\n0 0 0\n1 1 1\n"
    with pytest.raises(ValueError, match="float32"):
        parse_cube(text)


def test_cube_table_values_overflowing_float32_are_rejected():
    with pytest.raises(ValueError, match="float32"):
        parse_cube("LUT_1D_SIZE 2\n0 0 0\n1e39 1 1\n")
