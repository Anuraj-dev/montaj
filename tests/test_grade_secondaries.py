"""Secondaries, power windows, protect: skin and the film look (#44).

Contract: each secondary's mask selects what it says (pinned at the centre, edge and outside), and a
secondary whose `adjust` is the identity leaves the primary grade bit-identical; the look's grain is a
pure function of the frame number; grades without #44 fields keep their dump (so every hash); a
`center: face` window uses ingest's face boxes when they exist and the subject matte's centroid when not.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from montaj.analysis import ANALYSIS_VERSION, _sha
from montaj.gpu_policy import GpuPolicy
from montaj.grade import Adjust, Grade, Key, SplitTone, grade_warnings
from montaj.render.canvas import Canvas
from montaj.render.color import (LUMA, GradeContext, apply, centroid, ellipse_mask, film_grain, gate_weave,
                                 gradient_mask, grade, halation, hsl, key_mask, protect_skin, split_tone, to_linear)
from montaj.render.frame import Renderer
from montaj.spec import SpecError, layer_warnings, load_spec
from montaj.timeline import resolve
from tests.test_grade import _project
from tests.test_matte import EDGE, _load, _matte


def _ctx(subject=None, face=None, t: float = 0.0, k: float = 1.0) -> GradeContext:
    return GradeContext(k, t, 30.0, lambda: subject, face or (lambda pt: None))


def _lin(*rgb255) -> torch.Tensor:
    """One linear pixel per colour, as (3, 1, n)."""
    return to_linear(torch.tensor(rgb255, dtype=torch.float32).T.reshape(3, 1, -1) / 255.0)


def _luma(img: torch.Tensor) -> torch.Tensor:
    return img[0] * LUMA[0] + img[1] * LUMA[1] + img[2] * LUMA[2]


SKIN, SKY, GREY = (224, 172, 140), (40, 120, 230), (128, 128, 128)


# --- masks -----------------------------------------------------------------------------------------


def test_hsl_reads_display_values() -> None:
    h, s, y = hsl(_lin((255, 0, 0), (0, 0, 255), GREY))
    assert h.flatten().tolist() == pytest.approx([0, 240, 0], abs=1e-3)
    assert s.flatten().tolist() == pytest.approx([1, 1, 0], abs=1e-4)
    assert y.flatten()[2].item() == pytest.approx(128 / 255, abs=1e-4)


def test_key_mask_selects_skin_not_sky_or_grey_with_a_soft_edge() -> None:
    key = Key(hue=(15, 45), sat=(0.2, 0.9), luma=(0.2, 0.95), soft=0.05)
    m = key_mask(_lin(SKIN, SKY, GREY), key).flatten().tolist()
    assert m == [1.0, 0.0, 0.0]
    # Hue 15 + 10° is inside; 15 − 9° is inside the 18° soft edge, partly selected; 15 − 20° is out.
    hues = torch.tensor([25.0, 6.0, 355.0])
    rgb = torch.stack([torch.tensor(_hue_rgb(float(h))) for h in hues], 1).reshape(3, 1, -1)
    m = key_mask(to_linear(rgb), Key(hue=(15, 45), soft=0.05)).flatten().tolist()
    assert m[0] == 1.0 and 0.0 < m[1] < 1.0 and m[2] == 0.0
    # A wrapping range [340, 20] selects pure red.
    assert key_mask(_lin((255, 0, 0)), Key(hue=(340, 20), soft=0)).item() == 1.0


def _hue_rgb(h: float) -> tuple[float, float, float]:
    import colorsys

    return colorsys.hls_to_rgb(h / 360.0, 0.5, 0.7)


def test_ellipse_mask_is_one_inside_half_on_the_radius_zero_beyond_the_feather() -> None:
    m = ellipse_mask(200, 200, "cpu", 100.0, 100.0, 60.0, 30.0, 20.0)[0]
    assert m[100, 100] == 1.0
    assert m[100, 100 + 59] == pytest.approx(0.5, abs=0.06)  # pixel centre 159.5: 59.5 px out on x
    assert m[100 - 30, 100] == pytest.approx(0.5, abs=0.06)  # 29.5 px up on y
    assert m[100, 100 + 75] == 0.0 and m[100 - 45, 100] == 0.0  # past radius + feather/2
    assert m[100, 100 + 45] == 1.0  # inside radius − feather/2
    hard = ellipse_mask(200, 200, "cpu", 100.0, 100.0, 60.0, 30.0, 0.0)[0]
    assert set(hard.unique().tolist()) == {0.0, 1.0}


def test_gradient_mask_runs_from_one_to_zero() -> None:
    m = gradient_mask(101, 10, "cpu", (5.0, 0.5), (5.0, 100.5))[0]
    assert m[0, 5] == 1.0 and m[100, 5] == 0.0
    assert m[50, 5] == pytest.approx(0.5, abs=1e-5)
    col = m[:, 5]
    assert bool((col[1:] <= col[:-1]).all())


def test_centroid_of_a_matte() -> None:
    m = torch.zeros(1, 40, 60)
    m[:, 10:20, 0:30] = 1
    assert centroid(m) == pytest.approx((15.0, 15.0))
    assert centroid(torch.zeros(1, 4, 4)) is None


# --- each secondary: mask applies its adjust; identity adjust is bit-identical -------------------


SECONDARIES = [
    {"key": {"hue": [15, 45], "sat": [0.2, 0.9], "luma": [0.2, 0.95]}},
    {"window": {"ellipse": {"center": [32, 32], "radius": 20, "feather": 8}}},
    {"window": {"gradient": {"from": [0, 10], "to": [0, 54]}}},
    {"matte": "subject"},
    {"matte": "subject", "invert": True},
    {"window": {"ellipse": {"center": "subject", "radius": [10, 14]}}},
    {"window": {"ellipse": {"center": "face", "radius": 10}}},
]


def _picture() -> tuple[torch.Tensor, torch.Tensor]:
    """64x64: skin on the left half, sky on the right; the subject matte is the left half."""
    img = torch.empty(3, 64, 64)
    img[:, :, :32] = _lin(SKIN).reshape(3, 1, 1)
    img[:, :, 32:] = _lin(SKY).reshape(3, 1, 1)
    sub = torch.zeros(1, 64, 64)
    sub[:, :, :32] = 1
    return img, sub


@pytest.mark.parametrize("sec", SECONDARIES)
def test_identity_adjust_leaves_the_primary_bit_identical(sec: dict) -> None:
    img, sub = _picture()
    g = Grade(exposure=0.3, secondaries=[{**sec, "adjust": {}}])
    assert torch.equal(apply(img, g, _ctx(sub)), grade(img, Grade(exposure=0.3)))


@pytest.mark.parametrize(("sec", "where_in", "where_out"), [
    (SECONDARIES[0], (32, 8), (32, 56)),  # skin key: left yes, sky no
    (SECONDARIES[1], (32, 32), (2, 2)),  # ellipse at the centre
    (SECONDARIES[2], (2, 40), (62, 40)),  # gradient: top yes, bottom no
    (SECONDARIES[3], (32, 8), (32, 56)),  # subject: the left half
    (SECONDARIES[4], (32, 56), (32, 8)),  # inverted: the background
    (SECONDARIES[5], (32, 16), (32, 48)),  # ellipse on the subject centroid (16.5, 32.5)
    (SECONDARIES[6], (32, 16), (32, 48)),  # face without face boxes: the same centroid
])
def test_each_secondary_adjusts_inside_its_mask_only(sec: dict, where_in, where_out) -> None:
    img, sub = _picture()
    base = grade(img, Grade())
    out = apply(img, Grade(secondaries=[{**sec, "adjust": {"exposure": 1}}]), _ctx(sub))
    yi, xi = where_in
    yo, xo = where_out
    assert torch.allclose(out[:, yi, xi], base[:, yi, xi] * 2, rtol=1e-5)
    assert torch.equal(out[:, yo, xo], base[:, yo, xo])


def test_face_point_maps_through_the_context_and_offset_moves_it() -> None:
    img, sub = _picture()
    sec = {"window": {"ellipse": {"center": "face", "radius": 6, "face_point": [100, 200]}}, "adjust": {"exposure": 1}}
    out = apply(img, Grade(secondaries=[sec]), _ctx(sub, face=lambda pt: (pt[0] / 4, pt[1] / 4)))
    changed = (out != img).any(0)
    ys, xs = changed.nonzero(as_tuple=True)
    assert xs.float().mean().item() == pytest.approx(24.5, abs=0.6) and ys.float().mean().item() == pytest.approx(49.5, abs=0.6)
    sec["window"]["ellipse"]["offset"] = [10, -20]
    out = apply(img, Grade(secondaries=[sec]), _ctx(sub, face=lambda pt: (pt[0] / 4, pt[1] / 4), k=0.5))
    ys, xs = (out != img).any(0).nonzero(as_tuple=True)
    assert xs.float().mean().item() == pytest.approx(29.5, abs=0.6) and ys.float().mean().item() == pytest.approx(39.5, abs=0.6)


def test_window_without_a_subject_selects_nothing() -> None:
    img, _ = _picture()
    g = Grade(secondaries=[{"window": {"ellipse": {"center": "subject", "radius": 9}}, "adjust": {"exposure": 1}}])
    assert torch.equal(apply(img, g, _ctx(torch.zeros(1, 64, 64))), img)


def test_apply_never_writes_its_input() -> None:
    img, sub = _picture()
    before = img.clone()
    g = Grade(secondaries=[{"matte": "subject", "adjust": {"saturation": 0}}], protect="skin",
              look={"film": {"stock": "warm-negative", "halation": 0.5, "bloom": 0.5, "grain": {"amount": 0.5},
                             "gate_weave": 1}, "split_tone": {"shadows": "#103040", "highlights": "#ffd9a0"}})
    out = apply(img, g, _ctx(sub, t=3))
    assert torch.equal(img, before) and out.dtype == torch.float32 and out.shape == img.shape


# --- look ------------------------------------------------------------------------------------------


def test_split_tone_keeps_grey_luminance_and_tints_each_end() -> None:
    st = SplitTone(shadows="#103040", highlights="#ffd9a0", amount=0.5)
    dark, light = _lin((30, 30, 30)), _lin((230, 230, 230))
    for px in (dark, light):
        assert _luma(split_tone(px, st)).item() == pytest.approx(_luma(px).item(), rel=1e-4)
    d, li = split_tone(dark, st).flatten(), split_tone(light, st).flatten()
    assert d[2] > d[0] and li[0] > li[2]  # teal shadows, warm highlights
    assert torch.equal(split_tone(light, SplitTone(shadows="#103040", highlights="#ffd9a0", amount=0)), light)


def test_protect_skin_restores_skin_hue_and_keeps_graded_luminance() -> None:
    ref = _lin(SKIN, SKY)
    shifted = ref * torch.tensor([0.7, 1.0, 1.4]).reshape(3, 1, 1)  # a cold look
    out = protect_skin(shifted, ref)
    skin, sky = out[:, 0, 0], out[:, 0, 1]
    want = ref[:, 0, 0] * (_luma(shifted)[0, 0] / _luma(ref)[0, 0])
    assert torch.allclose(skin, want, rtol=1e-5)
    assert torch.equal(sky, shifted[:, 0, 1])


def test_protect_skin_in_apply_guards_only_the_look() -> None:
    img, sub = _picture()
    cold = {"split_tone": {"shadows": "#2060ff", "highlights": "#2060ff", "amount": 0.6}}
    free = apply(img, Grade(look=cold), _ctx(sub))
    kept = apply(img, Grade(look=cold, protect="skin"), _ctx(sub))
    h0, _, _ = hsl(img[:, :1, :1])
    hf, _, _ = hsl(free[:, :1, :1])
    hk, _, _ = hsl(kept[:, :1, :1])
    assert abs(hf.item() - h0.item()) > 10 and abs(hk.item() - h0.item()) < 0.5
    assert torch.equal(kept[:, :, 40:], free[:, :, 40:])  # sky is not skin


def test_halation_is_red_weighted_and_only_around_highlights() -> None:
    img = torch.full((3, 64, 64), 0.02)
    img[:, 30:34, 30:34] = 4.0
    out = halation(img, 0.5, 1.0)
    glow = out - img
    assert glow[:, :2, :2].abs().max() < 1e-4  # far from the highlight
    ring = glow[:, 32, 38]
    assert ring[0] > ring[1] > ring[2] > 0
    dark = torch.full((3, 16, 16), 0.1)
    assert torch.equal(halation(dark, 1.0, 1.0), dark)


def test_grain_is_a_pure_function_of_the_frame_and_follows_luminance() -> None:
    img = torch.full((3, 96, 96), 0.18)
    a, b, c = film_grain(img, 0.5, 1.0, 7, 1.0), film_grain(img, 0.5, 1.0, 7, 1.0), film_grain(img, 0.5, 1.0, 8, 1.0)
    assert torch.equal(a, b) and not torch.equal(a, c)
    n = (a - img)[0]
    assert abs(n.mean().item()) < 0.01 and n.std().item() > 0.01
    assert torch.equal(a[0], a[1])  # monochrome
    black = torch.zeros(3, 32, 32)
    assert torch.equal(film_grain(black, 1.0, 1.0, 3, 1.0), black)  # no grain where there is no light
    coarse = film_grain(img, 0.5, 4.0, 7, 1.0) - img
    assert (coarse[0, :, 1:] - coarse[0, :, :-1]).abs().mean() < (n[:, 1:] - n[:, :-1]).abs().mean()


def test_gate_weave_moves_the_frame_smoothly_over_time() -> None:
    img = torch.zeros(3, 64, 64)
    img[:, :, 32:] = 1.0
    a, b = gate_weave(img, 1.0, 10.0, 30.0, 1.0), gate_weave(img, 1.0, 10.5, 30.0, 1.0)
    assert not torch.equal(a, img)
    assert (a - b).abs().max() < 0.5  # half a frame later it has barely moved
    assert torch.equal(gate_weave(img, 1.0, 10.0, 30.0, 1.0), a)


def test_stock_matches_its_adjust_on_top_of_the_primary() -> None:
    from montaj.grade import STOCKS

    img, sub = _picture()
    for name, adj in STOCKS.items():
        out = apply(img, Grade(exposure=0.2, look={"film": {"stock": name}}), _ctx(sub))
        assert torch.equal(out, grade(grade(img, Grade(exposure=0.2)), Adjust(**adj))), name


def test_stocks_do_not_blow_up_near_black_chroma() -> None:
    """Regression: a stock curve with a lifted black point scaled dark noise by y'/y into blue blotches."""
    from montaj.grade import STOCKS

    dark = torch.tensor([0.0004, 0.0005, 0.0020]).reshape(3, 1, 1)
    spread = (dark.amax(0) - dark.amin(0)).item()
    for name, adj in STOCKS.items():
        out = grade(dark, Adjust(**adj))
        assert (out.amax(0) - out.amin(0)).item() < 2 * spread, name


# --- spec, hash, warnings --------------------------------------------------------------------------


def test_plain_grade_dump_is_unchanged() -> None:
    assert Grade(temp=800).model_dump() == Adjust(temp=800).model_dump()
    assert "secondaries" not in Grade(temp=800).model_dump()


def _errors(tmp: Path, grade_yaml: str, shot: str = "photo: a, hold: 5f") -> list[str]:
    p = _project(tmp, f"  - {{{shot}, grade: {grade_yaml}}}\n")
    with pytest.raises(SpecError) as ei:
        resolve(load_spec(p), tmp)
    return ei.value.errors


@pytest.mark.parametrize(("grade_yaml", "needle"), [
    ("{secondaries: [{adjust: {}}]}", "secondaries[0]: selects nothing"),
    ("{secondaries: [{key: {soft: 0.1}, adjust: {}}]}", "secondaries[0].key: has no range"),
    ("{secondaries: [{key: {hue: [0, 400]}, adjust: {}}]}", "secondaries[0].key.hue: 400.0 is out of range"),
    ("{secondaries: [{key: {sat: [0.8, 0.2]}, adjust: {}}]}", "key.sat: low end is above high end"),
    ("{secondaries: [{window: {}, adjust: {}}]}", "needs exactly one of ellipse or gradient"),
    ("{secondaries: [{window: {ellipse: {center: [1, 2], radius: 0}}, adjust: {}}]}", "radius: 0.0 must be above 0"),
    ("{secondaries: [{window: {gradient: {from: [1, 2], to: [1, 2]}}, adjust: {}}]}", "same point"),
    ("{secondaries: [{key: {hue: [1, 2]}, adjust: {exposure: 20}}]}", "secondaries[0].adjust.exposure"),
    ("{secondaries: [{matte: subject, adjust: {}}]}", "need the shot's subject matte"),
    ("{secondaries: [{window: {ellipse: {center: face, radius: 9}}, adjust: {}}]}", "need the shot's subject matte"),
    ("{secondaries: [{matte: person, adjust: {}}]}", "secondaries[0].matte: unknown 'person'"),
    ("{look: {film: {stock: kodak}}}", "look.film.stock: unknown 'kodak'"),
    ("{look: {film: {halation: 2}}}", "look.film.halation"),
    ("{look: {film: {grain: {amount: 0.2, size: 9}}}}", "look.film.grain.size"),
    ("{look: {split_tone: {shadows: teal, highlights: '#ffffff'}}}", "split_tone.shadows: 'teal' is not a colour"),
    ("{protect: face}", "grade.protect"),
])
def test_secondary_spec_errors(tmp_path: Path, grade_yaml: str, needle: str) -> None:
    errs = _errors(tmp_path, grade_yaml)
    assert any(needle in e for e in errs), errs


# --- renderer: matte secondaries and face windows on a real clip shot ----------------------------


def _render(tmp: Path, shots: str, f: int = 10, policy=None, device: str = "cpu"):
    tl = resolve(load_spec(_load(tmp, shots)), tmp)
    r = Renderer(tl, tmp, Canvas(tl.size[0], tl.size[1], device=device), policy=policy)
    return r, r.frame(f)


def test_matte_secondary_on_a_clip_shot_darkens_only_the_background(tmp_path: Path) -> None:
    _matte(tmp_path)
    _, bare = _render(tmp_path, "  - {clip: rec, hold: 1s, matte: subject}\n")
    sec = "{secondaries: [{matte: subject, invert: true, adjust: {exposure: -2}}]}"
    _, out = _render(tmp_path, f"  - {{clip: rec, hold: 1s, matte: subject, grade: {sec}}}\n")
    assert torch.allclose(out[:, :, : EDGE - 4], bare[:, :, : EDGE - 4], atol=2e-3)
    right_b, right_o = bare[:, :, EDGE + 4:].mean(), out[:, :, EDGE + 4:].mean()
    assert right_o < right_b * 0.6


def _analysis(tmp: Path, faces: list) -> None:
    assets = tmp / "assets"
    clip = assets / "rec.mkv"
    st = clip.stat()
    row = {"kind": "clip", "sha": _sha(clip), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "v": ANALYSIS_VERSION,
           "w": 320, "h": 240, "duration": 3, "fps": 30, "scenes": [{"s": 0, "e": 3, "faces": faces, "subject": [80, 120]}]}
    (assets / "analysis.json").write_text(json.dumps({"rec.mkv": row}))


def _changed_centre(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    ys, xs = ((a - b).abs().amax(0) > 0.01).nonzero(as_tuple=True)
    return xs.float().mean().item(), ys.float().mean().item()


def test_face_window_uses_ingest_face_boxes_else_the_matte_centroid(tmp_path: Path) -> None:
    _matte(tmp_path)
    shot = "  - {{clip: rec, hold: 1s, matte: subject{grade}}}\n"
    sec = ", grade: {secondaries: [{window: {ellipse: {center: face, radius: 12, feather: 0}}, adjust: {exposure: 1}}]}"
    _, bare = _render(tmp_path, shot.format(grade=""))
    spec = load_spec(_load(tmp_path, shot.format(grade=sec)))
    assert any("follows the subject matte's centroid" in w for w in layer_warnings(spec, tmp_path))
    r, out = _render(tmp_path, shot.format(grade=sec))
    x, y = _changed_centre(out, bare)
    # The matte is source x < 160 -> frame x < 54: its centroid is (27, 96).
    assert x == pytest.approx(27, abs=1.5) and y == pytest.approx(96, abs=1.5)
    # Ingest found a face at source (200, 60): the cover fit (0.8x, x − 92.5) puts it at frame (86, 48).
    _analysis(tmp_path, [{"box": [190, 50, 210, 70], "eyes": [], "score": 0.9}])
    spec = load_spec(_load(tmp_path, shot.format(grade=sec)))
    assert grade_warnings(spec, tmp_path) == []
    r, out = _render(tmp_path, shot.format(grade=sec))
    assert r.tl.shots[0].spec.grade.secondaries[0].window.ellipse.face_point == (200.0, 60.0)
    x, y = _changed_centre(out, bare)
    assert x == pytest.approx(86, abs=1.5) and y == pytest.approx(48, abs=1.5)


def test_no_grade_fields_render_bit_identical(tmp_path: Path) -> None:
    _matte(tmp_path)
    _, a = _render(tmp_path, "  - {clip: rec, hold: 1s, matte: subject, grade: {exposure: 0.5}}\n")
    _, b = _render(tmp_path, "  - {clip: rec, hold: 1s, matte: subject, grade: {exposure: 0.5, secondaries: "
                             "[{matte: subject, adjust: {}}]}}\n")
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.gpu
def test_gpu_secondaries_and_film_look_stay_on_gpu_and_match_cpu(tmp_path: Path) -> None:
    _matte(tmp_path)
    p = _load(tmp_path, "  - {clip: rec, hold: 1s, matte: subject, grade: {secondaries: ["
                        "{matte: subject, invert: true, adjust: {exposure: -1, saturation: 0.5}}, "
                        "{window: {ellipse: {center: subject, radius: 40, feather: 30}}, adjust: {exposure: 0.4}}, "
                        "{key: {hue: [0, 60], sat: [0.1, 1]}, adjust: {temp: 300}}, "
                        "{window: {gradient: {from: [540, 0], to: [540, 900]}}, adjust: {temp: -400}}], "
                        "protect: skin, look: {film: {stock: warm-negative, halation: 0.4, bloom: 0.2, "
                        "grain: {amount: 0.3, size: 1.2}, gate_weave: 0.3}, "
                        "split_tone: {shadows: '#103040', highlights: '#ffd9a0'}}}}\n")
    p.write_text(p.read_text().replace("size: 108x192", "size: 180x320"))
    tl = resolve(load_spec(p), tmp_path)
    pol = GpuPolicy(strict=False)
    gpu = Renderer(tl, tmp_path, Canvas(180, 320, device="cuda"), policy=pol).frame(10)
    assert gpu.device.type == "cuda"
    assert pol.seams.counts[("grade", "gpu")] > 0 and "grade" not in pol.cpu_stages()
    again = Renderer(tl, tmp_path, Canvas(180, 320, device="cuda")).frame(10)
    assert torch.equal(gpu, again)  # grain seeded per frame: deterministic
    # Same pipeline on the CPU, grain off (its RNG differs per device): the pictures agree.
    p.write_text(p.read_text().replace("amount: 0.3", "amount: 0"))
    tl = resolve(load_spec(p), tmp_path)
    g2 = Renderer(tl, tmp_path, Canvas(180, 320, device="cuda")).frame(10).cpu()
    c2 = Renderer(tl, tmp_path, Canvas(180, 320, device="cpu")).frame(10)
    assert (g2 - c2).abs().mean() < 3e-3
