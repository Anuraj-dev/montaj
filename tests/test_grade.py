"""Linear-light working space and the per-shot primary grade (#43).

Contract: every operator is the identity at its default and matches a hand-checked value; the grade is
a pure float32 function that never writes into its input; absent fields keep every hash; a named grade
is materialised onto the shots that use it; a fade between bright shots has no mid-fade dip in linear.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from montaj.gpu_policy import GpuPolicy, GpuStrictError
from montaj.grade import Grade
from montaj.project import segment_hash
from montaj.render.canvas import Canvas
from montaj.render.color import LUMA, grade, monotone_curve, to_display, to_linear, wb_gains
from montaj.render.frame import Renderer
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

# Three channels in shadow, mid and highlight, as linear light.
PX = torch.tensor([0.05, 0.2, 0.6]).reshape(3, 1, 1)


def _close(img: torch.Tensor, want, tol: float = 1e-5) -> None:
    got = img.flatten().tolist()
    assert got == pytest.approx(list(want), abs=tol), got


# --- transforms ----------------------------------------------------------------------------------


def test_srgb_transforms_pin_and_round_trip() -> None:
    assert to_linear(torch.tensor(0.5)).item() == pytest.approx(0.214041, abs=1e-6)
    assert to_linear(torch.tensor(0.04)).item() == pytest.approx(0.04 / 12.92, abs=1e-7)  # linear toe
    assert to_display(torch.tensor(0.2)).item() == pytest.approx(0.484529, abs=1e-6)
    ramp = torch.linspace(0, 1, 1001)
    assert torch.allclose(to_display(to_linear(ramp)), ramp, atol=2e-6)
    assert to_display(torch.tensor([-0.5, 3.0])).tolist() == [0.0, 1.0]  # output transform clamps


# --- operators: identity and one pinned value each -----------------------------------------------


@pytest.mark.parametrize("field", [
    {"exposure": 0}, {"temp": 0}, {"tint": 0}, {"contrast": 1}, {"contrast": 1, "pivot": 0.2},
    {"lift": 0}, {"lift": [0, 0, 0]}, {"gamma": 1}, {"gamma": [1, 1, 1]}, {"gain": 1}, {"gain": [1, 1, 1]},
    {"saturation": 1}, {"vibrance": 0}, {"curves": {"luma": [[0, 0], [1, 1]]}},
    {"curves": {"luma": [[0, 0], [0.5, 0.5], [1, 1]]}},
])
def test_each_operator_is_identity_at_default(field: dict) -> None:
    img = torch.rand(3, 16, 16) * 1.5  # above 1 too: exposure headroom must survive the identity
    _close(grade(img, Grade(**field)), img.flatten().tolist())


@pytest.mark.parametrize(("field", "want"), [
    ({"exposure": 1}, [0.1, 0.4, 1.2]),
    ({"exposure": -2}, [0.0125, 0.05, 0.15]),
    ({"temp": 1000}, [0.054855, 0.197579, 0.500405]),
    ({"tint": 50}, [0.056420, 0.189774, 0.677041]),
    # p = lin(0.46) = 0.178868; 0.2 -> p * (0.2 / p) ** 1.5.
    ({"contrast": 1.5}, [0.026436, 0.211485, 1.098908]),
    # display 0.248461 + 0.1 * (1 - 0.248461) = 0.323615 -> linear 0.085159.
    ({"lift": 0.1}, [0.085159, 0.248965, 0.634724]),
    ({"gamma": 2}, [0.212007, 0.442428, 0.773948]),
    ({"gain": [1.2, 1, 0.8]}, [0.071939, 0.2, 0.364956]),
    # Y = 0.196990; half way from each channel to Y.
    ({"saturation": 0.5}, [0.123495, 0.198495, 0.398495]),
    ({"vibrance": 0.5}, [0.043875, 0.200125, 0.616792]),
    ({"curves": {"luma": [[0, 0], [0.5, 0.6], [1, 1]]}}, [0.075296, 0.301185, 0.903556]),
])
def test_each_operator_pinned_value(field: dict, want: list[float]) -> None:
    _close(grade(PX, Grade(**field)), want)


def test_white_balance_direction_and_luma_kept() -> None:
    assert wb_gains(0, 0) == (1.0, 1.0, 1.0)
    r, g, b = wb_gains(1000, 0)
    assert r > 1 > b  # warmer
    r2, _, b2 = wb_gains(-1000, 0)
    assert b2 > 1 > r2
    _, gm, _ = wb_gains(0, 50)
    assert gm < 1  # + tint is magenta
    for gains in (wb_gains(1000, 0), wb_gains(-3000, -40)):
        assert sum(c * w for c, w in zip(gains, LUMA)) == pytest.approx(1.0, abs=1e-12)


def test_saturation_zero_is_luminance_and_vibrance_favours_muted() -> None:
    grey = grade(PX, Grade(saturation=0)).flatten()
    assert torch.allclose(grey, torch.full((3,), 0.19699), atol=1e-5)
    muted = torch.tensor([0.30, 0.25, 0.22]).reshape(3, 1, 1)
    vivid = torch.tensor([0.80, 0.10, 0.02]).reshape(3, 1, 1)
    spread = lambda t: float(t.max() - t.min())  # noqa: E731
    gain_m = spread(grade(muted, Grade(vibrance=0.8))) / spread(muted)
    gain_v = spread(grade(vivid, Grade(vibrance=0.8))) / spread(vivid)
    assert gain_m > gain_v > 1


def test_luma_curve_is_monotone_and_continuous_past_its_ends() -> None:
    pts = [[0.1, 0.05], [0.4, 0.5], [0.6, 0.55], [0.9, 0.95]]
    x = torch.linspace(-0.2, 1.4, 1601)
    y = monotone_curve(pts, x)
    assert bool((y[1:] - y[:-1] >= -1e-6).all())
    knots = monotone_curve(pts, torch.tensor([p[0] for p in pts]))
    _close(knots, [p[1] for p in pts], 1e-6)
    assert monotone_curve(pts, torch.tensor([0.0])).item() == pytest.approx(-0.05, abs=1e-6)  # slope 1
    assert monotone_curve(pts, torch.tensor([1.2])).item() == pytest.approx(1.25, abs=1e-6)


def test_grade_is_pure_float32_and_never_writes_its_input() -> None:
    img = torch.rand(3, 8, 8)
    before = img.clone()
    for g in (Grade(), Grade(exposure=1, temp=500, contrast=1.2, lift=0.05, saturation=1.3)):
        out = grade(img, g)
        assert out.dtype == torch.float32 and out is not img
        out.add_(1)
        assert torch.equal(img, before)


# --- spec ----------------------------------------------------------------------------------------


def _png(path: Path, rgb: tuple[int, int, int], size=(32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, rgb).save(path)


# Two bright, complementary photos: a straight sRGB mix between them dips dark at mid-fade.
RED = (255, 64, 26)
SKY = (26, 153, 255)


def _project(tmp: Path, shots: str, *, size: str = "108x192", video: str = "", top: str = "") -> Path:
    _png(tmp / "assets" / "a.png", RED)
    _png(tmp / "assets" / "b.png", SKY)
    p = tmp / "montaj.yaml"
    p.write_text(f"video:\n  size: {size}\n  fps: 10\n  look: clean\n{video}assets: assets\n{top}shots:\n{shots}")
    return p


def _errors(tmp: Path, shots: str, **kw) -> list[str]:
    with pytest.raises(SpecError) as ei:
        resolve(load_spec(_project(tmp, shots, **kw)), tmp)
    return ei.value.errors


TWO = "  - {photo: a, hold: 10f}\n  - {photo: b, hold: 10f}\n"


@pytest.mark.parametrize(("shots", "top", "needle"), [
    ("  - {photo: a, hold: 5f, grade: warm}\n", "", "shots[0].grade: unknown grade 'warm' (define it under grades:; have: none defined)"),
    ("  - {photo: a, hold: 5f, grade: wram}\n", "grades:\n  warm: {exposure: 1}\n", "have: warm"),
    ("  - {photo: a, hold: 5f, grade: {exposure: 9}}\n", "", "shots[0].grade.exposure: 9.0 is out of range"),
    ("  - {photo: a, hold: 5f, grade: {temp: 5000}}\n", "", "shots[0].grade.temp"),
    ("  - {photo: a, hold: 5f, grade: {gamma: [1, 0, 1]}}\n", "", "shots[0].grade.gamma: 0.0 is out of range"),
    ("  - {photo: a, hold: 5f, grade: {contrast: 0}}\n", "", "shots[0].grade.contrast"),
    ("  - {photo: a, hold: 5f, grade: {curves: {luma: [[0, 0]]}}}\n", "", "needs at least 2 points"),
    ("  - {photo: a, hold: 5f, grade: {curves: {luma: [[0.5, 0], [0.5, 1]]}}}\n", "", "curves.luma[1]: x does not increase"),
    ("  - {photo: a, hold: 5f}\n", "grades:\n  Warm: {exposure: 1}\n", "grades.Warm: bad name"),
    ("  - {photo: a, hold: 5f}\n", "grades:\n  warm: {saturation: -1}\n", "grades.warm.saturation"),
    ("  - {blank: true, hold: 5f, grade: {exposure: 1}}\n  - {photo: a, hold: 5f}\n", "",
     "shots[0].grade: a blank shot has no picture to grade"),
])
def test_grade_spec_errors(tmp_path: Path, shots: str, top: str, needle: str) -> None:
    errs = _errors(tmp_path, shots, top=top)
    assert any(needle in e for e in errs), errs


def test_grade_schema_errors_name_the_yaml_path(tmp_path: Path) -> None:
    for bad, needle in (("{exposure: true}", "shots[0].grade.exposure"), ("{exposure: 1, hue: 3}", "shots[0].grade.hue"),
                        ("{lift: [0, 1]}", "shots[0].grade.lift")):
        errs = _errors(tmp_path, f"  - {{photo: a, hold: 5f, grade: {bad}}}\n")
        assert any(needle in e and "grade_inline" not in e for e in errs), errs
    errs = _errors(tmp_path, TWO, video="  color: {working: hdr}\n")
    assert any("video.color.working" in e for e in errs), errs


def _hashes(p: Path) -> list[str]:
    tl = resolve(load_spec(p), p.parent)

    class Shas(dict):
        def __missing__(self, key):
            return "sha-" + key

    return [segment_hash(tl, seg, Shas(), "e1") for seg in tl.segments]


def test_hashes_move_only_when_colour_is_set(tmp_path: Path) -> None:
    base = _hashes(_project(tmp_path, TWO))
    assert len(base) == 2
    assert _hashes(_project(tmp_path, TWO, top="grades:\n  warm: {temp: 800}\n")) == base  # unused grade
    lin = _hashes(_project(tmp_path, TWO, video="  color: {working: linear}\n"))
    assert lin[0] != base[0] and lin[1] != base[1]


def test_named_grade_rehashes_only_its_shots(tmp_path: Path) -> None:
    shots = "  - {photo: a, hold: 10f, grade: day}\n  - {photo: b, hold: 10f}\n"
    one = _hashes(_project(tmp_path, shots, top="grades:\n  day: {temp: 800}\n"))
    two = _hashes(_project(tmp_path, shots, top="grades:\n  day: {temp: 900}\n"))
    assert one[0] != two[0] and one[1] == two[1]
    inline = _hashes(_project(tmp_path, "  - {photo: a, hold: 10f, grade: {temp: 800}}\n  - {photo: b, hold: 10f}\n"))
    assert inline == one  # the name resolves to the same values as writing them inline
    tl = resolve(load_spec(tmp_path / "montaj.yaml"), tmp_path)
    assert tl.shots[0].spec.grade == Grade(temp=800)


# --- renderer ------------------------------------------------------------------------------------


def _renderer(tmp: Path, shots: str, *, device: str = "cpu", policy=None, **kw) -> Renderer:
    tl = resolve(load_spec(_project(tmp, shots, **kw)), tmp)
    return Renderer(tl, tmp, Canvas(tl.size[0], tl.size[1], device=device), policy=policy)


def _centre(img: torch.Tensor) -> torch.Tensor:
    _, h, w = img.shape
    return img[:, h // 2 - 4:h // 2 + 4, w // 2 - 4:w // 2 + 4].mean((1, 2)).cpu()


def test_display_mode_without_grade_is_bit_identical(tmp_path: Path) -> None:
    shots = TWO.replace("{photo: b,", "{photo: b, in: {type: fade, dur: 6f},")
    plain = _renderer(tmp_path, shots)
    explicit = _renderer(tmp_path, shots, video="  color: {working: display}\n")
    for f in range(0, 20, 3):
        assert torch.equal(plain.frame(f), explicit.frame(f))


def test_linear_mode_round_trips_a_still_and_grades_apply(tmp_path: Path) -> None:
    disp = _renderer(tmp_path, TWO).frame(2)
    lin = _renderer(tmp_path, TWO, video="  color: {working: linear}\n").frame(2)
    assert torch.allclose(disp, lin, atol=1e-5)
    up = _renderer(tmp_path, TWO.replace("hold: 10f}", "hold: 10f, grade: {exposure: 1}}", 1)).frame(2)
    want = to_display(to_linear(_centre(disp)) * 2)
    assert torch.allclose(_centre(up), want, atol=2e-3)  # clean look's sharpen is flat on a solid
    assert torch.equal(_renderer(tmp_path, TWO.replace("hold: 10f}", "hold: 10f, grade: {exposure: 1}}", 1)).frame(15),
                       _renderer(tmp_path, TWO).frame(15))  # shot 1 is ungraded


def _fade_luminance(tmp: Path, mode: str, device: str = "cpu", size: str = "108x192") -> list[float]:
    """Rec.709 linear luminance at the frame centre for every frame of a 10-frame fade A -> B."""
    shots = "  - {photo: a, hold: 10f}\n  - {photo: b, hold: 16f, in: {type: fade, dur: 10f}}\n"
    r = _renderer(tmp, shots, device=device, size=size, video=f"  color: {{working: {mode}}}\n")
    out = []
    for f in range(8, 22):
        c = to_linear(_centre(r.frame(f)))
        out.append(float(sum(c[i] * LUMA[i] for i in range(3))))
    return out


def _assert_no_dip(tmp: Path, device: str, size: str) -> None:
    disp = _fade_luminance(tmp, "display", device, size)
    lin = _fade_luminance(tmp, "linear", device, size)
    ends = min(disp[0], disp[-1])
    print(f"\nfade luminance {device} {size}: endpoints A={disp[0]:.4f} B={disp[-1]:.4f}")
    print("display:", " ".join(f"{v:.4f}" for v in disp), f"min {min(disp):.4f}")
    print("linear: ", " ".join(f"{v:.4f}" for v in lin), f"min {min(lin):.4f}")
    assert abs(lin[0] - disp[0]) < 2e-3 and abs(lin[-1] - disp[-1]) < 2e-3
    assert min(disp) < ends - 0.05  # display mode: a real mid-fade dip
    assert min(lin) >= ends - 2e-3  # linear mode: never darker than the darker end


def test_linear_fade_has_no_mid_fade_dip(tmp_path: Path) -> None:
    _assert_no_dip(tmp_path, "cpu", "108x192")


def test_linear_dip_passes_through_the_exact_colour(tmp_path: Path) -> None:
    shots = "  - {photo: a, hold: 10f}\n  - {photo: b, hold: 16f, in: {type: dip, dur: 10f, color: '#336699'}}\n"
    r = _renderer(tmp_path, shots, video="  color: {working: linear}\n")
    tr = r.tl.plan(15).scene
    mid = tr.t0 + tr.dur / 2
    img = to_display(r._eval(tr, float(mid)))
    _close(_centre(img), (0x33 / 255, 0x66 / 255, 0x99 / 255), 2e-4)


def test_strict_policy_refuses_a_cpu_grade(tmp_path: Path) -> None:
    graded = TWO.replace("hold: 10f}", "hold: 10f, grade: {exposure: 1}}", 1)
    with pytest.raises(GpuStrictError):
        _renderer(tmp_path, graded, policy=GpuPolicy(strict=True)).frame(2)
    with pytest.raises(GpuStrictError):
        _renderer(tmp_path, TWO, policy=GpuPolicy(strict=True), video="  color: {working: linear}\n").frame(2)


@pytest.mark.gpu
def test_gpu_linear_fade_has_no_dip_and_grade_stays_on_gpu(tmp_path: Path) -> None:
    _assert_no_dip(tmp_path, "cuda", "180x320")
    pol = GpuPolicy(strict=False)  # PNG fixtures decode on the CPU; only the grade seam is under test
    shots = ("  - {photo: a, hold: 10f, grade: {exposure: 0.5, temp: 600, curves: {luma: [[0, 0], [0.5, 0.55], [1, 1]]}}}\n"
             "  - {photo: b, hold: 16f, in: {type: fade, dur: 10f}}\n")
    r = _renderer(tmp_path, shots, device="cuda", policy=pol, size="180x320", video="  color: {working: linear}\n")
    gpu = r.frame(12)
    assert gpu.device.type == "cuda"
    assert pol.seams.counts[("grade", "gpu")] > 0 and "grade" not in pol.cpu_stages()
    cpu = _renderer(tmp_path, shots, size="180x320", video="  color: {working: linear}\n").frame(12)
    assert torch.allclose(_centre(gpu), _centre(cpu), atol=2e-3)
