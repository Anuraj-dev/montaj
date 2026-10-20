"""wall-pullback and burn-white: spec rules, timeline window, cache key, and CPU pixel pins."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from montaj.project import file_sha, segment_hash
from montaj.render.canvas import Canvas
from montaj.render.core import outc
from montaj.render.transitions import burn_white, wall_pullback
from montaj.spec import SpecError, load_spec
from montaj.timeline import Trans, resolve


def _write(tmp: Path, shots: str) -> Path:
    assets = tmp / "assets"
    assets.mkdir(exist_ok=True)
    for s in ("a", "b"):
        Image.new("RGB", (8, 8), (20, 40, 60)).save(assets / f"{s}.jpg", "JPEG")
    p = tmp / "montaj.yaml"
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\n"
        f"assets: assets\nshots:\n{shots}\n"
    )
    return p


def _shots(extra: str) -> str:
    return f"  - {{photo: a, hold: 40f}}\n  - {{photo: b, hold: 40f, in: {{{extra}}}}}\n"


def _resolve(tmp: Path, extra: str):
    return resolve(load_spec(_write(tmp, _shots(extra))), tmp)


@pytest.mark.parametrize("kind", ["wall-pullback", "burn-white"])
def test_window_is_first_dur_frames_of_incoming_shot(tmp_path: Path, kind: str):
    tl = _resolve(tmp_path, f"type: {kind}, dur: 20f")
    assert tl.plan(39).scene.__class__.__name__ == "Still"
    sc = tl.plan(40).scene
    assert isinstance(sc, Trans) and sc.kind == kind and sc.t0 == 40.0 and sc.dur == 20.0
    assert isinstance(tl.plan(59).scene, Trans)
    assert not isinstance(tl.plan(60).scene, Trans)


def test_pullback_zoom_defaults_and_overrides(tmp_path: Path):
    assert _resolve(tmp_path, "type: wall-pullback, dur: 20f").plan(45).scene.zoom == 3.2
    other = tmp_path / "z"
    other.mkdir()
    assert _resolve(other, "type: wall-pullback, dur: 20f, zoom: 5").plan(45).scene.zoom == 5.0


def test_burn_white_has_no_zoom(tmp_path: Path):
    assert _resolve(tmp_path, "type: burn-white, dur: 20f").plan(45).scene.zoom is None


@pytest.mark.parametrize(
    "extra, needle",
    [
        ("type: wall-pullback, dur: 20f, zoom: 1", "shots[1].in.zoom"),
        ("type: wall-pullback, dur: 20f, zoom: 0.5", "shots[1].in.zoom"),
        ("type: burn-white, dur: 20f, zoom: 2", "shots[1].in.zoom"),
        ("type: wall-pullback, dur: 50f", "longer than hold"),
        ("type: burn-white, dur: 50f", "longer than hold"),
        ("type: burn, dur: 20f", "wall-pullback"),
    ],
)
def test_validate_rejects_with_hint(tmp_path: Path, extra: str, needle: str):
    p = _write(tmp_path, _shots(extra))
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert any(needle in e and "(" in e for e in ei.value.errors), ei.value.errors


@pytest.mark.parametrize("kind", ["wall-pullback", "burn-white"])
def test_first_shot_rejects(tmp_path: Path, kind: str):
    p = _write(tmp_path, f"  - {{photo: a, hold: 40f, in: {{type: {kind}, dur: 10f}}}}\n")
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert any("shots[0].in" in e for e in ei.value.errors)


def test_overlapping_windows_rejected(tmp_path: Path):
    shots = (
        "  - {photo: a, hold: 40f}\n"
        "  - {photo: b, hold: 20f, in: {type: burn-white, dur: 20f}}\n"
        "  - {photo: a, hold: 20f, in: {type: whip, dur: 10f}}\n"
    )
    with pytest.raises(SpecError) as ei:
        load_spec(_write(tmp_path, shots))
    assert any("overlap" in e for e in ei.value.errors), ei.value.errors


@pytest.mark.parametrize("field, a, b", [("dur", "20f", "21f"), ("zoom", "3", "4")])
def test_pullback_params_reach_hash_of_both_shots(tmp_path: Path, field: str, a: str, b: str):
    def hashes(tmp: Path, value: str):
        tmp.mkdir()
        base = "type: wall-pullback, dur: 20f, zoom: 3"
        extra = base.replace(f"{field}: {a}", f"{field}: {value}")
        tl = _resolve(tmp, extra)
        shas = {s: file_sha(tmp / "assets" / f"{s}.jpg") for s in ("a", "b")}
        return [segment_hash(tl, seg, shas, "e1") for seg in tl.segments]

    h0, h1 = hashes(tmp_path / "x", a), hashes(tmp_path / "y", b)
    assert h0 != h1


def test_default_zoom_equals_explicit_default_hash(tmp_path: Path):
    def h(tmp: Path, extra: str):
        tmp.mkdir()
        tl = _resolve(tmp, extra)
        shas = {s: file_sha(tmp / "assets" / f"{s}.jpg") for s in ("a", "b")}
        return [segment_hash(tl, seg, shas, "e1") for seg in tl.segments]

    assert h(tmp_path / "x", "type: wall-pullback, dur: 20f") == h(
        tmp_path / "y", "type: wall-pullback, dur: 20f, zoom: 3.2"
    )


# --- pixel pins (CPU) -------------------------------------------------------

W, H = 54, 96


@pytest.fixture(scope="module")
def cv():
    return Canvas(W, H, device="cpu")


def _frames():
    a = torch.zeros(3, H, W)
    a[0] = 0.8  # red outgoing
    b = torch.zeros(3, H, W)
    b[2] = 0.6  # blue incoming
    b[:, :, : W // 2] += 0.2
    return a, b


def test_pullback_ends_are_exact(cv):
    a, b = _frames()
    assert torch.equal(wall_pullback(cv, a, b, 0.0), a)
    assert torch.equal(wall_pullback(cv, a, b, 1.0), b)


def test_pullback_card_shrinks_then_dissolves(cv):
    a, b = _frames()
    x = 0.4
    e = outc(x)
    out = wall_pullback(cv, a, b, x)
    # Camera scale k = zoom -> 1; the card is a scaled by k / zoom, centred.
    k = 3.2 + (1.0 - 3.2) * e
    half = k / 3.2
    assert 0.0 < half < 1.0
    cy, cx = H // 2, W // 2
    assert torch.allclose(out[:, cy, cx], a[:, cy, cx], atol=1e-5)  # centre is the opaque card
    # Corner is outside the card: not the outgoing red, and nothing was added to it.
    assert out[0, 0, 0] < 0.5 and out[2, 0, 0] > 0.0
    # A later frame is past the hold: the card is translucent, so centre is a mix.
    late = wall_pullback(cv, a, b, 0.95)
    assert not torch.allclose(late[:, cy, cx], a[:, cy, cx], atol=1e-3)
    assert late.min() >= 0.0 and late.max() <= 1.0


def test_pullback_is_deterministic_and_pure(cv):
    a, b = _frames()
    a0, b0 = a.clone(), b.clone()
    x1 = wall_pullback(cv, a, b, 0.3)
    x2 = wall_pullback(cv, a, b, 0.3)
    assert torch.equal(x1, x2) and torch.equal(a, a0) and torch.equal(b, b0)


def test_burn_white_ends_and_midpoint():
    a, b = _frames()
    assert torch.equal(burn_white(a, b, 0.0), a)
    assert torch.equal(burn_white(a, b, 1.0), b)
    assert torch.equal(burn_white(a, b, 0.5), torch.ones_like(a))


def test_burn_white_clips_highlights_first_and_only_brightens():
    a, b = _frames()
    out = burn_white(a, b, 0.2)
    assert out.min() >= a.min() and out.max() <= 1.0
    assert (out >= a - 1e-6).all()  # first half only brightens the outgoing shot
    # Red channel (0.8) clips before the zero channels reach it.
    assert out[0].mean() > a[0].mean() and out[1].mean() > 0.0
    emerge = burn_white(a, b, 0.8)
    assert (emerge >= b - 1e-6).all()  # second half settles down onto the incoming shot
    # Monotone: brightness never dips below the min of the neighbours in each half.
    first = [burn_white(a, b, x).mean().item() for x in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
    second = [burn_white(a, b, x).mean().item() for x in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    assert first == sorted(first) and second == sorted(second, reverse=True)
