"""Keyframes on layer properties (#35): easing curves, key evaluation, resolve, hash."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from montaj.easing import parse_ease
from montaj.project import _shot_payload
from montaj.render.layers import LayerState, layer_state
from montaj.spec import Layer, LayerKey, SpecError, load_spec
from montaj.timeline import resolve
from tests.test_layers import _errors, _project


def e(name: str, x: float) -> float:
    return parse_ease(name)(x)


@pytest.mark.parametrize("name", ["back-in", "back-out", "bounce", "elastic", "spring", "spring(180, 14)",
                                  "cubic-bezier(0.25, 0.1, 0.25, 1)", "steps(4)", "smooth"])
def test_every_easing_starts_at_0_and_ends_at_1(name: str) -> None:
    assert e(name, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert e(name, 1.0) == pytest.approx(1.0, abs=1e-9)


def test_back_overshoots_out_and_undershoots_in() -> None:
    xs = [i / 100 for i in range(101)]
    assert max(e("back-out", x) for x in xs) == pytest.approx(1.1, abs=0.01)  # ~10% overshoot
    assert min(e("back-in", x) for x in xs) < -0.09
    assert e("back-out", 0.5) == pytest.approx(1 + 2.70158 * -0.125 + 1.70158 * 0.25)  # 1.0877


def test_spring_overshoots_then_settles() -> None:
    xs = [i / 200 for i in range(201)]
    ys = [e("spring", x) for x in xs]
    assert max(ys) == pytest.approx(1.16, abs=0.02)  # zeta 0.5: exp(-pi zeta/sqrt(1 - zeta^2)) ~ 16%
    assert max(ys) > 1.1
    assert all(abs(y - 1) < 0.02 for y in ys[int(0.9 * 200):])  # envelope e^(-5x) is under 1% by 0.9
    # Stiffer and heavier damping: no overshoot at critical damping.
    assert max(e("spring(100, 20)", x) for x in xs) < 1.0 + 1e-3
    assert e("spring(100, 10)", 0.3) == pytest.approx(
        1 - math.exp(-1.5) * (math.cos(10 * 0.866025 * 0.3) + 0.5 / 0.866025 * math.sin(10 * 0.866025 * 0.3)),
        abs=0.03)  # closed form of the underdamped response, plus the small end correction


def test_elastic_and_bounce_five_points() -> None:
    assert e("bounce", 0.25) == pytest.approx(7.5625 * 0.25**2)
    assert e("bounce", 0.5) == pytest.approx(7.5625 * (0.5 - 1.5 / 2.75) ** 2 + 0.75)
    assert e("bounce", 0.75) == pytest.approx(7.5625 * (0.75 - 2.25 / 2.75) ** 2 + 0.9375)
    assert max(e("elastic", x / 100) for x in range(101)) > 1.2  # rings past 1
    assert e("elastic", 0.5) == pytest.approx(2**-5 * math.sin((5 - 0.75) * 2 * math.pi / 3) + 1)


def test_cubic_bezier_matches_known_points() -> None:
    # CSS `ease-in-out` is cubic-bezier(.42, 0, .58, 1): symmetric, so exactly 0.5 at the middle.
    assert e("cubic-bezier(.42, 0, .58, 1)", 0.5) == pytest.approx(0.5, abs=1e-6)
    # Linear control points reproduce the identity.
    for x in (0.1, 0.37, 0.8):
        assert e("cubic-bezier(0, 0, 1, 1)", x) == pytest.approx(x, abs=1e-6)
    # CSS `ease` (.25, .1, .25, 1) at x=.5 is about 0.8024 (browser reference value).
    assert e("cubic-bezier(0.25, 0.1, 0.25, 1)", 0.5) == pytest.approx(0.8024, abs=1e-3)
    assert e("cubic-bezier(0.3, -0.5, 0.7, 1.5)", 0.1) < 0  # y control points may leave 0..1


def test_steps_and_bad_names() -> None:
    assert [e("steps(4)", x) for x in (0, 0.24, 0.25, 0.6, 1.0)] == [0, 0, 0.25, 0.5, 1.0]
    for bad in ("wobble", "steps(0)", "steps(2.5)", "spring(0, 3)", "cubic-bezier(2, 0, 0.5, 1)", "cubic-bezier(1, 2)"):
        assert parse_ease(bad) is None


def _layer(**kw) -> Layer:
    return Layer(image="a", **kw)


def test_keys_hold_before_first_and_after_last_and_interpolate() -> None:
    lay = _layer(scale=[LayerKey(t=10, v=0.5, ease="linear"), LayerKey(t=20, v=1.5, ease="linear")])
    assert layer_state(lay, 0.0, 60).scale == 0.5 and layer_state(lay, 10.0, 60).scale == 0.5
    assert layer_state(lay, 15.0, 60).scale == pytest.approx(1.0)
    assert layer_state(lay, 20.0, 60).scale == 1.5 and layer_state(lay, 50.0, 60).scale == 1.5


def test_overshoot_key_leaves_the_range_between_its_values() -> None:
    lay = _layer(scale=[LayerKey(t=0, v=0.6), LayerKey(t=10, v=1.0, ease="back-out")])
    scales = [layer_state(lay, float(t), 60).scale for t in range(0, 12)]
    assert max(scales) > 1.0 and scales[-1] == 1.0


def test_pos_keys_shift_from_the_first_key_and_rotate_and_opacity_are_keyable() -> None:
    lay = _layer(pos=[LayerKey(t=0, v=(540, 2100), ease="linear"), LayerKey(t=10, v=(540, 1500), ease="linear")],
                 rotate=[LayerKey(t=0, v=-20), LayerKey(t=10, v=0, ease="linear")],
                 opacity=[LayerKey(t=0, v=0, ease="linear"), LayerKey(t=10, v=1, ease="linear")])
    s = layer_state(lay, 5.0, 60)
    assert (s.dx, s.dy) == (0.0, pytest.approx(-300.0))
    assert s.opacity == pytest.approx(0.5)
    assert layer_state(lay, 0.0, 60) is None  # opacity 0 is off
    assert layer_state(lay, 12.0, 60) == LayerState(1.0, 0.0, 1.0, 0.0, -600.0)


def test_spring_opacity_is_clamped_to_one() -> None:
    lay = _layer(opacity=[LayerKey(t=0, v=0.2), LayerKey(t=20, v=1.0, ease="spring")])
    assert max(layer_state(lay, float(t), 60).opacity for t in range(1, 21)) == 1.0


def test_resolve_turns_key_times_into_frames(tmp_path: Path) -> None:
    shots = ("  - {hold: 2s, layers: [{clip: rec}, {image: tag, pos: [54, 200], "
             "scale: [{t: 0f, v: 0.6}, {t: 0.5s, v: 1, ease: spring}]}]}\n")
    tl = resolve(load_spec(_project(tmp_path, shots)), tmp_path)
    keys = tl.shots[0].spec.layers[1].scale
    assert [k.t for k in keys] == [0, 15] and keys[1].ease == "spring"


def test_unset_key_lists_leave_the_hash_payload_alone(tmp_path: Path) -> None:
    tl = resolve(load_spec(_project(tmp_path, "  - {hold: 1s, layers: [{clip: rec}, {image: tag, scale: 2}]}\n")), tmp_path)
    assert _shot_payload(tl.shots[0].spec)["layers"] == [{"clip": "rec"}, {"image": "tag", "scale": 2.0}]


def test_validation_errors(tmp_path: Path) -> None:
    def err(layer: str) -> str:
        return "\n".join(_errors(tmp_path, f"  - {{hold: 1s, layers: [{{clip: rec}}, {{image: tag, {layer}}}]}}\n"))

    assert "unknown 'wobble'" in err("scale: [{t: 0f, v: 1}, {t: 5f, v: 2, ease: wobble}]")
    assert "not after the previous key" in err("scale: [{t: 5f, v: 1}, {t: 5f, v: 2}]")
    assert "needs a number" in err("scale: [{t: 0f, v: [1, 2]}]")
    assert "needs [x, y]" in err("pos: [{t: 0f, v: 3}]")
    assert "after the end of the 30f hold" in err("scale: [{t: 0f, v: 1}, {t: 2s, v: 2}]")
    assert "got -1" in err("scale: [{t: 0f, v: 1}, {t: 5f, v: -1}]")
    assert "0 to 1" in err("opacity: [{t: 0f, v: 1.5}]")


@pytest.mark.gpu
def test_strict_keyframed_layer_film_is_all_gpu(tmp_path: Path) -> None:
    from montaj.gpu_policy import GpuPolicy
    from montaj.pipeline import render
    from tests.test_layers import _gpu_film

    path = _gpu_film(tmp_path)
    path.write_text(path.read_text().replace(
        "scale: 6, rotate: -8,",
        "scale: [{t: 0f, v: 2}, {t: 10f, v: 6, ease: spring}], rotate: [{t: 0f, v: -30}, {t: 10f, v: -8, ease: back-out}],"
    ).replace("pos: [540, 400]", "pos: [{t: 0f, v: [540, 900]}, {t: 10f, v: [540, 400], ease: 'cubic-bezier(0.3, 0, 0.2, 1)'}]"))
    pol = GpuPolicy(strict=True)
    result = render(path, "final", pol)
    assert result.gpu == "all" and result.rendered > 0
