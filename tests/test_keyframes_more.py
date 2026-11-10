"""#35 part 2: shared easings on every keyed spot, keyable box/radius/shadow."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from montaj.render.core import EASE
from montaj.render.layers import layer_state, sample
from montaj.spec import Layer, LayerKey, load_spec
from montaj.timeline import resolve
from tests.test_layers import _errors, _project, _renderer


def test_core_ease_builds_the_new_names_and_still_rejects_bad_ones() -> None:
    assert EASE["linear"](0.25) == 0.25
    assert EASE["back-out"](0.5) == pytest.approx(1.0877, abs=1e-3)
    assert EASE["steps(4)"](0.6) == 0.5
    assert EASE["cubic-bezier(0, 0, 1, 1)"](0.3) == pytest.approx(0.3, abs=1e-6)
    with pytest.raises(KeyError):
        EASE["wobble"]
    assert "back-out" not in EASE  # lookups build on demand; the table itself is never written to


def _film(tmp: Path, ease: str) -> Path:
    path = _project(tmp, "  - {hold: 1s, layers: [{clip: rec}]}\n")
    path.write_text(path.read_text().replace(
        "shots:", f"""tracks:
  dust: [{{t: 0s, v: 0.2}}, {{t: 1s, v: 0.8, ease: {ease}}}]
overlays:
  - {{shape: ring, from: 0s, to: 1s, center: [540, 900], radius: [100, 300], ease: {ease}}}
shots:"""))
    return path


def test_tracks_overlays_accept_the_new_easings(tmp_path: Path) -> None:
    tl = resolve(load_spec(_film(tmp_path, "back-out")), tmp_path)
    assert tl._tracks["dust"][1][2](0.5) == pytest.approx(1.0877, abs=1e-3)
    assert tl.overlays[0].ease == "back-out"


def test_bad_easing_is_rejected_on_tracks_and_overlays(tmp_path: Path) -> None:
    from montaj.spec import SpecError

    with pytest.raises(SpecError) as ei:
        load_spec(_film(tmp_path, "wobble"))
    assert sum("unknown 'wobble'" in e for e in ei.value.errors) == 2


def test_camera_and_wall_keys_accept_the_new_easings(tmp_path: Path) -> None:
    shots = "  - {clip: rec, hold: 1s, camera: [{t: 0f, zoom: 1}, {t: 20f, zoom: 1.5, ease: spring}]}\n"
    load_spec(_project(tmp_path, shots))
    bad = _errors(tmp_path, "  - {clip: rec, hold: 1s, camera: [{t: 0f, zoom: 1}, {t: 20f, zoom: 1.5, ease: wobble}]}\n")
    assert any("unknown 'wobble'" in e for e in bad)


def _layer(**kw) -> Layer:
    return Layer(image="a", **kw)


def test_radius_and_shadow_are_keyable() -> None:
    lay = _layer(radius=[LayerKey(t=0, v=0, ease="linear"), LayerKey(t=10, v=40, ease="linear")],
                 shadow=[LayerKey(t=0, v=0.2), LayerKey(t=10, v=1.0, ease="linear")])
    assert sample(lay.radius, 5.0, 0.0) == pytest.approx(20.0)
    assert sample(lay.shadow, 10.0, 0.0) == 1.0
    assert layer_state(lay, 5.0, 60) is not None


def test_box_keys_sample_to_a_rect() -> None:
    lay = _layer(box=[LayerKey(t=0, v=(0, 0, 100, 200), ease="linear"), LayerKey(t=10, v=(0, 0, 300, 400), ease="linear")])
    assert sample(lay.box, 5.0, None) == pytest.approx((0, 0, 200, 300))


def test_keyed_box_validation(tmp_path: Path) -> None:
    def err(layer: str) -> str:
        return "\n".join(_errors(tmp_path, f"  - {{hold: 1s, layers: [{{clip: rec}}, {{image: tag, {layer}}}]}}\n"))

    assert "needs [x0, y0, x1, y1]" in err("box: [{t: 0f, v: [0, 0, 5, 5]}, {t: 5f, v: [1, 2]}]")
    assert "is not a box" in err("box: [{t: 0f, v: [0, 0, 5, 5]}, {t: 5f, v: [9, 2, 3, 4]}]")
    assert "fixed box" in err("fit: contain, box: [{t: 0f, v: [0, 0, 5, 5]}, {t: 5f, v: [0, 0, 9, 9]}]")
    assert "0 to 1" in err("shadow: [{t: 0f, v: 2}]")
    assert "got -3" in err("radius: [{t: 0f, v: -3}]")


def test_keyed_box_grows_the_layer_on_screen(tmp_path: Path) -> None:
    shots = """  - hold: 20f
    layers:
      - {clip: rec}
      - {image: tag, box: [{t: 0f, v: [0, 0, 200, 200], ease: linear}, {t: 10f, v: [0, 0, 800, 800]}], radius: [{t: 0f, v: 0}, {t: 10f, v: 60}]}
"""
    rend, tl = _renderer(tmp_path, shots)
    scene = tl.plan(0).scene
    small, big = rend._layers_frame(scene, 0.0), rend._layers_frame(scene, 10.0)
    assert not torch.equal(small, big)
    k = rend.cv.k
    assert rend._dest_at(tl.shots[0].spec.layers[1], rend._layer_plans(0)[1], 10.0) == pytest.approx((0, 0, 800 * k, 800 * k))
