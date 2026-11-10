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


# ---------------------------------------------------------------- anchors


def _anchored(tmp: Path, shots: str, header: str = "  bpm: 120\n") -> Path:
    path = _project(tmp, shots)
    path.write_text(path.read_text().replace("  look: warm-film\n", "  look: warm-film\n" + header))
    return path


def test_beat_anchor_snaps_to_the_grid_and_leaves_no_trace(tmp_path: Path) -> None:
    shots = ("  - {hold: 2s, layers: [{clip: rec}]}\n"
             "  - {hold: 2s, layers: [{clip: rec}, {image: tag, scale: [{on: beat, t: 0.4s, v: 1}, {t: 20f, v: 2}]}]}\n")
    tl = resolve(load_spec(_anchored(tmp_path, shots)), tmp_path)
    keys = tl.shots[1].spec.layers[1].scale
    assert [k.t for k in keys] == [15, 20]  # 120 bpm = a beat per 15f; shot start 60f is on a beat, 0.4s = 12f -> 15f
    assert all(k.on is None for k in keys)
    assert "on" not in keys[0].model_dump(exclude_none=True) or keys[0].on is None


def test_word_anchor_follows_the_word_and_checks_it_exists(tmp_path: Path) -> None:
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "song.wav").write_bytes(b"RIFF")
    (tmp_path / "music" / "markers.json").write_text('{"words": [{"i": 3, "w": "hey", "s": 0.5, "e": 0.9}]}')
    audio = "audio:\n  track: music/song.wav\n  markers: music/markers.json\n"
    shots = "  - {hold: 2s, layers: [{clip: rec}, {image: tag, rotate: [{on: word:3, t: 0.2s, v: 0}, {t: 40f, v: 9}]}]}\n"
    path = _anchored(tmp_path, shots, header="")
    path.write_text(path.read_text() + audio)
    tl = resolve(load_spec(path), tmp_path)
    assert [k.t for k in tl.shots[0].spec.layers[1].rotate] == [21, 40]  # 0.5s = 15f, plus 0.2s = 6f
    path.write_text(path.read_text().replace("word:3", "word:9"))
    from montaj.spec import SpecError

    with pytest.raises(SpecError) as ei:
        load_spec(path)
    assert any("unknown word 9" in e for e in ei.value.errors)


def test_anchor_errors(tmp_path: Path) -> None:
    def err(key: str, header: str = "  bpm: 120\n") -> str:
        shots = f"  - {{hold: 2s, layers: [{{clip: rec}}, {{image: tag, scale: [{key}]}}]}}\n"
        from montaj.spec import SpecError

        with pytest.raises(SpecError) as ei:
            load_spec(_anchored(tmp_path, shots, header))
        return "\n".join(ei.value.errors)

    assert "use beat or word:N" in err("{on: sometime, v: 1}")
    assert "bpm" in err("{on: beat, v: 1}", header="")
    assert "needs audio.markers" in err("{on: 'word:1', v: 1}")


def _pair(**child) -> list[Layer]:
    return [Layer(image="a", id="p", rotate=[LayerKey(t=0, v=0), LayerKey(t=10, v=90)]),
            Layer(image="b", parent="p", **child)]


def test_child_inherits_the_parent_rotation_about_the_parent_centre() -> None:
    layers = _pair()
    from montaj.render.layers import world_state

    centres = {0: (0.0, 0.0), 1: (100.0, 0.0)}
    s = world_state(layers, 1, 10, 30, centres.__getitem__)
    assert s.rotate == pytest.approx(90)
    # a quarter turn clockwise on screen carries the child from right of the parent to below it
    assert (s.dx, s.dy) == pytest.approx((-100.0, 100.0), abs=1e-6)
    assert world_state(layers, 1, 0, 30, centres.__getitem__).dx == pytest.approx(0.0)


def test_anchor_pivots_a_rotation_off_the_centre() -> None:
    layer = Layer(image="a", anchor=(100.0, 0.0), rotate=[LayerKey(t=0, v=0), LayerKey(t=10, v=90)])
    s = layer_state(layer, 10, 30)
    assert (s.dx, s.dy) == pytest.approx((100.0, -100.0), abs=1e-6)  # clockwise about a pivot on its right: the centre swings up


def test_parent_errors(tmp_path: Path) -> None:
    def err(layers: str) -> str:
        return "\n".join(_errors(tmp_path, f"  - {{hold: 1s, layers: [{layers}]}}\n"))

    assert "no layer has id" in err("{image: tag, parent: nope}")
    assert "loop" in err("{image: tag, id: a, parent: b}, {image: tag, id: b, parent: a}")
    assert "used twice" in err("{image: tag, id: a}, {image: tag, id: a}")


def test_unset_parent_fields_leave_hashes_alone() -> None:
    dumped = Layer(image="a").model_dump_json()
    assert not any(f in dumped for f in ('"parent"', '"anchor"', '"id"'))


def test_inspect_track_prints_the_curve(tmp_path: Path) -> None:
    from montaj import inspector

    _project(tmp_path, "  - {hold: 1s, layers: [{clip: rec}, {image: tag, id: s, scale: [{t: 0f, v: 1}, {t: 10f, v: 2}]}]}\n")
    for layer in ("1", "s"):
        out = inspector.track(tmp_path, f"0.{layer}.scale")
        assert out["keys"] == [0, 10] and out["values"][0] == 1.0 and out["values"][-1] == 2.0
    assert inspector.lines(out)[0].startswith("OK track 0.s.scale keys=[0, 10] 0f=1.0")
    with pytest.raises(ValueError, match="not keyed"):
        inspector.track(tmp_path, "0.0.opacity")
