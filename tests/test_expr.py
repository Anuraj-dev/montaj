"""#52: safe `expr` on keyable layer properties, wider `inspect --track`, keyed-box cover re-crop."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from montaj import expr as xp
from montaj import inspector
from montaj.render.layers import layer_state, sample
from montaj.spec import LayerExpr
from montaj.timeline import resolve
from montaj.spec import load_spec
from tests.test_layers import _errors, _project, _renderer

ENV = {"frame": 12.0}


def ev(text: str, **env: float) -> float:
    return xp.evaluate(text, {**ENV, **env})


def test_evaluator_arithmetic_functions_and_conditionals() -> None:
    assert ev("1 + 2 * 3 - 4 / 2") == 5.0
    assert ev("-frame % 5 + 2 ** 3") == 11.0
    assert ev("clamp(frame / 6, 0, 1.5)") == 1.5
    assert ev("lerp(10, 20, 0.25) + min(3, 4, 2) + max(1, 2) + abs(-1)") == 17.5
    assert ev("sin(0) + cos(0)") == 1.0
    assert ev("1 if frame > 10 else 0") == 1.0 and ev("1 if frame < 10 else 0") == 0.0
    assert ev("1 < frame < 20") == 1.0 and ev("1 < frame < 5") == 0.0


def test_noise_is_deterministic_and_bounded() -> None:
    vals = [ev(f"noise(7, {x / 7})") for x in range(200)]
    assert vals == [ev(f"noise(7, {x / 7})") for x in range(200)]
    assert all(-1.0 <= v <= 1.0 for v in vals) and len(set(vals)) > 50
    assert ev("noise(1, 3)") != ev("noise(2, 3)")


@pytest.mark.parametrize("text,token", [
    ("__import__('os').system('x')", "__import__"),
    ("().__class__", "attribute access"),
    ("frame.real", "attribute access"),
    ("[1, 2][0]", "indexing"),
    ("(lambda: 1)()", "lambda"),
    ("[x for x in (1, 2)]", "ListComp"),
    ("open('f')", "'open'"),
    ("foo + 1", "'foo'"),
    ("'abc'", "not a number"),
    ("sin(1, 2)", "sin() takes 1"),
    ("word(frame)", "word(frame)"),
    ("2 ** 100", "too large"),
    ("10 ** 999999999", "too large"),
    ("1 +", "syntax error"),
    ("frame and 1", "BoolOp"),
    ("1e30", "out of range"),
    ("x" * 500, "too long"),
])
def test_unsafe_or_bad_expressions_are_rejected_naming_the_token(text: str, token: str) -> None:
    with pytest.raises(xp.ExprError) as ei:
        xp.validate(text)
    assert token in str(ei.value)


def test_runtime_guards_for_computed_exponents_and_division() -> None:
    with pytest.raises(xp.ExprError, match="too large"):
        ev("9 ** 9 ** 9")
    with pytest.raises(xp.ExprError):
        ev("1 / (frame - 12)")
    with pytest.raises(xp.ExprError):
        ev("frame % 0")


def test_bind_folds_constants_so_only_frame_remains() -> None:
    bound = xp.bind("t * 2 + beat + word(3) + fps + bpm", fps=30, bpm=120, start=60, beat0=0, word_frames={3: 90})
    # frame 15 on the shot clock: t = 0.5 s, beat = (15 + 60) / 15 = 5, word 3 = (90 - 60) / 30 = 1 s
    assert xp.evaluate(bound, {"frame": 15.0}) == pytest.approx(1.0 + 5.0 + 1.0 + 30 + 120)
    assert xp.words("word(3) + word(1)") == [1, 3]
    with pytest.raises(xp.ExprError, match="beat needs video.bpm"):
        xp.bind("beat", fps=30, bpm=None, start=0, beat0=0, word_frames={})


def _shots(prop: str) -> str:
    return f"  - {{hold: 1s, layers: [{{clip: rec}}, {{image: tag, {prop}}}]}}\n"


def test_expr_loads_resolves_and_evaluates_per_frame(tmp_path: Path) -> None:
    path = _project(tmp_path, _shots("pos: {expr: ['54 + 20 * sin(t * 6)', '96']}, scale: {expr: 1 + t}, "
                                     "opacity: {expr: 'clamp(frame / 10, 0, 1)'}"))
    tl = resolve(load_spec(path), tmp_path)
    layer = tl.shots[0].spec.layers[1]
    assert isinstance(layer.scale, LayerExpr) and "t" not in layer.scale.expr.replace("frame", "")
    st = layer_state(layer, 15.0, 30)
    assert st.scale == pytest.approx(1.5) and st.opacity == 1.0
    assert st.dx == pytest.approx(20 * math.sin(3.0))
    assert layer_state(layer, 0.0, 30) is None  # opacity 0 hides the layer


def test_expr_changes_the_segment_hash(tmp_path: Path) -> None:
    a = resolve(load_spec(_project(tmp_path, _shots("scale: {expr: 1 + t}"))), tmp_path).shots[0].spec
    b = resolve(load_spec(_project(tmp_path, _shots("scale: {expr: 1 + 2 * t}"))), tmp_path).shots[0].spec
    assert a.model_dump_json() != b.model_dump_json()


@pytest.mark.parametrize("prop,token", [
    ("scale: {expr: 'foo + 1'}", "'foo'"),
    ("scale: {expr: '__import__(1)'}", "__import__"),
    ("scale: {expr: 'frame.real'}", "attribute access"),
    ("scale: {expr: '2 ** 99'}", "too large"),
    ("scale: {expr: 'beat'}", "video.bpm"),
    ("scale: {expr: '1 / (frame - 0)'}", "division by zero"),
    ("pos: {expr: '5'}", "needs [x, y]"),
    ("scale: {expr: ['1', '2']}", "needs one expression"),
    ("box: {expr: ['0', '0', '9']}", "needs [x0, y0, x1, y1]"),
])
def test_bad_expr_gives_a_line_naming_the_token(tmp_path: Path, prop: str, token: str) -> None:
    errs = _errors(tmp_path, _shots(prop))
    assert any(("expr" in e or "bpm" in token) and token in e for e in errs), errs


def test_expr_word_needs_audio_markers(tmp_path: Path) -> None:
    errs = _errors(tmp_path, _shots("scale: {expr: '1 + word(2)'}"))
    assert any("word:2 needs audio.markers" in e for e in errs), errs


def test_expr_beat_uses_the_grid(tmp_path: Path) -> None:
    path = _project(tmp_path, _shots("scale: {expr: 'beat'}"))
    path.write_text(path.read_text().replace("  look: warm-film\n", "  look: warm-film\n  bpm: 120\n"))
    layer = resolve(load_spec(path), tmp_path).shots[0].spec.layers[1]
    assert sample(layer.scale, 30.0, 1.0) == pytest.approx(2.0)  # 120 bpm = a beat per 15f


def test_inspect_track_samples_expr_tracks_and_camera(tmp_path: Path) -> None:
    path = _project(tmp_path, _shots("scale: {expr: 1 + frame / 10}")
                    + "  - {clip: rec, hold: 1s, camera: [{t: 0f, zoom: 1}, {t: 20f, zoom: 2, ease: linear}]}\n")
    path.write_text(path.read_text().replace("shots:", "tracks:\n  dust: [{t: 0s, v: 0}, {t: 1s, v: 1, ease: linear}]\nshots:"))
    out = inspector.track(tmp_path, "0.1.scale", samples=30)
    assert len(out["frames"]) == 30 and out["keys"] == [] and out["values"][0] == 1.0
    assert out["values"][-1] == pytest.approx(1 + out["frames"][-1] / 10, abs=1e-3)
    dust = inspector.track(tmp_path, "tracks.dust", samples=4)
    assert dust["frames"] == [0, 10, 20, 30] and dust["values"] == [0.0, 0.3333, 0.6667, 1.0]
    zoom = inspector.track(tmp_path, "1.camera.zoom", samples=3)
    assert zoom["frames"] == [0, 10, 20] and zoom["values"][0] == 1.0 and zoom["values"][-1] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="no camera focus keys"):
        inspector.track(tmp_path, "1.camera.focus")
    with pytest.raises(ValueError, match="--samples"):
        inspector.track(tmp_path, "0.1.scale", samples=1)


def test_keyed_box_recrops_to_keep_the_source_aspect(tmp_path: Path) -> None:
    shots = ("  - hold: 20f\n    layers:\n      - {clip: rec}\n"
             "      - {image: tag, box: [{t: 0f, v: [0, 0, 400, 400], ease: linear}, {t: 10f, v: [0, 0, 400, 800]}]}\n")
    path = _project(tmp_path, shots, size="180x320")
    tl = resolve(load_spec(path), tmp_path)
    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer

    rend = Renderer(tl, tmp_path, Canvas(180, 320, device="cpu"))
    layer, plan = tl.shots[0].spec.layers[1], rend._layer_plans(0)[1]
    for local in (0.0, 5.0, 10.0):
        dest = rend._dest_at(layer, plan, local)
        x0, y0, x1, y1 = rend._src_at(layer, plan, dest)
        assert (x1 - x0) / (y1 - y0) == pytest.approx((dest[2] - dest[0]) / (dest[3] - dest[1]), rel=1e-4)
    a, b = rend._src_at(layer, plan, rend._dest_at(layer, plan, 0.0)), rend._src_at(layer, plan, rend._dest_at(layer, plan, 10.0))
    assert a != b
    assert rend._layers_frame(tl.plan(0).scene, 10.0).shape == (3, 320, 180)
