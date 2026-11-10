"""`custom:` code-drawn shots, effects and transitions (#37): spec, hash, renderer, errors."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from montaj.gpu_policy import GpuPolicy
from montaj.project import file_sha, segment_hash
from montaj.render import custom as C
from montaj.render.canvas import Canvas
from montaj.render.custom import CustomError, check_output
from montaj.render.frame import Renderer
from montaj.spec import SpecError, load_spec
from montaj.timeline import Custom, Trans, resolve

SOLID = "def draw(ctx):\n    c = ctx.params.get('c', 0.25)\n    return ctx.uv[..., 0].new_full((3, ctx.H, ctx.W), c)\n"
INVERT = "def draw(ctx):\n    return 1 - ctx.inputs[0]\n"
WIPE = (
    "def draw(ctx):\n"
    "    a, b = ctx.inputs\n"
    "    m = (ctx.uv[..., 0] < ctx.p * 4 - 2).float()\n"
    "    return a * (1 - m) + b * m\n"
)
VIDEO = "video:\n  size: 64x32\n  fps: 10\n  look: warm-film\n  background: '#000000'\n"


def _project(tmp: Path, shots: str, files: dict[str, str] | None = None) -> Path:
    (tmp / "assets").mkdir(exist_ok=True)
    for rel, body in (files or {}).items():
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_text(body)
    p = tmp / "montaj.yaml"
    p.write_text(VIDEO + "assets: assets\nshots:\n" + shots)
    return p


def _errors(p: Path) -> list[str]:
    with pytest.raises(SpecError) as ei:
        resolve(load_spec(p), p.parent)
    return ei.value.errors


def _shas(tmp: Path) -> dict[str, str]:
    from montaj.pipeline import _asset_shas

    return _asset_shas(load_spec(tmp / "montaj.yaml"), tmp)


def _hashes(tmp: Path) -> list[str]:
    tl = resolve(load_spec(tmp / "montaj.yaml"), tmp)
    shas = _shas(tmp)
    return [segment_hash(tl, seg, shas, "e1") for seg in tl.segments]


THREE = (
    "  - {custom: fx/a.py, hold: 10f, params: {c: 0.25}}\n"
    "  - {custom: fx/b.py, hold: 10f, effects: [{custom: fx/inv.py}]}\n"
    "  - {custom: fx/a.py, hold: 10f, params: {c: 0.75}, in: {type: custom, src: fx/wipe.py, dur: 5f}}\n"
    "  - {custom: fx/b.py, hold: 10f}\n"
)
FILES = {"fx/a.py": SOLID, "fx/b.py": SOLID, "fx/inv.py": INVERT, "fx/wipe.py": WIPE}


# --- spec ----------------------------------------------------------------------------------------


def test_custom_fields_resolve_to_scenes(tmp_path: Path) -> None:
    tl = resolve(load_spec(_project(tmp_path, THREE, FILES)), tmp_path)
    assert [s.kind for s in tl.shots] == ["custom"] * 4
    assert tl.plan(0).scene == Custom(shot=0, t0=0, end=10)
    tr = tl.plan(20).scene
    assert isinstance(tr, Trans) and tr.kind == "custom" and tr.src == "fx/wipe.py" and tr.params == "{}"
    assert tl.plan(25).scene == Custom(shot=2, t0=20, end=30)


@pytest.mark.parametrize(
    ("shots", "files", "want"),
    [
        ("  - {custom: fx/missing.py, hold: 5f}\n", {}, "shots[0].custom: fx/missing.py not found"),
        ("  - {custom: fx/a.txt, hold: 5f}\n", {"fx/a.txt": SOLID}, "is not a .py file"),
        ("  - {custom: fx/a.py, hold: 5f}\n", {"fx/a.py": "x = 1\n\ndef draw(ctx)\n"}, "fx/a.py:3: syntax error"),
        ("  - {custom: fx/a.py, hold: 5f}\n", {"fx/a.py": "x = 1\n"}, "fx/a.py: no top-level draw(ctx)"),
        ("  - {blank: true, hold: 5f, params: {a: 1}}\n", {}, "shots[0].params: params needs custom"),
        ("  - {blank: true, custom: fx/a.py, hold: 5f}\n", {"fx/a.py": SOLID}, "card and custom are mutually exclusive"),
        ("  - {blank: true, hold: 5f}\n  - {blank: true, hold: 5f, in: {type: custom, dur: 2f}}\n", {},
         "shots[1].in.src: required for custom"),
        ("  - {blank: true, hold: 5f}\n  - {blank: true, hold: 5f, in: {type: fade, dur: 2f, src: fx/a.py}}\n",
         {"fx/a.py": SOLID}, "shots[1].in.src: not used by fade"),
        ("  - {custom: fx/a.py, hold: 5f, effects: [{custom: fx/nope.py}]}\n", {"fx/a.py": SOLID},
         "shots[0].effects[0].custom: fx/nope.py not found"),
    ],
)
def test_bad_custom_specs_are_one_line_errors(tmp_path: Path, shots: str, files: dict, want: str) -> None:
    errs = _errors(_project(tmp_path, shots, files))
    assert any(want in e for e in errs), errs


def test_unset_custom_fields_stay_out_of_the_dump(tmp_path: Path) -> None:
    p = _project(tmp_path, "  - {blank: true, hold: 5f}\n  - {blank: true, hold: 5f, in: {type: fade, dur: 2f}}\n")
    spec = load_spec(p)
    dumped = [s.model_dump(mode="json", by_alias=True) for s in spec.shots]
    assert all(k not in d for d in dumped for k in ("custom", "params", "effects"))
    assert "src" not in dumped[1]["in"] and "params" not in dumped[1]["in"]


# --- hash ----------------------------------------------------------------------------------------


def test_editing_one_file_rehashes_exactly_its_segments(tmp_path: Path) -> None:
    _project(tmp_path, THREE, FILES)
    base = _hashes(tmp_path)
    assert len(set(base)) == 4

    def changed(rel: str) -> list[int]:
        path = tmp_path / rel
        old = path.read_text()
        path.write_text(old + "# edited\n")
        try:
            return [i for i, (x, y) in enumerate(zip(base, _hashes(tmp_path))) if x != y]
        finally:
            path.write_text(old)

    assert changed("fx/a.py") == [0, 2]  # shots 0 and 2 draw it
    assert changed("fx/b.py") == [1, 2, 3]  # segment 2 opens on the wipe whose outgoing side is shot 1
    assert changed("fx/inv.py") == [1, 2]  # shot 1's effect rides into the wipe too
    assert changed("fx/wipe.py") == [2]  # only the segment holding the transition window
    assert _hashes(tmp_path) == base


def test_params_and_helper_version_reach_the_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _project(tmp_path, THREE, FILES)
    base = _hashes(tmp_path)
    _project(tmp_path, THREE.replace("c: 0.25", "c: 0.3"), FILES)
    assert [x != y for x, y in zip(base, _hashes(tmp_path))] == [True, False, False, False]
    _project(tmp_path, THREE.replace("dur: 5f}", "dur: 5f, params: {edge: 2}}"), FILES)
    assert [x != y for x, y in zip(base, _hashes(tmp_path))] == [False, False, True, False]  # only the segment holding the wipe
    _project(tmp_path, THREE, FILES)
    monkeypatch.setattr(C, "HELPER_VERSION", C.HELPER_VERSION + 1)
    assert all(x != y for x, y in zip(base, _hashes(tmp_path)))


def test_asset_shas_hold_custom_file_bytes(tmp_path: Path) -> None:
    _project(tmp_path, THREE, FILES)
    shas = _shas(tmp_path)
    assert {k for k in shas if k.startswith("custom:")} == {f"custom:{rel}" for rel in FILES}
    assert shas["custom:fx/a.py"] == file_sha(tmp_path / "fx/a.py")


# --- renderer (CPU canvas) -----------------------------------------------------------------------


def _renderer(tmp: Path, shots: str, files: dict[str, str], policy=None) -> Renderer:
    tl = resolve(load_spec(_project(tmp, shots, files)), tmp)
    return Renderer(tl, tmp, Canvas(tl.size[0], tl.size[1], device="cpu"), policy=policy)


def _raw(r: Renderer, f: int) -> torch.Tensor:
    """The composited scene before the look, so values are exactly what draw returned."""
    return r._eval(r.tl.plan(f).scene, float(f))


def test_custom_shot_effect_and_transition_pixels(tmp_path: Path) -> None:
    r = _renderer(tmp_path, THREE, FILES)
    assert torch.allclose(_raw(r, 0), torch.full((3, 32, 64), 0.25))
    assert torch.allclose(_raw(r, 15), torch.full((3, 32, 64), 0.75))  # 1 - 0.25: the effect inverts shot 1
    mid = _raw(r, 22)  # wipe p = 0.5: left half shot 2 (0.75), right half the inverted shot 1 (0.75)
    first = _raw(r, 20)  # p = 0: all outgoing
    assert torch.allclose(first, torch.full((3, 32, 64), 0.75))
    assert torch.allclose(_raw(r, 25), torch.full((3, 32, 64), 0.75))
    assert mid.shape == (3, 32, 64)
    assert r.frame(0).shape == (3, 32, 64)


def test_transition_sees_outgoing_then_incoming(tmp_path: Path) -> None:
    shots = (
        "  - {custom: fx/a.py, hold: 10f, params: {c: 0.0}}\n"
        "  - {custom: fx/a.py, hold: 10f, params: {c: 1.0}, in: {type: custom, src: fx/wipe.py, dur: 5f}}\n"
    )
    r = _renderer(tmp_path, shots, FILES)
    x = _raw(r, 12)  # p = 0.5: uv.x < 0 is the incoming shot
    assert float(x[0, 0, 0]) == 1.0 and float(x[0, 0, -1]) == 0.0
    assert float(_raw(r, 10).max()) == 0.0


def test_rgba_composites_over_the_background(tmp_path: Path) -> None:
    half = "def draw(ctx):\n    import torch\n    return torch.full((4, ctx.H, ctx.W), 0.5, device=ctx.device)\n"
    r = _renderer(tmp_path, "  - {custom: fx/h.py, hold: 4f}\n", {"fx/h.py": half})
    assert torch.allclose(_raw(r, 0), torch.full((3, 32, 64), 0.5))  # 0.5 premultiplied over black


def test_ctx_clock_progress_and_params(tmp_path: Path) -> None:
    probe = (
        "def draw(ctx):\n"
        "    v = ctx.t * 100 + ctx.p * 10 + ctx.params['k']\n"
        "    return ctx.uv[..., 0].new_full((3, ctx.H, ctx.W), v / 1000)\n"
    )
    r = _renderer(tmp_path, "  - {blank: true, hold: 3f}\n  - {custom: fx/p.py, hold: 11f, params: {k: 1}}\n",
                  {"fx/p.py": probe})
    assert float(_raw(r, 3)[0, 0, 0]) == pytest.approx(0.001)  # t 0, p 0
    assert float(_raw(r, 13)[0, 0, 0]) == pytest.approx((100 + 10 + 1) / 1000)  # t 1 s, p 1 on the last frame


def test_helpers_are_deterministic_and_shaped(tmp_path: Path) -> None:
    cv = Canvas(48, 24, device="cpu")
    runner = C.CustomRunner(tmp_path, cv.W, cv.H, "cpu", 30, None, 0.0)
    ctx = runner.ctx({}, t=0.0, p=0.0, f=0.0, inputs=[])
    assert ctx.uv.shape == (24, 48, 2)
    assert float(ctx.uv[..., 1].abs().max()) == pytest.approx(1 - 1 / 24)  # short side spans [-1, 1]
    assert torch.equal(ctx.noise(3), ctx.noise(3)) and not torch.equal(ctx.noise(3), ctx.noise(4))
    f1 = ctx.fbm(ctx.uv * 4, seed=2)
    assert torch.equal(f1, ctx.fbm(ctx.uv * 4, seed=2)) and 0 <= float(f1.min()) and float(f1.max()) < 1
    d = ctx.sd_circle(ctx.uv, 0.5)
    assert float(ctx.fill(d)[12, 24]) == 1.0 and float(ctx.fill(d)[0, 0]) == 0.0
    assert ctx.palette(ctx.uv[..., 0]).shape == (3, 24, 48)
    img = torch.rand(3, 24, 48)
    assert torch.allclose(ctx.sample(img, ctx.uv), img, atol=1e-5)  # identity warp
    assert ctx.blur(img, 2.0).shape == img.shape
    r, a = ctx.polar(ctx.uv)
    assert r.shape == a.shape == (24, 48)


# --- errors --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "line", "want"),
    [
        ("def draw(ctx):\n    x = 1\n    return x / 0\n", 3, "ZeroDivisionError: division by zero"),
        ("def draw(ctx):\n    import torch\n    return torch.zeros(3, 2, 2)\n", 1, "draw returned shape (3, 2, 2)"),
        ("def draw(ctx):\n    import torch\n    return torch.zeros(3, ctx.H, ctx.W, dtype=torch.float64)\n", 1,
         "draw returned float64, not float32"),
        ("def draw(ctx):\n    return ctx.uv[..., 0].new_full((3, ctx.H, ctx.W), float('nan'))\n", 1,
         "draw returned NaN or inf"),
        ("def draw(ctx):\n    return 0.5\n", 1, "draw returned float, not a tensor"),
        ("import math\nraise ValueError('top')\ndef draw(ctx):\n    pass\n", 2, "ValueError: top"),
    ],
)
def test_bad_file_is_custom_error_with_file_line(tmp_path: Path, body: str, line: int, want: str) -> None:
    r = _renderer(tmp_path, "  - {custom: fx/bad.py, hold: 4f}\n", {"fx/bad.py": body})
    with pytest.raises(CustomError) as ei:
        r.frame(0)
    msg = str(ei.value)
    assert msg.startswith(f"custom fx/bad.py:{line}: ") and want in msg, msg
    assert msg.endswith(")")  # `(<hint>)`
    assert isinstance(ei.value, RuntimeError)  # the CLI prints it as `ERR render: custom fx/bad.py:N: …`


def test_cpu_tensor_on_a_cuda_canvas_is_an_error_not_a_copy() -> None:
    with pytest.raises(CustomError) as ei:
        check_output(torch.zeros(3, 4, 4), "fx/a.py", 1, 4, 4, "cuda")
    assert "draw returned a cpu tensor, the canvas is cuda" in str(ei.value)
    assert "ctx.device" in str(ei.value)


def test_policy_records_a_custom_seam(tmp_path: Path) -> None:
    pol = GpuPolicy(strict=False)
    r = _renderer(tmp_path, "  - {custom: fx/a.py, hold: 4f}\n", FILES, policy=pol)
    r.frame(0)
    assert pol.cpu_stages() == ["custom"]  # a CPU canvas is a fallback
    strict = GpuPolicy(strict=True)
    r = _renderer(tmp_path, "  - {custom: fx/a.py, hold: 4f}\n", FILES, policy=strict)
    from montaj.gpu_policy import GpuStrictError

    with pytest.raises(GpuStrictError):
        r.frame(0)


def test_probe_checks_range_and_determinism(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text(SOLID)
    frames, ms, peak = C.probe(tmp_path, "ok.py", {"c": 0.5}, kind="shot", W=16, H=8, device="cpu", frames=3)
    assert len(frames) == 3 and ms >= 0 and peak == 0
    (tmp_path / "hot.py").write_text("def draw(ctx):\n    return ctx.uv[..., 0].new_full((3, ctx.H, ctx.W), 2.0)\n")
    with pytest.raises(CustomError, match="values outside 0..1"):
        C.probe(tmp_path, "hot.py", None, kind="shot", W=16, H=8, device="cpu")
    (tmp_path / "rand.py").write_text(
        "import torch\ndef draw(ctx):\n    return torch.rand(3, ctx.H, ctx.W, device=ctx.device)\n"
    )
    with pytest.raises(CustomError, match="two calls at p=0.00 differ"):
        C.probe(tmp_path, "rand.py", None, kind="shot", W=16, H=8, device="cpu")
    (tmp_path / "fx.py").write_text(INVERT)
    frames, _, _ = C.probe(tmp_path, "fx.py", None, kind="effect", W=16, H=8, device="cpu", frames=2)
    assert frames[0].shape == (3, 8, 16)


# --- GPU -----------------------------------------------------------------------------------------


@pytest.mark.gpu
def test_strict_gpu_render_of_custom_shots_is_gpu_all(tmp_path: Path) -> None:
    from montaj.pipeline import render

    shots = THREE.replace("hold: 10f", "hold: 6f").replace("dur: 5f", "dur: 3f")
    p = _project(tmp_path, shots, FILES)
    p.write_text(p.read_text().replace("size: 64x32", "size: 360x640"))
    result = render(p, "final", GpuPolicy(strict=True))
    assert result.gpu == "all" and result.rendered == 4
    again = render(p, "final", GpuPolicy(strict=True))
    assert again.cached == 4
    (tmp_path / "fx/wipe.py").write_text(WIPE + "# edited\n")
    third = render(p, "final", GpuPolicy(strict=True))
    assert third.rendered == 1 and third.cached == 3


@pytest.mark.gpu
def test_cpu_tensor_from_draw_on_cuda_is_an_err_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from montaj.cli import main

    body = "import torch\n\ndef draw(ctx):\n    return torch.zeros(3, ctx.H, ctx.W)\n"
    p = _project(tmp_path, "  - {custom: fx/cpu.py, hold: 4f}\n", {"fx/cpu.py": body})
    p.write_text(p.read_text().replace("size: 64x32", "size: 360x640"))
    code = main(["-C", str(tmp_path), "render", "--final"])
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == 1
    assert out.startswith("ERR render: custom fx/cpu.py:3: draw returned a cpu tensor, the canvas is cuda"), out


# --- example recipe and `montaj custom test` -----------------------------------------------------

RECIPE = Path(__file__).resolve().parents[1] / "recipes" / "custom"


def test_custom_recipe_validates_and_every_draw_passes_the_probe(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(RECIPE, tmp_path, dirs_exist_ok=True)
    (tmp_path / "assets").mkdir()
    tl = resolve(load_spec(tmp_path / "montaj.yaml"), tmp_path)
    assert [s.kind for s in tl.shots] == ["custom", "custom"]
    kinds = {"starfield": "shot", "blackhole": "shot", "chroma": "effect", "portal": "transition"}
    for stem, kind in kinds.items():
        frames, _, _ = C.probe(tmp_path, f"fx/{stem}.py", None, kind=kind, W=36, H=64, device="cpu", frames=3)
        assert all(f.shape == (3, 64, 36) for f in frames)


def test_custom_test_command_prints_ok_and_writes_a_strip(tmp_path: Path, capsys, monkeypatch) -> None:
    from montaj.cli import main

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    (tmp_path / "fx").mkdir()
    (tmp_path / "fx/a.py").write_text(SOLID)
    code = main(["-C", str(tmp_path), "custom", "test", "fx/a.py", "--size", "32x16", "--params", '{"c": 0.5}'])
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == 0 and out.startswith("OK fx/a.py ") and out.endswith("strip build/custom-a.jpg"), out
    assert (tmp_path / "build/custom-a.jpg").is_file()
    (tmp_path / "fx/bad.py").write_text("def draw(ctx):\n    return ctx.uv\n")
    code = main(["-C", str(tmp_path), "custom", "test", "fx/bad.py", "--size", "32x16"])
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == 1 and out.startswith("ERR custom: custom fx/bad.py:1: draw returned shape (16, 32, 2)"), out
