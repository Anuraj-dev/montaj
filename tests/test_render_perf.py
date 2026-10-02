"""Wall frames stay pixel-matched to film2 and cheap enough to render.

The wall shot is 16 prints. Sampling each one full-frame dominates `render --final`.
These frames span the recipe camera: close on print 34 (zoom ~1.6), the wide 0.42
hold, and the move back in. Budget is 0.2s/frame; the gate is looser so a shared GPU
can spike without hiding a return to the old ~0.56s path.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
V2 = Path.home() / "Anuraj-dev/montaj-projects/_oracle/v2"
SRC = V2 / "src"
# f=500 still on the 1.606 close-up, 560..670 on the 0.42 wide hold, 700 zooming back.
FRAMES = (500, 530, 560, 620, 670, 700)
WALL_FRAME_LIMIT_S = 0.40


@pytest.fixture(scope="module")
def wall_renderer(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    proj = tmp_path_factory.mktemp("wall-perf")
    assets = proj / "assets"
    assets.mkdir()
    for src in sorted(SRC.glob("*.jpg")):
        shutil.copyfile(src, assets / src.name)
    spec_path = proj / "montaj.yaml"
    shutil.copyfile(ROOT / "recipes" / "birthday-short.yaml", spec_path)

    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer
    from montaj.spec import load_spec
    from montaj.timeline import resolve

    tl = resolve(load_spec(spec_path), spec_path.parent)
    cv = Canvas(1080, 1920, device="cuda")
    renderer = Renderer(tl, spec_path.parent, cv)
    yield renderer
    del renderer, cv
    torch.cuda.empty_cache()


def _zooms(renderer, frames) -> list[float]:
    from montaj.render.wall import wall_camera

    shot = next(s for s in renderer.tl.shots if s.kind == "wall")
    wall = renderer._wall(shot.index)
    cam = shot.spec.wall.camera.model_dump()
    return [wall_camera(wall, cam, f - shot.start)[1] for f in frames]


def test_wall_frame_time(wall_renderer):
    frames = FRAMES
    with torch.no_grad():
        wall_renderer.frame(frames[0])
        torch.cuda.synchronize()
        times = []
        for f in frames:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            wall_renderer.frame(f)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    zooms = _zooms(wall_renderer, frames)
    assert min(zooms) <= 0.42 + 1e-6 and max(zooms) > 1.2, f"zoom span {zooms}"
    assert max(times) < WALL_FRAME_LIMIT_S, (
        f"wall frame budget {WALL_FRAME_LIMIT_S:.2f}s, got "
        + ", ".join(f"f={f} {dt:.3f}s" for f, dt in zip(frames, times))
    )


def test_wall_frame_parity(wall_renderer):
    """Full frames, finish included, against film2.render_frame."""
    if not SRC.is_dir():
        pytest.skip("film2 sources are not on this machine")
    added = str(V2) not in sys.path
    if added:
        sys.path.insert(0, str(V2))
    cwd = os.getcwd()
    try:
        os.chdir(V2)  # film2 opens src/ and depth/ relative to here
        film2 = importlib.import_module("film2")
        with torch.no_grad():
            if getattr(film2, "WALL", None) is None:
                film2.build_wall()
            zooms = _zooms(wall_renderer, FRAMES)
            assert min(zooms) <= 0.42 + 1e-6 and max(zooms) > 1.2, f"zoom span {zooms}"
            for f in FRAMES:
                got = wall_renderer.frame(f)
                want = film2.render_frame(f)
                delta = (got - want).abs()
                mean, mx = float(delta.mean()), float(delta.max())
                assert mean < 1e-3 and mx < 2 / 255, f"f={f}: mean {mean:.3e} max {mx:.3e}"
                del got, want, delta
    finally:
        os.chdir(cwd)
        if added and str(V2) in sys.path:
            sys.path.remove(str(V2))
        torch.cuda.empty_cache()
