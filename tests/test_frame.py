"""Full-frame parity with film2.render_frame on the birthday recipe.

Photos are copied byte-for-byte. Ingest re-encodes and would move pixels.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
V2 = Path.home() / "Anuraj-dev/montaj-projects/_oracle/v2"
SRC = V2 / "src"

# Integer frames that land on each moment film2 is supposed to match.
FRAMES = (
    (10, "still"),
    (70, "swirl"),
    (152, "chain-flash"),
    (225, "whip-x"),
    (352, "fade-38-29"),
    (391, "whip-y"),
    (457, "fade-wall"),
    (620, "wall-flight"),
    (800, "outro-glow"),
)


def test_birthday_frames_match_film2(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    assets = tmp_path / "assets"
    assets.mkdir()
    for src in sorted(SRC.glob("*.jpg")):
        shutil.copyfile(src, assets / src.name)
    spec_path = tmp_path / "montaj.yaml"
    shutil.copyfile(ROOT / "recipes" / "birthday-short.yaml", spec_path)

    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer
    from montaj.spec import load_spec
    from montaj.timeline import resolve

    tl = resolve(load_spec(spec_path), spec_path.parent)
    assert tl.n_frames == 840 and tl.size == (1080, 1920)

    added = str(V2) not in sys.path
    if added:
        sys.path.insert(0, str(V2))
    cwd = os.getcwd()
    try:
        os.chdir(V2)  # film2 opens src/<stem>.jpg and depth/<stem>.png relative to here
        film2 = importlib.import_module("film2")
        with torch.no_grad():
            film2.build_wall()
            cv = Canvas(tl.size[0], tl.size[1], device="cuda")
            renderer = Renderer(tl, spec_path.parent, cv)
            for f, name in FRAMES:
                got = renderer.frame(f)
                want = film2.render_frame(f)
                delta = (got - want).abs()
                mean, mx = float(delta.mean()), float(delta.max())
                assert mean < 1e-3 and mx < 2 / 255, f"{name} f={f}: mean {mean:.3e} max {mx:.3e}"
                del got, want, delta
            del renderer, cv
    finally:
        os.chdir(cwd)
        if added and str(V2) in sys.path:
            sys.path.remove(str(V2))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
