"""#62 item 7: the Auren rewrite's mascot — one `frames:` layer replacing one image layer per 0.1 s — shows the
same image on every film frame as the per-image layers it replaces (the segment hash differs; the picture does not)."""
from __future__ import annotations

from pathlib import Path

from montaj.project import frame_index
from montaj.render.layers import _in_window
from montaj.spec import load_spec
from montaj.timeline import resolve
from tests.test_i59_frames_layer import N, _project

FPS = 60
STEP = 0.1  # seconds per image, as the Auren mascot flipbook
EACH = "".join(f"      - {{image: m{i:02d}, from: {i * STEP:.1f}s, to: {(i + 1) * STEP:.1f}s, pos: [54, 96]}}\n"
               for i in range(N))
ONE = f"      - {{frames: \"m*.png\", every: {STEP}s, from: 0s, to: {N * STEP:.1f}s, pos: [54, 96]}}\n"


def _layers(tmp: Path, layers: str):
    p = _project(tmp, f"  - hold: 1s\n    layers:\n{layers}", fps=FPS)
    tl = resolve(load_spec(p), tmp)
    return tl.shots[0].spec.layers


def test_one_frames_layer_shows_what_the_image_layers_showed(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    each, (seq,) = _layers(tmp_path / "a", EACH), _layers(tmp_path / "b", ONE)
    shown = []
    for f in range(-5, FPS + 5):  # past both ends: nothing shows outside the window
        want = next((ly.image for ly in each if _in_window(ly, f)), None)
        i = frame_index(seq, f, N, FPS)
        got = None if i is None else f"m{i:02d}"
        assert got == want, f
        shown.append(got)
    assert [f"m{i:02d}" for i in range(N)] == sorted({s for s in shown if s})
