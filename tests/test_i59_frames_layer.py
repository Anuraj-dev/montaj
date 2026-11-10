"""#59: an image sequence (PNG frames with alpha) as one `frames:` layer."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from montaj.analysis import describe_all
from montaj.pipeline import _asset_shas
from montaj.project import _TRAIL_FRAMES, find_frames, frame_index, segment_hash
from montaj.render.canvas import Canvas
from montaj.render.frame import Renderer
from montaj.spec import Layer, SpecError, load_spec
from montaj.timeline import resolve

N = 5


def _frame(i: int, size=(40, 20)) -> Image.Image:
    """RGBA: a block whose colour and x position change with `i`; the rest transparent."""
    arr = np.zeros((size[1], size[0], 4), np.uint8)
    x = 4 * i
    arr[:, x:x + 10] = (min(255, 40 * i + 30), max(0, 200 - 30 * i), 90, 255)
    arr[:2, :] = (10, 10, 10, 128)  # half-transparent strip: alpha must survive
    return Image.fromarray(arr, "RGBA")


def _project(tmp: Path, shots: str, *, fps: int = 30) -> Path:
    assets = tmp / "assets"
    (assets / "seq").mkdir(parents=True, exist_ok=True)
    for i in range(N):
        _frame(i).save(assets / "seq" / f"{i:03d}.png")
        _frame(i).save(assets / f"m{i:02d}.png")
    (assets / "bg.png").exists() or Image.new("RGB", (64, 64), (20, 40, 60)).save(assets / "bg.png")
    path = tmp / "montaj.yaml"
    path.write_text(f"video:\n  size: 108x192\n  fps: {fps}\n  look: warm-film\nassets: assets\nshots:\n{shots}\n")
    return path


def _render(tmp: Path, shots: str, **kw):
    tl = resolve(load_spec(_project(tmp, shots, **kw)), tmp)
    return Renderer(tl, tmp, Canvas(tl.size[0], tl.size[1], device="cpu")), tl


def _errors(tmp: Path, shots: str) -> list[str]:
    with pytest.raises(SpecError) as ei:
        load_spec(_project(tmp, shots))
    return ei.value.errors


# ---------------------------------------------------------------- frame index


def _lay(**kw) -> Layer:
    return Layer(frames="seq", **kw)


def test_frame_index_steps_and_end_modes() -> None:
    assert [frame_index(_lay(), t, 3, 30) for t in range(5)] == [0, 1, 2, 2, 2]  # default: hold the last
    assert [frame_index(_lay(loop=True), t, 3, 30) for t in range(5)] == [0, 1, 2, 0, 1]
    assert [frame_index(_lay(once=True), t, 3, 30) for t in range(5)] == [0, 1, 2, None, None]
    assert [frame_index(_lay(every=2), t, 9, 30) for t in range(6)] == [0, 0, 1, 1, 2, 2]
    assert [frame_index(_lay(fps=10), t, 9, 30) for t in range(7)] == [0, 0, 0, 1, 1, 1, 2]
    assert [frame_index(_lay(start=2, from_=3, to=6), t, 9, 30) for t in range(7)] == [None, None, None, 2, 3, 4, None]
    # Motion-blur sub-samples (+-0.25 frame) show their frame's image.
    assert {frame_index(_lay(), 3 + d, 9, 30) for d in (-0.25, 0.0, 0.25)} == {3}
    # Before an unset `from` (incoming transition) it shows the start image.
    assert frame_index(_lay(start=1), -4.0, 9, 30) == 1


def test_trail_default_matches_the_renderer() -> None:
    from montaj.render.effects import TRAIL_FRAMES

    assert _TRAIL_FRAMES == TRAIL_FRAMES


def test_frames_folder_and_glob_in_name_order(tmp_path: Path) -> None:
    d = tmp_path / "a" / "s"
    d.mkdir(parents=True)
    for n in ("f10.png", "f2.png", "f1.png", ".hidden.png", "notes.txt"):
        (d / n).write_bytes(b"x")
    assert [p.name for p in find_frames(tmp_path / "a", "s")] == ["f1.png", "f2.png", "f10.png"]
    assert [p.name for p in find_frames(tmp_path / "a", "s/f1*.png")] == ["f1.png", "f10.png"]
    with pytest.raises(FileNotFoundError, match="have: s"):
        find_frames(tmp_path / "a", "nope")


# ---------------------------------------------------------------- validate


@pytest.mark.parametrize(
    ("layer", "needle"),
    [
        ("{frames: nope}", "no folder 'nope'"),
        ("{frames: seq, image: bg}", "exactly one of clip or image (or frames)"),
        ("{image: bg, fps: 12}", "fps needs frames"),
        ("{image: bg, loop: true}", "loop needs frames"),
        ("{frames: seq, fps: 12, every: 2f}", "fps and every are mutually exclusive"),
        ("{frames: seq, loop: true, once: true}", "loop, hold and once are mutually exclusive"),
        ("{frames: seq, start: 5}", "past the last image (5 frames)"),
        ("{frames: seq, fps: 0}", "must be > 0"),
        ("{frames: seq, mask: subject}", "a frames layer has no matte"),
    ],
)
def test_frames_errors_are_helpful(tmp_path: Path, layer: str, needle: str) -> None:
    errs = _errors(tmp_path, f"  - {{hold: 10f, layers: [{layer}]}}\n")
    assert any(needle in e for e in errs), errs


def test_frames_of_mixed_sizes_are_an_error(tmp_path: Path) -> None:
    _project(tmp_path, "  - {photo: bg, hold: 1f}\n")
    _frame(9, size=(30, 20)).save(tmp_path / "assets" / "seq" / "009.png")
    errs = _errors(tmp_path, "  - {hold: 10f, layers: [{frames: seq}]}\n")
    assert any("009.png is 30x20, the first frame is 40x20" in e for e in errs), errs


def test_unset_frames_fields_stay_out_of_the_dump(tmp_path: Path) -> None:
    from montaj.project import _shot_payload

    tl = resolve(load_spec(_project(tmp_path, "  - {hold: 10f, layers: [{image: bg}, {frames: seq, every: 0.1s}]}\n")),
                 tmp_path)
    assert _shot_payload(tl.shots[0].spec)["layers"] == [{"image": "bg"}, {"frames": "seq", "every": 3}]


# ---------------------------------------------------------------- render


# The Auren run's 35 lines, shrunk: one image layer per frame, each shown for `every` film frames.
def _per_frame_layers(every: int, pos="[540, 900]", scale=6) -> str:
    return "".join(f"      - {{image: m{i:02d}, from: {i * every}f, to: {(i + 1) * every}f, pos: {pos}, scale: {scale}}}\n"
                   for i in range(N))


def test_one_frames_line_renders_the_per_frame_layers(tmp_path: Path) -> None:
    every = 2
    hold = N * every
    old = f"  - hold: {hold}f\n    layers:\n      - {{image: bg}}\n" + _per_frame_layers(every)
    new = (f"  - hold: {hold}f\n    layers:\n      - {{image: bg}}\n"
           f"      - {{frames: seq, every: {every}f, pos: [540, 900], scale: 6}}\n")
    a, tla = _render(tmp_path, old)
    b, tlb = _render(tmp_path, new)
    for f in range(hold):
        assert torch.equal(a.frame(f), b.frame(f)), f
    # And the frames really move: consecutive images differ.
    assert not torch.equal(b.frame(0), b.frame(every))


def test_once_disappears_and_loop_wraps(tmp_path: Path) -> None:
    bare, _ = _render(tmp_path, "  - hold: 8f\n    layers:\n      - {image: bg}\n")
    once, _ = _render(tmp_path, "  - hold: 8f\n    layers:\n      - {image: bg}\n"
                                "      - {frames: seq, once: true, pos: [540, 900], scale: 6}\n")
    loop, _ = _render(tmp_path, "  - hold: 8f\n    layers:\n      - {image: bg}\n"
                                "      - {frames: seq, loop: true, pos: [540, 900], scale: 6}\n")
    assert not torch.equal(once.frame(4), bare.frame(4))
    assert torch.equal(once.frame(5), bare.frame(5))  # past the 5th image
    assert torch.equal(loop.frame(5), loop.frame(0))


def test_frames_stay_bounded_in_the_cache(tmp_path: Path) -> None:
    rend, _ = _render(tmp_path, "  - hold: 10f\n    layers:\n      - {frames: seq, loop: true}\n")
    rend.begin_segment([0])
    for f in range(10):
        rend.frame(f)
    assert 0 < len(list(rend._layer_frames)) <= rend._layer_frames.cap


# ---------------------------------------------------------------- cache


_THREE = """  - {photo: bg, hold: 4f}
  - hold: 6f
    layers:
      - {image: bg}
      - {frames: seq, every: 2f, pos: [540, 900]}
  - hold: 4f
    layers:
      - {image: bg}
      - {frames: seq, start: 3, every: 2f, pos: [540, 900]}
"""


def test_changing_one_frame_rehashes_only_the_segments_that_show_it(tmp_path: Path) -> None:
    spec = load_spec(_project(tmp_path, _THREE))
    tl = resolve(spec, tmp_path)
    shas = _asset_shas(spec, tmp_path)
    assert shas["frames:seq"] == str(N) and {f"frames:seq:{i}" for i in range(N)} <= set(shas)

    def hashes(s):
        return [segment_hash(tl, seg, s, "e1") for seg in tl.segments]

    base = hashes(shas)
    # Segment 1 shows images 0-2, segment 2 images 3-4 (hold, not past the end).
    seq = tmp_path / "assets" / "seq"
    _frame(1, size=(40, 20)).rotate(180).save(seq / "001.png")
    got = hashes(_asset_shas(spec, tmp_path))
    assert [g != b for g, b in zip(got, base)] == [False, True, False]
    _frame(4).rotate(180).save(seq / "004.png")
    got2 = hashes(_asset_shas(spec, tmp_path))
    assert [g != b for g, b in zip(got2, got)] == [False, False, True]
    # Adding an image no segment shows changes nothing.
    shutil.copy(seq / "004.png", seq / "099.png")
    assert hashes(_asset_shas(spec, tmp_path)) == got2


def test_trails_hash_the_images_they_redraw(tmp_path: Path) -> None:
    shots = ("  - hold: 3f\n    layers:\n      - {frames: seq, start: 2, effects: [{type: trails, frames: 2}]}\n"
             "  - hold: 3f\n    layers:\n      - {frames: seq}\n")
    spec = load_spec(_project(tmp_path, shots))
    tl = resolve(spec, tmp_path)
    shas = _asset_shas(spec, tmp_path)
    seg0 = tl.segments[0]
    base = segment_hash(tl, seg0, shas, "e1")
    # Segment 0 shows images 2-4 directly and clamps ghosts at 2: image 1 never appears, image 2 does.
    assert segment_hash(tl, seg0, {**shas, "frames:seq:1": "x"}, "e1") == base
    assert segment_hash(tl, seg0, {**shas, "frames:seq:2": "x"}, "e1") != base


def test_films_without_frames_keep_their_hash(tmp_path: Path) -> None:
    spec = load_spec(_project(tmp_path, "  - hold: 4f\n    layers:\n      - {image: bg}\n"))
    tl = resolve(spec, tmp_path)
    shas = _asset_shas(spec, tmp_path)
    assert not any(k.startswith("frames:") for k in shas)
    # Pinned before #59: the payload carries no "frames" key, so the digest is the old one.
    from montaj import project

    seen = []
    real = project.json.dumps
    project.json.dumps = lambda obj, **kw: seen.append(obj) or real(obj, **kw)
    try:
        segment_hash(tl, tl.segments[0], shas, "e1")
    finally:
        project.json.dumps = real
    assert "frames" not in seen[-1]


# ---------------------------------------------------------------- ingest --list


def test_ingest_list_prints_one_line_per_frames_folder(tmp_path: Path) -> None:
    _project(tmp_path, "  - {photo: bg, hold: 1f}\n")
    assets = tmp_path / "assets"
    (assets / "mattes").mkdir()
    Image.new("L", (4, 4)).save(assets / "mattes" / "abc.subject-v1.png")
    (assets / "empty").mkdir()
    opaque = assets / "plates"
    opaque.mkdir()
    Image.new("RGB", (8, 6)).save(opaque / "a.jpg")
    lines = describe_all(assets)
    assert "seq frames 5 40x20 alpha" in lines
    assert "plates frames 1 8x6 opaque" in lines
    assert not any(ln.startswith(("mattes", "empty", "000.png")) for ln in lines)
    assert len(lines) == 2 + 1 + N  # seq, plates, bg.png, m00..m04


def test_cli_ingest_list_shows_the_folder_as_one_asset(tmp_path: Path, capsys) -> None:
    from montaj import analysis, cli

    proj = tmp_path / "p"
    (proj / "assets" / "mascot").mkdir(parents=True)
    (proj / "montaj.yaml").write_text("video: {size: 108x192, fps: 30, look: warm-film}\nassets: assets\nshots: []\n")
    for i in range(3):
        _frame(i).save(proj / "assets" / "mascot" / f"{i:05d}.png")
    assert cli.main(["-C", str(proj), "ingest", "--list"]) == 0
    assert capsys.readouterr().out.strip().splitlines() == ["mascot frames 3 40x20 alpha", "OK 1 assets"]
    assert analysis.media(proj / "assets") == []  # never analysed as photos
