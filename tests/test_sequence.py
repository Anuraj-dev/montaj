"""Sequence shots: one hold cycles photos or colours on a fixed interval."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from montaj.cli import _intended_black, _intended_freeze
from montaj.pipeline import _asset_shas
from montaj.project import file_sha, segment_hash
from montaj.render.canvas import Canvas
from montaj.render.core import still
from montaj.render.frame import Renderer
from montaj.review.summary import shot_label
from montaj.spec import SpecError, load_spec
from montaj.timeline import Sequence, resolve, sequence_index


def _jpeg(folder: Path, stem: str, color: tuple[int, int, int]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.jpg"
    Image.new("RGB", (32, 40), color).save(path, quality=95)
    return path


def _write(tmp: Path, shots: str, stems: dict[str, tuple[int, int, int]] | None = None, *, fps: int = 30) -> Path:
    assets = tmp / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for stem, color in (stems or {}).items():
        _jpeg(assets, stem, color)
    path = tmp / "montaj.yaml"
    path.write_text(
        f"video:\n  size: 180x320\n  fps: {fps}\n  look: warm-film\n"
        "  background: \"#102030\"\n  motion_blur: 1\n"
        f"assets: assets\nshots:\n{shots}"
    )
    return path


def _errors(tmp: Path, shots: str, stems: dict | None = None) -> list[str]:
    with pytest.raises(SpecError) as ei:
        load_spec(_write(tmp, shots, stems))
    return ei.value.errors


def _open(path: Path):
    tl = resolve(load_spec(path), path.parent)
    cv = Canvas(tl.size[0], tl.size[1], device="cpu")
    return tl, Renderer(tl, path.parent, cv)


def _solid(img: torch.Tensor, rgb: tuple[int, int, int]) -> bool:
    want = torch.tensor(rgb, dtype=torch.float32) / 255
    return torch.allclose(img, want[:, None, None].expand_as(img))


def test_sequence_index_holds_item_zero_before_the_shot() -> None:
    assert sequence_index(10, 4, 3, 9.9) == 0
    assert sequence_index(10, 4, 3, 10) == 0
    assert sequence_index(10, 4, 3, 13.9) == 0
    assert sequence_index(10, 4, 3, 14) == 1
    assert sequence_index(10, 4, 3, 22) == 0  # 3 steps of 4, wrapped


def test_twelve_frame_flicker_is_one_shot(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "  - hold: 24f\n"
        "    sequence:\n"
        "      every: 12f\n"
        '      items: ["#ffffff", "#000000"]\n',
    )
    spec = load_spec(path)
    assert len(spec.shots) == 1
    assert spec.shots[0].sequence is not None
    assert spec.shots[0].sequence.every == "12f"
    tl, renderer = _open(path)
    assert tl.n_frames == 24
    assert tl.shots[0].kind == "sequence"
    assert tl.shots[0].spec.sequence.every == 12
    assert shot_label(tl.shots[0]) == "sequence"
    assert isinstance(tl.plan(0).scene, Sequence)
    assert isinstance(tl.plan(11).scene, Sequence)
    assert isinstance(tl.plan(12).scene, Sequence)
    for f in range(12):
        assert _solid(renderer._eval(tl.plan(f).scene, float(f)), (255, 255, 255))
    for f in range(12, 24):
        assert _solid(renderer._eval(tl.plan(f).scene, float(f)), (0, 0, 0))
    first = renderer.frame(0)
    later = renderer.frame(12)
    assert torch.equal(first, renderer.frame(0))
    assert torch.equal(later, renderer.frame(12))
    assert not torch.equal(first, later)


def test_seconds_interval_matches_frames_and_the_list_wraps(tmp_path: Path) -> None:
    frames = _write(
        tmp_path / "frames",
        "  - hold: 16f\n"
        "    sequence:\n"
        "      every: 4f\n"
        '      items: ["#ff0000", "#00ff00", "#0000ff"]\n',
    )
    seconds = _write(
        tmp_path / "seconds",
        "  - hold: 16f\n"
        "    sequence:\n"
        "      every: 0.4s\n"
        '      items: ["#ff0000", "#00ff00", "#0000ff"]\n',
        fps=10,
    )
    tl_f, r_f = _open(frames)
    tl_s, r_s = _open(seconds)
    assert tl_f.shots[0].spec.sequence.every == 4
    assert tl_s.shots[0].spec.sequence.every == 4
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0))
    for f, rgb in zip((0, 4, 8, 12), colors):
        assert _solid(r_f._eval(tl_f.plan(f).scene, float(f)), rgb)
        assert _solid(r_s._eval(tl_s.plan(f).scene, float(f)), rgb)
    # The last partial step is still the third colour (frames 8 and 9 of a 10f hold).
    short = _write(
        tmp_path / "short",
        "  - hold: 10f\n"
        "    sequence:\n"
        "      every: 4f\n"
        '      items: ["#ff0000", "#00ff00", "#0000ff"]\n',
    )
    tl, renderer = _open(short)
    assert _solid(renderer._eval(tl.plan(8).scene, 8.0), (0, 0, 255))
    assert _solid(renderer._eval(tl.plan(9).scene, 9.0), (0, 0, 255))


def test_photo_items_match_stills_and_mix_with_colour(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "  - hold: 8f\n"
        "    sequence:\n"
        "      every: 4f\n"
        "      items:\n"
        "        - {photo: 16}\n"
        '        - {color: "#112233"}\n'
        "        - b\n",
        {"16": (220, 30, 30), "b": (20, 30, 220)},
    )
    tl, renderer = _open(path)
    seq = tl.shots[0].spec.sequence
    assert [item.photo for item in seq.items] == ["16", None, "b"]
    photo = renderer._photo("16", None, False)
    assert torch.allclose(renderer._eval(tl.plan(0).scene, 0.0), still(renderer.cv, photo))
    assert _solid(renderer._eval(tl.plan(4).scene, 4.0), (0x11, 0x22, 0x33))
    long = _write(
        tmp_path / "long",
        "  - hold: 12f\n"
        "    sequence:\n"
        "      every: 4f\n"
        "      items: [16, b]\n",
        {"16": (220, 30, 30), "b": (20, 30, 220)},
    )
    tl, renderer = _open(long)
    assert torch.allclose(
        renderer._eval(tl.plan(0).scene, 0.0),
        still(renderer.cv, renderer._photo("16", None, False)),
    )
    assert torch.allclose(
        renderer._eval(tl.plan(4).scene, 4.0),
        still(renderer.cv, renderer._photo("b", None, False)),
    )
    assert torch.allclose(
        renderer._eval(tl.plan(8).scene, 8.0),
        still(renderer.cv, renderer._photo("16", None, False)),
    )


def test_fade_in_uses_the_sequence_clock(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "  - hold: 8f\n"
        "    in: {type: fade, dur: 4f}\n"
        "    sequence:\n"
        "      every: 8f\n"
        '      items: ["#ffffff", "#000000"]\n',
    )
    tl, renderer = _open(path)
    bg = torch.tensor([0x10, 0x20, 0x30], dtype=torch.float32) / 255
    at0 = renderer._eval(tl.plan(0).scene, 0.0)
    assert torch.allclose(at0, bg[:, None, None].expand_as(at0))
    assert _solid(renderer._eval(tl.plan(4).scene, 4.0), (255, 255, 255))


def test_hash_includes_the_list_and_the_interval(tmp_path: Path) -> None:
    shots = (
        "  - {photo: b, hold: 8f}\n"
        "  - hold: 8f\n"
        "    sequence:\n"
        "      every: 4f\n"
        '      items: [a, "#000000"]\n'
    )
    path = _write(tmp_path, shots, {"a": (10, 20, 30), "b": (40, 50, 60)})
    tl = resolve(load_spec(path), path.parent)
    shas = _asset_shas(tl.spec, path.parent)
    assert set(shas) == {"photo:a", "photo:b"}
    h0 = [segment_hash(tl, seg, shas, "e1") for seg in tl.segments]
    changed = dict(shas)
    changed["photo:a"] = "0" * 64
    h_photo = [segment_hash(tl, seg, changed, "e1") for seg in tl.segments]
    assert h_photo[0] == h0[0]
    assert h_photo[1] != h0[1]

    listed = _write(
        tmp_path / "list",
        shots.replace('"#000000"', '"#ffffff"'),
        {"a": (10, 20, 30), "b": (40, 50, 60)},
    )
    tl_list = resolve(load_spec(listed), listed.parent)
    h_list = [segment_hash(tl_list, seg, shas, "e1") for seg in tl_list.segments]
    assert h_list[0] == h0[0]
    assert h_list[1] != h0[1]

    slower = _write(
        tmp_path / "slower",
        shots.replace("every: 4f", "every: 8f"),
        {"a": (10, 20, 30), "b": (40, 50, 60)},
    )
    tl_slow = resolve(load_spec(slower), slower.parent)
    h_slow = [segment_hash(tl_slow, seg, shas, "e1") for seg in tl_slow.segments]
    assert h_slow[0] == h0[0]
    assert h_slow[1] != h0[1]

    same = _write(
        tmp_path / "same",
        "  - hold: 8f\n"
        "    sequence:\n"
        "      every: 0.4s\n"
        '      items: ["#ffffff", "#000000"]\n',
        fps=30,
    )
    frames = _write(
        tmp_path / "frames",
        "  - hold: 8f\n"
        "    sequence:\n"
        "      every: 12f\n"
        '      items: ["#ffffff", "#000000"]\n',
    )
    tl_same = resolve(load_spec(same), same.parent)
    tl_frames = resolve(load_spec(frames), frames.parent)
    assert segment_hash(tl_same, tl_same.segments[0], {}, "e1") == segment_hash(
        tl_frames, tl_frames.segments[0], {}, "e1"
    )


def test_check_treats_a_colour_step_as_intended(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "  - {photo: a, hold: 30f}\n"
        "  - hold: 24f\n"
        "    sequence:\n"
        "      every: 12f\n"
        '      items: ["#ffffff", "#000000"]\n',
        {"a": (180, 90, 40)},
    )
    freeze = _intended_freeze(tmp_path)
    black = _intended_black(tmp_path)

    def covers(spans, lo: float, hi: float) -> bool:
        return any(a - 1e-6 <= lo and b + 1e-6 >= hi for a, b in spans)

    assert covers(freeze, 0.0, 1.0)  # still
    assert covers(freeze, 1.0, 1.8)  # both colour steps
    assert covers(black, 1.4, 1.8)  # the dark step
    assert not any(max(a, 1.05) < min(b, 1.35) for a, b in black)

    faded = _write(
        tmp_path / "fade",
        "  - {photo: a, hold: 30f}\n"
        "  - hold: 60f\n"
        "    in: {type: fade, dur: 30f}\n"
        "    sequence:\n"
        "      every: 30f\n"
        '      items: ["#ffffff", "#000000"]\n',
        {"a": (180, 90, 40)},
    )
    spans = _intended_freeze(faded.parent)
    assert not any(max(a, 1.2) < min(b, 1.8) for a, b in spans)
    assert covers(spans, 2.0, 3.0)


def test_sequence_rejects_bad_values(tmp_path: Path) -> None:
    beats = _errors(
        tmp_path / "beats",
        "  - hold: 24f\n    sequence: {every: 4b, items: [a, b]}\n",
        {"a": (1, 2, 3), "b": (4, 5, 6)},
    )
    assert any("shots[0].sequence.every" in e and "beats are not allowed" in e for e in beats)
    bare = _errors(
        tmp_path / "bare",
        '  - hold: 24f\n    sequence: {every: 12, items: ["#ffffff", "#000000"]}\n',
    )
    assert any("shots[0].sequence.every" in e and "bare number" in e for e in bare)
    zero = _errors(
        tmp_path / "zero",
        '  - hold: 24f\n    sequence: {every: 0f, items: ["#ffffff", "#000000"]}\n',
    )
    assert any("shots[0].sequence.every" in e and "at least 1 frame" in e for e in zero)
    short = _errors(
        tmp_path / "short",
        '  - hold: 24f\n    sequence: {every: 12f, items: ["#ffffff"]}\n',
    )
    assert any("shots[0].sequence.items" in e and "at least two" in e for e in short)
    color = _errors(
        tmp_path / "color",
        "  - hold: 24f\n    sequence:\n      every: 12f\n      items:\n        - {color: \"#fff\"}\n        - {color: blue}\n",
    )
    assert any("items[0].color" in e and "#rrggbb" in e for e in color)
    assert any("items[1].color" in e and "blue" in e for e in color)
    both = _errors(
        tmp_path / "both",
        "  - hold: 24f\n    sequence:\n      every: 12f\n      items:\n        - {photo: a, color: \"#ffffff\"}\n        - b\n",
        {"a": (1, 2, 3), "b": (4, 5, 6)},
    )
    assert any("items[0]" in e and "mutually exclusive" in e for e in both)
    extra = _errors(
        tmp_path / "extra",
        "  - {photo: a, hold: 8f, sequence: {every: 4f, items: [a, b]}}\n",
        {"a": (1, 2, 3), "b": (4, 5, 6)},
    )
    assert any("shots[0]" in e and "mutually exclusive" in e for e in extra)
    framing = _errors(
        tmp_path / "framing",
        '  - hold: 8f\n    crop: [0, 0, 4, 4]\n    sequence: {every: 4f, items: ["#ffffff", "#000000"]}\n',
    )
    assert any("shots[0].crop" in e and "not used on a sequence" in e for e in framing)
    missing = _errors(
        tmp_path / "missing",
        '  - hold: 8f\n    sequence: {every: 4f, items: ["#ffffff", gone]}\n',
    )
    assert any("shots[0].sequence.items[1]" in e and "gone" in e and "have:" in e for e in missing)
