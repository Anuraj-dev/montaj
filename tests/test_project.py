from pathlib import Path

import pytest
from PIL import Image

from montaj.project import file_sha, find_photo, segment_hash
from montaj.spec import load_spec
from montaj.timeline import resolve


def _jpeg(folder: Path, stem: str, color=(20, 40, 60)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{stem}.jpg"
    Image.new("RGB", (8, 8), color).save(p, "JPEG")
    return p


def _resolve(tmp: Path, shots: str, stems: list[str], *, extra=""):
    assets = tmp / "assets"
    for s in stems:
        _jpeg(assets, s)
    p = tmp / "montaj.yaml"
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\n"
        f"assets: assets\nshots:\n{shots}{extra}\n"
    )
    spec = load_spec(p)
    return resolve(spec, p.parent), {s: file_sha(assets / f"{s}.jpg") for s in stems}


def test_find_photo_and_sha(tmp_path: Path):
    assets = tmp_path / "assets"
    jpg = _jpeg(assets, "16")
    Image.new("RGB", (4, 4), (1, 2, 3)).save(assets / "32.PNG", "PNG")
    assert find_photo(assets, "16") == jpg
    assert find_photo(assets, "32").name == "32.PNG"
    digest = file_sha(jpg)
    assert len(digest) == 64 and digest == file_sha(jpg)
    with pytest.raises(FileNotFoundError) as ei:
        find_photo(assets, "99")
    assert "99" in str(ei.value) and "have:" in str(ei.value)
    assert "16" in str(ei.value)


def test_hash_locality_edit_one_shot(tmp_path: Path):
    shots = "  - {photo: a, hold: 10f}\n  - {photo: b, hold: 10f}\n  - {photo: c, hold: 10f}\n"
    tl, shas = _resolve(tmp_path, shots, ["a", "b", "c"])
    h0 = [segment_hash(tl, seg, shas, "e1") for seg in tl.segments]
    spec2 = tl.spec.model_copy(deep=True)
    spec2.shots[1].crop = (1.0, 2.0, 3.0, 4.0)
    tl2 = resolve(spec2, tmp_path)
    h1 = [segment_hash(tl2, seg, shas, "e1") for seg in tl2.segments]
    assert h0[0] == h1[0]
    assert h0[1] != h1[1]
    assert h0[2] == h1[2]


def test_hash_locality_whip_references_both_shots(tmp_path: Path):
    shots = (
        "  - {photo: a, hold: 20f}\n"
        "  - {photo: b, hold: 20f, in: {type: whip, dur: 10f}}\n"
    )
    tl, shas = _resolve(tmp_path, shots, ["a", "b"])
    h0 = [segment_hash(tl, seg, shas, "e1") for seg in tl.segments]
    spec_b = tl.spec.model_copy(deep=True)
    spec_b.shots[1].crop = (0.0, 0.0, 4.0, 4.0)
    h_b = [segment_hash(resolve(spec_b, tmp_path), seg, shas, "e1") for seg in tl.segments]
    assert h0[0] != h_b[0]
    assert h0[1] != h_b[1]
    spec_a = tl.spec.model_copy(deep=True)
    spec_a.shots[0].focus = (3.0, 4.0)
    h_a = [segment_hash(resolve(spec_a, tmp_path), seg, shas, "e1") for seg in tl.segments]
    assert h0[0] != h_a[0]
    assert h0[1] != h_a[1]


def test_hash_video_block_and_engine_version(tmp_path: Path):
    shots = "  - {photo: a, hold: 10f}\n"
    tl, shas = _resolve(tmp_path, shots, ["a"])
    a = segment_hash(tl, tl.segments[0], shas, "e1")
    b = segment_hash(tl, tl.segments[0], shas, "e2")
    assert a != b
    spec2 = tl.spec.model_copy(deep=True)
    spec2.video.look = "other"
    tl2 = resolve(spec2, tmp_path)
    assert segment_hash(tl2, tl2.segments[0], shas, "e1") != a


def test_hash_wall_prints_included(tmp_path: Path):
    wall = """
  - {photo: a, hold: 8f}
  - hold: 12f
    wall:
      grid: 10x10
      seed: 1
      jitter: 0
      rot: 0
      print_height: 8
      prints:
        - {photo: a, cell: [0, 0], straight: true}
        - {photo: b, cell: [1, 0], straight: true}
      light: {center: [0, 0], radius: [1, 1]}
      camera:
        pos: [{t: 0f}]
        zoom: [{t: 0f, value: 1}]
"""
    tl, shas = _resolve(tmp_path, wall, ["a", "b"])
    h = segment_hash(tl, tl.segments[1], shas, "e1")
    shas2 = dict(shas)
    shas2["b"] = "0" * 64
    assert segment_hash(tl, tl.segments[1], shas2, "e1") != h
    assert segment_hash(tl, tl.segments[0], shas2, "e1") == segment_hash(tl, tl.segments[0], shas, "e1")
