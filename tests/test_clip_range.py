"""Clip reuse: `clip` as a path to a finished video plus `range: [start, end]`."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from montaj.pipeline import _asset_shas
from montaj.project import resolve_clip, segment_hash
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve


def _video(path: Path, seconds: int = 4, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"color=c={color}:size=64x48:rate=30:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def _film(tmp: Path, shot: dict, with_assets: bool = False) -> Path:
    if with_assets:
        (tmp / "assets").mkdir()
    spec = {"video": {"size": "1080x1920", "fps": 30, "look": "warm-film"}, "shots": [shot]}
    p = tmp / "montaj.yaml"
    p.write_text(yaml.safe_dump(spec))
    return p


def _hash(tmp: Path, spec_path: Path) -> str:
    spec = load_spec(spec_path)
    tl = resolve(spec, tmp)
    return segment_hash(tl, tl.segments[0], _asset_shas(spec, tmp), "e1")


def test_range_sets_shot_length_and_offset(tmp_path: Path) -> None:
    _video(tmp_path / "old" / "final.mp4")
    p = _film(tmp_path, {"clip": "old/final.mp4", "range": ["1s", "3.5s"]})
    tl = resolve(load_spec(p), tmp_path)
    assert (tl.shots[0].start, tl.shots[0].end) == (0, 75)
    assert tl.shots[0].kind == "clip"


def test_hash_changes_with_range_and_file_bytes(tmp_path: Path) -> None:
    src = tmp_path / "old" / "final.mp4"
    _video(src)
    base = _hash(tmp_path, _film(tmp_path, {"clip": "old/final.mp4", "range": ["1s", "3s"]}))
    again = _hash(tmp_path, _film(tmp_path, {"clip": "old/final.mp4", "range": ["1s", "3s"]}))
    moved = _hash(tmp_path, _film(tmp_path, {"clip": "old/final.mp4", "range": ["0s", "2s"]}))
    assert base == again
    assert base != moved
    _video(src, color="blue")
    changed = _hash(tmp_path, _film(tmp_path, {"clip": "old/final.mp4", "range": ["1s", "3s"]}))
    assert changed != base


def test_unset_range_leaves_hash_payload_alone(tmp_path: Path) -> None:
    from montaj.project import _shot_payload

    _video(tmp_path / "a.mp4")
    spec = load_spec(_film(tmp_path, {"clip": "a.mp4", "hold": "1s"}))
    assert "range" not in _shot_payload(spec.shots[0])


def test_assets_folder_not_needed_for_path_clip(tmp_path: Path) -> None:
    _video(tmp_path / "a.mp4")
    load_spec(_film(tmp_path, {"clip": "a.mp4", "range": ["0s", "1s"]}))
    assert not (tmp_path / "assets").exists()


def test_stem_clip_still_resolves_in_assets(tmp_path: Path) -> None:
    _video(tmp_path / "assets" / "intro.mp4")
    assert resolve_clip(tmp_path, tmp_path / "assets", "intro").name == "intro.mp4"


@pytest.mark.parametrize(
    ("shot", "needle"),
    [
        ({"clip": "a.mp4", "range": ["1s", "9s"]}, "clip is"),
        ({"clip": "a.mp4", "range": ["2s", "1s"]}, "end must be after start"),
        ({"clip": "a.mp4", "range": ["0s", "1s"], "hold": "1s"}, "range and hold are mutually exclusive"),
        ({"clip": "a.mp4", "range": ["0s", "1s"], "until": "2s"}, "range and until are mutually exclusive"),
        ({"clip": "a.mp4", "range": ["0s", "1s"], "clip_in": "1s"}, "range and clip_in are mutually exclusive"),
        ({"clip": "gone.mp4", "range": ["0s", "1s"]}, "not found"),
        ({"clip": "a.mp4", "range": ["0s"]}, "need [start, end]"),
        ({"photo": "x", "range": ["0s", "1s"]}, "range needs clip"),
    ],
)
def test_bad_range_is_rejected(tmp_path: Path, shot: dict, needle: str) -> None:
    _video(tmp_path / "a.mp4", seconds=4)
    p = _film(tmp_path, shot, with_assets=True)
    with pytest.raises(SpecError) as exc:
        load_spec(p)
    assert needle in str(exc.value)
