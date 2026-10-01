"""Segment cache and preview sizing. The render itself is the only GPU part."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from montaj.encode import frame_count, probe
from montaj.pipeline import _apply_mode, _encode_segment, engine_version, render
from montaj.project import file_sha, find_photo, segment_hash
from montaj.spec import Spec, load_spec
from montaj.timeline import Segment, Still, Trans, Wall, resolve

SPEC = """\
video:
  size: 640x960
  fps: 30
  look: warm-film
  motion_blur: 1
assets: assets
shots:
  - {photo: a, hold: 6f}
  - {photo: b, hold: 8f, in: {type: fade, dur: 3f}}
  - {photo: c, hold: 8f, in: {type: whip, dur: 4f, axis: x}}
  - {photo: d, hold: 6f}
"""
EDITED = SPEC.replace(
    "{photo: b, hold: 8f, in: {type: fade, dur: 3f}}",
    "{photo: b, hold: 8f, zoom: 1.5, in: {type: fade, dur: 3f}}",
)


def test_preview_halves_size_and_caps_blur() -> None:
    spec = Spec.model_validate({
        "video": {"size": "1080x1920", "fps": 30, "look": "warm-film", "motion_blur": 6},
        "shots": [{"photo": "a", "hold": "2f"}],
    })
    preview = _apply_mode(spec, "preview")
    assert preview.video.size == "540x960"
    assert preview.video.motion_blur == 2
    assert spec.video.size == "1080x1920" and spec.video.motion_blur == 6

    odd = spec.model_copy(update={"video": spec.video.model_copy(update={"size": "1002x1002", "motion_blur": 1})})
    halved = _apply_mode(odd, "preview")
    assert halved.video.size == "500x500"
    assert halved.video.motion_blur == 1
    assert _apply_mode(spec, "final") is spec
    with pytest.raises(ValueError):
        _apply_mode(spec, "draft")


def _project(tmp: Path, text: str) -> Path:
    assets = tmp / "assets"
    assets.mkdir(exist_ok=True)
    for i, stem in enumerate("abcd"):
        Image.new("RGB", (32, 48), (30 + 40 * i, 70, 140)).save(assets / f"{stem}.jpg", quality=95)
    path = tmp / "montaj.yaml"
    path.write_text(text)
    return path


def _scene_shots(scene) -> set[int]:
    if isinstance(scene, (Still, Wall)):
        return {scene.shot}
    if isinstance(scene, Trans):
        return _scene_shots(scene.a) | _scene_shots(scene.b)
    raise AssertionError(type(scene))


def _referencing(tl, index: int) -> set[int]:
    found = set()
    for seg in tl.segments:
        used: set[int] = set()
        for f in range(seg.start, seg.end):
            used |= _scene_shots(tl.plan(f).scene)
        if index in used:
            found.add(seg.index)
    return found


def _ident(path: Path) -> tuple[int, int]:
    st = path.stat()
    return (st.st_ino, st.st_size)


@pytest.mark.gpu
def test_cache_rerenders_only_segments_that_reference_the_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.chdir(tmp_path)  # a relative build/ log must not land in the repo
    spec_path = _project(tmp_path, SPEC)
    log = tmp_path / "probe.log"

    first = render(spec_path, "preview")
    assert (first.rendered, first.cached) == (4, 0)
    assert first.n_frames == 28 and first.size == (320, 480)
    assert first.seconds == pytest.approx(28 / 30)
    assert first.path == (tmp_path / "out" / "preview.mp4").resolve()
    assert probe(first.path, log=log).size == (320, 480)
    assert frame_count(first.path, log=log) == 28
    assert "miss" in (tmp_path / "build" / "render.log").read_text()

    seg_dir = tmp_path / "build" / "segments"
    cached_files = {p.name: _ident(p) for p in seg_dir.glob("*.mp4") if not p.name.startswith(".")}
    assert len(cached_files) == 4
    assert not list(seg_dir.glob(".*.partial.mp4"))

    second = render(spec_path, "preview")
    assert (second.rendered, second.cached) == (0, 4)
    assert frame_count(second.path, log=log) == 28
    for name, ident in cached_files.items():
        assert _ident(seg_dir / name) == ident

    spec_path.write_text(EDITED)
    # The pipeline hashes the preview spec, whose size differs from the file on disk.
    preview_tl = resolve(_apply_mode(load_spec(spec_path), "preview"), tmp_path)
    edited = 1  # shot b
    expect = _referencing(preview_tl, edited)
    assert expect == {1, 2}  # its own fade segment, and the whip that reaches back into it
    shas = {pid: file_sha(find_photo(tmp_path / "assets", pid)) for pid in "abcd"}
    new_hash = {
        seg.index: segment_hash(preview_tl, seg, shas, engine_version())
        for seg in preview_tl.segments
    }

    third = render(spec_path, "preview")
    assert third.rendered == len(expect)
    assert third.cached == len(preview_tl.segments) - len(expect)
    assert frame_count(third.path, log=log) == third.n_frames == 28
    assert probe(third.path, log=log).size == (320, 480)
    for index, digest in new_hash.items():
        path = seg_dir / f"{digest}.mp4"
        assert path.is_file()
        if index in expect:
            assert f"{digest}.mp4" not in cached_files
        else:
            assert cached_files[f"{digest}.mp4"] == _ident(path)
    assert not list(seg_dir.glob(".*.partial.mp4"))


def test_encode_segment_temp_is_unique_per_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two misses of one segment must not share `.<hash>.partial.mp4`."""
    seen: list[Path] = []

    class FakeEnc:
        n = 0

        def __init__(self, path, w, h, fps, log=None):
            FakeEnc.n += 1
            self.payload = f"v{FakeEnc.n}".encode()
            self.path = Path(path)
            seen.append(self.path)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.path.write_bytes(self.payload)

        def write(self, frame):
            raise AssertionError("an empty segment must not encode a frame")

    monkeypatch.setattr("montaj.pipeline.Encoder", FakeEnc)
    dest = tmp_path / "abc123.mp4"
    seg = Segment(index=0, start=0, end=0)
    renderer = SimpleNamespace(cv=SimpleNamespace(W=8, H=8))
    log = tmp_path / "render.log"
    _encode_segment(renderer, seg, dest, 30, log)
    _encode_segment(renderer, seg, dest, 30, log)

    shared = f".{dest.stem}.partial.mp4"
    pid = str(os.getpid())
    assert len(seen) == 2 and seen[0] != seen[1]
    assert dest.read_bytes() == b"v2"
    for path in seen:
        assert path.name != shared
        assert pid in path.name
        assert path.name.endswith(".partial.mp4")
        assert not path.exists()
    assert not list(tmp_path.glob(".*.partial.mp4"))
