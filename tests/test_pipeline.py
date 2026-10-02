"""Segment cache and preview sizing. The render itself is the only GPU part."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from montaj.encode import frame_count, probe
from montaj.pipeline import _apply_mode, _asset_shas, _encode_segment, engine_version, render
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


def test_engine_version_bumped_for_m2() -> None:
    assert engine_version() == "0.1.0+4"


def test_asset_shas_cover_a_morph_only_photo_and_a_clip(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "a.jpg").write_bytes(b"photo-a")
    (assets / "b.png").write_bytes(b"morph-only")
    (assets / "reel.mp4").write_bytes(b"clip-bytes")
    (tmp_path / "montaj.yaml").write_text(
        "video:\n  size: 64x64\n  fps: 30\n  look: warm-film\nassets: assets\nshots:\n"
        "  - {photo: a, hold: 8f, morph: {photo: b, at: 2f, dur: 4f}}\n"
        "  - {clip: reel, hold: 6f, clip_in: 1s}\n"
    )
    spec = load_spec(tmp_path / "montaj.yaml")
    shas = _asset_shas(spec, tmp_path)
    assert set(shas) == {"photo:a", "photo:b", "clip:reel"}
    tl = resolve(spec, tmp_path)
    for seg in tl.segments:
        digest = segment_hash(tl, seg, shas, engine_version())
        assert len(digest) == 64


def test_photo_and_clip_sha_keys_do_not_collide(tmp_path: Path) -> None:
    """assets/reel.jpg and assets/reel.mp4 both count. Editing the clip misses only the clip segment."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "reel.jpg").write_bytes(b"photo-v1")
    (assets / "reel.mp4").write_bytes(b"clip-v1")
    (tmp_path / "montaj.yaml").write_text(
        "video:\n  size: 64x64\n  fps: 30\n  look: warm-film\nassets: assets\nshots:\n"
        "  - {photo: reel, hold: 4f}\n"
        "  - {clip: reel, hold: 4f}\n"
    )
    spec = load_spec(tmp_path / "montaj.yaml")
    shas = _asset_shas(spec, tmp_path)
    assert shas["photo:reel"] != shas["clip:reel"]
    tl = resolve(spec, tmp_path)
    photo_h = segment_hash(tl, tl.segments[0], shas, engine_version())
    clip_h = segment_hash(tl, tl.segments[1], shas, engine_version())
    (assets / "reel.mp4").write_bytes(b"clip-v2-edited")
    shas2 = _asset_shas(spec, tmp_path)
    assert segment_hash(tl, tl.segments[0], shas2, engine_version()) == photo_h
    assert segment_hash(tl, tl.segments[1], shas2, engine_version()) != clip_h


def test_render_holds_the_gpu_and_muxes_the_track(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager

    assets = tmp_path / "assets"
    assets.mkdir()
    Image.new("RGB", (8, 8), (20, 40, 60)).save(assets / "a.jpg")
    music = tmp_path / "music"
    music.mkdir()
    (music / "song.wav").write_bytes(b"RIFFstub")
    spec_path = tmp_path / "montaj.yaml"
    spec_path.write_text(
        "video:\n  size: 64x64\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
        "assets: assets\naudio:\n  track: music/song.wav\n  fade_out: 3s\n  loudnorm: true\n"
        "shots:\n  - {photo: a, hold: 4f}\n"
    )
    held = {"n": 0, "on": False}
    seen: list[tuple[str, bool]] = []

    @contextmanager
    def fake_hold(log=None):
        held["n"] += 1
        held["on"] = True
        try:
            yield tmp_path / "lock"
        finally:
            held["on"] = False

    class FakeCV:
        def __init__(self, W, H, device="cuda"):
            seen.append(("canvas", held["on"]))
            self.W, self.H, self.device = W, H, "cpu"

    class FakeRenderer:
        def __init__(self, tl, spec_dir, cv):
            seen.append(("renderer", held["on"]))
            self.cv = cv

        def begin_segment(self, shots):
            seen.append(("begin", held["on"]))

        def frame(self, f):
            seen.append(("frame", held["on"]))
            return torch.zeros(3, self.cv.H, self.cv.W)

    class FakeEnc:
        def __init__(self, path, w, h, fps, log=None):
            seen.append(("encoder", held["on"]))
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.path.write_bytes(b"seg")

        def write(self, frame):
            seen.append(("write", held["on"]))
            return None

    muxed: dict = {}

    def fake_concat(chunks, out, **_kwargs):
        seen.append(("concat", held["on"]))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"cat")
        return Path(out)

    def fake_mux(video, audio, out, loudnorm=True, fade_out=0.0, *, log=None):
        seen.append(("mux", held["on"]))
        muxed["args"] = (Path(audio).name, loudnorm, fade_out)
        Path(out).write_bytes(b"muxed")
        return Path(out)

    monkeypatch.setattr("montaj.pipeline.gpu_hold", fake_hold)
    monkeypatch.setattr("montaj.pipeline.Canvas", FakeCV)
    monkeypatch.setattr("montaj.pipeline.Renderer", FakeRenderer)
    monkeypatch.setattr("montaj.pipeline.Encoder", FakeEnc)
    monkeypatch.setattr("montaj.pipeline.concat", fake_concat)
    monkeypatch.setattr("montaj.pipeline.mux", fake_mux)

    result = render(spec_path, "preview")
    assert held["n"] == 1 and held["on"] is False
    for name in ("canvas", "renderer", "encoder", "frame", "write", "concat", "mux"):
        assert any(op == name and inside for op, inside in seen), seen
    assert all(inside for _op, inside in seen)
    assert result.audio is True
    assert muxed["args"] == ("song.wav", True, 3.0)
    assert result.path.read_bytes() == b"muxed"
    assert (result.rendered, result.cached) == (1, 0)

    spec_path.write_text(
        "video:\n  size: 64x64\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
        "assets: assets\nshots:\n  - {photo: a, hold: 4f}\n"
    )
    muxed.clear()
    silent = render(spec_path, "final")
    assert silent.audio is False and "args" not in muxed
    assert silent.path.read_bytes() == b"cat"


@pytest.mark.gpu
def test_cached_rerender_does_not_rasterize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.chdir(tmp_path)
    calls = {"n": 0}

    def fake_lines(lines, *, scale, cache_dir, chromium=None):
        calls["n"] += 1
        from montaj.render.text import RasterLine, Unit
        out = []
        for line in lines:
            fill = torch.zeros(4, 2, 2)
            unit = Unit(line.text, fill, None, (0, 0, 2, 2), None, (0, 0, 2, 2), False)
            out.append(RasterLine((unit,), (0, 0, 2, 2), None, None, scale, False))
        return out

    monkeypatch.setattr("montaj.render.frame.raster_lines", fake_lines)
    monkeypatch.setattr("montaj.render.frame.draw_text", lambda cv, img, *a, **k: img)
    _project(
        tmp_path,
        "video:\n  size: 640x960\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
        "assets: assets\nshots:\n  - {photo: a, hold: 2f}\n"
        "text:\n  - from: 0s\n    to: 2f\n    lines:\n      - {text: Hi, y: 10, at: 0s}\n",
    )
    # _project writes four jpegs and ignores the yaml's photo list. One is enough; a.jpg exists.
    first = render(tmp_path / "montaj.yaml", "preview")
    assert (first.rendered, first.cached) == (1, 0)
    assert calls["n"] == 1
    second = render(tmp_path / "montaj.yaml", "preview")
    assert (second.rendered, second.cached) == (0, 1)
    assert calls["n"] == 1


def test_render_tells_the_renderer_which_shots_a_segment_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every segment, including a cache hit, names the shots its frames can read."""
    from contextlib import contextmanager

    assets = tmp_path / "assets"
    assets.mkdir()
    Image.new("RGB", (8, 8), (20, 40, 60)).save(assets / "a.jpg")
    Image.new("RGB", (8, 8), (60, 40, 20)).save(assets / "b.jpg")
    spec_path = tmp_path / "montaj.yaml"
    spec_path.write_text(
        "video:\n  size: 64x64\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
        "assets: assets\nshots:\n"
        "  - {photo: a, hold: 4f}\n"
        "  - {photo: b, hold: 4f, in: {type: fade, dur: 2f}}\n"
    )
    held = {"on": False}
    log: list[tuple] = []

    @contextmanager
    def fake_hold(log_path=None):
        held["on"] = True
        try:
            yield tmp_path / "lock"
        finally:
            held["on"] = False

    class FakeCV:
        def __init__(self, W, H, device="cuda"):
            assert held["on"]
            self.W, self.H = W, H

    class FakeRenderer:
        def __init__(self, tl, spec_dir, cv):
            assert held["on"]
            self.cv = cv

        def begin_segment(self, shots):
            assert held["on"]
            log.append(("begin", frozenset(shots)))

        def frame(self, f):
            assert held["on"]
            log.append(("frame", f))
            return torch.zeros(3, self.cv.H, self.cv.W)

    class FakeEnc:
        def __init__(self, path, w, h, fps, log=None):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.path.write_bytes(b"seg")

        def write(self, frame):
            return None

    def fake_concat(chunks, out, **_kwargs):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"cat")
        return Path(out)

    monkeypatch.setattr("montaj.pipeline.gpu_hold", fake_hold)
    monkeypatch.setattr("montaj.pipeline.Canvas", FakeCV)
    monkeypatch.setattr("montaj.pipeline.Renderer", FakeRenderer)
    monkeypatch.setattr("montaj.pipeline.Encoder", FakeEnc)
    monkeypatch.setattr("montaj.pipeline.concat", fake_concat)

    first = render(spec_path, "preview")
    assert (first.rendered, first.cached) == (2, 0)
    assert log == [
        ("begin", frozenset({0})),
        ("frame", 0), ("frame", 1), ("frame", 2), ("frame", 3),
        ("begin", frozenset({0, 1})),
        ("frame", 4), ("frame", 5), ("frame", 6), ("frame", 7),
    ]
    second = render(spec_path, "preview")
    assert (second.rendered, second.cached) == (0, 2)
    assert log[10:] == [("begin", frozenset({0})), ("begin", frozenset({0, 1}))]


def test_mux_two_pass_lands_a_minus_6_tone_on_minus_14(tmp_path: Path) -> None:
    """A steady tone at -6 LUFS muxes to -14 ± 0.5. Single-pass dynamic loudnorm misses this."""
    import subprocess

    from montaj import encode, qa

    log = tmp_path / "mux.log"
    seconds = 3.0
    raw = tmp_path / "raw.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"sine=frequency=1000:sample_rate=48000:duration={seconds}", str(raw)],
        check=True,
    )
    measured = qa.loudness(raw, log=log)
    assert measured is not None
    tone = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw),
         "-af", f"volume={-6.0 - measured:.4f}dB", str(tone)],
        check=True,
    )
    assert qa.loudness(tone, log=log) == pytest.approx(-6.0, abs=0.4)
    video = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"color=c=black:size=320x240:rate=25:duration={seconds}",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    out = encode.mux(video, tone, tmp_path / "m.mp4", log=log)
    assert "linear=true" in log.read_text()
    assert qa.loudness(out, log=log) == pytest.approx(-14.0, abs=0.5)
