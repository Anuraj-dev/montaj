"""Bounded renderer caches. Pixels match an unbounded cache; sizes stay at the cap.

Seam: `Renderer.frame` and `ClipSource.frame`. Cache length is part of the contract
the caps exist to enforce, so the tests read it after every frame.
"""
from __future__ import annotations

import math
import subprocess
import weakref
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from montaj.render.canvas import Canvas
from montaj.render.clip import _CACHE_FRAMES, ClipSource
from montaj.render.core import Photo
from montaj.render.frame import Renderer
from montaj.spec import load_spec
from montaj.timeline import resolve

_TIGHT = {
    "photos": 1,
    "bgs": 2,
    "cards": 1,
    "stills": 2,
    "walls": 1,
    "clips": 2,
    "captions": 1,
}


def _ffmpeg(*args: str) -> bytes:
    proc = subprocess.run(["ffmpeg", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr.decode(errors="replace")[-400:])
    return proc.stdout


def _resident(clip: ClipSource) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for entry in clip._cache.values():
        out.extend(entry)
    return out


def _oracle_frames(path: Path, n: int, w: int, h: int, clip_in: float = 0.0) -> list[torch.Tensor]:
    """Same ffmpeg invocation ClipSource uses, not ClipSource itself."""
    raw = _ffmpeg(
        "-v", "error", "-ss", f"{clip_in:.6f}", "-i", str(path),
        "-an", "-vf", f"fps=30,scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
        "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    )
    frame = w * h * 3
    assert len(raw) >= n * frame
    frames = []
    for i in range(n):
        chunk = raw[i * frame:(i + 1) * frame]
        arr = np.frombuffer(chunk, np.uint8).reshape(h, w, 3).copy()
        frames.append(torch.from_numpy(arr.transpose(2, 0, 1)))
    return frames


def _assert_uint8_cpu(frames: list[torch.Tensor]) -> None:
    assert frames
    for tensor in frames:
        assert tensor.dtype == torch.uint8
        assert tensor.device.type == "cpu"


def test_clip_longer_than_the_cap_keeps_only_cap_frames(tmp_path: Path, monkeypatch) -> None:
    n = _CACHE_FRAMES + 12
    w, h = 32, 24
    src = tmp_path / "long.mkv"
    _ffmpeg(
        "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=32x24:rate=30:duration=8",
        "-frames:v", str(n), "-c:v", "ffv1", "-pix_fmt", "rgb24", str(src),
    )
    ref = _oracle_frames(src, n, w, h)
    assert len(ref) == n and not torch.equal(ref[0], ref[_CACHE_FRAMES])

    decoded: list[int] = []
    orig = ClipSource._decode

    def spy(self, clip_in, w, h, nframes):
        got = orig(self, clip_in, w, h, nframes)
        decoded.append(len(got))
        return got

    monkeypatch.setattr(ClipSource, "_decode", spy)

    jumped = ClipSource(src, 30)
    last = jumped.frame(float(n - 1), 0.0, 0.0, w, h, nframes=n)
    assert torch.equal(last, ref[-1].float() / 255)
    stayed = _resident(jumped)
    assert len(stayed) <= _CACHE_FRAMES
    _assert_uint8_cpu(stayed)

    clip = ClipSource(src, 30)
    for i in range(n):
        got = clip.frame(float(i), 0.0, 0.0, w, h, nframes=n)
        assert torch.equal(got, ref[i].float() / 255), i
        stayed = _resident(clip)
        assert len(stayed) <= _CACHE_FRAMES
        _assert_uint8_cpu(stayed)
        if i == 0:
            # Reading the first frame must not have swallowed the rest of the hold.
            assert any(torch.equal(t, ref[0]) for t in stayed)
            assert not any(torch.equal(t, ref[_CACHE_FRAMES]) for t in stayed)
    # The whole hold can still be returned. The source, not the caller, stays bounded.
    whole = ClipSource(src, 30)
    got = whole.frames(0.0, w, h, nframes=n)
    assert len(got) == n
    for i, frame in enumerate(got):
        assert frame.dtype == torch.uint8 and torch.equal(frame, ref[i])
    assert len(_resident(whole)) <= _CACHE_FRAMES
    assert all(k <= _CACHE_FRAMES for k in decoded)


def test_two_clip_windows_share_one_frame_budget(tmp_path: Path, monkeypatch) -> None:
    """A fade reads two windows of one file. Each keeps its recent frames; decoder opens stay one per window."""
    cap = 8
    monkeypatch.setattr("montaj.render.clip._CACHE_FRAMES", cap)
    w, h, hold = 32, 24, 48
    src = tmp_path / "reel.mkv"
    _ffmpeg(
        "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=32x24:rate=30:duration=4",
        "-frames:v", "120", "-c:v", "ffv1", "-pix_fmt", "rgb24", str(src),
    )
    outgoing = _oracle_frames(src, 40, w, h, 0.0)
    incoming = _oracle_frames(src, 16, w, h, 0.5)
    assert not torch.equal(outgoing[0], incoming[0])

    opens = {"n": 0}
    orig = ClipSource._spawn

    def spy(self, clip_in, width, height, nframes):
        opens["n"] += 1
        return orig(self, clip_in, width, height, nframes)

    monkeypatch.setattr(ClipSource, "_spawn", spy)
    clip = ClipSource(src, 30)
    for i in range(25):
        got = clip.frame(float(i), 0.0, 0.0, w, h, nframes=hold)
        assert torch.equal(got, outgoing[i].float() / 255), i
        stayed = _resident(clip)
        assert len(stayed) <= cap
        _assert_uint8_cpu(stayed)
    assert opens["n"] == 1

    # motion_blur 2 samples t−0.25 and t+0.25, so each side asks for the previous index too.
    for k in range(8):
        for delta in (-0.25, 0.25):
            t = 25 + k + delta
            got = clip.frame(t, 0.0, 0.0, w, h, nframes=hold)
            assert torch.equal(got, outgoing[math.floor(t)].float() / 255)
            t_in = k + delta
            got_in = clip.frame(t_in, 0.0, 0.5, w, h, nframes=hold)
            assert torch.equal(got_in, incoming[max(0, math.floor(t_in))].float() / 255)
        stayed = _resident(clip)
        assert len(stayed) <= cap
        _assert_uint8_cpu(stayed)
        assert len(clip._cache) == 2
    assert opens["n"] == 2


def _photo(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (48, 64), rgb)
    ImageDraw.Draw(im).rectangle((6, 8, 28, 40), fill=(rgb[2], rgb[0], rgb[1]))
    im.save(path, quality=95)


def _film(tmp: Path) -> Path:
    assets = tmp / "assets"
    for i in range(12):
        _photo(assets / f"p{i:02d}.jpg", (20 + i * 18, 30 + i * 11, 220 - i * 12))
    _photo(assets / "morph.jpg", (240, 200, 40))
    shots = []
    for i in range(12):
        extra = ""
        if i >= 5:
            extra += ", frame: {tilt: [-1.5, 1.2]}"
        if i == 7:
            extra += ", morph: {photo: morph, at: 0f, dur: 3f}"
        inn = "" if i == 0 else ", in: {type: fade, dur: 2f}"
        shots.append(f"  - {{photo: p{i:02d}, hold: 3f{extra}{inn}}}")
    path = tmp / "montaj.yaml"
    path.write_text(
        "video:\n  size: 64x96\n  fps: 30\n  look: warm-film\n"
        "  background: \"#102030\"\n  motion_blur: 1\n"
        "assets: assets\nshots:\n" + "\n".join(shots) + "\n"
    )
    return path


def _open(path: Path, caps: dict[str, int]) -> tuple:
    tl = resolve(load_spec(path), path.parent)
    cv = Canvas(tl.size[0], tl.size[1], device="cpu")
    return tl, Renderer(tl, path.parent, cv, cache_caps=caps)


def _within(renderer: Renderer, caps: dict[str, int]) -> None:
    for name, cap in caps.items():
        assert len(getattr(renderer, f"_{name}")) <= cap, name


def test_tight_caps_match_an_unbounded_renderer(tmp_path: Path) -> None:
    path = _film(tmp_path)
    tl, tight = _open(path, _TIGHT)
    _, wide = _open(path, {name: 10_000 for name in _TIGHT})
    assert len(tl.shots) >= 12
    assert any(shot.spec.frame is not None for shot in tl.shots)
    assert any(shot.spec.morph is not None for shot in tl.shots)
    assert any(shot.spec.in_ is not None and shot.spec.in_.type == "fade" for shot in tl.shots)

    first = tight.frame(0)
    _within(tight, _TIGHT)
    assert torch.equal(first, wide.frame(0))
    photo = weakref.ref(tight._photos.get(("p00", None, False)))
    assert photo() is not None

    for f in range(1, tl.n_frames):
        got = tight.frame(f)
        _within(tight, _TIGHT)
        assert torch.equal(got, wide.frame(f)), f

    assert photo() is None
    assert len(wide._photos) > _TIGHT["photos"]
    assert len(wide._bgs) > _TIGHT["bgs"]
    assert len(wide._cards) > _TIGHT["cards"]
    assert len(wide._stills) > _TIGHT["stills"]


def test_segment_shots_include_the_neighbours_a_transition_touches(tmp_path: Path) -> None:
    from montaj.pipeline import _segment_shots

    assets = tmp_path / "assets"
    for stem in "abcd":
        _photo(assets / f"{stem}.jpg", (40, 80, 120))
    path = tmp_path / "montaj.yaml"
    path.write_text(
        "video:\n  size: 64x96\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
        "assets: assets\nshots:\n"
        "  - {photo: a, hold: 6f}\n"
        "  - {photo: b, hold: 8f, in: {type: fade, dur: 3f}}\n"
        "  - {photo: c, hold: 8f, in: {type: whip, dur: 4f, axis: x}}\n"
        "  - {photo: d, hold: 6f}\n"
    )
    tl = resolve(load_spec(path), tmp_path)
    assert [(seg.start, seg.end) for seg in tl.segments] == [(0, 6), (6, 14), (14, 22), (22, 28)]
    # Fade occupies the head of shot b. Whip starts in the tail of shot b and finishes in shot c.
    assert [_segment_shots(tl, seg) for seg in tl.segments] == [
        frozenset({0}),
        frozenset({0, 1, 2}),
        frozenset({1, 2}),
        frozenset({3}),
    ]


def _lifetime_film(tmp: Path) -> Path:
    assets = tmp / "assets"
    _photo(assets / "p0.jpg", (20, 40, 200))
    _photo(assets / "p1.jpg", (200, 30, 40))
    for i in range(4):
        _photo(assets / f"w{i}.jpg", (30 + 40 * i, 90, 210 - 30 * i))
    _ffmpeg(
        "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=32x24:rate=30:duration=2",
        "-frames:v", "60", "-c:v", "ffv1", "-pix_fmt", "rgb24", str(assets / "reel.mkv"),
    )
    prints = "\n".join(
        f"        - {{photo: w{i}, cell: [{i % 2}, {i // 2}], straight: true}}" for i in range(4)
    )
    path = tmp / "montaj.yaml"
    path.write_text(
        "video:\n  size: 64x96\n  fps: 30\n  look: warm-film\n"
        "  background: \"#102030\"\n  motion_blur: 2\n"
        "assets: assets\nshots:\n"
        "  - {photo: p0, hold: 3f, frame: {caption: Arshiya, tilt: [-1.5, 1.2]}}\n"
        "  - hold: 3f\n"
        "    wall:\n"
        "      grid: 200x260\n"
        "      seed: 1\n"
        "      jitter: 0\n"
        "      rot: 0\n"
        "      print_height: 80\n"
        "      prints:\n"
        f"{prints}\n"
        "      light: {center: [0, 0], radius: [400, 300]}\n"
        "      camera:\n"
        "        pos: [{t: 0f, print: w0, at: [24, 32]}]\n"
        "        zoom: [{t: 0f, value: 0.4}]\n"
        "  - {photo: p1, hold: 3f}\n"
        "  - {photo: p0, hold: 3f, frame: {caption: Arshiya, tilt: [0.6, -0.4]}}\n"
        "  - {clip: reel, hold: 8f, clip_in: 0s}\n"
        "  - {clip: reel, hold: 8f, clip_in: 0.4s, in: {type: fade, dur: 4f}}\n"
    )
    return path


def _render_segments(renderer: Renderer, tl, prepare: bool) -> list[torch.Tensor]:
    shots_for = None
    if prepare:
        from montaj.pipeline import _segment_shots
        shots_for = _segment_shots
    frames = [torch.empty(0) for _ in range(tl.n_frames)]
    for seg in tl.segments:
        if shots_for is not None:
            renderer.begin_segment(shots_for(tl, seg))
        for f in range(seg.start, seg.end):
            frames[f] = renderer.frame(f)
    return frames


def test_segment_lifetime_releases_a_wall_and_reloads_on_the_next_use(tmp_path: Path, monkeypatch) -> None:
    """A wall keeps its prints while its segment can read them, then drops them. Pixels stay put."""
    monkeypatch.setattr("montaj.render.clip._CACHE_FRAMES", 8)
    path = _lifetime_film(tmp_path)
    tl, wide = _open(path, {name: 10_000 for name in _TIGHT})
    assert tl.spec.video.motion_blur >= 2
    assert any(shot.spec.frame is not None and shot.spec.frame.caption for shot in tl.shots)
    wall_index = next(shot.index for shot in tl.shots if shot.kind == "wall")
    assert tl.shots[wall_index + 1].kind == "still"
    assert any(shot.kind == "clip" and shot.spec.clip == "reel" for shot in tl.shots)
    wide_frames = _render_segments(wide, tl, prepare=False)
    wide_wall = wide._walls.get(wall_index)
    assert wide_wall is not None
    wide_refs = [weakref.ref(layer["img"]) for layer in wide_wall.order]
    del wide_wall
    assert len(wide_refs) == 4 and len({id(ref()) for ref in wide_refs}) == 4
    assert all(ref() is not None for ref in wide_refs)

    caps = {**_TIGHT, "photos": 64, "captions": 4}
    _, life = _open(path, caps)
    inits = {"n": 0}
    seen: set[tuple] = set()
    orig_init = Photo.__init__

    def spy_init(self, cv, photo_path, crop=None, card=False, pad=90):
        inits["n"] += 1
        seen.add((str(photo_path), None if crop is None else tuple(crop), bool(card)))
        orig_init(self, cv, photo_path, crop, card, pad)

    monkeypatch.setattr(Photo, "__init__", spy_init)
    from montaj.render import frame as frame_mod

    captions = {"n": 0}
    orig_caption = frame_mod.raster_caption

    def spy_caption(*args, **kwargs):
        captions["n"] += 1
        return orig_caption(*args, **kwargs)

    monkeypatch.setattr(frame_mod, "raster_caption", spy_caption)
    opens = {"n": 0}
    orig_spawn = ClipSource._spawn

    def spy_spawn(self, clip_in, width, height, nframes):
        opens["n"] += 1
        return orig_spawn(self, clip_in, width, height, nframes)

    monkeypatch.setattr(ClipSource, "_spawn", spy_spawn)

    from montaj.pipeline import _segment_shots

    life_refs: list[weakref.ReferenceType] | None = None
    released = False
    for seg in tl.segments:
        life.begin_segment(_segment_shots(tl, seg))
        if life_refs is not None:
            assert life._walls.get(wall_index) is None
            for stem in (f"w{i}" for i in range(4)):
                assert (stem, None, True) not in life._photos
            assert all(ref() is None for ref in life_refs)
            life_refs = None
            released = True
        for f in range(seg.start, seg.end):
            got = life.frame(f)
            _within(life, caps)
            assert torch.equal(got, wide_frames[f]), f
        if seg.index == wall_index:
            wall = life._walls.get(wall_index)
            assert wall is not None
            life_refs = [weakref.ref(layer["img"]) for layer in wall.order]
            assert len(life_refs) == 4 and len({id(ref()) for ref in life_refs}) == 4
            del wall
    assert released
    # The unbounded renderer still holds the wall. The prepared one let those photos go.
    assert all(ref() is not None for ref in wide_refs)
    assert inits["n"] > len(seen)
    assert captions["n"] >= 2
    assert opens["n"] == 2
