"""ClipSource decodes a cover-centred clip. Frame i is source frame i."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest
import torch

from montaj.render.clip import _CACHE_FRAMES, ClipSource


def _ffmpeg(*args: str) -> bytes:
    proc = subprocess.run(["ffmpeg", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr.decode(errors="replace")[-400:])
    return proc.stdout


def _raw_frame(path, index: int) -> torch.Tensor:
    raw = _ffmpeg(
        "-v", "error", "-i", str(path),
        "-vf", f"select=eq(n\\,{index})", "-fps_mode", "vfr", "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    )
    arr = np.frombuffer(raw, np.uint8).reshape(240, 320, 3).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255


def _src(tmp_path):
    src = tmp_path / "src.mkv"
    _ffmpeg(
        "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=2",
        "-frames:v", "60", "-c:v", "ffv1", "-pix_fmt", "rgb24", str(src),
    )
    return src


def test_clip_frame_is_source_frame(tmp_path) -> None:
    src = _src(tmp_path)
    clip = ClipSource(src, 30)
    frames = clip.frames(0.0, 320, 240, hold=2.0)
    assert len(frames) == 60
    assert frames[0].dtype == torch.uint8 and frames[0].device.type == "cpu"
    assert frames[0].shape == (3, 240, 320)
    assert not torch.equal(frames[0], frames[1])
    for i in (0, 1, 17, 59):
        # floor(t − start) with a fractional t still lands on source frame i.
        got = clip.frame(100 + i + 0.9, 100, 0.0, 320, 240, nframes=60)
        assert got.dtype == torch.float32 and got.device.type == "cpu"
        assert torch.equal(got, frames[i].float() / 255)
        assert torch.equal(got, _raw_frame(src, i))
        assert frames[i].dtype == torch.uint8
    assert torch.equal(clip.frame(99.9, 100, 0.0, 320, 240, hold=2.0), frames[0].float() / 255)
    assert torch.equal(clip.frame(1000, 100, 0.0, 320, 240, nframes=60), frames[-1].float() / 255)
    # `-ss` is before `-i`, so decoded frame 0 is one second into the file.
    later = clip.frames(1.0, 320, 240, hold=1.0)
    assert len(later) == 30
    assert later[0].dtype == torch.uint8
    assert torch.equal(later[0].float() / 255, frames[30].float() / 255)


def test_clip_decodes_the_hold_and_bounds_the_cache(tmp_path) -> None:
    src = _src(tmp_path)
    clip = ClipSource(src, 30)
    # A 5-frame hold of a 60-frame source does not keep the tail.
    short = clip.frames(0.0, 320, 240, nframes=5)
    assert len(short) == 5
    assert sum(len(v) for v in clip._cache.values()) == 5
    first = clip.frames(0.0, 160, 120, nframes=60)
    second = clip.frames(0.0, 80, 60, hold=2.0)
    assert len(first) == 60 and len(second) == 60
    # 5 + 60 + 60 exceeds the cap, so only the newest window stays.
    assert _CACHE_FRAMES < len(first) + len(second)
    assert list(clip._cache) == [(0.0, 80, 60, 60)]
    assert sum(len(v) for v in clip._cache.values()) <= _CACHE_FRAMES
    # Eviction drops the cache entry, not the tensors already returned.
    assert short[0].shape == (3, 240, 320) and first[0].dtype == torch.uint8
    cached = clip._cache[(0.0, 80, 60, 60)][0]
    got = clip.frame(0.0, 0.0, 0.0, 80, 60, hold=2.0)
    assert got.dtype == torch.float32 and torch.equal(got, second[0].float() / 255)
    assert cached.dtype == torch.uint8 and torch.equal(cached, second[0])
    with pytest.raises(TypeError):
        clip.frames(0.0, 8, 8)
    with pytest.raises(ValueError):
        clip.frames(0.0, 8, 8, nframes=10, hold=1.0)
