"""NVDEC clip decode matches the ffmpeg reader: same frame index, colour within a pinned PSNR."""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from montaj.render import clip as clip_mod
from montaj.render import nvdec
from montaj.render.clip import ClipSource

# Measured minimum on these clips is 52.2 dB (sharp, saturated testsrc2 edges); project
# footage measured 49.4-58.4 dB. The rest is swscale's fixed-point rounding.
_MIN_PSNR = 50.0


def _ffmpeg(*args: str) -> None:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr.decode(errors="replace")[-400:])


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    """Short GOPs and B-frames so a seek lands mid-GOP; one bt709-tagged, one untagged (bt601)."""
    d = tmp_path_factory.mktemp("nvdec")
    h264 = d / "h264.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=3", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-g", "12", "-bf", "2", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709", str(h264))
    hevc = d / "hevc.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=480x848:rate=30:duration=3", "-c:v", "libx265",
            "-x265-params", "log-level=error:keyint=15:bframes=3", "-pix_fmt", "yuv420p", str(hevc))
    return {"h264": h264, "hevc": hevc}


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def _both(path: Path, clip_in: float, w: int, h: int, n: int, crop=None):
    """(ffmpeg frames, nvdec frames) as uint8 CUDA tensors."""
    proc, _err, thread = ClipSource(path, 30)._spawn(clip_in, w, h, n, crop)
    raw = proc.stdout.read()
    proc.wait()
    thread.join()
    ref = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy()).permute(0, 3, 1, 2).cuda()
    choice = nvdec.choose(path, "cuda")
    assert choice.gpu, choice.reason
    reader = nvdec.NvdecReader(path, choice, clip_in, 30, w, h, n, crop, "cuda")
    got = []
    while (frame := reader.read()) is not None:
        got.append(reader.expand(frame))
    reader.close()
    return ref, torch.stack(got) if got else torch.empty(0, 3, h, w, dtype=torch.uint8, device="cuda")


CASES = [
    # (codec, clip_in, w, h, nframes, crop)
    ("h264", 0.0, 270, 480, 30, None),          # cover: portrait out of landscape, 24 -> 30 fps
    ("h264", 1.37, 320, 180, 30, None),         # seek mid-GOP
    ("h264", 2.5, 270, 480, 40, None),          # runs past EOF: count comes from ffmpeg's fps tail
    ("h264", 0.9, 200, 150, 20, (101, 53, 501, 353)),  # crop/camera box
    ("hevc", 0.0, 540, 960, 30, None),
    ("hevc", 1.7, 270, 480, 20, (30, 100, 430, 811)),
]


@pytest.mark.gpu
@pytest.mark.parametrize("codec,clip_in,w,h,n,crop", CASES)
def test_nvdec_matches_ffmpeg_frame_for_frame(clips, codec, clip_in, w, h, n, crop) -> None:
    ref, got = _both(clips[codec], clip_in, w, h, n, crop)
    assert got.shape == ref.shape
    for i in range(len(ref)):
        own = _psnr(got[i], ref[i])
        assert own >= _MIN_PSNR, (i, own)
        # Exact index: no neighbouring ffmpeg frame is a better match than frame i.
        for j in (i - 1, i + 1):
            if 0 <= j < len(ref) and not torch.equal(ref[j], ref[i]):
                assert own > _psnr(got[i], ref[j]), (i, j)


@pytest.mark.gpu
def test_clip_source_on_cuda_decodes_with_nvdec(clips) -> None:
    lines: list[str] = []
    gpu = ClipSource(clips["h264"], 30, device="cuda", log=lines.append)
    cpu = ClipSource(clips["h264"], 30)
    assert gpu.decoder.gpu and not cpu.decoder.gpu
    assert lines == [f"clip h264.mp4 decode nvdec: {gpu.decoder.reason}"]
    a = gpu.frames(0.5, 160, 90, 12)
    b = cpu.frames(0.5, 160, 90, 12)
    assert all(t.is_cuda and t.dtype == torch.uint8 for t in a)
    assert len(a) == len(b)
    assert min(_psnr(x, y.cuda()) for x, y in zip(a, b)) >= _MIN_PSNR
    gpu.close()


@pytest.mark.gpu
def test_choose_falls_back_for_codecs_nvdec_cannot_read(tmp_path) -> None:
    ffv1 = tmp_path / "a.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1", "-c:v", "ffv1", str(ffv1))
    choice = nvdec.choose(ffv1, "cuda")
    assert choice.name == "ffmpeg" and "ffv1" in choice.reason


def test_choose_on_cpu_is_ffmpeg(tmp_path) -> None:
    choice = nvdec.choose(tmp_path / "missing.mp4", "cpu")
    assert (choice.name, choice.gpu) == ("ffmpeg", False)


def test_failed_nvdec_open_demotes_the_source_to_ffmpeg(tmp_path, monkeypatch) -> None:
    src = tmp_path / "a.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1", "-c:v", "ffv1",
            "-pix_fmt", "rgb24", str(src))
    monkeypatch.setattr(clip_mod.nvdec, "choose", lambda path, device: nvdec.Choice("nvdec", "forced"))

    def boom(*args, **kwargs):
        raise RuntimeError("no decoder")

    monkeypatch.setattr(clip_mod.nvdec, "NvdecReader", boom)
    lines: list[str] = []
    source = ClipSource(src, 30, log=lines.append)
    frames = source.frames(0, 64, 48, 5)
    assert len(frames) == 5 and not frames[0].is_cuda
    assert source.decoder.name == "ffmpeg"
    assert lines[-1].startswith("clip a.mkv decode ffmpeg: NVDEC open failed (RuntimeError: no decoder)")


@pytest.mark.gpu
def test_nvdec_decode_keeps_stdout_clean(clips) -> None:
    """PyNvVideoCodec logs INFO to stdout unless quieted before import; commands print one line."""
    code = ("from montaj.render import nvdec; import sys; p = sys.argv[1]; "
            "r = nvdec.NvdecReader(p, nvdec.choose(p, 'cuda'), 0.5, 30, 64, 36, 3, None, 'cuda'); "
            "[r.read() for _ in range(3)]; r.close()")
    proc = subprocess.run([sys.executable, "-c", code, str(clips["h264"])], capture_output=True, text=True,
                          env={k: v for k, v in os.environ.items() if k != "LOGGER_LEVEL"})
    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stdout == ""


@pytest.mark.gpu
def test_window_cache_keeps_the_smaller_of_nv12_and_rgb(clips) -> None:
    """640x360 source: an upscaled window caches NV12 surfaces, a downscaled one RGB; reads are RGB."""
    gpu = ClipSource(clips["h264"], 30, device="cuda")
    cpu = ClipSource(clips["h264"], 30)
    for w, h, stored in ((540, 960, (540, 640)), (160, 90, (3, 90, 160))):
        a = gpu.frames(0.5, w, h, 12)
        window = gpu._cache[clip_mod.window_key(0.5, w, h, 12)]
        held = list(window._frames.values())
        assert all(t.dtype == torch.uint8 and tuple(t.shape) == stored for t in held)
        # 24 -> 30 fps repeats a frame by reference instead of storing it twice.
        assert len({t.data_ptr() for t in held}) < len(held) if len(stored) == 2 else True
        assert all(t.shape == (3, h, w) for t in a)
        b = cpu.frames(0.5, w, h, 12)
        assert min(_psnr(x, y.cuda()) for x, y in zip(a, b)) >= _MIN_PSNR
        assert torch.equal(window[0], a[0])
    gpu.close()
