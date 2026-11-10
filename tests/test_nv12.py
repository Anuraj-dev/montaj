"""RGB -> NV12 on the encoder side: decode-back PSNR against the former rgb24 + swscale path."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from montaj import encode

W, H, FPS = 320, 240, 25
# Lossless x264 isolates the colour conversion from coding loss. Measured 2026-11: new vs old
# decoded frames 47.8 dB (the only difference is chroma siting); vs the source both ~32 dB
# (4:2:0 on a noisy pattern). Untagged NV12 input measured 26.9 dB here: ffmpeg re-converted it.
MIN_NEW_VS_OLD_DB = 44.0
# The new path must not be worse than swscale against the source RGB.
MAX_SOURCE_LOSS_DB = 0.5
# NVENC at cq 16 adds coding loss on both sides, so the floor is lower.
MIN_NVENC_NEW_VS_OLD_DB = 38.0


def pattern(n: int, seed: int = 0) -> list[np.ndarray]:
    """Photo-like frames: smooth colour gradients, hard edges, a saturated disc and mild noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = []
    for i in range(n):
        r = 0.5 + 0.5 * np.sin(xx / 37.0 + i * 0.3)
        g = yy / H
        b = 0.5 + 0.5 * np.cos((xx + yy) / 53.0 - i * 0.2)
        img = np.stack([r, g, b], axis=-1)
        img[(xx - 160 - 4 * i) ** 2 + (yy - 120) ** 2 < 50 ** 2] = (0.95, 0.1, 0.2)
        img[:, 40 + i: 44 + i] = 1.0
        img += rng.normal(0.0, 0.01, img.shape)
        out.append((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    return out


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def encode_rgb24(out: Path, frames: list[np.ndarray], codec_args: list[str]) -> Path:
    """The pre-NV12 path, verbatim: rgb24 on stdin, swscale converts to yuv420p."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           *codec_args, "-pix_fmt", "yuv420p", *encode.BT709_ARGS, *encode.BT709_BSF, str(out)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    return out


def encode_nv12(out: Path, frames, codec: str, log: Path, **kw) -> Path:
    with encode.Encoder(out, W, H, FPS, encoder=codec, log=log, **kw) as enc:
        for f in frames:
            enc.write(f)
    return out


def test_nv12_layout_and_bt709_limited_levels() -> None:
    """White -> Y 235, black -> Y 16, grey chroma 128; a pure-red 2×2 block -> Cr 240."""
    img = np.zeros((4, 6, 3), np.uint8)
    img[:, :2] = 255
    img[:2, 2:4] = (255, 0, 0)
    nv = encode.rgb_to_nv12(img).numpy()
    assert nv.shape == (4 * 6 + 2 * 3 * 2,)
    y, uv = nv[:24].reshape(4, 6), nv[24:].reshape(2, 3, 2)
    assert y[0, 0] == 235 and y[3, 5] == 16
    assert tuple(uv[0, 0]) == (128, 128) and tuple(uv[1, 2]) == (128, 128)
    assert y[0, 2] == round(16 + 219 * 0.2126)
    assert tuple(uv[0, 1]) == (round(128 - 224 * 0.2126 / 1.8556), 240)


def test_float_chw_and_uint8_hwc_agree() -> None:
    img = pattern(1)[0]
    chw = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    assert torch.equal(encode.rgb_to_nv12(img), encode.rgb_to_nv12(chw))


def test_odd_size_buffer_matches_ffmpeg_nv12() -> None:
    """ffmpeg's nv12 frame size for odd sides is W*H + 2*ceil(W/2)*ceil(H/2)."""
    assert encode.rgb_to_nv12(np.zeros((5, 7, 3), np.uint8)).numel() == 35 + 2 * 4 * 3


def test_encoder_rejects_float_wrong_shape(tmp_path: Path) -> None:
    with encode.Encoder(tmp_path / "bad.mp4", W, H, FPS, encoder="libx264", log=tmp_path / "e.log") as enc:
        with pytest.raises(encode.EncodeError):
            enc.write(torch.zeros(3, H, W + 2))


def test_decode_back_psnr_vs_rgb24_path_lossless(tmp_path: Path) -> None:
    src = pattern(4)
    old = decode(encode_rgb24(tmp_path / "old.mp4", src, ["-c:v", "libx264", "-qp", "0", "-preset", "fast"]))
    new = decode(encode_nv12(tmp_path / "new.mp4", src, "libx264", tmp_path / "e.log", crf=0, preset="fast"))
    ref = np.stack(src)
    assert new.shape == old.shape == ref.shape
    assert psnr(new, old) >= MIN_NEW_VS_OLD_DB, psnr(new, old)
    assert psnr(new, ref) >= psnr(old, ref) - MAX_SOURCE_LOSS_DB, (psnr(new, ref), psnr(old, ref))


def test_lossless_planes_are_exactly_ours(tmp_path: Path) -> None:
    """ffmpeg passes our NV12 through untouched: decoded Y/U/V equal rgb_to_nv12's bytes."""
    src = pattern(1)
    out = encode_nv12(tmp_path / "new.mp4", src, "libx264", tmp_path / "e.log", crf=0, preset="fast")
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(out),
         "-f", "rawvideo", "-pix_fmt", "nv12", "-"],
        stdout=subprocess.PIPE, check=True,
    ).stdout
    assert np.array_equal(np.frombuffer(raw, np.uint8), encode.rgb_to_nv12(src[0]).numpy())


@pytest.mark.gpu
def test_decode_back_psnr_nvenc_from_cuda_float(tmp_path: Path) -> None:
    """The render path: float CHW on the GPU, converted there, encoded by NVENC."""
    if not encode.nvenc_usable():
        pytest.skip("h264_nvenc cannot encode here")
    src = pattern(6)
    gpu = [torch.from_numpy(f).cuda().permute(2, 0, 1).float() / 255.0 for f in src]
    old = decode(encode_rgb24(tmp_path / "old.mp4", src, encode.vcodec_args("h264_nvenc")))
    new = decode(encode_nv12(tmp_path / "new.mp4", gpu, "h264_nvenc", tmp_path / "e.log"))
    ref = np.stack(src)
    assert psnr(new, old) >= MIN_NVENC_NEW_VS_OLD_DB, psnr(new, old)
    assert psnr(new, ref) >= psnr(old, ref) - MAX_SOURCE_LOSS_DB, (psnr(new, ref), psnr(old, ref))
    # CUDA and CPU float arithmetic may round a .5 boundary differently, never by more than 1.
    diff = encode.rgb_to_nv12(gpu[0]).cpu().int() - encode.rgb_to_nv12(src[0]).int()
    assert diff.abs().max() <= 1
