"""Encoder routes: zero-copy NVENC from CUDA surfaces vs the ffmpeg pipe it falls back to."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from montaj import doctor, encode
from test_nv12 import FPS, H, W, decode, pattern, psnr

# Same encoder, same settings, same NV12 bytes: measured 2026-11 the two routes decode to identical
# frames (inf dB, every frame). The floor leaves room for a driver that rounds rate control apart.
MIN_ZERO_COPY_VS_PIPE_DB = 45.0
MAX_SOURCE_LOSS_DB = 0.5


# ---- route selection (CPU) ----------------------------------------------------------------


def test_route_pipe_for_non_nvenc_codecs_and_host_frames() -> None:
    assert encode.choose_route("libx264", W, H, "cuda:0") == encode.Route(encode.ROUTE_PIPE, "codec libx264")
    for device in (None, "cpu", torch.device("cpu")):
        route = encode.choose_route("h264_nvenc", W, H, device)
        assert route.name == encode.ROUTE_PIPE and not route.zero_copy, route


def test_route_pipe_for_odd_sizes_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(encode, "zero_copy_error", lambda index=0: None)
    assert encode.choose_route("h264_nvenc", W + 1, H, "cuda:0").reason == f"odd size {W + 1}x{H}"
    assert encode.choose_route("h264_nvenc", W, H, "cuda:0").zero_copy
    monkeypatch.setenv(encode.ZERO_COPY_ENV, "0")
    assert encode.choose_route("h264_nvenc", W, H, "cuda:0").reason == f"{encode.ZERO_COPY_ENV}=0"


def test_route_reports_why_zero_copy_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(encode, "zero_copy_error", lambda index=0: "NVENC session failed (x)")
    route = encode.choose_route("h264_nvenc", W, H, "cuda:0")
    assert route == encode.Route(encode.ROUTE_PIPE, "NVENC session failed (x)")
    assert str(route) == "ffmpeg-pipe: NVENC session failed (x)"


def test_encoder_exposes_and_logs_its_route(tmp_path: Path) -> None:
    log = tmp_path / "e.log"
    with encode.Encoder(tmp_path / "x.mp4", W, H, FPS, encoder="libx264", log=log, device="cuda:0") as enc:
        enc.write(pattern(1)[0])
    assert enc.route == encode.Route(encode.ROUTE_PIPE, "codec libx264")
    assert "encode route ffmpeg-pipe: codec libx264" in log.read_text()


def test_doctor_reports_the_encode_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(doctor, "prefer_encoder", lambda: "h264_nvenc")
    monkeypatch.setattr(encode, "zero_copy_error", lambda index=0: None)
    assert doctor.encode_route_line().startswith("OK encode route: nvenc-cuda: PyNvVideoCodec")
    monkeypatch.setattr(encode, "zero_copy_error", lambda index=0: "PyNvVideoCodec unavailable (x)")
    assert doctor.encode_route_line() == "WARN encode route: ffmpeg-pipe: PyNvVideoCodec unavailable (x)"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert doctor.encode_route_line() == "WARN encode route: ffmpeg-pipe: no CUDA device"


# ---- zero copy (GPU) ----------------------------------------------------------------------


def _need_zero_copy() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    error = encode.zero_copy_error(torch.cuda.current_device())
    if error is not None:
        pytest.skip(error)


def _gpu_frames(n: int) -> tuple[list[np.ndarray], list[torch.Tensor]]:
    src = pattern(n)
    return src, [torch.from_numpy(f).cuda().permute(2, 0, 1).float() / 255.0 for f in src]


def _encode(path: Path, frames, device: str | None, size: tuple[int, int] = (W, H)) -> encode.Encoder:
    with encode.Encoder(path, *size, FPS, encoder="h264_nvenc", log=path.with_suffix(".log"), device=device) as enc:
        for f in frames:
            enc.write(f)
    return enc


def _ffprobe(path: Path, *args: str) -> dict:
    out = subprocess.run([encode.FFPROBE, "-v", "error", *args, "-of", "json", str(path)],
                         stdout=subprocess.PIPE, check=True).stdout
    return json.loads(out)


@pytest.mark.gpu
def test_zero_copy_decodes_like_the_pipe(tmp_path: Path) -> None:
    """60 frames (two B-frame groups past the first GOP's start) through both routes."""
    _need_zero_copy()
    src, gpu = _gpu_frames(60)
    zc = _encode(tmp_path / "zc.mp4", gpu, "cuda")
    pipe = _encode(tmp_path / "pipe.mp4", gpu, None)
    assert zc.route.zero_copy and not pipe.route.zero_copy
    a, b, ref = decode(tmp_path / "zc.mp4"), decode(tmp_path / "pipe.mp4"), np.stack(src)
    assert a.shape == b.shape == ref.shape
    worst = min(psnr(x, y) for x, y in zip(a, b))
    assert worst >= MIN_ZERO_COPY_VS_PIPE_DB, worst
    assert psnr(a, ref) >= psnr(b, ref) - MAX_SOURCE_LOSS_DB


@pytest.mark.gpu
def test_zero_copy_stream_tags_and_timing(tmp_path: Path) -> None:
    _need_zero_copy()
    _, gpu = _gpu_frames(30)
    out = tmp_path / "zc.mp4"
    _encode(out, gpu, "cuda")
    s = _ffprobe(out, "-show_streams", "-select_streams", "v:0")["streams"][0]
    assert (s["codec_name"], s["profile"], s["pix_fmt"]) == ("h264", "High", "yuv420p")
    assert (s["color_range"], s["color_space"], s["color_transfer"], s["color_primaries"]) == ("tv",) + ("bt709",) * 3
    assert (s["r_frame_rate"], int(s["nb_frames"])) == (f"{FPS}/1", 30)
    pts = sorted(int(p["pts"]) for p in _ffprobe(out, "-show_packets", "-select_streams", "v:0")["packets"])
    step = pts[1] - pts[0]
    assert pts == [pts[0] + i * step for i in range(30)] and pts[0] == 0
    assert not list(tmp_path.glob(".*.es.mp4")), "elementary-stream temp file left behind"


def _forbid_host_copies(m: pytest.MonkeyPatch) -> None:
    """Every torch path that brings a CUDA tensor to host memory raises. (The CUDA profiler would
    count DtoH memcpys directly, but CUPTI fails to initialise on this Blackwell driver.)"""

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("frame copied to the host")

    for name in ("cpu", "numpy", "tolist", "item", "__array__"):
        m.setattr(torch.Tensor, name, boom)
    to, copy_ = torch.Tensor.to, torch.Tensor.copy_

    def guarded_to(self: torch.Tensor, *args: object, **kw: object) -> torch.Tensor:
        out = to(self, *args, **kw)
        if self.is_cuda and not out.is_cuda:
            boom()
        return out

    def guarded_copy(self: torch.Tensor, src: torch.Tensor, *args: object, **kw: object) -> torch.Tensor:
        if src.is_cuda and not self.is_cuda:
            boom()
        return copy_(self, src, *args, **kw)

    m.setattr(torch.Tensor, "to", guarded_to)
    m.setattr(torch.Tensor, "copy_", guarded_copy)


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cuda", None])
def test_zero_copy_never_copies_frames_to_the_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   device: str | None) -> None:
    """The write loop on the zero-copy route never touches host memory; the same guard catches the
    pipe route's copy, so it is not vacuous."""
    _need_zero_copy()
    _, gpu = _gpu_frames(20)
    enc = encode.Encoder(tmp_path / "x.mp4", W, H, FPS, encoder="h264_nvenc", log=tmp_path / "e.log", device=device)
    with monkeypatch.context() as m:
        _forbid_host_copies(m)
        if enc.route.zero_copy:
            for t in gpu:
                enc.write(t)
            torch.cuda.synchronize()
        else:
            with pytest.raises(AssertionError, match="host"):
                enc.write(gpu[0])
    enc.close()
    if device is not None:
        assert encode.frame_count(tmp_path / "x.mp4") == 20


@pytest.mark.gpu
def test_zero_copy_chunks_concat_to_exact_frames(tmp_path: Path) -> None:
    """Uneven zero-copy segments join with uniform pts (46+15+23 = 84), like the pipe's."""
    _need_zero_copy()
    _, gpu = _gpu_frames(46)
    counts = (46, 15, 23)
    chunks = [_encode(tmp_path / f"c{n}.mp4", gpu[:n], "cuda").path for n in counts]
    out = encode.concat(chunks, tmp_path / "joined.mp4", fps=FPS, frames=counts, log=tmp_path / "c.log")
    pts = sorted(int(p["pts"]) for p in _ffprobe(out, "-show_packets", "-select_streams", "v:0")["packets"])
    assert len(pts) == 84
    assert len({b - a for a, b in zip(pts, pts[1:])}) == 1


@pytest.mark.gpu
def test_zero_copy_rejects_sizes_below_the_nvenc_minimum(tmp_path: Path) -> None:
    _need_zero_copy()
    frame = torch.zeros(3, 128, 128, device="cuda")
    with pytest.raises(encode.EncodeError):
        _encode(tmp_path / "small.mp4", [frame], "cuda", size=(128, 128))
    assert not list(tmp_path.glob(".*.es.mp4"))


@pytest.mark.gpu
def test_zero_copy_keeps_stdout_quiet(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """PyNvVideoCodec logs INFO lines to stdout unless LOGGER_LEVEL is set before its import."""
    _need_zero_copy()
    _, gpu = _gpu_frames(5)
    capfd.readouterr()
    _encode(tmp_path / "zc.mp4", gpu, "cuda")
    assert capfd.readouterr().out == ""


@pytest.mark.gpu
def test_zero_copy_abort_leaves_no_files(tmp_path: Path) -> None:
    _need_zero_copy()
    _, gpu = _gpu_frames(3)
    out = tmp_path / "zc.mp4"
    with pytest.raises(RuntimeError, match="boom"):
        with encode.Encoder(out, W, H, FPS, encoder="h264_nvenc", log=tmp_path / "e.log", device="cuda") as enc:
            for t in gpu:
                enc.write(t)
            raise RuntimeError("boom")
    assert not out.exists() and not list(tmp_path.glob(".*.es.mp4"))
