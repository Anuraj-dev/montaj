"""ffmpeg/ffprobe seam: RGB frames in (converted to NV12 here), finished MP4s out.

Every function that shells out to ffmpeg tees stderr into a log file
(`<log argument>` or `build/<name>.log`) and returns parsed data, never prints.
A non-zero ffmpeg exit raises `EncodeError` carrying the last stderr lines.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from montaj.gpu_policy import GpuPolicy

FFMPEG = os.environ.get("MONTAJ_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("MONTAJ_FFPROBE", "ffprobe")

# film2 / render.sh encode settings; the fallback keeps visual parity on machines without NVENC.
NVENC_CQ = 16
NVENC_ARGS = ("-preset", "p7", "-tune", "hq", "-rc", "vbr", "-b:v", "0", "-profile:v", "high")
X264_CRF = 16
X264_PRESET = "slow"
AUDIO_ARGS = ("-c:a", "aac", "-ar", "48000")
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"
# The film is rendered in sRGB/bt709 and must say so, or players guess. ffmpeg's h264 encoders in
# this build drop -color_primaries/-color_trc (only -colorspace survives to the container), so the
# VUI goes in through the bitstream filter — which is also what a stream copy needs to carry it.
BT709_ARGS = ("-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709")
BT709_BSF = ("-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1")
NVENC_PROBE_SIZE = "320x240"  # NVENC rejects frames below ~145 px a side

# Zero-copy route: PyNvVideoCodec encodes the CUDA NV12 surface in-process. These keys mirror
# NVENC_ARGS (ffmpeg's -g default for h264_nvenc is 250; P7 picks the same 3 B-frames and High
# profile either way). Unknown keys are silently ignored by PyNvVideoCodec, so only known ones go here.
NVENC_DIRECT_CONFIG = {
    "preset": "P7", "tuning_info": "high_quality", "rc": "vbr", "bitrate": "0", "maxbitrate": "0",
    "gop": "250",
}
# Its elementary stream has no colour VUI and no limited-range flag; the remux writes both.
NVENC_DIRECT_BSF = ("-bsf:v", BT709_BSF[1] + ":video_full_range_flag=0")
ROUTE_ZERO_COPY = "nvenc-cuda"
ROUTE_PIPE = "ffmpeg-pipe"
ZERO_COPY_ENV = "MONTAJ_ZERO_COPY"  # "0" forces the ffmpeg pipe (A/B timing, debugging)

_MOV_EXTS = {".mp4", ".m4v", ".mov"}


class EncodeError(RuntimeError):
    """ffmpeg/ffprobe failed. `tail` holds the last stderr lines."""

    def __init__(self, what: str, tail: str = "") -> None:
        super().__init__(f"{what}: {tail.strip()[-800:]}" if tail else what)
        self.what = what
        self.tail = tail


def log_file(name: str, log: Path | str | None = None) -> Path:
    """Log destination: explicit path, else `build/<name>.log`. Always created."""
    path = Path(log) if log is not None else Path("build") / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def tail_lines(text: str, n: int = 6) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _append_log(path: Path, cmd: Sequence[str], stderr: bytes) -> str:
    """Add the invocation and its stderr to the log; return the decoded stderr."""
    text = stderr.decode("utf-8", "replace")
    with open(path, "ab") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n".encode())
        fh.write(stderr)
    return text


def run(
    cmd: Sequence[str],
    *,
    log: Path | str | None = None,
    desc: str = "ffmpeg",
    check: bool = True,
) -> tuple[int, str]:
    """Run `cmd`, append its stderr to the log. Returns (returncode, stderr)."""
    proc = subprocess.run(list(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    stderr = _append_log(log_file(desc, log), cmd, proc.stderr)
    if check and proc.returncode != 0:
        raise EncodeError(f"{desc} failed ({proc.returncode})", tail_lines(stderr))
    return proc.returncode, stderr


def capture(cmd: Sequence[str], *, log: Path | str | None = None, desc: str = "ffmpeg") -> str:
    """Run `cmd` and return stdout (stderr still goes to the log)."""
    proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stderr = _append_log(log_file(desc, log), cmd, proc.stderr)
    if proc.returncode != 0:
        raise EncodeError(f"{desc} failed ({proc.returncode})", tail_lines(stderr))
    return proc.stdout.decode("utf-8", "replace")


def tool_version(tool: str) -> str | None:
    """First line of `<tool> -version`, or None when it is missing/unusable."""
    try:
        out = subprocess.run([tool, "-version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.decode("utf-8", "replace").splitlines()
    return first[0].strip() if first else None


@cache
def has_encoder(name: str) -> bool:
    """Probe `ffmpeg -encoders` once per process; NVENC presence does not change mid-render."""
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return False
    return any(line.split()[1:2] == [name] for line in out.stdout.decode("utf-8", "replace").splitlines() if line.strip())


@cache
def nvenc_usable() -> bool:
    """Listed *and* able to encode a frame. A machine without a working driver lists h264_nvenc
    and then fails at encode time, so capability is a runtime question, not a build-flag question."""
    if not has_encoder("h264_nvenc"):
        return False
    try:
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", f"color=c=black:size={NVENC_PROBE_SIZE}", "-frames:v", "1",
             "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def vcodec_args(encoder: str, *, cq: int = NVENC_CQ, crf: int = X264_CRF, preset: str = X264_PRESET) -> list[str]:
    """Encoder flags for `encoder`; NVENC keeps film2's settings, libx264 is the parity fallback."""
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", *NVENC_ARGS, "-cq", str(cq)]
    if encoder == "libx264":
        return ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]
    return ["-c:v", encoder]


NVENC_MIN_SIDE = 160  # NVENC rejects frames under ~145 px a side (docs/STATE.md); round up


def nvenc_hint() -> str:
    """Why NVENC is not in use here, for a strict-mode error."""
    if not has_encoder("h264_nvenc"):
        return "this ffmpeg has no h264_nvenc; install an ffmpeg built with NVENC (`montaj doctor`)"
    return "h264_nvenc is listed but cannot encode (check the NVIDIA driver; `montaj doctor`)"


def prefer_encoder() -> str:
    """h264_nvenc when it actually encodes here, else libx264."""
    return "h264_nvenc" if nvenc_usable() else "libx264"


@dataclass(frozen=True)
class MediaInfo:
    """`duration` is the video stream's, never the container's: a long music track must not make
    the film look long. `audio_duration` stays independent so `av_offset` can expose a mismatch."""

    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    audio_codec: str | None
    audio_duration: float | None
    sample_rate: int | None

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def av_offset(self) -> float | None:
        """Audio minus video seconds, or None when either length is unknown."""
        if not self.has_audio or self.audio_duration is None:
            return None
        return self.audio_duration - self.duration


def _ratio(text: str | None) -> float:
    if not text or "/" not in text:
        return 0.0
    num, _, den = text.partition("/")
    try:
        return float(num) / float(den) if float(den) else 0.0
    except ValueError:
        return 0.0


def probe(path: Path | str, *, log: Path | str | None = None) -> MediaInfo:
    """Container/stream summary from ffprobe."""
    path = Path(path)
    if not path.exists():
        raise EncodeError(f"probe: {path} does not exist")
    raw = capture(
        [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        log=log,
        desc="ffprobe",
    )
    data = json.loads(raw or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    def _dur(*candidates: str | None) -> float:
        for value in candidates:
            if value and value != "N/A":
                return float(value)
        return 0.0

    return MediaInfo(
        path=path,
        duration=_dur(video.get("duration"), fmt.get("duration")),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate")),
        has_audio=audio is not None,
        audio_codec=(audio or {}).get("codec_name"),
        audio_duration=_dur((audio or {}).get("duration")) if audio else None,
        sample_rate=int((audio or {}).get("sample_rate") or 0) or None,
    )


def frame_count(path: Path | str, *, log: Path | str | None = None) -> int:
    """Exact decoded frame count (ffprobe `-count_frames`), not the container's estimate."""
    out = capture(
        [
            FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nk=1:nw=1", str(path),
        ],
        log=log,
        desc="ffprobe",
    )
    for line in out.splitlines():
        if line.strip().isdigit():
            return int(line.strip())
    raise EncodeError(f"frame_count: no frame count for {path}", out)


def rgb_to_nv12(frame: torch.Tensor | np.ndarray) -> torch.Tensor:
    """R'G'B' frame -> NV12 bytes (bt709 matrix, limited range), on the frame's own device.

    Takes the renderer's (3, H, W) float in 0..1 or an (H, W, 3) uint8 image. Returns a flat uint8
    tensor: the H×W luma plane, then ceil(H/2)×ceil(W/2) interleaved CbCr pairs. Chroma is the 2×2
    box mean (centre-sited); an odd edge is replicated so the buffer is the size ffmpeg expects and
    odd sizes fail (or not) in the encoder exactly as they did with rgb24.
    """
    t = torch.as_tensor(frame)
    if t.dtype == torch.uint8:
        t = t.permute(2, 0, 1).float() / 255.0
    rgb = t.float().clamp(0.0, 1.0)
    _, H, W = rgb.shape
    luma = 16.0 + 219.0 * (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    # Cb/Cr are linear in R'G'B', so subsampling RGB first equals subsampling Cb/Cr, at a quarter
    # of the arithmetic.
    sub = rgb.unsqueeze(0)
    if H % 2 or W % 2:
        sub = F.pad(sub, (0, W % 2, 0, H % 2), mode="replicate")
    sub = F.avg_pool2d(sub, 2).squeeze(0)
    ys = 0.2126 * sub[0] + 0.7152 * sub[1] + 0.0722 * sub[2]
    cb = 128.0 + 224.0 / 1.8556 * (sub[2] - ys)
    cr = 128.0 + 224.0 / 1.5748 * (sub[0] - ys)
    chroma = torch.stack((cb, cr), dim=-1)
    # Every value is inside 16..240 by construction; +0.5 then truncation rounds to nearest.
    return torch.cat((luma.reshape(-1), chroma.reshape(-1))).add_(0.5).to(torch.uint8)


@dataclass(frozen=True)
class Route:
    """Which path an `Encoder` sends frames down, and why. `choose_route` is the only place that
    decides; `Encoder.route` and the `encode route …` log line report it."""

    name: str  # ROUTE_ZERO_COPY or ROUTE_PIPE
    reason: str

    @property
    def zero_copy(self) -> bool:
        return self.name == ROUTE_ZERO_COPY

    def __str__(self) -> str:
        return f"{self.name}: {self.reason}"


def _pynvc():
    """Import PyNvVideoCodec. Its native logger prints INFO lines to stdout and reads LOGGER_LEVEL
    once, at import; Montaj commands keep stdout to one result line, so quiet it first."""
    os.environ.setdefault("LOGGER_LEVEL", "ERROR")
    import PyNvVideoCodec

    return PyNvVideoCodec


def _nvenc_direct_config(fps: float, cq: int, device_index: int) -> dict[str, str]:
    return {"codec": "h264", **NVENC_DIRECT_CONFIG, "cq": str(cq), "fps": f"{fps:g}", "gpu_id": str(device_index)}


@cache
def zero_copy_error(device_index: int = 0) -> str | None:
    """None when PyNvVideoCodec can open an NVENC session on this CUDA device, else why not.
    Probed once per process, like `nvenc_usable`."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        nvc = _pynvc()
    except Exception as exc:  # ImportError, or the package's own driver-version check
        return f"PyNvVideoCodec unavailable ({type(exc).__name__}: {exc})"
    try:
        w, h = (int(v) for v in NVENC_PROBE_SIZE.split("x"))
        nvc.CreateEncoder(w, h, "NV12", False, **_nvenc_direct_config(25, NVENC_CQ, device_index)).EndEncode()
    except Exception as exc:
        return f"NVENC session failed ({' '.join(str(exc).split())[:200]})"
    return None


def choose_route(codec: str, W: int, H: int, device: str | torch.device | None) -> Route:
    """Zero copy when NVENC encodes, frames live on a CUDA device, the size is even (odd sizes keep
    the pipe's ceil-chroma NV12 layout and its encoder behaviour) and PyNvVideoCodec works here.
    Everything else takes the step-1 pipe; #29's strict mode turns that fallback into an error."""
    dev = torch.device(device) if device is not None else None
    if codec != "h264_nvenc":
        return Route(ROUTE_PIPE, f"codec {codec}")
    if dev is None or dev.type != "cuda":
        return Route(ROUTE_PIPE, f"frames on {dev or 'cpu'}")
    if os.environ.get(ZERO_COPY_ENV, "1") == "0":
        return Route(ROUTE_PIPE, f"{ZERO_COPY_ENV}=0")
    if W % 2 or H % 2:
        return Route(ROUTE_PIPE, f"odd size {W}x{H}")
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    error = zero_copy_error(index)
    if error is not None:
        return Route(ROUTE_PIPE, error)
    from importlib.metadata import version

    return Route(ROUTE_ZERO_COPY, f"PyNvVideoCodec {version('pynvvideocodec')} on cuda:{index}")


class _PipeSink:
    """Step-1 route: NV12 bytes copied to the host and piped into ffmpeg, which encodes."""

    def __init__(self, enc: "Encoder", cq: int, crf: int, preset: str) -> None:
        self.log = enc.log
        cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            # Tag the input as what rgb_to_nv12 produced. Untagged, ffmpeg's colorspace negotiation
            # treats it as unspecified and swscale re-converts it to the bt709 output.
            "-f", "rawvideo", "-pix_fmt", "nv12", "-s", f"{enc.W}x{enc.H}",
            "-color_range", "tv", *BT709_ARGS,
            "-r", f"{enc.fps:g}", "-i", "-",
            *vcodec_args(enc.codec, cq=cq, crf=crf, preset=preset),
            "-pix_fmt", "nv12", *BT709_ARGS, *BT709_BSF,
        ]
        if enc.path.suffix.lower() in _MOV_EXTS:
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(enc.path))
        self._fh = open(self.log, "ab")
        self._fh.write(f"\n$ {' '.join(cmd)}\n".encode())
        self._fh.flush()
        self._done = False
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._fh)
        if self._proc.poll() is not None:
            self.finish()

    def write(self, t: torch.Tensor) -> None:
        data = rgb_to_nv12(t).cpu().numpy()
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(memoryview(data))
        except BrokenPipeError:
            self.finish()
            raise EncodeError("encode: ffmpeg closed stdin early", _log_tail(self.log)) from None

    def finish(self) -> None:
        """Drain the pipe, collect the exit code, raise if ffmpeg failed."""
        if self._done:
            return
        self._done = True
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = self._proc.wait()
        self._fh.close()
        if rc != 0:
            raise EncodeError(f"encode failed ({rc})", _log_tail(self.log))

    def abort(self) -> None:
        if self._done:
            return
        self._done = True
        self._proc.kill()
        self._proc.wait()
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except OSError:  # BrokenPipeError flushing what ffmpeg never read
                pass
        self._fh.close()


class _CAI:
    """One plane as a bare __cuda_array_interface__. PyNvVideoCodec prefers __dlpack__ when an
    object has it and calls it with an argument torch's Tensor.__dlpack__ rejects."""

    def __init__(self, t: torch.Tensor) -> None:
        self.__cuda_array_interface__ = t.__cuda_array_interface__


class _NV12Planes:
    """NVENC's NV12 input: a (H, W, 1) luma plane and an (H/2, W/2, 2) CbCr plane in one buffer."""

    def __init__(self, nv12: torch.Tensor, W: int, H: int) -> None:
        self._keep = nv12
        self._planes = [_CAI(nv12[: W * H].view(H, W, 1)), _CAI(nv12[W * H:].view(H // 2, W // 2, 2))]

    def cuda(self) -> list[_CAI]:
        return self._planes


class _NvencDirectSink:
    """Zero-copy route: NVENC reads the CUDA NV12 surface through PyNvVideoCodec. Frames never
    leave the card; only the compressed packets reach the host.

    Packets carry the encoder's display-order timestamps into PyNvVideoCodec's MP4 muxer (raw
    Annex-B through ffmpeg's h264 demuxer loses them across B-frames: pts = dts, first frames
    negative). ffmpeg then remuxes by stream copy, adding the bt709 VUI, container tags and faststart.
    """

    # 512 ticks a frame on a fps×512 clock: 12800 at 25 fps, ffmpeg's own mp4 choice.
    _TICKS = 512

    def __init__(self, enc: "Encoder", cq: int) -> None:
        self.enc = enc
        self.device = torch.device(enc.device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        nvc = _pynvc()
        rate = Fraction(enc.fps).limit_denominator(1001)
        self._es = enc.path.with_name(f".{enc.path.name}.es.mp4")
        self._stream = torch.cuda.Stream(self.device)
        self._n = 0
        self._done = False
        try:
            # The encoder copies its input on _stream, so writes only have to order _stream after
            # the renderer's stream; no host sync.
            self._nvenc = nvc.CreateEncoder(
                enc.W, enc.H, "NV12", False, cudastream=self._stream.cuda_stream,
                **_nvenc_direct_config(enc.fps, cq, self.device.index),
            )
            self._mux = nvc.FFmpegMuxer(
                file_path=str(self._es), media_format=nvc.MEDIA_FORMAT.MP4, codec="h264",
                width=enc.W, height=enc.H, fps_num=rate.numerator, fps_den=rate.denominator,
                timebase_num=1, timebase_den=rate.numerator * self._TICKS,
                extradata=self._nvenc.GetSequenceParams(),
            )
            self._mux.SetUniformPtsIncrement(rate.denominator * self._TICKS)
        except Exception as exc:
            self._done = True
            raise EncodeError(f"encode: NVENC zero-copy setup failed for {enc.W}x{enc.H}",
                              " ".join(str(exc).split())) from None
        self._pic = nvc.NV_ENC_PIC_PARAMS

    def _queue(self, packets: list[dict]) -> None:
        for p in packets:
            self._mux.MuxVideoPacket(bytes(p["data"]), p["picture_type"], p["timestamp"])

    def write(self, t: torch.Tensor) -> None:
        producer = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(producer)
        with torch.cuda.stream(self._stream):
            src = t.to(self.device)
            nv12 = rgb_to_nv12(src)
        if t.device == self.device:
            # The caller frees the frame after write; keep its memory until _stream has read it.
            t.record_stream(self._stream)
        pic = self._pic()
        pic.inputTimeStamp = self._n
        try:
            self._queue(self._nvenc.Encode(_NV12Planes(nv12, self.enc.W, self.enc.H), pic))
        except Exception as exc:
            self.abort()
            raise EncodeError(f"encode: NVENC failed at frame {self._n}", " ".join(str(exc).split())) from None
        self._n += 1

    def finish(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._queue(self._nvenc.EndEncode())
            self._mux.Finalize()
            del self._mux, self._nvenc
            cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(self._es),
                   "-map", "0:v", "-c:v", "copy", *BT709_ARGS, *NVENC_DIRECT_BSF]
            if self.enc.path.suffix.lower() in _MOV_EXTS:
                cmd += ["-movflags", "+faststart"]
            run([*cmd, str(self.enc.path)], log=self.enc.log, desc="encode")
        except EncodeError:
            raise
        except Exception as exc:
            raise EncodeError("encode: NVENC flush failed", " ".join(str(exc).split())) from None
        finally:
            self._es.unlink(missing_ok=True)

    def abort(self) -> None:
        if self._done:
            return
        self._done = True
        # The muxer's destructor finalizes a partial file; drop it, then remove what it wrote.
        self.__dict__.pop("_mux", None)
        self.__dict__.pop("_nvenc", None)
        self._es.unlink(missing_ok=True)


def _log_tail(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return tail_lines(data.decode("utf-8", "replace"))


class Encoder:
    """Context manager: RGB frames in, an H.264 MP4 out, by one of two routes (see `choose_route`).

    `write` takes the renderer's (3, H, W) float tensor or an (H, W, 3) uint8 image; either is
    converted to bt709 limited NV12 where it lives. Pass `device="cuda"` when frames arrive on the
    GPU: with NVENC that is the zero-copy route (NVENC reads the CUDA surface, ffmpeg only remuxes).
    Otherwise the step-1 route copies the 1.5 bytes/px NV12 buffer to the host and pipes it to
    ffmpeg, which hands it to the encoder untouched (swscale converts nothing).
    """

    def __init__(
        self,
        path: Path | str,
        W: int,
        H: int,
        fps: float,
        *,
        encoder: str | None = None,
        log: Path | str | None = None,
        cq: int = NVENC_CQ,
        crf: int = X264_CRF,
        preset: str = X264_PRESET,
        device: str | torch.device | None = None,
        policy: GpuPolicy | None = None,
    ) -> None:
        self.path = Path(path)
        self.W, self.H, self.fps = int(W), int(H), float(fps)
        self.codec = encoder or prefer_encoder()
        if policy is not None:
            if self.codec == "h264_nvenc":
                policy.gpu("encode")
            else:
                policy.cpu("encode", f"{self.codec} encode of {self.path.name}", nvenc_hint())
        self.device = device
        self.log = log_file("encode", log)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.route = choose_route(self.codec, self.W, self.H, device)
        with open(self.log, "a", encoding="utf-8") as fh:
            fh.write(f"encode route {self.route}\n")
        self._closed = False
        self._sink: _PipeSink | _NvencDirectSink = (
            _NvencDirectSink(self, cq) if self.route.zero_copy else _PipeSink(self, cq, crf, preset)
        )

    def write(self, frame: torch.Tensor | np.ndarray) -> None:
        """Append one frame: float (3, H, W) in 0..1 on any device, or uint8 (H, W, 3)."""
        if self._closed:
            raise EncodeError("Encoder.write after close")
        t = torch.as_tensor(frame)
        want = (self.H, self.W, 3) if t.dtype == torch.uint8 else (3, self.H, self.W)
        if tuple(t.shape) != want or not (t.dtype == torch.uint8 or t.is_floating_point()):
            raise EncodeError(
                f"frame must be uint8 ({self.H}, {self.W}, 3) or float (3, {self.H}, {self.W}), "
                f"got {t.dtype} {tuple(t.shape)}"
            )
        self._sink.write(t)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._sink.finish()

    def __enter__(self) -> "Encoder":
        return self

    def abort(self) -> None:
        """Stop without finishing the file: kill ffmpeg, or drop the NVENC session and its temp file."""
        if not self._closed:
            self._closed = True
            self._sink.abort()

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def _concat_quote(path: Path) -> str:
    """concat demuxer quoting: single quotes, escaped as '\\''."""
    return "'" + path.resolve().as_posix().replace("'", "'\\''") + "'"


def concat(
    chunks: Sequence[Path | str],
    out: Path | str,
    *,
    fps: float,
    frames: Sequence[int],
    log: Path | str | None = None,
) -> Path:
    """Join segments with the concat demuxer and `-c copy` (no re-encode).

    Each file gets a `duration` of n/fps from the caller so the next file starts on
    the 1/fps grid, even when a muxer stored a slightly short last-packet duration.
    Frame counts are passed in: probing them would decode every chunk.
    """
    chunks = [Path(c) for c in chunks]
    if not chunks:
        raise EncodeError("concat: no chunks")
    if fps <= 0:
        raise EncodeError(f"concat: fps must be positive, got {fps}")
    if len(frames) != len(chunks):
        raise EncodeError(f"concat: expected {len(chunks)} frame counts, got {len(frames)}")
    missing = [str(c) for c in chunks if not c.exists()]
    if missing:
        raise EncodeError(f"concat: missing chunks {missing}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = out.with_suffix(".concat.txt")
    lines = ["ffconcat version 1.0\n"]
    for chunk, n in zip(chunks, frames, strict=True):
        if n <= 0:
            raise EncodeError(f"concat: frame count must be positive, got {n} for {chunk}")
        lines.append(f"file {_concat_quote(chunk)}\n")
        lines.append(f"duration {format(n / fps, '.17g')}\n")
    listing.write_text("".join(lines))
    run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out)],
        log=log,
        desc="concat",
    )
    return out


def _parse_loudnorm(stderr: str) -> dict:
    """The last loudnorm JSON object in ffmpeg's stderr."""
    decoder = json.JSONDecoder()
    found = None
    for i, ch in enumerate(stderr):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(stderr[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "input_i" in obj:
            found = obj
    if found is None:
        raise EncodeError("mux: loudnorm measurement missing", tail_lines(stderr))
    return found


def _loudnorm_field(stats: dict, key: str) -> str:
    if key not in stats:
        raise EncodeError(f"mux: loudnorm json has no {key}")
    text = str(stats[key]).strip()
    if text.lower() in {"-inf", "inf", "nan"}:
        raise EncodeError(f"mux: loudnorm measured {key}={text}")
    return text


def _measure_loudnorm(audio: Path | str, duration: float, log: Path | str | None) -> dict:
    """Pass 1: integrated loudness of the trimmed track. No fade and no apad.

    The fade is applied after the gain, so it must not change the measurement.
    Padding silence into the analysis would pull a short track away from -14.
    """
    af = (
        f"atrim=start=0:end={duration:.6f},asetpts=PTS-STARTPTS,"
        f"{LOUDNORM}:print_format=json"
    )
    _code, stderr = run(
        [FFMPEG, "-y", "-hide_banner", "-i", str(audio), "-af", af, "-f", "null", "-"],
        log=log,
        desc="mux",
    )
    return _parse_loudnorm(stderr)


def _linear_loudnorm(stats: dict) -> str:
    return (
        f"{LOUDNORM}:measured_I={_loudnorm_field(stats, 'input_i')}:"
        f"measured_TP={_loudnorm_field(stats, 'input_tp')}:"
        f"measured_LRA={_loudnorm_field(stats, 'input_lra')}:"
        f"measured_thresh={_loudnorm_field(stats, 'input_thresh')}:"
        f"offset={_loudnorm_field(stats, 'target_offset')}:linear=true"
    )


def mux(
    video: Path | str,
    audio: Path | str,
    out: Path | str,
    loudnorm: bool = True,
    fade_out: float = 0.0,
    *,
    log: Path | str | None = None,
) -> Path:
    """Attach audio to a finished video: video copied, audio at -14 LUFS as AAC 256k/48k.

    Loudnorm is two-pass and linear. Pass 1 measures the track trimmed to the film.
    Pass 2 applies `measured_I/TP/LRA/thresh` and `offset` with `linear=true`.
    The film's length wins: short audio is padded (apad), long audio is cut.
    The fade, when set, runs after the gain so the body stays put and the ending decays.

    `fade_out` seconds fade out over the *film's* last seconds (`audio.fade_out` in a spec), so
    the music lands on the picture's end rather than the track's.
    """
    for src in (video, audio):
        if not Path(src).exists():
            raise EncodeError(f"mux: {src} does not exist")
    info = probe(video, log=log)
    if info.duration <= 0:
        raise EncodeError(f"mux: unknown video duration for {video}")
    if fade_out < 0:
        raise EncodeError(f"mux: fade_out must not be negative, got {fade_out}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if loudnorm:
        stats = _measure_loudnorm(audio, info.duration, log)
        chain = [
            f"atrim=start=0:end={info.duration:.6f}",
            "asetpts=PTS-STARTPTS",
            _linear_loudnorm(stats),
        ]
    else:
        chain = []
    if fade_out > 0:
        start = max(0.0, info.duration - fade_out)
        duration = min(fade_out, info.duration)
        chain.append(f"afade=t=out:st={start:.6f}:d={duration:.6f}")
    chain.append("apad")
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-i", str(audio),
        "-filter_complex", f"[1:a]{','.join(chain)}[a]", "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", *AUDIO_ARGS, "-b:a", "256k", "-t", f"{info.duration:.6f}",
        "-movflags", "+faststart", str(out),
    ]
    run(cmd, log=log, desc="mux")
    return out


PRESETS = ("master", "whatsapp")


def export(
    src: Path | str,
    target: Path | str,
    preset: str = "master",
    *,
    log: Path | str | None = None,
) -> Path:
    """Delivery copy: `master` copies both streams untouched, `whatsapp` is 720 wide / cq 27 / aac 160k."""
    if preset not in PRESETS:
        raise EncodeError(f"export: unknown preset '{preset}' (have: {', '.join(PRESETS)})")
    if not Path(src).exists():
        raise EncodeError(f"export: {src} does not exist")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    codec = (
        # A master copy stays bit-identical, so it only gets the container tags; the bitstream
        # filter would rewrite the SPS and break "copy" into a re-mux.
        ["-c", "copy", *BT709_ARGS] if preset == "master"
        else [*vcodec_args(prefer_encoder(), cq=27, crf=24), "-vf", "scale=720:-2",
              "-c:a", "aac", "-b:a", "160k", *BT709_ARGS, *BT709_BSF]
    )
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), *codec,
           "-movflags", "+faststart", str(target)]
    run(cmd, log=log, desc="export")
    return target
