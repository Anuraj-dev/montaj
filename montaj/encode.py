"""ffmpeg/ffprobe seam: raw RGB frames in, finished MP4s out.

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
from functools import cache
from pathlib import Path

import numpy as np

FFMPEG = os.environ.get("MONTAJ_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("MONTAJ_FFPROBE", "ffprobe")

# film2 / render.sh encode settings; the fallback keeps visual parity on machines without NVENC.
NVENC_CQ = 16
NVENC_ARGS = ("-preset", "p7", "-tune", "hq", "-rc", "vbr", "-b:v", "0", "-profile:v", "high")
X264_CRF = 16
X264_PRESET = "slow"
AUDIO_ARGS = ("-c:a", "aac", "-ar", "48000")
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"
NVENC_PROBE_SIZE = "320x240"  # NVENC rejects frames below ~145 px a side

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


class Encoder:
    """Context manager streaming uint8 HWC frames into ffmpeg stdin as rawvideo rgb24."""

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
    ) -> None:
        self.path = Path(path)
        self.W, self.H, self.fps = int(W), int(H), float(fps)
        self.codec = encoder or prefer_encoder()
        self.log = log_file("encode", log)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}",
            "-r", f"{self.fps:g}", "-i", "-",
            *vcodec_args(self.codec, cq=cq, crf=crf, preset=preset),
            "-pix_fmt", "yuv420p",
        ]
        if self.path.suffix.lower() in _MOV_EXTS:
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(self.path))
        self._cmd = cmd
        self._fh = open(self.log, "ab")
        self._fh.write(f"\n$ {' '.join(cmd)}\n".encode())
        self._fh.flush()
        self._closed = False
        self._rc = 0
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._fh)
        if self._proc.poll() is not None:
            self._finish()

    def write(self, frame: np.ndarray) -> None:
        """Append one frame; shape must be (H, W, 3) uint8."""
        if self._closed:
            raise EncodeError("Encoder.write after close")
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.shape != (self.H, self.W, 3):
            raise EncodeError(f"frame must be uint8 ({self.H}, {self.W}, 3), got {arr.dtype} {arr.shape}")
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(np.ascontiguousarray(arr).tobytes())
        except BrokenPipeError:
            self._finish()
            raise EncodeError("encode: ffmpeg closed stdin early", self._tail()) from None

    def _tail(self) -> str:
        try:
            data = self.log.read_bytes()
        except OSError:
            return ""
        return tail_lines(data.decode("utf-8", "replace"))

    def _finish(self) -> int:
        """Drain the pipe, collect the exit code, raise if ffmpeg failed."""
        if self._closed:
            return self._rc
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        self._rc = self._proc.wait()
        self._fh.close()
        self._closed = True
        if self._rc != 0:
            raise EncodeError(f"encode failed ({self._rc})", self._tail())
        return self._rc

    def close(self) -> None:
        self._finish()

    def __enter__(self) -> "Encoder":
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is None:
            self._finish()
        elif not self._closed:
            self._proc.kill()
            self._proc.wait()
            self._fh.close()
            self._closed = True


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


def mux(
    video: Path | str,
    audio: Path | str,
    out: Path | str,
    loudnorm: bool = True,
    *,
    log: Path | str | None = None,
) -> Path:
    """Attach audio to a finished video: video copied, audio loudnorm'd to -14 LUFS as AAC 256k/48k.

    The film's length wins: short audio is padded (apad), long audio is cut, so adding music
    never truncates the picture. Padding goes *after* loudnorm — normalising an endless apad
    stream makes loudnorm's single-pass analysis see a different signal.
    """
    for src in (video, audio):
        if not Path(src).exists():
            raise EncodeError(f"mux: {src} does not exist")
    info = probe(video, log=log)
    if info.duration <= 0:
        raise EncodeError(f"mux: unknown video duration for {video}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chain = f"{LOUDNORM},apad" if loudnorm else "apad"
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-i", str(audio),
        "-filter_complex", f"[1:a]{chain}[a]", "-map", "0:v", "-map", "[a]",
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
        ["-c", "copy"] if preset == "master"
        else [*vcodec_args(prefer_encoder(), cq=27, crf=24), "-vf", "scale=720:-2",
              "-c:a", "aac", "-b:a", "160k"]
    )
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), *codec,
           "-movflags", "+faststart", str(target)]
    run(cmd, log=log, desc="export")
    return target
