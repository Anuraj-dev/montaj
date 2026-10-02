"""Run an agent-owned numpy composition and inspect the resulting audio."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from montaj.audio._worker import append_log
from montaj.encode import FFMPEG, FFPROBE, EncodeError
from montaj.qa import loudness

class ComposeError(RuntimeError):
    """The script or the resulting audio failed the compose contract."""


@dataclass(frozen=True)
class Composition:
    path: Path
    duration: float
    sample_rate: int
    channels: int
    peak_dbfs: float
    lufs: float


def compose(
    script: Path | str,
    out: Path | str,
    *,
    root: Path | str,
    timeout: float = 600.0,
    log: Path | str | None = None,
) -> Composition:
    """Run ``script`` with Montaj's interpreter, then inspect its audio with ffmpeg."""
    if timeout <= 0:
        raise ComposeError(f"timeout must be positive, got {timeout:g} (see build/music.log)")
    root_path = Path(root).expanduser().resolve()
    script_path = Path(script).expanduser().resolve()
    out_path = Path(out).expanduser()
    if not out_path.is_absolute():
        out_path = (root_path / out_path).resolve()
    else:
        out_path = out_path.resolve()
    log_path = Path(log) if log is not None else Path("build/music.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "MONTAJ_OUT": str(out_path), "MONTAJ_SR": "48000"}
    cmd = [sys.executable, str(script_path), str(out_path)]
    backup: Path | None = None
    try:
        backup = _stash_old_output(out_path)
        try:
            proc = subprocess.run(
                cmd,
                cwd=root_path,
                env=env,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr)
            append_log(log_path, cmd, stdout, stderr + f"\ntimed out after {timeout:g}s\n")
            raise ComposeError(
                f"{_last_line(stderr) or f'timed out after {timeout:g}s'} (see build/music.log)"
            ) from None
        except OSError as exc:
            append_log(log_path, cmd, "", str(exc))
            raise ComposeError(f"{exc} (see build/music.log)") from None
        stdout = _as_text(proc.stdout)
        stderr = _as_text(proc.stderr)
        append_log(log_path, cmd, stdout, stderr)
        if proc.returncode != 0:
            reason = _last_line(stderr) or f"exit {proc.returncode}"
            raise ComposeError(f"{reason} (see build/music.log)")

        if not out_path.is_file():
            raise ComposeError(f"output WAV missing: {out_path} (see build/music.log)")
        try:
            duration, sample_rate, channels = _probe_audio(out_path, log_path)
            if duration < 1.0:
                raise ComposeError(f"output WAV shorter than 1s ({duration:.2f}s) (see build/music.log)")
            peak = _peak_dbfs(out_path, log_path)
            if peak < -60.0:
                raise ComposeError(f"output WAV is silent (peak {format_db(peak)} dBFS) (see build/music.log)")
            measured_lufs = loudness(out_path, log=log_path)
        except EncodeError as exc:
            raise ComposeError(f"output WAV unreadable: {_last_line(str(exc))} (see build/music.log)") from None
        if measured_lufs is None:
            raise ComposeError(f"output WAV has no measurable loudness (see build/music.log)")
        result = Composition(out_path, duration, sample_rate, channels, peak, measured_lufs)
    except Exception:
        _restore_old_output(out_path, backup)
        raise
    if backup is not None:
        backup.unlink(missing_ok=True)
    return result


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _stash_old_output(path: Path) -> Path | None:
    """Move an existing output aside so a successful run must create a new file."""
    if not path.is_file():
        return None
    fd, raw_backup = tempfile.mkstemp(prefix=f".{path.name}.montaj-old-", dir=path.parent)
    os.close(fd)
    backup = Path(raw_backup)
    backup.unlink()
    os.replace(path, backup)
    return backup


def _restore_old_output(path: Path, backup: Path | None) -> None:
    if backup is None:
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
    os.replace(backup, path)


def _probe_audio(path: Path, log: Path) -> tuple[float, int, int]:
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    append_log(log, cmd, proc.stdout, proc.stderr)
    if proc.returncode != 0:
        raise EncodeError("ffprobe failed", _last_line(proc.stderr))
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise EncodeError(f"ffprobe returned invalid JSON: {exc}") from None
    streams = data.get("streams") or []
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if not isinstance(audio, dict):
        raise EncodeError("ffprobe found no audio stream")
    duration = _number(audio.get("duration")) or _number((data.get("format") or {}).get("duration"))
    sample_rate = int(_number(audio.get("sample_rate")))
    channels = int(_number(audio.get("channels")))
    if duration <= 0 or sample_rate <= 0 or channels <= 0:
        raise EncodeError("ffprobe returned incomplete audio metadata")
    return duration, sample_rate, channels


def _number(value: object) -> float:
    try:
        return float(value) if value not in (None, "", "N/A") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _peak_dbfs(path: Path, log: Path) -> float:
    """Decode float samples and measure their peak before result-line rounding."""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-sn", "-dn",
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stderr = _as_text(proc.stderr)
    append_log(log, cmd, "", stderr)
    if proc.returncode != 0:
        raise EncodeError(f"ffmpeg failed ({proc.returncode})", _last_line(stderr))
    samples = np.frombuffer(proc.stdout, dtype="<f4")
    if not samples.size:
        raise EncodeError("ffmpeg returned no audio samples", _last_line(stderr))
    peak = float(np.max(np.abs(samples)))
    if not np.isfinite(peak):
        raise EncodeError("ffmpeg returned a non-finite sample peak", _last_line(stderr))
    return float("-inf") if peak == 0.0 else float(20.0 * np.log10(peak))


def format_db(value: float) -> str:
    """Format a dB value for an agent-facing result line."""
    if value == float("-inf"):
        return "-inf"
    rounded = round(value, 1)
    return "0.0" if abs(rounded) < 0.05 else f"{rounded:.1f}"
