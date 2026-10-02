"""Environment check. `doctor()` returns `OK|WARN|FAIL <check>: <detail>` lines, never prints.

The TRAP lines are the hard-won facts from `docs/research/prior-art.md`; they travel with
the command so an agent hits them before burning time rediscovering them.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from montaj.config import Config, ConfigError
from montaj.config import load as load_config
from montaj.encode import FFPROBE, FFMPEG, has_encoder, nvenc_usable, tool_version

CHROMIUM_CANDIDATES = ("google-chrome-stable", "google-chrome", "chromium", "chromium-browser")
WHISPER_IMPORT_TIMEOUT = 20  # a plain import; the model load is the slow part and lives in the worker
# docs/research/prior-art.md "Known traps" — keep in sync with that file.
TRAPS: tuple[str, ...] = (
    "TRAP chromium-cpu: headless Chromium renders on CPU (SwiftShader) whatever GPU flags say - don't retry",
    "TRAP ace-step-python: ACE-Step venv bin/python symlink is dangling; run uv CPython 3.12 with its site-packages",
    "TRAP ace-step-vram: 8 GB VRAM - ACE-Step needs offload_to_cpu + offload_dit_to_cpu, and no render at the same time",
    "TRAP ace-step-path: ACE-Step lives at ~/.local/share/montaj/ACE-Step-1.5 (18 GB) - reference via config, don't move",
    "TRAP gpu-lock: music gen/analyze and render share the one GPU - montaj locks ~/.cache/montaj/gpu.lock, don't run them side by side",
    "TRAP still-photos: no warp/drift/depth-parallax on stills - motion lives in transitions, cuts on a beat grid",
)


def cuda_line() -> str:
    import torch  # imported here: doctor should work in a CPU-only venv too

    if not torch.cuda.is_available():
        return "FAIL cuda: torch.cuda.is_available() is False"
    cap = ".".join(str(n) for n in torch.cuda.get_device_capability(0))
    cu = ".".join(str(n) for n in (torch.version.cuda or "").split(".")[:2])
    return f"OK cuda: {torch.cuda.get_device_name(0)} sm_{cap}, torch {torch.__version__}{f' cu{cu}' if cu else ''}"


def _version(tool: str, name: str) -> str:
    banner = tool_version(tool)
    if not banner:
        return f"FAIL {name}: not found"
    parts = banner.split(" ")
    return f"OK {name}: {parts[2] if len(parts) > 2 else banner}"


def free_vram() -> tuple[int, int] | None:
    """(free, total) MiB from nvidia-smi, or None when it cannot answer."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        free, total = out.stdout.decode().strip().splitlines()[0].split(",")
        return int(free.strip()), int(total.strip())
    except (IndexError, ValueError):
        return None


def nvenc_line() -> str:
    """Listed is not the same as usable: a build can list h264_nvenc with no working driver."""
    if not has_encoder("h264_nvenc"):
        return "WARN nvenc: h264_nvenc missing, libx264 -crf 16 -preset slow fallback"
    if nvenc_usable():
        return "OK nvenc: h264_nvenc available"
    return "WARN nvenc: h264_nvenc listed but cannot encode (no usable driver), libx264 fallback"


def doctor() -> list[str]:
    """Environment report for the CLI (or an agent) to print as-is."""
    lines = [cuda_line(), nvenc_line()]

    lines.append(_version(FFMPEG, "ffmpeg"))
    lines.append(_version(FFPROBE, "ffprobe"))

    chrome = next((c for c in CHROMIUM_CANDIDATES if shutil.which(c)), None)
    lines.append(f"OK chromium: {chrome}" if chrome else "FAIL chromium: not found")

    vram = free_vram()
    lines.append(f"OK vram: {vram[0]} MiB free of {vram[1]} MiB" if vram else "WARN vram: nvidia-smi unavailable")

    lines.extend(config_lines())

    lines.extend(TRAPS)
    return lines


def _exists_line(name: str, path: str) -> str:
    return f"OK {name}: {path}" if Path(path).exists() else f"FAIL {name}: {path} does not exist"


def faster_whisper_line(python: str, *, timeout: float = WHISPER_IMPORT_TIMEOUT) -> str:
    """Can that interpreter import faster_whisper? A subprocess: montaj must not import it."""
    try:
        proc = subprocess.run(
            [python, "-c", "import faster_whisper"], capture_output=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return f"FAIL faster_whisper: import from {python} did not finish in {timeout:g}s"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"FAIL faster_whisper: cannot run {python}: {exc}"
    if proc.returncode != 0:
        tail = " ".join(proc.stderr.decode("utf-8", "replace").split())[-160:]
        return f"FAIL faster_whisper: {tail or f'exit {proc.returncode}'}"
    return f"OK faster_whisper: importable by {Path(python).name}"


def config_lines(config: Config | None = None) -> list[str]:
    """One OK/FAIL line per config key, then the whisper import check.

    Separate processes with separate torch builds cannot be checked in-process, so the last line
    is an actual interpreter launch — with a timeout, because a wrong path can hang on NFS.
    """
    try:
        cfg = config if config is not None else load_config()
    except ConfigError as exc:
        return [f"FAIL config: {exc}"]
    lines = [f"OK config: {cfg.path}"]
    for key in ("ace_step_dir", "ace_python", "ace_site_packages", "whisper_python"):
        lines.append(_exists_line(key, getattr(cfg, key)))
    lines.append(faster_whisper_line(cfg.whisper_python))
    return lines


def chromium() -> Path | None:
    """Path to the Chromium binary used for text rasterisation, or None."""
    found = next((c for c in CHROMIUM_CANDIDATES if shutil.which(c)), None)
    return Path(found) if found else None