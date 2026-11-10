"""Environment check. `doctor()` returns `OK|WARN|FAIL <check>: <detail>` lines, never prints.

The TRAP lines are the hard-won facts from `docs/research/prior-art.md`; they travel with
the command so an agent hits them before burning time rediscovering them.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from montaj.config import Config, ConfigError
from montaj.config import load as load_config
from montaj.encode import (
    FFMPEG, FFPROBE, TEN_BIT_ENCODERS, choose_route, has_encoder, nvenc_hint, nvenc_usable, prefer_encoder,
    ten_bit_error, tool_version,
)
from montaj.gpu_policy import GpuPolicy, load_policy, resolve_device

CHROMIUM_CANDIDATES = ("google-chrome-stable", "google-chrome", "chromium", "chromium-browser")
WHISPER_IMPORT_TIMEOUT = 20  # a plain import; the model load is the slow part and lives in the worker
# docs/research/prior-art.md "Known traps" — keep in sync with that file.
TRAPS: tuple[str, ...] = (
    "TRAP chromium-cpu: headless Chromium renders on CPU (SwiftShader) whatever GPU flags say - don't retry",
    "TRAP ace-step-python: ACE-Step venv bin/python symlink is dangling; run uv CPython 3.12 with its site-packages",
    "TRAP ace-step-vram: 8 GB VRAM - ACE-Step needs offload_to_cpu + offload_dit_to_cpu, and no render at the same time",
    "TRAP ace-step-path: ACE-Step lives at ~/.local/share/montaj/ACE-Step-1.5 (18 GB) - reference via config, don't move",
    "TRAP gpu-lock: music gen/analyze and render share the one GPU - montaj locks ~/.cache/montaj/gpu.lock, don't run them side by side",
    "TRAP gpu-strict: [gpu] strict=true in config.toml turns every CPU pixel fallback (ffmpeg decode, Chromium text, PIL photo, libx264) into ERR gpu-strict; page raster in capture stays CPU; print/wall shadows are a bit-exact torch port of PIL GaussianBlur, so walls render strict",
    "TRAP p010-typestr: PyNvVideoCodec takes P010 planes only with __cuda_array_interface__ typestr '|u2'; torch's '<u2' fails 'Invalid typestr'",
    "TRAP hevc-tag: a stream copy of HEVC into MP4 writes 'hev1' unless -tag:v hvc1 - QuickTime/Safari refuse hev1; concat/mux/export pass it",
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


def ten_bit_line() -> str:
    """`render --codec hevc10|av1` and the 10-bit export targets: can this NVENC encode them?"""
    ok = [c for c in TEN_BIT_ENCODERS if ten_bit_error(c) is None]
    bad = [c for c in TEN_BIT_ENCODERS if c not in ok]
    if not bad:
        return f"OK nvenc 10-bit: {', '.join(f'{c} ({TEN_BIT_ENCODERS[c]})' for c in ok)}"
    return f"WARN nvenc 10-bit: {', '.join(bad)} unavailable ({ten_bit_error(bad[0])}); those codecs and targets ERR"


def encode_route_line() -> str:
    """The route a 1080x1920 GPU render encodes through. WARN means frames cross to the host."""
    import torch

    if not torch.cuda.is_available():
        return "WARN encode route: ffmpeg-pipe: no CUDA device"
    route = choose_route(prefer_encoder(), 1080, 1920, "cuda")
    return f"{'OK' if route.zero_copy else 'WARN'} encode route: {route}"


def doctor() -> list[str]:
    """Environment report for the CLI (or an agent) to print as-is."""
    lines = [cuda_line(), nvenc_line(), ten_bit_line(), encode_route_line()]
    lines.extend(stage_lines())

    lines.append(_version(FFMPEG, "ffmpeg"))
    lines.append(_version(FFPROBE, "ffprobe"))

    chrome = next((c for c in CHROMIUM_CANDIDATES if shutil.which(c)), None)
    lines.append(f"OK chromium: {chrome}" if chrome else "FAIL chromium: not found")

    lines.append(text_engine_line())

    vram = free_vram()
    lines.append(f"OK vram: {vram[0]} MiB free of {vram[1]} MiB" if vram else "WARN vram: nvidia-smi unavailable")

    lines.extend(config_lines())
    lines.extend(skill_lines())

    lines.extend(TRAPS)
    return lines


# The agent skill is a symlink into the checkout, so a moved checkout leaves it dangling. Doctor
# warns; `montaj skill link` (link_skill) repairs it. Dirs that do not exist are not ours to create.
SKILL = "make-video"
AGENT_SKILL_DIRS = (".claude/skills", ".codex/skills")


def skill_source() -> Path:
    """This checkout's skill. The repo root comes from the package location, never the cwd."""
    return Path(__file__).resolve().parents[1] / "skills" / SKILL


def skill_dirs() -> list[Path]:
    """Existing agent skill dirs under $HOME (Path.home reads HOME, so tests can use a temp home)."""
    home = Path.home()
    return [d for d in (home / rel for rel in AGENT_SKILL_DIRS) if d.is_dir()]


def _skill_state(link: Path) -> tuple[str, Path | None]:
    """missing | real | dangling | other (with the target it reaches) | ok."""
    if not os.path.lexists(link):
        return "missing", None
    if not link.is_symlink():
        return "real", None
    if not link.exists():
        return "dangling", None
    target = link.resolve()
    return ("ok", None) if target == skill_source().resolve() else ("other", target)


def skill_lines() -> list[str]:
    """One line per existing agent skill dir: OK when make-video resolves to this checkout."""
    source = skill_source()
    if not (source / "SKILL.md").is_file():
        return [f"WARN skill {SKILL}: {source} missing (this checkout has no skill to link)"]
    lines: list[str] = []
    for d in skill_dirs():
        link = d / SKILL
        state, target = _skill_state(link)
        if state == "ok":
            lines.append(f"OK skill {SKILL}: {link}")
        elif state == "missing":
            lines.append(f"WARN skill {SKILL}: no link in {d} (montaj skill link)")
        elif state == "dangling":
            lines.append(f"WARN skill {SKILL}: {link} is dangling (montaj skill link)")
        elif state == "other":
            lines.append(f"WARN skill {SKILL}: {link} points at another checkout {target} (montaj skill link)")
        else:
            lines.append(f"WARN skill {SKILL}: {link} is not a symlink (remove it, then montaj skill link)")
    return lines


def link_skill() -> list[str]:
    """Make `make-video` in each existing agent skill dir a symlink straight at this checkout.

    Idempotent. A real directory or file at the link path is reported and left alone; the result
    line is last, so its OK/ERR status is the command's status.
    """
    source = skill_source()
    if not (source / "SKILL.md").is_file():
        return [f"ERR skill link: {source} has no SKILL.md to link"]
    dirs = skill_dirs()
    if not dirs:
        return ["ERR skill link: no ~/.claude/skills or ~/.codex/skills to link into (create one first)"]
    lines: list[str] = []
    linked: list[Path] = []
    for d in dirs:
        link = d / SKILL
        # Leave a correct link alone, even one stored through a symlinked parent; a chain through
        # another skill link (codex -> claude -> checkout) is rewritten to point straight here.
        stored = Path(os.readlink(link)) if link.is_symlink() else None
        if stored is not None and link.exists() and not (link.parent / stored).is_symlink() \
                and link.resolve() == source.resolve():
            linked.append(d)
            continue
        if os.path.lexists(link) and not link.is_symlink():
            lines.append(f"ERR skill {SKILL}: {link} is not a symlink: remove it first")
            continue
        # Build the new link beside the old one and rename it over, so a failure never loses the old link.
        tmp = d / f".{SKILL}.tmp"
        try:
            if os.path.lexists(tmp):
                tmp.unlink()
            tmp.symlink_to(source, target_is_directory=True)
            os.replace(tmp, link)
        except OSError as e:
            lines.append(f"ERR skill {SKILL}: {link}: {e.strerror or e}")
            continue
        linked.append(d)
    failed = len(dirs) - len(linked)
    if failed:
        lines.append(f"ERR skill link: {failed} of {len(dirs)} dirs not linked (see above)")
    else:
        names = ", ".join(str(d) for d in linked)
        lines.append(f"OK skill {SKILL} linked in {len(linked)} dir{'s' if len(linked) != 1 else ''}: {names}")
    return lines


def stage_lines(policy: GpuPolicy | None = None) -> list[str]:
    """One line per pixel stage: does it run on the GPU here? Under `[gpu] strict` a stage that
    would fall back to the CPU is a FAIL (the render would stop with `ERR gpu-strict`); the named
    exceptions (Chromium text styles, page raster, PNG/WebP photos) stay WARN."""
    import torch

    try:
        pol = policy if policy is not None else load_policy()
    except RuntimeError as exc:
        return [f"FAIL gpu-policy: {exc}"]
    bad = "FAIL" if pol.strict else "WARN"
    if not torch.cuda.is_available():
        return [f"{bad} composite: no CUDA device", f"OK gpu-policy: strict={str(pol.strict).lower()}"]
    try:
        device = resolve_device(pol.device)
    except ValueError as exc:
        return [f"FAIL gpu-policy: {exc}"]
    lines = [f"OK gpu-policy: strict={str(pol.strict).lower()} device={device}"]
    name = torch.cuda.get_device_name(torch.device(device))
    lines.append(f"OK composite: {device} {name}")

    from montaj.render import nvdec

    lines.append(
        "OK decode: nvdec (a clip NVDEC cannot open falls back to ffmpeg on the CPU)" if nvdec._IMPORT_ERROR is None
        else f"{bad} decode: ffmpeg on the CPU, PyNvVideoCodec unavailable ({nvdec._IMPORT_ERROR})"
    )
    lines.append(photo_line(device, bad))
    lines.append(text_stage_line())
    codec = prefer_encoder()
    route = choose_route(codec, 1080, 1920, device)
    if codec != "h264_nvenc":
        lines.append(f"{bad} encode: {codec} on the CPU ({nvenc_hint()})")
    elif route.zero_copy:
        lines.append("OK encode: nvenc zero-copy")
    else:
        lines.append(f"WARN encode: nvenc via ffmpeg pipe ({route.reason})")
    lines.append(
        f"WARN capture: page raster is CPU (Chromium); encode {'nvenc' if codec == 'h264_nvenc' else codec}"
    )
    return lines


def photo_line(device: str, bad: str = "WARN") -> str:
    """JPEG decodes on nvJPEG; a tiny real decode proves it. PNG/WebP are CPU (PIL) at load."""
    import io

    import torch
    from PIL import Image

    try:
        from torchvision.io import ImageReadMode, decode_jpeg

        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (200, 30, 60)).save(buf, "JPEG")
        decode_jpeg(torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8), device=device, mode=ImageReadMode.RGB)
    except Exception as exc:
        return f"{bad} photo: nvJPEG unavailable, PIL on the CPU ({type(exc).__name__}: {' '.join(str(exc).split())[:100]})"
    return "OK photo: nvjpeg (JPEG); PNG/WebP decode on the CPU (PIL), convert to JPEG for strict"


def text_stage_line() -> str:
    from montaj.render.glyphs import STYLE_ENGINE

    cpu = [s for s, e in STYLE_ENGINE.items() if e != "gpu"]
    try:
        import fontTools  # noqa: F401
        import uharfbuzz  # noqa: F401
    except ImportError as exc:
        return f"WARN text: gpu engine unavailable ({exc.name} missing: uv sync)"
    if cpu:
        return f"WARN text: gpu engine ready; styles {', '.join(cpu)} default to Chromium (CPU), set engine: gpu per line"
    return "OK text: gpu"


def text_engine_line() -> str:
    """Which engine draws each text style by default, and whether the gpu engine can shape."""
    from montaj.render.glyphs import STYLE_ENGINE

    styles = ", ".join(f"{s} {e}" for s, e in STYLE_ENGINE.items())
    try:
        import fontTools
        import uharfbuzz
    except ImportError as exc:
        return f"FAIL text-engine: {styles}; gpu unavailable ({exc.name} missing: uv sync)"
    return f"OK text-engine: {styles}; gpu opt-in per line (engine: gpu), harfbuzz {uharfbuzz.version_string()}, fonttools {fontTools.version}"


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
        cfg = config if config is not None else load_config(required=())
    except ConfigError as exc:
        return [f"FAIL config: {exc}"]
    lines = [f"OK config: {cfg.path}"]
    for key in ("ace_step_dir", "ace_python", "ace_site_packages", "whisper_python"):
        value = getattr(cfg, key)
        lines.append(f"FAIL {key}: missing" if not value else _exists_line(key, value))
    lines.append(
        faster_whisper_line(cfg.whisper_python) if cfg.whisper_python
        else "FAIL faster_whisper: whisper_python missing"
    )
    return lines


def chromium() -> Path | None:
    """Path to the Chromium binary used for text rasterisation, or None."""
    found = next((c for c in CHROMIUM_CANDIDATES if shutil.which(c)), None)
    return Path(found) if found else None
