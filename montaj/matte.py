"""Subject mattes (#36): a per-asset alpha computed once, offline, on cuda:0; the renderer only reads it.

`montaj ingest --matte <stem>` runs person segmentation (torchvision DeepLabV3-ResNet101, COCO with
VOC labels, class 15; weights pinned by sha256 and fetched once into the models cache) on every
source frame, refines the coarse probability against the full-resolution frame with a guided filter
(He et al.), and steadies it in time: a pixel that barely changed keeps half of its previous value,
one that moved takes the new value outright, so edges stop shimmering without ghosting a waving hand.

Mattes are files keyed by the source bytes: `<assets>/mattes/<sha16>.subject-v1.mp4` for a clip (grey
H.264 on the source's own frame clock, so it decodes through the same `ClipSource` path, crop and
camera as its clip) and `.png` for a photo. An unchanged source finds its matte and is not recomputed;
a changed one has a new sha and so no matte until it is re-ingested. `mattes/index.json` holds a
stat shortcut (size + mtime → sha) so lookups at render time do not re-hash a large clip.

Every pixel step (decode, inference, refine, encode) runs on the device. Under `[gpu] strict` a
non-CUDA device is a `GpuStrictError` before any work starts.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import urllib.request
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

MATTE_URL = "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth"
MATTE_SHA = "586e9e4e203fcbf17e1ad45533d8d33ab133fc762bf03101c5dd743995c08c0d"
MATTE_FILE = "deeplabv3-resnet101-coco.pth"
MATTE_TAG = "subject-v1"  # bump when the model or refine changes: every matte is then recomputed
MATTES_DIR = "mattes"
PERSON = 15  # VOC class index
INFER_SHORT = 520  # the model's training short side
GUIDE_EPS = 1e-4
STEADY = 0.5  # share of the previous matte kept where a pixel barely changed
STEADY_DELTA = 0.25  # a change larger than this is motion, not shimmer
JUMP_WARN = 0.02  # mean frame-to-frame matte change (0..1) above which ingest warns
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_CLIP_EXTS = (".mp4", ".mov", ".webm", ".mkv")

# A segmenter maps (3, h, w) float RGB 0..1 on the device to a (1, h', w') subject probability.
Segmenter = Callable[[torch.Tensor], torch.Tensor]


class MatteError(RuntimeError):
    """Ingest could not make a matte. `str(exc)` is the ERR payload."""


class MatteMissing(FileNotFoundError):
    """No matte for this source's current bytes. `hint` completes the ERR line."""

    def __init__(self, problem: str, hint: str) -> None:
        super().__init__(problem)
        self.hint = hint


# ---- where mattes live ---------------------------------------------------------------------------


def mattes_dir(assets: Path | str) -> Path:
    return Path(assets) / MATTES_DIR


def _index_path(assets: Path) -> Path:
    return mattes_dir(assets) / "index.json"


def _load_index(assets: Path) -> dict[str, dict]:
    path = _index_path(assets)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _key(assets: Path, src: Path) -> str:
    src = Path(src).resolve()
    try:
        return src.relative_to(Path(assets).resolve()).as_posix()
    except ValueError:
        return str(src)


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_sha(assets: Path | str, src: Path | str, index: dict[str, dict] | None = None) -> str:
    """sha256 of `src`, through the index's stat shortcut when size and mtime still match."""
    assets, src = Path(assets), Path(src)
    row = (index if index is not None else _load_index(assets)).get(_key(assets, src))
    st = src.stat()
    if isinstance(row, dict) and row.get("size") == st.st_size and row.get("mtime_ns") == st.st_mtime_ns:
        return str(row["sha"])
    return _file_sha(src)


def is_clip(src: Path | str) -> bool:
    return Path(src).suffix.lower() in _CLIP_EXTS


def matte_path(assets: Path | str, src: Path | str, sha: str | None = None) -> Path:
    """Where `src`'s matte lives (it may not exist yet)."""
    sha = sha or source_sha(assets, src)
    return mattes_dir(assets) / f"{sha[:16]}.{MATTE_TAG}.{'mp4' if is_clip(src) else 'png'}"


def find_matte(assets: Path | str, src: Path | str, stem: str) -> Path:
    """The matte for `src`'s current bytes, or MatteMissing naming the ingest command."""
    path = matte_path(assets, src)
    if not path.is_file():
        raise MatteMissing(f"no subject matte for {stem!r} (its bytes changed or it was never matted)",
                           f"run montaj ingest --matte {stem}")
    return path


# ---- the model -----------------------------------------------------------------------------------


def matte_weights() -> Path:
    """The pinned DeepLabV3 weights, downloaded on first use."""
    from montaj.vision import models_dir

    path = models_dir() / MATTE_FILE
    if path.is_file() and _file_sha(path) == MATTE_SHA:
        return path
    hint = f"download {MATTE_URL} to {path}"
    tmp = path.with_suffix(".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MATTE_URL, timeout=120) as resp, tmp.open("wb") as fh:
            for block in iter(lambda: resp.read(1 << 20), b""):
                fh.write(block)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise MatteError(f"matte model missing, fetch failed ({exc}): {hint}") from exc
    if _file_sha(tmp) != MATTE_SHA:
        tmp.unlink(missing_ok=True)
        raise MatteError(f"matte model download has the wrong sha256: {hint}")
    tmp.replace(path)
    return path


class PersonSegmenter:
    """DeepLabV3-ResNet101 in fp16 on the device; returns the person probability at model resolution."""

    def __init__(self, device) -> None:
        from torchvision.models.segmentation import deeplabv3_resnet101

        self.device = torch.device(device)
        net = deeplabv3_resnet101(weights=None, weights_backbone=None, num_classes=21, aux_loss=True)
        net.load_state_dict(torch.load(matte_weights(), map_location="cpu", weights_only=True))
        net.aux_classifier = None
        half = self.device.type == "cuda"
        self.net = (net.half() if half else net).eval().to(self.device)
        self.dtype = torch.float16 if half else torch.float32
        self.mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        h, w = int(rgb.shape[1]), int(rgb.shape[2])
        s = INFER_SHORT / min(h, w)
        ih, iw = max(8, round(h * s / 8) * 8), max(8, round(w * s / 8) * 8)
        x = F.interpolate(rgb[None], size=(ih, iw), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dtype)
        logits = self.net(x)["out"].float()
        return logits.softmax(dim=1)[0, PERSON:PERSON + 1]


# ---- refine and steady ---------------------------------------------------------------------------


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def refine(prob: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Upsample a coarse (1, h', w') probability to the frame and snap its edges to the frame's with a
    grey guided filter (radius ~ short side / 135, so 8 px at 1080). (1, h, w) in 0..1."""
    h, w = int(rgb.shape[1]), int(rgb.shape[2])
    p = F.interpolate(prob[None], size=(h, w), mode="bilinear", align_corners=False)
    guide = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])[None, None]
    r = max(2, round(min(h, w) / 135))
    mi, mp = _box(guide, r), _box(p, r)
    cov = _box(guide * p, r) - mi * mp
    var = _box(guide * guide, r) - mi * mi
    a = cov / (var + GUIDE_EPS)
    b = mp - a * mi
    q = _box(a, r) * guide + _box(b, r)
    return q[0].clamp(0.0, 1.0)


def steady(cur: torch.Tensor, prev: torch.Tensor | None) -> torch.Tensor:
    """Temporal smoothing that leaves motion alone (see the module docstring)."""
    if prev is None:
        return cur
    still = ((cur - prev).abs() < STEADY_DELTA).to(cur.dtype)
    return cur + still * STEADY * (prev - cur)


# ---- ingest ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MatteResult:
    stem: str
    path: Path
    made: bool  # False: an existing matte for these bytes was reused
    frames: int
    jump: float  # mean frame-to-frame change, 0..1 (0 for a photo or a cached matte)


def _rate(path: Path) -> tuple[Fraction, int]:
    """(frame rate, frame count) of the first video stream."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise MatteError(f"matte: ffprobe failed on {path.name}: {proc.stderr.strip()[-200:]}")
    data = json.loads(proc.stdout or "{}")
    st = (data.get("streams") or [{}])[0]
    rate = Fraction(0)
    for name in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = Fraction(st.get(name) or "0")
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            break
    if rate <= 0:
        raise MatteError(f"matte: {path.name} has no frame rate")
    dur = st.get("duration") or (data.get("format") or {}).get("duration")
    try:
        count = math.floor(float(dur) * rate + 1e-6)
    except (TypeError, ValueError):
        count = int(st.get("nb_frames") or 0)
    if count <= 0:
        raise MatteError(f"matte: {path.name} has no frames")
    return rate, count


def _clip_matte(src: Path, out: Path, segment: Segmenter, device, policy, log) -> tuple[int, float]:
    from montaj.encode import Encoder
    from montaj.render.clip import ClipSource
    from montaj.render.nvdec import probe

    info = probe(src)
    if info is None:
        raise MatteError(f"matte: {src.name} has no video stream")
    w, h = info.width, info.height
    if w % 2 or h % 2:
        raise MatteError(f"matte: {src.name} is {w}x{h}; re-encode it to even dimensions")
    rate, count = _rate(src)
    say = (lambda m: _note(log, m)) if log is not None else None
    reader = ClipSource(src, float(rate), device=device, log=say, policy=policy)
    tmp = out.with_name(f".{out.name}.partial.mp4")
    prev: torch.Tensor | None = None
    jumps: list[float] = []
    try:
        with torch.no_grad(), Encoder(tmp, w, h, float(rate), log=log, device=device, policy=policy) as enc:
            for i in range(count):
                rgb = reader.frame(float(i), 0.0, 0.0, w, h, nframes=count, device=device)
                m = steady(refine(segment(rgb), rgb), prev)
                if prev is not None:
                    jumps.append(float((m - prev).abs().mean()))
                prev = m
                enc.write(m.expand(3, h, w).contiguous())
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        reader.close()
    return count, (sum(jumps) / len(jumps) if jumps else 0.0)


def _photo_matte(src: Path, out: Path, segment: Segmenter, device, policy) -> None:
    from PIL import Image

    from montaj.render.layers import load_rgba

    with torch.no_grad():
        rgba = load_rgba(src, device, policy)
        rgb = rgba[:3] / rgba[3:4].clamp(min=1e-6) if rgba.shape[0] == 4 else rgba
        m = refine(segment(rgb.clamp(0, 1)), rgb.clamp(0, 1)) * rgba[3:4]
        grey = (m[0] * 255 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
    tmp = out.with_name(f".{out.name}.partial.png")
    Image.fromarray(np.ascontiguousarray(grey), "L").save(tmp)  # serialising bytes, not pixel work
    tmp.replace(out)


def _note(log, message: str) -> None:
    if log is None:
        return
    path = Path(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def make_mattes(
    assets: Path | str,
    sources: list[tuple[str, Path]],
    *,
    device,
    policy=None,
    log: Path | str | None = None,
    segmenter: Callable[[object], Segmenter] | None = None,
) -> list[MatteResult]:
    """Matte each `(stem, source path)` whose bytes have no matte yet; reuse the rest.

    `segmenter(device)` builds the model (default `PersonSegmenter`); it is called at most once, and
    only when a source needs work. Runs under the GPU lock. Writes `mattes/index.json`.
    """
    from montaj.gpulock import hold

    assets = Path(assets)
    dev = torch.device(device)
    if policy is not None:
        if dev.type == "cuda":
            policy.gpu("matte")
        else:
            policy.cpu("matte", f"matte inference for {', '.join(s for s, _ in sources)} on {dev}",
                       "run ingest --matte on a CUDA device ([gpu] device in the machine config)")
    index = _load_index(assets)
    mattes_dir(assets).mkdir(parents=True, exist_ok=True)
    results: list[MatteResult] = []
    todo: list[tuple[str, Path, Path]] = []
    for stem, src in sources:
        st = Path(src).stat()
        sha = source_sha(assets, src, index)
        index[_key(assets, src)] = {"sha": sha, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        out = matte_path(assets, src, sha)
        if out.is_file():
            results.append(MatteResult(stem, out, False, 0, 0.0))
        else:
            todo.append((stem, Path(src), out))
    if todo:
        with hold(log):
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            segment = (segmenter or PersonSegmenter)(dev)
            for stem, src, out in todo:
                if is_clip(src):
                    n, jump = _clip_matte(src, out, segment, dev, policy, log)
                else:
                    _photo_matte(src, out, segment, dev, policy)
                    n, jump = 1, 0.0
                results.append(MatteResult(stem, out, True, n, jump))
            if dev.type == "cuda":
                peak = torch.cuda.max_memory_allocated(dev) / 2**30
                _note(log, f"matte: torch peak {peak:.2f} GB on {dev}")
            del segment
    _index_path(assets).write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    order = {stem: i for i, (stem, _) in enumerate(sources)}
    return sorted(results, key=lambda r: order[r.stem])
