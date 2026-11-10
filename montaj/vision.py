"""GPU models and measurements behind `montaj ingest` analysis.

Faces: RetinaFace with a MobileNet-0.25 backbone (Pytorch_Retinaface, MIT), the 1.7 MB weights
fetched once into `~/.cache/montaj/models/` (`MONTAJ_MODELS` overrides) and pinned by sha256.
Everything else is plain tensor maths, no model: spectral-residual saliency for `subject`,
Laplacian variance for `sharpness`, mean luma for `brightness`, k-means for dominant colours,
histogram and frame differences for clip cuts and motion, and audio energy for speech spans.

`GpuAnalyzer` holds the loaded net as instance state, built once per ingest; nothing here is
module-level mutable. Coordinates are source pixels of the file analysed.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import subprocess
import urllib.request
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

FACE_URL = "https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth"
FACE_SHA = "2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220"
FACE_FILE = "retinaface-mobilenet0.25.pth"
MODELS_ENV = "MONTAJ_MODELS"
FACE_SCORE = 0.8  # RetinaFace's vis threshold is 0.6; 0.8 drops profile/background false hits
FACE_NMS = 0.4
FACE_MAX_SIDE = 2048
CLIP_W, CLIP_H = 128, 72  # stats frames: cuts and motion need shape, not detail
CUT_DIST = 0.45  # half-L1 colour-histogram distance (0..1) between neighbouring frames
CUT_MIN_GAP_S = 0.5
AUDIO_RATE = 16000
AUDIO_WIN_S = 0.02
SPEECH_FLOOR_DB = -45.0  # quieter than this is never speech
SPEECH_OVER_NOISE_DB = 12.0
SPEECH_MIN_S = 0.2
SPEECH_BRIDGE_S = 0.3
SILENCE_MIN_S = 0.3
MAX_SCENES = 200


class AnalysisError(RuntimeError):
    pass


def models_dir() -> Path:
    given = os.environ.get(MODELS_ENV, "").strip()
    return Path(given).expanduser() if given else Path.home() / ".cache" / "montaj" / "models"


def face_weights() -> Path:
    """The pinned RetinaFace weights, downloaded on first use."""
    path = models_dir() / FACE_FILE
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == FACE_SHA:
        return path
    hint = f"download {FACE_URL} to {path}, or ingest --fast"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FACE_URL, timeout=60) as resp:
            data = resp.read()
    except OSError as exc:
        raise AnalysisError(f"face model missing, fetch failed ({exc}): {hint}") from exc
    if hashlib.sha256(data).hexdigest() != FACE_SHA:
        raise AnalysisError(f"face model download has the wrong sha256: {hint}")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


# --- RetinaFace (MobileNet-0.25), layer names match the published state dict --------------------


def _conv_bn(inp: int, oup: int, stride: int = 1, leaky: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup),
                         nn.LeakyReLU(leaky, inplace=True))


def _conv_bn_no_relu(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, 3, 1, 1, bias=False), nn.BatchNorm2d(oup))


def _conv_bn1x1(inp: int, oup: int, leaky: float) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
                         nn.LeakyReLU(leaky, inplace=True))


def _conv_dw(inp: int, oup: int, stride: int, leaky: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp),
        nn.LeakyReLU(leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(leaky, inplace=True),
    )


class _MobileNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(_conv_bn(3, 8, 2, 0.1), _conv_dw(8, 16, 1), _conv_dw(16, 32, 2),
                                    _conv_dw(32, 32, 1), _conv_dw(32, 64, 2), _conv_dw(64, 64, 1))
        self.stage2 = nn.Sequential(_conv_dw(64, 128, 2), *[_conv_dw(128, 128, 1) for _ in range(5)])
        self.stage3 = nn.Sequential(_conv_dw(128, 256, 2), _conv_dw(256, 256, 1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        a = self.stage1(x)
        b = self.stage2(a)
        return [a, b, self.stage3(b)]


class _FPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output1 = _conv_bn1x1(64, 64, 0.1)
        self.output2 = _conv_bn1x1(128, 64, 0.1)
        self.output3 = _conv_bn1x1(256, 64, 0.1)
        self.merge1 = _conv_bn(64, 64, leaky=0.1)
        self.merge2 = _conv_bn(64, 64, leaky=0.1)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        o1, o2, o3 = self.output1(xs[0]), self.output2(xs[1]), self.output3(xs[2])
        o2 = self.merge2(o2 + F.interpolate(o3, size=o2.shape[2:], mode="nearest"))
        o1 = self.merge1(o1 + F.interpolate(o2, size=o1.shape[2:], mode="nearest"))
        return [o1, o2, o3]


class _SSH(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv3X3 = _conv_bn_no_relu(64, 32)
        self.conv5X5_1 = _conv_bn(64, 16, leaky=0.1)
        self.conv5X5_2 = _conv_bn_no_relu(16, 16)
        self.conv7X7_2 = _conv_bn(16, 16, leaky=0.1)
        self.conv7x7_3 = _conv_bn_no_relu(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c5_1 = self.conv5X5_1(x)
        c7 = self.conv7x7_3(self.conv7X7_2(c5_1))
        return F.relu(torch.cat([self.conv3X3(x), self.conv5X5_2(c5_1), c7], dim=1))


class _Head(nn.Module):
    def __init__(self, out: int) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(64, out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1x1(x).permute(0, 2, 3, 1)


class RetinaFace(nn.Module):
    MIN_SIZES = ((16, 32), (64, 128), (256, 512))
    STEPS = (8, 16, 32)
    MEAN_BGR = (104.0, 117.0, 123.0)

    def __init__(self) -> None:
        super().__init__()
        self.body = _MobileNet()
        self.fpn = _FPN()
        self.ssh1, self.ssh2, self.ssh3 = _SSH(), _SSH(), _SSH()
        self.ClassHead = nn.ModuleList([_Head(4) for _ in range(3)])
        self.BboxHead = nn.ModuleList([_Head(8) for _ in range(3)])
        self.LandmarkHead = nn.ModuleList([_Head(20) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fpn = self.fpn(self.body(x))
        feats = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
        n = x.shape[0]
        loc = torch.cat([h(f).reshape(n, -1, 4) for h, f in zip(self.BboxHead, feats)], 1)
        conf = torch.cat([h(f).reshape(n, -1, 2) for h, f in zip(self.ClassHead, feats)], 1)
        ldm = torch.cat([h(f).reshape(n, -1, 10) for h, f in zip(self.LandmarkHead, feats)], 1)
        return loc, conf.softmax(-1), ldm

    @classmethod
    def priors(cls, h: int, w: int, device: torch.device) -> torch.Tensor:
        rows = []
        for step, sizes in zip(cls.STEPS, cls.MIN_SIZES):
            fh, fw = math.ceil(h / step), math.ceil(w / step)
            for i, j in product(range(fh), range(fw)):
                for s in sizes:
                    rows.append(((j + 0.5) * step / w, (i + 0.5) * step / h, s / w, s / h))
        return torch.tensor(rows, dtype=torch.float32, device=device)


def load_face_net(device: torch.device) -> RetinaFace:
    net = RetinaFace()
    net.load_state_dict(torch.load(face_weights(), map_location="cpu", weights_only=True))
    return net.eval().to(device)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, thresh: float) -> list[int]:
    """Greedy NMS; a handful of faces survive the score cut, so a Python loop is fine."""
    order = scores.argsort(descending=True).tolist()
    keep: list[int] = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        rest = torch.tensor(order, device=boxes.device)
        xx0 = torch.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = torch.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = torch.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = torch.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (xx1 - xx0).clamp(min=0) * (yy1 - yy0).clamp(min=0)
        iou = inter / (area[i] + area[rest] - inter)
        order = [o for o, v in zip(order, (iou <= thresh).tolist()) if v]
    return keep


@torch.inference_mode()
def detect_faces(net: RetinaFace, rgb: torch.Tensor, *, score: float = FACE_SCORE) -> list[dict]:
    """Faces in an (H, W, 3) uint8 RGB tensor, largest first.

    Each is `{box: [x0, y0, x1, y1], eyes: [[x, y], [x, y]], score}` in source px; `eyes` is image
    left then image right.
    """
    h0, w0 = rgb.shape[:2]
    img = rgb.permute(2, 0, 1)[None].float()
    k = min(1.0, FACE_MAX_SIDE / max(h0, w0))
    if k < 1.0:
        img = F.interpolate(img, size=(round(h0 * k), round(w0 * k)), mode="area")
    h, w = img.shape[2:]
    bgr = img[:, [2, 1, 0]] - torch.tensor(RetinaFace.MEAN_BGR, device=img.device).view(1, 3, 1, 1)
    loc, conf, ldm = net(bgr)
    pri = RetinaFace.priors(h, w, img.device)
    s = conf[0, :, 1]
    hit = s > score
    if not bool(hit.any()):
        return []
    loc, ldm, pri, s = loc[0][hit], ldm[0][hit], pri[hit], s[hit]
    cxcy = pri[:, :2] + loc[:, :2] * 0.1 * pri[:, 2:]
    wh = pri[:, 2:] * torch.exp(loc[:, 2:] * 0.2)
    boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)
    pts = pri[:, None, :2] + ldm.view(-1, 5, 2) * 0.1 * pri[:, None, 2:]
    scale = torch.tensor([w0, h0], device=img.device, dtype=torch.float32)
    boxes = boxes * scale.repeat(2)
    pts = pts * scale
    keep = _nms(boxes, s, FACE_NMS)
    faces = []
    for i in keep:
        x0, y0, x1, y1 = (float(v) for v in boxes[i].clamp(min=0).tolist())
        x1, y1 = min(x1, w0), min(y1, h0)
        eyes = [[round(float(pts[i, j, 0]), 1), round(float(pts[i, j, 1]), 1)] for j in (0, 1)]
        faces.append({"box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)], "eyes": eyes,
                      "score": round(float(s[i]), 3)})
    faces.sort(key=lambda f: -(f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
    return faces


# --- photo measurements ------------------------------------------------------------------------


def _luma(img: torch.Tensor) -> torch.Tensor:
    """(1, 3, H, W) float -> (1, 1, H, W), Rec.709 weights."""
    wts = torch.tensor([0.2126, 0.7152, 0.0722], device=img.device).view(1, 3, 1, 1)
    return (img * wts).sum(1, keepdim=True)


def _fit(img: torch.Tensor, side: int) -> torch.Tensor:
    h, w = img.shape[2:]
    k = side / max(h, w)
    if k >= 1.0:
        return img
    return F.interpolate(img, size=(max(1, round(h * k)), max(1, round(w * k))), mode="area")


def _gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    g = torch.exp(-(t**2) / (2 * sigma**2))
    g = g / g.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))


def subject_point(img: torch.Tensor, w0: int, h0: int) -> list[float]:
    """Spectral-residual saliency centroid (Hou & Zhang 2007), in source px."""
    g = _luma(_fit(img, 64))[0, 0]
    spec = torch.fft.fft2(g)
    amp = torch.log(spec.abs() + 1e-8)
    avg = F.avg_pool2d(F.pad(amp[None, None], (1, 1, 1, 1), mode="replicate"), 3, 1)[0, 0]
    sal = torch.fft.ifft2(torch.exp((amp - avg) + 1j * torch.angle(spec))).abs() ** 2
    sal = _gauss(sal[None, None], 2.5)[0, 0]
    wts = (sal - 2.0 * sal.mean()).clamp(min=0)
    if float(wts.sum()) <= 0:
        wts = sal
    hh, ww = sal.shape
    ys = (torch.arange(hh, device=sal.device, dtype=torch.float32) + 0.5) / hh
    xs = (torch.arange(ww, device=sal.device, dtype=torch.float32) + 0.5) / ww
    tot = wts.sum()
    cx = float((wts.sum(0) * xs).sum() / tot)
    cy = float((wts.sum(1) * ys).sum() / tot)
    return [round(cx * w0, 1), round(cy * h0, 1)]


def sharpness(img: torch.Tensor) -> float:
    """Variance of the Laplacian of 0..255 luma at <= 1024 px: higher is sharper."""
    g = _luma(_fit(img, 1024)) * 255.0
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=g.device, dtype=torch.float32)
    return round(float(F.conv2d(g, lap.view(1, 1, 3, 3)).var()), 1)


def dominant_colors(img: torch.Tensor, k: int = 5, iters: int = 12) -> list[str]:
    """k-means over a 64 px thumbnail; hex colours, most-covered first. Deterministic init."""
    px = _fit(img, 64)[0].reshape(3, -1).T.contiguous()
    order = _luma(px.T.reshape(1, 3, -1, 1)).flatten().argsort()
    k = min(k, px.shape[0])
    cent = px[order[((torch.arange(k, device=px.device) + 0.5) * len(order) / k).long()]].clone()
    for _ in range(iters):
        lab = torch.cdist(px, cent).argmin(1)
        for c in range(k):
            sel = lab == c
            if bool(sel.any()):
                cent[c] = px[sel].mean(0)
    lab = torch.cdist(px, cent).argmin(1)
    counts = torch.bincount(lab, minlength=k)
    out = []
    for c in counts.argsort(descending=True).tolist():
        if int(counts[c]) == 0:
            continue
        r, g, b = (int(round(float(v) * 255)) for v in cent[c].clamp(0, 1))
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


def photo_stats(img_u8: torch.Tensor) -> dict:
    """subject, sharpness, brightness, colors for an (H, W, 3) uint8 tensor."""
    h0, w0 = img_u8.shape[:2]
    img = img_u8.permute(2, 0, 1)[None].float() / 255.0
    return {
        "subject": subject_point(img, w0, h0),
        "sharpness": sharpness(img),
        "brightness": round(float(_luma(img).mean()), 3),
        "colors": dominant_colors(img),
    }


# --- clip measurements -------------------------------------------------------------------------


def _ffmpeg() -> str:
    from montaj.encode import FFMPEG

    return FFMPEG


def _stream(cmd: list[str], chunk: int):
    """Yield stdout in `chunk`-byte pieces; raise with the stderr tail on failure."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    try:
        while True:
            buf = proc.stdout.read(chunk)
            if not buf:
                break
            yield buf
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        rc = proc.wait()
    if rc != 0:
        raise AnalysisError(f"ffmpeg failed ({rc}): {err.strip().splitlines()[-1:] or ''}")


def frame_stats(path: Path, fps: float, device: torch.device) -> tuple[list[float], list[float], int]:
    """(cut times s, motion per second 0..1, frames read) from 128x72 frames at native rate."""
    cmd = [_ffmpeg(), "-v", "error", "-hwaccel", "cuda", "-i", str(path), "-map", "0:v:0",
           "-vf", f"scale={CLIP_W}:{CLIP_H}:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    fbytes = CLIP_W * CLIP_H * 3
    prev: torch.Tensor | None = None
    prev_hist: torch.Tensor | None = None
    dists: list[float] = []
    diffs: list[float] = []
    n = 0
    pending = b""
    for buf in _stream(cmd, fbytes * 120):
        pending += buf
        whole = len(pending) // fbytes
        if not whole:
            continue
        arr = np.frombuffer(pending[: whole * fbytes], dtype=np.uint8).reshape(whole, CLIP_H, CLIP_W, 3)
        pending = pending[whole * fbytes:]
        x = torch.from_numpy(arr.copy()).to(device)
        q = (x.long() // 32)  # 8 levels per channel -> 512-bin joint histogram
        idx = (q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]).reshape(whole, -1)
        hist = torch.zeros(whole, 512, device=device).scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
        hist = hist / idx.shape[1]
        rgb = x.float() / 255.0  # per channel: equal-luma colour changes still count as motion
        if prev is not None:
            hist = torch.cat([prev_hist, hist])
            rgb = torch.cat([prev, rgb])
        dists += (0.5 * (hist[1:] - hist[:-1]).abs().sum(1)).tolist()
        diffs += (rgb[1:] - rgb[:-1]).abs().mean((1, 2, 3)).tolist()
        prev, prev_hist = rgb[-1:], hist[-1:]
        n += whole
    rate = fps if fps > 0 else 30.0
    cuts: list[float] = []
    for i, d in enumerate(dists):
        t = (i + 1) / rate
        if d >= CUT_DIST and (not cuts or t - cuts[-1] >= CUT_MIN_GAP_S):
            cuts.append(round(t, 3))
    motion: list[float] = []
    per = max(1, int(round(rate)))
    for s in range(0, len(diffs), per):
        part = diffs[s:s + per]
        motion.append(round(sum(part) / len(part), 4))
    return cuts, motion, n


def _spans(active: list[bool], win: float, min_len: float, bridge: float) -> list[tuple[float, float]]:
    runs: list[list[float]] = []
    for i, on in enumerate(active):
        if not on:
            continue
        t0, t1 = i * win, (i + 1) * win
        if runs and t0 - runs[-1][1] <= bridge + 1e-9:
            runs[-1][1] = t1
        else:
            runs.append([t0, t1])
    return [(a, b) for a, b in runs if b - a >= min_len - 1e-9]


def speech_spans(samples: np.ndarray, rate: int, duration: float) -> tuple[list[dict], list[dict]]:
    """Energy VAD: loud spans (speech) and the quiet gaps between them (silence).

    It hears loudness, not voices: music or traffic counts as speech. `samples` is mono float -1..1.
    """
    win = int(rate * AUDIO_WIN_S)
    n = len(samples) // win
    if n == 0:
        return [], ([{"s": 0.0, "e": round(duration, 3)}] if duration >= SILENCE_MIN_S else [])
    x = samples[: n * win].astype(np.float64).reshape(n, win)  # 50 windows/s: CPU is plenty
    db = 20 * np.log10(np.sqrt((x * x).mean(1)) + 1e-9)
    noise = float(np.quantile(db, 0.1))
    thr = max(SPEECH_FLOOR_DB, noise + SPEECH_OVER_NOISE_DB)
    active = (db > thr).tolist()
    speech = _spans(active, AUDIO_WIN_S, SPEECH_MIN_S, SPEECH_BRIDGE_S)
    end = max(duration, n * AUDIO_WIN_S)
    silence = []
    edge = 0.0
    for a, b in speech + [(end, end)]:
        if a - edge >= SILENCE_MIN_S:
            silence.append({"s": round(edge, 3), "e": round(a, 3)})
        edge = b
    return [{"s": round(a, 3), "e": round(b, 3)} for a, b in speech], silence


def read_audio(path: Path) -> np.ndarray:
    cmd = [_ffmpeg(), "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(AUDIO_RATE),
           "-f", "s16le", "-"]
    raw = b"".join(_stream(cmd, 1 << 20))
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def frame_at(path: Path, t: float) -> np.ndarray:
    """One full-size RGB frame at `t` seconds (ffmpeg's accurate seek)."""
    cmd = [_ffmpeg(), "-v", "error", "-ss", f"{max(t, 0.0):.3f}", "-i", str(path), "-map", "0:v:0",
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    raw = b"".join(_stream(cmd, 1 << 20))
    if not raw:
        raise AnalysisError(f"no frame at {t:.3f}s in {path.name}")
    with Image.open(io.BytesIO(raw)) as im:
        return np.asarray(im.convert("RGB"))


class GpuAnalyzer:
    """Loads RetinaFace once; `photo()` / `clip()` return the analysis rows ingest stores.

    `peak` records the most VRAM (bytes) each stage allocated above what was already held, for the log.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        if device is None:
            if not torch.cuda.is_available():
                raise AnalysisError("analysis needs CUDA (torch.cuda.is_available() is False): use ingest --fast")
            device = torch.device("cuda")
        self.device = device
        self.peak: dict[str, int] = {}
        with self._measure("load"):
            self.net = load_face_net(device)

    @contextmanager
    def _measure(self, stage: str) -> Iterator[None]:
        cuda = self.device.type == "cuda"
        base = 0
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            base = torch.cuda.memory_allocated(self.device)  # other users of this process's torch
        yield
        if cuda:
            got = torch.cuda.max_memory_allocated(self.device) - base
            self.peak[stage] = max(self.peak.get(stage, 0), got)

    def _tensor(self, rgb: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.require(rgb, requirements=["C", "W"])).to(self.device)

    def photo(self, path: Path) -> dict:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        img = self._tensor(rgb)
        with self._measure("faces"):
            faces = detect_faces(self.net, img)
        with self._measure("stats"):
            stats = photo_stats(img)
        return {"w": int(rgb.shape[1]), "h": int(rgb.shape[0]), "faces": faces, **stats}

    def clip(self, path: Path) -> dict:
        from montaj.encode import probe

        info = probe(path)
        with self._measure("clip"):
            cuts, motion, n = frame_stats(path, info.fps, self.device)
        duration = info.duration or (n / info.fps if info.fps else 0.0)
        speech: list[dict] = []
        silence: list[dict] = []
        if info.has_audio:
            speech, silence = speech_spans(read_audio(path), AUDIO_RATE, duration)
        bounds = [0.0, *cuts, duration]
        scenes = []
        w = h = 0
        for s, e in list(zip(bounds[:-1], bounds[1:]))[:MAX_SCENES]:
            rgb = frame_at(path, (s + e) / 2)
            h, w = rgb.shape[:2]
            img = self._tensor(rgb)
            with self._measure("faces"):
                faces = detect_faces(self.net, img)
            pt = subject_point(img.permute(2, 0, 1)[None].float() / 255.0, w, h)
            scenes.append({"s": round(s, 3), "e": round(e, 3), "faces": faces, "subject": pt})
        return {"w": w or info.width, "h": h or info.height, "duration": round(duration, 3),
                "fps": round(info.fps, 3), "cuts": cuts, "speech": speech, "silence": silence,
                "motion": motion, "scenes": scenes}
