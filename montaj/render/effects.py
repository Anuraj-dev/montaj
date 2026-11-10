"""Built-in effects (#38): shake, blur, bloom, chroma and trails, on the canvas device.

Each effect is a pure function of its spec fields and the shot clock (`t_local`, frames from the shot
start, fractional under motion blur): shake reads a seeded hash of time, never RNG state, and trails
re-evaluate the picture at earlier times instead of remembering frames. So a segment renders the same
on its own as inside the whole film, which is what the segment cache needs.

Images are (3, H, W) opaque frames or (4, h, w) premultiplied layer windows. Amount 0 returns the input
unchanged (no resample). Lengths are design px (1080 wide), scaled by `Canvas.k` by the caller.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

from montaj.render.shots import css_blur

# Defaults for unset fields. They are engine constants: changing one changes films without a hash change.
SHAKE_AMOUNT = 12.0  # design px, peak offset per axis
SHAKE_RATE = 12.0  # noise cells per second: about a dozen direction changes a second
BLUR_AMOUNT = 8.0  # gaussian sigma, design px
BLOOM_AMOUNT = 0.6
BLOOM_THRESHOLD = 0.75  # luma above which a pixel glows
BLOOM_RADIUS = 24.0  # glow sigma, design px
CHROMA_AMOUNT = 0.3
CHROMA_PX = 32.0  # design px of red/blue split at the corner per unit amount
TRAIL_FRAMES = 4
TRAIL_DECAY = 0.6  # weight of each older frame relative to the next newer one
TRAIL_AMOUNT = 1.0
HIT_FRAMES = 8  # `at` without `dur`

DEFAULT_AMOUNT = {"shake": SHAKE_AMOUNT, "blur": BLUR_AMOUNT, "bloom": BLOOM_AMOUNT, "chroma": CHROMA_AMOUNT,
                  "trails": TRAIL_AMOUNT}
_LUMA = (0.2126, 0.7152, 0.0722)


def _keys(keys, t: float) -> float:
    """Linear between keyframes (t whole frames, increasing), held before the first and after the last."""
    if t <= keys[0].t:
        return float(keys[0].v)
    for a, b in zip(keys, keys[1:]):
        if t < b.t:
            x = (t - a.t) / float(b.t - a.t)
            return float(a.v) + (float(b.v) - float(a.v)) * x
    return float(keys[-1].v)


def level(fx, t_local: float) -> float:
    """The effect's amount at `t_local`: the number or keyframes, times the `at`/`dur` hit envelope.

    A hit is full strength at `at` and falls linearly to nothing at `at + dur`; outside it the effect is off.
    """
    a = fx.amount
    v = DEFAULT_AMOUNT[fx.type] if a is None else _keys(a, t_local) if isinstance(a, list) else float(a)
    if fx.at is not None:
        dur = float(fx.dur if fx.dur is not None else HIT_FRAMES)
        x = (t_local - float(fx.at)) / dur
        if x < 0.0 or x >= 1.0:
            return 0.0
        v *= 1.0 - x
    return max(0.0, v)


# ---- shake ------------------------------------------------------------------------------------------


def _hash(i: int, seed: int, ch: int) -> float:
    """Integer lattice hash to [0, 1). Python ints, so every device and run agrees."""
    h = (i * 374761393 + seed * 668265263 + ch * 144665 + 1013904223) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return h / 4294967296.0


def _vnoise(x: float, seed: int, ch: int) -> float:
    i = math.floor(x)
    f = x - i
    u = f * f * (3.0 - 2.0 * f)
    a, b = _hash(i, seed, ch), _hash(i + 1, seed, ch)
    return a + (b - a) * u


def shake_offset(seed: int, t_local: float, fps: float, amp: float) -> tuple[float, float]:
    """(dx, dy) within ±amp: smooth value noise of time, so motion-blur sub-frames smear it."""
    x = t_local / float(fps) * SHAKE_RATE
    return (amp * (2.0 * _vnoise(x, seed, 0) - 1.0), amp * (2.0 * _vnoise(x, seed, 1) - 1.0))


def shake_frame(img: torch.Tensor, dx: float, dy: float, amp: float) -> torch.Tensor:
    """Move an opaque frame by (dx, dy) device px, overscanned so no edge shows for any |d| <= amp.

    The zoom grows with `amp` (the current envelope), so it eases back to 1 as a hit decays.
    """
    if amp <= 0:
        return img
    _c, H, W = img.shape
    amp = min(amp, min(W, H) / 4.0)
    s = 1.0 / (1.0 - 2.0 * amp / min(W, H))
    theta = torch.tensor([[1.0 / s, 0.0, -2.0 * dx / W], [0.0, 1.0 / s, -2.0 * dy / H]],
                         device=img.device, dtype=img.dtype)[None]
    grid = F.affine_grid(theta, [1, img.shape[0], H, W], align_corners=False)
    return F.grid_sample(img[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]


# ---- blur and bloom --------------------------------------------------------------------------------


def blur(img: torch.Tensor, sigma: float, edge: str = "replicate") -> torch.Tensor:
    """Gaussian blur, sigma device px. `replicate` for a full frame (no dark rim), `zero` for a layer."""
    if sigma < 1e-3:
        return img
    if edge == "zero":
        return css_blur(img, sigma)
    r = int(math.ceil(3.0 * sigma))
    H, W = img.shape[-2:]
    r = min(r, H - 1, W - 1)  # replicate pad needs pad < size
    padded = F.pad(img[None], (r, r, r, r), mode="replicate")[0]
    return css_blur(padded, sigma)[:, r:r + H, r:r + W]


def bloom(img: torch.Tensor, amount: float, threshold: float, sigma: float, edge: str = "replicate") -> torch.Tensor:
    """Pixels brighter than `threshold` (luma) glow: their blurred light is added, `amount` strong."""
    if amount <= 0:
        return img
    rgb = img[:3]
    luma = _LUMA[0] * rgb[0] + _LUMA[1] * rgb[1] + _LUMA[2] * rgb[2]
    w = ((luma - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
    glow = blur(rgb * w[None], sigma, edge)
    out = (rgb + amount * glow).clamp(0.0, 1.0)
    return out if img.shape[0] == 3 else torch.cat((out, img[3:]), dim=0)


# ---- chromatic aberration --------------------------------------------------------------------------


def _scaled(img: torch.Tensor, f: float, cx: float, cy: float) -> torch.Tensor:
    """`img` sampled at c + (p − c)·f, c = (cx, cy) in px: f < 1 enlarges about c."""
    _c, h, w = img.shape
    nx, ny = 2.0 * cx / w - 1.0, 2.0 * cy / h - 1.0
    theta = torch.tensor([[f, 0.0, (1.0 - f) * nx], [0.0, f, (1.0 - f) * ny]], device=img.device,
                         dtype=img.dtype)[None]
    grid = F.affine_grid(theta, [1, img.shape[0], h, w], align_corners=False)
    return F.grid_sample(img[None], grid, mode="bilinear", padding_mode="border" if img.shape[0] != 2 else "zeros",
                         align_corners=False)[0]


def chroma(img: torch.Tensor, shift: float, cx: float, cy: float, ref: float) -> torch.Tensor:
    """Radial red/blue split: red pushed out, blue pulled in, `shift` device px at distance `ref` from c."""
    if shift <= 0 or ref <= 0:
        return img
    s = shift / ref
    if img.shape[0] == 3:
        r = _scaled(img[0:1], 1.0 - s, cx, cy)
        b = _scaled(img[2:3], 1.0 + s, cx, cy)
        return torch.cat((r, img[1:2], b), dim=0)
    # Premultiplied: each split channel carries its own coverage; the union is the new alpha.
    ra = _scaled(torch.cat((img[0:1], img[3:4])), 1.0 - s, cx, cy)
    ba = _scaled(torch.cat((img[2:3], img[3:4])), 1.0 + s, cx, cy)
    a = torch.maximum(img[3:4], torch.maximum(ra[1:2], ba[1:2]))
    return torch.cat((ra[0:1], img[1:2], ba[0:1], a), dim=0)


# ---- trails -----------------------------------------------------------------------------------------


def trails(sample: Callable[[int], torch.Tensor], amount: float, frames: int, decay: float) -> torch.Tensor:
    """The picture averaged with its last `frames` frames (each `decay` times the weight of the next newer).

    `sample(i)` is the picture `i` frames ago, evaluated afresh, so no state carries between frames.
    `amount` mixes from the plain picture (0) to the full average (1).
    """
    cur = sample(0)
    if amount <= 0 or frames <= 0:
        return cur
    acc = cur.clone()
    total = 1.0
    for i in range(1, frames + 1):
        w = decay ** i
        acc += w * sample(i)
        total += w
    return cur + (acc / total - cur) * min(1.0, amount)


# ---- one frame-level effect ---------------------------------------------------------------------------


def frame_effect(cv, img: torch.Tensor, fx, t_local: float, fps: float) -> torch.Tensor:
    """Apply one non-trails built-in effect to an opaque (3, H, W) frame."""
    v = level(fx, t_local)
    if v <= 0:
        return img
    k = float(cv.k)
    if fx.type == "shake":
        amp = v * k
        dx, dy = shake_offset(fx.seed or 0, t_local, fps, amp)
        return shake_frame(img, dx, dy, amp)
    if fx.type == "blur":
        return blur(img, v * k)
    if fx.type == "bloom":
        thr = BLOOM_THRESHOLD if fx.threshold is None else float(fx.threshold)
        return bloom(img, v, thr, (BLOOM_RADIUS if fx.radius is None else float(fx.radius)) * k)
    if fx.type == "chroma":
        _c, H, W = img.shape
        return chroma(img, v * CHROMA_PX * k, W / 2.0, H / 2.0, math.hypot(W, H) / 2.0)
    raise ValueError(f"effect: unknown {fx.type!r}")


def layer_pad(cv, fxs, t_local: float) -> float:
    """Device px a layer's window must grow so its pixel effects (blur, bloom, chroma) are not cut off."""
    k = float(cv.k)
    pad = 0.0
    for fx in fxs:
        v = level(fx, t_local)
        if v <= 0:
            continue
        if fx.type == "blur":
            pad += 3.0 * v * k
        elif fx.type == "bloom":
            pad += 3.0 * (BLOOM_RADIUS if fx.radius is None else float(fx.radius)) * k
        elif fx.type == "chroma":
            pad += 1.5 * v * CHROMA_PX * k
    return pad


def layer_pixels(cv, fxs, t_local: float, ref: float):
    """A `composite` post-filter running the layer's blur/bloom/chroma in list order, or None.

    The filter gets the layer's premultiplied window (4, h, w) and the layer centre in window px.
    `ref` is the layer's on-screen half-diagonal (device px), the chroma split's reference distance.
    """
    steps = [(fx, level(fx, t_local)) for fx in fxs if fx.type in ("blur", "bloom", "chroma")]
    steps = [(fx, v) for fx, v in steps if v > 0]
    if not steps:
        return None
    k = float(cv.k)

    def post(rgba: torch.Tensor, cx: float, cy: float) -> torch.Tensor:
        for fx, v in steps:
            if fx.type == "blur":
                rgba = blur(rgba, v * k, edge="zero")
            elif fx.type == "bloom":
                thr = BLOOM_THRESHOLD if fx.threshold is None else float(fx.threshold)
                rgba = bloom(rgba, v, thr, (BLOOM_RADIUS if fx.radius is None else float(fx.radius)) * k, edge="zero")
            else:
                rgba = chroma(rgba, v * CHROMA_PX * k, cx, cy, ref)
        return rgba

    return post
