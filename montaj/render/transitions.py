"""Screen-space transitions. Port of film2.swirl/whip/fade and film.shift."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .core import cubic, smooth


def shift(cv, img, dx: float, dy: float):
    th = torch.tensor([[1, 0, -2 * dx / cv.W], [0, 1, -2 * dy / cv.H]],
                      device=cv.device, dtype=torch.float32)[None]
    g = F.affine_grid(th, (1, 3, cv.H, cv.W), align_corners=False)
    return F.grid_sample(img[None], g, mode="bilinear", padding_mode="zeros", align_corners=False)[0]


def swirl(cv, a, b, x: float, center: tuple):
    cx, cy = center
    r = torch.sqrt((cv.XX - cx) ** 2 + (cv.YY - cy) ** 2) / (1500 * cv.k)
    ang = torch.atan2(cv.YY - cy, cv.XX - cx)
    prog = cubic(x) * 1.5 - 0.15
    m = torch.sigmoid((prog - r + 0.05 * torch.sin(5 * ang + 9 * r) + 0.06 * cv.lownoise) / 0.035)
    return a * (1 - m) + b * m


def _slide(cv, a, b, e: float, axis: str, d: float = 1):
    D = (cv.W if axis == "x" else cv.H) * d
    off = -e * D
    if axis == "x":
        return shift(cv, a, off, 0) + shift(cv, b, off + D, 0)
    return shift(cv, a, 0, off) + shift(cv, b, 0, off + D)


def whip(cv, a, b, x: float, axis: str = "x", d: float = 1):
    return _slide(cv, a, b, cubic(x), axis, d)


def push(cv, a, b, x: float, axis: str = "x"):
    """b slides in and shoves a out, 1:1, on a gentler ease than whip."""
    if x <= 0:
        return a
    if x >= 1:
        return b
    return _slide(cv, a, b, smooth(x), axis)


def fade(a, b, x: float):
    e = smooth(x)
    return a * (1 - e) + b * e


def rgb(raw: str) -> tuple[float, float, float]:
    """`#rrggbb` -> 0..1 floats. spec.py has already validated the string."""
    n = int(raw[1:], 16)
    return ((n >> 16) & 255) / 255.0, ((n >> 8) & 255) / 255.0, (n & 255) / 255.0


def dip(cv, a, b, x: float, color: str = "#000000"):
    """a settles into a flat colour on the first half, b emerges from it on the second."""
    col = torch.tensor(rgb(color), device=cv.device, dtype=torch.float32)[:, None, None]
    if x < 0.5:
        p = smooth(x * 2)
        return a * (1 - p) + col * p
    q = smooth((x - 0.5) * 2)
    return col * (1 - q) + b * q


def _zoom_blur(cv, img, amount: float, taps: int):
    """Mean of `taps` centre-anchored zooms of img between 1x and (1 + amount)x."""
    if amount <= 0:
        return img
    acc = torch.zeros_like(img)
    for j in range(taps):
        s = 1 + amount * j / (taps - 1)
        th = torch.tensor([[1 / s, 0, 0], [0, 1 / s, 0]], device=cv.device, dtype=torch.float32)[None]
        g = F.affine_grid(th, (1, 3, cv.H, cv.W), align_corners=False)
        acc = acc + F.grid_sample(img[None], g, mode="bilinear", padding_mode="border", align_corners=False)[0]
    return acc / taps


def zoom(cv, a, b, x: float, strength: float = 1.0, taps: int = 8):
    """Zoom-blur punch: a rushes in and smears, b arrives from the same rush and settles."""
    if x <= 0:
        return a
    if x >= 1:
        return b
    e = smooth(x)
    amount = 0.35 * strength
    w = smooth((x - 0.4) / 0.2)  # crossfade only through the middle fifth
    out = None
    if w < 1:
        out = _zoom_blur(cv, a, amount * e, taps) * (1 - w)
    if w > 0:
        part = _zoom_blur(cv, b, amount * (1 - e), taps) * w
        out = part if out is None else out + part
    return out


def _glitch_rng(seed: int, frame: int) -> torch.Generator:
    # A private CPU generator: draws are identical on every device and never touch global state.
    return torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + frame)


def glitch(cv, a, b, x: float, strength: float = 1.0, seed: int = 0, frame: int = 0):
    """Block-noise swap a -> b, horizontal band tears and R/B channel offsets.

    Everything random comes from (seed, frame), so a frame renders the same every time.
    The tear and channel offsets scale with sin(pi x): none at either end.
    """
    if x <= 0:
        return a
    if x >= 1:
        return b
    H, W = cv.H, cv.W
    g = _glitch_rng(seed, frame)
    env = math.sin(math.pi * x) * strength
    bh, bw = max(1, round(H / 24)), max(1, round(W / 12))
    rows, cols = -(-H // bh), -(-W // bw)
    swap = (torch.rand(rows, cols, generator=g) < x)
    swap = swap.repeat_interleave(bh, 0).repeat_interleave(bw, 1)[:H, :W].to(cv.device)
    img = torch.where(swap[None], b, a)

    n_bands = 6
    cuts = torch.sort(torch.randint(0, H, (n_bands - 1,), generator=g)).values
    amp = torch.rand(n_bands, generator=g) * 2 - 1
    live = (torch.rand(n_bands, generator=g) < min(1.0, 0.35 + 0.5 * env)).float()
    band_dx = torch.round(amp * live * 0.1 * W * env).long()
    ys = torch.arange(H)
    row_dx = band_dx[torch.bucketize(ys, cuts, right=True)]
    ch = int(round(0.012 * W * env))

    xs = torch.arange(W)
    ys, xs, row_dx = ys.to(cv.device), xs.to(cv.device), row_dx.to(cv.device)
    out = []
    for c, off in enumerate((ch, 0, -ch)):  # R right, B left
        idx = (xs[None, :] - row_dx[:, None] - off) % W
        out.append(img[c][ys[:, None], idx])
    return torch.stack(out)


def rgb_split(cv, img, amount: float, progress: float):
    """One-off R/B pull along x. `progress` runs 0..1 over the hit; the split decays to nothing."""
    d = int(round(32 * cv.k * amount * (1 - progress) ** 2))
    if d == 0:
        return img
    xs = torch.arange(cv.W, device=cv.device)
    r = img[0][:, (xs - d).clamp(0, cv.W - 1)]
    bl = img[2][:, (xs + d).clamp(0, cv.W - 1)]
    return torch.stack((r, img[1], bl))
