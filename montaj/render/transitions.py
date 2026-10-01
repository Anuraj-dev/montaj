"""Screen-space transitions. Port of film2.swirl/whip/fade and film.shift."""
from __future__ import annotations

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


def whip(cv, a, b, x: float, axis: str = "x", d: float = 1):
    e = cubic(x)
    D = (cv.W if axis == "x" else cv.H) * d
    off = -e * D
    if axis == "x":
        return shift(cv, a, off, 0) + shift(cv, b, off + D, 0)
    return shift(cv, a, 0, off) + shift(cv, b, 0, off + D)


def fade(a, b, x: float):
    e = smooth(x)
    return a * (1 - e) + b * e
