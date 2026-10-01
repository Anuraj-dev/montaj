"""Warm-film finish. Port of film2.finish/bloom, parametrised by look + effect lists."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .core import smooth

LOOKS = {"warm-film": {
    "sharpen": 0.3, "scurve": 0.3, "lift": 0.012, "gain": 0.98, "vignette": 0.28,
    "leak_sigma": 8.0, "leak_w": 520.0, "leak_h": 900.0,
    "leak_dx": 280.0, "leak_drift": 24.0, "leak_dy": 8.0,
    "col": [1.0, 0.55, 0.25],
    "intro_bloom": 0.9, "intro_lift": 0.35, "intro_lift_frac": 0.8,
    "outro_bloom": 0.55, "outro_rim": 0.10, "outro_ramp": 45.0,
}}


def _odd(x: float) -> int:
    x = max(1, int(round(x)))
    return x if x % 2 else x + 1


def _binomial(n: int, device) -> torch.Tensor:
    """Separable binomial of odd size n; n=5 is film2's [1,4,6,4,1]/256."""
    m = n - 1
    row: list[int] = []
    c = 1
    for i in range(n):
        row.append(c)
        c = c * (m - i) // (i + 1)
    v = torch.tensor(row, device=device, dtype=torch.float32)
    return (v[:, None] * v[None]) / (v.sum() * v.sum())


def bloom(cv, img, k: float):
    k1, k2 = _odd(9 * cv.k), _odd(41 * cv.k)
    g = F.avg_pool2d(F.avg_pool2d(img[None], k1, 1, k1 // 2), k2, 1, k2 // 2)[0]
    return 1 - (1 - img) * (1 - (g**1.5) * k)


def finish(cv, img, f: float, look: dict, flashes: list, leaks: list, intro, outro):
    """flashes [(at,k)], leaks [(at,k,dir)], intro ("white-lift",dur)|None, outro ("glow",dur,start)|None."""
    L = look
    n = max(3, _odd(5 * cv.k))
    gk = _binomial(n, cv.device)
    bl = F.conv2d(img[:, None], gk[None, None], padding=n // 2)[:, 0]
    img = (img + L["sharpen"] * (img - bl)).clamp(0, 1)
    img = img * img * (3 - 2 * img) * L["scurve"] + img * (1 - L["scurve"])
    vig = (1 - L["vignette"] * ((((cv.XX - cv.W / 2) / (cv.W * 0.62)) ** 2
          + ((cv.YY - cv.H / 2) / (cv.H * 0.62)) ** 2).clamp(0, 1.3)))[None]
    img = (L["lift"] + img * L["gain"]) * vig
    leak = torch.zeros_like(img[:1])
    for c, k, d in leaks:
        e = math.exp(-((f - c) / L["leak_sigma"]) ** 2) * k
        if e < 1e-3:
            continue
        x = cv.W / 2 + d * (f - c) * L["leak_drift"] * cv.k + d * L["leak_dx"] * cv.k
        y = cv.H * 0.35 + (f - c) * L["leak_dy"] * cv.k
        leak = leak + e * torch.exp(-(((cv.XX - x) / (L["leak_w"] * cv.k)) ** 2
                                      + ((cv.YY - y) / (L["leak_h"] * cv.k)) ** 2))[None]
    col = torch.tensor(L["col"], device=cv.device)[:, None, None]
    img = 1 - (1 - img) * (1 - (leak * col).clamp(0, 1))
    for c, k in flashes:
        if 0 <= f - c < 6:
            img = img + (1 - img) * k * (1 - (f - c) / 6) ** 2
    if intro is not None and f < intro[1]:  # soft white bloom settling onto the picture
        img = bloom(cv, img, L["intro_bloom"] * (1 - smooth(f / intro[1])))
    if intro is not None:  # lift width is a fraction of dur so it dies inside the window
        lift_w = L["intro_lift_frac"] * intro[1]
        img = 1 - (1 - img) * (1 - L["intro_lift"] * (1 - smooth(f / lift_w)))
    if outro is not None and f >= outro[2]:  # ending glow grows and stays
        kk = smooth((f - outro[2]) / L["outro_ramp"])
        img = bloom(cv, img, L["outro_bloom"] * kk)
        rim = torch.exp(-(((cv.XX - cv.W / 2) / (cv.W * 0.55)) ** 2
                          + ((cv.YY - cv.H * 0.42) / (cv.H * 0.45)) ** 2))[None]
        img = 1 - (1 - img) * (1 - rim * col * L["outro_rim"] * kk)
    return img.clamp(0, 1)
