"""Bit-exact torch port of Pillow's `ImageFilter.GaussianBlur` for 8-bit images (#50).

Pillow (src/libImaging/BoxBlur.c, unchanged since 2017) does not run a true Gaussian: it runs `passes`
(3) extended box blurs per axis, x first, each rounding to uint8 in 8.24 fixed point. Every frame of
the wall carries that rounding in its print shadow, and the film2 oracle uses the same PIL call, so a
float Gaussian would change the parity target. This port reproduces the integer maths exactly, on any
device: box radius from sigma (float32, as the C does), replicate edges, and the
`(acc*ww + (far_l+far_r)*fw + 2^23) >> 24` rounding of each pass.
"""
from __future__ import annotations

import numpy as np
import torch


def box_radius(sigma: float, passes: int = 3) -> float:
    """`_gaussian_blur_radius`: the extended box radius whose `passes` repeats have variance sigma²,
    evaluated in the C's mix of float32 and double so the fixed-point weights match bit for bit."""
    f = np.float32
    r = f(sigma)  # PyArg "f": the Python float arrives as a C float
    sigma2 = f(r * r / f(passes))
    L = f(np.sqrt(12.0 * np.float64(sigma2) + 1.0))
    l = f(np.floor((np.float64(L) - 1.0) / 2.0))
    a = f(f(2) * l + f(1)) * f(l * f(l + f(1)) - f(3) * sigma2)
    a = f(a / f(f(6) * f(sigma2 - f(l + f(1)) * f(l + f(1)))))
    return float(f(l + a))


def _weights(radius: float) -> tuple[int, int, int]:
    """(integer radius, ww, fw) of `ImagingHorizontalBoxBlur`: the in-box and the fractional far weight."""
    f = np.float32
    r = int(radius)
    ww = int(f(1 << 24) / f(f(f(radius) * f(2)) + f(1)))  # float division, truncated to UINT32
    fw = ((1 << 24) - (r * 2 + 1) * ww) // 2
    return r, ww, fw


def _box_pass(x: torch.Tensor, r: int, ww: int, fw: int) -> torch.Tensor:
    """One extended box blur along the last axis of int64 values 0..255, edges replicated."""
    n = x.shape[-1]
    idx = torch.arange(-r - 1, n + r + 1, device=x.device).clamp_(0, n - 1)
    p = x[..., idx]  # p[j] = x[clamp(j - r - 1)]
    c = torch.nn.functional.pad(p.cumsum(-1), (1, 0))
    acc = c[..., 2 * r + 2:2 * r + 2 + n] - c[..., 1:1 + n]  # sum of x[clamp(x-r .. x+r)]
    far = p[..., :n] + p[..., 2 * r + 2:2 * r + 2 + n]  # x[clamp(x-r-1)] + x[clamp(x+r+1)]
    return (acc * ww + far * fw + (1 << 23)) >> 24


def gaussian_blur_u8(img: torch.Tensor, sigma: float, passes: int = 3) -> torch.Tensor:
    """`Image.filter(ImageFilter.GaussianBlur(sigma))` for a (..., H, W) uint8 tensor, on its device.
    Returns uint8, identical to Pillow's result."""
    if sigma == 0:
        return img.clone()
    radius = box_radius(sigma, passes)
    x = img.to(torch.int64)
    if radius != 0:
        r, ww, fw = _weights(radius)
        for _ in range(passes):
            x = _box_pass(x, r, ww, fw)
        xt = x.transpose(-1, -2)
        for _ in range(passes):
            xt = _box_pass(xt, r, ww, fw)
        x = xt.transpose(-1, -2)
    return x.to(torch.uint8)
