"""sRGB <-> linear light and the per-shot primary grade (#43). Pure torch, float32 on the canvas device.

`grade(img, g)` takes linear-light (3, H, W) and returns a new tensor; it never writes into `img`,
because cached stills reach it directly. Operator order, each documented in SPEC-REFERENCE:
exposure -> white balance -> contrast -> lift/gamma/gain -> saturation -> vibrance -> luma curve.
"""
from __future__ import annotations

import math

import torch

LUMA = (0.2126, 0.7152, 0.0722)  # Rec.709 / sRGB primaries, on linear values
_D65_K = 6500.0
# CIE XYZ -> linear sRGB (D65).
_XYZ_TO_RGB = ((3.2406, -1.5372, -0.4986), (-0.9689, 1.8758, 0.0415), (0.0557, -0.2040, 1.0570))
_EPS = 1e-6


def to_linear(img: torch.Tensor) -> torch.Tensor:
    """sRGB-encoded 0..1 -> linear light (IEC 61966-2-1 EOTF). Negatives clamp to 0."""
    x = img.float().clamp(min=0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _encode(lin: torch.Tensor) -> torch.Tensor:
    """Linear -> sRGB encoding without the top clamp, so graded highlights above 1 survive."""
    x = lin.clamp(min=0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


def to_display(lin: torch.Tensor) -> torch.Tensor:
    """Linear light -> sRGB-encoded 0..1, clamped: the output transform."""
    return _encode(lin).clamp(0, 1)


def _luma(img: torch.Tensor) -> torch.Tensor:
    return img[0:1] * LUMA[0] + img[1:2] * LUMA[1] + img[2:3] * LUMA[2]


def _planck_rgb(kelvin: float) -> tuple[float, float, float]:
    """Linear sRGB of a Planckian illuminant at Y = 1 (Kim et al. 2002 xy fit, 1667-25000 K)."""
    t = kelvin
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    X, Y, Z = x / y, 1.0, (1 - x - y) / y
    r, g, b = (m[0] * X + m[1] * Y + m[2] * Z for m in _XYZ_TO_RGB)
    return r, g, b


def wb_gains(temp: float, tint: float) -> tuple[float, float, float]:
    """Per-channel gains. `temp` K warmer (+) or cooler (-) than 6500 K; `tint` + magenta, - green.

    The gains are the ratio of the target illuminant to 6500 K, so temp 0 tint 0 is exactly 1, and
    they are normalised so a grey's Rec.709 luminance does not move.
    """
    if temp == 0 and tint == 0:
        return 1.0, 1.0, 1.0
    ref = _planck_rgb(_D65_K)
    tgt = _planck_rgb(_D65_K - temp)
    r, g, b = (tgt[i] / ref[i] for i in range(3))
    g *= 2.0 ** (-tint / 200.0)
    n = r * LUMA[0] + g * LUMA[1] + b * LUMA[2]
    return r / n, g / n, b / n


def _vec(v, like: torch.Tensor) -> torch.Tensor:
    """A number or [r, g, b] as a (3|1, 1, 1) tensor on `like`'s device."""
    return torch.as_tensor(v, device=like.device, dtype=torch.float32).reshape(-1, 1, 1)


def _is_identity(v, unit: float) -> bool:
    return all(float(c) == unit for c in (v if isinstance(v, (list, tuple)) else (v,)))


def monotone_curve(points, x: torch.Tensor) -> torch.Tensor:
    """Fritsch-Carlson monotone cubic through `points` [(x, y)], evaluated at `x`.

    Outside the first/last knot the curve continues at slope 1 from that knot, so it stays continuous.
    """
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    n = len(xs)
    d = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(n - 1)]
    m = [d[0]] + [0.0 if d[i - 1] * d[i] <= 0 else (d[i - 1] + d[i]) / 2 for i in range(1, n - 1)] + [d[-1]]
    for i in range(n - 1):
        if d[i] == 0:
            m[i] = m[i + 1] = 0.0
            continue
        a, b = m[i] / d[i], m[i + 1] / d[i]
        s = a * a + b * b
        if s > 9:
            tau = 3 / math.sqrt(s)
            m[i], m[i + 1] = tau * a * d[i], tau * b * d[i]
    dev = x.device
    kx = torch.tensor(xs, device=dev, dtype=torch.float32)
    ky = torch.tensor(ys, device=dev, dtype=torch.float32)
    km = torch.tensor(m, device=dev, dtype=torch.float32)
    i = (torch.bucketize(x.contiguous(), kx, right=True) - 1).clamp(0, n - 2)
    x0, x1, y0, y1, m0, m1 = kx[i], kx[i + 1], ky[i], ky[i + 1], km[i], km[i + 1]
    h = x1 - x0
    t = ((x - x0) / h).clamp(0, 1)
    t2, t3 = t * t, t * t * t
    inner = ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * m0
             + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * m1)
    out = torch.where(x < xs[0], x - xs[0] + ys[0], inner)
    return torch.where(x > xs[-1], x - xs[-1] + ys[-1], out)


def grade(img: torch.Tensor, g) -> torch.Tensor:
    """Primary grade on linear light. `g` is a `montaj.grade.Grade`. Returns a new float32 tensor."""
    x = img.float()
    if g.exposure:
        x = x * (2.0 ** g.exposure)
    if g.temp or g.tint:
        x = x * _vec(wb_gains(g.temp, g.tint), x)
    if g.contrast != 1:
        p = float(to_linear(torch.tensor(g.pivot)))
        x = p * (x.clamp(min=0) / p) ** g.contrast
    if not (_is_identity(g.lift, 0) and _is_identity(g.gamma, 1) and _is_identity(g.gain, 1)):
        # Wheels act on display-encoded values, where colourists expect them.
        e = _encode(x)
        e = _vec(g.gain, x) * (e + _vec(g.lift, x) * (1 - e))
        e = e.clamp(min=0) ** (1 / _vec(g.gamma, x))
        x = to_linear(e)
    if g.saturation != 1:
        y = _luma(x)
        x = y + (x - y) * g.saturation
    if g.vibrance:
        y = _luma(x)
        hi = x.amax(0, keepdim=True)
        sat = (hi - x.amin(0, keepdim=True)) / hi.clamp(min=_EPS)
        x = y + (x - y) * (1 + g.vibrance * (1 - sat.clamp(0, 1)))
    if g.curves is not None and g.curves.luma is not None:
        y = _luma(x).clamp(min=0)
        y2 = to_linear(monotone_curve(g.curves.luma, _encode(y)))
        x = x * torch.where(y > _EPS, y2 / y.clamp(min=_EPS), torch.ones_like(y))
    return x if x is not img else x.clone()
