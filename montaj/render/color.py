"""sRGB <-> linear light and the per-shot primary grade (#43). Pure torch, float32 on the canvas device.

`grade(img, g)` takes linear-light (3, H, W) and returns a new tensor; it never writes into `img`,
because cached stills reach it directly. Operator order, each documented in SPEC-REFERENCE:
exposure -> white balance -> contrast -> lift/gamma/gain -> saturation -> vibrance -> luma curve.
`apply(img, g, ctx)` adds the #44 secondaries and film look after it (section at the end).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

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


# ---- #44: secondaries, protect, film look ---------------------------------------------------------
#
# `apply(img, g, ctx)` = primary grade -> each secondary (mask · its own primary adjust) -> stock and split
# tone (skin restored under `protect: skin`) -> halation, bloom -> grain -> gate weave. All on linear light.
# Masks are (1, H, W) in 0..1. Lengths are design px (1080 wide) scaled by `ctx.k`.

HALATION_THRESHOLD = 0.5  # linear luma where highlights start to glow
HALATION_SIGMA = 10.0  # design px
HALATION_TINT = (1.0, 0.28, 0.06)  # red-weighted: film's red layer scatters furthest
BLOOM_SIGMA = 28.0
GRAIN_STRENGTH = 0.12  # linear amplitude at amount 1, mid-grey
WEAVE_PX = 3.0  # design px at gate_weave 1
WEAVE_RATE = 2.5  # noise cells per second


@dataclass(frozen=True)
class GradeContext:
    """What a secondary may read besides the picture. `subject()` -> (1, H, W) | None; `face(pt)` maps a
    source-px point to device px (None when the shot cannot). `t` is the film frame clock (float)."""

    k: float
    t: float
    fps: float
    subject: Callable[[], torch.Tensor | None]
    face: Callable[[tuple[float, float]], tuple[float, float] | None]


def _smooth(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


def _band(v: torch.Tensor, lo: float, hi: float, soft: float) -> torch.Tensor:
    """1 inside [lo, hi], falling to 0 over `soft` outside it."""
    if soft <= 0:
        return ((v >= lo) & (v <= hi)).float()
    return _smooth((v - lo + soft) / soft) * _smooth((hi + soft - v) / soft)


def hsl(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(hue degrees, HSL saturation, Rec.709 luma), each (1, H, W), of linear `img` on display values."""
    e = _encode(img).clamp(0, 1)
    r, g, b = e[0:1], e[1:2], e[2:3]
    hi, lo = e.amax(0, keepdim=True), e.amin(0, keepdim=True)
    c = hi - lo
    cs = c.clamp(min=_EPS)
    h = torch.where(hi == r, torch.remainder((g - b) / cs, 6.0),
                    torch.where(hi == g, (b - r) / cs + 2.0, (r - g) / cs + 4.0)) * 60.0
    h = torch.where(c > _EPS, h, torch.zeros_like(h))
    lum = (hi + lo) / 2
    s = torch.where(c > _EPS, c / (1 - (2 * lum - 1).abs()).clamp(min=_EPS), torch.zeros_like(c)).clamp(0, 1)
    y = e[0:1] * LUMA[0] + e[1:2] * LUMA[1] + e[2:3] * LUMA[2]
    return h, s, y


def key_mask(img: torch.Tensor, key) -> torch.Tensor:
    """HSL qualifier. `soft` is the edge as a fraction of each scale (hue: 360°)."""
    h, s, y = hsl(img)
    m = torch.ones_like(y)
    if key.hue is not None:
        lo, hi = float(key.hue[0]), float(key.hue[1])
        width = (hi - lo) % 360.0 if lo != hi else 0.0
        mid = (lo + width / 2) % 360.0
        d = torch.remainder(h - mid + 180.0, 360.0) - 180.0  # signed distance to the range centre
        m = m * _band(d, -width / 2, width / 2, key.soft * 360.0)
    if key.sat is not None:
        m = m * _band(s, float(key.sat[0]), float(key.sat[1]), key.soft)
    if key.luma is not None:
        m = m * _band(y, float(key.luma[0]), float(key.luma[1]), key.soft)
    return m


def ellipse_mask(H: int, W: int, device, cx: float, cy: float, rx: float, ry: float, feather: float) -> torch.Tensor:
    """Ellipse SDF (first-order distance), device px; 0.5 on the radius, soft over `feather` centred there."""
    ys = torch.arange(H, device=device, dtype=torch.float32)[:, None] + 0.5 - cy
    xs = torch.arange(W, device=device, dtype=torch.float32)[None, :] + 0.5 - cx
    f = torch.sqrt((xs / rx) ** 2 + (ys / ry) ** 2)
    grad = torch.sqrt((xs / rx ** 2) ** 2 + (ys / ry ** 2) ** 2) / f.clamp(min=_EPS)
    d = torch.where(f > _EPS, (f - 1) / grad.clamp(min=_EPS), torch.full_like(f, -min(rx, ry)))
    if feather <= 0:
        return (d <= 0).float()[None]
    return (1 - _smooth(d / feather + 0.5))[None]


def gradient_mask(H: int, W: int, device, a: tuple[float, float], b: tuple[float, float]) -> torch.Tensor:
    """1 at `a`, 0 at `b`, smooth in between along a->b; device px."""
    ys = torch.arange(H, device=device, dtype=torch.float32)[:, None] + 0.5
    xs = torch.arange(W, device=device, dtype=torch.float32)[None, :] + 0.5
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = ((xs - a[0]) * dx + (ys - a[1]) * dy) / (dx * dx + dy * dy)
    return (1 - _smooth(t))[None]


def centroid(m: torch.Tensor) -> tuple[float, float] | None:
    """Weighted centre (device px) of a (1, H, W) matte, or None when it is empty."""
    w = m[0].float()
    total = float(w.sum())
    if total < 1e-3:
        return None
    H, W = w.shape
    ys = torch.arange(H, device=w.device, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=w.device, dtype=torch.float32) + 0.5
    return float((w.sum(0) * xs).sum()) / total, float((w.sum(1) * ys).sum()) / total


def _ellipse_center(e, ctx: GradeContext, subject) -> tuple[float, float] | None:
    if isinstance(e.center, str):
        pt = ctx.face(tuple(e.face_point)) if e.center == "face" and e.face_point is not None else None
        if pt is None:
            m = subject()
            pt = None if m is None else centroid(m)
        if pt is None:
            return None
    else:
        pt = (float(e.center[0]) * ctx.k, float(e.center[1]) * ctx.k)
    if e.offset is not None:
        pt = (pt[0] + float(e.offset[0]) * ctx.k, pt[1] + float(e.offset[1]) * ctx.k)
    return pt


def secondary_mask(img: torch.Tensor, sec, ctx: GradeContext, subject) -> torch.Tensor:
    """The product of a secondary's key, window and matte, then `invert`."""
    _c, H, W = img.shape
    m = torch.ones(1, H, W, device=img.device, dtype=torch.float32)
    if sec.key is not None:
        m = m * key_mask(img, sec.key)
    w = sec.window
    if w is not None and w.ellipse is not None:
        e = w.ellipse
        c = _ellipse_center(e, ctx, subject)
        if c is None:  # no subject this frame: the window selects nothing
            m = torch.zeros_like(m)
        else:
            rx, ry = e.radius if isinstance(e.radius, tuple) else (e.radius, e.radius)
            m = m * ellipse_mask(H, W, img.device, c[0], c[1], float(rx) * ctx.k, float(ry) * ctx.k,
                                 float(e.feather) * ctx.k)
    if w is not None and w.gradient is not None:
        gr = w.gradient
        m = m * gradient_mask(H, W, img.device, (gr.from_[0] * ctx.k, gr.from_[1] * ctx.k),
                              (gr.to[0] * ctx.k, gr.to[1] * ctx.k))
    if sec.matte is not None:
        s = subject()
        m = m * (torch.zeros_like(m) if s is None else s.to(m.dtype).clamp(0, 1))
    return 1 - m if sec.invert else m


def split_tone(img: torch.Tensor, st) -> torch.Tensor:
    """Shadows toward one colour, highlights toward another; luminance-preserving gains.

    Each tone is its linear colour normalised to luma 1, so a blend of the two keeps a grey's luminance.
    `balance` moves the crossover (display luma 0.5 - 0.5·balance); `amount` 0..1 lerps from no change.
    """
    from montaj.grade import hex_rgb

    if st.amount <= 0:
        return img
    def tone(raw: str) -> torch.Tensor:
        c = to_linear(torch.tensor(hex_rgb(raw), device=img.device)).reshape(3, 1, 1)
        y = float((c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2]).item())
        c = c / max(y, _EPS)
        return 1 + st.amount * (c - 1)

    gs, gh = tone(st.shadows), tone(st.highlights)
    p = 0.5 - 0.5 * float(st.balance)
    w = _smooth((_encode(_luma(img)).clamp(0, 1) - p + 0.25) / 0.5)
    return img * (gs + (gh - gs) * w)


def protect_skin(graded: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Where `ref` is skin, keep its hue and saturation and take only the graded luminance."""
    from montaj.grade import SKIN_KEY

    m = key_mask(ref, SKIN_KEY)
    yr, yg = _luma(ref), _luma(graded)
    kept = ref * torch.where(yr > _EPS, yg / yr.clamp(min=_EPS), torch.ones_like(yr))
    return graded + (kept - graded) * m


def halation(img: torch.Tensor, amount: float, k: float) -> torch.Tensor:
    """Red-weighted glow around highlights: light scattered back through the film base."""
    from montaj.render.effects import blur

    w = ((_luma(img) - HALATION_THRESHOLD) / (1 - HALATION_THRESHOLD)).clamp(0, 1)
    glow = blur(_luma(img) * w, HALATION_SIGMA * k)
    return img + amount * glow * _vec(HALATION_TINT, img)


def film_bloom(img: torch.Tensor, amount: float, k: float) -> torch.Tensor:
    from montaj.render.effects import blur

    w = ((_luma(img) - HALATION_THRESHOLD) / (1 - HALATION_THRESHOLD)).clamp(0, 1)
    return img + amount * blur(img * w, BLOOM_SIGMA * k)


def grain_noise(H: int, W: int, device, frame: int, size_px: float) -> torch.Tensor:
    """(1, H, W) unit-variance noise, a pure function of the frame number (cache-safe, deterministic)."""
    h, w = max(1, round(H / size_px)), max(1, round(W / size_px))
    gen = torch.Generator(device=device).manual_seed((frame * 2654435761 + 0x5EED) & 0x7FFFFFFF)
    n = torch.randn(1, 1, h, w, generator=gen, device=device)
    if (h, w) != (H, W):
        n = F.interpolate(n, size=(H, W), mode="bilinear", align_corners=False)
        n = n / n.std().clamp(min=_EPS)
    return n[0]


def film_grain(img: torch.Tensor, amount: float, size: float, frame: int, k: float) -> torch.Tensor:
    """Monochrome grain in linear light, strongest in the midtones (sqrt of luma, fading in highlights)."""
    _c, H, W = img.shape
    y = _luma(img).clamp(0, 1)
    n = grain_noise(H, W, img.device, frame, max(1.0, size * k))
    return (img + amount * GRAIN_STRENGTH * n * y.sqrt() * (1 - 0.5 * y)).clamp(min=0)


def gate_weave(img: torch.Tensor, amount: float, t: float, fps: float, k: float) -> torch.Tensor:
    """The frame wanders a few px in the gate: smooth noise of time, overscanned so no edge shows."""
    from montaj.render.effects import _vnoise, shake_frame

    amp = amount * WEAVE_PX * k
    x = t / float(fps) * WEAVE_RATE
    dx = amp * (2 * _vnoise(x, 4401, 0) - 1)
    dy = amp * 0.6 * (2 * _vnoise(x, 4401, 1) - 1)  # film weaves more sideways than vertically
    return shake_frame(img, dx, dy, amp)


def apply(img: torch.Tensor, g, ctx: GradeContext) -> torch.Tensor:
    """The full grade on linear (3, H, W): primary, secondaries, look. Never writes into `img`."""
    from montaj.grade import STOCKS, Adjust

    x = grade(img, g)
    cache: list = []

    def subject():
        if not cache:
            cache.append(ctx.subject())
        return cache[0]

    for sec in g.secondaries or ():
        m = secondary_mask(x, sec, ctx, subject)
        x = x + (grade(x, sec.adjust) - x) * m
    look = g.look
    if look is None and g.protect is None:
        return x
    ref = x
    film = look.film if look is not None else None
    if film is not None and film.stock is not None:
        x = grade(x, Adjust(**STOCKS[film.stock]))
    if look is not None and look.split_tone is not None:
        x = split_tone(x, look.split_tone)
    if g.protect == "skin":
        x = protect_skin(x, ref)
    if film is None:
        return x
    if film.halation > 0:
        x = halation(x, film.halation, ctx.k)
    if film.bloom > 0:
        x = film_bloom(x, film.bloom, ctx.k)
    if film.grain is not None and film.grain.amount > 0:
        x = film_grain(x, film.grain.amount, film.grain.size, math.floor(ctx.t + 0.5), ctx.k)
    if film.gate_weave > 0:
        x = gate_weave(x, film.gate_weave, ctx.t, ctx.fps, ctx.k)
    return x
