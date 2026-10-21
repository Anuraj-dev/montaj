"""Vector HUD overlays: ring/arc, brackets, polyline, waveform. Pure `fn(cv, img, overlays, f)`.

Each shape is a signed distance in device px, anti-aliased with a 1 px linear ramp
(coverage = 0.5 - d), drawn on a bounding window so a small ring costs a small tensor.
Design-px values scale by `cv.k`. A `glow` blurs the coverage and screens it under the
crisp shape. The renderer reads only the ResolvedOverlay values it is handed (no files).
A waveform's bar heights are `levels` computed at resolve time; this module never opens audio.
"""
from __future__ import annotations

import math

import torch

from montaj.timeline import waveform_index

from .core import EASE
from .shots import css_blur

_TWO_PI = 2.0 * math.pi


def _rgb(color: str) -> tuple[float, float, float]:
    n = int(color[1:], 16)
    return ((n >> 16) & 255) / 255.0, ((n >> 8) & 255) / 255.0, (n & 255) / 255.0


def _progress(f: float, a: float, b: float, ease: str) -> float:
    """Eased 0..1 position of frame f inside [a, b]."""
    x = 0.0 if b <= a else (f - a) / (b - a)
    return EASE[ease](min(1.0, max(0.0, x)))


def _lerp(pair: tuple[float, float], e: float) -> float:
    return pair[0] + (pair[1] - pair[0]) * e


def _ring_cov(X, Y, c, r: float, half: float, sweep: float) -> torch.Tensor:
    dx, dy = X - c[0], Y - c[1]
    rho = torch.sqrt(dx * dx + dy * dy)
    cov = (0.5 - (torch.abs(rho - r) - half)).clamp(0.0, 1.0)
    if sweep >= 360.0:
        return cov
    # Angle runs clockwise from 12 o'clock over 0..2π. The ramps are in px along the arc:
    # the start edge only matters just past 12 o'clock, the end edge at the sweep angle.
    ang = torch.atan2(dx, -dy) % _TWO_PI
    rr = rho.clamp(min=1.0)
    start = torch.where(ang < math.pi, (ang * rr + 0.5).clamp(0.0, 1.0), torch.ones_like(ang))
    return cov * start * ((math.radians(sweep) - ang) * rr + 0.5).clamp(0.0, 1.0)


def _seg_dist(X, Y, a, b) -> torch.Tensor:
    abx, aby = b[0] - a[0], b[1] - a[1]
    denom = max(abx * abx + aby * aby, 1e-12)
    t = (((X - a[0]) * abx + (Y - a[1]) * aby) / denom).clamp(0.0, 1.0)
    return torch.hypot(X - (a[0] + t * abx), Y - (a[1] + t * aby))


def _poly_dist(X, Y, pts, visible: float | None) -> torch.Tensor | None:
    """Distance to the first `visible` px of a polyline (all of it when None). None if empty."""
    dist = None
    run = 0.0
    for a, b in zip(pts, pts[1:]):
        n = math.hypot(b[0] - a[0], b[1] - a[1])
        if n <= 0.0:
            continue
        frac = 1.0 if visible is None else min(1.0, max(0.0, (visible - run) / n))
        run += n
        if frac <= 0.0:
            continue
        end = (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)
        d = _seg_dist(X, Y, a, end)
        dist = d if dist is None else torch.minimum(dist, d)
    return dist


def _brackets(box, arm: float, p: float, k: float):
    """Four L-shaped corners in device px; arms grow from the corner with p."""
    x0, y0, x1, y1 = (v * k for v in box)
    a = min(arm * k, (x1 - x0) / 2, (y1 - y0) / 2) * p
    return [
        [(x0 + a, y0), (x0, y0), (x0, y0 + a)],
        [(x1 - a, y0), (x1, y0), (x1, y0 + a)],
        [(x1 - a, y1), (x1, y1), (x1, y1 - a)],
        [(x0 + a, y1), (x0, y1), (x0, y1 - a)],
    ]


def _shape(ov, f: float, k: float):
    """(kind, bounds), or None when nothing is visible at f.

    kind is ("ring", (cx, cy), r, sweep, half) or ("poly", [pts...], visible, half); bounds is
    the shape's own extent (x0, y0, x1, y1) in device px, before the half-width margin.
    """
    e = _progress(f, ov.from_, ov.to, ov.ease)
    half = ov.width * k / 2.0
    if ov.shape == "ring":
        r = _lerp(ov.radius, e) * k
        sweep = _lerp(ov.sweep, e)
        if sweep <= 0.0 or r + half <= 0.0:
            return None
        c = (ov.center[0] * k, ov.center[1] * k)
        return ("ring", c, r, sweep, half), (c[0] - r, c[1] - r, c[0] + r, c[1] + r)
    p = 1.0 if ov.draw is None else _progress(f, ov.from_, ov.from_ + ov.draw, ov.ease)
    if p <= 0.0:
        return None
    if ov.shape == "brackets":
        polys = _brackets(ov.box, ov.arm, p, k)
        visible = None
    else:
        polys = [[(x * k, y * k) for x, y in ov.points]]
        total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(polys[0], polys[0][1:]))
        visible = None if p >= 1.0 else p * total
    xs = [x for poly in polys for x, _ in poly]
    ys = [y for poly in polys for _, y in poly]
    return ("poly", polys, visible, half), (min(xs), min(ys), max(xs), max(ys))


def _coverage(kind, X, Y) -> torch.Tensor | None:
    if kind[0] == "ring":
        _, c, r, sweep, half = kind
        return _ring_cov(X, Y, c, r, half, sweep)
    _, polys, visible, half = kind
    dist = None
    for poly in polys:
        d = _poly_dist(X, Y, poly, visible)
        if d is not None:
            dist = d if dist is None else torch.minimum(dist, d)
    return None if dist is None else (0.5 - (dist - half)).clamp(0.0, 1.0)


def _pixel_span(a: float, b: float, limit: int) -> tuple[int, int]:
    """Device pixels whose centres lie in [a, b), clipped to [0, limit)."""
    lo = max(0, math.ceil(a - 0.5))
    hi = min(limit, math.floor(b - 0.5 - 1e-9) + 1)
    return lo, hi


def _draw_waveform(cv, img: torch.Tensor, ov, f: float) -> torch.Tensor:
    """Vertical bars in `box`, mirrored about its horizontal centre. Nothing outside the box."""
    if not ov.levels or ov.box is None:
        return img
    i = waveform_index(ov.from_, f)
    if i < 0 or i >= len(ov.levels):
        return img
    row = ov.levels[i]
    if not row or max(row) <= 0.0:
        return img
    k = float(cv.k)
    x0, y0, x1, y1 = (float(v) * k for v in ov.box)
    if x1 <= x0 or y1 <= y0:
        return img
    ix0, ix1 = _pixel_span(x0, x1, cv.W)
    iy0, iy1 = _pixel_span(y0, y1, cv.H)
    if ix1 <= ix0 or iy1 <= iy0:
        return img
    X = (torch.arange(ix0, ix1, device=img.device, dtype=img.dtype) + 0.5)[None, :].expand(iy1 - iy0, -1)
    Y = (torch.arange(iy0, iy1, device=img.device, dtype=img.dtype) + 0.5)[:, None].expand(-1, ix1 - ix0)
    n = len(row)
    slot = (x1 - x0) / n
    hw = min(float(ov.width) * k, slot) / 2.0
    cy = (y0 + y1) / 2.0
    half_box = (y1 - y0) / 2.0
    cov = torch.zeros((iy1 - iy0, ix1 - ix0), device=img.device, dtype=img.dtype)
    zero = torch.zeros((), device=img.device, dtype=img.dtype)
    for bi, level in enumerate(row):
        if level <= 0.0:
            continue
        cx = x0 + (bi + 0.5) * slot
        hh = float(level) * half_box
        dx = (X - cx).abs() - hw
        dy = (Y - cy).abs() - hh
        dist = torch.hypot(dx.clamp(min=0), dy.clamp(min=0)) + torch.minimum(torch.maximum(dx, dy), zero)
        cov = torch.maximum(cov, (0.5 - dist).clamp(0.0, 1.0))
    mask = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
    cov = torch.where(mask, cov, torch.zeros_like(cov))
    sigma = (2.0 * ov.width + 4.0) * k if ov.glow > 0 else 0.0
    g = None
    if sigma > 0.0:
        g = (css_blur(cov[None], sigma)[0] * ov.glow * ov.opacity).clamp(0.0, 1.0)
        g = torch.where(mask, g, torch.zeros_like(g))
    rgb = torch.tensor(_rgb(ov.color), device=img.device, dtype=img.dtype)[:, None, None]
    win = img[:, iy0:iy1, ix0:ix1]
    if g is not None:
        win = 1.0 - (1.0 - win) * (1.0 - rgb * g)
    painted = win * (1.0 - cov * ov.opacity) + rgb * (cov * ov.opacity)
    img[:, iy0:iy1, ix0:ix1] = torch.where(mask[None], painted, img[:, iy0:iy1, ix0:ix1])
    return img


def draw_overlay(cv, img: torch.Tensor, ov, f: float) -> torch.Tensor:
    """Paint one overlay at frame f onto `img` (3, H, W) in place; returns it."""
    if ov.shape == "waveform":
        return _draw_waveform(cv, img, ov, f)
    k = float(cv.k)
    got = _shape(ov, f, k)
    if got is None:
        return img
    kind, (bx0, by0, bx1, by1) = got
    half = kind[-1]
    sigma = (2.0 * ov.width + 4.0) * k if ov.glow > 0 else 0.0
    pad = math.ceil(half + 2.0 + 3.0 * sigma)
    # Rasterise and blur on a window that reaches `pad` past the canvas: an offscreen stroke
    # still lights visible pixels. Only the canvas part is composited.
    x0, y0 = max(-pad, int(math.floor(bx0)) - pad), max(-pad, int(math.floor(by0)) - pad)
    x1, y1 = min(cv.W + pad, int(math.ceil(bx1)) + pad), min(cv.H + pad, int(math.ceil(by1)) + pad)
    if x1 <= x0 or y1 <= y0:
        return img
    X = (torch.arange(x0, x1, device=img.device, dtype=img.dtype) + 0.5)[None, :].expand(y1 - y0, -1)
    Y = (torch.arange(y0, y1, device=img.device, dtype=img.dtype) + 0.5)[:, None].expand(-1, x1 - x0)
    cov = _coverage(kind, X, Y)
    if cov is None:
        return img
    g = None
    if sigma > 0:
        g = (css_blur(cov[None], sigma)[0] * ov.glow * ov.opacity).clamp(0.0, 1.0)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, cv.W), min(y1, cv.H)
    if cx1 <= cx0 or cy1 <= cy0:
        return img
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    cov = cov[sl]
    rgb = torch.tensor(_rgb(ov.color), device=img.device, dtype=img.dtype)[:, None, None]
    win = img[:, cy0:cy1, cx0:cx1]
    if g is not None:
        # Screen the blurred coverage under the shape so glow brightens, never darkens.
        win = 1.0 - (1.0 - win) * (1.0 - rgb * g[sl])
    a = cov * ov.opacity
    img[:, cy0:cy1, cx0:cx1] = win * (1.0 - a) + rgb * a
    return img


def draw_overlays(cv, img: torch.Tensor, overlays, f: float) -> torch.Tensor:
    """Composite `overlays` (in list order) over `img` at frame f. Returns a new tensor."""
    overlays = tuple(overlays)
    if not overlays:
        return img
    with torch.no_grad():
        out = img.clone()
        for ov in overlays:
            draw_overlay(cv, out, ov, f)
        return out
