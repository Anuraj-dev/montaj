"""Golden-film finish layers. One pure function per layer; film.html 364–436.

`t` is seconds, `f` is frames. Design px are multiplied by `cv.k`.
`mulberry32` is film.html line 74, 32-bit integer semantics.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from .shots import css_blur

_W, _H = 1080.0, 1920.0


def _bits(x: int) -> int:
    return x & 0xFFFFFFFF


def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x


def _imul(a: int, b: int) -> int:
    return _i32(_bits(a) * _bits(b))


def _xor(a: int, b: int) -> int:
    return _i32(_bits(a) ^ _bits(b))


def _or(a: int, b: int) -> int:
    return _i32(_bits(a) | _bits(b))


def mulberry32(seed: int):
    """film.html line 74. Each call returns the next float in [0, 1)."""
    s = _i32(int(seed))

    def nxt() -> float:
        nonlocal s
        s = _i32(s + 0x6D2B79F5)
        t = _imul(_xor(s, _bits(s) >> 15), _or(1, s))
        t = _xor(t + _imul(_xor(t, _bits(t) >> 7), _or(61, t)), t)
        return _bits(_xor(t, _bits(t) >> 14)) / 4294967296

    return nxt


def mulberry32_block(seed: int, n: int) -> np.ndarray:
    """The first `n` draws of `mulberry32(seed)` at once, bit-identical.

    The state only ever adds 0x6D2B79F5, so draw `j` is a pure function of `seed + (j + 1)·0x6D2B79F5`.
    uint32 arrays wrap like JS 32-bit ints; `>>` on uint32 is JS `>>>`.
    """
    j = np.arange(1, n + 1, dtype=np.uint64)
    a = ((np.uint64(int(seed) & 0xFFFFFFFF) + j * np.uint64(0x6D2B79F5)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    t = (a ^ (a >> np.uint32(15))) * (a | np.uint32(1))
    t = (t + (t ^ (t >> np.uint32(7))) * (t | np.uint32(61))) ^ t
    return (t ^ (t >> np.uint32(14))).astype(np.float64) / 4294967296


def mote_params() -> list[tuple[float, float, float, float, float, float]]:
    """70 motes from mulberry32(22), film.html line 366. Design px on 1080×1920."""
    return list(_motes())


@lru_cache(maxsize=1)
def _motes() -> tuple[tuple[float, float, float, float, float, float], ...]:
    """Constant per process (fixed seed), so computed once."""
    r = mulberry32(22)
    out = []
    for _ in range(70):
        x, y = r() * _W, r() * _H
        s = 1.5 + r() * r() * 9  # two draws
        v = 8 + r() * 30
        ph = r() * 6.28
        dr = r() * 40
        out.append((x, y, s, v, ph, dr))
    return tuple(out)


def burst_parts(index: int, n: int) -> list[tuple[float, float, float, float, float]]:
    """film.html lines 368–370. Seed is `100 + index`. Returns vx, vy, s, life, tw."""
    r = mulberry32(100 + int(index))
    parts = []
    for _ in range(int(n)):
        a = r() * 6.283
        sp = 250 + r() * 900
        parts.append((math.cos(a) * sp, math.sin(a) * sp - 200, 1.5 + r() * 4, 1.4 + r() * 1.8, r() * 6))
    return parts


def _screen(img: torch.Tensor, premul: torch.Tensor) -> torch.Tensor:
    """Screen of a premultiplied layer: `b + c·(1−b)`."""
    return img + premul * (1 - img)


def _radial(dist: torch.Tensor, radius: float, stops: list[tuple[float, tuple, float]]):
    """Canvas radial gradient, straight-alpha interpolation, then premultiply.

    Chromium's canvas (unlike CSS) interpolates these stops in straight RGBA.
    """
    u = (dist / radius).clamp(0, 1)
    rgb = torch.zeros(dist.shape + (3,), device=dist.device, dtype=dist.dtype)
    alpha = torch.zeros_like(dist)
    for (t0, c0, a0), (t1, c1, a1) in zip(stops, stops[1:]):
        span = max(t1 - t0, 1e-6)
        w = ((u - t0) / span).clamp(0, 1)
        active = (u >= t0) & (u <= t1 if t1 < stops[-1][0] else u >= t0)
        # Last segment includes the endpoint; earlier segments own their right edge.
        col = torch.tensor(c0, device=dist.device, dtype=dist.dtype) * (1 - w[..., None]) \
            + torch.tensor(c1, device=dist.device, dtype=dist.dtype) * w[..., None]
        al = a0 * (1 - w) + a1 * w
        rgb = torch.where(active[..., None], col, rgb)
        alpha = torch.where(active, al, alpha)
    return rgb * alpha[..., None], alpha


def _stamp(acc_p: torch.Tensor, acc_a: torch.Tensor, cx: float, cy: float, radius: float,
           stops: list[tuple[float, tuple, float]]) -> None:
    if radius <= 0.4:
        return
    H, W = acc_a.shape
    x0 = max(0, int(math.floor(cx - radius)))
    x1 = min(W, int(math.ceil(cx + radius)) + 1)
    y0 = max(0, int(math.floor(cy - radius)))
    y1 = min(H, int(math.ceil(cy + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    xs = torch.arange(x0, x1, device=acc_a.device, dtype=torch.float32) + 0.5
    ys = torch.arange(y0, y1, device=acc_a.device, dtype=torch.float32) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    dist = torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    prem, al = _radial(dist, radius, stops)
    acc_p[:, y0:y1, x0:x1] += prem.permute(2, 0, 1)
    acc_a[y0:y1, x0:x1] += al


def leaks(cv, img: torch.Tensor, t: float, glow: float, dur: float) -> torch.Tensor:
    """Two discs, blur 60, screen. film.html lines 426–429. `t` and `dur` in seconds.

    The blur runs on the whole disc, including the part past the stage, and `.L`
    clips afterwards (film.html line 18). A screen-sized pad cuts that tail off.
    """
    lk = max(0.0, min(1.0, float(glow))) * (1 - max(0.0, min(1.0, (t - (dur - 2.0)) / 2.0)))
    if lk <= 1e-4:
        return img
    k = cv.k
    sigma = 60.0 * k
    blur_r = int(math.ceil(sigma * 3))
    specs = (
        (900.0, -300 + 180 * math.sin(t * 0.23), 200 + 300 * math.sin(t * 0.17 + 1),
         (1.0, 140 / 255, 60 / 255), 0.55 * lk),
        (1000.0, 500 + 160 * math.cos(t * 0.19), 1000 + 280 * math.sin(t * 0.21),
         (1.0, 70 / 255, 110 / 255), 0.35 * lk),
    )
    out = img.clone()
    for size, left, top, rgb, alpha in specs:
        cx = (left + size / 2) * k
        cy = (top + size / 2) * k
        reach = 0.65 * (size * k / 2) * math.sqrt(2)  # farthest-corner, stop at 65%
        x0 = int(math.floor(cx - reach)) - blur_r
        y0 = int(math.floor(cy - reach)) - blur_r
        x1 = int(math.ceil(cx + reach)) + blur_r + 1
        y1 = int(math.ceil(cy + reach)) + blur_r + 1
        xs = torch.arange(x0, x1, device=img.device, dtype=img.dtype) + 0.5
        ys = torch.arange(y0, y1, device=img.device, dtype=img.dtype) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        fade = (1 - torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2) / reach).clamp(0, 1)
        # Premultiplied sRGB: transparent stop keeps the hue, alpha falls linearly.
        buf = torch.stack([torch.tensor(c, device=img.device, dtype=img.dtype) * alpha * fade for c in rgb])
        blurred = css_blur(buf, sigma)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(cv.W, x1), min(cv.H, y1)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        patch = blurred[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
        out[:, sy0:sy1, sx0:sx1] = _screen(out[:, sy0:sy1, sx0:sx1], patch)
    return out.clamp(0, 1)


def dust(cv, img: torch.Tensor, t: float, level: float, bursts: tuple = ()) -> torch.Tensor:
    """Motes + sparkle bursts drawn additive ('lighter'), then source-over. film.html 373–397.

    `level` is the dust track. `bursts` is `(index, at, x, y, n)` in seconds and design px.
    Mote level is `clamp(level, 0, 1.2) · clamp(t / 2s)`.
    """
    k = cv.k
    strength = max(0.0, min(1.2, float(level))) * max(0.0, min(1.0, t / 2.0))
    acc_p = torch.zeros(3, cv.H, cv.W, device=img.device)
    acc_a = torch.zeros(cv.H, cv.W, device=img.device)
    if strength > 0:
        for x, y, s, v, ph, dr in mote_params():
            yy = (y - v * t) % _H
            xx = x + math.sin(t * 0.4 + ph) * dr
            a = strength * (0.35 + 0.35 * math.sin(t * 1.7 + ph * 3))
            if a <= 0:
                continue
            rad = s * 3 * k
            stops = (
                (0.0, (255 / 255, 226 / 255, 160 / 255), a),
                (0.35, (240 / 255, 190 / 255, 110 / 255), a * 0.45),
                (1.0, (240 / 255, 190 / 255, 110 / 255), 0.0),
            )
            _stamp(acc_p, acc_a, xx * k, yy * k, rad, stops)
    for index, at, bx, by, n in bursts:
        dt = t - float(at)
        if dt < 0 or dt > 3.4:
            continue
        for vx, vy, s, life, tw in burst_parts(int(index), int(n)):
            if dt > life:
                continue
            drag = (1 - math.exp(-dt * 1.6)) / 1.6
            px = (bx + vx * drag) * k
            py = (by + vy * drag + 90 * dt * dt) * k
            a = (1 - dt / life) * (0.6 + 0.4 * math.sin(dt * 20 + tw))
            stops = (
                (0.0, (255 / 255, 248 / 255, 225 / 255), a),
                (0.3, (255 / 255, 205 / 255, 120 / 255), a * 0.6),
                (1.0, (255 / 255, 190 / 255, 100 / 255), 0.0),
            )
            _stamp(acc_p, acc_a, px, py, s * 4 * k, stops)
    acc_a = acc_a.clamp(0, 1)
    acc_p = torch.minimum(acc_p, acc_a[None]).clamp(0, 1)
    return (acc_p + (1 - acc_a) * img).clamp(0, 1)


def vignette(cv, img: torch.Tensor) -> torch.Tensor:
    """film.html line 37. Ellipse 75%×60% at 50% 48%, premultiplied black stops."""
    rx, ry = 0.75 * cv.W, 0.60 * cv.H
    dx = (cv.XX + 0.5 - 0.50 * cv.W) / rx
    dy = (cv.YY + 0.5 - 0.48 * cv.H) / ry
    d = torch.sqrt(dx * dx + dy * dy)
    a = torch.zeros_like(d)
    m = (d > 0.45) & (d <= 0.85)
    a = torch.where(m, 0.55 * (d - 0.45) / 0.40, a)
    m = (d > 0.85) & (d <= 1)
    a = torch.where(m, 0.55 + 0.30 * (d - 0.85) / 0.15, a)
    a = torch.where(d > 1, torch.full_like(a, 0.85), a)
    return (img * (1 - a)).clamp(0, 1)


@lru_cache(maxsize=4)
def _grain_bytes(i: int) -> np.ndarray:
    """film.html line 399. One rng draw per pixel, stored as a uint8 gray. Pure in `i`, so memoised."""
    vals = mulberry32_block(7 + i, 256 * 256)
    # ToUint8Clamp rounds halves to even, which is what np.rint does.
    byte = np.clip(np.rint(vals * 255.0), 0, 255).astype(np.float32).reshape(256, 256) / 255.0
    byte.flags.writeable = False
    return byte


def _grain_tile(i: int, device) -> torch.Tensor:
    return torch.from_numpy(_grain_bytes(i).copy()).to(device)


def grain(cv, img: torch.Tensor, f: float) -> torch.Tensor:
    """Tile `f % 4`, offset `((f·97) % 256, (f·57) % 256)` design px. Overlay at .16.

    film.html lines 36, 421–422.
    """
    fi = int(f)
    tile = _grain_tile(fi % 4, img.device)
    ox = (fi * 97) % 256 * cv.k
    oy = (fi * 57) % 256 * cv.k
    span = 256 * cv.k
    ix = torch.floor(torch.remainder(cv.XX - ox, span) / cv.k).long() % 256
    iy = torch.floor(torch.remainder(cv.YY - oy, span) / cv.k).long() % 256
    s = tile[iy, ix]
    b = img
    blended = torch.where(b <= 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))
    return (b * (1 - 0.16) + blended * 0.16).clamp(0, 1)


def _hit_opacity(t: float, hits) -> float:
    """film.html line 423, with each hit's own k. `d` in seconds. Max over hits."""
    op = 0.0
    for at, k in hits:
        d = t - float(at)
        v = ((d + 0.08) / 0.08) * k if d < 0 else k * math.exp(-3.2 * d)
        if v > op:
            op = v
    return max(0.0, min(1.0, op))


def hits(cv, img: torch.Tensor, t: float, hit_list) -> torch.Tensor:
    """Golden radial at 50% 45%, screen. film.html lines 39 and 423–424."""
    op = _hit_opacity(t, hit_list)
    if op <= 1e-4:
        return img
    cx, cy = 0.5 * cv.W, 0.45 * cv.H
    R = math.hypot(max(cx, cv.W - cx), max(cy, cv.H - cy))  # farthest corner
    dist = torch.sqrt((cv.XX + 0.5 - cx) ** 2 + (cv.YY + 0.5 - cy) ** 2)
    u = (dist / R).clamp(0, 1)
    c0 = torch.tensor((255, 248, 230), device=img.device, dtype=img.dtype) / 255
    c1 = torch.tensor((243, 196, 122), device=img.device, dtype=img.dtype) / 255
    c2 = torch.tensor((176, 102, 42), device=img.device, dtype=img.dtype) / 255
    w = (u / 0.6).clamp(0, 1)
    w2 = ((u - 0.6) / 0.4).clamp(0, 1)
    first = (u <= 0.6)[:, :, None]
    col = torch.where(first, c0 * (1 - w[:, :, None]) + c1 * w[:, :, None],
                      c1 * (1 - w2[:, :, None]) + c2 * w2[:, :, None])
    return _screen(img, col.permute(2, 0, 1) * op).clamp(0, 1)


def bars(cv, img: torch.Tensor, height: float) -> torch.Tensor:
    """Black letterbox, `height` design px. film.html lines 41 and 430."""
    h = float(height) * cv.k
    if h <= 0:
        return img
    rows = torch.arange(cv.H, device=img.device, dtype=img.dtype)
    top = (h - rows).clamp(0, 1)
    bot = (rows + 1 - cv.H + h).clamp(0, 1)
    cov = torch.maximum(top, bot)[None, :, None]
    return (img * (1 - cov)).clamp(0, 1)


def fade_out(img: torch.Tensor, f: float, start: float, dur: float) -> torch.Tensor:
    """film.html line 435. `f`, `start`, `dur` in frames. Multiply by `1 − clamp((f−start)/dur)`."""
    if dur <= 0:
        return img
    a = 1.0 - max(0.0, min(1.0, (float(f) - float(start)) / float(dur)))
    return img * a
