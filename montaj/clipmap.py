"""Clip time and camera maps. Pure functions of the shot spec, in film frames.

`source_index` maps a shot frame to the source frame it shows, so speed and freeze
are a table the renderer reads, never state. `camera_at` evaluates the punch-in keys.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from montaj.render.core import EASE, key, keys_s

# Midpoint steps per frame when a speed curve is integrated.
_STEPS = 16
# `floor` of an exact product such as 1.5 * 8 must not land one frame low.
_EPS = 1e-6

# (t in frames, value, ease name)
SpeedKeys = Sequence[tuple[float, float, str]]
# (t in frames, zoom or None to hold, focus in source px or None to hold, ease name)
CameraKeys = Sequence[tuple[float, "float | None", "tuple[float, float] | None", str]]


def source_index(speed: float | SpeedKeys | None, freeze: float | None, n: int) -> tuple[int, ...] | None:
    """Source frame (film fps, from `clip_in`) for shot frames `0..n-1`; None means identity.

    The source position is the integral of the speed curve, floored to a frame.
    From `freeze` on it stops. Speed is never <= 0, so the table never runs backwards.
    """
    if speed is None and freeze is None:
        return None
    stop = n if freeze is None else min(n, max(0, int(freeze)))
    pos = [0.0] * n
    if isinstance(speed, (int, float)) or speed is None:
        rate = 1.0 if speed is None else float(speed)
        pos = [rate * min(f, stop) for f in range(n)]
    else:
        keys = [(float(t), float(v), EASE[e]) for t, v, e in speed]
        acc = 0.0
        for f in range(n):
            pos[f] = acc
            if f >= stop:
                continue
            for s in range(_STEPS):
                acc += float(key(f + (s + 0.5) / _STEPS, keys)) / _STEPS
    return tuple(int(math.floor(p + _EPS)) for p in pos)


def camera_at(keys: CameraKeys, crop: tuple[float, float, float, float], t: float) -> tuple[float, float, float]:
    """`(zoom, fx, fy)` at shot-clock `t`: zoom is a multiple of cover, `fx, fy` are fractions of `crop`.

    A key that leaves out `zoom` or `focus` holds the previous one. The first key's
    defaults are zoom 1 and the crop centre. Zoom moves geometrically between keys.
    """
    x0, y0, x1, y1 = crop
    zoom, focus = 1.0, ((x0 + x1) / 2, (y0 + y1) / 2)
    zk, fk = [], []
    for kt, kz, kf, ease in keys:
        zoom = zoom if kz is None else float(kz)
        focus = focus if kf is None else (float(kf[0]), float(kf[1]))
        fn = EASE[ease]
        zk.append((float(kt), zoom, fn))
        fk.append((float(kt), focus, fn))
    z = keys_s(t, zk)
    fx, fy = key(t, fk)
    return z, (fx - x0) / (x1 - x0), (fy - y0) / (y1 - y0)
