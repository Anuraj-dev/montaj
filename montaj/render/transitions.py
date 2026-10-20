"""Screen-space transitions. Port of film2.swirl/whip/fade and film.shift.

``wall_pullback`` and ``burn_white`` are spec transitions. They composite the
two frames the timeline already resolved, so a pull-back or a burn is not a
stack of keyed layers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import clamp01, cubic, outc, smooth

# Magnification at the start of a wall pull-back. The camera eases to 1.
DEFAULT_PULLBACK_ZOOM = 3.2
# The outgoing card stays opaque until the ease has nearly settled, then dissolves
# so the last frame is the incoming picture and not a card sitting on it.
_CARD_HOLD = 0.92
# Extra exposure while burning to white. 0 would be a flat dip; this clips highlights first.
_BURN_HEAT = 4.0


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


def _scale_about_center(cv, img, k: float):
    """Zoom ``img`` about the frame centre. ``k > 1`` crops in; ``k < 1`` shrinks onto black."""
    if abs(k - 1.0) <= 1e-6:
        return img
    inv = 1.0 / k
    th = torch.tensor([[inv, 0.0, 0.0], [0.0, inv, 0.0]], device=img.device, dtype=img.dtype)[None]
    grid = F.affine_grid(th, (1, 3, cv.H, cv.W), align_corners=False)
    return F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]


def wall_pullback(cv, a, b, x: float, zoom: float = DEFAULT_PULLBACK_ZOOM):
    """Pull back from ``zoom`` onto ``b``. ``a`` shrinks as the centre card.

    ``x = 0`` is ``a``. ``x = 1`` is ``b``. The camera scale is ``zoom → 1`` on
    ``outc``. Pixels outside the shrinking card are ``b`` at that scale, so the
    incoming frame opens as a wall around the outgoing shot.
    """
    e = outc(x)
    if e <= 0.0:
        return a
    k = zoom + (1.0 - zoom) * e
    view = _scale_about_center(cv, b, k)
    if e >= 1.0:
        return view
    card = _scale_about_center(cv, a, k / zoom)
    alpha = 1.0 if e <= _CARD_HOLD else (1.0 - e) / (1.0 - _CARD_HOLD)
    nx = (cv.XX + 0.5) / cv.W * 2 - 1
    ny = (cv.YY + 0.5) / cv.H * 2 - 1
    inside = (nx.abs() <= (k / zoom)) & (ny.abs() <= (k / zoom))
    mask = inside.to(dtype=a.dtype) * alpha
    return view * (1.0 - mask) + card * mask


def _lift_to_white(img, t: float):
    """``t = 0`` leaves ``img``. ``t = 1`` is white. Gain clips highlights before the mix ends."""
    if t <= 0.0:
        return img
    if t >= 1.0:
        return torch.ones_like(img)
    gain = 1.0 + _BURN_HEAT * t
    clipped = torch.clamp(img * gain, 0.0, 1.0)
    return clipped * (1.0 - t) + t


def burn_white(a, b, x: float):
    """Outgoing shot burns to white; incoming shot emerges from it.

    The first half lifts ``a`` to white. The second half settles from white onto ``b``.
    ``x = 0`` is ``a``, ``x = 0.5`` is white, ``x = 1`` is ``b``.
    """
    x = clamp01(x)
    if x <= 0.5:
        return _lift_to_white(a, smooth(x * 2.0))
    return _lift_to_white(b, smooth((1.0 - x) * 2.0))
