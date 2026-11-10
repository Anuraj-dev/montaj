"""Closed-form easings for keyframes (#35): progress x in [0, 1] -> eased progress.

Unlike `render.core.EASE` these may leave [0, 1] (back, spring, elastic, bounce overshoot and settle), so
they never clamp their output; `x` is clamped on the way in. All are pure functions of x, so a frame's
value depends only on its time and the spec. No torch: `spec` validates easing names with this module.

Names: `back-in`, `back-out`, `bounce`, `elastic`, `spring` / `spring(stiffness, damping)`,
`cubic-bezier(x1, y1, x2, y2)`, `steps(n)`, plus the core set (`linear`, `smooth`, `cubic`, ...).
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable

BACK = 1.70158
SPRING_STIFFNESS = 100.0
SPRING_DAMPING = 10.0
NEW_EASES = ("back-in", "back-out", "bounce", "elastic", "spring", "spring(stiffness, damping)",
             "cubic-bezier(x1, y1, x2, y2)", "steps(n)")

_CALL = re.compile(r"^([a-z-]+)\(([^()]*)\)$")


def _c01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def back_in(x: float) -> float:
    x = _c01(x)
    return (BACK + 1) * x**3 - BACK * x**2


def back_out(x: float) -> float:
    x = _c01(x)
    return 1 + (BACK + 1) * (x - 1) ** 3 + BACK * (x - 1) ** 2


def bounce(x: float) -> float:
    """Penner ease-out bounce: three decaying bounces off 1."""
    x = _c01(x)
    n, d = 7.5625, 2.75
    if x < 1 / d:
        return n * x * x
    if x < 2 / d:
        x -= 1.5 / d
        return n * x * x + 0.75
    if x < 2.5 / d:
        x -= 2.25 / d
        return n * x * x + 0.9375
    x -= 2.625 / d
    return n * x * x + 0.984375


def elastic(x: float) -> float:
    """Ease-out elastic: rings around 1 and settles."""
    x = _c01(x)
    if x in (0.0, 1.0):
        return x
    return 2 ** (-10 * x) * math.sin((10 * x - 0.75) * 2 * math.pi / 3) + 1


def spring(stiffness: float = SPRING_STIFFNESS, damping: float = SPRING_DAMPING) -> Callable[[float], float]:
    """Unit-mass damped spring released from rest toward 1, over x seconds-of-the-key (x in 0..1).

    Closed form for under-, critically and over-damped motion. A linear term makes x = 1 land exactly on 1
    (the raw response is within a percent there), so the hold after the key does not jump.
    """
    w = math.sqrt(stiffness)
    zeta = damping / (2 * w)

    def raw(x: float) -> float:
        if zeta < 1.0:
            wd = w * math.sqrt(1 - zeta * zeta)
            return 1 - math.exp(-zeta * w * x) * (math.cos(wd * x) + zeta * w / wd * math.sin(wd * x))
        if zeta == 1.0:
            return 1 - math.exp(-w * x) * (1 + w * x)
        r = w * math.sqrt(zeta * zeta - 1)
        a, b = zeta * w - r, zeta * w + r
        return 1 - (b * math.exp(-a * x) - a * math.exp(-b * x)) / (b - a)

    miss = 1.0 - raw(1.0)
    return lambda x: raw(_c01(x)) + miss * _c01(x)


def cubic_bezier(x1: float, y1: float, x2: float, y2: float) -> Callable[[float], float]:
    """CSS `cubic-bezier`: P0 = (0, 0), P3 = (1, 1); solve the curve's x for its parameter, return y."""
    def bx(s: float) -> float:
        return 3 * (1 - s) ** 2 * s * x1 + 3 * (1 - s) * s * s * x2 + s**3

    def by(s: float) -> float:
        return 3 * (1 - s) ** 2 * s * y1 + 3 * (1 - s) * s * s * y2 + s**3

    def fn(x: float) -> float:
        x = _c01(x)
        lo, hi = 0.0, 1.0
        for _ in range(60):  # x(s) is monotonic for x1, x2 in [0, 1]
            mid = (lo + hi) / 2
            if bx(mid) < x:
                lo = mid
            else:
                hi = mid
        return by((lo + hi) / 2)

    return fn


def steps(n: int) -> Callable[[float], float]:
    return lambda x: min(1.0, math.floor(_c01(x) * n + 1e-9) / n)


def _nums(args: str, count: int | None) -> list[float] | None:
    try:
        vals = [float(a) for a in args.split(",")]
    except ValueError:
        return None
    if (count is not None and len(vals) != count) or not all(math.isfinite(v) for v in vals):
        return None
    return vals


def parse_ease(name: str) -> Callable[[float], float] | None:
    """The easing named `name`, or None when it is not one (or its arguments are out of range)."""
    from montaj.render.core import EASE as CORE_EASE  # torch: imported only when a render or validate needs it

    if name in CORE_EASE:
        return CORE_EASE[name]
    simple = {"back-in": back_in, "back-out": back_out, "bounce": bounce, "elastic": elastic}
    if name in simple:
        return simple[name]
    if name == "spring":
        return spring()
    m = _CALL.match(name.replace(" ", ""))
    if m is None:
        return None
    head, args = m.groups()
    if head == "spring":
        v = _nums(args, 2)
        return spring(*v) if v and v[0] > 0 and v[1] >= 0 else None
    if head == "cubic-bezier":
        v = _nums(args, 4)
        return cubic_bezier(*v) if v and 0 <= v[0] <= 1 and 0 <= v[2] <= 1 else None
    if head == "steps":
        v = _nums(args, 1)
        return steps(int(v[0])) if v and v[0] >= 1 and v[0] == int(v[0]) else None
    return None
