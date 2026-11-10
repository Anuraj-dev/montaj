"""Per-unit text motion for gpu-engine lines: entries, exits, emphasis shake and idle loops.

`timeline._resolve_texts` materialises a `Motion` per resolved line (plain values, frames), so the
segment hash sees every number the renderer uses. `unit_poses` turns a motion and a time into one
`Pose` per reveal unit, in closed form: a frame is a pure function of `t`, with no simulation state.
Shake noise is hashed from a seed, so renders repeat exactly. Distances are design px.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from montaj.render.core import clamp01, outc, sine

IN_TYPES = ("pop", "slam", "slide", "drop", "rise", "type", "flip", "scale-in")
OUT_TYPES = ("cut", "fade", "pop-out", "slide")
IDLE_TYPES = ("float", "pulse")
SIDES = ("left", "right", "top", "bottom")
_DIR = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, -1.0), "bottom": (0.0, 1.0)}

# Defaults in seconds (entry/exit length) and design px, per type.
IN_DUR = {"pop": 0.2, "slam": 0.2, "slide": 0.3, "drop": 0.5, "rise": 0.9, "type": 0.0, "flip": 0.3,
          "scale-in": 0.25}
IN_OVERSHOOT = {"pop": 1.08, "slam": 1.06, "slide": 1.04, "flip": 1.0, "scale-in": 1.0}
IN_DISTANCE = {"slide": 160.0, "drop": 220.0}
OUT_DUR = {"cut": 0.0, "fade": 0.2, "pop-out": 0.2, "slide": 0.25}
OUT_DISTANCE = 160.0
IDLE_AMOUNT = {"float": 4.0, "pulse": 4.0}
IDLE_PERIOD = 1.5  # s
SHAKE_DUR = 0.25  # s after the unit lands
POP_FROM = 0.8  # pop starts at this scale
SLAM_FROM = 1.8  # slam starts at this scale
POP_OUT_PEAK = 1.08  # pop-out swells to this before shrinking away
SLAM_SETTLE = 2.0  # frames
PEAK_AT = 0.6  # fraction of an overshooting entry spent reaching the peak


@dataclass(frozen=True)
class Motion:
    """A resolved line's motion. Times are frames; distances design px; `from_`/`out_to` are sides.

    The unit clock is the line's own: unit i enters at `ResolvedLine.at + i·stagger` and starts its
    exit at `out_at + i·out_stagger`. `out_at` is per piece, so a wrapped line's last unit ends its
    exit exactly at the block's `to`. `emph_kind` (None = `kind`) and `shake` apply to units holding
    `*emphasis*` glyphs.
    """

    kind: str
    dur: float
    overshoot: float = 1.0
    distance: float = 0.0
    from_: str | None = None
    emph_kind: str | None = None
    emph_dur: float = 0.0
    emph_overshoot: float = 1.0
    emph_distance: float = 0.0
    shake: float = 0.0
    shake_dur: float = 0.0
    out_kind: str | None = None
    out_at: float = 0.0
    out_dur: float = 0.0
    out_stagger: float = 0.0
    out_to: str | None = None
    out_distance: float = 0.0
    idle_kind: str | None = None
    idle_amount: float = 0.0
    idle_period: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class Pose:
    """One unit's change from its laid-out pose, about the unit's ink centre. `sy` squashes vertically."""

    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    sy: float = 1.0
    rotate: float = 0.0
    opacity: float = 1.0
    blur: float = 0.0


REST = Pose()
HIDDEN = Pose(opacity=0.0)


def _outq(x: float) -> float:
    x = clamp01(x)
    return 1 - (1 - x) ** 2


def _inq(x: float) -> float:
    x = clamp01(x)
    return x * x


def overshoot(p: float, start: float, peak: float) -> float:
    """`start` → `peak` (ease-out) over the first PEAK_AT of p, then → 1 (sine). No peak: `start` → 1 (outc)."""
    if peak <= 1.0 + 1e-9:
        return start + (1.0 - start) * outc(p)
    if p < PEAK_AT:
        return start + (peak - start) * _outq(p / PEAK_AT)
    return peak + (1.0 - peak) * sine((p - PEAK_AT) / (1.0 - PEAK_AT))


def bounce(p: float) -> float:
    """Penner's ease-out bounce: 0 → 1 with three decaying bounces off 1."""
    p = clamp01(p)
    n, d = 7.5625, 2.75
    if p < 1 / d:
        return n * p * p
    if p < 2 / d:
        p -= 1.5 / d
        return n * p * p + 0.75
    if p < 2.5 / d:
        p -= 2.25 / d
        return n * p * p + 0.9375
    p -= 2.625 / d
    return n * p * p + 0.984375


def settle_time(kind: str, dur: float) -> float:
    """Frames from a unit's start until its entry is at rest."""
    return dur + (SLAM_SETTLE if kind == "slam" else 0.0)


def entry(kind: str, u: float, dur: float, overshoot_: float, distance: float, side: str | None) -> Pose:
    """Pose `u` frames after the unit starts. Before 0 the unit is hidden; after the settle it is at rest."""
    if u < 0:
        return HIDDEN
    if kind == "type":
        return REST
    if u >= settle_time(kind, dur) or dur <= 0:
        return REST
    p = u / dur
    if kind == "rise":
        # Exactly the legacy gpu reveal (text._paint_glyphs): σ = (1−k)·12 design px, rise, opacity k.
        k = outc(p)
        if k >= 1 - 1e-6:
            return REST
        return Pose(dy=(1 - k) * distance, opacity=k, blur=(1 - k) * 24.0)
    if kind == "pop":
        return Pose(scale=overshoot(p, POP_FROM, overshoot_), opacity=min(1.0, p / 0.35))
    if kind == "slam":
        if u < dur:  # accelerate down onto the line
            return Pose(scale=SLAM_FROM + (1.0 - SLAM_FROM) * _inq(p), opacity=min(1.0, p / 0.25))
        q = (u - dur) / SLAM_SETTLE  # squash past rest and back
        return Pose(scale=1.0 - (overshoot_ - 1.0) * math.sin(math.pi * q))
    if kind == "slide":
        ox, oy = _DIR[side or "left"]
        rest = 1.0 - overshoot(p, 0.0, overshoot_)
        return Pose(dx=ox * distance * rest, dy=oy * distance * rest, opacity=min(1.0, p / 0.3))
    if kind == "drop":
        return Pose(dy=-distance * (1.0 - bounce(p)), opacity=min(1.0, p / 0.2))
    if kind == "flip":
        return Pose(sy=overshoot(p, 0.0, overshoot_), opacity=min(1.0, p / 0.3))
    if kind == "scale-in":
        return Pose(scale=overshoot(p, 0.0, overshoot_), opacity=min(1.0, p / 0.3))
    raise ValueError(f"unknown entry {kind!r}")


def exit_(kind: str, u: float, dur: float, overshoot_: float, distance: float, side: str | None) -> Pose:
    """Pose `u` frames after the unit starts leaving. Every exit is hidden from `dur` on."""
    if u < 0:
        return REST
    if u >= dur:
        return HIDDEN
    q = u / dur
    if kind == "fade":
        return Pose(opacity=1.0 - sine(q))
    if kind == "pop-out":
        peak = max(overshoot_, 1.0)
        s = 1.0 + (peak - 1.0) * _outq(q / 0.3) if q < 0.3 else peak * (1.0 - _inq((q - 0.3) / 0.7))
        return Pose(scale=s, opacity=1.0 - clamp01((q - 0.5) / 0.5))
    if kind == "slide":
        ox, oy = _DIR[side or "right"]
        e = _inq(q)
        return Pose(dx=ox * distance * e, dy=oy * distance * e, opacity=1.0 - clamp01((q - 0.4) / 0.6))
    raise ValueError(f"unknown exit {kind!r}")  # `cut` has dur 0 and never gets here


def _hash01(seed: int, unit: int, step: int, axis: int) -> float:
    """Deterministic value in [-1, 1] (splitmix64 of the four integers)."""
    x = (seed * 0x9E3779B97F4A7C15 + unit * 0xBF58476D1CE4E5B9 + step * 0x94D049BB133111EB + axis * 0x2545F4914F6CDD1D)
    x &= 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return (x / 0xFFFFFFFFFFFFFFFF) * 2.0 - 1.0


def noise(seed: int, unit: int, s: float, axis: int) -> float:
    """Value noise with one lattice point per frame, smoothly interpolated: continuous in sub-frame time."""
    i = math.floor(s)
    f = s - i
    w = f * f * (3 - 2 * f)
    a, b = _hash01(seed, unit, i, axis), _hash01(seed, unit, i + 1, axis)
    return a + (b - a) * w


def _compose(a: Pose, b: Pose) -> Pose:
    return Pose(dx=a.dx + b.dx, dy=a.dy + b.dy, scale=a.scale * b.scale, sy=a.sy * b.sy,
                rotate=a.rotate + b.rotate, opacity=a.opacity * b.opacity, blur=a.blur + b.blur)


def unit_pose(m: Motion, unit: int, emph: bool, t: float, at: float, stagger: float) -> Pose:
    """Pose of reveal unit `unit` (of this piece) at frame `t`; `at`/`stagger` are the line's reveal clock."""
    u = t - (at + unit * stagger)
    if emph and m.emph_kind is not None:
        kind, dur, ov, distance = m.emph_kind, m.emph_dur, m.emph_overshoot, m.emph_distance
    else:
        kind, dur, ov, distance = m.kind, m.dur, m.overshoot, m.distance
    pose = entry(kind, u, dur, ov, distance, m.from_)
    if pose.opacity <= 0.0:
        return pose
    land = settle_time(kind, dur)
    if m.idle_kind is not None and u > land and m.idle_period > 0:
        w = math.sin(2 * math.pi * (u - land) / m.idle_period)
        if m.idle_kind == "float":
            pose = _compose(pose, Pose(dy=-m.idle_amount * w))
        else:
            pose = _compose(pose, Pose(scale=1.0 + m.idle_amount / 100.0 * w))
    if emph and m.shake > 0 and m.shake_dur > 0:
        s = u - dur  # the hit: the unit reaches its rest size
        if 0 <= s < m.shake_dur:
            amp = m.shake * (1.0 - s / m.shake_dur)
            pose = _compose(pose, Pose(dx=amp * noise(m.seed, unit, s, 0), dy=amp * noise(m.seed, unit, s, 1),
                                       rotate=0.4 * amp * noise(m.seed, unit, s, 2)))
    if m.out_kind is not None:
        v = t - (m.out_at + unit * m.out_stagger)
        if v >= 0:
            pose = _compose(pose, exit_(m.out_kind, v, m.out_dur, POP_OUT_PEAK, m.out_distance, m.out_to))
    return pose


def unit_poses(m: Motion, n: int, emph_units: frozenset[int], t: float, at: float, stagger: float) -> list[Pose]:
    return [unit_pose(m, i, i in emph_units, t, at, stagger) for i in range(n)]
