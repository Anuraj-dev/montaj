"""Colour spec (#43): `video.color`, `shots[].grade`, top-level `grades:`. Pixel math is `render/color.py`.

A named grade (`grade: warm-day`) is replaced by its `Grade` in `resolve_grades`, so the shot's segment
hash carries the values: editing one named grade re-renders only the shots that use it.
"""
from __future__ import annotations

import math
import re
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Discriminator, Field, Tag

GRADE_TAGS = ("grade_name", "grade_inline")  # union tags, not spec keys
GRADE_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
TEMP_MAX = 4000.0  # K either side of 6500
TINT_MAX = 100.0
EXPOSURE_MAX = 8.0  # stops


def _reject_bool(v: object) -> object:
    if isinstance(v, bool):
        raise ValueError("need a number, not a boolean")
    return v


Num = Annotated[float, BeforeValidator(_reject_bool)]
Wheel = Num | tuple[Num, Num, Num]  # one value for all channels, or [r, g, b]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ColorSpec(_Frozen):
    working: Literal["display", "linear"] = "display"


class Curves(_Frozen):
    # [[x, y], ...] on display-encoded luma, x strictly increasing, both 0..1.
    luma: list[tuple[Num, Num]] | None = None


class Grade(_Frozen):
    exposure: Num = 0.0
    temp: Num = 0.0
    tint: Num = 0.0
    contrast: Num = 1.0
    pivot: Num = 0.46
    lift: Wheel = 0.0
    gamma: Wheel = 1.0
    gain: Wheel = 1.0
    saturation: Num = 1.0
    vibrance: Num = 0.0
    curves: Curves | None = Field(default=None, exclude_if=lambda v: v is None)


def _grade_kind(v: object) -> str:
    return "grade_name" if isinstance(v, str) else "grade_inline"


GradeRef = Annotated[
    Annotated[str, Tag("grade_name")] | Annotated[Grade, Tag("grade_inline")],
    Discriminator(_grade_kind),
]


def _line(path: str, problem: str, hint: str) -> str:
    return f"{path}: {problem} ({hint})"


def _check_grade(g: Grade, path: str, errors: list[str]) -> None:
    def num(name: str, v, ok, hint: str) -> None:
        for c in v if isinstance(v, tuple) else (v,):
            if not math.isfinite(c) or not ok(c):
                errors.append(_line(f"{path}.{name}", f"{c!r} is out of range", hint))
                return

    num("exposure", g.exposure, lambda v: abs(v) <= EXPOSURE_MAX, f"stops, -{EXPOSURE_MAX:g}..{EXPOSURE_MAX:g}")
    num("temp", g.temp, lambda v: abs(v) <= TEMP_MAX, f"kelvin offset, -{TEMP_MAX:g}..{TEMP_MAX:g}; + is warmer")
    num("tint", g.tint, lambda v: abs(v) <= TINT_MAX, f"-{TINT_MAX:g}..{TINT_MAX:g}; + is magenta")
    num("contrast", g.contrast, lambda v: v > 0, "above 0; 1 is unchanged")
    num("pivot", g.pivot, lambda v: 0 < v < 1, "display value between 0 and 1; 0.46 is 18% grey")
    num("lift", g.lift, lambda v: -1 <= v <= 1, "-1..1; 0 is unchanged")
    num("gamma", g.gamma, lambda v: v > 0, "above 0; 1 is unchanged")
    num("gain", g.gain, lambda v: 0 <= v <= 8, "0..8; 1 is unchanged")
    num("saturation", g.saturation, lambda v: 0 <= v <= 4, "0..4; 1 is unchanged")
    num("vibrance", g.vibrance, lambda v: -1 <= v <= 1, "-1..1; 0 is unchanged")
    pts = g.curves.luma if g.curves is not None else None
    if pts is None:
        return
    where = f"{path}.curves.luma"
    if len(pts) < 2:
        errors.append(_line(where, "needs at least 2 points", "[[0, 0], [1, 1]] is the identity"))
        return
    for i, (x, y) in enumerate(pts):
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
            errors.append(_line(f"{where}[{i}]", f"[{x:g}, {y:g}] is outside 0..1", "points are [x, y] in 0..1"))
            return
        if i and x <= pts[i - 1][0]:
            errors.append(_line(f"{where}[{i}]", "x does not increase", "list points left to right, x strictly increasing"))
            return


def check_grades(spec, errors: list[str]) -> None:
    """Range checks for `grades:` and every `shots[].grade`; a named grade must exist."""
    named = spec.grades or {}
    for name, g in named.items():
        if not GRADE_NAME.match(name):
            errors.append(_line(f"grades.{name}", "bad name", "lowercase letters, digits, - and _, starting with a letter"))
        _check_grade(g, f"grades.{name}", errors)
    for i, sh in enumerate(spec.shots):
        g = sh.grade
        if g is None:
            continue
        path = f"shots[{i}].grade"
        if sh.blank:
            errors.append(_line(path, "a blank shot has no picture to grade", "remove grade"))
        elif isinstance(g, str):
            if g not in named:
                have = ", ".join(sorted(named)) or "none defined"
                errors.append(_line(path, f"unknown grade {g!r}", f"define it under grades:; have: {have}"))
        else:
            _check_grade(g, path, errors)


def resolve_grades(spec):
    """Replace each named `shots[].grade` with its `Grade`. Call after `check_grades` passed."""
    named = spec.grades or {}
    if not any(isinstance(sh.grade, str) for sh in spec.shots):
        return spec
    shots = [sh.model_copy(update={"grade": named[sh.grade]}) if isinstance(sh.grade, str) else sh
             for sh in spec.shots]
    return spec.model_copy(update={"shots": shots})
