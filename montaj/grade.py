"""Colour spec (#43, #44): `video.color`, `shots[].grade`, top-level `grades:`. Pixel math is `render/color.py`.

A named grade (`grade: warm-day`) is replaced by its `Grade` in `resolve_grades`, so the shot's segment
hash carries the values: editing one named grade re-renders only the shots that use it.

#44 adds, all optional and absent from the dump when unset: `secondaries` (an HSL key, a window or the
subject matte, each with its own primary `adjust`), `protect: skin` and `look` (film stock, halation,
bloom, grain, gate weave, split tone). `center: face` is resolved from `analysis.json` face boxes in
`resolve_faces`; without them the window follows the subject matte's centroid.
"""
from __future__ import annotations

import math
import re
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Discriminator, Field, StrictBool, Tag

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


class Adjust(_Frozen):
    """The primary operators. A secondary's `adjust` is one; `Grade` adds the #44 fields."""

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


def _none(v: object) -> bool:
    return v is None


Pair = tuple[Num, Num]
Center = Literal["face", "subject"] | Pair


class Key(_Frozen):
    """HSL qualifier on display-encoded values: hue in degrees (lo > hi wraps through red), sat and luma 0..1."""

    hue: Pair | None = Field(default=None, exclude_if=_none)
    sat: Pair | None = Field(default=None, exclude_if=_none)
    luma: Pair | None = Field(default=None, exclude_if=_none)
    soft: Num = 0.05


class Ellipse(_Frozen):
    center: Center
    radius: Num | Pair
    feather: Num = 0.0
    offset: Pair | None = Field(default=None, exclude_if=_none)
    # Set by `resolve_faces` from analysis.json (source px); the renderer maps it through the clip's camera.
    face_point: Pair | None = Field(default=None, exclude_if=_none)


class Gradient(_Frozen):
    from_: Pair = Field(alias="from")
    to: Pair


class Window(_Frozen):
    ellipse: Ellipse | None = Field(default=None, exclude_if=_none)
    gradient: Gradient | None = Field(default=None, exclude_if=_none)


class Secondary(_Frozen):
    key: Key | None = Field(default=None, exclude_if=_none)
    window: Window | None = Field(default=None, exclude_if=_none)
    matte: str | None = Field(default=None, exclude_if=_none)
    invert: StrictBool = False
    adjust: Adjust


class Grain(_Frozen):
    amount: Num
    size: Num = 1.0


class Film(_Frozen):
    stock: str | None = Field(default=None, exclude_if=_none)
    halation: Num = 0.0
    bloom: Num = 0.0
    grain: Grain | None = Field(default=None, exclude_if=_none)
    gate_weave: Num = 0.0


class SplitTone(_Frozen):
    shadows: str
    highlights: str
    balance: Num = 0.0
    amount: Num = 0.3


class GradeLook(_Frozen):
    film: Film | None = Field(default=None, exclude_if=_none)
    split_tone: SplitTone | None = Field(default=None, exclude_if=_none)


class Grade(Adjust):
    secondaries: list[Secondary] | None = Field(default=None, exclude_if=_none)
    protect: Literal["skin"] | None = Field(default=None, exclude_if=_none)
    look: GradeLook | None = Field(default=None, exclude_if=_none)


# Film stocks: a primary adjust on top of the shot's grade. Curves give the toe and shoulder.
# Stock curves start at (0, 0): `curves.luma` scales each pixel by y'/y, so a lifted black point would
# multiply near-black chroma noise into colour blotches. The faded black comes from additive `lift`.
_FILM_CURVE = ((0.0, 0.0), (0.25, 0.22), (0.75, 0.79), (1.0, 0.97))
STOCKS: dict[str, dict] = {
    "warm-negative": {"temp": 300.0, "tint": 4.0, "contrast": 1.05, "lift": 0.025, "saturation": 0.9,
                      "curves": {"luma": _FILM_CURVE}},
    "cool-negative": {"temp": -250.0, "tint": -4.0, "contrast": 1.05, "lift": 0.025, "saturation": 0.88,
                      "curves": {"luma": _FILM_CURVE}},
    "print": {"temp": 120.0, "contrast": 1.15, "saturation": 1.05, "lift": (0.01, 0.018, 0.025),
              "curves": {"luma": ((0.0, 0.0), (0.25, 0.2), (0.75, 0.82), (1.0, 0.98))}},
    "bleach-bypass": {"contrast": 1.25, "saturation": 0.55, "curves": {"luma": ((0.0, 0.0), (0.25, 0.2),
                                                                               (0.75, 0.83), (1.0, 1.0))}},
}
# `protect: skin`: the skin-tone qualifier whose hue the look may not move.
SKIN_KEY = Key(hue=(10.0, 50.0), sat=(0.12, 0.75), luma=(0.12, 0.95), soft=0.05)


def extras(g) -> bool:
    """True when a grade has any #44 field; plain primary grades keep the #43 path bit for bit."""
    return bool(getattr(g, "secondaries", None)) or getattr(g, "protect", None) is not None \
        or getattr(g, "look", None) is not None


def hex_rgb(raw: str) -> tuple[float, float, float] | None:
    """`#rrggbb` -> display 0..1, or None."""
    text = str(raw).strip()
    if len(text) != 7 or text[0] != "#":
        return None
    try:
        n = int(text[1:], 16)
    except ValueError:
        return None
    return ((n >> 16) & 255) / 255.0, ((n >> 8) & 255) / 255.0, (n & 255) / 255.0


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


def _check_extras(g: Grade, path: str, errors: list[str]) -> None:
    """Ranges of the #44 fields. Shot context (does it have a matte?) is `_check_matte_use`."""
    def rng(where: str, v, lo: float, hi: float, hint: str) -> bool:
        if not (math.isfinite(v) and lo <= v <= hi):
            errors.append(_line(where, f"{v!r} is out of range", hint))
            return False
        return True

    for j, sec in enumerate(g.secondaries or ()):
        at = f"{path}.secondaries[{j}]"
        if sec.key is None and sec.window is None and sec.matte is None:
            errors.append(_line(at, "selects nothing", "add key, window or matte: subject"))
        if sec.matte is not None and sec.matte != "subject":
            errors.append(_line(f"{at}.matte", f"unknown {sec.matte!r}", "use subject"))
        k = sec.key
        if k is not None:
            if k.hue is None and k.sat is None and k.luma is None:
                errors.append(_line(f"{at}.key", "has no range", "add hue, sat or luma: [lo, hi]"))
            if k.hue is not None:
                for v in k.hue:
                    rng(f"{at}.key.hue", v, 0, 360, "degrees 0..360; [340, 20] wraps through red")
            for name in ("sat", "luma"):
                pair = getattr(k, name)
                if pair is not None and all(rng(f"{at}.key.{name}", v, 0, 1, "0..1") for v in pair) and pair[0] > pair[1]:
                    errors.append(_line(f"{at}.key.{name}", "low end is above high end", "write [lo, hi]"))
            rng(f"{at}.key.soft", k.soft, 0, 1, "edge width as a fraction of each range's scale, 0..1")
        w = sec.window
        if w is not None:
            if (w.ellipse is None) == (w.gradient is None):
                errors.append(_line(f"{at}.window", "needs exactly one of ellipse or gradient", "pick one per secondary"))
            e = w.ellipse
            if e is not None:
                for v in e.radius if isinstance(e.radius, tuple) else (e.radius,):
                    if not (math.isfinite(v) and v > 0):
                        errors.append(_line(f"{at}.window.ellipse.radius", f"{v!r} must be above 0", "design px"))
                        break
                rng(f"{at}.window.ellipse.feather", e.feather, 0, 10000, "design px of soft edge, 0 or more")
            gr = w.gradient
            if gr is not None and tuple(gr.from_) == tuple(gr.to):
                errors.append(_line(f"{at}.window.gradient", "from and to are the same point", "design px [x, y], apart"))
        _check_grade(sec.adjust, f"{at}.adjust", errors)
    look = g.look
    if look is None:
        return
    film = look.film
    if film is not None:
        if film.stock is not None and film.stock not in STOCKS:
            errors.append(_line(f"{path}.look.film.stock", f"unknown {film.stock!r}", f"use {', '.join(STOCKS)}"))
        for name in ("halation", "bloom", "gate_weave"):
            rng(f"{path}.look.film.{name}", getattr(film, name), 0, 1, "0..1; 0 is off")
        if film.grain is not None:
            rng(f"{path}.look.film.grain.amount", film.grain.amount, 0, 1, "0..1; 0 is off")
            rng(f"{path}.look.film.grain.size", film.grain.size, 0.5, 4, "0.5..4; grain cell in design px")
    st = look.split_tone
    if st is not None:
        for name in ("shadows", "highlights"):
            if hex_rgb(getattr(st, name)) is None:
                errors.append(_line(f"{path}.look.split_tone.{name}", f"{getattr(st, name)!r} is not a colour", '"#rrggbb"'))
        rng(f"{path}.look.split_tone.balance", st.balance, -1, 1, "-1..1; + gives more of the range to highlights")
        rng(f"{path}.look.split_tone.amount", st.amount, 0, 1, "0..1; how far toward the tone colours")


def _uses_subject(g: Grade) -> bool:
    for sec in g.secondaries or ():
        e = sec.window.ellipse if sec.window is not None else None
        if sec.matte is not None or (e is not None and e.center in ("face", "subject")):
            return True
    return False


def _check_matte_use(sh, g: Grade, path: str, errors: list[str]) -> None:
    """`matte: subject` and `center: face|subject` read the shot's subject matte."""
    if not _uses_subject(g):
        return
    if sh.matte is None and not any(getattr(layer, "mask", None) for layer in sh.layers or ()):
        errors.append(_line(f"{path}.secondaries", "matte and center: face|subject need the shot's subject matte",
                            "add matte: subject to the clip shot, or mask: subject to a layer"))


def check_grades(spec, errors: list[str]) -> None:
    """Range checks for `grades:` and every `shots[].grade`; a named grade must exist."""
    named = spec.grades or {}
    for name, g in named.items():
        if not GRADE_NAME.match(name):
            errors.append(_line(f"grades.{name}", "bad name", "lowercase letters, digits, - and _, starting with a letter"))
        _check_grade(g, f"grades.{name}", errors)
        _check_extras(g, f"grades.{name}", errors)
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
                _check_matte_use(sh, named[g], path, errors)
        else:
            _check_grade(g, path, errors)
            _check_extras(g, path, errors)
            _check_matte_use(sh, g, path, errors)


def resolve_grades(spec):
    """Replace each named `shots[].grade` with its `Grade`. Call after `check_grades` passed."""
    named = spec.grades or {}
    if not any(isinstance(sh.grade, str) for sh in spec.shots):
        return spec
    shots = [sh.model_copy(update={"grade": named[sh.grade]}) if isinstance(sh.grade, str) else sh
             for sh in spec.shots]
    return spec.model_copy(update={"shots": shots})


def _face_points(spec, spec_dir, i: int, sh):
    """Source-px centre of the largest face ingest found where this clip shot starts, or None."""
    if sh.clip is None or spec_dir is None:
        return None
    from pathlib import Path

    from montaj.analysis import AnalysisBook, face_point
    from montaj.project import clip_is_file, find_clip
    from montaj.spec import _clip_start_s

    if clip_is_file(sh.clip):
        return None
    assets = Path(spec_dir) / spec.assets
    try:
        file = find_clip(assets, sh.clip)
    except FileNotFoundError:
        return None
    row, _why = AnalysisBook(assets).row(file)
    if row is None:
        return None
    t = _clip_start_s(spec, sh.range[0] if sh.range is not None else sh.clip_in)
    scenes = row.get("scenes") or []
    scene = next((s for s in scenes if s["s"] <= t < s["e"]), scenes[-1] if scenes else {})
    faces = scene.get("faces") or []
    if not faces:
        return None
    x, y = face_point(faces[0])
    return (float(x), float(y))


def _face_windows(g) -> list[tuple[int, Ellipse]]:
    out = []
    for j, sec in enumerate(g.secondaries or ()):
        e = sec.window.ellipse if sec.window is not None else None
        if e is not None and e.center == "face":
            out.append((j, e))
    return out


def resolve_faces(spec, spec_dir):
    """Write ingest's face point onto each `center: face` window of a clip shot (after `resolve_grades`).

    The point is part of the shot's grade, so the segment hash sees it. Without one the window follows the
    subject matte's centroid (`grade_warnings` says so).
    """
    shots = list(spec.shots)
    changed = False
    for i, sh in enumerate(shots):
        g = sh.grade
        if not isinstance(g, Grade) or not _face_windows(g):
            continue
        pt = _face_points(spec, spec_dir, i, sh)
        if pt is None:
            continue
        secs = list(g.secondaries)
        for j, e in _face_windows(g):
            win = secs[j].window.model_copy(update={"ellipse": e.model_copy(update={"face_point": pt})})
            secs[j] = secs[j].model_copy(update={"window": win})
        shots[i] = sh.model_copy(update={"grade": g.model_copy(update={"secondaries": secs})})
        changed = True
    return spec.model_copy(update={"shots": shots}) if changed else spec


def grade_warnings(spec, spec_dir) -> list[str]:
    """`WARN` for each `center: face` that falls back to the subject matte's centroid."""
    named = spec.grades or {}
    out = []
    for i, sh in enumerate(spec.shots):
        g = named.get(sh.grade) if isinstance(sh.grade, str) else sh.grade
        if not isinstance(g, Grade) or not _face_windows(g):
            continue
        if _face_points(spec, spec_dir, i, sh) is None:
            src = sh.clip if sh.clip is not None else "this shot"
            out.append(f"WARN shots[{i}].grade: center: face has no face boxes for {src} in analysis.json; "
                       f"the window follows the subject matte's centroid (run montaj ingest for faces)")
    return out
