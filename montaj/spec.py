"""YAML spec schema: pydantic models, duration parsing, SpecError."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, WrapSerializer

EASE = ("linear", "smooth", "cubic", "outc", "inc", "expo", "ramp")
TRANS_TYPES = ("cut", "fade", "swirl", "whip")
_DUR = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([fbs])$")
_BARE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_WH = re.compile(r"^(\d+)x(\d+)$")


def _photo_id(v: object) -> object:
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    return v


PhotoId = Annotated[str, BeforeValidator(_photo_id)]


def _int_px(v: object) -> object:
    """Crops are source-pixel slices. Integral floats (YAML `60.0`) become ints."""
    if isinstance(v, bool):
        raise ValueError(f"{v!r} is not a whole pixel")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if math.isfinite(v) and v == int(v):
            return int(v)
        raise ValueError(f"{v!r} is not a whole pixel")
    return v


def _dump_int_px(v: object, handler):
    """Assignment can park an integral float on the field; dump it as an int."""
    if isinstance(v, float) and math.isfinite(v) and v == int(v):
        v = int(v)
    return handler(v)


IntPx = Annotated[int, BeforeValidator(_int_px), WrapSerializer(_dump_int_px)]
CropBox = tuple[IntPx, IntPx, IntPx, IntPx]


def _reject_bare_duration(v: object) -> object:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        raise ValueError("bare number")
    return v


DurationStr = Annotated[str, BeforeValidator(_reject_bare_duration)]


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SpecError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def _line(path: str, problem: str, hint: str) -> str:
    return f"{path}: {problem} ({hint})"


class IntroOutro(Frozen):
    type: str
    dur: DurationStr


class Video(Frozen):
    size: str
    fps: int
    bpm: float | None = None
    look: str
    intro: IntroOutro | None = None
    outro: IntroOutro | None = None
    motion_blur: int = 1


class Transition(Frozen):
    type: str
    dur: DurationStr | None = None
    center: tuple[float, float] | None = None
    axis: str = "x"


class WallPrint(Frozen):
    photo: PhotoId
    cell: tuple[int, int]
    straight: bool = False
    scale: float | None = None
    crop: CropBox | None = None


class WallLight(Frozen):
    center: tuple[float, float]
    radius: tuple[float, float]


class PosKey(Frozen):
    t: DurationStr | int
    print: PhotoId | None = None
    at: tuple[float, float] | None = None
    ease: str = "smooth"


class ZoomKey(Frozen):
    t: DurationStr | int
    value: float | None = None
    print: PhotoId | None = None
    ease: str = "smooth"


class WallCamera(Frozen):
    pos: list[PosKey]
    zoom: list[ZoomKey]


class WallSpec(Frozen):
    grid: str
    seed: int
    jitter: float
    rot: float
    print_height: float
    prints: list[WallPrint]
    light: WallLight
    camera: WallCamera


class Shot(Frozen):
    photo: PhotoId | None = None
    wall: WallSpec | None = None
    hold: DurationStr
    crop: CropBox | None = None
    focus: tuple[float, float] | None = None
    zoom: float | None = None
    in_: Transition | None = Field(default=None, alias="in")
    flash: float | None = None


class Fx(Frozen):
    leak: float | None = None
    flash: float | None = None
    at: DurationStr
    dir: int | None = None


class Spec(Frozen):
    video: Video
    assets: str = "assets"
    shots: list[Shot]
    fx: list[Fx] = []


def _yaml_path(loc: tuple) -> str:
    parts: list[str] = []
    for x in loc:
        if x == "in_":
            x = "in"
        if isinstance(x, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{x}]"
            else:
                parts.append(f"[{x}]")
        else:
            parts.append(str(x))
    return ".".join(parts) or "."


_PYDANTIC_HINTS = {
    "missing": "required",
    "extra_forbidden": "remove this field",
    "string_type": "need a string",
    "int_type": "need an integer",
    "int_parsing": "need an integer",
    "float_type": "need a number",
    "bool_type": "need a boolean",
    "list_type": "need a list",
    "tuple_type": "need a list",
    "too_short": "wrong length",
    "too_long": "wrong length",
    "model_type": "need a mapping",
    "dict_type": "need a mapping",
}


def _from_pydantic(exc: ValidationError) -> list[str]:
    lines = []
    for err in exc.errors():
        path = _yaml_path(tuple(err["loc"]))
        msg = err["msg"]
        if msg.lower().startswith("value error, "):
            msg = msg[13:]
        hint = _PYDANTIC_HINTS.get(err["type"], "check the spec")
        if "bare number" in msg:
            msg = "bare number"
            hint = "use 22f, 4b or 1.5s"
        elif "whole pixel" in msg:
            hint = "use integer source-px"
        lines.append(_line(path, msg, hint))
    return lines


def _fmt_num(v: float) -> str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "-inf" if v < 0 else "inf"
    return repr(v)


def round_frame(x: float) -> int:
    if not math.isfinite(x):
        raise ValueError("non-finite frame count")
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def parse_dur(raw: str, fps: int, bpm: float | None, path: str, errors: list[str]) -> int | None:
    if not isinstance(raw, str) or not _DUR.match(raw.strip()):
        text = raw if isinstance(raw, str) else repr(raw)
        if isinstance(raw, (int, float)) or (isinstance(raw, str) and _BARE.match(raw.strip())):
            errors.append(_line(path, "bare number", "use 22f, 4b or 1.5s"))
        else:
            errors.append(_line(path, f"invalid duration {text!r}", "use 22f, 4b or 1.5s"))
        return None
    n_s, unit = _DUR.match(raw.strip()).groups()
    n = float(n_s)
    if not math.isfinite(n) or n < 0:
        errors.append(_line(path, "negative duration", "use a positive 22f, 4b or 1.5s"))
        return None
    if unit == "f":
        frames = n
    elif fps <= 0:
        return None
    elif unit == "s":
        frames = n * fps
    elif bpm is None or not math.isfinite(bpm) or bpm <= 0:
        return None
    else:
        frames = n * 60.0 / bpm * fps
    if not math.isfinite(frames):
        errors.append(_line(path, "duration is not finite", "use a smaller duration"))
        return None
    return round_frame(frames)


def parse_wh(raw: str, path: str, errors: list[str], *, even: bool = False) -> tuple[int, int] | None:
    if not isinstance(raw, str) or not _WH.match(raw.strip()):
        errors.append(_line(path, f"expected WxH, got {raw!r}", "use 1080x1920"))
        return None
    w, h = (int(x) for x in _WH.match(raw.strip()).groups())
    if w <= 0 or h <= 0:
        errors.append(_line(path, f"sides must be positive, got {w}x{h}", "use positive even sizes"))
        return None
    if even and (w % 2 or h % 2):
        errors.append(_line(path, f"both sides must be even, got {w}x{h}", "use even WxH"))
        return None
    return w, h


def _duration_fields(spec: Spec) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if spec.video.intro:
        out.append(("video.intro.dur", spec.video.intro.dur))
    if spec.video.outro:
        out.append(("video.outro.dur", spec.video.outro.dur))
    for i, sh in enumerate(spec.shots):
        out.append((f"shots[{i}].hold", sh.hold))
        if sh.in_ is not None and sh.in_.dur is not None:
            out.append((f"shots[{i}].in.dur", sh.in_.dur))
        if sh.wall is not None:
            for k, key in enumerate(sh.wall.camera.pos):
                out.append((f"shots[{i}].wall.camera.pos[{k}].t", key.t))
            for k, key in enumerate(sh.wall.camera.zoom):
                out.append((f"shots[{i}].wall.camera.zoom[{k}].t", key.t))
    for i, fx in enumerate(spec.fx):
        out.append((f"fx[{i}].at", fx.at))
    return out


def _collect_photos(data: object, assets_dir: Path, errors: list[str]) -> None:
    from montaj.project import find_photo

    if not isinstance(data, dict):
        return
    if not assets_dir.is_dir():
        rel = data.get("assets", "assets")
        errors.append(_line("assets", f"{rel!r} not a directory", "create it or fix the path"))
        return
    shots = data.get("shots")
    if not isinstance(shots, list):
        return
    for i, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue
        if "photo" in shot and shot["photo"] is not None:
            pid = _photo_id(shot["photo"])
            if isinstance(pid, str):
                _require_photo(assets_dir, pid, f"shots[{i}].photo", errors, find_photo)
        wall = shot.get("wall")
        if isinstance(wall, dict):
            prints = wall.get("prints")
            if isinstance(prints, list):
                for j, pr in enumerate(prints):
                    if isinstance(pr, dict) and pr.get("photo") is not None:
                        pid = _photo_id(pr["photo"])
                        if isinstance(pid, str):
                            _require_photo(
                                assets_dir, pid, f"shots[{i}].wall.prints[{j}].photo", errors, find_photo
                            )


def _require_photo(assets_dir: Path, pid: str, path: str, errors: list[str], find_photo) -> None:
    try:
        find_photo(assets_dir, pid)
    except FileNotFoundError as e:
        msg = str(e)
        have = "none"
        if "(have: " in msg:
            have = msg.split("(have: ", 1)[1].rstrip(")")
        errors.append(_line(path, f"{pid!r} not in assets/", f"have: {have}"))


def validate_spec(spec: Spec, errors: list[str]) -> dict[str, int]:
    """Fill `errors`; return map of duration yaml-path -> frames for paths that parsed."""
    fps = spec.video.fps
    bpm = spec.video.bpm
    if fps <= 0:
        errors.append(_line("video.fps", f"must be positive, got {fps}", "use fps: 30"))
    if bpm is not None and (not math.isfinite(bpm) or bpm <= 0):
        errors.append(_line("video.bpm", f"must be finite and positive, got {_fmt_num(bpm)}", "use bpm: 120"))
    if spec.video.motion_blur < 1:
        errors.append(_line("video.motion_blur", "must be >= 1", "1 = off"))
    parse_wh(spec.video.size, "video.size", errors, even=True)

    fields = _duration_fields(spec)
    uses_beats = any(isinstance(raw, str) and raw.strip().endswith("b") for _, raw in fields)
    if uses_beats and bpm is None:
        errors.append(_line("video.bpm", "required when a duration uses beats", "add bpm: 120"))

    parsed: dict[str, int] = {}
    for path, raw in fields:
        n = parse_dur(raw, fps, bpm, path, errors)
        if n is not None:
            parsed[path] = n

    if not spec.shots:
        errors.append(_line("shots", "need at least one shot", "add a shot"))

    holds: list[int | None] = []
    for i, sh in enumerate(spec.shots):
        h = parsed.get(f"shots[{i}].hold")
        holds.append(h)
        if h is not None and h <= 0:
            errors.append(_line(f"shots[{i}].hold", f"must be > 0, got {h}f", "lengthen hold"))
        has_p = sh.photo is not None
        has_w = sh.wall is not None
        if has_p and has_w:
            errors.append(_line(f"shots[{i}]", "photo and wall are mutually exclusive", "remove one"))
        elif not has_p and not has_w:
            errors.append(_line(f"shots[{i}]", "need photo or wall", "add one"))
        if sh.wall is not None:
            parse_wh(sh.wall.grid, f"shots[{i}].wall.grid", errors, even=False)
            if not sh.wall.prints:
                errors.append(_line(f"shots[{i}].wall.prints", "need at least one print", "add a print"))
            print_ids = [pr.photo for pr in sh.wall.prints]
            have_prints = ", ".join(sorted(set(print_ids))) or "none"
            for j, pr in enumerate(sh.wall.prints):
                if pr.scale is not None and (not math.isfinite(pr.scale) or pr.scale <= 0):
                    errors.append(
                        _line(
                            f"shots[{i}].wall.prints[{j}].scale",
                            f"must be finite and positive, got {_fmt_num(pr.scale)}",
                            "use a scale > 0",
                        )
                    )
            for k, key in enumerate(sh.wall.camera.pos):
                if key.ease not in EASE:
                    errors.append(
                        _line(
                            f"shots[{i}].wall.camera.pos[{k}].ease",
                            f"unknown {key.ease!r}",
                            "use linear, smooth, cubic, outc, inc, expo or ramp",
                        )
                    )
                if key.print is not None and key.print not in print_ids:
                    errors.append(
                        _line(
                            f"shots[{i}].wall.camera.pos[{k}].print",
                            f"{key.print!r} not in this wall's prints",
                            f"have: {have_prints}",
                        )
                    )
            for k, key in enumerate(sh.wall.camera.zoom):
                if key.ease not in EASE:
                    errors.append(
                        _line(
                            f"shots[{i}].wall.camera.zoom[{k}].ease",
                            f"unknown {key.ease!r}",
                            "use linear, smooth, cubic, outc, inc, expo or ramp",
                        )
                    )
                if key.print is not None and key.print not in print_ids:
                    errors.append(
                        _line(
                            f"shots[{i}].wall.camera.zoom[{k}].print",
                            f"{key.print!r} not in this wall's prints",
                            f"have: {have_prints}",
                        )
                    )
                if key.value is not None and (not math.isfinite(key.value) or key.value <= 0):
                    errors.append(
                        _line(
                            f"shots[{i}].wall.camera.zoom[{k}].value",
                            f"must be finite and positive, got {_fmt_num(key.value)}",
                            "use a value > 0",
                        )
                    )
        tr = sh.in_
        if tr is None:
            continue
        if tr.type not in TRANS_TYPES:
            errors.append(
                _line(f"shots[{i}].in.type", f"unknown {tr.type!r}", "use cut, fade, swirl or whip")
            )
            continue
        if tr.type == "cut":
            continue
        if tr.type == "whip" and tr.axis not in ("x", "y"):
            errors.append(_line(f"shots[{i}].in.axis", f"must be x or y, got {tr.axis!r}", "use x or y"))
        if tr.dur is None:
            errors.append(_line(f"shots[{i}].in.dur", f"required for {tr.type}", "use 22f, 4b or 1.5s"))
            continue
        d = parsed.get(f"shots[{i}].in.dur")
        if d is None:
            continue
        if d <= 0:
            errors.append(_line(f"shots[{i}].in.dur", f"must be > 0, got {d}f", "lengthen dur"))
            continue
        if i == 0:
            errors.append(
                _line(f"shots[{i}].in", "no previous shot to transition from", "remove in or use type: cut")
            )
        if tr.type in ("fade", "swirl"):
            if h is not None and d > h:
                errors.append(
                    _line(
                        f"shots[{i}].in.dur",
                        f"{tr.type} {d}f longer than hold {h}f",
                        "shorten dur or lengthen hold",
                    )
                )
        elif tr.type == "whip":
            half = d / 2
            if h is not None and half > h:
                errors.append(
                    _line(
                        f"shots[{i}].in.dur",
                        f"whip half {half:g}f longer than hold {h}f",
                        "shorten dur or lengthen hold",
                    )
                )
            if i > 0 and holds[i - 1] is not None and half > holds[i - 1]:
                errors.append(
                    _line(
                        f"shots[{i}].in.dur",
                        f"whip half {half:g}f longer than previous hold {holds[i - 1]}f",
                        f"shorten dur or lengthen shots[{i - 1}].hold",
                    )
                )

    _reject_overlap_windows(spec, holds, parsed, errors)

    for i, fx in enumerate(spec.fx):
        has_leak = fx.leak is not None
        has_flash = fx.flash is not None
        if has_leak == has_flash:
            errors.append(_line(f"fx[{i}]", "need leak or flash", "set one"))
        if has_leak and fx.dir not in (1, -1):
            errors.append(_line(f"fx[{i}].dir", f"must be 1 or -1, got {fx.dir!r}", "use dir: 1 or dir: -1"))
        if has_flash and fx.dir is not None:
            errors.append(_line(f"fx[{i}].dir", "dir is for leak", "remove dir"))
    return parsed


def _trans_window(kind: str, start: int, dur: int) -> tuple[float, float]:
    if kind in ("fade", "swirl"):
        return float(start), float(start + dur)
    half = dur / 2
    return float(start) - half, float(start) + half


def _reject_overlap_windows(
    spec: Spec, holds: list[int | None], parsed: dict[str, int], errors: list[str]
) -> None:
    t = 0
    starts: list[int | None] = []
    for h in holds:
        starts.append(t if h is not None else None)
        t += h or 0
    wins: list[tuple[int, float, float]] = []
    for i, sh in enumerate(spec.shots):
        tr = sh.in_
        if tr is None or tr.type not in ("fade", "swirl", "whip"):
            continue
        d = parsed.get(f"shots[{i}].in.dur")
        st = starts[i]
        if d is None or st is None:
            continue
        lo, hi = _trans_window(tr.type, st, d)
        wins.append((i, lo, hi))
    for a, (i, a0, a1) in enumerate(wins):
        for j, b0, b1 in wins[a + 1 :]:
            if max(a0, b0) < min(a1, b1):
                errors.append(
                    _line(
                        f"shots[{j}].in.dur",
                        f"overlaps shots[{i}].in",
                        f"shorten one transition or lengthen shots[{j - 1}]",
                    )
                )


def load_spec(path: Path | str) -> Spec:
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as e:
        raise SpecError([_line(".", f"cannot read {path}", str(e))]) from e
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise SpecError([_line(".", "invalid YAML", str(e))]) from e
    if not isinstance(data, dict):
        raise SpecError([_line(".", f"spec must be a mapping, got {type(data).__name__}", "start with video:")])

    errors: list[str] = []
    spec: Spec | None = None
    try:
        spec = Spec.model_validate(data)
    except ValidationError as e:
        errors.extend(_from_pydantic(e))

    assets_rel = data.get("assets", "assets")
    assets_dir = path.parent / str(assets_rel)
    _collect_photos(data, assets_dir, errors)

    if spec is not None:
        validate_spec(spec, errors)
    if errors:
        raise SpecError(errors)
    assert spec is not None
    return spec
