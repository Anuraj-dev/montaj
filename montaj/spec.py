"""YAML spec schema: pydantic models, duration parsing, SpecError."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, WrapSerializer

EASE = ("linear", "smooth", "cubic", "outc", "inc", "expo", "ramp", "sine")
EASE_HINT = "use linear, smooth, cubic, outc, inc, expo, ramp or sine"
TRANS_TYPES = ("cut", "fade", "swirl", "whip")
TEXT_STYLES = ("serif", "script", "caps", "deva")
TEXT_BY = ("word", "char")
TEXT_REVEAL = ("rise", "pop")
TEXT_COLORS = ("cream", "gold", "ink")
_DUR = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([fbs])$")
_BARE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_WH = re.compile(r"^(\d+)x(\d+)$")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_MARKER = re.compile(r"^([a-z][a-z0-9_]*)([+-](?:\d+(?:\.\d*)?|\.\d+)[fbs])?$")
_WORD = re.compile(r"^word:(\d+)$")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


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
TimeExpr = DurationStr


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
    beat0: DurationStr | None = None
    look: str
    intro: IntroOutro | None = None
    outro: IntroOutro | None = None
    motion_blur: int = 1
    background: str = "#000000"


class Transition(Frozen):
    type: str
    dur: DurationStr | None = None
    center: tuple[float, float] | None = None
    axis: str = "x"
    blur: float = 0.0


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


class Drift(Frozen):
    zoom: tuple[float, float] = (1.0, 1.0)
    pan: tuple[float, float] = (0.0, 0.0)


class Tone(Frozen):
    brightness: float = 1.0
    saturate: float = 1.0
    contrast: float = 1.0


class FrameCard(Frozen):
    caption: str | None = None
    tilt: tuple[float, float] | None = None


class Morph(Frozen):
    photo: PhotoId
    at: DurationStr
    dur: DurationStr
    center: tuple[float, float] | None = None


class Shot(Frozen):
    photo: PhotoId | None = None
    wall: WallSpec | None = None
    clip: PhotoId | None = None
    hold: DurationStr | None = None
    until: TimeExpr | None = None
    clip_in: DurationStr | None = None
    crop: CropBox | None = None
    focus: tuple[float, float] | None = None
    zoom: float | None = None
    drift: Drift | None = None
    tone: Tone | None = None
    frame: FrameCard | None = None
    morph: Morph | None = None
    pulse: str | None = None
    blank: bool = False
    in_: Transition | None = Field(default=None, alias="in")
    flash: float | None = None


class TextLine(Frozen):
    text: str
    y: float
    at: TimeExpr
    style: str = "serif"
    italic: bool = False
    gold: bool = False
    color: str | None = None
    size: float | None = None
    weight: float | None = None
    tracking: float | None = None
    line_height: float | None = None
    by: str = "word"
    stagger: DurationStr | None = None
    dur: DurationStr | None = None
    rise: float = 26.0
    reveal: str = "rise"
    sweep: tuple[TimeExpr, TimeExpr] | None = None
    shadow: bool = True
    indent: float = 0.0


class TextBlock(Frozen):
    from_: TimeExpr = Field(alias="from")
    to: TimeExpr
    fade_in: DurationStr | None = None
    fade_out: DurationStr | None = None
    blur: float = 10.0
    lines: list[TextLine]


class Sub(Frozen):
    from_: TimeExpr = Field(alias="from")
    to: TimeExpr
    text: str


class TrackKey(Frozen):
    t: TimeExpr
    v: float
    ease: str | None = None


class Tracks(Frozen):
    dust: list[TrackKey] = Field(default_factory=list)
    glow: list[TrackKey] = Field(default_factory=list)
    bars: list[TrackKey] = Field(default_factory=list)


class Audio(Frozen):
    track: str
    markers: str | None = None
    fade_out: DurationStr | None = None
    loudnorm: bool = True


class Fx(Frozen):
    leak: float | None = None
    flash: float | None = None
    hit: float | None = None
    burst: float | None = None
    at: TimeExpr
    dir: int | None = None
    pos: tuple[float, float] | None = None


class Spec(Frozen):
    video: Video
    assets: str = "assets"
    shots: list[Shot]
    fx: list[Fx] = Field(default_factory=list)
    markers: dict[str, TimeExpr] = Field(default_factory=dict)
    audio: Audio | None = None
    text: list[TextBlock] = Field(default_factory=list)
    subs: list[Sub] = Field(default_factory=list)
    tracks: Tracks | None = None


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


def _dur_frames(raw: str, fps: int, bpm: float | None, path: str, errors: list[str]) -> float | None:
    """Unrounded duration in frames. Text reveals need sub-frame precision."""
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
    return frames


def parse_dur(raw: str, fps: int, bpm: float | None, path: str, errors: list[str]) -> int | None:
    frames = _dur_frames(raw, fps, bpm, path, errors)
    return None if frames is None else round_frame(frames)


def _parse_time(
    raw: object,
    markers: dict[str, float],
    fps: int,
    bpm: float | None,
    path: str,
    errors: list[str],
    *,
    beat0: float = 0.0,
) -> float | None:
    """Duration from film start, or `<marker>[±duration]`. Returns float frames.

    A plain beat duration (`8b`) is a grid time: beat0 + 8 beats. Offsets (`ch1+2b`)
    are spans and do not add beat0.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        errors.append(_line(path, "bare number", "use 22f, 4b or 1.5s"))
        return None
    if not isinstance(raw, str):
        errors.append(_line(path, f"invalid time {raw!r}", "use 12s, 360f, 16b or a marker"))
        return None
    text = raw.strip()
    if _BARE.match(text):
        errors.append(_line(path, "bare number", "use 22f, 4b or 1.5s"))
        return None
    dur = _DUR.match(text)
    if dur:
        frames = _dur_frames(text, fps, bpm, path, errors)
        if frames is None:
            return None
        if dur.group(2) == "b":
            return beat0 + frames
        return frames
    m = _MARKER.match(text)
    if not m:
        errors.append(_line(path, f"invalid time {text!r}", "use 12s, 360f, 16b or a marker"))
        return None
    name, off = m.group(1), m.group(2)
    if name not in markers:
        have = ", ".join(sorted(markers)) or "none"
        errors.append(_line(path, f"unknown marker {name!r}", f"have: {have}"))
        return None
    base = markers[name]
    if not off:
        return base
    sign = 1.0 if off[0] == "+" else -1.0
    extra = _dur_frames(off[1:], fps, bpm, path, errors)
    if extra is None:
        return None
    return base + sign * extra


def parse_time(
    raw: object,
    markers: dict[str, float],
    fps: int,
    bpm: float | None,
    path: str = ".",
    *,
    beat0: float = 0.0,
) -> float:
    """Float frames, unrounded. Raises SpecError on a bad expression."""
    errors: list[str] = []
    val = _parse_time(raw, markers, fps, bpm, path, errors, beat0=beat0)
    if errors or val is None:
        raise SpecError(errors or [_line(path, "invalid time", "use 12s, 360f, 16b or a marker")])
    return val


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
        if sh.hold is not None:
            out.append((f"shots[{i}].hold", sh.hold))
        if sh.in_ is not None and sh.in_.dur is not None:
            out.append((f"shots[{i}].in.dur", sh.in_.dur))
        if sh.wall is not None:
            for k, key in enumerate(sh.wall.camera.pos):
                out.append((f"shots[{i}].wall.camera.pos[{k}].t", key.t))
            for k, key in enumerate(sh.wall.camera.zoom):
                out.append((f"shots[{i}].wall.camera.zoom[{k}].t", key.t))
    return out


def _unrounded_durs(spec: Spec) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if spec.audio is not None and spec.audio.fade_out is not None:
        out.append(("audio.fade_out", spec.audio.fade_out))
    for i, sh in enumerate(spec.shots):
        if sh.clip_in is not None:
            out.append((f"shots[{i}].clip_in", sh.clip_in))
        if sh.morph is not None:
            out.append((f"shots[{i}].morph.at", sh.morph.at))
            out.append((f"shots[{i}].morph.dur", sh.morph.dur))
    for i, block in enumerate(spec.text):
        if block.fade_in:
            out.append((f"text[{i}].fade_in", block.fade_in))
        if block.fade_out:
            out.append((f"text[{i}].fade_out", block.fade_out))
        for j, line in enumerate(block.lines):
            if line.stagger:
                out.append((f"text[{i}].lines[{j}].stagger", line.stagger))
            if line.dur:
                out.append((f"text[{i}].lines[{j}].dur", line.dur))
    return out


def _expr_uses_beats(raw: str) -> bool:
    """True when a duration unit is beats (`4b`, `ch1+2b`). Marker names like `ch1b` are not."""
    text = raw.strip()
    m = _DUR.match(text)
    if m:
        return m.group(2) == "b"
    m = _MARKER.match(text)
    if m and m.group(2):
        return m.group(2)[-1] == "b"
    return False


def _time_strings(spec: Spec) -> list[str]:
    """Every duration / time-expression string, for the beats-need-bpm check."""
    out: list[str] = [raw for _, raw in _duration_fields(spec) if isinstance(raw, str)]
    out.extend(raw for _, raw in _unrounded_durs(spec))
    for v in spec.markers.values():
        if isinstance(v, str):
            out.append(v)
    for sh in spec.shots:
        if sh.until is not None:
            out.append(sh.until)
    for fx in spec.fx:
        if isinstance(fx.at, str):
            out.append(fx.at)
    for block in spec.text:
        out.extend([block.from_, block.to])
        if block.fade_in:
            out.append(block.fade_in)
        if block.fade_out:
            out.append(block.fade_out)
        for line in block.lines:
            out.append(line.at)
            if line.stagger:
                out.append(line.stagger)
            if line.dur:
                out.append(line.dur)
            if line.sweep:
                out.extend(line.sweep)
    for sub in spec.subs:
        out.extend([sub.from_, sub.to])
    if spec.tracks is not None:
        for keys in (spec.tracks.dust, spec.tracks.glow, spec.tracks.bars):
            out.extend(k.t for k in keys)
    return out


def _collect_photos(data: object, assets_dir: Path, spec_dir: Path, errors: list[str]) -> None:
    from montaj.project import find_clip, find_photo

    if not isinstance(data, dict):
        return
    audio = data.get("audio")
    if isinstance(audio, dict) and audio.get("track"):
        track = spec_dir / str(audio["track"])
        if not track.is_file():
            errors.append(
                _line("audio.track", f"{audio['track']!r} not found", "path relative to the spec file")
            )
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
        if "clip" in shot and shot["clip"] is not None:
            pid = _photo_id(shot["clip"])
            if isinstance(pid, str):
                _require_photo(assets_dir, pid, f"shots[{i}].clip", errors, find_clip)
        morph = shot.get("morph")
        if isinstance(morph, dict) and morph.get("photo") is not None:
            pid = _photo_id(morph["photo"])
            if isinstance(pid, str):
                _require_photo(assets_dir, pid, f"shots[{i}].morph.photo", errors, find_photo)
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


_MARKERS_GRID_HINT = 'use {"bpm": 80, "beats": [0.41, ...]} from montaj music analyze'


def _read_markers_json(spec: Spec, spec_dir: Path | None, errors: list[str]) -> object | None:
    """Parsed audio.markers JSON, or None after recording an error."""
    if spec.audio is None or not spec.audio.markers:
        return None
    rel = spec.audio.markers
    if spec_dir is None:
        errors.append(_line("audio.markers", f"{rel!r} needs a spec file", "load the spec from disk"))
        return None
    path = spec_dir / rel
    if not path.is_file():
        errors.append(_line("audio.markers", f"{rel!r} not found", "path relative to the spec file"))
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        errors.append(_line("audio.markers", "invalid JSON", str(e)))
        return None


def _finite_number(v: object) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    x = float(v)
    return x if math.isfinite(x) else None


def _markers_grid(data: object, errors: list[str]) -> tuple[float | None, float | None]:
    """`(bpm, beat0_seconds)` from a markers file. One error on `audio.markers` if bad."""
    if not isinstance(data, dict):
        errors.append(_line("audio.markers", "invalid bpm/beats", _MARKERS_GRID_HINT))
        return None, None
    bpm: float | None = None
    if "bpm" in data:
        n = _finite_number(data["bpm"])
        if n is None or n <= 0:
            errors.append(_line("audio.markers", "invalid bpm/beats", _MARKERS_GRID_HINT))
            return None, None
        bpm = n
    beat0_s: float | None = None
    if "beats" in data:
        beats = data["beats"]
        if not isinstance(beats, list):
            errors.append(_line("audio.markers", "invalid bpm/beats", _MARKERS_GRID_HINT))
            return bpm, None
        nums: list[float] = []
        for x in beats:
            n = _finite_number(x)
            if n is None or n < 0:
                errors.append(_line("audio.markers", "invalid bpm/beats", _MARKERS_GRID_HINT))
                return bpm, None
            nums.append(n)
        if nums:
            beat0_s = nums[0]
    return bpm, beat0_s


def _parse_beat0(raw: str, fps: int, errors: list[str]) -> float | None:
    """`video.beat0` as frames. `b` units are an error."""
    path = "video.beat0"
    if isinstance(raw, str):
        m = _DUR.match(raw.strip())
        if m and m.group(2) == "b":
            errors.append(_line(path, "beats are not allowed", "use 0.5s or 12f"))
            return None
    return _dur_frames(raw, fps, None, path, errors)


def _format_beat0(frames: float, fps: int) -> str:
    """Canonical seconds string for a resolved phase (`0s`, `0.41s`)."""
    seconds = 0.0 if fps <= 0 else frames / float(fps)
    if seconds == 0:
        return "0s"
    return f"{seconds:g}s"


def _beat_period(fps: int, bpm: float | None) -> float | None:
    if bpm is None or not math.isfinite(bpm) or bpm <= 0 or fps <= 0:
        return None
    period = 60.0 / bpm * fps
    return period if math.isfinite(period) and period > 0 else None


def _whole_beats(raw: object) -> int | None:
    """Integer N when `raw` is `Nb`; None for fractional beats or other units."""
    if not isinstance(raw, str):
        return None
    m = _DUR.match(raw.strip())
    if not m or m.group(2) != "b":
        return None
    n = float(m.group(1))
    if not math.isfinite(n) or n != int(n):
        return None
    return int(n)


def _on_beat_grid(u: float, beat0: float, period: float) -> bool:
    k = round((u - beat0) / period)
    return abs(u - (beat0 + k * period)) <= 1e-6


def _snap_forward_to_grid(u: float, beat0: float, period: float) -> float:
    if _on_beat_grid(u, beat0, period):
        return u
    return beat0 + math.ceil((u - beat0) / period) * period


def layout_shots(
    spec: Spec,
    markers: dict[str, float],
    fps: int,
    bpm: float | None,
    beat0: float,
    errors: list[str],
) -> tuple[list[tuple[int, int] | None], dict[str, float]]:
    """Lay shots out on an unrounded clock. Returns (bounds, until times).

    `u` is running time in frames (float). Shot end frame is `round_frame(end_u)`;
    the next shot starts there; `u` continues from `end_u` so rounding does not drift.
    Whole-beat holds snap forward onto `beat0 + k·T`.
    """
    period = _beat_period(fps, bpm)
    bounds: list[tuple[int, int] | None] = []
    until_times: dict[str, float] = {}
    u = 0.0
    start = 0
    for i, sh in enumerate(spec.shots):
        has_hold = sh.hold is not None
        has_until = sh.until is not None
        if has_hold and has_until:
            errors.append(_line(f"shots[{i}]", "hold and until are mutually exclusive", "remove one"))
            bounds.append(None)
            continue
        if not has_hold and not has_until:
            errors.append(_line(f"shots[{i}]", "need hold or until", "set one"))
            bounds.append(None)
            continue
        if has_until:
            until_f = _parse_time(
                sh.until, markers, fps, bpm, f"shots[{i}].until", errors, beat0=beat0
            )
            if until_f is None:
                bounds.append(None)
                continue
            until_times[f"shots[{i}].until"] = until_f
            end_u = until_f
        else:
            span = _dur_frames(sh.hold, fps, bpm, f"shots[{i}].hold", errors)
            if span is None:
                bounds.append(None)
                continue
            end_u = u + span
            if period is not None and _whole_beats(sh.hold) is not None:
                end_u = _snap_forward_to_grid(end_u, beat0, period)
        end = round_frame(end_u)
        if end <= start:
            if has_until:
                errors.append(
                    _line(
                        f"shots[{i}].until",
                        f"ends at {end}f, not after start {start}f",
                        f"use a time after {start}f",
                    )
                )
            else:
                errors.append(
                    _line(f"shots[{i}].hold", f"must be > 0, got {end - start}f", "lengthen hold")
                )
            bounds.append(None)
            continue
        bounds.append((start, end))
        start = end
        u = end_u
    return bounds, until_times


def _load_word_starts(
    spec: Spec,
    spec_dir: Path | None,
    errors: list[str],
    *,
    data: object | None = None,
    loaded: bool = False,
) -> dict[int, float]:
    """Map word `i` -> start seconds from audio.markers JSON."""
    if spec.audio is None or not spec.audio.markers:
        return {}
    if not loaded:
        data = _read_markers_json(spec, spec_dir, errors)
        if data is None:
            return {}
    elif data is None:
        return {}
    words = data.get("words") if isinstance(data, dict) else None
    if not isinstance(words, list):
        errors.append(
            _line("audio.markers", "need a words list", 'use {"words": [{"i", "w", "s", "e"}]}')
        )
        return {}
    out: dict[int, float] = {}
    for j, w in enumerate(words):
        if not isinstance(w, dict) or "i" not in w or "s" not in w:
            continue
        try:
            i = int(w["i"])
        except (TypeError, ValueError):
            errors.append(
                _line(f"audio.markers.words[{j}].i", f"need an integer, got {w['i']!r}", "use i: 1")
            )
            continue
        s = w["s"]
        try:
            if isinstance(s, bool):
                raise TypeError
            sec = float(s)
        except (TypeError, ValueError):
            errors.append(
                _line(f"audio.markers.words[{j}].s", f"not a number, got {s!r}", "use seconds like 12.0")
            )
            continue
        if not math.isfinite(sec):
            errors.append(
                _line(
                    f"audio.markers.words[{j}].s",
                    f"must be finite, got {_fmt_num(sec)}",
                    "use seconds like 12.0",
                )
            )
            continue
        out[i] = sec
    return out


def resolve_markers(
    spec: Spec,
    spec_dir: Path | None,
    fps: int,
    bpm: float | None,
    errors: list[str],
    *,
    beat0: float = 0.0,
    markers_json: object | None = None,
    markers_loaded: bool = False,
) -> dict[str, float]:
    """Resolve `markers` to float frames. Acyclic name→name refs are allowed."""
    uses_word = [
        (name, expr.strip())
        for name, expr in spec.markers.items()
        if isinstance(expr, str) and _WORD.match(expr.strip())
    ]
    word_s: dict[int, float] = {}
    if uses_word:
        if spec.audio is None or not spec.audio.markers:
            for name, expr in uses_word:
                errors.append(
                    _line(f"markers.{name}", f"{expr} needs audio.markers", "set audio.markers")
                )
        else:
            word_s = _load_word_starts(
                spec, spec_dir, errors, data=markers_json, loaded=markers_loaded
            )

    for name in spec.markers:
        if not _NAME.match(name):
            errors.append(_line(f"markers.{name}", f"invalid name {name!r}", "use [a-z][a-z0-9_]*"))

    resolved: dict[str, float] = {}
    pending: dict[str, str] = {}
    for name, expr in spec.markers.items():
        if not _NAME.match(name):
            continue
        path = f"markers.{name}"
        if not isinstance(expr, str):
            errors.append(_line(path, "need a string", "use 5.2s or word:1"))
            continue
        text = expr.strip()
        wm = _WORD.match(text)
        if wm:
            i = int(wm.group(1))
            if i not in word_s:
                if spec.audio is not None and spec.audio.markers:
                    have = ", ".join(str(k) for k in sorted(word_s)) or "none"
                    errors.append(_line(path, f"unknown word {i}", f"have: {have}"))
            else:
                resolved[name] = word_s[i] * fps
            continue
        if _DUR.match(text):
            val = _parse_time(text, {}, fps, bpm, path, errors, beat0=beat0)
            if val is not None:
                resolved[name] = val
            continue
        pending[name] = text

    progress = True
    while pending and progress:
        progress = False
        for name, expr in list(pending.items()):
            bag: list[str] = []
            val = _parse_time(expr, resolved, fps, bpm, f"markers.{name}", bag, beat0=beat0)
            if val is not None:
                resolved[name] = val
                del pending[name]
                progress = True
            elif any("unknown marker" not in e for e in bag):
                errors.extend(bag)
                del pending[name]

    for name, expr in pending.items():
        m = _MARKER.match(expr)
        ref = m.group(1) if m else None
        if ref is not None and (ref in pending or ref in spec.markers) and ref not in resolved:
            errors.append(
                _line(
                    f"markers.{name}",
                    f"cyclic marker {ref!r}",
                    "markers may only reference already-resolved names",
                )
            )
        else:
            _parse_time(expr, resolved, fps, bpm, f"markers.{name}", errors, beat0=beat0)
    return resolved


def _check_color(path: str, color: str | None, errors: list[str]) -> None:
    if color is None:
        return
    if color not in TEXT_COLORS and not _HEX.match(color):
        errors.append(_line(path, f"unknown colour {color!r}", "use cream, gold, ink or #rrggbb"))


def _check_ease(path: str, ease: str, errors: list[str]) -> None:
    if ease not in EASE:
        errors.append(_line(path, f"unknown {ease!r}", EASE_HINT))


def validate_spec(
    spec: Spec, errors: list[str], spec_dir: Path | None = None
) -> tuple[dict[str, int], dict[str, float], dict[str, float], float | None, float]:
    """Fill `errors`; return (rounded frames, float frames, markers, bpm, beat0 frames)."""
    fps = spec.video.fps
    specified_bpm = spec.video.bpm is not None
    if fps <= 0:
        errors.append(_line("video.fps", f"must be positive, got {fps}", "use fps: 30"))
    if specified_bpm and (not math.isfinite(spec.video.bpm) or spec.video.bpm <= 0):
        errors.append(
            _line(
                "video.bpm",
                f"must be finite and positive, got {_fmt_num(spec.video.bpm)}",
                "use bpm: 120",
            )
        )
        bpm: float | None = None
    else:
        bpm = spec.video.bpm
    if spec.video.motion_blur < 1:
        errors.append(_line("video.motion_blur", "must be >= 1", "1 = off"))
    parse_wh(spec.video.size, "video.size", errors, even=True)

    beat0_locked = spec.video.beat0 is not None
    beat0 = 0.0
    if beat0_locked:
        parsed_b0 = _parse_beat0(spec.video.beat0, fps, errors)
        if parsed_b0 is not None:
            beat0 = parsed_b0

    markers_json: object | None = None
    markers_loaded = False
    if spec.audio is not None and spec.audio.markers:
        markers_json = _read_markers_json(spec, spec_dir, errors)
        markers_loaded = True
        if markers_json is not None:
            m_bpm, m_beat0_s = _markers_grid(markers_json, errors)
            if not specified_bpm:
                if bpm is None:
                    bpm = m_bpm
                if not beat0_locked and m_beat0_s is not None:
                    beat0 = m_beat0_s * fps

    uses_beats = any(isinstance(raw, str) and _expr_uses_beats(raw) for raw in _time_strings(spec))
    if uses_beats and bpm is None and not specified_bpm:
        errors.append(
            _line(
                "video.bpm",
                "required when a duration uses beats",
                "add bpm: 120, or audio.markers from montaj music analyze",
            )
        )

    markers = resolve_markers(
        spec,
        spec_dir,
        fps,
        bpm,
        errors,
        beat0=beat0,
        markers_json=markers_json,
        markers_loaded=markers_loaded,
    )

    parsed: dict[str, int] = {}
    times: dict[str, float] = {}
    for path, raw in _duration_fields(spec):
        if path.endswith("].hold") and path.count(".") == 1:
            continue
        n = parse_dur(raw, fps, bpm, path, errors)
        if n is not None:
            parsed[path] = n
            times[path] = float(n)
    for path, raw in _unrounded_durs(spec):
        n = _dur_frames(raw, fps, bpm, path, errors)
        if n is not None:
            times[path] = n

    if not spec.shots:
        errors.append(_line("shots", "need at least one shot", "add a shot"))

    bounds, until_times = layout_shots(spec, markers, fps, bpm, beat0, errors)
    times.update(until_times)
    holds: list[int | None] = []
    for i, span in enumerate(bounds):
        if span is None:
            holds.append(None)
            continue
        start, end = span
        h = end - start
        parsed[f"shots[{i}].hold"] = h
        parsed[f"shots[{i}].start"] = start
        parsed[f"shots[{i}].end"] = end
        holds.append(h)

    for i, sh in enumerate(spec.shots):
        h = holds[i]
        has_p = sh.photo is not None
        has_w = sh.wall is not None
        has_c = sh.clip is not None
        has_b = sh.blank
        n_src = has_p + has_w + has_c + has_b
        if n_src > 1:
            errors.append(
                _line(f"shots[{i}]", "photo, wall, clip and blank are mutually exclusive", "remove extra")
            )
        elif n_src == 0:
            errors.append(_line(f"shots[{i}]", "need photo, wall or clip", "add one"))
        if not has_p:
            for name in ("drift", "frame", "morph", "tone", "pulse"):
                if getattr(sh, name) is not None:
                    errors.append(_line(f"shots[{i}].{name}", f"{name} needs photo", "add photo:"))
        if sh.clip_in is not None and not has_c:
            errors.append(_line(f"shots[{i}].clip_in", "clip_in needs clip", "add clip:"))
        if sh.zoom is not None and sh.drift is not None:
            errors.append(_line(f"shots[{i}]", "zoom and drift are mutually exclusive", "remove one"))
        if sh.pulse is not None:
            if sh.pulse != "heartbeat":
                errors.append(_line(f"shots[{i}].pulse", f"unknown {sh.pulse!r}", "use heartbeat"))
            elif bpm is None:
                errors.append(_line(f"shots[{i}].pulse", "needs video.bpm", "add bpm: 76"))
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
                            EASE_HINT,
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
                            EASE_HINT,
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
        if i == 0 and tr.type != "fade":
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
        kinds = [fx.leak is not None, fx.flash is not None, fx.hit is not None, fx.burst is not None]
        if sum(kinds) != 1:
            errors.append(_line(f"fx[{i}]", "need leak, flash, hit or burst", "set one"))
        if fx.leak is not None and fx.dir not in (1, -1):
            errors.append(_line(f"fx[{i}].dir", f"must be 1 or -1, got {fx.dir!r}", "use dir: 1 or dir: -1"))
        if fx.flash is not None and fx.dir is not None:
            errors.append(_line(f"fx[{i}].dir", "dir is for leak", "remove dir"))
        if (fx.hit is not None or fx.burst is not None) and fx.dir is not None:
            errors.append(_line(f"fx[{i}].dir", "dir is for leak", "remove dir"))
        if fx.burst is not None and fx.pos is None:
            errors.append(_line(f"fx[{i}].pos", "required for burst", "set pos: [x, y]"))
        at_f = _parse_time(fx.at, markers, fps, bpm, f"fx[{i}].at", errors, beat0=beat0)
        if at_f is not None:
            times[f"fx[{i}].at"] = at_f
            parsed[f"fx[{i}].at"] = round_frame(at_f)

    _validate_text(spec, markers, fps, bpm, times, errors, beat0=beat0)
    _validate_tracks(spec, markers, fps, bpm, times, errors, beat0=beat0)
    return parsed, times, markers, bpm, beat0


def _validate_text(
    spec: Spec,
    markers: dict[str, float],
    fps: int,
    bpm: float | None,
    times: dict[str, float],
    errors: list[str],
    *,
    beat0: float = 0.0,
) -> None:
    for i, block in enumerate(spec.text):
        for field, raw in (("from", block.from_), ("to", block.to)):
            val = _parse_time(raw, markers, fps, bpm, f"text[{i}].{field}", errors, beat0=beat0)
            if val is not None:
                times[f"text[{i}].{field}"] = val
        if not block.lines:
            errors.append(_line(f"text[{i}].lines", "need at least one line", "add a line"))
        for j, line in enumerate(block.lines):
            p = f"text[{i}].lines[{j}]"
            if line.style not in TEXT_STYLES:
                errors.append(_line(f"{p}.style", f"unknown {line.style!r}", "use serif, script, caps or deva"))
            if line.italic and line.style != "serif":
                errors.append(_line(f"{p}.italic", "italic is serif-only", "set style: serif"))
            if line.by not in TEXT_BY:
                errors.append(_line(f"{p}.by", f"unknown {line.by!r}", "use word or char"))
            if line.reveal not in TEXT_REVEAL:
                errors.append(_line(f"{p}.reveal", f"unknown {line.reveal!r}", "use rise or pop"))
            _check_color(f"{p}.color", line.color, errors)
            val = _parse_time(line.at, markers, fps, bpm, f"{p}.at", errors, beat0=beat0)
            if val is not None:
                times[f"{p}.at"] = val
            if line.sweep:
                for k, raw in enumerate(line.sweep):
                    sv = _parse_time(raw, markers, fps, bpm, f"{p}.sweep[{k}]", errors, beat0=beat0)
                    if sv is not None:
                        times[f"{p}.sweep[{k}]"] = sv
    for i, sub in enumerate(spec.subs):
        for field, raw in (("from", sub.from_), ("to", sub.to)):
            val = _parse_time(raw, markers, fps, bpm, f"subs[{i}].{field}", errors, beat0=beat0)
            if val is not None:
                times[f"subs[{i}].{field}"] = val


def _validate_tracks(
    spec: Spec,
    markers: dict[str, float],
    fps: int,
    bpm: float | None,
    times: dict[str, float],
    errors: list[str],
    *,
    beat0: float = 0.0,
) -> None:
    if spec.tracks is None:
        return
    for name in ("dust", "glow", "bars"):
        keys: list[TrackKey] = getattr(spec.tracks, name)
        for k, key in enumerate(keys):
            path = f"tracks.{name}[{k}]"
            val = _parse_time(key.t, markers, fps, bpm, f"{path}.t", errors, beat0=beat0)
            if val is not None:
                times[f"{path}.t"] = val
            if key.ease is not None:
                _check_ease(f"{path}.ease", key.ease, errors)


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
    _collect_photos(data, assets_dir, path.parent, errors)

    if spec is not None:
        validate_spec(spec, errors, spec_dir=path.parent)
    if errors:
        raise SpecError(errors)
    assert spec is not None
    return spec
