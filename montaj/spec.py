"""YAML spec schema: pydantic models, duration parsing, SpecError."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    StrictInt,
    Tag,
    ValidationError,
    WrapSerializer,
    model_validator,
)

from montaj.events import (
    Hit,
    Miss,
    compile_at_event,
    compile_event_marker,
    format_seconds,
    is_event_marker,
    load_events_file,
    match_event,
)

EASE = ("linear", "smooth", "cubic", "outc", "inc", "expo", "ramp", "sine")
EASE_HINT = "use linear, smooth, cubic, outc, inc, expo, ramp or sine"
TRANS_TYPES = ("cut", "fade", "swirl", "whip", "wall-pullback", "burn-white")
# These occupy [start, start+dur) of the incoming shot. Whip is centred on the cut.
INCOMING_TRANS = ("fade", "swirl", "wall-pullback", "burn-white")
TRANS_HINT = "use cut, fade, swirl, whip, wall-pullback or burn-white"
# Starting magnification of a wall pull-back. Materialised onto the shot so the hash sees it.
WALL_PULLBACK_ZOOM = 3.2
LOOK_NAMES = ("warm-film", "golden-film", "clean", "cool-tech")
INTRO_TYPES = ("white-lift",)
OUTRO_TYPES = ("glow", "fade")
TEXT_STYLES = ("serif", "script", "caps", "deva")
TEXT_BY = ("word", "char")
TEXT_REVEAL = ("rise", "pop")
TEXT_COLORS = ("cream", "gold", "ink")
BACKDROP_TYPES = ("gradient", "solid")
BACKDROP_PLACES = ("bottom", "top", "band")
BACKDROP_STRENGTH = 0.8
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


def _reject_bool(v: object) -> object:
    """Pydantic turns `true` into `1.0` before a later numeric check can see the bool."""
    if isinstance(v, bool):
        raise ValueError("need a number, not a boolean")
    return v


RealNum = Annotated[float, BeforeValidator(_reject_bool)]


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


def _reject_bare_duration_or_bool(v: object) -> object:
    """Pydantic would read YAML `true` as the integer 1, a one-frame interval."""
    if isinstance(v, bool):
        raise ValueError("bare number")
    return _reject_bare_duration(v)


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


class LutLook(Frozen):
    lut: str
    strength: RealNum = 1.0
    vignette: RealNum = 0.0
    grain: RealNum = 0.0


def _look_kind(v: object) -> str:
    """A name or a `{lut: …}` mapping. The tag keeps pydantic from reporting both branches."""
    return "look_lut" if isinstance(v, (dict, LutLook)) else "look_name"


Look = Annotated[
    Annotated[str, Tag("look_name")] | Annotated[LutLook, Tag("look_lut")],
    Discriminator(_look_kind),
]


class Video(Frozen):
    size: str
    fps: int
    bpm: float | None = None
    beat0: DurationStr | None = None
    look: Look
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
    # Absent stays out of model_dump so films without a pull-back keep their segment hash.
    zoom: RealNum | None = Field(default=None, exclude_if=lambda v: v is None)


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


def _as_sequence_item(v: object) -> object:
    """A list entry is a photo stem, `#rrggbb`, or `{photo}` / `{color}`."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return {"photo": str(v)}
    if isinstance(v, str):
        text = v.strip()
        if not text:
            raise ValueError("need a photo stem or #rrggbb")
        if text.startswith("#"):
            return {"color": text}
        return {"photo": text}
    return v


class SequenceItem(Frozen):
    photo: PhotoId | None = None
    color: str | None = None


class SequenceSpec(Frozen):
    every: Annotated[str | int, BeforeValidator(_reject_bare_duration_or_bool)]  # int only after resolve
    items: list[Annotated[SequenceItem, BeforeValidator(_as_sequence_item)]]


class CameraKey(Frozen):
    t: DurationStr | int
    zoom: RealNum | None = None
    focus: tuple[float, float] | None = None
    ease: str = "smooth"


class SpeedKey(Frozen):
    t: DurationStr | int
    v: RealNum
    ease: str = "smooth"


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


class CardSpec(Frozen):
    """One shot's picture: a solid colour with text lines on it."""

    color: str
    text: list[TextLine]


class Shot(Frozen):
    photo: PhotoId | None = None
    wall: WallSpec | None = None
    clip: PhotoId | None = None
    hold: DurationStr | None = None
    until: TimeExpr | None = None
    clip_in: DurationStr | None = None
    # Authoring-only: `resolve` folds the pick into clip_in. Absent stays out of model_dump
    # so an existing film keeps its segment hash.
    at_event: dict[str, Any] | None = Field(default=None, exclude_if=lambda v: v is None)
    pre: DurationStr | None = Field(default=None, exclude_if=lambda v: v is None)
    range: tuple[DurationStr, DurationStr] | None = None
    crop: CropBox | None = None
    focus: tuple[float, float] | None = None
    zoom: float | None = None
    # Absent stays out of model_dump so films that never use these keep their segment hash.
    camera: list[CameraKey] | None = Field(default=None, exclude_if=lambda v: v is None)
    speed: RealNum | list[SpeedKey] | None = Field(default=None, exclude_if=lambda v: v is None)
    freeze: DurationStr | int | None = Field(default=None, exclude_if=lambda v: v is None)
    drift: Drift | None = None
    tone: Tone | None = None
    frame: FrameCard | None = None
    morph: Morph | None = None
    pulse: str | None = None
    blank: bool = False
    # Absent stays out of model_dump so films without a sequence keep their segment hash.
    sequence: SequenceSpec | None = Field(default=None, exclude_if=lambda v: v is None)
    card: CardSpec | None = Field(default=None, exclude_if=lambda v: v is None)
    in_: Transition | None = Field(default=None, alias="in")
    flash: float | None = None


class TextBackdrop(Frozen):
    """Dark scrim behind a text block. The compositor paints it; no image asset."""

    type: str = "gradient"
    strength: RealNum = BACKDROP_STRENGTH
    place: str = "band"


def _coerce_backdrop(v: object) -> object:
    """`true`, `gradient` and `solid` are the default scrim. A mapping sets the knobs."""
    if v is None or isinstance(v, TextBackdrop):
        return v
    if v is True:
        return {}
    if isinstance(v, str):
        return {"type": v}
    if v is False:
        raise ValueError("omit backdrop to turn it off")
    return v


class TextBlock(Frozen):
    from_: TimeExpr = Field(alias="from")
    to: TimeExpr
    fade_in: DurationStr | None = None
    fade_out: DurationStr | None = None
    blur: float = 10.0
    # Absent stays out of model_dump so a film with no scrim keeps its segment hash.
    backdrop: Annotated[TextBackdrop | None, BeforeValidator(_coerce_backdrop)] = Field(
        default=None, exclude_if=lambda v: v is None
    )
    lines: list[TextLine]


class Sub(Frozen):
    from_: TimeExpr = Field(alias="from")
    to: TimeExpr
    text: str
    size: RealNum | None = None
    color: str | None = None


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
    burst: RealNum | None = None
    at: TimeExpr
    dir: int | None = None
    pos: tuple[float, float] | None = None


class ExpectSpan(Frozen):
    """One designed black or frozen window. Times are film-start expressions, so markers work."""

    black: tuple[TimeExpr, TimeExpr] | None = None
    frozen: tuple[TimeExpr, TimeExpr] | None = None
    why: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> ExpectSpan:
        if self.black is None and self.frozen is None:
            raise ValueError("need black or frozen")
        if self.black is not None and self.frozen is not None:
            raise ValueError("set black or frozen, not both")
        if self.why is None:
            return self
        why = self.why.strip()
        if not why or "\n" in self.why or "\r" in self.why:
            raise ValueError("why must be one non-empty line")
        if why != self.why:
            return self.model_copy(update={"why": why})
        return self


class LoudnessTarget(Frozen):
    """Override `check`'s integrated-loudness target. `tol` is LU around `i`."""

    i: RealNum
    tol: RealNum = 1.0


class CheckSpec(Frozen):
    expect: list[ExpectSpan] = Field(default_factory=list)
    loudness: LoudnessTarget | None = None
    digits: Literal["none", "spec"] | None = None
    faces: Literal["none"] | None = None
    sample: StrictInt = 3  # strict: YAML `true` must not coerce to 1


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
    check: CheckSpec | None = None


def _yaml_path(loc: tuple) -> str:
    parts: list[str] = []
    for x in loc:
        if x == "in_":
            x = "in"
        if x in ("look_name", "look_lut"):  # discriminator tags, not spec keys
            continue
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
        elif re.search(r"shots\[\d+\]\.range(\[\d+\])?$", path) and err["type"] in (
            "too_short", "too_long", "tuple_type", "missing"
        ):
            msg = "need [start, end]"
            hint = "use [start, end] like [1.5s, 4s]"
        elif "whole pixel" in msg:
            hint = "use integer source-px"
        elif "not a boolean" in msg:
            hint = "use a number"
        elif msg in ("need black or frozen", "set black or frozen, not both"):
            hint = "use black: [0s, 2s] or frozen: [logo, logo+0.5s]"
        elif msg == "why must be one non-empty line":
            hint = "a short reason, one line"
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
        if sh.freeze is not None:
            out.append((f"shots[{i}].freeze", sh.freeze))
        for k, ckey in enumerate(sh.camera or ()):
            out.append((f"shots[{i}].camera[{k}].t", ckey.t))
        if isinstance(sh.speed, list):
            for k, skey in enumerate(sh.speed):
                out.append((f"shots[{i}].speed[{k}].t", skey.t))
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
        if sh.pre is not None:
            out.append((f"shots[{i}].pre", sh.pre))
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
        if sh.range is not None:
            out.extend(sh.range)
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
    for sh in spec.shots:
        if sh.card is None:
            continue
        for line in sh.card.text:
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
    if spec.check is not None:
        for item in spec.check.expect:
            pair = item.black if item.black is not None else item.frozen
            if pair:
                out.extend(pair)
    return out


def _needs_assets_dir(shots: object) -> bool:
    """False when the film's only media is path clips. Blanks may sit beside them.

    A stem, a photo, or a blank-only film still needs the assets folder.
    """
    from montaj.project import clip_is_file

    if not isinstance(shots, list) or not shots:
        return True
    saw_path = False
    for shot in shots:
        if not isinstance(shot, dict):
            return True
        if shot.get("photo") is not None or shot.get("wall") is not None:
            return True
        morph = shot.get("morph")
        if isinstance(morph, dict) and morph.get("photo") is not None:
            return True
        clip = shot.get("clip")
        if clip is not None:
            pid = _photo_id(clip)
            if isinstance(pid, str) and clip_is_file(pid):
                saw_path = True
                continue
            return True
        if shot.get("blank") is True:
            continue
        return True
    return not saw_path


def _note_path_clip(shot: dict, i: int, spec_dir: Path, assets_dir: Path, errors: list[str]) -> None:
    from montaj.project import AssetMissing, clip_is_file, resolve_clip

    clip = shot.get("clip")
    if clip is None:
        return
    pid = _photo_id(clip)
    if not isinstance(pid, str) or not clip_is_file(pid):
        return
    try:
        resolve_clip(spec_dir, assets_dir, pid)
    except AssetMissing as e:
        errors.append(_line(f"shots[{i}].clip", str(e), e.hint))


def _collect_photos(data: object, assets_dir: Path, spec_dir: Path, errors: list[str]) -> None:
    from montaj.project import clip_is_file, find_clip, find_photo

    if not isinstance(data, dict):
        return
    audio = data.get("audio")
    if isinstance(audio, dict) and audio.get("track"):
        track = spec_dir / str(audio["track"])
        if not track.is_file():
            errors.append(
                _line("audio.track", f"{audio['track']!r} not found", "path relative to the spec file")
            )
    shots = data.get("shots")
    if not isinstance(shots, list):
        shots = []
    if not assets_dir.is_dir():
        if _needs_assets_dir(shots):
            rel = data.get("assets", "assets")
            errors.append(_line("assets", f"{rel!r} not a directory", "create it or fix the path"))
        for i, shot in enumerate(shots):
            if isinstance(shot, dict):
                _note_path_clip(shot, i, spec_dir, assets_dir, errors)
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
                if clip_is_file(pid):
                    _note_path_clip(shot, i, spec_dir, assets_dir, errors)
                else:
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
        seq = shot.get("sequence")
        if isinstance(seq, dict):
            items = seq.get("items")
            if isinstance(items, list):
                for j, item in enumerate(items):
                    found = _sequence_photo_id(item)
                    if found is not None:
                        pid, suffix = found
                        _require_photo(
                            assets_dir, pid, f"shots[{i}].sequence.items[{j}]{suffix}", errors, find_photo
                        )


def _sequence_photo_id(item: object) -> tuple[str, str] | None:
    """`(photo id, path suffix)` for one raw sequence entry, or None for a colour."""
    if isinstance(item, bool):
        return None
    if isinstance(item, int):
        return str(item), ""
    if isinstance(item, str):
        text = item.strip()
        if not text or text.startswith("#"):
            return None
        return text, ""
    if isinstance(item, dict) and item.get("photo") is not None:
        pid = _photo_id(item["photo"])
        if isinstance(pid, str):
            return pid, ".photo"
    return None


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


def _length_conflict(has_hold: bool, has_until: bool, has_range: bool) -> str:
    if has_hold and has_until and not has_range:
        return "hold and until are mutually exclusive"
    if has_range and has_hold and not has_until:
        return "range and hold are mutually exclusive"
    if has_range and has_until and not has_hold:
        return "range and until are mutually exclusive"
    return "hold, until and range are mutually exclusive"


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
        has_range = sh.range is not None
        chosen = has_hold + has_until + has_range
        if chosen > 1:
            errors.append(_line(f"shots[{i}]", _length_conflict(has_hold, has_until, has_range), "remove one"))
            bounds.append(None)
            continue
        if chosen == 0:
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
        elif sh.range is not None:
            # Source span, not a film-start time: `1b` is one beat long, and beat0 is not added.
            start_f = _dur_frames(sh.range[0], fps, bpm, f"shots[{i}].range[0]", errors)
            end_f = _dur_frames(sh.range[1], fps, bpm, f"shots[{i}].range[1]", errors)
            if start_f is None or end_f is None:
                bounds.append(None)
                continue
            if end_f <= start_f:
                errors.append(
                    _line(
                        f"shots[{i}].range",
                        f"end must be after start ({sh.range[0]} .. {sh.range[1]})",
                        "use a later end",
                    )
                )
                bounds.append(None)
                continue
            end_u = u + (end_f - start_f)
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
            if has_range:
                errors.append(
                    _line(
                        f"shots[{i}].range",
                        f"shorter than one frame ({sh.range[0]} .. {sh.range[1]})",
                        "lengthen the range",
                    )
                )
            elif has_until:
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


class _Events:
    """One read of each `assets/<stem>.events.json` per validate. Not process-global."""

    def __init__(self, assets: Path | None, errors: list[str]):
        self.assets = assets
        self.errors = errors
        self._cache: dict[str, list[dict] | None] = {}

    def get(self, stem: str, path: str) -> list[dict] | None:
        if stem in self._cache:
            return self._cache[stem]
        rows = _read_events(self.assets, stem, path, self.errors)
        self._cache[stem] = rows
        return rows


def _assets_dir(spec: Spec, spec_dir: Path | None) -> Path | None:
    if spec_dir is None:
        return None
    rel = Path(spec.assets)
    return rel if rel.is_absolute() else spec_dir / rel


def _read_events(assets: Path | None, stem: str, path: str, errors: list[str]) -> list[dict] | None:
    if assets is None:
        errors.append(_line(path, f"no events for {stem!r}", "load the spec from disk"))
        return None
    file = assets / f"{stem}.events.json"
    if not file.is_file():
        errors.append(_line(path, f"no events for {stem!r}", f"add assets/{stem}.events.json"))
        return None
    return load_events_file(file, f"assets/{stem}.events.json", errors)


def _clip_in_from_event(
    sh: Shot, i: int, fps: int, times: dict[str, float], events: _Events, errors: list[str]
) -> str | None:
    """Matched event time minus `pre`, as a duration string. None if this shot already errored."""
    path = f"shots[{i}]"
    if sh.pre is not None and f"shots[{i}].pre" not in times:
        return None
    sel = compile_at_event(sh.at_event, f"shots[{i}].at_event", errors)
    if sel is None or sh.clip is None or fps <= 0:
        return None
    rows = events.get(sh.clip, path)
    if rows is None:
        return None
    if not rows:
        errors.append(_line(path, f"no events in {sh.clip}", "have: none"))
        return None
    t = _event_seconds(match_event(rows, sel), sh.clip, path, errors)
    if t is None:
        return None
    pre_s = times.get(f"shots[{i}].pre", 0.0) / fps
    clip_s = t - pre_s
    if clip_s < -1e-9:
        errors.append(
            _line(
                f"shots[{i}].pre",
                f"{sh.pre} starts before the clip",
                f"event at {t:.6g}s",
            )
        )
        return None
    return format_seconds(0.0 if clip_s < 0 else clip_s)


def _event_seconds(found: Hit | Miss, stem: str, path: str, errors: list[str]) -> float | None:
    if isinstance(found, Miss):
        errors.append(_line(path, f"no event {found.label} in {stem}", found.hint))
        return None
    t = float(found.event["t"])
    if t < 0:
        errors.append(
            _line(
                path,
                f"event at {t:.6g}s is before the clip start",
                "run montaj ingest-rec so t is video time",
            )
        )
        return None
    return t


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
    events: _Events | None = None,
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
        if is_event_marker(text):
            compiled = compile_event_marker(text, path, errors)
            if compiled is not None and fps > 0:
                stem, sel = compiled
                book = events if events is not None else _Events(_assets_dir(spec, spec_dir), errors)
                rows = book.get(stem, path)
                if rows is not None:
                    if not rows:
                        errors.append(_line(path, f"no events in {stem}", "have: none"))
                    else:
                        t = _event_seconds(match_event(rows, sel), stem, path, errors)
                        if t is not None:
                            resolved[name] = t * fps
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


def _check_backdrop(path: str, bd: TextBackdrop, errors: list[str]) -> None:
    if bd.type not in BACKDROP_TYPES:
        errors.append(_line(f"{path}.type", f"unknown {bd.type!r}", "use gradient or solid"))
    if not math.isfinite(bd.strength) or bd.strength < 0 or bd.strength > 1:
        errors.append(
            _line(
                f"{path}.strength",
                f"must be from 0 to 1, got {_fmt_num(bd.strength)}",
                "use a fraction",
            )
        )
    if bd.place not in BACKDROP_PLACES:
        errors.append(_line(f"{path}.place", f"unknown {bd.place!r}", "use bottom, top or band"))


def _choices(names: tuple[str, ...]) -> str:
    return " or ".join(names)


def _whole_number(v: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v == int(v)


def _check_clip_range(
    spec: Spec,
    spec_dir: Path,
    i: int,
    sh: Shot,
    bpm: float | None,
    errors: list[str],
) -> None:
    """Reject a range that runs past the source, or a file that is not a video."""
    from montaj.encode import EncodeError, probe
    from montaj.project import resolve_clip

    if spec.video.fps <= 0 or sh.clip is None or sh.range is None:
        return
    try:
        path = resolve_clip(spec_dir, spec_dir / spec.assets, sh.clip)
    except FileNotFoundError:
        return
    try:
        info = probe(path, log=spec_dir / "build" / "spec.log")
    except EncodeError:
        errors.append(_line(f"shots[{i}].clip", f"cannot read {sh.clip!r}", "file is not a video"))
        return
    # probe() falls back to the container duration, so an audio-only file needs its own check.
    if info.width <= 0 or info.height <= 0:
        errors.append(_line(f"shots[{i}].clip", f"{sh.clip!r} has no video stream", "file is not a video"))
        return
    if info.duration <= 0:
        errors.append(
            _line(f"shots[{i}].clip", f"cannot read duration of {sh.clip!r}", "file is not a video")
        )
        return
    bag: list[str] = []
    start_f = _dur_frames(sh.range[0], spec.video.fps, bpm, f"shots[{i}].range[0]", bag)
    if start_f is not None and start_f / spec.video.fps >= info.duration - 1e-6:
        errors.append(
            _line(
                f"shots[{i}].range",
                f"starts at {sh.range[0]}, clip is {info.duration:.3f}s",
                "start inside the clip",
            )
        )
        return
    end_f = _dur_frames(sh.range[1], spec.video.fps, bpm, f"shots[{i}].range[1]", bag)
    if end_f is None:
        return
    # Container duration is often a frame short of n/fps. One frame of slack.
    if end_f / spec.video.fps > info.duration + 1.0 / spec.video.fps:
        errors.append(
            _line(
                f"shots[{i}].range",
                f"ends at {sh.range[1]}, clip is {info.duration:.3f}s",
                "shorten the range",
            )
        )


def _check_lut_look(look: LutLook, errors: list[str], spec_dir: Path | None) -> None:
    from montaj.cube import load_cube

    for name in ("strength", "vignette", "grain"):
        v = getattr(look, name)
        if not math.isfinite(v) or not 0.0 <= v <= 1.0:
            errors.append(_line(f"video.look.{name}", f"must be 0..1, got {_fmt_num(v)}", "use a number from 0 to 1"))
    if not look.lut.lower().endswith(".cube"):
        errors.append(_line("video.look.lut", f"{look.lut!r} is not a .cube file", "use a path ending in .cube"))
        return
    if spec_dir is None:
        return
    path = spec_dir / look.lut
    if not path.is_file():
        errors.append(_line("video.look.lut", f"{look.lut!r} not found", "path relative to the spec file"))
        return
    try:
        load_cube(path)
    except (OSError, ValueError) as e:
        errors.append(_line("video.look.lut", f"{look.lut!r}: {e}", "use a 3D .cube up to 65 points (or a 1D .cube)"))


def _check_finish(spec: Spec, errors: list[str], spec_dir: Path | None = None) -> None:
    look = spec.video.look
    if isinstance(look, LutLook):
        _check_lut_look(look, errors, spec_dir)
    elif look not in LOOK_NAMES:
        errors.append(_line("video.look", f"unknown {look!r}", f"use {', '.join(LOOK_NAMES)} or {{lut: file.cube}}"))
    intro = spec.video.intro
    if intro is not None and intro.type not in INTRO_TYPES:
        errors.append(
            _line("video.intro.type", f"unknown {intro.type!r}", f"use {_choices(INTRO_TYPES)}")
        )
    outro = spec.video.outro
    if outro is not None and outro.type not in OUTRO_TYPES:
        errors.append(
            _line("video.outro.type", f"unknown {outro.type!r}", f"use {_choices(OUTRO_TYPES)}")
        )


def _probe_clip(spec: Spec, spec_dir: Path | None, stem: str) -> tuple[int, int, float] | None:
    """(width, height, seconds) of a clip asset, or None when it cannot be probed."""
    if spec_dir is None:
        return None
    from montaj.encode import EncodeError, probe
    from montaj.project import resolve_clip

    try:
        info = probe(resolve_clip(spec_dir, spec_dir / spec.assets, stem))
    except (EncodeError, FileNotFoundError, OSError, ValueError):
        return None
    if info.width <= 0 or info.height <= 0:
        return None
    return info.width, info.height, info.duration


def _finite_point(v: tuple[float, float]) -> bool:
    return all(math.isfinite(c) for c in v)


def _check_clip_shot(
    spec: Spec,
    i: int,
    sh: Shot,
    hold: int | None,
    parsed: dict[str, int],
    times: dict[str, float],
    spec_dir: Path | None,
    errors: list[str],
    clip_in_s: float | None = None,
) -> None:
    """crop / focus / zoom / camera / speed / freeze on a clip shot. `clip_in_s` is an at_event pick."""
    from montaj.clipmap import source_index

    at = f"shots[{i}]"
    n_err = len(errors)
    if sh.camera is not None and (sh.focus is not None or sh.zoom is not None):
        errors.append(_line(at, "camera and focus/zoom are mutually exclusive", "put them in camera keys"))
    if sh.speed is not None and sh.range is not None:
        errors.append(_line(at, "speed and range are mutually exclusive", "use clip_in and hold with speed"))
    if sh.zoom is not None and (not math.isfinite(sh.zoom) or sh.zoom < 1):
        errors.append(_line(f"{at}.zoom", f"clip zoom is a multiple of cover, got {_fmt_num(sh.zoom)}", "use >= 1"))
    if sh.focus is not None and not _finite_point(sh.focus):
        errors.append(_line(f"{at}.focus", "must be finite", "use [x, y] in source px"))
    if sh.crop is not None:
        x0, y0, x1, y1 = sh.crop
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            errors.append(_line(f"{at}.crop", f"{list(sh.crop)} is not a box", "use [x0, y0, x1, y1] with x0 < x1, y0 < y1, all >= 0"))
    points = [(f"{at}.focus", sh.focus)] if sh.focus is not None else []
    if sh.camera is not None:
        if not sh.camera:
            errors.append(_line(f"{at}.camera", "need at least one key", "add {t: 0s, zoom: 1.0}"))
        last: int | None = None
        for k, ck in enumerate(sh.camera):
            path = f"{at}.camera[{k}]"
            if ck.ease not in EASE:
                errors.append(_line(f"{path}.ease", f"unknown {ck.ease!r}", EASE_HINT))
            if ck.zoom is not None and (not math.isfinite(ck.zoom) or ck.zoom < 1):
                errors.append(_line(f"{path}.zoom", f"zoom is a multiple of cover, got {_fmt_num(ck.zoom)}", "use >= 1"))
            if ck.focus is not None:
                if _finite_point(ck.focus):
                    points.append((f"{path}.focus", ck.focus))
                else:
                    errors.append(_line(f"{path}.focus", "must be finite", "use [x, y] in source px"))
            t = parsed.get(f"{path}.t")
            if t is not None:
                if last is not None and t <= last:
                    errors.append(_line(f"{path}.t", "keys must be in increasing time", "sort the keys"))
                last = t
    if isinstance(sh.speed, list):
        if not sh.speed:
            errors.append(_line(f"{at}.speed", "need at least one key", "add {t: 0s, v: 1}"))
        last = None
        for k, sk in enumerate(sh.speed):
            path = f"{at}.speed[{k}]"
            if sk.ease not in EASE:
                errors.append(_line(f"{path}.ease", f"unknown {sk.ease!r}", EASE_HINT))
            if not math.isfinite(sk.v) or sk.v <= 0:
                errors.append(_line(f"{path}.v", f"must be finite and > 0, got {_fmt_num(sk.v)}", "reverse playback is not supported"))
            t = parsed.get(f"{path}.t")
            if t is not None:
                if last is not None and t <= last:
                    errors.append(_line(f"{path}.t", "keys must be in increasing time", "sort the keys"))
                last = t
    elif sh.speed is not None and (not math.isfinite(sh.speed) or sh.speed <= 0):
        errors.append(_line(f"{at}.speed", f"must be finite and > 0, got {_fmt_num(sh.speed)}", "reverse playback is not supported"))
    freeze = parsed.get(f"{at}.freeze")
    if freeze is not None and hold is not None and freeze >= hold:
        errors.append(_line(f"{at}.freeze", f"{freeze}f is at or after the end of the {hold}f hold", "use a time inside the hold"))
    if len(errors) > n_err or not (sh.crop or sh.speed is not None or points):
        return
    # Source-dependent checks: crop and focus inside the frame, and speed not running off the end.
    info = _probe_clip(spec, spec_dir, sh.clip)
    if info is None:
        return
    w, h, seconds = info
    if sh.crop is not None and (sh.crop[2] > w or sh.crop[3] > h):
        errors.append(_line(f"{at}.crop", f"{list(sh.crop)} runs outside the {w}x{h} source", "keep the box inside the frame"))
        return
    box = sh.crop or (0, 0, w, h)
    for path, (px, py) in points:
        if not (box[0] <= px <= box[2] and box[1] <= py <= box[3]):
            errors.append(_line(path, f"[{_fmt_num(px)}, {_fmt_num(py)}] is outside the crop {list(box)}", "focus is in source px"))
    fps = spec.video.fps
    if sh.speed is None or hold is None or hold <= 0 or fps <= 0:
        return
    if isinstance(sh.speed, list):
        speed = [(parsed.get(f"{at}.speed[{k}].t", 0), sk.v, sk.ease) for k, sk in enumerate(sh.speed)]
    else:
        speed = float(sh.speed)
    table = source_index(speed, freeze, hold)
    assert table is not None
    clip_in = clip_in_s if clip_in_s is not None else times.get(f"{at}.clip_in", 0.0)
    needs = (clip_in + table[-1] + 1) / fps
    if needs > seconds + 1.5 / fps:
        errors.append(
            _line(at, f"clip ends at {seconds:.1f}s of source, needs {needs:.1f}s", "shorten hold, lower speed or clip_in")
        )


def _check_ease(path: str, ease: str, errors: list[str]) -> None:
    if ease not in EASE:
        errors.append(_line(path, f"unknown {ease!r}", EASE_HINT))


def _validate_sequence(
    seq: SequenceSpec,
    i: int,
    fps: int,
    bpm: float | None,
    parsed: dict[str, int],
    errors: list[str],
) -> None:
    path = f"shots[{i}].sequence.every"
    raw = seq.every
    if isinstance(raw, str):
        matched = _DUR.match(raw.strip())
        if matched and matched.group(2) == "b":
            errors.append(_line(path, "beats are not allowed", "use 12f or 0.4s"))
        else:
            n = parse_dur(raw, fps, bpm, path, errors)
            if n is not None and n < 1:
                errors.append(_line(path, f"must be at least 1 frame, got {n}f", "use 1f or longer"))
            elif n is not None:
                parsed[path] = n
    elif isinstance(raw, int) and not isinstance(raw, bool):
        if raw < 1:
            errors.append(_line(path, f"must be at least 1 frame, got {raw}f", "use 1f or longer"))
        else:
            parsed[path] = raw
    items_path = f"shots[{i}].sequence.items"
    if len(seq.items) < 2:
        errors.append(_line(items_path, "need at least two items", "add a photo or #rrggbb"))
    for j, item in enumerate(seq.items):
        ip = f"{items_path}[{j}]"
        has_photo = item.photo is not None
        has_color = item.color is not None
        if has_photo and has_color:
            errors.append(_line(ip, "photo and color are mutually exclusive", "set one"))
        elif not has_photo and not has_color:
            errors.append(_line(ip, "need a photo or #rrggbb", "set photo: or color:"))
        if has_color and (not isinstance(item.color, str) or not _HEX.match(item.color)):
            errors.append(_line(f"{ip}.color", f"unknown colour {item.color!r}", "use #rrggbb"))


def validate_spec(
    spec: Spec, errors: list[str], spec_dir: Path | None = None
) -> tuple[dict[str, int], dict[str, float], dict[str, float], float | None, float, dict[int, str]]:
    """Fill `errors`; return (rounded frames, float frames, markers, bpm, beat0 frames, clip_in).

    `clip_in` is the duration string `at_event` resolved for each shot index. The
    authoring spec keeps `at_event`; `resolve` copies this onto the shot the
    renderer seeks, so the segment hash sees the picked time.
    """
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

    events = _Events(_assets_dir(spec, spec_dir), errors)
    markers = resolve_markers(
        spec,
        spec_dir,
        fps,
        bpm,
        errors,
        beat0=beat0,
        markers_json=markers_json,
        markers_loaded=markers_loaded,
        events=events,
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

    clip_ins: dict[int, str] = {}
    for i, sh in enumerate(spec.shots):
        h = holds[i]
        has_p = sh.photo is not None
        has_w = sh.wall is not None
        has_c = sh.clip is not None
        has_b = sh.blank
        has_s = sh.sequence is not None
        has_card = sh.card is not None
        n_src = has_p + has_w + has_c + has_b + has_s + has_card
        if n_src > 1:
            errors.append(
                _line(
                    f"shots[{i}]",
                    "photo, wall, clip, blank, sequence and card are mutually exclusive",
                    "remove extra",
                )
            )
        elif n_src == 0:
            errors.append(_line(f"shots[{i}]", "need photo, wall, clip, blank, sequence or card", "add one"))
        if sh.card is not None:
            _validate_card(sh.card, i, fps, bpm, errors)
        if not has_p:
            for name in ("drift", "frame", "morph", "tone", "pulse"):
                if getattr(sh, name) is not None:
                    errors.append(_line(f"shots[{i}].{name}", f"{name} needs photo", "add photo:"))
        if sh.clip_in is not None and not has_c:
            errors.append(_line(f"shots[{i}].clip_in", "clip_in needs clip", "add clip:"))
        if sh.range is not None and not has_c:
            errors.append(_line(f"shots[{i}].range", "range needs clip", "add clip:"))
        if sh.range is not None and sh.clip_in is not None:
            errors.append(
                _line(f"shots[{i}]", "range and clip_in are mutually exclusive", "range is [start, end]")
            )
        if (
            sh.range is not None
            and has_c
            and sh.clip_in is None
            and sh.hold is None
            and sh.until is None
            and h is not None
            and spec_dir is not None
        ):
            _check_clip_range(spec, spec_dir, i, sh, bpm, errors)
        if sh.sequence is not None:
            _validate_sequence(sh.sequence, i, fps, bpm, parsed, errors)
            for name in ("crop", "focus", "zoom"):
                if getattr(sh, name) is not None:
                    errors.append(_line(f"shots[{i}].{name}", "not used on a sequence", "remove it"))
        if sh.at_event is not None:
            if not has_c:
                errors.append(_line(f"shots[{i}].at_event", "at_event needs clip", "add clip:"))
            elif sh.range is not None:
                errors.append(
                    _line(f"shots[{i}]", "range and at_event are mutually exclusive", "remove one")
                )
            elif sh.clip_in is not None:
                errors.append(
                    _line(f"shots[{i}]", "clip_in and at_event are mutually exclusive", "remove one")
                )
            else:
                got = _clip_in_from_event(sh, i, fps, times, events, errors)
                if got is not None:
                    clip_ins[i] = got
        if sh.pre is not None and sh.at_event is None:
            errors.append(_line(f"shots[{i}].pre", "pre needs at_event", "remove pre or add at_event"))
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
        if has_c:
            picked = clip_ins.get(i)
            _check_clip_shot(
                spec, i, sh, h, parsed, times, spec_dir, errors,
                None if picked is None else float(picked[:-1]),  # format_seconds: "<n>s"
            )
        else:
            for name in ("camera", "speed", "freeze"):
                if getattr(sh, name) is not None:
                    errors.append(_line(f"shots[{i}].{name}", f"{name} needs clip", "add clip:"))
        tr = sh.in_
        if tr is None:
            continue
        if tr.type not in TRANS_TYPES:
            errors.append(
                _line(f"shots[{i}].in.type", f"unknown {tr.type!r}", TRANS_HINT)
            )
            continue
        if tr.zoom is not None and tr.type != "wall-pullback":
            errors.append(
                _line(f"shots[{i}].in.zoom", "zoom is for wall-pullback", "remove zoom")
            )
        if tr.type == "wall-pullback" and tr.zoom is not None:
            if not math.isfinite(tr.zoom) or tr.zoom <= 1:
                errors.append(
                    _line(
                        f"shots[{i}].in.zoom",
                        f"must be finite and > 1, got {_fmt_num(tr.zoom)}",
                        "use a zoom above 1, for example 3.2",
                    )
                )
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
                _line(f"shots[{i}].in", "no previous shot to transition from", "remove in or use type: fade")
            )
        if tr.type in INCOMING_TRANS:
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
    _check_finish(spec, errors, spec_dir)
    if spec.check is not None and not 1 <= spec.check.sample <= 30:
        errors.append(_line("check.sample", f"must be 1 to 30, got {spec.check.sample}", "use sample: 3"))

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
        if fx.burst is not None and not _whole_number(fx.burst):
            errors.append(
                _line(
                    f"fx[{i}].burst",
                    f"need a whole number, got {_fmt_num(fx.burst)}",
                    "use an integer",
                )
            )
        if fx.burst is not None and fx.pos is None:
            errors.append(_line(f"fx[{i}].pos", "required for burst", "set pos: [x, y]"))
        at_f = _parse_time(fx.at, markers, fps, bpm, f"fx[{i}].at", errors, beat0=beat0)
        if at_f is not None:
            times[f"fx[{i}].at"] = at_f
            parsed[f"fx[{i}].at"] = round_frame(at_f)

    _validate_text(spec, markers, fps, bpm, times, errors, beat0=beat0)
    _validate_tracks(spec, markers, fps, bpm, times, errors, beat0=beat0)
    _validate_check(spec, markers, fps, bpm, times, errors, beat0=beat0)
    return parsed, times, markers, bpm, beat0, clip_ins


def _card_duration(
    raw: object, fps: int, bpm: float | None, path: str, errors: list[str]
) -> float | None:
    """A card line clock is a duration from the shot start, not a film marker."""
    if isinstance(raw, str) and _DUR.match(raw.strip()):
        return _dur_frames(raw, fps, bpm, path, errors)
    shown = raw if isinstance(raw, str) else repr(raw)
    bare = isinstance(raw, (int, float)) or (isinstance(raw, str) and _BARE.match(raw.strip()))
    if bare:
        errors.append(_line(path, "bare number", "time from this shot's start: 0s, 12f or 1b"))
    else:
        errors.append(
            _line(path, f"invalid duration {shown!r}", "time from this shot's start: 0s, 12f or 1b")
        )
    return None


def _check_text_line(path: str, line: TextLine, errors: list[str]) -> None:
    if line.style not in TEXT_STYLES:
        errors.append(_line(f"{path}.style", f"unknown {line.style!r}", "use serif, script, caps or deva"))
    if line.italic and line.style != "serif":
        errors.append(_line(f"{path}.italic", "italic is serif-only", "set style: serif"))
    if line.by not in TEXT_BY:
        errors.append(_line(f"{path}.by", f"unknown {line.by!r}", "use word or char"))
    if line.reveal not in TEXT_REVEAL:
        errors.append(_line(f"{path}.reveal", f"unknown {line.reveal!r}", "use rise or pop"))
    _check_color(f"{path}.color", line.color, errors)
    for name in ("y", "size", "weight", "tracking", "line_height", "rise", "indent"):
        v = getattr(line, name)
        if v is None:
            continue
        positive = name in ("size", "weight", "line_height")
        if not math.isfinite(v) or (positive and v <= 0):
            need = "finite and > 0" if positive else "finite"
            errors.append(
                _line(f"{path}.{name}", f"must be {need}, got {_fmt_num(v)}", f"use a design-px {name}")
            )


def _validate_card(
    card: CardSpec, index: int, fps: int, bpm: float | None, errors: list[str]
) -> None:
    path = f"shots[{index}].card"
    if not isinstance(card.color, str) or not _HEX.match(card.color):
        errors.append(_line(f"{path}.color", f"expected #rrggbb, got {card.color!r}", "use #112233"))
    if not card.text:
        errors.append(_line(f"{path}.text", "need at least one line", "add a line"))
    for j, line in enumerate(card.text):
        p = f"{path}.text[{j}]"
        _check_text_line(p, line, errors)
        _card_duration(line.at, fps, bpm, f"{p}.at", errors)
        if line.stagger is not None:
            _card_duration(line.stagger, fps, bpm, f"{p}.stagger", errors)
        if line.dur is not None:
            _card_duration(line.dur, fps, bpm, f"{p}.dur", errors)
        if line.sweep:
            for k, raw in enumerate(line.sweep):
                _card_duration(raw, fps, bpm, f"{p}.sweep[{k}]", errors)


def _sec_label(frames: float, fps: int) -> str:
    seconds = frames / float(fps) if fps > 0 else frames
    text = f"{seconds:.4f}".rstrip("0").rstrip(".")
    return f"{text}s"


def _validate_check(
    spec: Spec,
    markers: dict[str, float],
    fps: int,
    bpm: float | None,
    times: dict[str, float],
    errors: list[str],
    *,
    beat0: float = 0.0,
) -> None:
    """Resolve `check.expect` spans and the optional loudness target.

    A span that never shows up in the render is not an error here: `check` reports it
    as INFO, so a stale annotation survives an edit instead of failing validate.
    """
    if spec.check is None:
        return
    loud = spec.check.loudness
    if loud is not None:
        if not math.isfinite(loud.i):
            errors.append(
                _line("check.loudness.i", f"must be finite, got {_fmt_num(loud.i)}", "use i: -14")
            )
        if not math.isfinite(loud.tol) or loud.tol < 0:
            errors.append(
                _line(
                    "check.loudness.tol",
                    f"must be >= 0, got {_fmt_num(loud.tol)}",
                    "use tol: 1",
                )
            )
    for i, item in enumerate(spec.check.expect):
        kind = "black" if item.black is not None else "frozen"
        pair = item.black if item.black is not None else item.frozen
        if pair is None:
            continue
        ends: list[float | None] = []
        for j, raw in enumerate(pair):
            val = _parse_time(
                raw, markers, fps, bpm, f"check.expect[{i}].{kind}[{j}]", errors, beat0=beat0
            )
            ends.append(val)
        if ends[0] is None or ends[1] is None:
            continue
        start, end = ends[0], ends[1]
        if end <= start:
            errors.append(
                _line(
                    f"check.expect[{i}].{kind}",
                    f"ends at or before the start ({_sec_label(start, fps)} to {_sec_label(end, fps)})",
                    "end must be later than start",
                )
            )
            continue
        times[f"check.expect[{i}].start"] = start
        times[f"check.expect[{i}].end"] = end


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
        if block.backdrop is not None:
            _check_backdrop(f"text[{i}].backdrop", block.backdrop, errors)
        if not block.lines:
            errors.append(_line(f"text[{i}].lines", "need at least one line", "add a line"))
        for j, line in enumerate(block.lines):
            p = f"text[{i}].lines[{j}]"
            _check_text_line(p, line, errors)
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
        if sub.size is not None and (not math.isfinite(sub.size) or sub.size <= 0):
            errors.append(
                _line(
                    f"subs[{i}].size",
                    f"must be finite and > 0, got {_fmt_num(sub.size)}",
                    "use a design-px size",
                )
            )
        _check_color(f"subs[{i}].color", sub.color, errors)


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
    if kind in INCOMING_TRANS:
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
        if tr is None or tr.type not in (*INCOMING_TRANS, "whip"):
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
