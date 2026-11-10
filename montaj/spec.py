"""YAML spec schema: pydantic models, duration parsing, SpecError."""

from __future__ import annotations

import ast
import contextlib
import difflib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal, get_args

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    StrictBool,
    StrictInt,
    Tag,
    ValidationError,
    WrapSerializer,
    model_validator,
)

from montaj import expr as xp
from montaj.easing import NEW_EASES, parse_ease
from montaj.expr import ExprError
from montaj.grade import GRADE_TAGS, ColorSpec, Grade, GradeRef, check_grades
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
TRANS_TYPES = (
    "cut", "fade", "swirl", "whip", "glitch", "zoom", "dip", "push", "wall-pullback", "burn-white", "custom",
)
# These occupy [start, start+dur) of the incoming shot. Whip is centred on the cut.
INCOMING_TRANS = ("fade", "swirl", "glitch", "zoom", "dip", "push", "wall-pullback", "burn-white", "custom")
TRANS_HINT = "use cut, fade, swirl, whip, glitch, zoom, dip, push, wall-pullback, burn-white or custom"
TRANS_STRENGTH_MAX = 4.0
# Starting magnification of a wall pull-back. Materialised onto the shot so the hash sees it.
WALL_PULLBACK_ZOOM = 3.2
LOOK_NAMES = ("warm-film", "golden-film", "clean", "cool-tech")
INTRO_TYPES = ("white-lift",)
OUTRO_TYPES = ("glow", "fade")
TEXT_STYLES = ("serif", "script", "caps", "deva", "sans", "mono")
TEXT_BY = ("word", "char")
TEXT_REVEAL = ("rise", "pop", "type")
TEXT_BOX = ("none", "chip")
TEXT_ALIGN = ("center", "left", "right")
TEXT_SIDES = ("left", "right")
# Weights that ship as font files; a missing weight would silently fall back to the nearest.
TEXT_WEIGHTS = {"sans": (400, 600, 700), "mono": (400, 700)}
TEXT_COLORS = ("cream", "gold", "ink")
OVERLAY_SHAPES = ("ring", "brackets", "line", "waveform")
WAVEFORM_BARS = 32  # default bar count; each bar is one slice of that frame's audio
WAVEFORM_BARS_MAX = 128
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

# Analysis sugar (`assets/analysis.json`): resolved to numbers before layout (`resolve_sugar`).
_FOCUS_REF = re.compile(r"^(?:face(?::([1-9]\d*))?|subject)$")
_SPEECH_REF = re.compile(r"^speech:([1-9]\d*)$")
_SUGAR_TAGS = ("focus_xy", "focus_ref", "range_span", "range_ref")  # union tags, not spec keys


def _focus_ref(v: str) -> str:
    if not _FOCUS_REF.match(v):
        raise ValueError(f"unknown focus {v!r}, want [x, y], face, face:N or subject")
    return v


def _speech_ref(v: str) -> str:
    if not _SPEECH_REF.match(v):
        raise ValueError(f"unknown range {v!r}, want [start, end] or speech:N")
    return v


# A string is sugar, anything else the number form, so a bad list keeps its plain error path.
Focus = Annotated[
    Annotated[tuple[float, float], Tag("focus_xy")] | Annotated[str, AfterValidator(_focus_ref), Tag("focus_ref")],
    Discriminator(lambda v: "focus_ref" if isinstance(v, str) else "focus_xy"),
]
ClipRange = Annotated[
    Annotated[tuple[DurationStr, DurationStr], Tag("range_span")]
    | Annotated[str, AfterValidator(_speech_ref), Tag("range_ref")],
    Discriminator(lambda v: "range_ref" if isinstance(v, str) else "range_span"),
]


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _none_value(v: object) -> bool:
    return v is None


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
    # #43: `{working: linear}` composites in linear light. Absent stays out of the hash.
    color: ColorSpec | None = Field(default=None, exclude_if=_none_value)


class Transition(Frozen):
    type: str
    dur: DurationStr | None = None
    center: tuple[float, float] | None = None
    axis: str = "x"
    blur: float = 0.0
    # Absent stays out of model_dump so films without these keep their segment hash.
    strength: RealNum | None = Field(default=None, exclude_if=lambda v: v is None)
    seed: Annotated[int, Field(strict=True)] | None = Field(default=None, exclude_if=lambda v: v is None)
    color: str | None = Field(default=None, exclude_if=lambda v: v is None)
    zoom: RealNum | None = Field(default=None, exclude_if=lambda v: v is None)
    # `type: custom`: the .py file (relative to montaj.yaml) and what its draw(ctx) reads as ctx.params.
    src: str | None = Field(default=None, exclude_if=lambda v: v is None)
    params: dict[str, Any] | None = Field(default=None, exclude_if=lambda v: v is None)


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
    focus: Focus | None = None
    ease: str = "smooth"


class SpeedKey(Frozen):
    t: DurationStr | int
    v: RealNum
    ease: str = "smooth"


class TextStroke(Frozen):
    """An outline outside the fill. `width` design px."""

    width: RealNum
    color: str = "#000000"


class TextShadow(Frozen):
    """One drop shadow; replaces the style's own. Offsets and blur radius in design px."""

    x: RealNum = 0.0
    y: RealNum = 6.0
    blur: RealNum = 12.0
    color: str = "#000000"
    opacity: RealNum = 0.6


class TextEmphasis(Frozen):
    """How `*marked*` words differ from the rest of the line."""

    color: str = "#ffe600"
    scale: RealNum = 1.0
    # Motion (#22): the marked words' entry type and impact shake. Unset they stay out of the dump.
    in_: str | None = Field(default=None, alias="in", exclude_if=lambda v: v is None)
    shake: RealNum | None = Field(default=None, exclude_if=lambda v: v is None)


def _as_typed(v: object) -> object:
    """`in: pop` is `in: {type: pop}`."""
    return {"type": v} if isinstance(v, str) else v


class TextIn(Frozen):
    """How each unit of a gpu line enters. Durations and `by` override the line's own."""

    type: str
    by: str | None = None
    dur: DurationStr | None = None
    stagger: DurationStr | None = None
    overshoot: RealNum | None = None
    from_: str | None = Field(default=None, alias="from")
    distance: RealNum | None = None


class TextOut(Frozen):
    """How each unit leaves; the last unit's exit ends at the block's `to`."""

    type: str
    dur: DurationStr | None = None
    stagger: DurationStr | None = None
    to: str | None = None
    distance: RealNum | None = None


class TextIdle(Frozen):
    """A loop while the unit is at rest."""

    type: str
    amount: RealNum | None = None


class TextBar(Frozen):
    """A rounded bar hugging one word of a gpu line (#23 `highlight`). `from` slides it from another word."""

    word: StrictInt
    from_: StrictInt | None = Field(default=None, alias="from", exclude_if=_none_value)
    color: str = "#ffe600"
    pad: RealNum = 14.0


def _none(v: object) -> bool:
    return v is None


class TextLine(Frozen):
    text: str
    # Required unless `pos` is set (checked in `_check_text_line`).
    y: float | None = None
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
    # A mapping is a custom shadow (gpu engine); a bool switches the style's own.
    shadow: bool | TextShadow = True
    indent: float = 0.0
    # Added after the first release: left out of the dump at their default so older films keep their hashes.
    align: str = Field(default="center", exclude_if=lambda v: v == "center")
    box: str = Field(default="none", exclude_if=lambda v: v == "none")
    box_color: str | None = Field(default=None, exclude_if=lambda v: v is None)
    dot: str | None = Field(default=None, exclude_if=lambda v: v is None)
    cursor: bool = Field(default=False, exclude_if=lambda v: v is False)
    # None = the style's engine (render/glyphs.STYLE_ENGINE). "gpu" shapes with HarfBuzz at resolve time.
    engine: str | None = Field(default=None, exclude_if=lambda v: v is None)
    # Fit and wrap are resolved into size, text and y at resolve time (timeline._resolve_texts); unset they stay out of the dump.
    fit: bool = Field(default=False, exclude_if=lambda v: v is False)
    wrap: str | None = Field(default=None, exclude_if=lambda v: v is None)
    max_width: float | None = Field(default=None, exclude_if=lambda v: v is None)
    max_lines: int | None = Field(default=None, exclude_if=lambda v: v is None)
    # Text styling (#21). Each one is drawn by the gpu engine and selects it when `engine` is unset.
    font: str | None = Field(default=None, exclude_if=_none)
    case: str | None = Field(default=None, exclude_if=_none)
    emphasis: TextEmphasis | None = Field(default=None, exclude_if=_none)
    stroke: TextStroke | None = Field(default=None, exclude_if=_none)
    glow: RealNum | None = Field(default=None, exclude_if=_none)
    pos: tuple[RealNum, RealNum] | None = Field(default=None, exclude_if=_none)
    rotate: RealNum | None = Field(default=None, exclude_if=_none)
    # Text motion (#22), gpu engine. Resolved into `timeline.ResolvedLine.motion`.
    in_: Annotated[TextIn | None, BeforeValidator(_as_typed)] = Field(default=None, alias="in", exclude_if=_none)
    out: Annotated[TextOut | None, BeforeValidator(_as_typed)] = Field(default=None, exclude_if=_none)
    idle: Annotated[TextIdle | None, BeforeValidator(_as_typed)] = Field(default=None, exclude_if=_none)
    # A bar behind one word (#23); drawn by the gpu engine. Unset it stays out of the dump.
    bar: TextBar | None = Field(default=None, exclude_if=_none)


def gpu_fields(line: TextLine) -> list[str]:
    """The fields of `line` only the gpu engine draws. Any of them makes the line a gpu line."""
    out = [n for n in ("font", "case", "emphasis", "stroke", "glow", "pos", "rotate", "in_", "out", "idle", "bar")
           if getattr(line, n) is not None]
    out = ["in" if n == "in_" else n for n in out]
    if isinstance(line.shadow, TextShadow):
        out.append("shadow")
    if "*" in line.text:  # `*emphasis*` markup (or an escaped `\*`)
        out.append("text")
    return out


def line_engine(line: TextLine) -> str:
    """`engine` when set; else gpu when a gpu-only field is used; else the style's engine."""
    from montaj.render.glyphs import STYLE_ENGINE

    if line.engine is not None:
        return line.engine
    if gpu_fields(line):
        return "gpu"
    return STYLE_ENGINE.get(line.style, "chromium")


class LowerThird(Frozen):
    title: str
    subtitle: str | None = None
    at: TimeExpr | None = None
    side: str = "left"
    y: float = 1500.0  # 9:16 value; `Spec` rescales an unset y to the frame height


def _expand_lower_third(data: object) -> object:
    """`lower_third` is sugar: it becomes a title chip and a subtitle line before anything else sees it."""
    if not isinstance(data, dict) or data.get("lower_third") is None:
        return data
    data = dict(data)
    raw = data.pop("lower_third")
    lt = LowerThird.model_validate(raw)
    if lt.side not in TEXT_SIDES:
        raise ValueError(f"lower_third.side unknown {lt.side!r}; use {_choices(TEXT_SIDES)}")
    if not math.isfinite(lt.y):
        raise ValueError(f"lower_third.y must be finite, got {_fmt_num(lt.y)}; use a design-px y")
    if data.get("lines"):
        raise ValueError("lower_third replaces lines; use one or the other")
    at = lt.at if lt.at is not None else data.get("from")
    lines = [{"text": lt.title, "style": "sans", "weight": 700, "size": 54, "y": lt.y, "at": at,
              "align": lt.side, "box": "chip", "dot": "gold"}]
    if lt.subtitle:
        lines.append({"text": lt.subtitle, "style": "sans", "size": 32, "y": lt.y + 110, "at": at,
                      "align": lt.side, "color": "cream"})
    data["lines"] = lines
    data.setdefault("fade_in", "0.4s")
    data.setdefault("fade_out", "0.4s")
    return data


LAYER_BLENDS = ("normal", "screen", "add", "multiply", "overlay", "soft-light", "difference")
LAYER_FITS = ("cover", "contain")
LAYER_ANIMS = ("fade", "pop")
MATTES = ("subject",)  # #36: the person matte `montaj ingest --matte` computes


EFFECT_TYPES = ("shake", "blur", "bloom", "chroma", "trails")
# The built-in knobs each effect type reads besides amount, at and dur.
EFFECT_KNOBS = {"shake": ("seed",), "blur": (), "bloom": ("threshold", "radius"), "chroma": (),
                "trails": ("frames", "decay")}


class EffectKey(Frozen):
    """One `amount` keyframe: `t` a duration from the shot start (whole frames after resolve)."""

    t: DurationStr | int
    v: RealNum


class Effect(Frozen):
    """`shots[].effects[]` / `layers[].effects[]`: a custom file (`custom`) or a built-in (`type`, #38).

    A custom file redraws the shot's picture (ctx.inputs[0]). A built-in reads `amount` (a number or
    keyframes) and the knobs `EFFECT_KNOBS` names for its type; `at` + `dur` make it a hit that decays to
    nothing. `at` is a film time; after resolve it, `dur` and keyframe `t` are whole frames from the
    shot start. Unset fields stay out of the dump, so a custom-only film keeps its segment hash.
    """

    custom: str | None = Field(default=None, exclude_if=_none)
    params: dict[str, Any] | None = Field(default=None, exclude_if=_none)
    type: str | None = Field(default=None, exclude_if=_none)
    amount: RealNum | list[EffectKey] | None = Field(default=None, exclude_if=_none)
    at: TimeExpr | int | None = Field(default=None, exclude_if=_none)
    dur: DurationStr | int | None = Field(default=None, exclude_if=_none)
    seed: StrictInt | None = Field(default=None, exclude_if=_none)
    threshold: RealNum | None = Field(default=None, exclude_if=_none)
    radius: RealNum | None = Field(default=None, exclude_if=_none)
    frames: StrictInt | None = Field(default=None, exclude_if=_none)
    decay: RealNum | None = Field(default=None, exclude_if=_none)



class LayerAnim(Frozen):
    """A layer's entrance or exit. `dur` becomes whole frames in `resolve`."""

    type: str
    dur: DurationStr | int


class LayerKey(Frozen):
    """A keyframe on a layer property (#35). `t` is the shot clock (whole frames after `resolve`); `v` is a
    number, or `[x, y]` for `pos`. `ease` shapes the move that arrives at this key; the first key's is unused."""

    t: DurationStr | int = "0f"
    v: RealNum | tuple[RealNum, RealNum] | tuple[RealNum, RealNum, RealNum, RealNum]
    ease: str = "smooth"
    # `beat` snaps the key to the nearest beat of the grid; `word:N` lands it on audio word N. `t` is then the
    # offset from the shot start (`beat`) or from the word. Folded into `t` by `resolve`, so it never reaches the hash.
    on: str | None = Field(default=None, exclude_if=_none)

    @model_validator(mode="before")
    @classmethod
    def _yaml_on(cls, data: Any) -> Any:
        """YAML 1.1 reads an unquoted `on:` key as the boolean True; it means the `on` field."""
        if isinstance(data, dict) and True in data and "on" not in data:
            data = {("on" if k is True else k): v for k, v in data.items()}
        return data


class LayerExpr(Frozen):
    """A layer property computed per frame (#52): `expr` is one expression (a number), or a list of them for
    `pos` (2) and `box` (4). See `montaj.expr`. `resolve` binds the film's constants into it, so the hash
    sees the bound text."""

    expr: str | list[str]


LAYER_KEYED = ("pos", "scale", "rotate", "opacity", "box", "radius", "shadow")
_KEY_ARITY = {"pos": 2, "box": 4}  # every other keyed property is a number


class Layer(Frozen):
    """One source composited over the layers before it (#34): source + placement + transform + opacity + blend.

    Unset fields stay out of the dump, so only what a film sets reaches its segment hash. Placement is a
    `box` (design px) or a `pos` (centre, natural size: one source px per design px); neither is the full frame.
    `LAYER_KEYED` properties are a value or a list of `LayerKey`s; the renderer evaluates them
    per frame (`render.layers.layer_state`).
    """

    clip: PhotoId | None = Field(default=None, exclude_if=_none)
    image: PhotoId | None = Field(default=None, exclude_if=_none)
    # `id` names a layer for `parent`, which inherits that layer's transform; `anchor` is the pivot's offset from
    # the layer's centre (design px) that scale and rotate act about.
    id: str | None = Field(default=None, exclude_if=_none)
    parent: str | None = Field(default=None, exclude_if=_none)
    anchor: tuple[RealNum, RealNum] | None = Field(default=None, exclude_if=_none)
    clip_in: DurationStr | None = Field(default=None, exclude_if=_none)
    focus: Focus | None = Field(default=None, exclude_if=_none)
    # Shot clock: frames from the shot start. Ints after resolve.
    from_: DurationStr | int | None = Field(default=None, alias="from", exclude_if=_none)
    to: DurationStr | int | None = Field(default=None, exclude_if=_none)
    box: tuple[RealNum, RealNum, RealNum, RealNum] | list[LayerKey] | LayerExpr | None = Field(default=None, exclude_if=_none)
    fit: str | None = Field(default=None, exclude_if=_none)
    pos: tuple[RealNum, RealNum] | list[LayerKey] | LayerExpr | None = Field(default=None, exclude_if=_none)
    scale: RealNum | list[LayerKey] | LayerExpr | None = Field(default=None, exclude_if=_none)
    rotate: RealNum | list[LayerKey] | LayerExpr | None = Field(default=None, exclude_if=_none)
    opacity: RealNum | list[LayerKey] | LayerExpr | None = Field(default=None, exclude_if=_none)
    blend: str | None = Field(default=None, exclude_if=_none)
    radius: RealNum | list[LayerKey] | LayerExpr | None = Field(default=None, exclude_if=_none)
    shadow: RealNum | list[LayerKey] | LayerExpr | None = Field(default=None, exclude_if=_none)
    in_: LayerAnim | None = Field(default=None, alias="in", exclude_if=_none)
    out: LayerAnim | None = Field(default=None, exclude_if=_none)
    # Built-in effects on this layer (#38): shake moves it, trails echo it, the rest filter its pixels.
    effects: list[Effect] | None = Field(default=None, exclude_if=_none)
    # `subject`: cut the layer to its source's matte (#36); it also becomes the plane `behind` text sits under.
    mask: str | None = Field(default=None, exclude_if=_none)


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
    range: ClipRange | None = None
    crop: CropBox | None = None
    focus: Focus | None = None
    zoom: float | None = None
    # Absent stays out of model_dump so films that never use these keep their segment hash.
    camera: list[CameraKey] | None = Field(default=None, exclude_if=lambda v: v is None)
    speed: RealNum | list[SpeedKey] | None = Field(default=None, exclude_if=lambda v: v is None)
    freeze: DurationStr | int | None = Field(default=None, exclude_if=lambda v: v is None)
    drift: Drift | None = None
    tone: Tone | None = None
    # #43 primary grade: inline, or a `grades:` name that `resolve_grades` replaces. Absent stays out.
    grade: GradeRef | None = Field(default=None, exclude_if=_none_value)
    frame: FrameCard | None = None
    morph: Morph | None = None
    pulse: str | None = None
    blank: bool = False
    # Absent stays out of model_dump so films without a sequence keep their segment hash.
    sequence: SequenceSpec | None = Field(default=None, exclude_if=lambda v: v is None)
    card: CardSpec | None = Field(default=None, exclude_if=lambda v: v is None)
    # Code-drawn shot (#37): a .py file whose draw(ctx) paints the frame; params reach ctx.params.
    custom: str | None = Field(default=None, exclude_if=lambda v: v is None)
    params: dict[str, Any] | None = Field(default=None, exclude_if=lambda v: v is None)
    effects: list[Effect] | None = Field(default=None, exclude_if=lambda v: v is None)
    # Back to front; replaces the single source (#34). Absent stays out of model_dump.
    layers: list[Layer] | None = Field(default=None, exclude_if=lambda v: v is None)
    # `subject` on a clip shot: its ingested matte is the plane `behind` text sits under (#36). Absent stays out.
    matte: str | None = Field(default=None, exclude_if=lambda v: v is None)
    in_: Transition | None = Field(default=None, alias="in")
    flash: float | None = None
    # Linked clip audio (#40): the clip's own sound follows this shot. Audio-only, so
    # `project._shot_payload` keeps them out of the segment hash; absent stays out of model_dump.
    audio: StrictBool | None = Field(default=None, exclude_if=lambda v: v is None)
    gain: str | None = Field(default=None, exclude_if=lambda v: v is None)
    audio_lead: str | None = Field(default=None, exclude_if=lambda v: v is None)
    audio_tail: str | None = Field(default=None, exclude_if=lambda v: v is None)
    # Transcript cuts (#39): authoring-only. `expand_cuts` turns the shot into ordinary `range` shots at
    # load, so these never reach layout, the renderer or the segment hash.
    cuts: Literal["auto"] | None = Field(default=None, exclude_if=lambda v: v is None)
    cuts_pad: tuple[DurationStr, DurationStr] | None = Field(default=None, exclude_if=lambda v: v is None)
    cuts_punch: Literal["alternate"] | None = Field(default=None, exclude_if=lambda v: v is None)
    drop: list[int] | None = Field(default=None, exclude_if=lambda v: v is None)
    id: str | None = Field(default=None, exclude_if=lambda v: v is None)


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
    # Draw the block inside the sub-frame average (#22). False stays out of the dump.
    motion_blur: bool = Field(default=False, exclude_if=lambda v: v is False)
    # Absent stays out of model_dump so a film with no scrim keeps its segment hash.
    backdrop: Annotated[TextBackdrop | None, BeforeValidator(_coerce_backdrop)] = Field(
        default=None, exclude_if=lambda v: v is None
    )
    # `subject`: drawn between the picture and the on-screen subject's matte (#36). Absent stays out.
    behind: str | None = Field(default=None, exclude_if=lambda v: v is None)
    lines: list[TextLine]

    @model_validator(mode="before")
    @classmethod
    def _lt(cls, data: object) -> object:
        return _expand_lower_third(data)


class Sub(Frozen):
    from_: TimeExpr = Field(alias="from")
    to: TimeExpr
    text: str
    size: RealNum | None = None
    color: str | None = None
    # `drop` removes trailing `.`/`,` from the text at load (#24). Unset leaves the text as written, so films
    # made before it keep their hashes; `validate` warns about a trailing mark on an unset sub.
    punctuation: Literal["drop", "keep"] | None = Field(default=None, exclude_if=_none)


class TrackKey(Frozen):
    t: TimeExpr
    v: float
    ease: str | None = None


class Tracks(Frozen):
    dust: list[TrackKey] = Field(default_factory=list)
    glow: list[TrackKey] = Field(default_factory=list)
    bars: list[TrackKey] = Field(default_factory=list)


class AudioEdit(Frozen):
    cut: tuple[TimeExpr, TimeExpr] | None = None
    loop: tuple[TimeExpr, TimeExpr] | None = None
    times: int | None = None


class AudioClip(Frozen):
    id: str | None = None
    src: str
    stream: int | None = None
    in_: DurationStr | None = Field(default=None, alias="in")
    out: DurationStr | None = None
    at: TimeExpr | list[TimeExpr]
    offset: DurationStr | None = None
    gain: str | None = None
    pan: RealNum | None = None


class AudioDuck(Frozen):
    under: list[str]
    depth: str = "-6dB"
    attack: DurationStr = "40ms"
    release: DurationStr = "200ms"


class Audio(Frozen):
    # Absent: no music bed. Linked clip audio (`audio: true` shots) still mixes over silence.
    track: str | None = Field(default=None, exclude_if=lambda v: v is None)
    markers: str | None = None
    fade_out: DurationStr | None = None
    loudnorm: bool = True
    edit: list[AudioEdit] = Field(default_factory=list)
    clips: list[AudioClip] = Field(default_factory=list)
    duck: AudioDuck | None = None
    # #41: `auto` adds transition sounds at layout (`emphasis.expand_emphasis`); both are consumed there, so only clips are hashed.
    sfx: Literal["auto"] | None = Field(default=None, exclude_if=lambda v: v is None)
    sfx_gain: str | None = Field(default=None, exclude_if=lambda v: v is None)


class Fx(Frozen):
    leak: float | None = None
    flash: float | None = None
    hit: float | None = None
    burst: RealNum | None = None
    rgb_split: RealNum | None = None
    dur: DurationStr | None = None
    at: TimeExpr
    dir: int | None = None
    pos: tuple[float, float] | None = None


class Overlay(Frozen):
    """Vector HUD shape drawn over the picture. Which fields apply depends on `shape`."""

    shape: str
    from_: TimeExpr = Field(alias="from")
    to: TimeExpr
    center: tuple[RealNum, RealNum] | None = None
    radius: RealNum | tuple[RealNum, RealNum] | None = None
    sweep: RealNum | tuple[RealNum, RealNum] | None = None
    box: tuple[RealNum, RealNum, RealNum, RealNum] | None = None
    arm: RealNum | None = None
    points: list[tuple[RealNum, RealNum]] | None = None
    draw: DurationStr | None = None
    width: RealNum = 3.0
    color: str = "#ffffff"
    opacity: RealNum = 1.0
    glow: RealNum = 0.0
    ease: str = "smooth"
    above_finish: bool = False
    # Absent on the other shapes, and left out of model_dump, so a ring film's dump stays what it was.
    source: str | None = Field(default=None, exclude_if=lambda v: v is None)
    # `true` would otherwise become 1, a legal bar count.
    bars: Annotated[int | None, BeforeValidator(_reject_bool)] = Field(
        default=None, exclude_if=lambda v: v is None
    )


class ExpectSpan(Frozen):
    """One designed black or frozen window. Times are film-start expressions, so markers work."""

    black: tuple[TimeExpr, TimeExpr] | None = None
    frozen: tuple[TimeExpr, TimeExpr] | None = None
    # A designed caption exception (#24): caption WARNs inside the span are dropped.
    caption: tuple[TimeExpr, TimeExpr] | None = Field(default=None, exclude_if=_none)
    why: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> ExpectSpan:
        kinds = [k for k in ("black", "frozen", "caption") if getattr(self, k) is not None]
        if not kinds:
            raise ValueError("need black, frozen or caption")
        if len(kinds) > 1:
            raise ValueError("set one of black, frozen, caption")
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


class PacingTarget(Frozen):
    """`check.pacing`: targets `check` judges the cut rhythm against. Unset fields are never judged."""

    max_shot: DurationStr | None = None
    min_per10s: RealNum | None = None
    hook_text_by: DurationStr | None = None


class SafeZones(Frozen):
    """`check.safe`: fractions of the frame a caption keeps clear of. Unset sides take the aspect's default."""

    top: RealNum | None = None
    bottom: RealNum | None = None
    left: RealNum | None = None
    right: RealNum | None = None


class CheckSpec(Frozen):
    safe: SafeZones | None = Field(default=None, exclude_if=_none)
    expect: list[ExpectSpan] = Field(default_factory=list)
    pacing: PacingTarget | None = None
    loudness: LoudnessTarget | None = None
    digits: Literal["none", "spec"] | None = None
    faces: Literal["none"] | None = None
    sample: StrictInt = 3  # strict: YAML `true` must not coerce to 1


class CaptionOverride(Frozen):
    """One sparse fix: `word` (index) with `text` / `emphasis`, or `chunk_break_before` (word index)."""

    word: StrictInt | None = None
    text: str | None = None
    emphasis: bool | None = None
    chunk_break_before: StrictInt | None = None


class Captions(Frozen):
    """Word timings in, styled caption chunks out. `captions.expand` turns this into `text` blocks."""

    source: str
    preset: str = "punch"
    words_per_chunk: tuple[StrictInt, StrictInt] = (1, 3)
    accent: str = "#ffe600"
    emphasis: str | list[StrictInt] = "auto"
    # `keep` leaves trailing `.` and `,` on the chunks; the default drops them (docs/CAPTIONS.md).
    punctuation: Literal["drop", "keep"] = "drop"
    y: RealNum | None = None
    offset: DurationStr | None = None
    overrides: list[CaptionOverride] = Field(default_factory=list)


class Emphasis(Frozen):
    """One word-anchored beat (#41): exactly one of `word` / `match` / `on`, then any of the actions."""

    word: StrictInt | None = Field(default=None, exclude_if=_none)
    match: str | None = Field(default=None, exclude_if=_none)
    on: str | None = Field(default=None, exclude_if=_none)
    zoom: RealNum | None = Field(default=None, exclude_if=_none)
    shake: RealNum | None = Field(default=None, exclude_if=_none)
    flash: RealNum | None = Field(default=None, exclude_if=_none)
    sfx: str | None = Field(default=None, exclude_if=_none)
    caption: Literal["accent"] | None = Field(default=None, exclude_if=_none)


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
    overlays: list[Overlay] = Field(default_factory=list)
    check: CheckSpec | None = None
    # Set by `expand_captions`: index of the first `text` block it made. Never dumped, so never hashed.
    caption_from: int | None = Field(default=None, exclude=True)
    captions: Captions | None = Field(default=None, exclude_if=_none_value)
    grades: dict[str, Grade] | None = Field(default=None, exclude_if=_none_value)
    # Authoring-only (#41): `expand_emphasis` turns it into effects, camera keys, fx, clips and caption overrides.
    emphasis: list[Emphasis] | None = Field(default=None, exclude_if=_none)

    @model_validator(mode="before")
    @classmethod
    def _lower_third_y(cls, data: object) -> object:
        """An unset `lower_third.y` is 1500 on a 1920-tall frame, the same fraction of other heights, and always
        leaves 210 design px below it for the chip and the subtitle line."""
        if not isinstance(data, dict) or not isinstance(data.get("text"), list):
            return data
        video = data.get("video")
        m = _WH.match(str(video.get("size", "")).strip()) if isinstance(video, dict) else None
        if not m or int(m[1]) <= 0:
            return data
        w, h = int(m[1]), int(m[2])
        design_h = 1080 * h / w
        y = 1500.0 if h * 1080 == 1920 * w else min(1500.0 * design_h / 1920.0, design_h - 210.0)
        blocks = [dict(b, lower_third={"y": y, **b["lower_third"]})
                  if isinstance(b, dict) and isinstance(b.get("lower_third"), dict) else b for b in data["text"]]
        return {**data, "text": blocks}


def _yaml_path(loc: tuple) -> str:
    parts: list[str] = []
    for x in loc:
        if x == "in_":
            x = "in"
        if x in ("look_name", "look_lut") or x in _SUGAR_TAGS or x in GRADE_TAGS:  # discriminator tags, not spec keys
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


def _models_in(ann: object) -> list[type[BaseModel]]:
    """Every model class named by a field annotation (through Optional, unions, lists, dicts)."""
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return [ann]
    return [m for arg in get_args(ann) for m in _models_in(arg)]


def _known_keys(loc: tuple) -> list[str]:
    """Field names and aliases legal beside the unknown key at `loc`: walk the models down from `Spec`."""
    models: list[type[BaseModel]] = [Spec]
    for part in loc[:-1]:
        if isinstance(part, int):
            continue
        nxt: list[type[BaseModel]] = []
        for m in models:
            for name, f in m.model_fields.items():
                if part in (name, f.alias):
                    nxt += _models_in(f.annotation)
        if not nxt:
            return []  # a union tag or an untyped mapping: no honest list
        models = nxt
    return sorted({k for m in models for name, f in m.model_fields.items() for k in (name, f.alias) if k})


def _unknown_key(loc: tuple) -> tuple[str, str]:
    """(problem, hint) for an unknown key: the nearest real key, else the keys that are legal there."""
    key = str(loc[-1])
    known = _known_keys(loc)
    near = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
    if near:
        return f"unknown key {key!r} — did you mean {near[0]!r}?", ""
    shown = ", ".join(known[:14]) + (", ..." if len(known) > 14 else "")
    return f"unknown key {key!r}", f"valid here: {shown}" if known else "montaj spec --search <word> finds a field"


def _from_pydantic(exc: ValidationError) -> list[str]:
    lines = []
    for err in exc.errors():
        path = _yaml_path(tuple(err["loc"]))
        if err["type"] == "extra_forbidden":
            problem, hint = _unknown_key(tuple(err["loc"]))
            lines.append(_line(path, problem, hint) if hint else f"{path}: {problem}")
            continue
        msg = err["msg"]
        if msg.lower().startswith("value error, "):
            msg = msg[13:]
        key = str(err["loc"][-1]) if err["loc"] else ""
        hint = _PYDANTIC_HINTS.get(err["type"], f"montaj spec --search {key}")
        if err["type"] == "missing":
            msg, hint = f"missing required key {key!r}", f"add `{key}:`; montaj spec --search {key} shows its form"
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


def _shot_effects(sh: Shot, i: int) -> list[tuple[str, Effect]]:
    """(path, effect) for the shot's effects, then each layer's."""
    out = [(f"shots[{i}].effects[{j}]", fx) for j, fx in enumerate(sh.effects or ())]
    for k, layer in enumerate(sh.layers or ()):
        out.extend((f"shots[{i}].layers[{k}].effects[{j}]", fx) for j, fx in enumerate(layer.effects or ()))
    return out


def _duration_fields(spec: Spec) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if spec.video.intro:
        out.append(("video.intro.dur", spec.video.intro.dur))
    if spec.video.outro:
        out.append(("video.outro.dur", spec.video.outro.dur))
    for i, fx in enumerate(spec.fx):
        if fx.dur is not None:
            out.append((f"fx[{i}].dur", fx.dur))
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
        for k, layer in enumerate(sh.layers or ()):
            for name, raw in (("from", layer.from_), ("to", layer.to),
                              ("in.dur", layer.in_ and layer.in_.dur), ("out.dur", layer.out and layer.out.dur)):
                if raw is not None:
                    out.append((f"shots[{i}].layers[{k}].{name}", raw))
            for prop in LAYER_KEYED:
                keys = getattr(layer, prop)
                if isinstance(keys, list):
                    for j, lkey in enumerate(keys):
                        if lkey.on is None:  # an anchored key's time is worked out against the grid or a word
                            out.append((f"shots[{i}].layers[{k}].{prop}[{j}].t", lkey.t))
        for path, fx in _shot_effects(sh, i):
            if fx.dur is not None:
                out.append((f"{path}.dur", fx.dur))
            if isinstance(fx.amount, list):
                for k, key in enumerate(fx.amount):
                    out.append((f"{path}.amount[{k}].t", key.t))
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
        for k, layer in enumerate(sh.layers or ()):
            if layer.clip_in is not None:
                out.append((f"shots[{i}].layers[{k}].clip_in", layer.clip_in))
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
            out.extend(_motion_durs(f"text[{i}].lines[{j}]", line))
    return out


def _motion_durs(path: str, line: TextLine) -> list[tuple[str, str]]:
    """`in`/`out` durations of a text line, keyed like the line's own `stagger`/`dur`."""
    out = []
    for name, m in (("in", line.in_), ("out", line.out)):
        if m is None:
            continue
        for field in ("dur", "stagger"):
            raw = getattr(m, field)
            if raw:
                out.append((f"{path}.{name}.{field}", raw))
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
    from montaj.audio.mix import time_strings

    out.extend(time_strings(spec))
    for sh in spec.shots:
        for layer in sh.layers or ():
            for prop in LAYER_KEYED:
                for lkey in getattr(layer, prop) if isinstance(getattr(layer, prop), list) else ():
                    if lkey.on is not None:
                        out.append(lkey.t if isinstance(lkey.t, str) else "0f")
                        if lkey.on == "beat":
                            out.append("1b")  # snapping to the grid needs a bpm
                for text in _expr_texts(getattr(layer, prop)):
                    with contextlib.suppress(ExprError):
                        if xp.uses(text, "beat", "bpm"):
                            out.append("1b")
        if sh.until is not None:
            out.append(sh.until)
        if sh.range is not None:
            out.extend(sh.range)
    for fx in spec.fx:
        if isinstance(fx.at, str):
            out.append(fx.at)
    for i, sh in enumerate(spec.shots):
        out.extend(fx.at for _, fx in _shot_effects(sh, i) if isinstance(fx.at, str))
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
            out.extend(raw for _, raw in _motion_durs("", line))
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
    for ov in spec.overlays:
        out.extend([ov.from_, ov.to])
        if ov.draw:
            out.append(ov.draw)
    if spec.tracks is not None:
        for keys in (spec.tracks.dust, spec.tracks.glow, spec.tracks.bars):
            out.extend(k.t for k in keys)
    if spec.check is not None:
        for item in spec.check.expect:
            pair = item.black or item.frozen or item.caption
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
        layers = shot.get("layers")
        for j, layer in enumerate(layers if isinstance(layers, list) else ()):
            if not isinstance(layer, dict):
                continue
            path = f"shots[{i}].layers[{j}]"
            if layer.get("image") is not None:
                pid = _photo_id(layer["image"])
                if isinstance(pid, str):
                    _require_photo(assets_dir, pid, f"{path}.image", errors, find_photo)
            if layer.get("clip") is not None:
                pid = _photo_id(layer["clip"])
                if isinstance(pid, str):
                    if clip_is_file(pid):
                        from montaj.project import AssetMissing, resolve_clip

                        try:
                            resolve_clip(spec_dir, assets_dir, pid)
                        except AssetMissing as e:
                            errors.append(_line(f"{path}.clip", str(e), e.hint))
                    else:
                        _require_photo(assets_dir, pid, f"{path}.clip", errors, find_clip)


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


def _has_sugar(sh: Shot) -> bool:
    if isinstance(sh.focus, str) or isinstance(sh.range, str):
        return True
    if any(isinstance(layer.focus, str) for layer in sh.layers or ()):
        return True
    return sh.camera is not None and any(isinstance(k.focus, str) for k in sh.camera)


def resolve_sugar(spec: Spec, spec_dir: Path | None, errors: list[str]) -> Spec:
    """`focus: face|face:N|subject` and `range: speech:N` become numbers from assets/analysis.json.

    Runs before layout, so the shot the renderer and the segment hash see holds only numbers. A
    failed lookup records one error and leaves a neutral value so later checks do not pile on.
    """
    if not any(_has_sugar(sh) for sh in spec.shots):
        return spec
    from montaj.analysis import AnalysisBook

    book = AnalysisBook(_assets_dir(spec, spec_dir))
    shots = [
        _resolve_shot_sugar(spec, sh, i, spec_dir, book, errors) if _has_sugar(sh) else sh
        for i, sh in enumerate(spec.shots)
    ]
    return spec.model_copy(update={"shots": shots})


def _sugar_row(spec: Spec, sh: Shot, spec_dir: Path | None, book, path: str, errors: list[str]):
    """(kind, stem, row) for the shot's photo or assets clip, or None after one error."""
    from montaj.project import clip_is_file, find_clip, find_photo

    if sh.photo is not None:
        kind, stem, find = "photo", sh.photo, find_photo
    elif sh.clip is not None:
        if clip_is_file(sh.clip):
            errors.append(_line(path, f"{sh.clip} is not analysed", "use a clip in assets/ and run montaj ingest"))
            return None
        kind, stem, find = "clip", sh.clip, find_clip
    else:
        errors.append(_line(path, "analysis sugar needs photo or clip", "use [x, y]"))
        return None
    assets = _assets_dir(spec, spec_dir)
    if assets is None:
        errors.append(_line(path, f"no analysis for {stem}", "load the spec from disk"))
        return None
    try:
        file = find(assets, stem)
    except FileNotFoundError:
        return None  # the missing asset is already its own error
    row, why = book.row(file)
    if row is None:
        errors.append(_line(path, why, "run montaj ingest"))
        return None
    return kind, stem, row


def _clip_start_s(spec: Spec, raw: str | None) -> float:
    """Source second a clip shot starts at, for picking its scene. Unparseable -> 0 (validate reports it)."""
    fps = spec.video.fps
    if raw is None or fps <= 0:
        return 0.0
    frames = _dur_frames(raw, fps, spec.video.bpm, "", [])
    return frames / fps if frames is not None and frames > 0 else 0.0


def _focus_point(ref: str, kind: str, stem: str, row: dict, t: float, path: str, errors: list[str]):
    from montaj.analysis import face_point

    if kind == "clip":
        scenes = row.get("scenes") or []
        scene = next((s for s in scenes if s["s"] <= t < s["e"]), scenes[-1] if scenes else {})
        faces, subject, where = scene.get("faces") or [], scene.get("subject"), f"{stem} at {_fmt_num(t)}s"
    else:
        faces, subject, where = row.get("faces") or [], row.get("subject"), stem
    if ref == "subject":
        if not subject:
            errors.append(_line(path, f"no subject in {where}", "use [x, y]"))
            return None
        return (float(subject[0]), float(subject[1]))
    m = _FOCUS_REF.match(ref)
    n = int(m.group(1) or 1) if m else 1
    if not faces:
        errors.append(_line(path, f"no face in {where}", "use subject or [x, y]"))
        return None
    if n > len(faces):
        have = ", ".join(f"face:{k + 1}" for k in range(len(faces)))
        errors.append(_line(path, f"no face {n} in {where}", f"have: {have}"))
        return None
    x, y = face_point(faces[n - 1])
    return (x, y)


def _resolve_shot_sugar(spec: Spec, sh: Shot, i: int, spec_dir: Path | None, book, errors: list[str]) -> Shot:
    at = f"shots[{i}]"
    update: dict[str, Any] = {}
    rng = sh.range
    if isinstance(rng, str):
        got = _sugar_row(spec, sh, spec_dir, book, f"{at}.range", errors)
        rng = ("0s", "1s")  # neutral stand-in after an error; the spec fails anyway
        if got is not None and got[0] == "clip":
            spans = got[2].get("speech") or []
            n = int(_SPEECH_REF.match(sh.range).group(1))
            if n > len(spans):
                errors.append(_line(f"{at}.range", f"no speech span {n} in {got[1]}", f"have: {len(spans)}"))
            else:
                span = spans[n - 1]
                rng = (format_seconds(span["s"]), format_seconds(span["e"]))
        update["range"] = rng
    focus_refs = isinstance(sh.focus, str) or (sh.camera is not None and any(isinstance(k.focus, str) for k in sh.camera))
    if focus_refs:
        got = _sugar_row(spec, sh, spec_dir, book, f"{at}.focus", errors)
        t = _clip_start_s(spec, rng[0] if isinstance(rng, tuple) else sh.clip_in)

        def point(ref: str, path: str):
            return None if got is None else _focus_point(ref, got[0], got[1], got[2], t, path, errors)

        if isinstance(sh.focus, str):
            update["focus"] = point(sh.focus, f"{at}.focus")
        if sh.camera is not None:
            update["camera"] = [
                key.model_copy(update={"focus": point(key.focus, f"{at}.camera[{k}].focus")})
                if isinstance(key.focus, str) else key
                for k, key in enumerate(sh.camera)
            ]
    if sh.layers and any(isinstance(layer.focus, str) for layer in sh.layers):
        layers = []
        for k, layer in enumerate(sh.layers):
            if isinstance(layer.focus, str):
                path = f"{at}.layers[{k}].focus"
                # A layer is its own source: read its image or clip, at its own clip_in.
                src = SimpleNamespace(photo=layer.image, clip=layer.clip)
                got = _sugar_row(spec, src, spec_dir, book, path, errors)
                t = _clip_start_s(spec, layer.clip_in)
                xy = None if got is None else _focus_point(layer.focus, got[0], got[1], got[2], t, path, errors)
                layer = layer.model_copy(update={"focus": xy})
            layers.append(layer)
        update["layers"] = layers
    return sh.model_copy(update=update)


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


def _key_anchors(spec: Spec) -> list[tuple[str, str]]:
    """(path, `on` value) for every anchored layer key."""
    out = []
    for i, sh in enumerate(spec.shots):
        for k, layer in enumerate(sh.layers or ()):
            for prop in LAYER_KEYED:
                keys = getattr(layer, prop)
                for j, lkey in enumerate(keys if isinstance(keys, list) else ()):
                    if lkey.on is not None:
                        out.append((f"shots[{i}].layers[{k}].{prop}[{j}].on", lkey.on))
                for text in _expr_texts(getattr(layer, prop)):
                    with contextlib.suppress(ExprError):
                        out.extend((f"shots[{i}].layers[{k}].{prop}.expr", f"word:{n}") for n in xp.words(text))
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
    events: _Events | None = None,
) -> dict[str, float]:
    """Resolve `markers` to float frames. Acyclic name→name refs are allowed."""
    uses_word = [
        (name, expr.strip())
        for name, expr in spec.markers.items()
        if isinstance(expr, str) and _WORD.match(expr.strip())
    ]
    anchors = [(path, on) for path, on in _key_anchors(spec) if _WORD.match(on)]  # layer keys `on: word:N`
    uses_word += [(path, on) for path, on in anchors]
    word_s: dict[int, float] = {}
    if uses_word:
        if spec.audio is None or not spec.audio.markers:
            for name, expr in uses_word:
                errors.append(
                    _line(name if name.startswith("shots[") else f"markers.{name}", f"{expr} needs audio.markers", "set audio.markers")
                )
        else:
            word_s = _load_word_starts(
                spec, spec_dir, errors, data=markers_json, loaded=markers_loaded
            )
            if spec.audio.edit:
                from montaj.audio.mix import shift_words

                word_s = shift_words(spec, word_s, fps, bpm, beat0)

    for name in spec.markers:
        if not _NAME.match(name):
            errors.append(_line(f"markers.{name}", f"invalid name {name!r}", "use [a-z][a-z0-9_]*"))

    resolved: dict[str, float] = {}
    pending: dict[str, str] = {}
    for path, on in anchors:  # filed under their own text ("word:12"), which no marker name can be
        i = int(_WORD.match(on).group(1))
        if i in word_s:
            resolved[on] = word_s[i] * fps
        elif spec.audio is not None and spec.audio.markers:
            have = ", ".join(str(k) for k in sorted(word_s)) or "none"
            errors.append(_line(path, f"unknown word {i}", f"have: {have}"))
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
            _check_ease(f"{path}.ease", ck.ease, errors)
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


def _check_trans_params(i: int, tr: Transition, errors: list[str]) -> None:
    """strength / seed / color belong to specific transitions; reject them elsewhere."""
    p = f"shots[{i}].in"
    if tr.strength is not None:
        if tr.type not in ("glitch", "zoom"):
            errors.append(_line(f"{p}.strength", f"not used by {tr.type}", "use it on glitch or zoom"))
        elif not math.isfinite(tr.strength) or not 0 < tr.strength <= TRANS_STRENGTH_MAX:
            errors.append(
                _line(
                    f"{p}.strength",
                    f"must be > 0 and <= {TRANS_STRENGTH_MAX:g}, got {_fmt_num(tr.strength)}",
                    "use a value like 1.0",
                )
            )
    if tr.seed is not None:
        if tr.type != "glitch":
            errors.append(_line(f"{p}.seed", f"not used by {tr.type}", "use it on glitch"))
        elif tr.seed < 0:
            errors.append(_line(f"{p}.seed", f"must be >= 0, got {tr.seed}", "use a whole number like 3"))
    if tr.color is not None:
        if tr.type != "dip":
            errors.append(_line(f"{p}.color", f"not used by {tr.type}", "use it on dip"))
        elif not _HEX.match(tr.color):
            errors.append(_line(f"{p}.color", f"unknown colour {tr.color!r}", "use #rrggbb"))
    for name in ("src", "params"):
        if getattr(tr, name) is not None and tr.type != "custom":
            errors.append(_line(f"{p}.{name}", f"not used by {tr.type}", "use it on type: custom"))


def _check_custom_file(path: str, rel: str, spec_dir: Path | None, errors: list[str]) -> None:
    """A custom file exists, parses and defines top-level draw(ctx). It is not run here."""
    if not rel.endswith(".py"):
        errors.append(_line(path, f"{rel!r} is not a .py file", "name a Python file like fx/stars.py"))
        return
    if spec_dir is None:
        return
    file = Path(spec_dir) / rel
    if not file.is_file():
        errors.append(_line(path, f"{rel} not found", "paths are relative to montaj.yaml"))
        return
    try:
        tree = ast.parse(file.read_bytes(), filename=str(file))
    except SyntaxError as e:
        errors.append(_line(path, f"{rel}:{e.lineno}: syntax error: {e.msg}", "fix the file"))
        return
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "draw":
            if not node.args.args:
                errors.append(_line(path, f"{rel}:{node.lineno}: draw takes no ctx", "use def draw(ctx):"))
            return
    errors.append(_line(path, f"{rel}: no top-level draw(ctx)", "define def draw(ctx): ... return rgb"))


def _validate_custom_shot(sh: Shot, i: int, spec_dir: Path | None, errors: list[str]) -> None:
    if sh.custom is not None:
        _check_custom_file(f"shots[{i}].custom", sh.custom, spec_dir, errors)
        for name in ("crop", "focus", "zoom"):
            if getattr(sh, name) is not None:
                errors.append(_line(f"shots[{i}].{name}", "not used on a custom shot", "pass it in params"))
    elif sh.params is not None:
        errors.append(_line(f"shots[{i}].params", "params needs custom", "add custom: fx/name.py"))
    for j, fx in enumerate(sh.effects or ()):
        if fx.custom is not None:
            _check_custom_file(f"shots[{i}].effects[{j}].custom", fx.custom, spec_dir, errors)
    if sh.effects is not None and not sh.effects:
        errors.append(_line(f"shots[{i}].effects", "empty list", "remove effects or add {type: blur} or {custom: fx/name.py}"))
    if sh.blank and sh.effects:
        errors.append(_line(f"shots[{i}].effects", "not used on a blank shot", "use a custom shot"))


# Upper bounds that keep a typo from costing minutes a frame.
_EFFECT_MAX = {"shake": 200.0, "blur": 200.0, "bloom": 4.0, "chroma": 4.0, "trails": 1.0}
_TRAIL_FRAMES = 16


def _fmt_g(v: float) -> str:
    """`200`, not `200.0`: effect ranges read like the YAML they came from."""
    return f"{v:g}" if math.isfinite(v) else _fmt_num(v)


def _validate_effects(sh: Shot, i: int, markers: dict[str, float], fps: int,
                      bpm: float | None, beat0: float, parsed: dict[str, int], errors: list[str]) -> None:
    """Built-in effects (#38): one of custom/type, knobs that fit the type, finite ranges.

    `at` is parsed here (it may name a marker) into `parsed[<path>.at]`, film frames.
    """
    for k, layer in enumerate(sh.layers or ()):
        if layer.effects is not None and not layer.effects:
            errors.append(_line(f"shots[{i}].layers[{k}].effects", "empty list", "remove effects or add {type: blur}"))
    for path, fx in _shot_effects(sh, i):
        on_layer = ".layers[" in path
        if (fx.custom is None) == (fx.type is None):
            errors.append(_line(path, "need exactly one of custom or type", f"set type: {_choices(EFFECT_TYPES)}, or custom: fx/name.py"))
            continue
        if fx.custom is not None:
            if on_layer:
                errors.append(_line(f"{path}.custom", "custom effects run on a shot's picture", "move it to shots[].effects"))
            for name in ("amount", "at", "dur", *sorted({n for ks in EFFECT_KNOBS.values() for n in ks})):
                if getattr(fx, name) is not None:
                    errors.append(_line(f"{path}.{name}", "is for a built-in type", "pass it in params"))
            continue
        if fx.type not in EFFECT_TYPES:
            errors.append(_line(f"{path}.type", f"unknown {fx.type!r}", f"use {_choices(EFFECT_TYPES)}"))
            continue
        if fx.params is not None:
            errors.append(_line(f"{path}.params", "params is for custom", "set the effect's fields directly"))
        knobs = EFFECT_KNOBS[fx.type]
        for name in sorted({n for ks in EFFECT_KNOBS.values() for n in ks} - set(knobs)):
            if getattr(fx, name) is not None:
                errors.append(_line(f"{path}.{name}", f"not used by {fx.type}",
                                    f"{fx.type} takes amount, at, dur" + "".join(f", {n}" for n in knobs)))
        top = _EFFECT_MAX[fx.type]
        values = fx.amount if isinstance(fx.amount, list) else [] if fx.amount is None else [EffectKey(t="0f", v=fx.amount)]
        if isinstance(fx.amount, list) and not fx.amount:
            errors.append(_line(f"{path}.amount", "empty keyframe list", "use a number or [{t: 0f, v: 1}, ...]"))
        for key in values:
            if not math.isfinite(key.v) or not 0 <= key.v <= top:
                errors.append(_line(f"{path}.amount", f"must be 0 to {_fmt_g(top)}, got {_fmt_g(key.v)}",
                                    "0 is off"))
                break
        if isinstance(fx.amount, list):
            ts = [parsed.get(f"{path}.amount[{k}].t") for k in range(len(fx.amount))]
            if None not in ts and any(b <= a for a, b in zip(ts, ts[1:])):
                errors.append(_line(f"{path}.amount", "keyframe times must increase", "sort the keys by t"))
        if fx.dur is not None and fx.at is None:
            errors.append(_line(f"{path}.dur", "dur needs at", "add at: a time or marker"))
        d = parsed.get(f"{path}.dur")
        if d is not None and d <= 0:
            errors.append(_line(f"{path}.dur", f"must be > 0, got {d}f", "lengthen dur"))
        if fx.at is not None:
            at_f = _parse_time(fx.at, markers, fps, bpm, f"{path}.at", errors, beat0=beat0)
            if at_f is not None:
                parsed[f"{path}.at"] = round_frame(at_f)
        if fx.threshold is not None and (not math.isfinite(fx.threshold) or not 0 <= fx.threshold < 1):
            errors.append(_line(f"{path}.threshold", f"must be 0 to below 1, got {_fmt_g(fx.threshold)}", "use 0.75"))
        if fx.radius is not None and (not math.isfinite(fx.radius) or not 0 < fx.radius <= 400):
            errors.append(_line(f"{path}.radius", f"must be > 0 and at most 400, got {_fmt_g(fx.radius)}", "design px, use 24"))
        if fx.frames is not None and not 1 <= fx.frames <= _TRAIL_FRAMES:
            errors.append(_line(f"{path}.frames", f"must be 1 to {_TRAIL_FRAMES}, got {fx.frames}", "use 4"))
        if fx.decay is not None and (not math.isfinite(fx.decay) or not 0 < fx.decay <= 1):
            errors.append(_line(f"{path}.decay", f"must be above 0 and at most 1, got {_fmt_g(fx.decay)}", "use 0.6"))


def _check_ease(path: str, ease: str, errors: list[str]) -> None:
    if parse_ease(ease) is None:
        errors.append(_line(path, f"unknown {ease!r}", f"{EASE_HINT}; or {', '.join(NEW_EASES)}"))


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
    spec = resolve_sugar(spec, spec_dir, errors)
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
    if spec.audio is not None:
        from montaj.audio.mix import plan_audio

        plan_audio(spec, spec_dir, markers, fps, bpm, beat0, errors)

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

    _validate_mattes(spec, spec_dir, errors)
    check_grades(spec, errors)
    clip_ins: dict[int, str] = {}
    for i, sh in enumerate(spec.shots):
        h = holds[i]
        has_p = sh.photo is not None
        has_w = sh.wall is not None
        has_c = sh.clip is not None
        has_b = sh.blank
        has_s = sh.sequence is not None
        has_card = sh.card is not None
        has_custom = sh.custom is not None
        n_src = has_p + has_w + has_c + has_b + has_s + has_card + has_custom
        if sh.layers is not None and n_src:
            errors.append(_line(f"shots[{i}].layers", "layers replace the shot's single source", "move it into a layer"))
        elif n_src > 1:
            errors.append(
                _line(
                    f"shots[{i}]",
                    "photo, wall, clip, blank, sequence, card and custom are mutually exclusive",
                    "remove extra",
                )
            )
        elif n_src == 0 and sh.layers is None:
            errors.append(
                _line(f"shots[{i}]", "need photo, wall, clip, blank, sequence, card, custom or layers", "add one")
            )
        _validate_custom_shot(sh, i, spec_dir, errors)
        _validate_effects(sh, i, markers, fps, bpm, beat0, parsed, errors)
        if sh.card is not None:
            _validate_card(sh.card, i, fps, bpm, errors)
        if sh.layers is not None:
            _validate_layers(spec, i, sh, h, parsed, times, spec_dir, errors, (markers, fps, bpm, beat0))
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
                _check_ease(f"shots[{i}].wall.camera.pos[{k}].ease", key.ease, errors)
                if key.print is not None and key.print not in print_ids:
                    errors.append(
                        _line(
                            f"shots[{i}].wall.camera.pos[{k}].print",
                            f"{key.print!r} not in this wall's prints",
                            f"have: {have_prints}",
                        )
                    )
            for k, key in enumerate(sh.wall.camera.zoom):
                _check_ease(f"shots[{i}].wall.camera.zoom[{k}].ease", key.ease, errors)
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
        _check_trans_params(i, tr, errors)
        if tr.type == "custom":
            if tr.src is None:
                errors.append(_line(f"shots[{i}].in.src", "required for custom", "set src: fx/name.py"))
            else:
                _check_custom_file(f"shots[{i}].in.src", tr.src, spec_dir, errors)
        if tr.type == "cut":
            continue
        if tr.type in ("whip", "push") and tr.axis not in ("x", "y"):
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
    from montaj.audio.mix import check_links

    check_links(spec, spec_dir, holds, fps, bpm, errors)
    _check_finish(spec, errors, spec_dir)
    if spec.check is not None and not 1 <= spec.check.sample <= 30:
        errors.append(_line("check.sample", f"must be 1 to 30, got {spec.check.sample}", "use sample: 3"))

    for i, fx in enumerate(spec.fx):
        kinds = [
            fx.leak is not None,
            fx.flash is not None,
            fx.hit is not None,
            fx.burst is not None,
            fx.rgb_split is not None,
        ]
        if sum(kinds) != 1:
            errors.append(_line(f"fx[{i}]", "need leak, flash, hit, burst or rgb_split", "set one"))
        if fx.rgb_split is not None and (
            not math.isfinite(fx.rgb_split) or not 0 < fx.rgb_split <= TRANS_STRENGTH_MAX
        ):
            errors.append(
                _line(
                    f"fx[{i}].rgb_split",
                    f"must be > 0 and <= {TRANS_STRENGTH_MAX:g}, got {_fmt_num(fx.rgb_split)}",
                    "use a value like 0.8",
                )
            )
        if fx.dur is not None and fx.rgb_split is None:
            errors.append(_line(f"fx[{i}].dur", "dur is for rgb_split", "remove dur"))
        if fx.dur is not None and parsed.get(f"fx[{i}].dur", 1) <= 0:
            errors.append(_line(f"fx[{i}].dur", "must be > 0", "lengthen dur"))
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
    _validate_overlays(spec, markers, fps, bpm, times, errors, beat0=beat0)
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
    before = len(errors)
    _check_text_line_fields(path, line, errors)
    if len(errors) == before:  # measure only a line whose fields are sound
        over = _overflow_error(path, line)
        if over:
            errors.append(over)


def _check_text_line_fields(path: str, line: TextLine, errors: list[str]) -> None:
    if line.style not in TEXT_STYLES:
        errors.append(_line(f"{path}.style", f"unknown {line.style!r}", f"use {', '.join(TEXT_STYLES[:-1])} or {TEXT_STYLES[-1]}"))
    if line.italic and line.style != "serif":
        errors.append(_line(f"{path}.italic", "italic is serif-only", "set style: serif"))
    if line.by not in TEXT_BY:
        errors.append(_line(f"{path}.by", f"unknown {line.by!r}", "use word or char"))
    if line.reveal not in TEXT_REVEAL:
        errors.append(_line(f"{path}.reveal", f"unknown {line.reveal!r}", "use rise, pop or type"))
    if line.box not in TEXT_BOX:
        errors.append(_line(f"{path}.box", f"unknown {line.box!r}", "use none or chip"))
    if line.align not in TEXT_ALIGN:
        errors.append(_line(f"{path}.align", f"unknown {line.align!r}", "use center, left or right"))
    if line.box != "chip" and (line.dot is not None or line.box_color is not None):
        errors.append(_line(f"{path}.box", "dot and box_color need a box", "set box: chip"))
    if line.reveal == "type" and line.dur is not None:
        errors.append(_line(f"{path}.dur", "typewriter characters appear instantly", "remove dur, or use reveal: rise"))
    if line.cursor and line.reveal != "type":
        errors.append(_line(f"{path}.cursor", "cursor needs the typewriter reveal", "set reveal: type"))
    if line.y is None and line.pos is None:
        errors.append(_line(f"{path}.y", "missing", "set y (top of the line, design px) or pos: [x, y]"))
    if line.engine is not None or gpu_fields(line):
        _check_text_engine(path, line, errors)
    _check_text_fit_fields(path, line, errors)
    allowed = TEXT_WEIGHTS.get(line.style)
    if allowed and line.weight is not None and line.weight not in allowed:
        errors.append(_line(f"{path}.weight", f"{line.style} has no weight {_fmt_num(line.weight)}",
                            f"use {', '.join(str(w) for w in allowed)}"))
    _check_color(f"{path}.dot", line.dot, errors)
    _check_color(f"{path}.box_color", line.box_color, errors)
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


def _check_text_fit_fields(path: str, line: TextLine, errors: list[str]) -> None:
    if line.wrap is not None and line.wrap != "balance":
        errors.append(_line(f"{path}.wrap", f"unknown {line.wrap!r}", "use balance"))
    if line.max_lines is not None and (line.wrap is None or not 1 <= line.max_lines <= 4):
        errors.append(_line(f"{path}.max_lines", "needs wrap: balance and a count from 1 to 4", "set wrap: balance, max_lines: 2"))
    if line.max_width is not None and (not math.isfinite(line.max_width) or line.max_width <= 0):
        errors.append(_line(f"{path}.max_width", f"must be finite and > 0, got {_fmt_num(line.max_width)}", "use a design-px width"))


def _overflow_error(path: str, line: TextLine) -> str | None:
    """`ERR` text when a line, shaped with the bundled fonts, runs outside the frame (or past `max_width`).

    Needs no GPU and no browser. `fit` and `wrap` resolve the overflow at timeline time, so only a line
    they cannot save is reported. Returns None when the line fits or cannot be measured.
    """
    from montaj.render import glyphs

    if line.style not in TEXT_STYLES or not line.text.strip():
        return None
    size = float(line.size if line.size is not None else glyphs.STYLE_METRICS[line.style].size)
    book = glyphs.FontBook()
    # Measured as drawn: the line's font, case, emphasis scale, stroke and rotation (gpu lines). `pos`
    # replaces the aligned x; a wrapped, rotated stack is measured piece by piece about each piece's centre.
    kw = dict(style=line.style, weight=line.weight, italic=line.italic, tracking=line.tracking,
              line_height=line.line_height, font=line.font, case=line.case,
              emphasis_scale=line.emphasis.scale if line.emphasis is not None else 1.0,
              stroke=line.stroke.width if line.stroke is not None else 0.0, rotate=float(line.rotate or 0.0))
    place = dict(pos=(line.pos[0], 0.0)) if line.pos is not None else dict(align=line.align, indent=line.indent)
    wants = line.fit or line.wrap is not None
    text = glyphs.parse_emphasis(line.text)[0] if line_engine(line) == "gpu" else line.text
    if not wants and book.missing(line.style, line.weight, line.italic, text, font=line.font, case=line.case):
        return None  # Chromium falls back to a system font here; the bundled fonts cannot say how wide
    try:
        plan = glyphs.plan_text(book, line.text, size=size, fit=line.fit, balance=line.wrap == "balance",
                                max_width=line.max_width, max_lines=line.max_lines or 2, **kw)
        worst = 0.0
        for text in plan.lines:
            x, w, _h = glyphs.measure(book, text, size=plan.size, **place, **kw)
            worst = max(worst, w - line.max_width if line.max_width is not None else glyphs.frame_overflow(x, w))
        if worst <= 0.5:
            return None
        fit_to = glyphs.plan_text(book, line.text, size=size, fit=True, balance=line.wrap == "balance",
                                  max_width=line.max_width, max_lines=line.max_lines or 2, **kw)
    except ValueError as exc:
        return _line(path, str(exc), "shorten the text, or use a style whose fonts have these characters")
    edge = f"max_width {_fmt_num(line.max_width)}" if line.max_width is not None else "the frame"
    return _line(path, f"{round(worst)}px wider than {edge} at size {plan.size:g}",
                 f"fits at size {fit_to.size:g}; or set fit: true")


def _check_text_engine(path: str, line: TextLine, errors: list[str]) -> None:
    from montaj.render import glyphs

    if line.engine is not None and line.engine not in glyphs.ENGINES:
        errors.append(_line(f"{path}.engine", f"unknown {line.engine!r}", "use chromium or gpu"))
        return
    styled = gpu_fields(line)
    if line.engine == "chromium":
        for name in styled:
            what = "*emphasis* markup" if name == "text" else name
            errors.append(_line(f"{path}.{name}", f"{what} is drawn by the gpu engine",
                                "remove engine: chromium" if name != "text" else "remove engine: chromium, or escape * as \\*"))
        return
    if line_engine(line) != "gpu":
        return
    # The gpu engine draws fill, shadow and the rise/pop/type reveals; these still need Chromium.
    hint = "remove it, or drop engine: gpu" if not styled else f"remove it, or drop {', '.join(styled)}"
    for name, unset in (("gold", False), ("box", "none"), ("cursor", False)):
        if getattr(line, name) != unset:
            errors.append(_line(f"{path}.{name}", f"{name} is not drawn by the gpu engine yet", hint))
    _check_text_styling(path, line, errors)
    _check_text_motion(path, line, errors)
    if line.font is not None and line.font not in glyphs.FONTS:
        errors.append(_line(f"{path}.font", f"unknown {line.font!r}", f"use {', '.join(glyphs.FONTS)}"))
        return
    if line.font is not None and line.weight is not None:
        errors.append(_line(f"{path}.weight", f"{line.font} ships one weight", "remove weight, or pick another font"))
    if line.font is not None and line.italic:
        errors.append(_line(f"{path}.italic", f"{line.font} has no italic", "remove italic"))
    if line.style in TEXT_STYLES and (line.case is None or line.case in glyphs.CASES):
        try:
            plain, _ = glyphs.parse_emphasis(line.text)
        except ValueError:
            return  # reported by _check_text_styling
        missing = glyphs.FontBook().missing(line.style, line.weight, line.italic, plain, font=line.font, case=line.case)
        if missing:
            errors.append(_line(f"{path}.text", f"{line.font or line.style} fonts have no glyph for {''.join(missing)!r}",
                                "use another font or style" if styled else "use another style, or drop engine: gpu"))


def _check_text_motion(path: str, line: TextLine, errors: list[str]) -> None:
    """Field-level checks for the gpu-engine motion fields (#22)."""
    from montaj.render import motion as M

    def finite(p: str, v, lo: float, hi: float, hint: str) -> None:
        if v is not None and (not math.isfinite(v) or not lo <= v <= hi):
            errors.append(_line(p, f"must be from {_fmt_num(lo)} to {_fmt_num(hi)}, got {_fmt_num(v)}", hint))

    moving = line.in_ is not None or line.out is not None or line.idle is not None or (
        line.emphasis is not None and (line.emphasis.in_ is not None or line.emphasis.shake is not None))
    if moving and line.reveal == "pop":
        errors.append(_line(f"{path}.reveal", "reveal: pop is the whole-line pop; in, out, idle and emphasis motion are per unit",
                            "remove reveal and use in: pop"))
    elif line.in_ is not None and line.reveal != "rise":
        errors.append(_line(f"{path}.reveal", "in replaces reveal", f"remove reveal and use in: {line.reveal}"))
    if line.in_ is not None:
        m, p = line.in_, f"{path}.in"
        if m.type not in M.IN_TYPES:
            errors.append(_line(f"{p}.type", f"unknown {m.type!r}", f"use {', '.join(M.IN_TYPES)}"))
        if m.by is not None and m.by not in TEXT_BY:
            errors.append(_line(f"{p}.by", f"unknown {m.by!r}", "use word or char"))
        if m.type == "type" and (m.dur is not None or line.dur is not None):
            errors.append(_line(f"{p}.dur", "typewriter characters appear instantly", "remove dur"))
        if m.from_ is not None and (m.type != "slide" or m.from_ not in M.SIDES):
            errors.append(_line(f"{p}.from", f"from {m.from_!r} needs in: slide and a side", "use left, right, top or bottom with type: slide"))
        if m.overshoot is not None and m.type not in M.IN_OVERSHOOT:
            errors.append(_line(f"{p}.overshoot", f"{m.type} has no overshoot", "remove overshoot"))
        finite(f"{p}.overshoot", m.overshoot, 1.0, 2.0, "use 1.0-1.2")
        if m.distance is not None and m.type not in ("slide", "drop", "rise"):
            errors.append(_line(f"{p}.distance", f"{m.type} does not travel", "remove distance"))
        finite(f"{p}.distance", m.distance, 0.0, 2000.0, "use design px, e.g. 160")
    if line.out is not None:
        m, p = line.out, f"{path}.out"
        if m.type not in M.OUT_TYPES:
            errors.append(_line(f"{p}.type", f"unknown {m.type!r}", f"use {', '.join(M.OUT_TYPES)}"))
        if m.type == "cut" and m.dur is not None:
            errors.append(_line(f"{p}.dur", "a cut takes no time", "remove dur, or use type: fade"))
        if m.to is not None and (m.type != "slide" or m.to not in M.SIDES):
            errors.append(_line(f"{p}.to", f"to {m.to!r} needs out: slide and a side", "use left, right, top or bottom with type: slide"))
        if m.distance is not None and m.type != "slide":
            errors.append(_line(f"{p}.distance", f"{m.type} does not travel", "remove distance"))
        finite(f"{p}.distance", m.distance, 0.0, 2000.0, "use design px, e.g. 160")
    if line.idle is not None:
        if line.idle.type not in M.IDLE_TYPES:
            errors.append(_line(f"{path}.idle.type", f"unknown {line.idle.type!r}", "use float or pulse"))
        finite(f"{path}.idle.amount", line.idle.amount, 0.0, 100.0, "float: design px; pulse: percent")
    if line.emphasis is not None:
        e = line.emphasis
        if e.in_ is not None and (e.in_ not in M.IN_TYPES or e.in_ == "type"):
            errors.append(_line(f"{path}.emphasis.in", f"unknown {e.in_!r}",
                                f"use {', '.join(t for t in M.IN_TYPES if t != 'type')}"))
        finite(f"{path}.emphasis.shake", e.shake, 0.0, 60.0, "use design px, e.g. 6")


def _check_text_styling(path: str, line: TextLine, errors: list[str]) -> None:
    """Field-level checks for the gpu-engine styling fields (#21)."""
    from montaj.render import glyphs

    if line.case is not None and line.case not in glyphs.CASES:
        errors.append(_line(f"{path}.case", f"unknown {line.case!r}", "use upper, lower or as-is"))
    try:
        _plain, spans = glyphs.parse_emphasis(line.text)
    except ValueError:
        errors.append(_line(f"{path}.text", "unclosed *", "close the *emphasis*, or escape a literal asterisk as \\*"))
        spans = []
    if line.emphasis is not None:
        if not any(spans):
            errors.append(_line(f"{path}.emphasis", "no *marked* words in text", "wrap the words in *…*, or remove emphasis"))
        _check_color(f"{path}.emphasis.color", line.emphasis.color, errors)
        sc = line.emphasis.scale
        if not math.isfinite(sc) or sc <= 0 or sc > 3:
            errors.append(_line(f"{path}.emphasis.scale", f"must be > 0 and <= 3, got {_fmt_num(sc)}", "use 1.0-1.3"))
    if line.bar is not None:
        _check_color(f"{path}.bar.color", line.bar.color, errors)
        n_words = len(glyphs.parse_emphasis(line.text)[0].split(" "))
        for name, v in (("word", line.bar.word), ("from", line.bar.from_)):
            if v is not None and not 0 <= v < n_words:
                errors.append(_line(f"{path}.bar.{name}", f"{v} is not a word of the line", f"use 0 to {n_words - 1}"))
        if not math.isfinite(line.bar.pad) or line.bar.pad < 0:
            errors.append(_line(f"{path}.bar.pad", f"must be >= 0, got {_fmt_num(line.bar.pad)}", "use design px, e.g. 14"))
    if line.stroke is not None:
        _check_color(f"{path}.stroke.color", line.stroke.color, errors)
        w = line.stroke.width
        size = line.size if line.size is not None else glyphs.STYLE_METRICS.get(line.style, glyphs.STYLE_METRICS["serif"]).size
        cap = glyphs.SPREAD / glyphs.SDF_EM  # em of distance the glyph cells hold
        if not math.isfinite(w) or w < 0:
            errors.append(_line(f"{path}.stroke.width", f"must be >= 0, got {_fmt_num(w)}", "use design px, e.g. 8"))
        elif size and math.isfinite(size) and size > 0 and w > cap * size:
            errors.append(_line(f"{path}.stroke.width", f"{_fmt_num(w)} px is over {cap:.0%} of size {_fmt_num(size)}",
                                f"use at most {_fmt_num(round(cap * size, 1))}"))
    if isinstance(line.shadow, TextShadow):
        sh = line.shadow
        _check_color(f"{path}.shadow.color", sh.color, errors)
        for name in ("x", "y", "blur", "opacity"):
            v = getattr(sh, name)
            if not math.isfinite(v) or (name == "blur" and v < 0) or (name == "opacity" and not 0 <= v <= 1):
                need = "from 0 to 1" if name == "opacity" else ">= 0" if name == "blur" else "finite"
                errors.append(_line(f"{path}.shadow.{name}", f"must be {need}, got {_fmt_num(v)}", "use design px" if name != "opacity" else "use a fraction"))
    if line.glow is not None and (not math.isfinite(line.glow) or not 0 <= line.glow <= 1):
        errors.append(_line(f"{path}.glow", f"must be from 0 to 1, got {_fmt_num(line.glow)}", "use a fraction, e.g. 0.4"))
    if line.pos is not None:
        if not all(math.isfinite(v) for v in line.pos):
            errors.append(_line(f"{path}.pos", "must be finite", "use [x, y] in design px"))
        if line.y is not None:
            errors.append(_line(f"{path}.pos", "pos replaces y", "remove y"))
        if line.align != "center":
            errors.append(_line(f"{path}.pos", "pos replaces align", "remove align"))
    if line.rotate is not None and not math.isfinite(line.rotate):
        errors.append(_line(f"{path}.rotate", "must be finite", "use degrees, e.g. -3"))


CAPTION_MIN_WEIGHT = 600
CAPTION_MIN_SIZE = 0.05  # of the frame height
CAPTION_MAX_WORDS = 8  # an authored block this short, low in the frame, reads as a caption
STOP_ENDS = (".", "।")


def _line_y(line: TextLine) -> float | None:
    return line.pos[1] if line.pos is not None else line.y


def is_caption_block(spec: Spec, i: int, block: TextBlock, design_h: float) -> bool:
    """A block made by `captions`, or an authored one that looks like a caption: a few words, no `dot`
    (the lower-third marker), every line below 55% of the frame height."""
    if spec.caption_from is not None and i >= spec.caption_from:
        return True
    ys = [_line_y(ln) for ln in block.lines]
    words = sum(len(ln.text.split()) for ln in block.lines)
    return (bool(ys) and all(y is not None and y >= 0.55 * design_h for y in ys)
            and words <= CAPTION_MAX_WORDS and all(ln.dot is None for ln in block.lines))


def _caption_warnings(spec: Spec) -> list[str]:
    """Spec-only caption hygiene lines (#24): trailing full stops, static boxes, thin or small type."""
    from montaj.render import glyphs

    w, h = (int(x) for x in spec.video.size.lower().split("x"))
    design_h = glyphs.DESIGN_W * h / w
    out: list[str] = []
    for i, sub in enumerate(spec.subs):
        if sub.punctuation is None and sub.text.rstrip().endswith((".", ",", "।")) and not sub.text.rstrip().endswith(".."):
            out.append(f"WARN subs[{i}] ends with a full stop or comma; set punctuation: drop")
    for i, block in enumerate(spec.text):
        for j, line in enumerate(block.lines):
            plain = line.text.rstrip()
            if plain.endswith(STOP_ENDS) and not plain.endswith(".."):
                out.append(f"WARN text[{i}].lines[{j}] ends with a full stop")
        if not is_caption_block(spec, i, block, design_h):
            continue
        boxed = (block.backdrop is not None and block.backdrop.type == "solid") or any(ln.box == "chip" for ln in block.lines)
        if boxed:
            out.append(f"WARN caption-box text[{i}] uses a static box; use stroke + shadow")
        for j, line in enumerate(block.lines):
            if line.font is None:  # a display face is heavy by design
                weight = line.weight if line.weight is not None else glyphs.STYLE_METRICS.get(line.style, glyphs.STYLE_METRICS["serif"]).weight
                if weight < CAPTION_MIN_WEIGHT:
                    out.append(f"WARN caption-weight text[{i}].lines[{j}] weight {weight:g} (want {CAPTION_MIN_WEIGHT}+)")
            size = line.size if line.size is not None else glyphs.STYLE_METRICS.get(line.style, glyphs.STYLE_METRICS["serif"]).size
            if size < CAPTION_MIN_SIZE * design_h:
                out.append(f"WARN caption-size text[{i}].lines[{j}] size {size:g} (want {CAPTION_MIN_SIZE * design_h:.0f}+, "
                           f"{CAPTION_MIN_SIZE:.0%} of frame height)")
    return out


def text_warnings(spec: Spec) -> list[str]:
    """`WARN` lines for text that validates but will likely look wrong."""
    from montaj.render import glyphs

    out: list[str] = []
    out += _caption_warnings(spec)
    for i, block in enumerate(spec.text):
        for j, line in enumerate(block.lines):
            if line.stroke is None:
                continue
            size = line.size if line.size is not None else glyphs.STYLE_METRICS.get(line.style, glyphs.STYLE_METRICS["serif"]).size
            if size and line.stroke.width > 0.2 * size:
                out.append(f"WARN text[{i}].lines[{j}].stroke.width: {line.stroke.width:g} px is over 20% of "
                           f"size {size:g}; the outline will swallow the letters")
    return out


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
        if line.engine == "gpu":
            errors.append(_line(f"{p}.engine", "card lines are drawn by Chromium", "remove engine: gpu"))
        for name in gpu_fields(line):
            errors.append(_line(f"{p}.{name}", "card lines are drawn by Chromium, without styling",
                                "use a text block over a blank shot"))
        if line.fit or line.wrap is not None:
            errors.append(_line(f"{p}.fit", "fit and wrap are not available on card lines yet", "set size by hand, or use a text block"))
        _card_duration(line.at, fps, bpm, f"{p}.at", errors)
        if line.stagger is not None:
            _card_duration(line.stagger, fps, bpm, f"{p}.stagger", errors)
        if line.dur is not None:
            _card_duration(line.dur, fps, bpm, f"{p}.dur", errors)
        if line.sweep:
            for k, raw in enumerate(line.sweep):
                _card_duration(raw, fps, bpm, f"{p}.sweep[{k}]", errors)


def _design_frame(spec: Spec) -> tuple[float, float] | None:
    """(width, height) of the frame in design px: 1080 wide, the film's aspect."""
    m = _WH.match(str(spec.video.size))
    if not m or int(m.group(1)) <= 0:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    return 1080.0, h * 1080.0 / w


def _layer_source_size(spec: Spec, spec_dir: Path | None, layer: Layer) -> tuple[int, int] | None:
    """Pixel size of a layer's image (header only) or clip (probe), or None when unknown."""
    if spec_dir is None:
        return None
    if layer.clip is not None:
        info = _probe_clip(spec, spec_dir, layer.clip)
        return None if info is None else (info[0], info[1])
    if layer.image is None:
        return None
    from PIL import Image

    from montaj.project import find_photo

    try:
        with Image.open(find_photo(spec_dir / spec.assets, layer.image)) as im:
            return im.size
    except (FileNotFoundError, OSError, ValueError):
        return None


def _expr_texts(v) -> list[str]:
    """The expression strings of a `LayerExpr` property (none for a value or key list)."""
    if not isinstance(v, LayerExpr):
        return []
    return [x for x in (v.expr if isinstance(v.expr, list) else [v.expr]) if isinstance(x, str)]


def layer_rest(v, default: float) -> float:
    """A scalar layer property's resting value: itself, or the last key's (where the animation ends)."""
    if v is None or isinstance(v, LayerExpr):  # an expression has no rest value before it is bound
        return default
    return float(v[-1].v) if isinstance(v, list) else float(v)


def layer_rest_pos(layer: Layer) -> tuple[float, float]:
    pos = layer.pos
    p = pos[-1].v if isinstance(pos, list) else pos
    return float(p[0]), float(p[1])


def _layer_extent(spec: Spec, spec_dir: Path | None, layer: Layer) -> tuple[float, float, float, float] | None:
    """Screen bounding box (design px) of a layer at its set scale and rotation, or None when unknown."""
    frame = _design_frame(spec)
    if frame is None:
        return None
    if isinstance(layer.box, LayerExpr) or isinstance(layer.pos, LayerExpr):
        return None  # computed per frame: no fixed extent to check
    if layer.box is not None:
        bx = layer.box[-1].v if isinstance(layer.box, list) else layer.box
        x0, y0, x1, y1 = (float(v) for v in bx)
    elif layer.pos is not None:
        size = _layer_source_size(spec, spec_dir, layer)
        if size is None:
            return None
        cx, cy = layer_rest_pos(layer)
        x0, y0, x1, y1 = cx - size[0] / 2, cy - size[1] / 2, cx + size[0] / 2, cy + size[1] / 2
    else:
        x0, y0, x1, y1 = 0.0, 0.0, frame[0], frame[1]
    s = layer_rest(layer.scale, 1.0)
    th = math.radians(layer_rest(layer.rotate, 0.0))
    hw, hh = (x1 - x0) / 2 * s, (y1 - y0) / 2 * s
    ex = abs(math.cos(th)) * hw + abs(math.sin(th)) * hh
    ey = abs(math.sin(th)) * hw + abs(math.cos(th)) * hh
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - ex, cy - ey, cx + ex, cy + ey


def _anchor_keys(layer: Layer, p: str, start: int, parsed: dict[str, int],
                 grid: tuple[dict[str, float], int, float | None, float], errors: list[str]) -> None:
    """Whole shot-clock frames for keys with `on`: `beat` snaps shot start + `t` to the nearest beat of the grid,
    `word:N` is word N's start + `t`. Stored in `parsed` like any key time, for `resolve` and the order checks."""
    markers, fps, bpm, beat0 = grid
    for prop in LAYER_KEYED:
        keys = getattr(layer, prop)
        for j, key in enumerate(keys if isinstance(keys, list) else ()):
            if key.on is None:
                continue
            q = f"{p}.{prop}[{j}]"
            off = parse_dur(key.t, fps, bpm, f"{q}.t", errors) if isinstance(key.t, str) else 0
            if off is None:
                continue
            if key.on == "beat":
                if not bpm:
                    continue  # the beats-need-bpm error already says so
                bf = fps * 60.0 / bpm
                at = beat0 + round((start + off - beat0) / bf) * bf
            elif _WORD.match(key.on):
                if key.on not in markers:
                    continue  # resolve_markers already reported it
                at = markers[key.on] + off
            else:
                errors.append(_line(f"{q}.on", f"unknown {key.on!r}", "use beat or word:N"))
                continue
            frame = round(at) - start
            if frame < 0:
                errors.append(_line(f"{q}.on", f"{key.on} lands {-frame}f before the shot", "use a later beat or word, or another shot"))
            else:
                parsed[f"{q}.t"] = frame


def _bind_expr(text: str, start: int, grid: tuple[dict[str, float], int, float | None, float]) -> str:
    """`text` with this film's fps, bpm, beat grid and word times folded in (`montaj.expr.bind`)."""
    markers, fps, bpm, beat0 = grid
    return xp.bind(text, fps=fps, bpm=bpm, start=start, beat0=beat0,
                   word_frames={n: markers[f"word:{n}"] for n in xp.words(text) if f"word:{n}" in markers})


def bind_layer_exprs(layer: Layer, start: int, grid: tuple[dict[str, float], int, float | None, float]) -> Layer:
    """The layer with every `expr` bound (a no-op without one). Called by `resolve` after validation."""
    update = {}
    for prop in LAYER_KEYED:
        v = getattr(layer, prop)
        if isinstance(v, LayerExpr):
            parts = [_bind_expr(x, start, grid) for x in (v.expr if isinstance(v.expr, list) else [v.expr])]
            update[prop] = LayerExpr(expr=parts if isinstance(v.expr, list) else parts[0])
    return layer.model_copy(update=update) if update else layer


def _check_exprs(layer: Layer, p: str, start: int, hold: int | None,
                 grid: tuple[dict[str, float], int, float | None, float], errors: list[str]) -> None:
    """Syntax, arity, names and finiteness of every `expr`, with the token named. A word the audio lacks is
    reported by `resolve_markers`; a bad bpm by the beats-need-bpm check."""
    markers, fps, bpm, _ = grid
    for prop in LAYER_KEYED:
        v = getattr(layer, prop)
        if not isinstance(v, LayerExpr):
            continue
        q = f"{p}.{prop}.expr"
        arity = _KEY_ARITY.get(prop, 1)
        parts = v.expr if isinstance(v.expr, list) else [v.expr]
        if (len(parts) != arity) or (isinstance(v.expr, list) != (arity > 1)):
            errors.append(_line(q, {2: "needs [x, y]", 4: "needs [x0, y0, x1, y1]"}.get(arity, "needs one expression"),
                                "pos takes 2 expressions, box 4; other properties one string"))
            continue
        for j, text in enumerate(parts):
            qj = q if arity == 1 else f"{q}[{j}]"
            try:
                xp.validate(text)
                if (xp.uses(text, "beat", "bpm") and not bpm) or any(f"word:{n}" not in markers for n in xp.words(text)):
                    continue
                bound = _bind_expr(text, start, grid)
                for f in sorted({0, (hold or 60) // 2, max(0, (hold or 60) - 1)}):
                    xp.evaluate(bound, {"frame": float(f)})
            except ExprError as e:
                errors.append(_line(qj, f"{text!r}: {e}", xp.HINT))
                break


def _layer_values(path: str, raw, parsed: dict[str, int], errors: list[str], hold: int | None):
    """Yield (path, value) for a layer property that is a value or a key list, checking the keys' shape,
    order, easing and shot-clock range. Scalars (`scale`) yield floats, `pos` yields pairs."""
    arity = _KEY_ARITY.get(path.rsplit(".", 1)[-1], 0)
    if raw is None or isinstance(raw, LayerExpr):  # expressions are checked by `_check_exprs`
        return
    if not isinstance(raw, list):
        yield path, raw
        return
    if not raw:
        errors.append(_line(path, "empty key list", "give at least one {t, v}"))
        return
    prev = None
    for j, key in enumerate(raw):
        q = f"{path}[{j}]"
        if (len(key.v) if isinstance(key.v, tuple) else 0) != arity:
            errors.append(_line(f"{q}.v", {2: "needs [x, y]", 4: "needs [x0, y0, x1, y1]"}.get(arity, "needs a number"),
                                "pos keys are [x, y], box keys [x0, y0, x1, y1]; other properties are numbers"))
        else:
            yield f"{q}.v", key.v
        if parse_ease(key.ease) is None:
            errors.append(_line(f"{q}.ease", f"unknown {key.ease!r}", f"{EASE_HINT}; or {', '.join(NEW_EASES)}"))
        t = parsed.get(f"{q}.t")
        if t is None:
            continue
        if prev is not None and t <= prev:
            errors.append(_line(f"{q}.t", f"{t}f is not after the previous key at {prev}f", "keys need strictly increasing times"))
        if hold is not None and t > hold:
            errors.append(_line(f"{q}.t", f"{t}f is after the end of the {hold}f hold", "use a time inside the hold"))
        prev = t


def _validate_mattes(spec: Spec, spec_dir: Path | None, errors: list[str]) -> None:
    """#36: `matte:` on clip shots, `mask:` on layers, `behind:` on text, and the matte files they read."""
    from montaj.project import shot_mattes

    for i, sh in enumerate(spec.shots):
        at = f"shots[{i}]"
        if sh.matte is not None:
            if sh.matte not in MATTES:
                errors.append(_line(f"{at}.matte", f"unknown {sh.matte!r}", f"use {_choices(MATTES)}"))
            elif sh.clip is None:
                errors.append(_line(f"{at}.matte", "matte needs a clip shot", "use mask: subject on a layer"))
        for k, layer in enumerate(sh.layers or ()):
            if layer.mask is None:
                continue
            p = f"{at}.layers[{k}].mask"
            if layer.mask not in MATTES:
                errors.append(_line(p, f"unknown {layer.mask!r}", f"use {_choices(MATTES)}"))
            if layer.shadow:
                errors.append(_line(f"{at}.layers[{k}].shadow", "a cut-out layer has no box to shadow", "remove shadow"))
    used = [(i, kind, stem) for i, sh in enumerate(spec.shots) for kind, stem in shot_mattes(sh)]
    for i, block in enumerate(spec.text):
        if block.behind is None:
            continue
        if block.behind not in MATTES:
            errors.append(_line(f"text[{i}].behind", f"unknown {block.behind!r}", f"use {_choices(MATTES)}"))
        elif not used:
            errors.append(_line(f"text[{i}].behind", "no shot has a subject matte",
                                "add matte: subject to the clip shot, or mask: subject to a layer"))
    if spec_dir is None or not used:
        return
    from montaj.matte import MatteMissing, find_matte
    from montaj.project import matte_source

    assets = Path(spec_dir) / spec.assets
    seen: set[tuple[str, str]] = set()
    for i, kind, stem in used:
        if (kind, stem) in seen or not isinstance(stem, str):
            continue
        seen.add((kind, stem))
        try:
            find_matte(assets, matte_source(Path(spec_dir), assets, kind, stem), stem)
        except MatteMissing as e:
            errors.append(_line(f"shots[{i}]", str(e), e.hint))
        except FileNotFoundError:
            pass  # the missing asset itself is reported by the asset check


def _validate_layers(
    spec: Spec, i: int, sh: Shot, hold: int | None, parsed: dict[str, int], times: dict[str, float],
    spec_dir: Path | None, errors: list[str], grid: tuple[dict[str, float], int, float | None, float],
) -> None:
    at = f"shots[{i}]"
    if not sh.layers:
        errors.append(_line(f"{at}.layers", "need at least one layer", "add {clip: stem} or {image: stem}"))
        return
    for name in ("crop", "focus", "zoom"):
        if getattr(sh, name) is not None:
            errors.append(_line(f"{at}.{name}", "not used on a layers shot", "set it on the layer"))
    frame = _design_frame(spec)
    ids = [layer.id for layer in sh.layers if layer.id is not None]
    for dup in sorted({x for x in ids if ids.count(x) > 1}):
        errors.append(_line(f"{at}.layers", f"id {dup!r} is used twice", "give each layer its own id"))
    parent_of = {layer.id: layer.parent for layer in sh.layers if layer.id is not None}
    for k, layer in enumerate(sh.layers):
        p = f"{at}.layers[{k}]"
        if layer.parent is not None:
            if layer.parent not in ids:
                errors.append(_line(f"{p}.parent", f"no layer has id {layer.parent!r}", f"have: {', '.join(ids) or 'none'}"))
            else:
                seen, cur = {layer.id}, layer.parent
                while cur is not None and cur in parent_of:
                    if cur in seen:
                        errors.append(_line(f"{p}.parent", "parents form a loop", "a layer cannot be its own ancestor"))
                        break
                    seen.add(cur)
                    cur = parent_of[cur]
        n_err = len(errors)
        if (layer.clip is None) == (layer.image is None):
            errors.append(_line(p, "need exactly one of clip or image", "set clip: or image:"))
        if layer.clip_in is not None and layer.clip is None:
            errors.append(_line(f"{p}.clip_in", "clip_in needs clip", "add clip:"))
        if layer.id is not None and not _NAME.match(layer.id):
            errors.append(_line(f"{p}.id", f"invalid name {layer.id!r}", "use [a-z][a-z0-9_]*"))
        if layer.anchor is not None and not _finite_all(layer.anchor):
            errors.append(_line(f"{p}.anchor", "must be finite", "use [x, y] design px from the layer centre"))
        if layer.box is not None and layer.pos is not None:
            errors.append(_line(p, "box and pos are mutually exclusive", "box sizes the layer; pos keeps its natural size"))
        if layer.fit is not None and layer.fit not in LAYER_FITS:
            errors.append(_line(f"{p}.fit", f"unknown {layer.fit!r}", f"use {_choices(LAYER_FITS)}"))
        if layer.fit is not None and layer.pos is not None:
            errors.append(_line(f"{p}.fit", "fit is for a box or the full frame", "remove fit or use box"))
        if layer.blend is not None and layer.blend not in LAYER_BLENDS:
            errors.append(_line(f"{p}.blend", f"unknown {layer.blend!r}", f"use {_choices(LAYER_BLENDS)}"))
        _anchor_keys(layer, p, parsed.get(f"{at}.start", 0), parsed, grid, errors)
        _check_exprs(layer, p, parsed.get(f"{at}.start", 0), hold, grid, errors)
        for name, lo, hi, hint in (("opacity", 0.0, 1.0, "use a value like 0.8"), ("scale", 1e-9, None, "use 0.5 for half size"),
                                   ("rotate", None, None, "use degrees, clockwise"),
                                   ("shadow", 0.0, 1.0, "use a value like 0.8"), ("radius", 0.0, None, "use design px")):
            raw = getattr(layer, name)
            for q, val in _layer_values(f"{p}.{name}", raw, parsed, errors, hold):
                if not math.isfinite(val) or (lo is not None and val < lo) or (hi is not None and val > hi):
                    want = {"opacity": "0 to 1", "scale": "finite and > 0", "rotate": "finite", "shadow": "0 to 1",
                            "radius": "finite and >= 0"}[name]
                    errors.append(_line(q, f"must be {want}, got {_fmt_num(val)}", hint))
        if isinstance(layer.box, (list, LayerExpr)) and layer.fit == "contain":
            errors.append(_line(f"{p}.fit", "contain letterboxes inside a fixed box", "use a fixed box, or fit: cover"))
        for q, bx in _layer_values(f"{p}.box", layer.box, parsed, errors, hold):
            x0, y0, x1, y1 = bx
            if not _finite_all(bx) or x1 <= x0 or y1 <= y0:
                errors.append(_line(q.removesuffix(".v"), f"{[_fmt_num(v) for v in bx]} is not a box",
                                    "use [x0, y0, x1, y1] in design px with x0 < x1, y0 < y1"))
        for q, pt in _layer_values(f"{p}.pos", layer.pos, parsed, errors, hold):
            if not _finite_all(pt):
                errors.append(_line(q, "must be finite", "use [x, y] in design px"))
        if layer.focus is not None and not _finite_point(layer.focus):
            errors.append(_line(f"{p}.focus", "must be finite", "use [x, y] in source px"))
        for name, anim in (("in", layer.in_), ("out", layer.out)):
            if anim is None:
                continue
            if anim.type not in LAYER_ANIMS:
                errors.append(_line(f"{p}.{name}.type", f"unknown {anim.type!r}", f"use {_choices(LAYER_ANIMS)}"))
            d = parsed.get(f"{p}.{name}.dur")
            if d is not None and d <= 0:
                errors.append(_line(f"{p}.{name}.dur", f"must be > 0, got {d}f", "lengthen dur"))
        f0 = parsed.get(f"{p}.from", 0)
        f1 = parsed.get(f"{p}.to")
        if f1 is not None and f1 <= f0:
            errors.append(_line(f"{p}.to", f"{f1}f is not after from {f0}f", "to must be later than from"))
        if hold is not None:
            if f0 >= hold:
                errors.append(_line(f"{p}.from", f"{f0}f is at or after the end of the {hold}f hold", "use a time inside the hold"))
            if f1 is not None and f1 > hold:
                errors.append(_line(f"{p}.to", f"{f1}f is after the end of the {hold}f hold", "use a time inside the hold"))
        if len(errors) > n_err:
            continue
        if frame is not None:
            ext = _layer_extent(spec, spec_dir, layer)
            if ext is not None and (ext[2] <= 0 or ext[3] <= 0 or ext[0] >= frame[0] or ext[1] >= frame[1]):
                errors.append(_line(p, "layer is entirely off-frame",
                                    f"keep the box inside 0..{_fmt_num(frame[0])} x 0..{_fmt_num(frame[1])} design px"))
        if layer.clip is None or spec_dir is None:
            continue
        info = _probe_clip(spec, spec_dir, layer.clip)
        if info is None:
            continue
        w, h, seconds = info
        if layer.focus is not None and not (0 <= layer.focus[0] <= w and 0 <= layer.focus[1] <= h):
            errors.append(_line(f"{p}.focus", f"[{_fmt_num(layer.focus[0])}, {_fmt_num(layer.focus[1])}] is outside the {w}x{h} source", "focus is in source px"))
        fps = spec.video.fps
        end = f1 if f1 is not None else hold
        if end is None or fps <= 0:
            continue
        needs = (times.get(f"{p}.clip_in", 0.0) + end - f0) / fps
        if needs > seconds + 1.5 / fps:
            errors.append(_line(p, f"clip ends at {seconds:.1f}s of source, needs {needs:.1f}s", "shorten the layer or lower clip_in"))


def layer_warnings(spec: Spec, spec_dir: Path | None = None) -> list[str]:
    """`WARN` lines for layers that validate but sit partly off-frame, and grade fallbacks (#44)."""
    from montaj.grade import grade_warnings

    frame = _design_frame(spec)
    if frame is None:
        return grade_warnings(spec, spec_dir)
    out: list[str] = grade_warnings(spec, spec_dir)
    for i, sh in enumerate(spec.shots):
        for k, layer in enumerate(sh.layers or ()):
            ext = _layer_extent(spec, spec_dir, layer)
            if ext is None:
                continue
            eps = 1e-6
            if ext[0] < -eps or ext[1] < -eps or ext[2] > frame[0] + eps or ext[3] > frame[1] + eps:
                box = ", ".join(_fmt_num(round(v, 1)) for v in ext)
                out.append(f"WARN shots[{i}].layers[{k}]: partly off-frame (spans [{box}] in a "
                           f"{_fmt_num(frame[0])}x{_fmt_num(round(frame[1], 1))} frame)")
    return out


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
    pace = spec.check.pacing
    if pace is not None:
        for name in ("max_shot", "hook_text_by"):
            raw = getattr(pace, name)
            if raw is not None:
                _dur_frames(raw, fps, bpm, f"check.pacing.{name}", errors)
        if pace.min_per10s is not None and not (math.isfinite(pace.min_per10s) and pace.min_per10s >= 0):
            errors.append(_line("check.pacing.min_per10s", f"must be >= 0, got {_fmt_num(pace.min_per10s)}",
                                "use min_per10s: 4"))
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
        kind = "black" if item.black is not None else "frozen" if item.frozen is not None else "caption"
        pair = item.black or item.frozen or item.caption
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


def _finite_all(vals) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def _check_waveform_source(spec: Spec, ov: Overlay, path: str, errors: list[str]) -> None:
    """`source` is `music`, an `audio.clips[].id` or a linked shot's `id`. No audio block is its own error."""
    if spec.audio is None:
        errors.append(_line(f"{path}.source", "no audio block", "add audio: with a track or clips"))
        return
    if ov.source is None or ov.source == "music":
        return  # a missing source is already "required for waveform"
    from montaj.audio.mix import linked_ids

    ids = sorted({c.id for c in spec.audio.clips if c.id} | linked_ids(spec))
    if ov.source not in ids:
        errors.append(_line(
            f"{path}.source",
            f"unknown source {ov.source!r}",
            f"use music or a clip id (have: {', '.join(ids) or 'none'})",
        ))


def _validate_overlays(
    spec: Spec,
    markers: dict[str, float],
    fps: int,
    bpm: float | None,
    times: dict[str, float],
    errors: list[str],
    *,
    beat0: float = 0.0,
) -> None:
    # Fields each shape reads; anything else set on it is a mistake worth naming.
    uses = {
        "ring": {"center", "radius", "sweep"},
        "brackets": {"box", "arm", "draw"},
        "line": {"points", "draw"},
        "waveform": {"box", "source", "bars"},
    }
    needs = {
        "ring": ("center", "radius"),
        "brackets": ("box",),
        "line": ("points",),
        "waveform": ("box", "source"),
    }
    shape_fields = {"center", "radius", "sweep", "box", "arm", "points", "draw", "source", "bars"}
    for i, ov in enumerate(spec.overlays):
        p = f"overlays[{i}]"
        spans: dict[str, float] = {}
        for field, raw in (("from", ov.from_), ("to", ov.to)):
            val = _parse_time(raw, markers, fps, bpm, f"{p}.{field}", errors, beat0=beat0)
            if val is not None:
                times[f"{p}.{field}"] = val
                spans[field] = val
        if len(spans) == 2 and spans["to"] <= spans["from"]:
            errors.append(_line(f"{p}.to", "must be after from", "lengthen the window"))
        if ov.draw is not None:
            d = _dur_frames(ov.draw, fps, bpm, f"{p}.draw", errors)
            if d is not None:
                if d <= 0:
                    errors.append(_line(f"{p}.draw", "must be > 0", "use 0.3s, or omit draw"))
                times[f"{p}.draw"] = d
        if ov.shape not in OVERLAY_SHAPES:
            errors.append(_line(f"{p}.shape", f"unknown {ov.shape!r}", f"use {_choices(OVERLAY_SHAPES)}"))
        else:
            for name in needs[ov.shape]:
                if getattr(ov, name) is None:
                    errors.append(_line(f"{p}.{name}", f"required for {ov.shape}", f"set {name}"))
            for name in sorted(shape_fields - uses[ov.shape]):
                if getattr(ov, name) is not None:
                    errors.append(_line(f"{p}.{name}", f"not used by {ov.shape}", f"remove {name}"))
            if ov.shape == "waveform":
                _check_waveform_source(spec, ov, p, errors)
                if ov.bars is not None and not 1 <= ov.bars <= WAVEFORM_BARS_MAX:
                    errors.append(_line(
                        f"{p}.bars",
                        f"must be from 1 to {WAVEFORM_BARS_MAX}, got {ov.bars}",
                        f"use {WAVEFORM_BARS}",
                    ))
        if ov.center is not None and not _finite_all(ov.center):
            errors.append(_line(f"{p}.center", "must be finite", "use [x, y] in design px"))
        if ov.radius is not None:
            rs = ov.radius if isinstance(ov.radius, tuple) else (ov.radius,)
            if not _finite_all(rs) or min(rs) < 0:
                errors.append(_line(f"{p}.radius", "must be finite and >= 0", "use 220 or [40, 220]"))
        if ov.sweep is not None:
            ss = ov.sweep if isinstance(ov.sweep, tuple) else (ov.sweep,)
            if not _finite_all(ss) or min(ss) < 0 or max(ss) > 360:
                errors.append(_line(f"{p}.sweep", "degrees must be in 0..360", "use 90 or [0, 360]"))
        if ov.box is not None:
            x0, y0, x1, y1 = ov.box
            if not _finite_all(ov.box) or x1 <= x0 or y1 <= y0:
                errors.append(_line(f"{p}.box", "need [x0, y0, x1, y1] with x1 > x0 and y1 > y0", "design px"))
        if ov.arm is not None and (not math.isfinite(ov.arm) or ov.arm <= 0):
            errors.append(_line(f"{p}.arm", "must be finite and > 0", "use a design-px length such as 28"))
        if ov.points is not None:
            if len(ov.points) < 2:
                errors.append(_line(f"{p}.points", "need at least 2 points", "use [[x, y], [x, y]]"))
            elif not all(_finite_all(pt) for pt in ov.points):
                errors.append(_line(f"{p}.points", "must be finite", "use design px"))
        if not math.isfinite(ov.width) or ov.width <= 0:
            errors.append(_line(f"{p}.width", f"must be finite and > 0, got {_fmt_num(ov.width)}", "use 3"))
        if not math.isfinite(ov.opacity) or not 0.0 <= ov.opacity <= 1.0:
            errors.append(_line(f"{p}.opacity", f"must be in 0..1, got {_fmt_num(ov.opacity)}", "use 0.8"))
        if not math.isfinite(ov.glow) or ov.glow < 0:
            errors.append(_line(f"{p}.glow", f"must be finite and >= 0, got {_fmt_num(ov.glow)}", "use 0.6"))
        if not _HEX.match(ov.color):
            errors.append(_line(f"{p}.color", f"unknown colour {ov.color!r}", "use #rrggbb"))
        _check_ease(f"{p}.ease", ov.ease, errors)


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


def load_spec(path: Path | str, *, expand: bool = True) -> Spec:
    """Parse and validate. `expand=False` keeps a `captions` block unexpanded (for `montaj captions`)."""
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

    from montaj.expand import ExpandError, expand_spec

    try:
        data, smap = expand_spec(data, text, path, TEXT_STYLES)
    except ExpandError as e:
        raise SpecError(e.lines(text)) from e

    errors: list[str] = []
    spec: Spec | None = None
    try:
        spec = Spec.model_validate(data)
    except ValidationError as e:
        errors.extend(_from_pydantic(e))

    assets_rel = data.get("assets", "assets")
    assets_dir = path.parent / str(assets_rel)
    _collect_photos(data, assets_dir, path.parent, errors)

    if spec is not None and not errors and any(sh.cuts for sh in spec.shots):
        from montaj.cuts import expand_cuts

        spec = expand_cuts(spec, path.parent, errors)
    if spec is not None and not errors and (spec.emphasis or (spec.audio and spec.audio.sfx)):
        from montaj.emphasis import expand_emphasis

        spec = expand_emphasis(spec, path.parent, errors)
    if spec is not None and expand and spec.captions is not None and not errors:
        from montaj.captions import expand_captions

        spec = expand_captions(spec, path.parent, errors)
    if spec is not None and expand and spec.subs:
        from montaj.captions import normalise_subs

        spec = normalise_subs(spec)
    if spec is not None:
        validate_spec(spec, errors, spec_dir=path.parent)
    if errors:
        raise SpecError([smap.remap(e) for e in errors] if smap else errors)
    assert spec is not None
    return spec
