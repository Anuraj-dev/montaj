"""Resolve a Spec into a Timeline of FramePlans."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from montaj.render.core import EASE as EASE_FN, key as eval_key
from montaj.spec import Shot, Spec, SpecError, parse_wh, round_frame, validate_spec

_TRACK_DEFAULT = {"dust": "linear", "glow": "linear", "bars": "sine"}


@dataclass(frozen=True)
class Still:
    shot: int
    span: tuple[int, int] | None = None  # (v0, v1) for drift/pulse/morph/frame; else None


@dataclass(frozen=True)
class Wall:
    shot: int
    t0: int


@dataclass(frozen=True)
class Clip:
    shot: int
    t0: int


@dataclass(frozen=True)
class Blank:
    """Shot-0 fade source. Renderer paints video.background."""


@dataclass(frozen=True)
class Trans:
    kind: str
    a: Still | Wall | Clip | Blank | Trans
    b: Still | Wall | Clip | Blank | Trans
    t0: float
    dur: float
    axis: str = "x"
    center: tuple[float, float] | None = None
    blur: float = 0.0


type Scene = Still | Wall | Clip | Blank | Trans


@dataclass(frozen=True)
class ResolvedLine:
    text: str
    style: str
    italic: bool
    gold: bool
    color: str | None
    size: float | None
    weight: float | None
    tracking: float | None
    line_height: float | None
    y: float
    at: float
    by: str
    stagger: float
    dur: float
    rise: float
    reveal: str
    sweep: tuple[float, float]
    shadow: bool
    indent: float = 0.0


@dataclass(frozen=True)
class ResolvedText:
    from_: float
    to: float
    fade_in: float
    fade_out: float
    blur: float
    lines: tuple[ResolvedLine, ...]


@dataclass(frozen=True)
class ResolvedSub:
    from_: float
    to: float
    text: str


@dataclass(frozen=True)
class FramePlan:
    f: int
    scene: Scene
    flashes: tuple[tuple[int, float], ...]
    leaks: tuple[tuple[int, float, int], ...]
    intro: tuple[str, int] | None
    outro: tuple[str, int, int] | None
    texts: tuple[int, ...] = ()
    subs: tuple[int, ...] = ()
    hits: tuple[tuple[float, float], ...] = ()
    bursts: tuple[tuple[int, float, float, float, int], ...] = ()
    dust: float = 0.0
    glow: float = 0.0
    bars: float = 0.0
    fade_out: tuple[int, int] | None = None


@dataclass
class ResolvedShot:
    index: int
    start: int
    end: int
    kind: str
    spec: Shot


@dataclass
class Segment:
    index: int
    start: int
    end: int


def _base_scene(shot: ResolvedShot, span: tuple[int, int] | None = None) -> Still | Wall | Clip | Blank:
    if shot.kind == "blank":
        return Blank()
    if shot.kind == "wall":
        return Wall(shot=shot.index, t0=shot.start)
    if shot.kind == "clip":
        return Clip(shot=shot.index, t0=shot.start)
    return Still(shot=shot.index, span=span)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(spans):
        if hi <= lo:
            continue
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _pos_id(key) -> tuple[str, tuple[float, float]] | None:
    if key.print is not None and key.at is not None:
        return (str(key.print), (float(key.at[0]), float(key.at[1])))
    return None


def _zoom_id(key) -> tuple[float, str | None] | None:
    if key.value is not None:
        return (float(key.value), None if key.print is None else str(key.print))
    return None


def _track_holds(keys, ident, local_end: int) -> list[tuple[int, int]]:
    """Local [lo, hi) where a key track holds one resolved identity."""
    resolved: list[tuple[int, object]] = []
    cur: object | None = None
    for key in keys:
        got = ident(key)
        if got is not None:
            cur = got
        if cur is None:
            continue
        resolved.append((int(key.t), cur))
    if not resolved:
        return [(0, local_end)]
    spans = [(0, resolved[0][0] + 1)]
    for (t0, v0), (t1, v1) in zip(resolved, resolved[1:]):
        if v0 == v1:
            spans.append((t0, t1 + 1))
    spans.append((resolved[-1][0], local_end))
    clipped = [(max(0, lo), min(local_end, hi)) for lo, hi in spans]
    return _merge_spans(clipped)


def _intersect_spans(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            if hi > lo:
                out.append((lo, hi))
    return _merge_spans(out)


def wall_static_spans(shot: ResolvedShot) -> list[tuple[int, int]]:
    """Absolute [start, end) frames where both wall camera tracks hold.

    Identity is spec-only: pos matches on print+at, zoom on value+print. A
    key without a value holds the previous one. Does not load photos.
    """
    wall = shot.spec.wall
    if wall is None:
        return []
    local_end = shot.end - shot.start
    pos = _track_holds(wall.camera.pos, _pos_id, local_end)
    zoom = _track_holds(wall.camera.zoom, _zoom_id, local_end)
    # Drop 1-frame key junctions (one track arriving, the other leaving).
    return [
        (lo + shot.start, hi + shot.start)
        for lo, hi in _intersect_spans(pos, zoom)
        if hi - lo > 1
    ]


@dataclass
class Timeline:
    fps: int
    size: tuple[int, int]
    n_frames: int
    shots: list[ResolvedShot]
    segments: list[Segment]
    spec: Spec
    _windows: list[tuple[float, float, Trans]]
    _flashes: list[tuple[int, float]]
    _leaks: list[tuple[int, float, int]]
    _intro: tuple[str, int] | None
    _outro: tuple[str, int, int] | None
    markers: dict[str, float] = field(default_factory=dict)
    texts: list[ResolvedText] = field(default_factory=list)
    subs: list[ResolvedSub] = field(default_factory=list)
    _hits: list[tuple[float, float]] = field(default_factory=list)
    _bursts: list[tuple[int, float, float, float, int]] = field(default_factory=list)
    _tracks: dict[str, list] = field(default_factory=dict)
    _fade_out: tuple[int, int] | None = None
    _spans: dict[int, tuple[int, int] | None] = field(default_factory=dict)

    def plan(self, f: int) -> FramePlan:
        if not 0 <= f < self.n_frames:
            raise IndexError(f"frame {f} out of range [0, {self.n_frames})")
        scene: Scene | None = None
        for sh in self.shots:
            if sh.start <= f < sh.end:
                scene = _base_scene(sh, self._spans.get(sh.index))
                break
        for w0, w1, tr in self._windows:
            if w0 <= f < w1:
                scene = tr
        if scene is None:
            raise IndexError(f"frame {f} has no scene")
        flashes = tuple((at, k) for at, k in self._flashes if 0 <= f - at < 6)
        leaks = tuple((at, k, d) for at, k, d in self._leaks if abs(f - at) <= 24)
        intro = self._intro if self._intro is not None and f < self._intro[1] else None
        outro = self._outro if self._outro is not None and f >= self._outro[2] else None
        texts = tuple(
            i for i, b in enumerate(self.texts) if b.from_ - 1 <= f <= b.to + 1
        )
        # film.html 431: opacity from (from−.2s) to (to+.3s); ±1 frame like text.
        subs = tuple(
            i
            for i, s in enumerate(self.subs)
            if s.from_ - 0.2 * self.fps - 1 <= f <= s.to + 0.3 * self.fps + 1
        )
        hit_hi = 1.2 * self.fps
        hits = tuple((at, k) for at, k in self._hits if -3 <= f - at < hit_hi)
        burst_hi = 3.4 * self.fps
        bursts = tuple(b for b in self._bursts if 0 <= f - b[1] <= burst_hi)
        dust = _r6(_track_at(self._tracks.get("dust") or [], f))
        # film.html 427: leaks fade out over the film's last 2 s. Folded into the plan so the
        # renderer never reads the film length and only the tail segments hash it.
        tail = min(1.0, max(0.0, (f / self.fps - (self.n_frames / self.fps - 2.0)) / 2.0))
        glow = _r6(_track_at(self._tracks.get("glow") or [], f) * (1 - tail))
        bars = _r6(_track_at(self._tracks.get("bars") or [], f))
        fade_out = self._fade_out if self._fade_out is not None and f >= self._fade_out[1] else None
        return FramePlan(
            f=f,
            scene=scene,
            flashes=flashes,
            leaks=leaks,
            intro=intro,
            outro=outro,
            texts=texts,
            subs=subs,
            hits=hits,
            bursts=bursts,
            dust=dust,
            glow=glow,
            bars=bars,
            fade_out=fade_out,
        )


def _r6(v: float) -> float:
    return round(v, 6)


def _track_at(keys: list, t: float) -> float:
    if not keys:
        return 0.0
    return float(eval_key(float(t), keys))


def _shot_with_frame_keys(sh: Shot, i: int, parsed: dict[str, int]) -> Shot:
    if sh.wall is None:
        return sh
    pos = [
        key.model_copy(update={"t": parsed[f"shots[{i}].wall.camera.pos[{k}].t"]})
        for k, key in enumerate(sh.wall.camera.pos)
    ]
    zoom = [
        key.model_copy(update={"t": parsed[f"shots[{i}].wall.camera.zoom[{k}].t"]})
        for k, key in enumerate(sh.wall.camera.zoom)
    ]
    cam = sh.wall.camera.model_copy(update={"pos": pos, "zoom": zoom})
    return sh.model_copy(update={"wall": sh.wall.model_copy(update={"camera": cam})})


def _shot_kind(sh: Shot) -> str:
    if sh.blank:
        return "blank"
    if sh.wall is not None:
        return "wall"
    if sh.clip is not None:
        return "clip"
    return "still"


def _visible_span(resolved: list[ResolvedShot], i: int, parsed: dict[str, int]) -> tuple[int, int] | None:
    """[v0, v1) for Ken Burns / pulse / morph: whip start → end of next transition.

    film.html Photo t0/t1 (lines 122–161): drift k runs across the visible overlap.
    """
    sh = resolved[i]
    spec = sh.spec
    if spec.drift is None and spec.pulse is None and spec.morph is None and spec.frame is None:
        return None
    v0 = float(sh.start)
    tr = spec.in_
    if tr is not None and tr.type == "whip":
        dur = parsed.get(f"shots[{i}].in.dur")
        if dur is not None:
            v0 = float(sh.start) - dur / 2.0
    v1 = float(sh.end)
    if i + 1 < len(resolved):
        nxt = resolved[i + 1]
        ntr = nxt.spec.in_
        if ntr is not None and ntr.type in ("fade", "swirl", "whip"):
            nd = parsed.get(f"shots[{i + 1}].in.dur")
            if nd is not None:
                v1 = float(nxt.start) + (nd / 2.0 if ntr.type == "whip" else nd)
    return (round_frame(v0), round_frame(v1))


def _resolve_texts(spec: Spec, times: dict[str, float], fps: int) -> list[ResolvedText]:
    out: list[ResolvedText] = []
    for i, block in enumerate(spec.text):
        frm = times[f"text[{i}].from"]
        to = times[f"text[{i}].to"]
        fi = times.get(f"text[{i}].fade_in", 0.0)
        fo = times.get(f"text[{i}].fade_out", 0.8 * fps)
        lines: list[ResolvedLine] = []
        for j, line in enumerate(block.lines):
            stagger = times.get(
                f"text[{i}].lines[{j}].stagger",
                (0.07 if line.by == "char" else 0.16) * fps,
            )
            dur = times.get(f"text[{i}].lines[{j}].dur", 0.9 * fps)
            if line.sweep:
                sweep = (times[f"text[{i}].lines[{j}].sweep[0]"], times[f"text[{i}].lines[{j}].sweep[1]"])
            else:
                sweep = (frm, to)
            lines.append(
                ResolvedLine(
                    text=line.text,
                    style=line.style,
                    italic=line.italic,
                    gold=line.gold,
                    color=line.color,
                    size=line.size,
                    weight=line.weight,
                    tracking=line.tracking,
                    line_height=line.line_height,
                    y=line.y,
                    at=times[f"text[{i}].lines[{j}].at"],
                    by=line.by,
                    stagger=stagger,
                    dur=dur,
                    rise=line.rise,
                    reveal=line.reveal,
                    sweep=sweep,
                    shadow=line.shadow,
                    indent=line.indent,
                )
            )
        out.append(ResolvedText(from_=frm, to=to, fade_in=fi, fade_out=fo, blur=block.blur, lines=tuple(lines)))
    return out


def _resolve_tracks(spec: Spec, times: dict[str, float]) -> dict[str, list]:
    if spec.tracks is None:
        return {}
    out: dict[str, list] = {}
    for name in ("dust", "glow", "bars"):
        keys = getattr(spec.tracks, name)
        built = []
        default = _TRACK_DEFAULT[name]
        for k, key in enumerate(keys):
            ease_name = key.ease or default
            built.append((times[f"tracks.{name}[{k}].t"], float(key.v), EASE_FN[ease_name]))
        out[name] = built
    return out


def resolve(spec: Spec, spec_dir: Path) -> Timeline:
    errors: list[str] = []
    parsed, times, markers = validate_spec(spec, errors, spec_dir=spec_dir)
    size = parse_wh(spec.video.size, "video.size", errors, even=True)
    if errors:
        raise SpecError(errors)
    assert size is not None

    t = 0
    resolved: list[ResolvedShot] = []
    for i, sh in enumerate(spec.shots):
        hold = parsed[f"shots[{i}].hold"]
        resolved.append(
            ResolvedShot(
                index=i,
                start=t,
                end=t + hold,
                kind=_shot_kind(sh),
                spec=_shot_with_frame_keys(sh, i, parsed),
            )
        )
        t += hold
    n_frames = t
    segments = [Segment(index=i, start=s.start, end=s.end) for i, s in enumerate(resolved)]
    spans = {i: _visible_span(resolved, i, parsed) for i in range(len(resolved))}

    windows: list[tuple[float, float, Trans]] = []
    for i, sh in enumerate(resolved):
        tr = sh.spec.in_
        if tr is None or tr.type == "cut":
            continue
        dur = float(parsed[f"shots[{i}].in.dur"])
        b = _base_scene(sh, spans[i])
        if i == 0:
            a = Blank()
        else:
            prev = resolved[i - 1]
            a = _base_scene(prev, spans[prev.index])
        if tr.type in ("fade", "swirl"):
            t0 = float(sh.start)
            center = tr.center
            if tr.type == "swirl" and center is None and i > 0:
                center = resolved[i - 1].spec.focus
            windows.append(
                (t0, t0 + dur, Trans(kind=tr.type, a=a, b=b, t0=t0, dur=dur, center=center, blur=tr.blur))
            )
        elif tr.type == "whip":
            t0 = float(sh.start) - dur / 2
            windows.append(
                (t0, t0 + dur, Trans(kind="whip", a=a, b=b, t0=t0, dur=dur, axis=tr.axis, blur=tr.blur))
            )

    flashes: list[tuple[int, float]] = []
    for sh in resolved:
        if sh.spec.flash is not None:
            flashes.append((sh.start, float(sh.spec.flash)))
    leaks: list[tuple[int, float, int]] = []
    hits: list[tuple[float, float]] = []
    bursts: list[tuple[int, float, float, float, int]] = []
    burst_i = 0
    for i, fx in enumerate(spec.fx):
        at_i = parsed.get(f"fx[{i}].at")
        at_f = times.get(f"fx[{i}].at", float(at_i) if at_i is not None else 0.0)
        if fx.flash is not None and at_i is not None:
            flashes.append((at_i, float(fx.flash)))
        if fx.leak is not None and at_i is not None:
            leaks.append((at_i, float(fx.leak), int(fx.dir)))
        if fx.hit is not None:
            hits.append((at_f, float(fx.hit)))
        if fx.burst is not None and fx.pos is not None:
            bursts.append((burst_i, at_f, float(fx.pos[0]), float(fx.pos[1]), int(fx.burst)))
            burst_i += 1

    intro = None
    if spec.video.intro is not None:
        intro = (spec.video.intro.type, parsed["video.intro.dur"])
    outro = None
    fade_out = None
    if spec.video.outro is not None:
        dur = parsed["video.outro.dur"]
        start = n_frames - dur
        if spec.video.outro.type == "fade":
            fade_out = (dur, start)
        else:
            outro = (spec.video.outro.type, dur, start)

    texts = _resolve_texts(spec, times, spec.video.fps)
    subs = [
        ResolvedSub(from_=times[f"subs[{i}].from"], to=times[f"subs[{i}].to"], text=sub.text)
        for i, sub in enumerate(spec.subs)
    ]

    return Timeline(
        fps=spec.video.fps,
        size=size,
        n_frames=n_frames,
        shots=resolved,
        segments=segments,
        spec=spec,
        _windows=windows,
        _flashes=flashes,
        _leaks=leaks,
        _intro=intro,
        _outro=outro,
        markers=markers,
        texts=texts,
        subs=subs,
        _hits=hits,
        _bursts=bursts,
        _tracks=_resolve_tracks(spec, times),
        _fade_out=fade_out,
        _spans=spans,
    )
