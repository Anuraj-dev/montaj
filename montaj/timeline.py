"""Resolve a Spec into a Timeline of FramePlans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from montaj.spec import Shot, Spec, SpecError, parse_wh, validate_spec


@dataclass(frozen=True)
class Still:
    shot: int


@dataclass(frozen=True)
class Wall:
    shot: int
    t0: int


@dataclass(frozen=True)
class Trans:
    kind: str
    a: Still | Wall | Trans
    b: Still | Wall | Trans
    t0: float
    dur: float
    axis: str = "x"
    center: tuple[float, float] | None = None


type Scene = Still | Wall | Trans


@dataclass(frozen=True)
class FramePlan:
    f: int
    scene: Scene
    flashes: tuple[tuple[int, float], ...]
    leaks: tuple[tuple[int, float, int], ...]
    intro: tuple[str, int] | None
    outro: tuple[str, int, int] | None


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


def _base_scene(shot: ResolvedShot) -> Still | Wall:
    if shot.kind == "wall":
        return Wall(shot=shot.index, t0=shot.start)
    return Still(shot=shot.index)


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

    def plan(self, f: int) -> FramePlan:
        if not 0 <= f < self.n_frames:
            raise IndexError(f"frame {f} out of range [0, {self.n_frames})")
        scene: Scene | None = None
        for sh in self.shots:
            if sh.start <= f < sh.end:
                scene = _base_scene(sh)
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
        return FramePlan(f=f, scene=scene, flashes=flashes, leaks=leaks, intro=intro, outro=outro)


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


def resolve(spec: Spec, spec_dir: Path) -> Timeline:
    del spec_dir
    errors: list[str] = []
    parsed = validate_spec(spec, errors)
    size = parse_wh(spec.video.size, "video.size", errors, even=True)
    if errors:
        raise SpecError(errors)
    assert size is not None

    starts: list[int] = []
    t = 0
    resolved: list[ResolvedShot] = []
    for i, sh in enumerate(spec.shots):
        hold = parsed[f"shots[{i}].hold"]
        kind = "wall" if sh.wall is not None else "still"
        resolved.append(
            ResolvedShot(index=i, start=t, end=t + hold, kind=kind, spec=_shot_with_frame_keys(sh, i, parsed))
        )
        starts.append(t)
        t += hold
    n_frames = t
    segments = [Segment(index=i, start=s.start, end=s.end) for i, s in enumerate(resolved)]

    windows: list[tuple[float, float, Trans]] = []
    for i, sh in enumerate(resolved):
        tr = sh.spec.in_
        if tr is None or tr.type == "cut":
            continue
        dur = float(parsed[f"shots[{i}].in.dur"])
        prev = resolved[i - 1]
        a, b = _base_scene(prev), _base_scene(sh)
        if tr.type in ("fade", "swirl"):
            t0 = float(sh.start)
            center = tr.center
            if tr.type == "swirl" and center is None:
                center = prev.spec.focus
            windows.append(
                (t0, t0 + dur, Trans(kind=tr.type, a=a, b=b, t0=t0, dur=dur, center=center))
            )
        elif tr.type == "whip":
            t0 = float(sh.start) - dur / 2
            windows.append(
                (t0, t0 + dur, Trans(kind="whip", a=a, b=b, t0=t0, dur=dur, axis=tr.axis))
            )

    flashes: list[tuple[int, float]] = []
    for sh in resolved:
        if sh.spec.flash is not None:
            flashes.append((sh.start, float(sh.spec.flash)))
    leaks: list[tuple[int, float, int]] = []
    for i, fx in enumerate(spec.fx):
        at = parsed[f"fx[{i}].at"]
        if fx.flash is not None:
            flashes.append((at, float(fx.flash)))
        if fx.leak is not None:
            leaks.append((at, float(fx.leak), int(fx.dir)))

    intro = None
    if spec.video.intro is not None:
        intro = (spec.video.intro.type, parsed["video.intro.dur"])
    outro = None
    if spec.video.outro is not None:
        dur = parsed["video.outro.dur"]
        outro = (spec.video.outro.type, dur, n_frames - dur)

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
    )
