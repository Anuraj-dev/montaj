"""The audio mixer: beat-grid music edits, placed voice/SFX clips, ducking. Pure numpy, CPU only.

`audio.edit`, `audio.clips` and `audio.duck` resolve to a `MixPlan` (seconds, absolute paths).
`render_mix` turns the plan into one stereo 48 kHz float wav in `build/audio/`, keyed by a hash of
the plan and the input bytes; `encode.mux` then treats that wav exactly as it treats `audio.track`.
A spec without any of the three never reaches this module at render time (`MixPlan.active`).

`source_window` slices one source's part of that mix (same code, same duck and ceiling) for waveform levels.
Resolve stores those levels; the renderer does not call into here.

A clip shot with `audio: true` is a `Linked` placement (#40): its own sound, locked to the picture
through the same `clip_in` / range / speed / freeze the renderer reads, so a retime keeps lip sync.
`audio_lead` / `audio_tail` move the audio cut off the picture cut (J and L cuts); the neighbouring
linked shot's sound starts or stops at that moved cut. Speed is varispeed: pitch follows speed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from montaj.encode import FFMPEG, FFPROBE, EncodeError
from montaj.emphasis import SFX_ID
from montaj.spec import _DUR, _assets_dir, _dur_frames, _line, _parse_time

SR = 48000
MIX_REV = "1"  # bump when the mixed samples change for the same plan
CLIP_FADE_S = 0.015  # fade in/out on every placed clip
XFADE_S = 0.012  # equal-power crossfade centred on an edit join
LINK_XFADE_S = 0.020  # equal-power crossfade centred on the cut between two linked clip shots
DUCK_THRESHOLD_DB = -45.0  # a duck source counts as "speaking" above this RMS
DUCK_WINDOW_S = 0.120
PEAK_CEILING = 0.98  # a hotter mix is scaled down (and reported) before it is written
SYNTH = ("whoosh", "riser", "boom", "tick", "glitch", "pop", "click", "swish", "hit", "bass-drop", "shutter", "ding")
SRC_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm")
_INF = 1e12

_DB = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*dB$")
_SPAN = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(ms|s|f|b)$")
MAX_GAIN_DB = 120.0  # beyond this 10**(dB/20) is no sane level (and float32 overflows near 760 dB)
_EDIT_SLACK_S = 0.005  # probed duration vs decoded samples
_TIME_HINT = "use 12s, 360f, 16b or a marker"
_SPAN_HINT = "use 40ms, 0.5s, 12f or 1b"


# --- plan --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EditOp:
    kind: str  # "cut" | "loop"
    a: float  # seconds, in the music as it stands after the earlier edits
    b: float
    times: int = 1  # loop: total plays of [a, b)


@dataclass(frozen=True)
class Placed:
    id: str | None
    src: str  # absolute path, or "synth:<name>"
    stream: int
    in_s: float
    out_s: float | None
    at_s: float
    gain_db: float
    pan: float


@dataclass(frozen=True)
class Linked:
    """A clip shot's own sound over film samples `[a, b)`.

    Film sample `s` plays source frame `clip_in + pos(s * fps / SR - start)` (film-fps frames from
    the source start, the clock `-ss clip_in` gives the picture), where `pos` integrates `speed`
    and stops at `freeze`. `lag` is the audio stream's start after the container's, in seconds.
    """

    id: str | None
    src: str  # absolute path of the clip
    fps: int
    start: int  # shot start frame
    a: int
    b: int
    clip_in: float  # source frames
    speed: float | tuple[tuple[float, float, str], ...] | None
    freeze: float | None  # shot frames
    gain_db: float
    lag: float = 0.0
    xin: int = 0  # half-width (samples) of the crossfade with the linked shot before; 0: a plain fade-in
    xout: int = 0  # the same, with the linked shot after


@dataclass(frozen=True)
class DuckPlan:
    under: tuple[str, ...]
    depth_db: float
    attack_s: float
    release_s: float


@dataclass(frozen=True)
class MixPlan:
    track: str | None  # None: no music, a silent bed
    edits: tuple[EditOp, ...] = ()
    clips: tuple[Placed, ...] = ()
    duck: DuckPlan | None = None
    links: tuple[Linked, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.edits or self.clips or self.duck or self.links)


@dataclass(frozen=True)
class MixStats:
    duration: float
    peak_dbfs: float  # of the mix before any ceiling scaling
    clipped: bool  # the mix was scaled down to stay under the ceiling
    clips: int  # placements mixed in
    duck_db: float  # deepest reduction applied to the music (<= 0)
    hash: str = ""
    peak_s: float | None = None  # film time of the peak; None in stats written before #64
    peak_parts: tuple = ()  # ((label, dBFS), …) what each source adds at the peak sample, loudest first
    # when clipped: (kind, name, placements, dB) from `_peak_fix`: a cut that keeps the mix under the ceiling
    fix: tuple = ()


def _span_s(raw: object, fps: int, bpm: float | None, path: str, errors: list[str],
            *, signed: bool = False) -> float | None:
    """A span in seconds from `40ms`, `0.5s`, `12f` or `1b`."""
    if isinstance(raw, bool) or not isinstance(raw, str):
        errors.append(_line(path, "bare number" if isinstance(raw, (int, float)) else f"invalid {raw!r}",
                            _SPAN_HINT))
        return None
    m = _SPAN.match(raw.strip())
    if not m:
        errors.append(_line(path, f"invalid duration {raw!r}", _SPAN_HINT))
        return None
    n, unit = float(m.group(1)), m.group(2)
    if not math.isfinite(n) or (n < 0 and not signed):
        errors.append(_line(path, "negative duration", "use a positive duration"))
        return None
    if unit == "ms":
        return n / 1000.0
    if unit == "s":
        return n
    if unit == "f":
        return n / fps if fps > 0 else None
    if bpm is None or not math.isfinite(bpm) or bpm <= 0:
        return None  # validate_spec reports "video.bpm required" for any beat unit
    return n * 60.0 / bpm


def _db(raw: object, path: str, errors: list[str]) -> float | None:
    m = _DB.match(raw.strip()) if isinstance(raw, str) else None
    if not m:
        errors.append(_line(path, f"invalid level {raw!r}", "use -6dB"))
        return None
    value = float(m.group(1))
    if not math.isfinite(value) or abs(value) > MAX_GAIN_DB:
        errors.append(_line(path, f"level {raw!r} is out of range", f"use between -{MAX_GAIN_DB:.0f}dB and +{MAX_GAIN_DB:.0f}dB"))
        return None
    return value


def _edit_time(raw: object, fps: int, bpm: float | None, beat0: float, path: str,
               errors: list[str]) -> float | None:
    """Edit points are times in the music itself, so a marker (a film time) makes no sense."""
    if not isinstance(raw, str) or not _DUR.match(raw.strip()):
        errors.append(_line(path, f"invalid music time {raw!r}", "edit points are music times: 20b or 12.5s"))
        return None
    if fps <= 0:
        return None  # validate_spec reports video.fps; no time can be resolved without it
    frames = _parse_time(raw, {}, fps, bpm, path, errors, beat0=beat0)
    return None if frames is None else frames / fps


def edit_ops(edits: list, fps: int, bpm: float | None, beat0: float, errors: list[str]) -> list[EditOp]:
    """`audio.edit` as seconds. Appends to `errors`; a bad op is skipped."""
    ops: list[EditOp] = []
    for i, e in enumerate(edits):
        base = f"audio.edit[{i}]"
        kinds = [k for k in ("cut", "loop") if getattr(e, k) is not None]
        if len(kinds) != 1:
            errors.append(_line(base, "need exactly one of cut or loop", "use {cut: [20b, 24b]}"))
            continue
        kind = kinds[0]
        if e.times is not None and kind != "loop":
            errors.append(_line(f"{base}.times", "times needs loop", "use {loop: [8b, 16b], times: 2}"))
            continue
        pair = [_edit_time(raw, fps, bpm, beat0, f"{base}.{kind}[{j}]", errors)
                for j, raw in enumerate(getattr(e, kind))]
        if None in pair:
            continue
        a, b = pair
        if not 0 <= a < b:
            errors.append(_line(f"{base}.{kind}", "need 0 <= start < end", "use [20b, 24b]"))
            continue
        times = 1
        if kind == "loop":
            times = 2 if e.times is None else e.times
            if isinstance(times, bool) or not isinstance(times, int) or times < 2:
                errors.append(_line(f"{base}.times", f"need a whole number >= 2, got {times!r}",
                                    "times is the total plays of the section"))
                continue
        ops.append(EditOp(kind, a, b, times))
    return ops


def edit_range_error(ops: list[EditOp] | tuple[EditOp, ...], total_s: float) -> tuple[int, str] | None:
    """`(op index, message)` for the first op that reaches past the music as it stands then."""
    cur = total_s
    for i, op in enumerate(ops):
        if op.b > cur + _EDIT_SLACK_S:
            return i, (f"{op.kind} ends at {op.b:.3f}s but the music is {cur:.3f}s long at that point "
                       f"(earlier edits count)")
        span = op.b - op.a
        cur += -span if op.kind == "cut" else span * (op.times - 1)
    return None


def _pieces(ops: list[EditOp], total_s: float = _INF) -> list[tuple[float, float]]:
    """Source ranges `(start, length)` seconds that make up the edited music, in play order."""
    pieces: list[tuple[float, float]] = [(0.0, total_s)]

    def take(lo: float, hi: float) -> list[tuple[float, float]]:
        out, cur = [], 0.0
        for s, n in pieces:
            a, b = max(lo, cur), min(hi, cur + n)
            if a < b:
                out.append((s + (a - cur), b - a))
            cur += n
        return out

    for op in ops:
        before, middle, after = take(0.0, op.a), take(op.a, op.b), take(op.b, _INF)
        if op.kind == "cut":
            pieces = before + after
        else:
            pieces = before + middle * op.times + after
    return pieces


def shift_words(spec, words: dict[int, float], fps: int, bpm: float | None, beat0: float) -> dict[int, float]:
    """Word start seconds after `audio.edit`: a word lands where it is first played; cut words drop."""
    if spec.audio is None or not spec.audio.edit:
        return words
    errors: list[str] = []
    ops = edit_ops(spec.audio.edit, fps, bpm, beat0, errors)
    if errors or len(ops) != len(spec.audio.edit):
        return words  # validate_audio reports the bad edit
    pieces = _pieces(ops)
    out: dict[int, float] = {}
    for i, s in words.items():
        cur = 0.0
        for ps, n in pieces:
            if ps <= s < ps + n:
                out[i] = cur + (s - ps)
                break
            cur += n
    return out


# --- spec -> plan ------------------------------------------------------------------------------


def _find_src(spec_dir: Path, raw: str) -> Path | None:
    base = spec_dir / raw
    if base.is_file():
        return base
    for ext in SRC_EXTS:
        cand = base.with_name(base.name + ext)
        if cand.is_file():
            return cand
    return None


def time_strings(spec) -> list[str]:
    """Every audio time/duration string, so the beats-need-bpm check sees them."""
    a = spec.audio
    if a is None:
        return []
    out: list[str] = []
    for e in a.edit:
        out += [x for x in (e.cut or ()) + (e.loop or ()) if isinstance(x, str)]
    for c in a.clips:
        out += [x for x in ([c.at] if isinstance(c.at, str) else list(c.at)) if isinstance(x, str)]
        out += [x for x in (c.offset, c.in_, c.out) if isinstance(x, str)]
    if a.duck is not None:
        out += [x for x in (a.duck.attack, a.duck.release) if isinstance(x, str)]
    for sh in spec.shots:
        out += [x for x in (sh.audio_lead, sh.audio_tail) if isinstance(x, str)]
    return out


def audio_layers(sh) -> list:
    """The layers of a shot that carry their clip's own sound (`layers[].audio: true`)."""
    return [layer for layer in sh.layers or () if layer.audio and layer.clip is not None]


def has_links(spec) -> bool:
    """Any shot or layer mixes its clip's own sound."""
    return any(sh.audio or audio_layers(sh) for sh in spec.shots)


def linked_ids(spec) -> set[str]:
    """Ids of clip shots (and audio layers) whose own sound is in the mix: duck and waveform sources, like clip ids."""
    return {sh.id for sh in spec.shots if sh.audio and sh.id} | {
        layer.id for sh in spec.shots for layer in audio_layers(sh) if layer.id}


_LINK_FIELDS = ("gain", "audio_lead", "audio_tail", "id")


def check_links(spec, spec_dir: Path | None, holds: list[int | None], fps: int, bpm: float | None,
                errors: list[str]) -> None:
    """Shot-level linked audio fields. Runs after layout, so a lead or tail is checked against its neighbour."""
    n = len(spec.shots)
    lead: list[float] = [0.0] * n
    tail: list[float] = [0.0] * n
    for i, sh in enumerate(spec.shots):
        at = f"shots[{i}]"
        for k, layer in enumerate(sh.layers or ()):
            if not layer.audio:
                continue
            p = f"{at}.layers[{k}].audio"
            if layer.clip is None:
                errors.append(_line(p, "audio needs clip", "set it on a clip layer"))
                continue
            if spec.audio is None:
                errors.append(_line(p, "linked clip audio needs an audio block", "add audio: {track: ...}"))
            if spec_dir is not None:
                path = _link_src(spec, spec_dir, layer.clip)
                if path is not None and _audio_start(path) is None:
                    errors.append(_line(p, f"clip {layer.clip!r} has no audio stream",
                                        "remove audio: true, or use a clip with sound"))
        if not sh.audio:
            for name in _LINK_FIELDS:
                if getattr(sh, name) is not None:
                    errors.append(_line(f"{at}.{name}", f"{name} needs audio: true", "add audio: true"))
            continue
        if sh.clip is None:
            errors.append(_line(f"{at}.audio", "audio needs clip", "add clip:, or use audio.clips for a sound file"))
            continue
        if spec.audio is None:
            errors.append(_line(f"{at}.audio", "linked clip audio needs an audio block", "add audio: {track: ...}"))
        if sh.id is not None and not re.match(r"^[a-z][a-z0-9_]*$", sh.id):
            errors.append(_line(f"{at}.id", f"invalid id {sh.id!r}", "use [a-z][a-z0-9_]*"))
        if sh.gain is not None:
            _db(sh.gain, f"{at}.gain", errors)
        for name, out in (("audio_lead", lead), ("audio_tail", tail)):
            raw = getattr(sh, name)
            v = _span_s(raw, fps, bpm, f"{at}.{name}", errors) if raw is not None else None
            if v is not None:
                out[i] = v
        if spec_dir is not None:
            path = _link_src(spec, spec_dir, sh.clip)
            if path is not None and _audio_start(path) is None:
                errors.append(_line(f"{at}.audio", f"clip {sh.clip!r} has no audio stream",
                                    "remove audio: true, or use a clip with sound"))
    for i, sh in enumerate(spec.shots):
        if not sh.audio or fps <= 0:
            continue
        at = f"shots[{i}]"
        if lead[i] > 0:
            prev = holds[i - 1] if i > 0 else None
            if i == 0:
                errors.append(_line(f"{at}.audio_lead", "no shot before the first one to lead under", "remove audio_lead"))
            elif prev is not None and lead[i] * fps > prev + 1e-6:
                errors.append(_line(f"{at}.audio_lead", f"{lead[i] * fps:g}f is longer than shots[{i - 1}] ({prev}f)",
                                    "shorten audio_lead"))
            elif spec.shots[i - 1].audio and tail[i - 1] > 0:
                errors.append(_line(f"{at}.audio_lead", f"shots[{i - 1}].audio_tail already moves this audio cut",
                                    "keep one of audio_tail and audio_lead"))
        if tail[i] > 0:
            nxt = holds[i + 1] if i + 1 < n else None
            if i + 1 == n:
                errors.append(_line(f"{at}.audio_tail", "no shot after the last one to tail under", "remove audio_tail"))
            elif nxt is not None and tail[i] * fps > nxt + 1e-6:
                errors.append(_line(f"{at}.audio_tail", f"{tail[i] * fps:g}f is longer than shots[{i + 1}] ({nxt}f)",
                                    "shorten audio_tail"))


def _link_src(spec, spec_dir: Path, clip: str) -> Path | None:
    from montaj.project import AssetMissing, resolve_clip

    try:
        return resolve_clip(spec_dir, _assets_dir(spec, spec_dir) or spec_dir / "assets", clip)
    except AssetMissing:
        return None  # the clip check reports it


def _audio_start(path: Path) -> float | None:
    """Audio stream 0's start after the container's, seconds; None when the file has no audio."""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=start_time:format=start_time",
         "-of", "json", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    try:
        info = json.loads(proc.stdout.decode() or "{}")
    except ValueError:
        return None
    streams = info.get("streams") or []
    if not streams:
        return None

    def num(v: object) -> float:
        try:
            x = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return x if math.isfinite(x) else 0.0

    return num(streams[0].get("start_time")) - num((info.get("format") or {}).get("start_time"))


def _links(spec, spec_dir: Path | None, shots, fps: int, bpm: float | None) -> list[Linked]:
    """`Linked` placements from the laid-out shots (`.start`, `.end`, `.spec` with frame keys).

    The audio cut before shot `i` sits at its picture cut, moved back by its own `audio_lead` or on
    by the previous linked shot's `audio_tail`. `check_links` has already rejected bad values.
    An audio layer (#53) plays its clip from the layer's `from` (else that audio cut) to its `to`
    (else the next audio cut), at speed 1 from its `clip_in`, as the renderer reads the layer.
    """
    if spec_dir is None or fps <= 0:
        return []
    n = len(shots)
    linked = [bool(r.spec.audio and r.spec.clip) for r in shots]
    layered = [[] if linked[i] else audio_layers(r.spec) for i, r in enumerate(shots)]
    # A shot whose sound runs from its start / to its end joins its linked neighbour with a crossfade.
    heard_in = [linked[i] or any(not ly.from_ for ly in layered[i]) for i in range(n)]
    heard_out = [linked[i] or any(ly.to is None or shots[i].start + ly.to >= shots[i].end for ly in layered[i])
                 for i in range(n)]
    probed: dict[str, tuple[Path | None, float | None]] = {}

    def source(clip: str) -> tuple[Path | None, float | None]:
        if clip not in probed:
            path = _link_src(spec, spec_dir, clip)
            probed[clip] = (path, _audio_start(path) if path is not None else None)
        return probed[clip]

    def span(raw: str | None) -> int:
        v = _span_s(raw, fps, bpm, "", []) if raw is not None else None
        return 0 if v is None else round(v * SR)

    def cut(i: int) -> int:
        """Film sample of the audio cut between shots `i - 1` and `i`."""
        s = frame_sample(shots[i].start, fps)
        if i > 0 and linked[i - 1]:
            s += span(shots[i - 1].spec.audio_tail)
        if linked[i]:
            s -= span(shots[i].spec.audio_lead)
        return s

    half = round(LINK_XFADE_S * SR / 2)
    out: list[Linked] = []
    for i, r in enumerate(shots):
        sh = r.spec
        last = cut(i + 1) if i + 1 < n else frame_sample(r.end, fps) + span(sh.audio_tail if linked[i] else None)
        for layer in layered[i]:
            path, lag = source(layer.clip)
            if lag is None:
                continue
            start = r.start + (layer.from_ or 0)
            a = frame_sample(start, fps) if layer.from_ else cut(i)
            b = frame_sample(r.start + layer.to, fps) if layer.to is not None and r.start + layer.to < r.end else last
            clip_in = _dur_frames(layer.clip_in, fps, bpm, "", []) if layer.clip_in is not None else 0.0
            xin = half if not layer.from_ and i > 0 and heard_out[i - 1] and a > 0 else 0
            xout = half if b == last and i + 1 < n and heard_in[i + 1] else 0
            out.append(Linked(layer.id, str(path), fps, start, a, b, float(clip_in or 0.0), None, None, 0.0,
                              lag, xin, xout))
        if not linked[i]:
            continue
        path, lag = source(sh.clip)
        if lag is None:
            continue
        a = cut(i)
        raw_in = sh.range[0] if sh.range is not None else sh.clip_in
        clip_in = _dur_frames(raw_in, fps, bpm, "", []) if raw_in is not None else 0.0
        speed = sh.speed
        if isinstance(speed, list):
            speed = tuple((float(k.t), float(k.v), k.ease) for k in speed)
        elif speed is not None:
            speed = float(speed)
        gain = _db(sh.gain, "", []) if sh.gain is not None else 0.0
        # Back-to-back linked shots join with a crossfade, so a cut never clicks or dips.
        xin = half if i > 0 and heard_out[i - 1] and a > 0 else 0
        xout = half if i + 1 < n and heard_in[i + 1] else 0
        out.append(Linked(sh.id, str(path), fps, r.start, a, last, float(clip_in or 0.0), speed,
                          None if sh.freeze is None else float(sh.freeze), gain or 0.0, lag, xin, xout))
    return out


def plan_audio(spec, spec_dir: Path | None, markers: dict[str, float], fps: int, bpm: float | None,
               beat0: float, errors: list[str], shots=None) -> MixPlan | None:
    """Validate `audio.edit/clips/duck` and resolve them to seconds. None without an audio block.

    `shots` are the laid-out shots (`Timeline.shots`); with them, `audio: true` shots become links.
    """
    a = spec.audio
    if a is None:
        return None
    root = spec_dir or Path(".")
    assets = _assets_dir(spec, spec_dir)
    ops = edit_ops(a.edit, fps, bpm, beat0, errors)
    if ops and len(ops) == len(a.edit) and a.track:
        total = _track_seconds(root / a.track)
        bad = edit_range_error(ops, total) if total is not None else None
        if bad is not None:
            errors.append(_line(f"audio.edit[{bad[0]}]", bad[1], "keep edit points inside the track"))
    clips: list[Placed] = []
    for i, c in enumerate(a.clips):
        base = f"audio.clips[{i}]"
        src = _clip_src(c.src, root, assets, f"{base}.src", errors)
        if c.stream is not None and (isinstance(c.stream, bool) or c.stream < 0):
            errors.append(_line(f"{base}.stream", f"need an audio stream index >= 0, got {c.stream!r}", "use 0"))
        if c.id is not None and not re.match(r"^[a-z][a-z0-9_]*$", c.id):
            errors.append(_line(f"{base}.id", f"invalid id {c.id!r}", "use [a-z][a-z0-9_]*"))
        gain = _db(c.gain, f"{base}.gain", errors) if c.gain is not None else 0.0
        if c.pan is not None and not (math.isfinite(c.pan) and -1 <= c.pan <= 1):
            errors.append(_line(f"{base}.pan", f"must be in [-1, 1], got {c.pan!r}", "-1 left, 1 right"))
        t_in = _span_s(c.in_, fps, bpm, f"{base}.in", errors) if c.in_ is not None else 0.0
        t_out = _span_s(c.out, fps, bpm, f"{base}.out", errors) if c.out is not None else None
        if t_in is not None and t_out is not None and t_out <= t_in:
            errors.append(_line(f"{base}.out", "must be after in", "use out > in"))
        off = _span_s(c.offset, fps, bpm, f"{base}.offset", errors, signed=True) if c.offset is not None else 0.0
        raws = [c.at] if isinstance(c.at, str) else list(c.at)
        if not raws:
            errors.append(_line(f"{base}.at", "need at least one time", "use at: 2s or at: [2b, 4b]"))
        for j, raw in enumerate(raws):
            path = f"{base}.at" if isinstance(c.at, str) else f"{base}.at[{j}]"
            f = _parse_time(raw, markers, fps, bpm, path, errors, beat0=beat0)
            if None in (f, off, gain, t_in) or src is None or fps <= 0:
                continue
            clips.append(Placed(c.id, src, c.stream or 0, t_in, t_out, f / fps + off, gain, c.pan or 0.0))
    duck = None
    if a.duck is not None:
        d = a.duck
        ids = {c.id for c in a.clips if c.id} | linked_ids(spec)
        if not d.under:
            errors.append(_line("audio.duck.under", "need at least one clip id", "use under: [voice]"))
        for j, name in enumerate(d.under):
            if name not in ids:
                errors.append(_line(f"audio.duck.under[{j}]", f"unknown clip id {name!r}",
                                    f"have: {', '.join(sorted(ids)) or 'none'}"))
        depth = _db(d.depth, "audio.duck.depth", errors)
        if depth is not None and depth >= 0:
            errors.append(_line("audio.duck.depth", f"must be below 0 dB, got {d.depth}", "use -6dB"))
        att = _span_s(d.attack, fps, bpm, "audio.duck.attack", errors)
        rel = _span_s(d.release, fps, bpm, "audio.duck.release", errors)
        if None not in (depth, att, rel):
            duck = DuckPlan(tuple(d.under), depth, att, rel)
    links = tuple(_links(spec, spec_dir, shots, fps, bpm)) if shots is not None else ()
    return MixPlan(str((root / a.track).resolve()) if a.track else None, tuple(ops), tuple(clips), duck, links)


def _track_seconds(path: Path) -> float | None:
    """Container duration of the music, or None when it cannot be probed (reported elsewhere)."""
    if not path.is_file():
        return None
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    try:
        value = float(proc.stdout.decode().strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def _clip_src(raw: str, root: Path, assets: Path | None, path: str, errors: list[str]) -> str | None:
    if raw.startswith("synth:"):
        mine = _find_src(root / "sfx", raw[6:])  # a project sfx/ file overrides the built-in of that name
        if mine is not None:
            return str(mine.resolve())
        if raw[6:] not in SYNTH:
            errors.append(_line(path, f"unknown synth {raw[6:]!r}", f"have: {', '.join(SYNTH)}, or a file in sfx/"))
            return None
        return raw
    found = _find_src(root, raw)  # a spec-relative file wins over the same name in assets
    if found is None and assets is not None:
        found = _find_src(assets, raw)
    if found is None:
        where = "the spec file" if assets is None else f"the spec file or {assets}"
        errors.append(_line(path, f"{raw!r} not found",
                            f"path relative to {where} (extension optional) or synth:<{'|'.join(SYNTH)}>"))
        return None
    return str(found.resolve())


# --- synth -------------------------------------------------------------------------------------


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (peak / m)).astype(np.float32) if m > 0 else x.astype(np.float32)


def _lowpass_sweep(x: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    """One-pole lowpass with a per-sample cutoff in Hz."""
    a = 1.0 - np.exp(-2.0 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def synth(name: str) -> np.ndarray:
    """Seeded one-shot, mono float32 at 48 kHz. Same name, same samples."""
    rng = np.random.default_rng({n: 100 + i for i, n in enumerate(SYNTH)}[name])
    if name == "whoosh":
        n = int(0.7 * SR)
        t = np.linspace(0, 1, n)
        body = _lowpass_sweep(rng.standard_normal(n), 300.0 * (14.0 ** np.sin(np.pi * t) ** 0.8))
        return _norm(body * np.sin(np.pi * t) ** 2)
    if name == "riser":
        n = int(2.0 * SR)
        t = np.linspace(0, 1, n)
        phase = 2 * np.pi * np.cumsum(200.0 * 10.0 ** t) / SR
        tone = np.sin(phase) + 0.5 * np.sin(2 * phase)
        noise = _lowpass_sweep(rng.standard_normal(n), 400.0 * 20.0 ** t)
        return _norm((0.6 * tone + 0.8 * noise) * t**1.5)
    if name == "boom":
        n = int(1.6 * SR)
        t = np.arange(n) / SR
        freq = 35.0 + 70.0 * np.exp(-t * 14.0)
        sub = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 3.2)
        thump = _lowpass_sweep(rng.standard_normal(n), np.full(n, 220.0)) * np.exp(-t * 9.0)
        return _norm(sub + 0.6 * thump)
    if name == "tick":
        n = int(0.05 * SR)
        t = np.arange(n) / SR
        return _norm(np.sin(2 * np.pi * 2400.0 * t) * np.exp(-t * 120.0))
    if name in _SYNTH2:
        return _synth2(name, rng)
    n = int(0.25 * SR)  # glitch: sample-and-hold stutter with random segment lengths
    out = np.zeros(n)
    i = 0
    while i < n:
        seg = int(rng.integers(40, 1400))
        out[i : i + seg] = rng.standard_normal() if rng.random() < 0.5 else np.sin(np.arange(min(seg, n - i)) * rng.uniform(0.02, 0.4))
        i += seg
    return _norm(out * np.exp(-np.arange(n) / SR * 8.0))


_SYNTH2 = ("pop", "click", "swish", "hit", "bass-drop", "shutter", "ding")


def _synth2(name: str, rng: np.random.Generator) -> np.ndarray:
    """The #41 one-shots: short, dry, and level-normalised like the others."""
    def tt(sec: float) -> np.ndarray:
        return np.arange(int(sec * SR)) / SR

    if name == "pop":  # a bubble: falling sine with a fast decay
        t = tt(0.12)
        return _norm(np.sin(2 * np.pi * np.cumsum(900.0 * np.exp(-t * 18.0) + 180.0) / SR) * np.exp(-t * 28.0))
    if name == "click":  # a dry key click: two-tone burst
        t = tt(0.03)
        return _norm((np.sin(2 * np.pi * 3200.0 * t) + 0.6 * rng.standard_normal(len(t))) * np.exp(-t * 220.0))
    if name == "swish":  # a short airy sweep, quieter and faster than whoosh
        n = int(0.32 * SR)
        t = np.linspace(0, 1, n)
        body = _lowpass_sweep(rng.standard_normal(n), 800.0 * (9.0 ** t))
        return _norm(body * np.sin(np.pi * t) ** 1.5)
    if name == "hit":  # an impact: low thump plus a noise crack
        t = tt(0.5)
        thump = np.sin(2 * np.pi * np.cumsum(60.0 + 140.0 * np.exp(-t * 40.0)) / SR) * np.exp(-t * 9.0)
        crack = _lowpass_sweep(rng.standard_normal(len(t)), np.full(len(t), 3500.0)) * np.exp(-t * 60.0)
        return _norm(thump + 0.7 * crack)
    if name == "bass-drop":  # a long sub that falls an octave and a half
        t = tt(1.4)
        return _norm(np.sin(2 * np.pi * np.cumsum(30.0 + 110.0 * np.exp(-t * 3.5)) / SR) * np.exp(-t * 1.6))
    if name == "shutter":  # camera shutter: two clicks 55 ms apart
        out = np.zeros(int(0.16 * SR))
        for at in (0.0, 0.055):
            t = tt(0.045)
            k = int(at * SR)
            out[k:k + len(t)] += (rng.standard_normal(len(t)) + np.sin(2 * np.pi * 1800.0 * t)) * np.exp(-t * 110.0)
        return _norm(out)
    t = tt(1.1)  # ding: a bell, two inharmonic partials
    return _norm((np.sin(2 * np.pi * 1318.5 * t) + 0.4 * np.sin(2 * np.pi * 1318.5 * 2.76 * t)) * np.exp(-t * 4.5))


def peak_lead(src: str, stream: int = 0) -> float:
    """Seconds from the start of `src` to its loudest moment (4 ms RMS envelope): where a hit lands on a cut."""
    if src.startswith("synth:"):
        x = synth(src[6:])
    else:
        x = decode(src, stream)
        x = np.sqrt((x ** 2).mean(axis=1)) if x.ndim == 2 else x
    if len(x) == 0:
        return 0.0
    w = max(1, int(0.004 * SR))
    env = np.convolve(x.astype(np.float64) ** 2, np.ones(w) / w, mode="same")
    return float(np.argmax(env)) / SR


# --- IO ----------------------------------------------------------------------------------------


def decode(path: str, stream: int = 0, *, log: Path | None = None) -> np.ndarray:
    """Audio of `path` (or its `stream`-th audio stream) as float32 `(n, 2)` at 48 kHz."""
    if path.startswith("synth:"):
        mono = synth(path[6:])
        return np.stack([mono, mono], axis=1)
    # ffmpeg's mono -> stereo upmix drops 3 dB; decode mono as mono and duplicate it ourselves.
    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", f"a:{stream}", "-show_entries", "stream=channels",
         "-of", "csv=p=0", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    mono = probe.stdout.decode().strip() == "1"
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-i", path, "-map", f"0:a:{stream}",
           "-ac", "1" if mono else "2", "-ar", str(SR), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    err = proc.stderr.decode("utf-8", "replace")
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n{err}")
    if proc.returncode != 0 or not proc.stdout:
        tail = "\n".join(err.strip().splitlines()[-3:])
        raise EncodeError(f"mix: cannot read audio stream {stream} of {path}", tail)
    pcm = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    return np.repeat(pcm[:, None], 2, axis=1) if mono else pcm.reshape(-1, 2)


def write_wav(path: Path, audio: np.ndarray) -> None:
    """Stereo 32-bit float wav, written to a temp name and renamed."""
    data = np.ascontiguousarray(audio, dtype="<f4").tobytes()
    header = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " +
              struct.pack("<IHHIIHH", 16, 3, 2, SR, SR * 8, 8, 32) + b"data" + struct.pack("<I", len(data)))
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish(path, header + data)


def _publish(path: Path, data: bytes) -> None:
    """Write `data` to a unique temp name beside `path`, then rename: readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# --- mixing ------------------------------------------------------------------------------------


def _slice(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """`x[lo:hi]`, zero-padded where the range leaves the array."""
    out = np.zeros((hi - lo, x.shape[1]), dtype=x.dtype)
    a, b = max(lo, 0), min(hi, len(x))
    if a < b:
        out[a - lo : b - lo] = x[a:b]
    return out


def _edit_parts(ops: tuple[EditOp, ...] | list[EditOp], n: int) -> list[tuple[int, int]]:
    """Source sample ranges that concatenate to the edited music. Adjacent pieces merge."""
    parts: list[list[int]] = []
    for s, length in _pieces(list(ops), n / SR):
        s0 = round(s * SR)
        s1 = min(n, round((s + length) * SR))
        if s0 >= n or s1 <= s0:
            continue
        if parts and parts[-1][1] == s0:
            parts[-1][1] = s1
        else:
            parts.append([s0, s1])
    return [(a, b) for a, b in parts]


def apply_edits(music: np.ndarray, ops: tuple[EditOp, ...] | list[EditOp]) -> np.ndarray:
    """Splice `music` per `ops`. Joins get an equal-power crossfade that keeps the length exact."""
    n = len(music)
    bad = edit_range_error(ops, n / SR)
    if bad is not None:
        raise EncodeError(f"mix: audio.edit[{bad[0]}] {bad[1]}", "keep edit points inside the track")
    parts = _edit_parts(ops, n)
    if not parts:
        raise EncodeError("mix: audio.edit leaves no music", "check the cut/loop ranges against the track length")
    out = np.concatenate([music[a:b] for a, b in parts])
    half = round(XFADE_S * SR / 2)
    pos = 0
    for k in range(len(parts) - 1):
        pos += parts[k][1] - parts[k][0]
        h = min(half, (parts[k][1] - parts[k][0]) // 2, (parts[k + 1][1] - parts[k + 1][0]) // 2)
        if h < 1:
            continue
        theta = np.linspace(0.0, np.pi / 2, 2 * h, dtype=np.float32)[:, None]
        a = _slice(music, parts[k][1] - h, parts[k][1] + h)
        b = _slice(music, parts[k + 1][0] - h, parts[k + 1][0] + h)
        out[pos - h : pos + h] = a * np.cos(theta) + b * np.sin(theta)
    return out


def _placed_span(c: Placed, n_src: int) -> tuple[int, int, int]:
    """`(film start, source start, length)` after `in`/`out` and a negative `at` trim."""
    lo = round(c.in_s * SR)
    hi = n_src if c.out_s is None else min(n_src, round(c.out_s * SR))
    at = round(c.at_s * SR)
    if at < 0:  # drop the head that would play before the film starts
        lo -= at
        at = 0
    return at, lo, max(0, hi - lo)


def _shape_placement(x: np.ndarray, p0: int, length: int, gain_db: float, pan: float,
                     head_fade: bool = True, tail_fade: bool = True) -> None:
    """Fade, gain and pan `x` in place. `x` holds placement indices `[p0, p0+len(x))` of `length`.

    A side with its own crossfade (`head_fade` / `tail_fade` false) gets no plain fade.
    """
    fade_n = min(round(CLIP_FADE_S * SR), length // 2)
    if fade_n > 0 and len(x):
        ramp = np.sin(np.linspace(0.0, np.pi / 2, fade_n, dtype=np.float32)) ** 2
        idx = np.arange(p0, p0 + len(x))
        head = (idx < fade_n) & head_fade
        tail = (idx >= length - fade_n) & tail_fade
        if head.any():
            x[head] *= ramp[idx[head], None]
        if tail.any():
            x[tail] *= ramp[length - 1 - idx[tail], None]
    x *= np.float32(10 ** (gain_db / 20))
    if pan:
        ang = (pan + 1) * np.pi / 4
        x *= np.array([math.sqrt(2) * math.cos(ang), math.sqrt(2) * math.sin(ang)], dtype=np.float32)


def _render_clip(c: Placed, cache: dict, log: Path | None) -> tuple[int, np.ndarray]:
    """`(start sample, stereo samples)` for one placement, trimmed, faded, gained and panned."""
    key = (c.src, c.stream)
    if key not in cache:
        cache[key] = decode(c.src, c.stream, log=log)
    x = cache[key]
    at, src, length = _placed_span(c, len(x))
    if length <= 0:
        return at, np.zeros((0, 2), np.float32)
    piece = np.array(x[src:src + length], dtype=np.float32, copy=True)
    _shape_placement(piece, 0, length, c.gain_db, c.pan)
    return at, piece


def _source_frames(speed, freeze: float | None, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(source frames from clip_in, sounding)` at shot clock `u` (frames, may be negative).

    The integral of the speed curve, as `clipmap.source_index` takes it (16 midpoint steps a
    frame), unfloored. It stops at `freeze`, and from there the sound is muted.
    """
    stop = math.inf if freeze is None else float(freeze)
    uc = np.minimum(u, stop)
    live = u < stop
    if not isinstance(speed, tuple):
        return (1.0 if speed is None else float(speed)) * uc, live
    from montaj.render.core import EASE, key

    keys = [(t, v, EASE[e]) for t, v, e in speed]
    steps = 16
    j0 = min(0, math.floor(float(uc.min()) * steps) - 1) if len(uc) else 0
    j1 = max(0, math.ceil(float(uc.max()) * steps) + 1) if len(uc) else 0
    inc = np.array([float(key((j + 0.5) / steps, keys)) / steps for j in range(j0, j1)])
    grid = np.concatenate([[0.0], np.cumsum(inc)])
    grid -= grid[-j0]  # pos(0) = 0
    return np.interp(uc, np.arange(j0, j1 + 1) / steps, grid), live


def _read_at(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """`x` at fractional sample positions `q`, linearly interpolated; zero outside the array."""
    i0 = np.floor(q).astype(np.int64)
    frac = (q - i0).astype(np.float32)[:, None]
    pad = np.concatenate([np.zeros((1, x.shape[1]), x.dtype), x, np.zeros((1, x.shape[1]), x.dtype)])
    lo = np.clip(i0 + 1, 0, len(pad) - 1)
    hi = np.clip(i0 + 2, 0, len(pad) - 1)
    return pad[lo] * (1 - frac) + pad[hi] * frac


def _render_link(link: Linked, cache: dict, log: Path | None) -> tuple[int, np.ndarray]:
    """`(start sample, stereo samples)` for one linked shot: its source under its own clock."""
    key = (link.src, 0)
    if key not in cache:
        cache[key] = decode(link.src, 0, log=log)
    x = cache[key]
    a = max(0, link.a)  # nothing plays before the film starts
    if link.b <= a:
        return a, np.zeros((0, 2), np.float32)
    lo, hi = max(0, a - link.xin), link.b + link.xout  # a crossfade reads past the cut on both sides
    fps = link.fps
    if link.speed in (None, 1.0) and link.freeze is None:
        # Integer-exact: film sample s plays source sample s - shot start + clip_in.
        off = round((link.clip_in / fps - link.lag) * SR) - frame_sample(link.start, fps)
        piece = _slice(x, lo + off, hi + off).astype(np.float32, copy=True)
    else:
        s = np.arange(lo, hi, dtype=np.float64)
        pos, live = _source_frames(link.speed, link.freeze, s * fps / SR - link.start)
        piece = _read_at(x, ((link.clip_in + pos) / fps - link.lag) * SR).astype(np.float32)
        piece[~live] = 0.0
    n = len(piece)
    for width, head in ((link.xin, True), (link.xout, False)):  # equal-power: sin in, cos out
        w = min(2 * width, n // 2)
        if w > 0:
            theta = np.linspace(0.0, np.pi / 2, w, dtype=np.float32)[:, None]
            if head:
                piece[:w] *= np.sin(theta)
            else:
                piece[n - w :] *= np.cos(theta)
    _shape_placement(piece, 0, n, link.gain_db, 0.0, head_fade=not link.xin, tail_fade=not link.xout)
    return lo, piece


def frame_sample(frame: int, fps: int) -> int:
    """First 48 kHz sample of integer film frame `frame`."""
    return int(round(int(frame) * SR / float(fps)))


def _q_level(peak: float) -> float:
    """Clamp to 0..1 and quantise to 1/255. The same samples always land on the same step."""
    if not math.isfinite(peak) or peak <= 0.0:
        return 0.0
    q = int(math.floor(min(1.0, peak) * 255.0 + 0.5))
    return min(q, 255) / 255.0


def band_levels(samples: np.ndarray, bars: int) -> tuple[float, ...]:
    """Peak of each equal slice of one frame, mono mean of the channels, quantised to 1/255."""
    bars = int(bars)
    if bars <= 0:
        return ()
    n = int(samples.shape[0]) if getattr(samples, "ndim", 0) else 0
    if n <= 0:
        return (0.0,) * bars
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    edges = (np.arange(bars + 1) * n) // bars
    row: list[float] = []
    for i in range(bars):
        a, b = int(edges[i]), int(edges[i + 1])
        peak = float(np.max(np.abs(mono[a:b]))) if b > a else 0.0
        row.append(_q_level(peak))
    return tuple(row)


def duck_gain(source: np.ndarray, plan: DuckPlan, n: int) -> np.ndarray:
    """Per-sample linear music gain: `depth` while `source` speaks, eased by attack and release.

    The speech test is a 120 ms centred RMS window (so the duck leads the voice a little) against
    a fixed -45 dB threshold; the smoothing runs at 1 kHz and is interpolated to samples.
    """
    hop = SR // 1000
    m = len(source) // hop
    power = (source[: m * hop, 0] ** 2 + source[: m * hop, 1] ** 2) / 2 if m else np.zeros(0)
    power = power.reshape(m, hop).mean(axis=1) if m else power
    w = max(1, round(DUCK_WINDOW_S * 1000))
    rms = np.sqrt(np.convolve(np.pad(power, (w // 2, w - w // 2 - 1)), np.ones(w) / w, mode="valid"))
    speaking = (20 * np.log10(np.maximum(rms, 1e-9)) > DUCK_THRESHOLD_DB).astype(np.float64)
    ka = 1 - math.exp(-0.001 / plan.attack_s) if plan.attack_s > 0 else 1.0
    kr = 1 - math.exp(-0.001 / plan.release_s) if plan.release_s > 0 else 1.0
    state = np.empty(len(speaking))
    s = 0.0
    for i, target in enumerate(speaking):
        s += (ka if target > s else kr) * (target - s)
        state[i] = s
    t_ctrl = (np.arange(len(state)) + 0.5) * hop
    amount = np.interp(np.arange(n), t_ctrl, state) if len(state) else np.zeros(n)
    return 10 ** (plan.depth_db * amount / 20)


def _mix_parts(plan: MixPlan, cache: dict, log: Path | None) -> tuple[np.ndarray, list, float]:
    """`(music bed after edits and duck, [(clip, start, samples)], duck dB)`: what `mix_audio` sums."""
    if plan.track is None:
        music = np.zeros((0, 2), np.float32)
    else:
        key = (plan.track, 0)
        if key not in cache:
            cache[key] = decode(plan.track, 0, log=log)
        music = apply_edits(cache[key], plan.edits) if plan.edits else cache[key]
    placed = [(c, *_render_clip(c, cache, log)) for c in plan.clips]
    placed += [(link, *_render_link(link, cache, log)) for link in plan.links]
    total = max([len(music)] + [s + len(x) for _, s, x in placed])
    bed = np.zeros((total, 2), dtype=np.float32)
    bed[: len(music)] = music
    duck_db = 0.0
    if plan.duck is not None:
        src = np.zeros((total, 2), dtype=np.float32)
        for c, s, x in placed:
            if c.id in plan.duck.under:
                src[s : s + len(x)] += x
        g = duck_gain(src, plan.duck, total).astype(np.float32)
        bed *= g[:, None]
        duck_db = float(20 * np.log10(g.min())) if len(g) else 0.0
    return bed, placed, duck_db


TRACK = "music (track)"  # the bed in a check line; a clip may itself be called `music`


def _key(c: Placed | Linked) -> tuple[str, str]:
    """`(kind, name)` of the spec knob that sets a placement's level, as a check line names it.

    `sfx`: emphasis / `sfx: auto` sounds, all set by `audio.sfx_gain`; `id`: every clip (or linked
    shot) with that id; `src`: unnamed clips of that file (`synth:hit` as is).
    """
    if c.id == SFX_ID:
        return ("sfx", "")
    if c.id:
        return ("id", c.id)
    return ("src", c.src if c.src.startswith("synth:") else Path(c.src).name)


def _shown(key: tuple[str, str]) -> str:
    return {"track": TRACK, "sfx": "audio.sfx_gain"}.get(key[0], key[1])


def _db_of(v: float) -> float:
    return 20 * math.log10(abs(v)) if v else -120.0


def _cut_db(h: float) -> float:
    """dB for scale `h`, rounded up to 0.1 dB so the printed cut clears it."""
    return math.ceil((-20 * math.log10(h) + 0.005) * 10) / 10


def _max_scale(other: np.ndarray, part: np.ndarray) -> float | None:
    """Largest factor `g` in (0, 1] with `|other + g*part| <= PEAK_CEILING` on every sample, else None.

    Each sample bounds `g` to an interval holding 0 (once `other` alone fits), so the bounds intersect
    to `[0, H]` and any cut of at least `-20 log10 H` dB clears the whole mix, not just the peak.
    """
    if not len(other) or float(np.max(np.abs(other))) > PEAK_CEILING:
        return None
    nz = part != 0
    o, p = other[nz].astype(np.float64), part[nz].astype(np.float64)
    if not len(p):
        return None
    h = float(np.min(np.where(p > 0, (PEAK_CEILING - o) / p, (-PEAK_CEILING - o) / p)))
    return min(1.0, h) if h > 0 else None


def _peak_fix(full: np.ndarray, placed: list, plan: MixPlan, order: list[tuple[str, str]]) -> tuple:
    """`(kind, name, placements, dB)`: a cut that keeps `full` under the ceiling everywhere.

    Duck sources are never cut (a quieter voice moves the duck). The cheapest clip knob among the
    sources pushing the peak (`order`) wins, since the spec sets it; else the track (no spec field
    sets its level: re-compose or turn the file down); else every non-duck source together
    (`name` lists the spared duck ids). If even that cannot clear it, every source by the overshoot.
    """
    ducked = set(plan.duck.under) if plan.duck is not None else set()
    is_duck = [c.id in ducked for c, _, _ in placed]
    clip_best: tuple = ()
    for key in [k for k in order if k[0] != "track"]:
        part = np.zeros_like(full)
        n = 0
        for (c, s, x), duck in zip(placed, is_duck):
            if _key(c) == key and not duck:
                part[s : s + len(x)] += x
                n += 1
        if not n or (h := _max_scale(full - part, part)) is None:
            continue
        if not clip_best or _cut_db(h) < clip_best[3]:
            clip_best = (*key, n, _cut_db(h))
    if clip_best:
        return clip_best
    duck_part = np.zeros_like(full)
    for (c, s, x), duck in zip(placed, is_duck):
        if duck:
            duck_part[s : s + len(x)] += x
    placed_sum = np.zeros_like(full)
    for c, s, x in placed:
        placed_sum[s : s + len(x)] += x
    bed = full - placed_sum
    track = Path(plan.track).name if plan.track else ""
    if plan.track and (h := _max_scale(full - bed, bed)) is not None:
        return ("track", track, 0, _cut_db(h))
    if (h := _max_scale(duck_part, full - duck_part)) is not None:
        return ("all", ", ".join(sorted(ducked)), 0, _cut_db(h))
    return ("all", "", 0, _cut_db(PEAK_CEILING / float(np.max(np.abs(full)))))


def mix_audio(plan: MixPlan, *, log: Path | None = None) -> tuple[np.ndarray, MixStats]:
    bed, placed, duck_db = _mix_parts(plan, {}, log)
    for _, s, x in placed:
        bed[s : s + len(x)] += x
    peak = float(np.max(np.abs(bed))) if len(bed) else 0.0
    clipped = peak > PEAK_CEILING
    peak_s: float | None = None
    parts: list[tuple[str, float]] = []
    fix: tuple = ()
    if peak > 0:
        i, ch = divmod(int(np.argmax(np.abs(bed))), bed.shape[1])
        peak_s = i / SR
        at: dict[tuple[str, str], float] = {}
        for c, s, x in placed:
            if s <= i < s + len(x):
                at[_key(c)] = at.get(_key(c), 0.0) + float(x[i - s, ch])
        at = {("track", ""): float(bed[i, ch]) - sum(at.values()), **at}
        keyed = sorted(((k, v) for k, v in at.items() if abs(v) > 1e-3), key=lambda kv: -abs(kv[1]))
        parts = [(_shown(k), v) for k, v in keyed]
        if clipped:
            sign = math.copysign(1.0, float(bed[i, ch]))
            fix = _peak_fix(bed, placed, plan, [k for k, v in keyed if v * sign > 0])
        if clipped:
            bed *= np.float32(PEAK_CEILING / peak)
    peak_db = 20 * math.log10(peak) if peak > 0 else -120.0
    return bed, MixStats(len(bed) / SR, peak_db, clipped, len(placed), duck_db, "", peak_s,
                         tuple((n, round(_db_of(v), 2)) for n, v in parts), fix)


def source_window(plan: MixPlan, source: str, s0: int, s1: int, *, cache: dict | None = None) -> np.ndarray:
    """Stereo film-timeline samples `[s0, s1)` of one waveform source as it sits in the mix.

    `music` is the track after `audio.edit` and the duck; any other name is every placement
    with that id, summed. Both come from `_mix_parts` and take the mix's peak-ceiling scale, so
    the bars follow what `mix_audio` writes. A plan that is not `active` skips the mixer at
    render time, so it gets no scale. Loudnorm and `audio.fade_out` happen later, at mux.
    `cache` is the caller's dict, not module state.
    """
    if s1 < s0:
        raise ValueError(f"source window {s0}:{s1} is backwards")
    cache = {} if cache is None else cache
    if "parts" not in cache:
        bed, placed, _ = _mix_parts(plan, cache, None)
        scale = 1.0
        if plan.active:
            full = bed.copy()
            for _, s, x in placed:
                full[s : s + len(x)] += x
            peak = float(np.max(np.abs(full))) if len(full) else 0.0
            if peak > PEAK_CEILING:
                scale = PEAK_CEILING / peak
        cache["parts"] = (bed, placed, np.float32(scale))
    bed, placed, scale = cache["parts"]
    key = ("stem", source)
    if key not in cache:
        if source == "music":
            stem = bed
        else:
            stem = np.zeros_like(bed)
            for c, s, x in placed:
                if c.id == source:
                    stem[s : s + len(x)] += x
        cache[key] = stem * scale
    return _slice(cache[key], s0, s1)


def plan_hash(plan: MixPlan) -> str:
    """Digest of the plan and the bytes of every input file."""
    shas: dict[str, str] = {}
    for p in [*([plan.track] if plan.track else []), *(c.src for c in plan.clips), *(link.src for link in plan.links)]:
        if not p.startswith("synth:") and p not in shas:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            shas[p] = h.hexdigest()
    body = asdict(plan)
    if not plan.links:
        del body["links"]  # a plan without linked shots keeps the wav it had before links existed
    for link in body.get("links", ()):
        if not link["xin"] and not link["xout"]:
            del link["xin"], link["xout"]  # an uncrossfaded link keeps the wav it had before crossfades
    payload = json.dumps({"rev": MIX_REV, "plan": body, "shas": shas}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def render_mix(plan: MixPlan, build_dir: Path, *, log: Path | None = None) -> tuple[Path, MixStats]:
    """The mixed wav for `plan` in `build_dir/audio/`, reused when its hash already exists."""
    digest = plan_hash(plan)
    wav = build_dir / "audio" / f"{digest}.wav"
    meta = wav.with_suffix(".json")
    cached = json.loads(meta.read_text()) if wav.is_file() and meta.is_file() else {}
    if "peak_s" in cached:  # stats from before #64 lack the peak detail: mix again (same samples)
        parts = tuple(map(tuple, cached["peak_parts"]))
        stats = MixStats(**{**cached, "peak_parts": parts, "fix": tuple(cached["fix"])})
    else:
        audio, stats = mix_audio(plan, log=log)
        stats = MixStats(**{**asdict(stats), "hash": digest})
        write_wav(wav, audio)
        _publish(meta, json.dumps(asdict(stats)).encode())
    _publish(build_dir / "audio-mix.json", json.dumps(asdict(stats)).encode())
    return wav, stats


def _fix_text(fix: list | tuple) -> str:
    kind, name, n, cut = fix
    if kind == "sfx":
        return f"lower audio.sfx_gain by {cut:.1f} dB"
    if kind in ("id", "src"):
        what = f"clip {name}" if kind == "id" else f"{name} clip"
        return f"lower gain on {'every ' if n > 1 else 'the '}{what} by {cut:.1f} dB"
    if kind == "track":
        return f"re-compose {name} or turn the file down by {cut:.1f} dB (no spec field sets the track level)"
    spared = f" except duck sources ({name})" if name else ""
    return f"lower the track and every clip{spared} by {cut:.1f} dB"


def check_lines(stats: dict, *, loudnorm: bool = True) -> list[str]:
    """`montaj check` lines from `build/audio-mix.json`; `loudnorm` is the spec's `audio.loudnorm`."""
    parts = [f"{stats['clips']} clips"]
    if stats["duck_db"] < 0:
        parts.append(f"duck {stats['duck_db']:.1f} dB")
    lines = [f"OK audio-mix {', '.join(parts)}, peak {stats['peak_dbfs']:+.1f} dBFS"]
    if stats["clipped"]:
        # The ceiling scale is one clean gain on the whole mix (no distortion). With loudnorm at mux
        # the level comes back, so it is INFO; with loudnorm off the film really is quieter: WARN.
        scale = 20 * math.log10(PEAK_CEILING) - stats["peak_dbfs"]
        line = f"{'INFO' if loudnorm else 'WARN'} clip mix peak {stats['peak_dbfs']:+.1f} dBFS"
        if stats.get("peak_s") is not None:
            srcs = [f"{n} {v:+.1f}" for n, v in stats.get("peak_parts", ())]
            more = f", +{len(srcs) - 4} more" if len(srcs) > 4 else ""
            line += f" at {stats['peak_s']:.2f}s = {', '.join(srcs[:4])}{more}"
        line += f"; whole mix scaled {scale:.1f} dB, clean" + ("" if loudnorm else ", and loudnorm is off: "
                                                                "the film is that much quieter")
        if len(stats.get("fix") or ()) == 4:
            line += f" (to skip the scale {_fix_text(stats['fix'])})"
        lines.append(line)
    return lines


__all__ = ["DuckPlan", "EditOp", "Linked", "MixPlan", "MixStats", "Placed", "apply_edits", "band_levels",
           "audio_layers", "check_lines", "check_links", "decode", "duck_gain", "edit_ops", "edit_range_error",
           "frame_sample", "has_links", "linked_ids",
           "plan_audio", "plan_hash", "render_mix", "shift_words", "source_window", "synth"]
