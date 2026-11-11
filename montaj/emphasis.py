"""Word-anchored edits and automatic transition sound (#41).

`expand_emphasis` runs where `expand_captions` does (`spec.load_spec`, and `timeline.resolve` for a spec built
in code). It turns `emphasis:` and `audio.sfx` (`auto`, or a list of `auto`/`text`/`cuts`) into the mechanisms that already exist: shot `effects`
(shake), clip `camera` keys (zoom), top-level `fx` (flash), `audio.clips` (sounds) and `captions.overrides`
(accent). The expansion is what the segment hash and the mix hash see; `emphasis` and `sfx` are cleared.

Layout decides which shot a word falls in, so a dry `validate_spec` of the spec without them runs first.
Everything is anchored to a marker (`em<i>` = `word:<i>`) or a grid expression, so a word still lands after
`audio.edit`. A sound's start is its target time minus its own peak lead (`mix.peak_lead`): the loudest
moment of the sample, not its first sample, lands on the word or the cut.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from montaj.spec import (
    Audio, AudioClip, CameraKey, CaptionOverride, Effect, Fx, Spec, _dur_frames, _line, _parse_time, validate_spec,
)

SFX_ID = "sfx"
DEFAULT_SFX_GAIN = "-8dB"
SHAKE_DUR = "8f"
ZOOM_PRE, ZOOM_POST = 4, 12  # frames: ramp to the peak, then back to the shot's own zoom
EVERY = re.compile(r"^every\s+(\d+(?:\.\d+)?)([bsf])$")
MATCH = re.compile(r"^(.+?)(?:#([1-9]\d*))?$")
AUTO_WHOOSH = ("whip", "push", "zoom")  # transitions that get a whoosh; its peak lands on the cut / window middle
DROP_MARKER = "drop"
CLOSE_S = 0.080  # two sfx peaks nearer than this smear into one hit (`check` warns)
TEXT_HIT, CUT_HIT = "hit", "swish"  # `audio.sfx: [text]` on a slam's impact, `[cuts]` on a hard cut (#62)


@dataclass(frozen=True)
class _Event:
    expr: str  # time expression the actions anchor to
    entry: int  # index in `emphasis`
    word: int | None  # word index (caption accent), None for `every`
    seed: int


def load_words(spec: Spec, spec_dir: Path | None, errors: list[str]):
    """`[Word(i, w, s, e)]` from `audio.markers`, else from `captions.source`; [] with `errors` filled."""
    from montaj.captions import _source_path, read_words
    from montaj.cuts import film_words

    if spec_dir is None:
        errors.append(_line("emphasis", "needs the spec on disk", "load the spec from a file"))
        return []
    if spec.audio is not None and spec.audio.markers:
        path = spec_dir / spec.audio.markers
        return film_words(spec, spec_dir, path, read_words(path, errors, "audio.markers"))
    if spec.captions is not None:
        path = _source_path(spec, spec_dir, errors)
        return film_words(spec, spec_dir, path, read_words(path, errors)) if path is not None else []
    errors.append(_line("emphasis", "has no words", "set audio.markers, or captions.source"))
    return []


def _norm(text: str) -> str:
    return re.sub(r"[^\w']+", "", text.lower())


def find_match(words, text: str, nth: int = 1) -> int | None:
    """Word index `i` of the `nth` run of words equal to `text` (case and punctuation ignored)."""
    want = [_norm(t) for t in text.split()]
    want = [t for t in want if t]
    if not want:
        return None
    toks = [_norm(w.w) for w in words]
    hits = 0
    for p in range(len(toks) - len(want) + 1):
        if toks[p:p + len(want)] == want:
            hits += 1
            if hits == nth:
                return words[p].i
    return None


def _word_marker(spec: Spec, w, spec_dir: Path | None = None) -> str:
    if spec.audio is not None and spec.audio.markers:
        from montaj.cuts import _cut_clip_of

        if spec_dir is None or _cut_clip_of(spec, spec_dir, spec_dir / spec.audio.markers) is None:
            return f"word:{w.i}"
        return f"{w.s:.4f}s"  # `word:i` reads source time; a cut clip's word is already in film time
    offset = 0.0
    if spec.captions is not None and spec.captions.offset:
        frames = _dur_frames(spec.captions.offset, spec.video.fps, None, "captions.offset", [])
        offset = (frames or 0.0) / spec.video.fps
    return f"{w.s + offset:.4f}s"


def _entries(spec: Spec, words, errors: list[str], spec_dir: Path | None = None) -> tuple[list[_Event], dict[str, str]]:
    """The events each `emphasis` entry anchors to, and the `em<i>` markers they need."""
    by_i = {w.i: w for w in words}
    events: list[_Event] = []
    markers: dict[str, str] = {}
    for n, e in enumerate(spec.emphasis or ()):
        path = f"emphasis[{n}]"
        if sum(x is not None for x in (e.word, e.match, e.on)) != 1:
            errors.append(_line(path, "need exactly one of word, match, on", "word: 12, match: \"never\" or on: every 8b"))
            continue
        if not any(x is not None for x in (e.zoom, e.shake, e.flash, e.sfx, e.caption)):
            errors.append(_line(path, "no action", "add zoom, shake, flash, sfx or caption"))
            continue
        if e.on is not None:
            if e.caption is not None:
                errors.append(_line(f"{path}.caption", "needs a word", "use word: or match:"))
            m = EVERY.match(e.on.strip())
            if not m or float(m.group(1)) <= 0:
                errors.append(_line(f"{path}.on", f"unknown {e.on!r}", "use every 8b, every 2s or every 12f"))
                continue
            events.append(_Event(e.on.strip(), n, None, n))
            continue
        i = e.word
        if e.match is not None:
            m = MATCH.match(e.match.strip())
            i = find_match(words, m.group(1), int(m.group(2) or 1)) if m else None
            if i is None:
                errors.append(_line(f"{path}.match", f"{e.match!r} not in the words", "montaj words --grep <text>"))
                continue
        if i not in by_i:
            have = f"{min(by_i)} to {max(by_i)}" if by_i else "none"
            errors.append(_line(f"{path}.word", f"word {i} not in the source", f"have {have}"))
            continue
        name = f"em{i}"
        mark = _word_marker(spec, by_i[i], spec_dir)
        if markers.setdefault(name, mark) != mark or (name in spec.markers and spec.markers[name] != mark):
            errors.append(_line("markers", f"{name!r} is reserved for emphasis", "rename that marker"))
        events.append(_Event(name, n, i, n))
    return events, markers


def _every_exprs(ev: _Event, n_frames: float, fps: int, bpm: float | None, beat0: float, errors: list[str]):
    """`(expr, frame)` for each grid point of an `every` event inside the film."""
    m = EVERY.match(ev.expr)
    num, unit = m.group(1), m.group(2)
    if unit == "b" and not bpm:
        errors.append(_line(f"emphasis[{ev.entry}].on", "beats need a grid", "set video.bpm or audio.markers"))
        return
    out = []
    for k in range(0, 2000):
        expr = f"{float(num) * k:g}{unit}"
        f = _parse_time(expr, {}, fps, bpm, "", [], beat0=beat0)
        if f is None or f >= n_frames:
            break
        out.append((expr, f))
    yield from out


def _shot_at(parsed: dict[str, int], count: int, f: float) -> int | None:
    fr = round(f)
    for i in range(count):
        if parsed.get(f"shots[{i}].start", 1 << 30) <= fr < parsed.get(f"shots[{i}].end", -1):
            return i
    return None


def _zoom_keys(sh, peaks: list[tuple[int, float]], end_local: int) -> list[CameraKey]:
    """Camera keys: the shot's own zoom, ramping to each peak exactly at its frame and back."""
    base = float(sh.zoom) if sh.zoom else 1.0
    keys: list[tuple[int, float]] = [(0, base)]
    peaks = sorted(peaks)
    for n, (u, z) in enumerate(peaks):
        nxt = peaks[n + 1][0] if n + 1 < len(peaks) else None
        if u - ZOOM_PRE > keys[-1][0]:
            keys.append((u - ZOOM_PRE, base))
        if u > keys[-1][0]:
            keys.append((u, z))
        elif u == keys[-1][0]:
            keys[-1] = (u, z)
        else:
            continue
        out = min(u + ZOOM_POST, (end_local if nxt is None else nxt - ZOOM_PRE) - 1)
        if out > keys[-1][0]:
            keys.append((out, base))
    return [CameraKey(t=f"{t}f", zoom=z, focus=sh.focus) for t, z in keys]


def _sfx_clip(spec: Spec, spec_dir: Path, name: str, expr: str, t_s: float, gain: str, path: str,
              errors: list[str]) -> AudioClip | None:
    """A clip whose loudest moment lands at `t_s` (and, when `expr` is set, anchored to it)."""
    from montaj.audio.mix import _clip_src, peak_lead
    from montaj.spec import _assets_dir

    raw = name if ("/" in name or "." in name) else f"synth:{name}"
    src = _clip_src(raw, spec_dir, _assets_dir(spec, spec_dir), path, errors)
    if src is None:
        return None
    lead = peak_lead(src)
    ms = round(lead * 1000)
    if t_s < lead:  # the film starts mid-sound: skip into the sample instead of dropping it
        return AudioClip(id=SFX_ID, src=raw, at="0s", in_=f"{round((lead - t_s) * 1000)}ms", gain=gain)
    return AudioClip(id=SFX_ID, src=raw, at=expr, offset=f"-{ms}ms", gain=gain)


def _auto_events(spec: Spec, parsed: dict[str, int], markers: dict[str, float], bpm, beat0) -> list[tuple[str, float]]:
    """`(sound, film frame of its peak)` for transitions, type reveals and a `drop` marker."""
    fps = spec.video.fps
    out: list[tuple[str, float]] = []
    for i, sh in enumerate(spec.shots):
        start = parsed.get(f"shots[{i}].start")
        if start is None:
            continue
        tr = sh.in_
        if tr is not None and tr.type in AUTO_WHOOSH and i > 0:
            dur = parsed.get(f"shots[{i}].in.dur", 0)
            out.append(("whoosh", float(start) if tr.type == "whip" else start + dur / 2))
        if sh.card is not None:
            for line in sh.card.text:
                if line.reveal == "type":
                    at = _dur_frames(line.at, fps, bpm, "", [])
                    if at is not None:
                        out.append(("tick", start + at))
    for block in spec.text:
        for line in block.lines:
            if line.reveal == "type":
                f = _parse_time(line.at, markers, fps, bpm, "", [], beat0=beat0)
                if f is not None:
                    out.append(("tick", f))
    if DROP_MARKER in markers:
        out.append(("riser", markers[DROP_MARKER]))
    return out


def _sfx_kinds(spec: Spec) -> set[str]:
    """`audio.sfx` as a set: `auto` alone is `{"auto"}`; a list names each kind."""
    sfx = spec.audio.sfx if spec.audio is not None else None
    return set() if sfx is None else {sfx} if isinstance(sfx, str) else set(sfx)


def _hit_events(spec: Spec, parsed: dict[str, int], markers: dict[str, float], bpm, beat0,
                kinds: set[str]) -> list[tuple[str, float]]:
    """`(sound, film frame of its peak)` for slam text impacts (`text`) and hard cuts (`cuts`).

    A slam line hits when its first unit lands: `at` + its `in.dur` (else `dur`, else the slam default). A hard
    cut is a shot after the first with no `in` transition.
    """
    from montaj.render.motion import IN_DUR

    fps = spec.video.fps
    out: list[tuple[str, float]] = []
    if "text" in kinds:
        for block in spec.text:
            for line in block.lines:
                if line.in_ is None or line.in_.type != "slam":
                    continue
                at = _parse_time(line.at, markers, fps, bpm, "", [], beat0=beat0)
                raw = line.in_.dur or line.dur
                dur = _dur_frames(raw, fps, bpm, "", []) if raw else IN_DUR["slam"] * fps
                if at is not None and dur is not None:
                    out.append((TEXT_HIT, at + dur))
    if "cuts" in kinds:
        for i, sh in enumerate(spec.shots):
            start = parsed.get(f"shots[{i}].start")
            if i > 0 and sh.in_ is None and start is not None:
                out.append((CUT_HIT, float(start)))
    return out


def _placed_peaks(base: Spec, spec_dir: Path, markers: dict[str, float], bpm, beat0) -> list[float]:
    """Film seconds where the author's own clips start or peak; ducked-under clips (voices) are not hits."""
    from montaj.audio.mix import peak_lead, plan_audio

    plan = plan_audio(base, spec_dir, markers, base.video.fps, bpm, beat0, [])
    if plan is None:
        return []
    voices = set(plan.duck.under) if plan.duck is not None else set()
    out: list[float] = []
    for c in plan.clips:
        if c.id in voices:
            continue
        out += [c.at_s, c.at_s + peak_lead(c.src, c.stream) - c.in_s]
    return out


def expand_emphasis(spec: Spec, spec_dir: Path | None, errors: list[str]) -> Spec:
    """`spec` without `emphasis` / `audio.sfx`, their effects, keys, fx, clips and accents in place."""
    kinds = _sfx_kinds(spec)
    auto = "auto" in kinds
    if not spec.emphasis and not kinds:
        return spec
    words = []
    if any(e.word is not None or e.match is not None for e in spec.emphasis or ()):
        words = load_words(spec, spec_dir, errors)
        if not words:
            return spec
    before = len(errors)
    events, em_markers = _entries(spec, words, errors, spec_dir)
    if len(errors) > before or spec_dir is None:
        return spec
    gain = (spec.audio.sfx_gain if spec.audio and spec.audio.sfx_gain else DEFAULT_SFX_GAIN)
    audio = spec.audio.model_copy(update={"sfx": None, "sfx_gain": None}) if spec.audio else None
    base = spec.model_copy(update={"emphasis": None, "audio": audio, "markers": {**spec.markers, **em_markers}})
    dry: list[str] = []
    parsed, _, markers, bpm, beat0, _ = validate_spec(base, dry, spec_dir)
    n_shots = len(spec.shots)
    if dry or f"shots[{n_shots - 1}].end" not in parsed:
        return spec.model_copy(update={"emphasis": None, "audio": audio}) if not dry else spec  # validate reports
    fps = spec.video.fps
    n_frames = parsed[f"shots[{n_shots - 1}].end"]

    shots = list(spec.shots)
    fx = list(spec.fx)
    clips: list[AudioClip] = []
    overrides: list[CaptionOverride] = []
    zooms: dict[int, list[tuple[int, float]]] = {}
    peaks: list[float] = []  # film seconds of each sound this adds; `text` / `cuts` hits skip them
    entries = spec.emphasis or []
    for ev in events:
        e = entries[ev.entry]
        path = f"emphasis[{ev.entry}]"
        if ev.word is None:
            points = list(_every_exprs(ev, n_frames, fps, bpm, beat0, errors))
        else:
            f = markers.get(ev.expr)
            points = [(ev.expr, f)] if f is not None else []
        if ev.word is not None and not points:
            errors.append(_line(path, f"word {ev.word} did not resolve", "check audio.markers"))
        for expr, f in points:
            if not 0 <= f < n_frames:
                errors.append(_line(path, f"{ev.expr!r} lands at frame {round(f)}, outside the film", "pick a word inside it"))
                continue
            si = _shot_at(parsed, n_shots, f)
            start = parsed.get(f"shots[{si}].start", 0) if si is not None else 0
            if e.shake is not None and si is not None:
                if shots[si].blank:
                    errors.append(_line(f"{path}.shake", f"shot {si} is blank", "shake needs a picture"))
                else:
                    fxe = Effect(type="shake", amount=float(e.shake), at=expr, dur=SHAKE_DUR, seed=ev.seed)
                    shots[si] = shots[si].model_copy(update={"effects": [*(shots[si].effects or []), fxe]})
            if e.zoom is not None and si is not None:
                sh = shots[si]
                if sh.clip is None or sh.camera is not None:
                    why = "has camera keys" if sh.camera is not None else "is not a clip shot"
                    errors.append(_line(f"{path}.zoom", f"shot {si} {why}", "zoom needs a clip shot without camera"))
                elif not math.isfinite(e.zoom) or e.zoom < 1:
                    errors.append(_line(f"{path}.zoom", f"zoom is a multiple of cover, got {e.zoom}", "use >= 1"))
                else:
                    zooms.setdefault(si, []).append((round(f) - start, float(e.zoom)))
            if e.flash is not None:
                fx.append(Fx(flash=float(e.flash), at=expr))
            if e.sfx is not None:
                clip = _sfx_clip(spec, spec_dir, e.sfx, expr, f / fps, gain, f"{path}.sfx", errors)
                if clip is not None:
                    clips.append(clip)
                    peaks.append(f / fps)
            if e.caption is not None and ev.word is not None:
                overrides.append(CaptionOverride(word=ev.word, emphasis=True))
    for si, peaks in zooms.items():
        sh = shots[si]
        end_local = parsed[f"shots[{si}].end"] - parsed[f"shots[{si}].start"]
        shots[si] = sh.model_copy(update={"camera": _zoom_keys(sh, peaks, end_local), "zoom": None, "focus": None})
    if overrides:
        if spec.captions is None:
            errors.append(_line("emphasis", "caption: accent needs a captions block", "add captions: {source: ...}"))
        else:
            covered = {o.word for o in spec.captions.overrides if o.emphasis is not None}
            add = [o for o in overrides if o.word not in covered]
            caps = spec.captions.model_copy(update={"overrides": [*spec.captions.overrides, *add]})
            spec = spec.model_copy(update={"captions": caps})
    if auto:
        for name, f in _auto_events(spec, parsed, markers, bpm, beat0):
            clip = _sfx_clip(spec, spec_dir, name, f"{f / fps:.4f}s", f / fps, gain, "audio.sfx", errors)
            if clip is not None:
                clips.append(clip)
                peaks.append(f / fps)
    hits = _hit_events(spec, parsed, markers, bpm, beat0, kinds)
    if hits:
        # Never double a sound already on that frame: the author's clips, emphasis sfx, `auto`, earlier hits.
        peaks += _placed_peaks(base, spec_dir, markers, bpm, beat0)
        for name, f in sorted(hits, key=lambda h: h[1]):
            t = f / fps
            if any(abs(t - p) < CLOSE_S for p in peaks):
                continue
            clip = _sfx_clip(spec, spec_dir, name, f"{t:.4f}s", t, gain, "audio.sfx", errors)
            if clip is not None:
                clips.append(clip)
                peaks.append(t)
    if clips:
        audio = (audio or Audio()).model_copy(update={"clips": [*(audio.clips if audio else ()), *clips]})
    return spec.model_copy(update={"emphasis": None, "shots": shots, "fx": fx, "audio": audio,
                                   "markers": {**spec.markers, **em_markers}})


def sfx_peaks(plan) -> list[tuple[float, str]]:
    """`(film seconds of the loudest moment, name)` for each emphasis / auto sound in a `MixPlan`."""
    from montaj.audio.mix import peak_lead

    out = []
    for c in plan.clips:
        if c.id == SFX_ID:
            name = c.src[6:] if c.src.startswith("synth:") else Path(c.src).stem
            out.append((c.at_s + peak_lead(c.src, c.stream) - c.in_s, name))
    return sorted(out)


def sfx_lines(plan) -> list[str]:
    """`INFO sfx=whoosh*3 pop*2` and a `WARN` for each pair of peaks within 80 ms."""
    peaks = sfx_peaks(plan)
    if not peaks:
        return []
    counts: dict[str, int] = {}
    for _, name in peaks:
        counts[name] = counts.get(name, 0) + 1
    lines = ["INFO sfx=" + " ".join(f"{k}x{v}" for k, v in sorted(counts.items()))]
    for (a, na), (b, nb) in zip(peaks, peaks[1:]):
        if b - a < CLOSE_S:
            lines.append(f"WARN sfx {na}@{a:.2f}s and {nb}@{b:.2f}s peak {round((b - a) * 1000)}ms apart (< 80ms)")
    return lines
