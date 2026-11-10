"""Pacing and retention numbers for `montaj check`.

Everything comes from the resolved `Timeline` (shots, transitions, text/sub windows, beat grid,
audio clips) plus the optional `audio.markers` word list. Two numbers read the rendered file:
`first_motion` (low-res frames, first 5 s) and `silence` (ffmpeg silencedetect, audio only).
`metrics` is the whole truth (also `check --json`); `lines` turns it into INFO lines and, when
`check.pacing` is set and missed, WARN lines. An unset target never warns.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from montaj.audio.mix import shift_words
from montaj.encode import FFMPEG, probe, run
from montaj.spec import _dur_frames, _parse_beat0
from montaj.timeline import Timeline

Span = tuple[float, float]

MOTION_WINDOW_S = 5.0
MOTION_DIFF = 1.0  # mean abs difference (of 255) from frame 0 that counts as the picture moving
MOTION_SIZE = 64
SILENCE_MIN_S = 0.5
SILENCE_DB = -50
SPEECH_JOIN_S = 0.5  # words closer than this are one stretch of speech
OFF_LIST_MAX = 5
SILENCE_START = re.compile(r"silence_start:\s*(?P<s>-?[\d.]+)")
SILENCE_END = re.compile(r"silence_end:\s*(?P<e>[\d.]+)")


def _merge(spans: Sequence[Span], join: float = 0.0) -> list[Span]:
    out: list[Span] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + join:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _minus(spans: Sequence[Span], cover: Sequence[Span]) -> list[Span]:
    """Parts of `spans` not inside `cover` (both merged, sorted)."""
    out: list[Span] = []
    for a, b in spans:
        cur = a
        for c, d in cover:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
        if cur < b:
            out.append((cur, b))
    return out


def _total(spans: Sequence[Span]) -> float:
    return sum(b - a for a, b in spans)


def _beat0_frames(tl: Timeline) -> float:
    return _parse_beat0(tl.spec.video.beat0 or "0s", tl.fps, []) or 0.0


def _words(tl: Timeline, root: Path) -> list[Span]:
    """Word spans in film seconds from `audio.markers`, after `audio.edit`. Empty without speech data."""
    audio = tl.spec.audio
    if audio is None or not audio.markers:
        return []
    try:
        data = json.loads((root / audio.markers).read_text())
    except (OSError, ValueError):
        return []
    raw = data.get("words") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    starts: dict[int, float] = {}
    dur: dict[int, float] = {}
    for w in raw:
        try:
            i, s, e = int(w["i"]), float(w["s"]), float(w["e"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(s) and math.isfinite(e) and e > s:
            starts[i], dur[i] = s, e - s
    starts = shift_words(tl.spec, starts, tl.fps, tl.spec.video.bpm, _beat0_frames(tl))
    return [(s, s + dur[i]) for i, s in starts.items()]


def _text_spans(tl: Timeline) -> list[Span]:
    """When on-screen text (resolved text lines, subs) is visible, in seconds. Captions land here too."""
    out: list[Span] = []
    for block in tl.texts:
        shown = [ln.at for ln in block.lines if ln.text.strip()]
        if shown:
            out.append((min(shown) / tl.fps, block.to / tl.fps))
    out += [(s.from_ / tl.fps, s.to / tl.fps) for s in tl.subs if s.text.strip()]
    return [(a, b) for a, b in out if b > a]


def first_motion(video: Path | str, fps: float) -> float | None:
    """Seconds to the first frame that differs from frame 0 (low-res, first 5 s); None if it never moves.

    Read from the render, not the timeline: a still's drift, a camera key and a transition all show
    up the same way here, which a per-feature timeline reading would have to enumerate.
    """
    cmd = [FFMPEG, "-v", "error", "-i", str(video), "-t", str(MOTION_WINDOW_S), "-an",
           "-vf", f"scale={MOTION_SIZE}:{MOTION_SIZE},format=gray", "-f", "rawvideo", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return None
    size = MOTION_SIZE * MOTION_SIZE
    n = len(proc.stdout) // size
    if n < 2:
        return None
    frames = np.frombuffer(proc.stdout[: n * size], dtype=np.uint8).reshape(n, size).astype(np.int16)
    hit = np.nonzero(np.abs(frames[1:] - frames[0]).mean(axis=1) >= MOTION_DIFF)[0]
    return float((hit[0] + 1) / fps) if hit.size else None


def silences(video: Path | str, *, log: Path | str | None = None) -> list[Span]:
    _, err = run([FFMPEG, "-hide_banner", "-nostats", "-i", str(video), "-vn",
                  "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_S}", "-f", "null", "-"],
                 log=log, desc="qa")
    spans: list[Span] = []
    start: float | None = None
    for line in err.splitlines():
        if (m := SILENCE_START.search(line)) is not None:
            start = max(0.0, float(m["s"]))
        elif (m := SILENCE_END.search(line)) is not None and start is not None:
            spans.append((start, float(m["e"])))
            start = None
    return spans


def _kind(tl: Timeline, i: int) -> str:
    tr = tl.shots[i].spec.in_
    return "cut" if tr is None else tr.type


def _longest_run(kinds: list[str], first: int) -> tuple[int, int]:
    """(length, first shot index) of the longest run of equal kinds; `kinds[0]` belongs to shot `first`."""
    best, best_at, run_len, at = 0, first, 0, first
    for j, k in enumerate(kinds):
        if j and k == kinds[j - 1]:
            run_len += 1
        else:
            run_len, at = 1, first + j
        if run_len > best:
            best, best_at = run_len, at
    return best, best_at


def metrics(tl: Timeline, root: Path, video: Path | str | None = None, *,
            log: Path | str | None = None) -> dict:
    """Pacing numbers. `video` adds `first_motion` and audio silence; without it those are omitted."""
    fps = float(tl.fps)
    total = tl.n_frames / fps
    shots = tl.shots
    lens = [(s.end - s.start) / fps for s in shots]
    cuts = max(0, len(shots) - 1)
    longest = max(range(len(lens)), key=lens.__getitem__) if lens else 0
    out: dict = {"duration": round(total, 3)}
    out["pacing"] = {
        "cuts": cuts,
        "per10s": round(cuts * 10.0 / total, 2) if total > 0 else 0.0,
        "median_shot": round(statistics.median(lens), 3) if lens else 0.0,
        "longest": round(lens[longest], 3) if lens else 0.0,
        "longest_shot": longest,
    }
    texts = _text_spans(tl)
    hook: dict = {"first_text": round(min(a for a, _ in texts), 3) if texts else None,
                  "first_cut": round(shots[1].start / fps, 3) if len(shots) > 1 else None}
    if video is not None:
        fm = first_motion(video, fps)
        hook["first_motion"] = None if fm is None else round(fm, 3)
    out["hook"] = hook

    bpm = tl.spec.video.bpm
    if bpm and bpm > 0 and cuts:
        period = 60.0 / bpm
        beat0 = _beat0_frames(tl) / fps
        off: list[dict] = []
        for s in shots[1:]:
            t = s.start / fps
            delta = t - (beat0 + round((t - beat0) / period) * period)
            if abs(delta) > 0.5 / fps + 1e-9:
                off.append({"shot": s.index, "ms": round(delta * 1000)})
        out["beats"] = {"cuts_on_grid": cuts - len(off), "cuts": cuts, "off": off}

    words = _words(tl, root)
    if words:
        speech = _merge(words, SPEECH_JOIN_S)
        bare = _minus(speech, _merge(texts))
        out["captions"] = {"coverage": round(1.0 - _total(bare) / _total(speech), 3),
                           "max_gap": round(max((b - a for a, b in bare), default=0.0), 3)}

    audio = tl.spec.audio
    if audio is not None:
        under = set(audio.duck.under) if audio.duck else set()
        sfx = sum(1 if isinstance(c.at, str) else len(c.at) for c in audio.clips if c.id not in under)
        block: dict = {"sfx": sfx}
        if audio.duck is not None:
            block["music_under_voice_db"] = float(audio.duck.depth.strip().removesuffix("dB"))
        if video is not None and probe(video, log=log).has_audio:
            block["silence_spans"] = len(silences(video, log=log))
        out["audio"] = block

    kinds = [_kind(tl, i) for i in range(1, len(shots))]
    if kinds:
        n, at = _longest_run(kinds, 1)
        styles = {ln.style for blk in tl.texts for ln in blk.lines}
        out["variety"] = {"transition_kinds": len(set(kinds)), "text_styles": len(styles),
                          "repeated_in_a_row": n, "repeated_from": at}
    return out


def _s(v: float) -> str:
    return f"{v:.1f}s"


def lines(m: dict, target=None, *, fps: int = 30, bpm: float | None = None) -> list[str]:
    """INFO lines for `m`, then WARN lines for each missed `check.pacing` target."""
    out: list[str] = []
    p = m["pacing"]
    out.append(f"INFO pacing cuts={p['cuts']} per10s={p['per10s']:.1f} median_shot={_s(p['median_shot'])} "
               f"longest={_s(p['longest'])}@shot{p['longest_shot']}")
    h = m["hook"]

    def t(key: str) -> str:
        return "none" if h.get(key) is None else _s(h[key])

    out.append(f"INFO hook first_text={t('first_text')} first_cut={t('first_cut')}"
               + (f" first_motion={t('first_motion')}" if "first_motion" in h else ""))
    if "beats" in m:
        b = m["beats"]
        off = ", ".join(f"shot {o['shot']} {o['ms']:+d}ms" for o in b["off"][:OFF_LIST_MAX])
        if len(b["off"]) > OFF_LIST_MAX:
            off += ", ..."
        out.append(f"INFO beats cuts_on_grid={b['cuts_on_grid']}/{b['cuts']} off=[{off}]")
    if "captions" in m:
        c = m["captions"]
        out.append(f"INFO captions coverage={c['coverage'] * 100:.0f}% of speech  max_gap={_s(c['max_gap'])}")
    if "audio" in m:
        a = m["audio"]
        parts = []
        if "silence_spans" in a:
            parts.append(f"silence>{SILENCE_MIN_S:g}s={a['silence_spans']} spans")
        parts.append(f"sfx={a['sfx']}")
        if "music_under_voice_db" in a:
            parts.append(f"music_under_voice={a['music_under_voice_db']:g}dB")
        out.append("INFO audio " + "  ".join(parts))
    if "variety" in m:
        v = m["variety"]
        last = v["repeated_from"] + v["repeated_in_a_row"] - 1
        where = f"@shots {v['repeated_from']}-{last}" if v["repeated_in_a_row"] > 1 else ""
        out.append(f"INFO variety transitions={v['transition_kinds']} kinds  text_styles={v['text_styles']}  "
                   f"repeated_in_a_row={v['repeated_in_a_row']}{where}")
    if target is None:
        return out
    if target.max_shot is not None:
        limit = _dur_frames(target.max_shot, fps, bpm, "check.pacing.max_shot", [])
        if limit is not None and p["longest"] > limit / fps + 1e-9:
            out.append(f"WARN pacing longest={_s(p['longest'])}@shot{p['longest_shot']} "
                       f"(max_shot {target.max_shot})")
    if target.min_per10s is not None and p["per10s"] < target.min_per10s:
        out.append(f"WARN pacing per10s={p['per10s']:.1f} (min_per10s {target.min_per10s:g})")
    if target.hook_text_by is not None:
        limit = _dur_frames(target.hook_text_by, fps, bpm, "check.pacing.hook_text_by", [])
        first = h.get("first_text")
        if limit is not None and (first is None or first > limit / fps + 1e-9):
            seen = "no text" if first is None else f"first_text={_s(first)}"
            out.append(f"WARN hook {seen} (hook_text_by {target.hook_text_by})")
    return out


__all__ = ["first_motion", "lines", "metrics", "silences"]
