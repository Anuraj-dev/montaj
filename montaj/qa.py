"""QA on a rendered file: duration, size, fps, black/frozen spans, loudness, A/V offset.

`check` returns short `OK|WARN|INFO ...` lines for the CLI to print; the ffmpeg stderr it
parses goes to `build/qa.log`. Spans the caller marks as intended (`intended_freeze`) are
reported as `OK frozen` so a deliberate still hold is not a defect.

`clip_overruns` is the spec-side half of `montaj check`: a clip shot whose read window
runs past the source file. The CLI appends those lines to `check`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from montaj.encode import FFMPEG, probe, run
from montaj.project import find_clip
from montaj.spec import _dur_frames
from montaj.timeline import Timeline, clip_read_frames, transition_overlap

Span = tuple[float, float]
BLACK_MIN_S = 0.10  # shorter than this is a fade, not a defect
BLACK_PIX_TH = 0.10
FREEZE_NOISE_DB = -60.0
FREEZE_MIN_S = 0.5
LOUDNESS_TARGET = -14.0
LOUDNESS_TOL = 1.0
AV_TOL_S = 0.5
SPAN_EPS_S = 0.1  # slivers of a second are rounding, not a defect
BLACK_RE = re.compile(r"black_start:(?P<start>[\d.]+)\s+black_end:(?P<end>[\d.]+)")
FREEZE_START_RE = re.compile(r"freeze_start:\s*(?P<start>[\d.]+)")
FREEZE_END_RE = re.compile(r"freeze_end:\s*(?P<end>[\d.]+)")
EBU_I_RE = re.compile(r"^\s*I:\s*(?P<i>-?\d+(?:\.\d+)?)\s*LUFS", re.MULTILINE)
EBU_SUMMARY_RE = re.compile(r"Integrated loudness:\s*\n\s*I:\s*(?P<i>-?\d+(?:\.\d+)?)\s*LUFS")


def _detect(video: Path | str, filt: str, log: Path | str | None) -> str:
    """Run one analysis filter over the whole file; return its stderr.

    A non-zero exit raises: a missing filter or an unreadable file must not read as "clean".
    """
    _, stderr = run(
        [FFMPEG, "-hide_banner", "-nostats", "-i", str(video), "-an", "-vf", filt, "-f", "null", "-"],
        log=log,
        desc="qa",
    )
    return stderr


def black_spans(video: Path | str, *, log: Path | str | None = None,
                min_s: float = BLACK_MIN_S) -> list[Span]:
    """`(start, end)` seconds of near-black frames."""
    stderr = _detect(video, f"blackdetect=d={min_s}:pix_th={BLACK_PIX_TH}", log)
    return [(float(m["start"]), float(m["end"])) for m in BLACK_RE.finditer(stderr)]


def frozen_spans(video: Path | str, *, log: Path | str | None = None,
                 min_s: float = FREEZE_MIN_S) -> list[Span]:
    """`(start, end)` seconds where the picture does not change."""
    stderr = _detect(video, f"freezedetect=n={FREEZE_NOISE_DB}dB:d={min_s}", log)
    spans: list[Span] = []
    start: float | None = None
    for line in stderr.splitlines():
        if (hit := FREEZE_START_RE.search(line)) is not None:
            start = float(hit["start"])
        elif (hit := FREEZE_END_RE.search(line)) is not None and start is not None:
            spans.append((start, float(hit["end"])))
            start = None
    if start is not None:
        info = probe(video, log=log)
        spans.append((start, info.duration))
    return spans


def loudness(video: Path | str, *, log: Path | str | None = None) -> float | None:
    """Integrated loudness in LUFS (None when the file has no audio)."""
    info = probe(video, log=log)
    if not info.has_audio:
        return None
    _, stderr = run(
        [FFMPEG, "-hide_banner", "-nostats", "-i", str(video), "-vn",
         "-af", "ebur128=framelog=verbose", "-f", "null", "-"],
        log=log,
        desc="qa",
    )
    summary = EBU_SUMMARY_RE.search(stderr)
    if summary:
        return float(summary["i"])
    hits = EBU_I_RE.findall(stderr)
    return float(hits[-1]) if hits else None


def union(spans: Sequence[Span]) -> list[Span]:
    """Merge overlapping/touching ranges so coverage is never counted twice."""
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract(span: Span, ranges: Sequence[Span]) -> list[Span]:
    """`span` minus the union of `ranges`: the parts of a freeze nobody claimed as intended.

    [1,10] with only [1,6] intended still leaves [6,10] to report; a fully intended span yields [].
    """
    pieces = [span]
    for lo, hi in union(ranges):
        remaining: list[Span] = []
        for start, end in pieces:
            if hi <= start or lo >= end:
                remaining.append((start, end))
                continue
            if start < lo:
                remaining.append((start, lo))
            if hi < end:
                remaining.append((hi, end))
        pieces = remaining
    return [piece for piece in pieces if piece[1] - piece[0] > SPAN_EPS_S]


def check(video: Path | str, intended_freeze: Sequence[Span] = (), *, log: Path | str | None = None,
          av_tol: float | None = None, intended_black: Sequence[Span] = ()) -> list[str]:
    """One short line per check; the only stdout surface of the QA stage.

    `av_tol` overrides the 0.5 s default. The CLI passes one frame when the spec has audio.
    `intended_black` drops black spans that sit inside a designed dark window (opening fade
    from a dark ground, a dark blank and its transitions, the outro fade).
    """
    info = probe(video, log=log)
    tol = AV_TOL_S if av_tol is None else av_tol
    lines = [
        f"OK duration {info.duration:.2f}s",
        f"OK size {info.width}x{info.height}",
        f"OK fps {info.fps:.1f}",
    ]
    ignored_black = 0
    for span in black_spans(video, log=log):
        left = subtract(span, intended_black)
        ignored_black += 1 if not left else 0
        lines += [f"WARN black {start:.2f}-{end:.2f}s" for start, end in left]
    if ignored_black:
        lines.append(f"INFO black {ignored_black} span(s) ignored as intended")
    ignored = 0
    for span in frozen_spans(video, log=log):
        # A leftover shorter than the detector's own minimum is the eased tail of a transition
        # running into an intended hold, not a freeze in its own right.
        left = [p for p in subtract(span, intended_freeze) if p[1] - p[0] >= FREEZE_MIN_S]
        ignored += 1 if not left else 0
        lines += [f"WARN frozen {start:.1f}-{end:.1f}s" for start, end in left]
    if ignored:
        lines.append(f"INFO frozen {ignored} span(s) ignored as intended")
    if not info.has_audio:
        lines.append("INFO no audio")
        return lines
    i = loudness(video, log=log)
    if i is None:
        lines.append("WARN loudness unreadable")
    elif abs(i - LOUDNESS_TARGET) <= LOUDNESS_TOL:
        lines.append(f"OK loudness {i:.1f} LUFS")
    else:
        lines.append(f"WARN loudness {i:.1f} LUFS (want {LOUDNESS_TARGET:.0f})")
    if info.av_offset is None:
        lines.append("INFO av-offset unknown")
    elif abs(info.av_offset) > tol:
        lines.append(f"WARN av-offset {info.av_offset:+.2f}s")
    else:
        lines.append(f"OK av-offset {info.av_offset:+.2f}s")
    return lines


def _frames(raw: str | None, fps: int, bpm: float | None) -> float:
    if not raw:
        return 0.0
    frames = _dur_frames(raw, fps, bpm, "duration", [])
    return 0.0 if frames is None else float(frames)


def clip_overruns(tl: Timeline, assets: Path, *, log: Path | str | None = None) -> list[str]:
    """WARN when a clip shot's read window goes past the end of its source.

    The window is the hold plus the outgoing fade or swirl, or half a whip: the
    same frames the renderer asks ffmpeg for. `clip_in` moves the start. One
    frame of slack, so a container duration a frame under the hold stays quiet.
    Each line names the shot (`shots[i]` and the clip stem) and the overrun.
    """
    fps = int(tl.fps)
    if fps <= 0:
        return []
    bpm = tl.spec.video.bpm
    durations: dict[Path, float] = {}
    lines: list[str] = []
    for shot in tl.shots:
        sh = shot.spec
        if not sh.clip:
            continue
        nxt = shot.index + 1
        overlap = 0.0
        if nxt < len(tl.shots):
            tr = tl.shots[nxt].spec.in_
            if tr is not None and tr.type != "cut" and tr.dur is not None:
                overlap = transition_overlap(tr.type, _frames(tr.dur, fps, bpm))
        need = clip_read_frames(shot.end - shot.start, overlap) / float(fps)
        clip_in = _frames(sh.clip_in, fps, bpm) / float(fps)
        path = find_clip(assets, sh.clip)
        if path not in durations:
            durations[path] = probe(path, log=log).duration
        overrun = (clip_in + need) - durations[path]
        if overrun <= (1.0 / fps) + 1e-4:
            continue
        lines.append(f"WARN shots[{shot.index}] {sh.clip} runs {overrun:.2f}s past its clip")
    return lines


__all__ = [
    "Span", "black_spans", "check", "clip_overruns", "frozen_spans", "loudness", "subtract", "union",
]
