"""QA on a rendered file: duration, size, fps, black/frozen spans, loudness, A/V offset.

`check` returns short `OK|WARN|INFO ...` lines for the CLI to print; the ffmpeg stderr it
parses goes to `build/qa.log`. Spans the caller marks as intended are `INFO … ignored as
intended`, not defects. An `Expect` that matches a detected span adds its `why`; one that
matches nothing is `INFO expect … not found`.

`clip_overruns` is the spec-side half of `montaj check`: a clip shot whose read window
runs past the source file. The CLI appends those lines to `check`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple, Sequence

from montaj.encode import FFMPEG, FFPROBE, MediaInfo, VideoSettings, capture, probe, run
from montaj.project import resolve_clip
from montaj.spec import _dur_frames
from montaj.timeline import Timeline, clip_read_frames, transition_overlap

Span = tuple[float, float]


class Expect(NamedTuple):
    """A designed span in seconds. `kind` is `black` or `frozen`."""

    kind: str
    span: Span
    why: str | None = None
BLACK_MIN_S = 0.10  # shorter than this is a fade, not a defect
BLACK_PIX_TH = 0.05  # blackdetect pix_th: true black and the tail of a fade count; a dark ground (e.g. #0b1020, luma ~0.06) does not
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


def _span_label(span: Span) -> str:
    """`0–2s`, `1.5–2.25s`. Whole seconds drop the fraction."""

    def one(t: float) -> str:
        rounded = round(t)
        if abs(t - rounded) < 5e-3:
            return str(int(rounded))
        return f"{t:.2f}".rstrip("0").rstrip(".")

    return f"{one(span[0])}\u2013{one(span[1])}s"


def _matches(span: Span, detected: Sequence[Span]) -> bool:
    """True when `span` overlaps a detected span, with `SPAN_EPS_S` of slack for detector rounding."""
    lo, hi = span
    for start, end in detected:
        if max(lo, start) < min(hi, end) + SPAN_EPS_S:
            return True
    return False


def _note_expects(lines: list[str], kind: str, detected: Sequence[Span], expect: Sequence[Expect]) -> None:
    for item in expect:
        if item.kind != kind:
            continue
        label = _span_label(item.span)
        if _matches(item.span, detected):
            if item.why:
                lines.append(f"INFO {kind} {label} ignored as intended: {item.why}")
        else:
            lines.append(f"INFO expect {kind} {label} not found")


def video_line(info: MediaInfo, want: VideoSettings | None = None) -> str:
    """Codec, bit depth and colour tags as the file states them. bt709 primaries, transfer and
    matrix plus limited (tv) range on every target; `want` (an export target's settings) also
    pins the codec and depth, so a youtube file that came out 8-bit is a WARN, not a silent pass."""
    codec = info.vcodec or "unknown"
    depth = info.bit_depth
    tags = (info.color_space, info.color_primaries, info.color_transfer)
    shown = f"{codec} {depth or '?'}-bit {info.pix_fmt or '?'}"
    if any(t != "bt709" for t in tags) or info.color_range != "tv":
        got = "/".join(t or "unset" for t in tags) + f" range={info.color_range or 'unset'}"
        return f"WARN video {shown} colour tags {got} (want bt709/bt709/bt709 range=tv)"
    if want is not None:
        want_codec = "hevc" if want.codec == "hevc10" else want.codec
        if codec != want_codec or depth != want.depth:
            return f"WARN video {shown} bt709 (want {want_codec} {want.depth}-bit)"
    return f"OK video {shown} bt709"


def _loudness_want(target: float) -> str:
    if abs(target - round(target)) < 1e-6:
        return f"{target:.0f}"
    return f"{target:.1f}"


def check(video: Path | str, intended_freeze: Sequence[Span] = (), *, log: Path | str | None = None,
          av_tol: float | None = None, intended_black: Sequence[Span] = (),
          expect: Sequence[Expect] = (), loudness_target: float | None = None,
          loudness_tol: float | None = None, want: VideoSettings | None = None) -> list[str]:
    """One short line per check; the only stdout surface of the QA stage.

    `av_tol` overrides the 0.5 s default. The CLI passes one frame when the spec has audio.
    `intended_black` drops black spans that sit inside a designed dark window (opening fade
    from a dark ground, a dark blank and its transitions, the outro fade, `check.expect`).
    `expect` is that same declaration plus its `why`: a match names the reason, a miss is
    `INFO expect … not found`. `loudness_target` / `loudness_tol` override -14 LUFS ± 1.
    `want` is the codec/depth the file must have (see `video_line`).
    """
    info = probe(video, log=log)
    tol = AV_TOL_S if av_tol is None else av_tol
    target = LOUDNESS_TARGET if loudness_target is None else loudness_target
    ltol = LOUDNESS_TOL if loudness_tol is None else loudness_tol
    lines = [
        f"OK duration {info.duration:.2f}s",
        f"OK size {info.width}x{info.height}",
        f"OK fps {info.fps:.1f}",
        video_line(info, want),
    ]
    ignored_black = 0
    detected_black = black_spans(video, log=log)
    for span in detected_black:
        left = subtract(span, intended_black)
        ignored_black += 1 if not left else 0
        lines += [f"WARN black {start:.2f}-{end:.2f}s" for start, end in left]
    if ignored_black:
        lines.append(f"INFO black {ignored_black} span(s) ignored as intended")
    _note_expects(lines, "black", detected_black, expect)
    ignored = 0
    detected_frozen = frozen_spans(video, log=log)
    for span in detected_frozen:
        # A leftover shorter than the detector's own minimum is the eased tail of a transition
        # running into an intended hold, not a freeze in its own right.
        left = [p for p in subtract(span, intended_freeze) if p[1] - p[0] >= FREEZE_MIN_S]
        ignored += 1 if not left else 0
        lines += [f"WARN frozen {start:.1f}-{end:.1f}s" for start, end in left]
    if ignored:
        lines.append(f"INFO frozen {ignored} span(s) ignored as intended")
    _note_expects(lines, "frozen", detected_frozen, expect)
    if not info.has_audio:
        lines.append("INFO no audio")
        return lines
    i = loudness(video, log=log)
    if i is None:
        lines.append("WARN loudness unreadable")
    elif abs(i - target) <= ltol:
        lines.append(f"OK loudness {i:.1f} LUFS")
    else:
        lines.append(f"WARN loudness {i:.1f} LUFS (want {_loudness_want(target)})")
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


def video_end(path: Path, *, log: Path | str | None = None) -> float:
    """End of the video stream alone: the last packet's pts plus its duration.

    `probe().duration` falls back to the container duration (MKV has no stream
    duration), and a longer audio track would hide a video overrun.
    """
    out = capture(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0", str(path)],
        log=log,
        desc="ffprobe",
    )
    end = 0.0
    for line in out.splitlines():
        parts = line.strip().split(",")
        try:
            pts = float(parts[0])
            dur = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
        except ValueError:
            continue
        end = max(end, pts + dur)
    return end or probe(path, log=log).duration


def clip_overruns(
    tl: Timeline, assets: Path, *, spec_dir: Path | None = None, log: Path | str | None = None
) -> list[str]:
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
        # A range never reads past its end, so only the hold counts.
        if nxt < len(tl.shots) and sh.range is None:
            tr = tl.shots[nxt].spec.in_
            if tr is not None and tr.type != "cut" and tr.dur is not None:
                overlap = transition_overlap(tr.type, _frames(tr.dur, fps, bpm))
        need = clip_read_frames(shot.end - shot.start, overlap) / float(fps)
        clip_in = _frames(sh.range[0] if sh.range is not None else sh.clip_in, fps, bpm) / float(fps)
        path = resolve_clip(spec_dir if spec_dir is not None else assets.parent, assets, sh.clip)
        if path not in durations:
            durations[path] = video_end(path, log=log)
        overrun = (clip_in + need) - durations[path]
        if overrun <= (1.0 / fps) + 1e-4:
            continue
        lines.append(f"WARN shots[{shot.index}] {sh.clip} runs {overrun:.2f}s past its clip")
    return lines


__all__ = [
    "Expect", "Span", "black_spans", "check", "clip_overruns", "frozen_spans", "loudness", "subtract", "union",
]


CLICK_RATIO = 6.0  # a join's biggest sample-to-sample step vs the 99th percentile of its surroundings
CLICK_FLOOR = 0.02  # ... and at least this much of full scale, so a quiet join is never a click


def join_clicks(audio, joins: Sequence[float], sr: int = 48000) -> list[tuple[float, float, float]]:
    """`(second, step, around)` for each join whose waveform jumps: a click.

    `audio` is mono samples. The step is the largest sample-to-sample change within 3 ms of the
    join; `around` is the 99th percentile of that change over the 10-60 ms either side.
    """
    import numpy as np

    d = np.abs(np.diff(np.asarray(audio, dtype=np.float64)))
    near, lo, hi = round(0.003 * sr), round(0.010 * sr), round(0.060 * sr)
    bad = []
    for t in joins:
        c = round(t * sr)
        if c - hi < 0 or c + hi >= len(d):
            continue
        ref = np.concatenate([d[c - hi : c - lo], d[c + lo : c + hi]])
        around = float(np.percentile(ref, 99))
        step = float(d[c - near : c + near].max())
        if step > max(CLICK_RATIO * around, CLICK_FLOOR):
            bad.append((t, step, around))
    return bad


def join_lines(video: Path | str, joins: Sequence[float], *, log: Path | str | None = None) -> list[str]:
    """`OK joins N clean` or one `WARN click` per join, read from the rendered file's own audio."""
    if not joins:
        return []
    from montaj.audio.mix import decode
    from montaj.encode import EncodeError

    try:
        audio = decode(str(video), 0, log=Path(log) if log else None)
    except EncodeError:
        return ["INFO joins unchecked: no audio"]
    bad = join_clicks(audio.mean(axis=1), joins)
    lines = [f"WARN click {t:.2f}s step {step:.3f} vs {around:.3f} around" for t, step, around in bad]
    clean = len(joins) - len(bad)
    return [f"OK joins {clean}/{len(joins)} clean", *lines] if bad else [f"OK joins {len(joins)} clean"]
