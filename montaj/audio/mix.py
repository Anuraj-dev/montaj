"""The audio mixer: beat-grid music edits, placed voice/SFX clips, ducking. Pure numpy, CPU only.

`audio.edit`, `audio.clips` and `audio.duck` resolve to a `MixPlan` (seconds, absolute paths).
`render_mix` turns the plan into one stereo 48 kHz float wav in `build/audio/`, keyed by a hash of
the plan and the input bytes; `encode.mux` then treats that wav exactly as it treats `audio.track`.
A spec without any of the three never reaches this module at render time (`MixPlan.active`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from montaj.encode import FFMPEG, FFPROBE, EncodeError
from montaj.spec import _DUR, _line, _parse_time

SR = 48000
MIX_REV = "1"  # bump when the mixed samples change for the same plan
CLIP_FADE_S = 0.015  # fade in/out on every placed clip
XFADE_S = 0.012  # equal-power crossfade centred on an edit join
DUCK_THRESHOLD_DB = -45.0  # a duck source counts as "speaking" above this RMS
DUCK_WINDOW_S = 0.120
PEAK_CEILING = 0.98  # a hotter mix is scaled down (and reported) before it is written
SYNTH = ("whoosh", "riser", "boom", "tick", "glitch")
SRC_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm")
_INF = 1e12

_DB = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*dB$")
_SPAN = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(ms|s|f|b)$")
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
class DuckPlan:
    under: tuple[str, ...]
    depth_db: float
    attack_s: float
    release_s: float


@dataclass(frozen=True)
class MixPlan:
    track: str
    edits: tuple[EditOp, ...] = ()
    clips: tuple[Placed, ...] = ()
    duck: DuckPlan | None = None

    @property
    def active(self) -> bool:
        return bool(self.edits or self.clips or self.duck)


@dataclass(frozen=True)
class MixStats:
    duration: float
    peak_dbfs: float  # of the mix before any ceiling scaling
    clipped: bool  # the mix was scaled down to stay under the ceiling
    clips: int  # placements mixed in
    duck_db: float  # deepest reduction applied to the music (<= 0)
    hash: str = ""


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
    return float(m.group(1))


def _edit_time(raw: object, fps: int, bpm: float | None, beat0: float, path: str,
               errors: list[str]) -> float | None:
    """Edit points are times in the music itself, so a marker (a film time) makes no sense."""
    if not isinstance(raw, str) or not _DUR.match(raw.strip()):
        errors.append(_line(path, f"invalid music time {raw!r}", "edit points are music times: 20b or 12.5s"))
        return None
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


def _pieces(ops: list[EditOp]) -> list[tuple[float, float]]:
    """Source ranges `(start, length)` seconds that make up the edited music, in play order."""
    pieces: list[tuple[float, float]] = [(0.0, _INF)]

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
    return out


def plan_audio(spec, spec_dir: Path | None, markers: dict[str, float], fps: int, bpm: float | None,
               beat0: float, errors: list[str]) -> MixPlan | None:
    """Validate `audio.edit/clips/duck` and resolve them to seconds. None without an audio block."""
    a = spec.audio
    if a is None:
        return None
    root = spec_dir or Path(".")
    ops = edit_ops(a.edit, fps, bpm, beat0, errors)
    clips: list[Placed] = []
    for i, c in enumerate(a.clips):
        base = f"audio.clips[{i}]"
        src = _clip_src(c.src, root, f"{base}.src", errors)
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
        ids = {c.id for c in a.clips if c.id}
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
    return MixPlan(str((root / a.track).resolve()), tuple(ops), tuple(clips), duck)


def _clip_src(raw: str, root: Path, path: str, errors: list[str]) -> str | None:
    if raw.startswith("synth:"):
        if raw[6:] not in SYNTH:
            errors.append(_line(path, f"unknown synth {raw[6:]!r}", f"have: {', '.join(SYNTH)}"))
            return None
        return raw
    found = _find_src(root, raw)
    if found is None:
        errors.append(_line(path, f"{raw!r} not found",
                            f"path relative to the spec file (extension optional) or synth:<{'|'.join(SYNTH)}>"))
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
    n = int(0.25 * SR)  # glitch: sample-and-hold stutter with random segment lengths
    out = np.zeros(n)
    i = 0
    while i < n:
        seg = int(rng.integers(40, 1400))
        out[i : i + seg] = rng.standard_normal() if rng.random() < 0.5 else np.sin(np.arange(min(seg, n - i)) * rng.uniform(0.02, 0.4))
        i += seg
    return _norm(out * np.exp(-np.arange(n) / SR * 8.0))


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
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    tmp.write_bytes(header + data)
    os.replace(tmp, path)


# --- mixing ------------------------------------------------------------------------------------


def _slice(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """`x[lo:hi]`, zero-padded where the range leaves the array."""
    out = np.zeros((hi - lo, x.shape[1]), dtype=x.dtype)
    a, b = max(lo, 0), min(hi, len(x))
    if a < b:
        out[a - lo : b - lo] = x[a:b]
    return out


def apply_edits(music: np.ndarray, ops: tuple[EditOp, ...] | list[EditOp]) -> np.ndarray:
    """Splice `music` per `ops`. Joins get an equal-power crossfade that keeps the length exact."""
    n = len(music)
    parts: list[list[int]] = []
    for s, length in _pieces(list(ops)):
        s0 = round(s * SR)
        s1 = n if length >= _INF / 2 else min(n, round((s + length) * SR))
        if s0 >= n or s1 <= s0:
            continue
        if parts and parts[-1][1] == s0:
            parts[-1][1] = s1
        else:
            parts.append([s0, s1])
    if not parts:
        raise EncodeError("mix: audio.edit leaves no music", "check the cut/loop ranges against the track length")
    out = np.concatenate([music[s0:s1] for s0, s1 in parts])
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


def _render_clip(c: Placed, cache: dict, log: Path | None) -> tuple[int, np.ndarray]:
    """`(start sample, stereo samples)` for one placement, trimmed, faded, gained and panned."""
    key = (c.src, c.stream)
    if key not in cache:
        cache[key] = decode(c.src, c.stream, log=log)
    x = cache[key]
    lo = round(c.in_s * SR)
    hi = len(x) if c.out_s is None else min(len(x), round(c.out_s * SR))
    x = x[lo:hi].copy()
    start = round(c.at_s * SR)
    if start < 0:  # an offset can reach before the film start: drop the head
        x = x[-start:]
        start = 0
    if len(x) == 0:
        return start, x
    f = min(round(CLIP_FADE_S * SR), len(x) // 2)
    if f > 0:
        ramp = np.sin(np.linspace(0.0, np.pi / 2, f, dtype=np.float32)) ** 2
        x[:f] *= ramp[:, None]
        x[-f:] *= ramp[::-1, None]
    x *= np.float32(10 ** (c.gain_db / 20))
    if c.pan:
        ang = (c.pan + 1) * np.pi / 4
        x *= np.array([math.sqrt(2) * math.cos(ang), math.sqrt(2) * math.sin(ang)], dtype=np.float32)
    return start, x


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


def mix_audio(plan: MixPlan, *, log: Path | None = None) -> tuple[np.ndarray, MixStats]:
    music = decode(plan.track, 0, log=log)
    if plan.edits:
        music = apply_edits(music, plan.edits)
    cache: dict = {}
    placed = [(c, *_render_clip(c, cache, log)) for c in plan.clips]
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
    for _, s, x in placed:
        bed[s : s + len(x)] += x
    peak = float(np.max(np.abs(bed))) if len(bed) else 0.0
    clipped = peak > PEAK_CEILING
    if clipped:
        bed *= np.float32(PEAK_CEILING / peak)
    peak_db = 20 * math.log10(peak) if peak > 0 else -120.0
    return bed, MixStats(total / SR, peak_db, clipped, len(placed), duck_db)


def plan_hash(plan: MixPlan) -> str:
    """Digest of the plan and the bytes of every input file."""
    shas: dict[str, str] = {}
    for p in [plan.track, *(c.src for c in plan.clips)]:
        if not p.startswith("synth:") and p not in shas:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            shas[p] = h.hexdigest()
    payload = json.dumps({"rev": MIX_REV, "plan": asdict(plan), "shas": shas}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def render_mix(plan: MixPlan, build_dir: Path, *, log: Path | None = None) -> tuple[Path, MixStats]:
    """The mixed wav for `plan` in `build_dir/audio/`, reused when its hash already exists."""
    digest = plan_hash(plan)
    wav = build_dir / "audio" / f"{digest}.wav"
    meta = wav.with_suffix(".json")
    if wav.is_file() and meta.is_file():
        stats = MixStats(**json.loads(meta.read_text()))
    else:
        audio, stats = mix_audio(plan, log=log)
        stats = MixStats(**{**asdict(stats), "hash": digest})
        write_wav(wav, audio)
        meta.write_text(json.dumps(asdict(stats)))
    (build_dir / "audio-mix.json").write_text(json.dumps(asdict(stats)))
    return wav, stats


def check_lines(stats: dict) -> list[str]:
    """`montaj check` lines from `build/audio-mix.json`."""
    parts = [f"{stats['clips']} clips"]
    if stats["duck_db"] < 0:
        parts.append(f"duck {stats['duck_db']:.1f} dB")
    lines = [f"OK audio-mix {', '.join(parts)}, peak {stats['peak_dbfs']:+.1f} dBFS"]
    if stats["clipped"]:
        lines.append(f"WARN clip clipped (mix peak {stats['peak_dbfs']:+.1f} dBFS, scaled to -0.2 dBFS)")
    return lines


__all__ = ["DuckPlan", "EditOp", "MixPlan", "MixStats", "Placed", "apply_edits", "check_lines",
           "duck_gain", "edit_ops", "plan_audio", "plan_hash", "render_mix", "shift_words", "synth"]
