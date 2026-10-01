"""Whisper word timings + a beat grid for one song. Runs under `whisper_python`; imports nothing from montaj.

Port of `claude-test/bday-video/music/analyze.py`: the same nvidia-library preload (lines 4-6), the
same `large-v3-turbo` on cuda float16 with `word_timestamps=True, vad_filter=False, beam_size=5`
(lines 12-13), the same 0.5 s-hop RMS (lines 20-21).

What is new: the beat tracker. That venv has numpy but no librosa, so tempo comes from an onset
envelope plus autocorrelation and phase from the grid offset that collects the most onset energy
(002 §Audio).

Only numpy is imported at module scope, deliberately: the tracker stays unit-testable inside
montaj's own process. `faster_whisper` is imported inside `main`, and fd 1 is pointed at stderr
before that happens, so the model's own prints land in montaj's log and never in the result lines.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

MODEL = "large-v3-turbo"
SR = 16000  # decode_audio resamples here
HOP = 80  # onset-envelope hop: 5 ms, fine enough that a beat lands where the click is
WIN = 256  # 16 ms analysis window; short enough that the flux frame is not a window early
BPM_MIN, BPM_MAX = 60.0, 180.0
RMS_HOP = SR // 2  # 0.5 s loudness envelope (analyze.py:20)
SECONDS_DP = 2  # film.html's C map is written to hundredths of a second
RESULT_FD = 1


# --- onset envelope + tempo (pure numpy) -----------------------------------------------------

def onset_envelope(x: np.ndarray, *, sr: int = SR, hop: int = HOP, win: int = WIN) -> tuple[np.ndarray, float]:
    """Spectral-flux onset strength at `hop` samples, and the frame rate in Hz.

    Energy entering the window between two frames, summed over bins and half-wave rectified.
    Flux frame i is timed at (i + 1) / rate: the moment the onset entered the window, not the
    window's centre, which is why a click reads where it actually is.
    """
    if len(x) < win:
        x = np.pad(x.astype(np.float32), (0, win - len(x)))
    # Frames centred on sample j*hop, so flux frame j (the change entering window j+1) is timed at
    # (j + 1) * hop — an onset lands where it is, not a window off, and not a window early.
    padded = np.pad(x, (win // 2, 0))
    frames = np.lib.stride_tricks.sliding_window_view(padded, win)[::hop] * np.hanning(win)
    log_mag = np.log1p(64.0 * np.abs(np.fft.rfft(frames, axis=1)))
    env = np.maximum(0.0, np.diff(log_mag, axis=0)).sum(axis=1)
    return env / (env.max() or 1.0), sr / hop


def _autocorr_peak(env: np.ndarray, rate: float) -> float:
    """Lag (frames) whose autocorrelation of the onset envelope peaks inside 60..180 bpm.

    A bar of accented clicks peaks at the beat lag, not at half of it, so `argmax` lands on the
    beat rather than its double — that is what fixes the tempo octave. Returns 0.0 when the clip
    is too short to hold a lag in range.
    """
    lo, hi = int(np.ceil(rate * 60.0 / BPM_MAX)), int(np.floor(rate * 60.0 / BPM_MIN))
    centred = env - env.mean()
    lags = np.arange(lo, min(hi, len(centred) - 1) + 1)
    if lags.size == 0:
        return 0.0
    scores = np.array([float(np.dot(centred[: len(centred) - lag], centred[lag:])) for lag in lags])
    i = int(np.argmax(scores))
    return float(lags[i] + _peak_shift(scores, i))


def _peak_shift(ys: np.ndarray, i: int) -> float:
    """Sub-index shift of the peak at index `i`, parabolically refined so a quantised lag or
    a 1 ms phase grid does not cost real accuracy."""
    if not 0 < i < ys.size - 1:
        return 0.0
    a, b, c = float(ys[i - 1]), float(ys[i]), float(ys[i + 1])
    denom = a - 2.0 * b + c
    return float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0)) if denom else 0.0


_PHASE_COARSE = 0.01  # tempo search phase step, 10 ms
_PHASE_FINE = 0.001  # final phase search step, 1 ms


def _phase_grid(bpm: float, step: float) -> np.ndarray:
    """Candidate grid offsets for `bpm`, one beat period long."""
    return np.arange(0.0, 60.0 / bpm, step)


def _comb_scores(env: np.ndarray, rate: float, bpm: float, offsets: np.ndarray, duration: float) -> np.ndarray:
    """Mean onset strength on the beat grid `offset + n · 60 / bpm`, one score per offset."""
    period = 60.0 / bpm
    n_beats = int(np.ceil(duration / period)) + 1
    idx = np.round((offsets[:, None] + np.arange(n_beats)[None, :] * period) * rate).astype(int) - 1
    usable = (idx >= 0) & (idx < env.size)
    values = np.where(usable, env[np.clip(idx, 0, env.size - 1)], 0.0)
    return values.sum(axis=1) / np.maximum(usable.sum(axis=1), 1)


def _best_phase(env: np.ndarray, rate: float, bpm: float, duration: float, step: float = _PHASE_FINE) -> float:
    """Grid offset (seconds) that puts the most onset energy on the beats, searched at `step`."""
    offsets = _phase_grid(bpm, step)
    scores = _comb_scores(env, rate, bpm, offsets, duration)
    i = int(np.argmax(scores))
    return float(offsets[i] + _peak_shift(scores, i) * step)


def _refine_tempo(env: np.ndarray, rate: float, bpm: float, duration: float) -> float:
    """Tempora within ±5% of the ACF estimate, ranked by comb score at their best phase.

    The ACF alone is quantised by the envelope hop; the comb alone cannot tell 76 from 152. Using
    the comb only to refine keeps the octave the ACF decided.
    """
    lo = max(BPM_MIN, bpm * 0.95)
    hi = min(BPM_MAX, bpm * 1.05)
    best_bpm, best_score = float(np.clip(bpm, BPM_MIN, BPM_MAX)), -1.0
    if lo > hi:
        return best_bpm
    for cand in np.arange(lo, hi, 0.01):
        score = float(_comb_scores(env, rate, float(cand), _phase_grid(float(cand), _PHASE_COARSE), duration).max())
        if score > best_score:
            best_bpm, best_score = float(cand), score
    return best_bpm

def beat_track(env: np.ndarray, rate: float, bpm: float | None = None, *, duration: float | None = None) -> tuple[float, list[float]]:
    """(bpm, beat times in seconds) for an onset envelope sampled at `rate` Hz.

    `bpm` given is authoritative (no tempo search); otherwise autocorrelation proposes one and the
    comb refines it. Phase is always searched, so a known tempo still lands on the beats.
    """
    end = duration if duration is not None else (len(env) + 1) / rate
    if bpm is None:
        lag = _autocorr_peak(env, rate)
        bpm = _refine_tempo(env, rate, (60.0 * rate / lag) if lag else 120.0, end)
    beats = list(np.arange(_best_phase(env, rate, bpm, end), end + 1e-9, 60.0 / bpm))
    return round(float(bpm), 3), [round(float(b), 3) for b in beats]


def rms_envelope(x: np.ndarray, *, hop: int = RMS_HOP) -> list[float]:
    """RMS per `hop` samples — the 0.5 s loudness envelope of analyze.py:20-21."""
    return [float(np.sqrt(np.mean(x[i : i + hop] ** 2))) for i in range(0, len(x) - hop, hop)]


# --- markers.json --------------------------------------------------------------------------

def build_markers(
    segments: list[dict[str, object]],
    *,
    bpm: float,
    beats: list[float],
    rms: list[float],
    duration: float,
) -> dict[str, object]:
    """`{duration, bpm, beats, words, segments, rms}` — the file `word:<i>` markers read.

    Word `i` is 1-based and global across segments, matching `markers: {v1: word:1}` in a spec
    (002 §Time). Times are hundredths of a second, the precision film.html's C map is written at.
    """
    words: list[dict[str, object]] = []
    out_segments: list[dict[str, object]] = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        spoken = seg.get("words") or []
        first = last = None
        for word in spoken:
            text = str(word["w"]).strip()
            if not text or word.get("s") is None or word.get("e") is None:
                continue  # whisper can emit a word with no timing; it cannot anchor a marker
            if first is None:
                first = len(words) + 1
            last = len(words) + 1
            words.append({"i": last, "w": text, "s": round(float(word["s"]), SECONDS_DP),
                          "e": round(float(word["e"]), SECONDS_DP)})
        out_segments.append({
            "s": round(start, SECONDS_DP),
            "e": round(end, SECONDS_DP),
            "text": " ".join(str(seg.get("text", "")).split()),
            "first_word": first,
            "last_word": last,
        })
    return {
        "duration": round(float(duration), 3),
        "bpm": bpm,
        "beats": beats,
        "words": words,
        "segments": out_segments,
        "rms": [round(v, 6) for v in rms],
    }


def segment_line(seg: dict[str, object]) -> str:
    """`12.00-23.74 w1-w15 Jaipur ki dhup mein…` — one line per segment, words ellipsised."""
    words = f"w{seg['first_word']}-{seg['last_word']}" if seg["first_word"] else "w0-0"
    text = str(seg["text"])
    if len(text) > 60:
        text = text[:60].rstrip() + "…"
    return f"{seg['s']:.2f}-{seg['e']:.2f} {words} {text}".rstrip()


# --- whisper --------------------------------------------------------------------------------

def _emit(line: str) -> None:
    os.write(RESULT_FD, (line + "\n").encode())


def _claim_stdout() -> None:
    """Point fd 1 at stderr and keep the real one for `_emit` (analyze.py prints to stdout, but
    faster_whisper and ctranslate2 also print, and those must not reach montaj's protocol)."""
    global RESULT_FD
    RESULT_FD = os.dup(1)
    os.dup2(2, 1)


def _preload_cuda_libs() -> None:
    """analyze.py lines 4-6: torch's wheels keep CUDA in site-packages/nvidia; load them RTLD_GLOBAL
    so faster_whisper's ctranslate2 resolves the driver."""
    for lib in sorted(glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/*/lib/*.so*"))):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="whisper word timings + beat grid -> markers.json")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--bpm", type=float, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    ns = _parser().parse_args(argv)
    _claim_stdout()
    _preload_cuda_libs()
    from faster_whisper import WhisperModel, decode_audio

    audio = decode_audio(ns.wav)  # 16 kHz mono
    model = WhisperModel(MODEL, device="cuda", compute_type="float16")
    segs, _info = model.transcribe(
        audio,
        language=ns.lang,
        word_timestamps=True,
        vad_filter=False,
        beam_size=5,
        initial_prompt=ns.prompt or None,
    )
    raw = [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "words": [{"w": w.word, "s": w.start, "e": w.end} for w in (s.words or [])],
        }
        for s in segs
    ]
    duration = len(audio) / SR
    env, rate = onset_envelope(audio)
    bpm, beats = beat_track(env, rate, ns.bpm, duration=duration)
    markers = build_markers(raw, bpm=bpm, beats=beats, rms=rms_envelope(audio), duration=duration)
    Path(ns.out).write_text(json.dumps(markers, ensure_ascii=False, indent=1), encoding="utf-8")
    for seg in markers["segments"]:
        _emit(segment_line(seg))
    return 0


if __name__ == "__main__":
    sys.exit(main())