"""`montaj music analyze`: word timings, a beat grid and a loudness envelope -> `music/markers.json`.

Port of `claude-test/bday-video/music/analyze.py` — same model (large-v3-turbo, cuda float16,
`word_timestamps=True`, `vad_filter=False`, beam 5), an initial prompt built from `lyrics.txt`, same 0.5 s-hop
RMS (lines 20-21). Added on top, per 002 §Audio: a beat grid (the worker tracks it from an onset
envelope, since that venv has numpy and not librosa) and the `first_word`/`last_word` word indices
the `word:<i>` markers resolve against.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from montaj.audio import _whisper_worker as beat_worker
from montaj.audio._worker import run_worker
from montaj.config import WHISPER_KEYS, Config, ConfigError, load
from montaj.audio._worker import append_log
from montaj.encode import FFMPEG, log_file
from montaj.gpulock import hold

WORKER = Path(__file__).with_name("_whisper_worker.py")
DEFAULT_LANG = "hi"
PROMPT_CHARS = 240  # whisper's initial prompt is a short hint, not a transcript


def default_prompt(wav: Path) -> str:
    """Whisper hint: the words of `lyrics.txt` beside the wav (as `music gen` leaves it), else none.

    Names and non-English words in the hint are what it fixes most; section tags like `[verse]` are dropped.
    """
    lyrics = wav.parent / "lyrics.txt"
    if not lyrics.is_file():
        return ""
    lines = [l.strip() for l in lyrics.read_text(encoding="utf-8").splitlines()]
    text = ", ".join(l for l in lines if l and not (l.startswith("[") and l.endswith("]")))
    return text[:PROMPT_CHARS]
DEFAULT_OUT = "music/markers.json"


@dataclass(frozen=True)
class Analysis:
    """`music/markers.json` plus the lines the CLI prints: one per segment, newest information last."""

    path: Path
    bpm: float
    n_words: int
    n_beats: int
    lines: tuple[str, ...]
    duration: float
    first_beat: float | None
    first_word: float | None
    lang: str
    no_words: bool


def analyze(
    wav: Path | str,
    out: Path | str = DEFAULT_OUT,
    *,
    lang: str = DEFAULT_LANG,
    prompt: str | None = None,
    bpm: float | None = None,
    no_words: bool = False,
    config: Config | None = None,
    log: Path | str | None = None,
) -> Analysis:
    """Transcribe `wav`, or track only beats when ``no_words`` is true."""
    wav_path = Path(wav).expanduser()
    if not wav_path.is_file():
        raise FileNotFoundError(f"wav: {wav_path} does not exist")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_file("music", log)
    if bpm is not None and (not np.isfinite(bpm) or bpm <= 0):
        raise ValueError("bpm must be finite and positive")
    if no_words:
        return _analyze_without_words(wav_path, out_path, lang=lang, bpm=bpm, log=log_path)
    try:
        cfg = config or load(required=WHISPER_KEYS)
    except ConfigError as exc:
        raise ConfigError(f"{exc}; use --no-words for beat-only analysis") from None
    args: list[object] = [
        "--wav", wav_path, "--out", out_path, "--lang", lang,
        "--prompt", prompt if prompt is not None else default_prompt(wav_path),
    ]
    if bpm is not None:
        args += ["--bpm", bpm]  # absent means "estimate the tempo", not "the tempo is None"
    cmd = cfg.whisper_argv(WORKER, *args)
    with hold(log_path):
        lines = run_worker("whisper", cmd, log=log_path)
    if not out_path.is_file():
        raise RuntimeError("whisper: no markers file (worker exit 0 but nothing was written)")
    markers = json.loads(out_path.read_text(encoding="utf-8"))
    if not isinstance(markers, dict):
        raise RuntimeError(f"whisper: {out_path} is not a markers file")
    return Analysis(
        path=out_path,
        bpm=float(markers.get("bpm") or 0.0),
        n_words=len(markers.get("words") or ()),
        n_beats=len(markers.get("beats") or ()),
        lines=tuple(lines),
        duration=float(markers.get("duration") or 0.0),
        first_beat=_first_time(markers.get("beats")),
        first_word=_first_time(markers.get("words"), key="s"),
        lang=lang,
        no_words=False,
    )


def _analyze_without_words(
    wav: Path,
    out: Path,
    *,
    lang: str,
    bpm: float | None,
    log: Path,
) -> Analysis:
    audio = _decode_audio(wav, log)
    duration = len(audio) / beat_worker.SR
    envelope, rate = beat_worker.onset_envelope(audio)
    measured_bpm, beats = beat_worker.beat_track(envelope, rate, bpm, duration=duration)
    markers = beat_worker.build_markers(
        [], bpm=measured_bpm, beats=beats, rms=beat_worker.rms_envelope(audio), duration=duration,
    )
    out.write_text(json.dumps(markers, ensure_ascii=False, indent=1), encoding="utf-8")
    return Analysis(
        path=out,
        bpm=measured_bpm,
        n_words=0,
        n_beats=len(beats),
        lines=(),
        duration=duration,
        first_beat=_first_time(beats),
        first_word=None,
        lang=lang,
        no_words=True,
    )


def _decode_audio(wav: Path, log: Path) -> np.ndarray:
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(wav), "-vn", "-sn", "-dn",
        "-f", "f32le", "-ac", "1", "-ar", str(beat_worker.SR), "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stderr = proc.stderr.decode("utf-8", "replace")
    append_log(log, cmd, "", stderr)
    if proc.returncode != 0:
        tail = " ".join(stderr.splitlines()[-1:]).strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"ffmpeg decode failed: {tail}")
    audio = np.frombuffer(proc.stdout, dtype="<f4")
    if not audio.size:
        raise RuntimeError("ffmpeg decode returned no audio samples")
    return audio.astype(np.float32, copy=False)


def _first_time(values: object, *, key: str | None = None) -> float | None:
    if not isinstance(values, (list, tuple)) or not values:
        return None
    value = values[0]
    if key is not None:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
