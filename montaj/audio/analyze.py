"""`montaj music analyze`: word timings, a beat grid and a loudness envelope -> `music/markers.json`.

Port of `claude-test/bday-video/music/analyze.py` — same model (large-v3-turbo, cuda float16,
`word_timestamps=True`, `vad_filter=False`, beam 5), same initial prompt (line 14), same 0.5 s-hop
RMS (lines 20-21). Added on top, per 002 §Audio: a beat grid (the worker tracks it from an onset
envelope, since that venv has numpy and not librosa) and the `first_word`/`last_word` word indices
the `word:<i>` markers resolve against.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from montaj.audio._worker import run_worker
from montaj.config import Config, load
from montaj.encode import log_file
from montaj.gpulock import hold

WORKER = Path(__file__).with_name("_whisper_worker.py")
DEFAULT_LANG = "hi"
DEFAULT_PROMPT = "Arshiya, Samar, Jaipur, Jama Masjid, janamdin mubarak, chaar saal, baaees baras"
DEFAULT_OUT = "music/markers.json"


@dataclass(frozen=True)
class Analysis:
    """`music/markers.json` plus the lines the CLI prints: one per segment, newest information last."""

    path: Path
    bpm: float
    n_words: int
    n_beats: int
    lines: tuple[str, ...]


def analyze(
    wav: Path | str,
    out: Path | str = DEFAULT_OUT,
    *,
    lang: str = DEFAULT_LANG,
    prompt: str | None = None,
    bpm: float | None = None,
    config: Config | None = None,
    log: Path | str | None = None,
) -> Analysis:
    """Transcribe `wav`, track its beats, write `out`. `bpm` given skips tempo estimation."""
    cfg = config or load()
    wav_path = Path(wav).expanduser()
    if not wav_path.is_file():
        raise FileNotFoundError(f"wav: {wav_path} does not exist")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_file("music", log)
    args: list[object] = [
        "--wav", wav_path, "--out", out_path, "--lang", lang,
        "--prompt", prompt if prompt is not None else DEFAULT_PROMPT,
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
    )