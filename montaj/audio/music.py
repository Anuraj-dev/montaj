"""`montaj music gen`: ACE-Step candidates, one subprocess, the whole GPU to itself.

Port of `claude-test/bday-video/music/generate.py`, kept where it matters:
- `py.sh` line 3-4: a working CPython 3.12 plus the ACE-Step venv's site-packages (config).
- `generate.py` line 26: `offload_to_cpu=True, offload_dit_to_cpu=True` — on 8 GB the LM phase OOMs
  otherwise, which is also why this command holds the GPU lock.
- `generate.py` lines 30-44: the same `GenerationParams`, one `GenerationConfig(batch_size=1,
  audio_format='wav', use_random_seed=True)` per candidate.

What changed for montaj: candidates are saved as `music/cand-<seed>.wav` + `.json` (params and
seed) so a spec can name one and a re-run can reproduce it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from montaj.audio._worker import AudioError, parse_candidates, run_worker
from montaj.config import Config, load
from montaj.encode import log_file
from montaj.gpulock import hold

WORKER = Path(__file__).with_name("_ace_worker.py")
DEFAULT_BPM = 76.0  # film.html line 76: const BEAT = 60 / 76
DEFAULT_KEY = "D major"
DEFAULT_LANG = "hi"
DEFAULT_DURATION = 165.0
DEFAULT_N = 3

_TERM = re.compile(r"(?P<v>\d+(?:\.\d+)?)(?P<u>m|s)?")


def parse_duration(raw: str | float | int) -> float:
    """`165s`, `2m45s`, `165` -> seconds. Raises ValueError on anything else."""
    if isinstance(raw, (int, float)):
        seconds = float(raw)
        if seconds > 0:
            return seconds
        raise ValueError(f"duration: must be positive, got {raw}")
    text = str(raw).strip().lower()
    total, pos = 0.0, 0
    for term in _TERM.finditer(text):
        if term.start() != pos:
            break
        total += float(term["v"]) * (60.0 if term["u"] == "m" else 1.0)
        pos = term.end()
    if pos != len(text) or pos == 0:
        raise ValueError(f"duration: cannot read {raw!r} (use 165s, 2m45s or plain seconds)")
    if total <= 0:
        raise ValueError(f"duration: must be positive, got {raw!r}")
    return total


class PartialGenError(AudioError):
    """Some candidates were written, but fewer than `--n`. Files stay; the command is still a failure."""

    def __init__(self, files: tuple[Path, ...], seeds: tuple[str, ...], n: int, reason: str) -> None:
        self.files = files
        self.seeds = seeds
        self.n = n
        self.reason = reason
        super().__init__(f"music gen: {len(files)}/{n} candidates ({reason})")


@dataclass(frozen=True)
class Candidates:
    """What one `music gen` produced: the wavs to listen to and the seeds to reproduce them."""

    files: tuple[Path, ...]
    seeds: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.files)


def gen(
    out_dir: Path | str,
    *,
    caption: str,
    lyrics: Path | str,
    bpm: float = DEFAULT_BPM,
    key: str = DEFAULT_KEY,
    lang: str = DEFAULT_LANG,
    duration: float = DEFAULT_DURATION,
    n: int = DEFAULT_N,
    config: Config | None = None,
    log: Path | str | None = None,
) -> Candidates:
    """Generate `n` candidates into `out_dir` as `cand-<seed>.wav` / `cand-<seed>.json`."""
    cfg = config or load()
    lyrics_path = Path(lyrics).expanduser().resolve()
    if not lyrics_path.is_file():
        raise AudioError(f"lyrics: {lyrics_path} does not exist")
    if n < 1:
        raise AudioError(f"music gen: n must be at least 1, got {n}")
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_path = log_file("music", log)
    cmd = cfg.ace_argv(
        WORKER,
        "--ace-dir", cfg.ace_step_dir,
        "--out", out,
        "--lyrics", lyrics_path,
        "--caption", caption,
        "--bpm", bpm,
        "--key", key,
        "--lang", lang,
        "--duration", duration,
        "--n", n,
    )
    try:
        with hold(log_path):
            raw = run_worker("ace-step", cmd, log=log_path, env=cfg.ace_env(), cwd=cfg.ace_step_dir)
    except AudioError as exc:
        if not any(line.startswith("CAND ") for line in exc.lines):
            raise
        raw = exc.lines
    cand_lines, fail_reason = _cand_lines(raw)
    made = parse_candidates(cand_lines, "ace-step")
    files, seeds = [], []
    for item in made:
        wav = Path(str(item["path"]))
        if not wav.is_file():
            raise AudioError(f"ace-step: candidate {wav} is missing (worker said it wrote it)")
        files.append(wav)
        seeds.append(str(item["seed"]))
    if not files:
        raise AudioError("ace-step: no candidates (see the log)")
    if len(files) < n or fail_reason:
        raise PartialGenError(tuple(files), tuple(seeds), n, fail_reason or "incomplete")
    return Candidates(files=tuple(files), seeds=tuple(seeds))


def _cand_lines(lines: list[str]) -> tuple[list[str], str | None]:
    """Split worker stdout into `CAND` lines and the first `FAIL` reason, if any."""
    cands: list[str] = []
    reason: str | None = None
    for line in lines:
        if line.startswith("CAND "):
            cands.append(line)
        elif line.startswith("FAIL "):
            if reason is None:
                reason = line[5:].strip() or "incomplete"
        else:
            raise AudioError(f"ace-step: unexpected worker output {' '.join(line.split())!r}")
    return cands, reason