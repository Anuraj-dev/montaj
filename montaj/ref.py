"""`montaj ref <video>`: a contact sheet of a reference video sampled at its cuts, with the measured cut rate (issue #26)."""
from __future__ import annotations

import re
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from montaj.encode import FFMPEG, probe, run
from montaj.sheet import MAX_TILES, compose_grid, extract, stamp

SCENE_THRESHOLD = 0.3  # ffmpeg scene score above which a frame starts a new shot
MIN_SHOT_S = 0.2  # cuts closer than this are one flash, not two shots
_PTS = re.compile(r"pts_time:([0-9.]+)")


@dataclass(frozen=True)
class RefStats:
    duration: float
    cuts: tuple[float, ...]  # seconds a new shot starts, excluding t=0

    @property
    def shots(self) -> int:
        return len(self.cuts) + 1

    @property
    def per_10s(self) -> float:
        return len(self.cuts) / self.duration * 10 if self.duration > 0 else 0.0

    @property
    def median_shot(self) -> float:
        edges = [0.0, *self.cuts, self.duration]
        return statistics.median(b - a for a, b in zip(edges, edges[1:]))


def detect_cuts(video: Path, duration: float, *, threshold: float = SCENE_THRESHOLD, log: Path | None = None) -> tuple[float, ...]:
    """Cut times from ffmpeg's scene score. Cuts nearer than MIN_SHOT_S to the previous one are merged."""
    _, stderr = run([FFMPEG, "-hide_banner", "-i", str(video), "-an", "-vf", f"select='gt(scene,{threshold})',showinfo",
                     "-f", "null", "-"], log=log, desc="ref")
    times = sorted(float(m.group(1)) for m in _PTS.finditer(stderr))
    cuts: list[float] = []
    for t in times:
        if t >= MIN_SHOT_S and t < duration and (not cuts or t - cuts[-1] >= MIN_SHOT_S):
            cuts.append(t)
    return tuple(cuts)


def sample_times(stats: RefStats, n: int = MAX_TILES) -> list[float]:
    """The first moments of each shot (0.1 s in, so the transition has settled), evenly thinned to `n`."""
    starts = [0.0, *stats.cuts]
    pick = starts if len(starts) <= n else [starts[round(i * (len(starts) - 1) / (n - 1))] for i in range(n)]
    return [min(t + 0.1, max(0.0, stats.duration - 0.05)) for t in pick]


def study(video: Path, out: Path, *, log: Path | None = None) -> tuple[Path, RefStats, int]:
    """Write the sheet to `out`; return (path, stats, tiles shown)."""
    info = probe(video, log=log)
    stats = RefStats(info.duration, detect_cuts(video, info.duration, log=log))
    times = sample_times(stats)
    tmp = Path(tempfile.mkdtemp(prefix="montaj-ref-"))
    try:
        frames = extract(video, times, tmp / "frames", info=info, log=log)
        images = [Image.open(p) for _, p in frames]
        try:
            compose_grid(images, [stamp(i / info.fps) for i, _ in frames], out)
        finally:
            for im in images:
                im.close()
        return out, stats, len(frames)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
