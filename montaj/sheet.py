"""Frame contact sheets: pick frames from a video, draw their timestamps, save one JPEG.

`compose_grid` is the shared grid builder (ingest reuses it for photo sheets).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from montaj.encode import FFMPEG, MediaInfo, probe, run

MAX_TILES = 12
SHEET_MAX_W = 1536
SHEET_COLS = 4
TILE_RATIO = 4 / 3
BG = (18, 18, 20)
FG = (235, 235, 235)


def stamp(t: float) -> str:
    """Seconds as mm:ss.ff."""
    cs = max(0, int(round(t * 100)))
    return f"{cs // 6000:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def pick_times(video: Path | str, at: Sequence[float] | float | None = None, n: int = 12,
               info: MediaInfo | None = None) -> list[float]:
    """`at` (seconds) verbatim, else `n` evenly spaced midpoints of the duration."""
    if at is not None:
        times = [float(at)] if isinstance(at, (int, float)) else [float(t) for t in at]
        return [max(0.0, t) for t in times]
    info = info or probe(video)
    duration = info.duration
    if duration <= 0:
        raise ValueError(f"{video}: unknown duration, pass explicit times")
    count = max(1, min(int(n), MAX_TILES))
    return [(i + 0.5) * duration / count for i in range(count)]


def extract(video: Path | str, times: Sequence[float], dest: Path | str, *,
            info: MediaInfo | None = None, log: Path | str | None = None) -> list[tuple[int, Path]]:
    """One ffmpeg pass writing the wanted frames as `frame_%03d.jpg`; returns (index, file) in time order.

    Times are clamped to the clip and deduped, so near-identical requests collapse to one tile.
    """
    info = info or probe(video, log=log)
    if info.fps <= 0:
        raise ValueError(f"{video}: unknown fps")
    n_frames = int(round(info.duration * info.fps)) or 1
    picks = sorted({min(max(0, int(round(t * info.fps))), n_frames - 1) for t in times})
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    expr = "+".join(f"eq(n\\,{i})" for i in picks)
    run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"select='{expr}'", "-fps_mode", "passthrough", "-q:v", "2",
         str(dest / "frame_%03d.jpg")],
        log=log,
        desc="sheet",
    )
    return [(i, path) for i, path in zip(picks, sorted(dest.glob("frame_*.jpg")))]


def compose_grid(tiles: Sequence[Image.Image], labels: Sequence[str], out: Path | str, *,
                 max_w: int = SHEET_MAX_W, cols: int = SHEET_COLS, tile_ratio: float = TILE_RATIO) -> Path:
    """Grid of thumbnails with a caption band under each, ≤ `max_w` wide."""
    if not tiles:
        raise ValueError("compose_grid: no tiles")
    cols = max(1, min(int(cols), len(tiles)))
    rows = -(-len(tiles) // cols)
    tile_w = max_w // cols
    thumb_h = int(tile_w * tile_ratio)
    label_h = max(18, tile_w // 14)
    sheet = Image.new("RGB", (cols * tile_w, rows * (thumb_h + label_h)), BG)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=max(11, tile_w // 16))
    for i, tile in enumerate(tiles):
        x, y = (i % cols) * tile_w, (i // cols) * (thumb_h + label_h)
        thumb = ImageOps.contain(tile.convert("RGB"), (tile_w, thumb_h), method=Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x + (tile_w - thumb.width) // 2, y + (thumb_h - thumb.height) // 2))
        if i < len(labels):
            draw.text((x + 6, y + thumb_h + 4), labels[i], font=font, fill=FG)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out


def sheet(video: Path | str, out: Path | str, at: Sequence[float] | float | None = None, n: int = 12, *,
          log: Path | str | None = None) -> Path:
    """One JPEG contact sheet of ≤ 12 frames, ≤ 1536 px wide, each labelled mm:ss.ff."""
    times = pick_times(video, at, n)
    if len(times) > MAX_TILES:
        times = times[:MAX_TILES]
    times = sorted(times)
    info = probe(video, log=log)
    tmp = Path(tempfile.mkdtemp(prefix="montaj-sheet-"))
    try:
        frames = extract(video, times, tmp / "frames", info=info, log=log)
        if not frames:
            raise RuntimeError(f"sheet: no frames extracted from {video}")
        labels = [stamp(index / info.fps) for index, _ in frames]
        return compose_grid([Image.open(path) for _, path in frames], labels, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)