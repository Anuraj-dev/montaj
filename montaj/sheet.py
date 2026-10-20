"""Frame contact sheets: pick frames from a video, draw their timestamps, save JPEGs.

`compose_grid` is the shared grid builder (ingest reuses it for photo sheets).
The default sheet is one JPEG of ≤ 12 frames. `--every` samples a fixed interval
and writes more pages (`sheet.jpg`, `sheet-2.jpg`, …) when that is more than 12.
"""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from montaj.encode import FFMPEG, MediaInfo, probe, run

MAX_TILES = 12
# Frames per ffmpeg call. ffmpeg's expression parser recurses on a left-associated
# `a+b+c…` chain and fails past ~100 terms, so extraction runs in bounded batches.
EXTRACT_BATCH = 48
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


def every_times(duration: float, interval: float, *, fps: float | None = None) -> list[float]:
    """`0, interval, 2*interval, …` while each time is inside `duration`.

    A 60s film at 0.5s is 120 times (0.0 … 59.5): one sample per interval, the
    endpoint itself left out. `fps` rejects a step finer than one frame.
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError(f"interval {interval}: use a positive number of seconds, like 0.5s")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"unknown duration {duration}, pass explicit times")
    span = (duration - 1e-9) / interval
    if not math.isfinite(span):
        raise ValueError(
            f"interval {interval:g}s would sample too many frames across {duration:.2f}s; "
            "use a coarser interval like 0.5s"
        )
    # A duration shorter than the epsilon still gets the opening frame.
    n = max(1, int(math.floor(span)) + 1)
    if fps is not None and fps > 0:
        n_frames = max(1, int(round(duration * fps)))
        if n > n_frames:
            frame = 1.0 / fps
            raise ValueError(
                f"interval {interval:g}s is finer than one frame ({frame:.4f}s at {fps:g} fps); "
                "use a coarser step such as 0.5s"
            )
    times = [i * interval for i in range(n)]
    # `i * interval` can land on the endpoint from rounding. That frame is the
    # next film's, not this interval's.
    return [t for t in times if t < duration]


def _frame_number(path: Path) -> int:
    """`frame_12.jpg` -> 12; numeric so `frame_1000` sorts after `frame_999`."""
    return int(path.stem.rsplit("_", 1)[1])


def pair_frames(picks: Sequence[int], files: Sequence[Path]) -> list[tuple[int, Path]]:
    """Pair sorted frame indexes with ffmpeg's numbered output, in numeric file order."""
    return list(zip(picks, sorted(files, key=_frame_number)))


def extract(video: Path | str, times: Sequence[float], dest: Path | str, *,
            info: MediaInfo | None = None, log: Path | str | None = None) -> list[tuple[int, Path]]:
    """Write the wanted frames as JPEGs, `EXTRACT_BATCH` per ffmpeg pass; returns (index, file) in time order.

    Times are clamped to the clip and deduped, so near-identical requests collapse to one tile.
    """
    info = info or probe(video, log=log)
    if info.fps <= 0:
        raise ValueError(f"{video}: unknown fps")
    n_frames = int(round(info.duration * info.fps)) or 1
    picks = sorted({min(max(0, int(round(t * info.fps))), n_frames - 1) for t in times})
    dest = Path(dest)
    out: list[tuple[int, Path]] = []
    for b, start in enumerate(range(0, len(picks), EXTRACT_BATCH)):
        batch = picks[start:start + EXTRACT_BATCH]
        folder = dest / f"b{b:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        expr = "+".join(f"eq(n\\,{i})" for i in batch)
        run(
            [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
             "-vf", f"select='{expr}'", "-fps_mode", "passthrough", "-q:v", "2",
             str(folder / "frame_%d.jpg")],
            log=log,
            desc="sheet",
        )
        out += pair_frames(batch, list(folder.glob("frame_*.jpg")))
    return out


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


def page_path(out: Path | str, page: int) -> Path:
    """Page 0 keeps `out`. Later pages are `<stem>-2.jpg`, `<stem>-3.jpg`, …"""
    out = Path(out)
    if page <= 0:
        return out
    return out.with_name(f"{out.stem}-{page + 1}{out.suffix}")


def _drop_stale_pages(out: Path, keep_pages: int) -> None:
    """Remove pages left by a denser sheet at the same path."""
    page = keep_pages
    while True:
        path = page_path(out, page)
        if not path.is_file():
            return
        path.unlink()
        page += 1


def sheets(video: Path | str, out: Path | str, times: Sequence[float], *,
           per_page: int = MAX_TILES, log: Path | str | None = None) -> list[Path]:
    """One JPEG per page of ≤ `per_page` frames, in time order. Returns the paths written."""
    info = probe(video, log=log)
    tmp = Path(tempfile.mkdtemp(prefix="montaj-sheet-"))
    try:
        frames = extract(video, times, tmp / "frames", info=info, log=log)
        if not frames:
            raise RuntimeError(f"sheet: no frames extracted from {video}")
        step = max(1, int(per_page))
        paths: list[Path] = []
        for page, start in enumerate(range(0, len(frames), step)):
            chunk = frames[start:start + step]
            images = [Image.open(path) for _, path in chunk]
            try:
                labels = [stamp(index / info.fps) for index, _ in chunk]
                dest = page_path(out, page)
                compose_grid(images, labels, dest)
                paths.append(dest)
            finally:
                for image in images:
                    image.close()
        _drop_stale_pages(Path(out), len(paths))
        return paths
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sheet(video: Path | str, out: Path | str, at: Sequence[float] | float | None = None, n: int = 12, *,
          log: Path | str | None = None) -> Path:
    """One JPEG contact sheet of ≤ 12 frames, ≤ 1536 px wide, each labelled mm:ss.ff."""
    times = pick_times(video, at, n)
    if len(times) > MAX_TILES:
        times = times[:MAX_TILES]
    return sheets(video, out, sorted(times), log=log)[0]