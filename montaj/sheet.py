"""Frame contact sheets: pick frames from a video, draw their timestamps, save JPEGs.

`compose_grid` is the shared grid builder (ingest reuses it for photo sheets).
The default sheet is one JPEG of ≤ 12 frames. `--every` samples a fixed interval
and writes more pages (`sheet.jpg`, `sheet-2.jpg`, …) when that is more than 12.
"""
from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from montaj.encode import FFMPEG, MediaInfo, probe, run

if TYPE_CHECKING:
    from montaj.timeline import Timeline

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

# ---- motion sheets: strips, transitions, diffs ------------------------------------------------
#
# A motion sheet is rows of tiles, each row a list of frame indexes. The row builders below are
# pure (timeline in, indexes out). `motion_sheet` extracts through the same `extract` path as the
# contact sheet and draws one image whose long edge never exceeds SHEET_MAX_W.

STRIP_MAX_FRAMES = 12
STRIP_COLS = 8
TRANS_MAX_ROWS = 8
TRANS_STOPS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_TILE_W = 64
DIFF_AMP = 4
DIFF_THRESH = 12
DIFF_SAME = 0.005
BASE_DIR = "sheet-base"


@dataclass(frozen=True)
class Row:
    """One sheet row: a header label, and the (frame index, caption) of each tile."""
    label: str
    tiles: tuple[tuple[int, str], ...]


def step_frames(raw: str, fps: float) -> int:
    """`--step`: `1f` is frames, `0.04s` / `0.04` is seconds; always >= 1 frame."""
    text = str(raw).strip().lower()
    try:
        if text.endswith("f"):
            value = float(text[:-1])
        else:
            value = float(text[:-1] if text.endswith("s") else text) * fps
    except ValueError:
        raise ValueError(f"--step {raw!r}: use frames like 1f or seconds like 0.04s") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"--step {raw!r}: use a positive step like 1f")
    return max(1, int(round(value)))


def strip_indices(center: int, n: int, step: int, n_frames: int) -> list[int]:
    """`n` frames `step` apart, centred on `center`, shifted to stay inside [0, n_frames)."""
    n = max(1, min(int(n), STRIP_MAX_FRAMES))
    step = max(1, int(step))
    span = (n - 1) * step
    first = center - ((n - 1) * step) // 2
    first = max(0, min(first, max(0, n_frames - 1 - span)))
    return sorted({min(first + k * step, n_frames - 1) for k in range(n)})


def _shot_at(tl: Timeline, f: int) -> int:
    hit = [sh.index for sh in tl.shots if sh.start <= f < sh.end]
    return hit[0] if hit else len(tl.shots) - 1


def _trans_at(tl: Timeline, f: int) -> str:
    for w0, w1, tr in tl._windows:
        if w0 <= f < w1:
            return tr.kind
    return "-"


def strip_row(tl: Timeline | None, center: int, n: int, step: int, n_frames: int, fps: float) -> list[Row]:
    """One strip around `center`, wrapped to rows of STRIP_COLS; the header names time, shot, transition."""
    idx = strip_indices(center, n, step, n_frames)
    head = f"strip {stamp(center / fps)} step {step}f"
    if tl is not None:
        head += f"  shot #{_shot_at(tl, center)}  trans {_trans_at(tl, center)}"
    rows = []
    for k in range(0, len(idx), STRIP_COLS):
        chunk = idx[k:k + STRIP_COLS]
        rows.append(Row(head if k == 0 else "", tuple((i, stamp(i / fps)) for i in chunk)))
    return rows


def transition_rows(tl: Timeline, fps: float) -> tuple[list[Row], int]:
    """One row per transition (start, 25%, 50%, 75%, end); returns (rows, total transitions).

    Beyond TRANS_MAX_ROWS the rows are an even subsample, so one image still covers the film.
    """
    rows: list[Row] = []
    for w0, w1, tr in tl._windows:
        lo = max(0, int(round(w0)))
        hi = min(tl.n_frames - 1, max(lo, int(round(w1)) - 1))
        shot = _shot_at(tl, int(round(w0 + (tr.dur / 2 if tr.kind == "whip" else 0.0))) + 1)
        tiles: dict[int, str] = {}
        for stop in TRANS_STOPS:
            f = lo + int(round((hi - lo) * stop))
            tiles.setdefault(f, f"{int(stop * 100)}% {stamp(f / fps)}")
        rows.append(Row(f"#{shot} {tr.kind}  {stamp(lo / fps)}", tuple(sorted(tiles.items()))))
    rows.sort(key=lambda r: r.tiles[0][0])
    total = len(rows)
    if total > TRANS_MAX_ROWS:
        keep = sorted({round(i * (total - 1) / (TRANS_MAX_ROWS - 1)) for i in range(TRANS_MAX_ROWS)})
        rows = [rows[i] for i in keep]
    return rows, total


def _fit(rows: Sequence[Row], ratio: float) -> tuple[int, int, int, int]:
    """(tile_w, tile_h, head_h, cap_h) so the sheet is <= SHEET_MAX_W on both edges."""
    cols = max(len(r.tiles) for r in rows)
    tile_w = SHEET_MAX_W // cols
    while True:
        tile_h = int(tile_w / ratio)
        cap = max(14, tile_w // 12)
        head = max(16, tile_w // 10)
        total = sum((head if r.label else 0) + tile_h + cap for r in rows)
        if total <= SHEET_MAX_W or tile_w <= MIN_TILE_W:
            return tile_w, tile_h, head, cap
        tile_w = max(MIN_TILE_W, int(tile_w * SHEET_MAX_W / total))


def _changed(a: Image.Image, b: Image.Image) -> tuple[Image.Image, float]:
    """Amplified |a-b| (greyscale, so only change survives) and the fraction of pixels that moved."""
    diff = ImageChops.difference(a.convert("L"), b.convert("L"))
    frac = sum(diff.point(lambda v: 255 if v > DIFF_THRESH else 0).histogram()[255:]) / max(1, diff.width * diff.height)
    return diff.point(lambda v: min(255, v * DIFF_AMP)).convert("RGB"), frac


def compose_rows(rows: Sequence[Row], images: dict[int, Image.Image], out: Path | str, *,
                 base: dict[int, Image.Image] | None = None) -> Path:
    """Draw `rows` as one JPEG with a header band per labelled row and a caption under each tile.

    With `base`, each tile shows only what changed against the baseline frame of the same index.
    """
    if not rows:
        raise ValueError("compose_rows: no rows")
    first = next(iter(images.values()))
    tile_w, tile_h, head, cap = _fit(rows, first.width / first.height)
    cols = max(len(r.tiles) for r in rows)
    height = sum((head if r.label else 0) + tile_h + cap for r in rows)
    sheet = Image.new("RGB", (cols * tile_w, height), BG)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=max(11, tile_w // 14))
    y = 0
    for row in rows:
        if row.label:
            draw.text((6, y + 2), row.label, font=font, fill=FG)
            y += head
        for c, (idx, caption) in enumerate(row.tiles):
            x = c * tile_w
            thumb = ImageOps.contain(images[idx].convert("RGB"), (tile_w, tile_h), method=Image.Resampling.LANCZOS)
            if base is not None:
                if idx in base:
                    ref = ImageOps.contain(base[idx].convert("RGB"), thumb.size, method=Image.Resampling.LANCZOS)
                    thumb, frac = _changed(thumb, ref)
                    caption += " same" if frac < DIFF_SAME else f" d{frac * 100:.0f}%"
                else:
                    caption += " new"
            sheet.paste(thumb, (x + (tile_w - thumb.width) // 2, y + (tile_h - thumb.height) // 2))
            draw.text((x + 4, y + tile_h + 2), caption, font=font, fill=FG)
        y += tile_h + cap
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    return out


def _load_base(base_dir: Path, wanted: Sequence[int]) -> dict[int, Image.Image]:
    found = {}
    for i in wanted:
        p = base_dir / f"f{i}.jpg"
        if p.is_file():
            with Image.open(p) as im:
                found[i] = im.convert("RGB")
    return found


def motion_sheet(video: Path | str, out: Path | str, rows: Sequence[Row], *, base_dir: Path | str,
                 diff: bool = False, log: Path | str | None = None) -> Path:
    """Extract every row's frames, then draw them. A plain run also stores them as the diff baseline."""
    info = probe(video, log=log)
    base_dir = Path(base_dir)
    wanted = sorted({i for r in rows for i, _ in r.tiles})
    tmp = Path(tempfile.mkdtemp(prefix="montaj-motion-"))
    try:
        frames = extract(video, [i / info.fps for i in wanted], tmp / "frames", info=info, log=log)
        images = {i: Image.open(p) for i, p in frames}
        try:
            for r in rows:
                missing = [i for i, _ in r.tiles if i not in images]
                if missing:
                    raise RuntimeError(f"sheet: frames {missing} not extracted from {video}")
            base = None
            if diff:
                base = _load_base(base_dir, wanted)
                if not base:
                    raise FileNotFoundError(
                        "sheet --diff: no baseline (run `sheet --transitions` or `--strip` on the previous render first)")
            path = compose_rows(rows, images, out, base=base)
            if not diff:
                base_dir.mkdir(parents=True, exist_ok=True)
                for i, im in images.items():
                    small = ImageOps.contain(im.convert("RGB"), (SHEET_MAX_W // 4, SHEET_MAX_W // 4))
                    small.save(base_dir / f"f{i}.jpg", quality=85)
            return path
        finally:
            for im in images.values():
                im.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
