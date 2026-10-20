"""Photo ingest: EXIF orientation, size cap, stems, assets.json, contact sheet.

Source folders are messy (rotated phone shots, 12 MP originals). `ingest` normalises
them once into `<assets>/<stem>.jpg` so every later stage can assume RGB, longest
side ≤ 2048 and a stable stem = file name.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps

from montaj.sheet import SHEET_MAX_W, compose_grid

SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
MAX_SIDE = 2048
JPEG_QUALITY = 92
SHEET_COLS = 4


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class AssetInfo:
    """One normalised photo. `w`/`h` are post-EXIF, post-downscale pixels."""

    stem: str
    w: int
    h: int
    sha: str


def assets_json_path(assets_dir: Path | str) -> Path:
    return Path(assets_dir) / "assets.json"


def _sha(path: Path) -> str:
    """sha256 of the normalised file, so a re-render can tell whether the asset changed."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_assets(assets_dir: Path | str) -> list[AssetInfo]:
    """Read assets.json (empty list when missing)."""
    path = assets_json_path(assets_dir)
    if not path.exists():
        return []
    return [AssetInfo(**row) for row in json.loads(path.read_text())]


def contact_sheet(
    items: list[tuple[str, Image.Image]],
    out: Path | str,
    *,
    max_w: int = SHEET_MAX_W,
    cols: int = SHEET_COLS,
) -> Path:
    """One numbered grid of thumbnails, each labelled `<index> <stem>`, ≤ `max_w` wide."""
    if not items:
        raise IngestError("contact sheet: no images")
    tiles = [image for _, image in items]
    labels = [f"{i + 1}  {stem}" for i, (stem, _) in enumerate(items)]
    return compose_grid(tiles, labels, out, max_w=max_w, cols=cols)


def sources(src_dir: Path | str) -> list[Path]:
    """Image files in `src_dir`, sorted by stem (case-insensitive extension match)."""
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise IngestError(f"ingest: {src_dir} is not a directory")
    files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES and not p.name.startswith(".")]
    return sorted(files, key=lambda p: p.stem)


def ingest(
    src_dir: Path | str,
    assets_dir: Path | str,
    *,
    max_side: int = MAX_SIDE,
    quality: int = JPEG_QUALITY,
    sheet_path: Path | str | None = None,
    write_sheet: bool = True,
) -> list[AssetInfo]:
    """Normalise every image in `src_dir` into `assets_dir`.

    Writes `<assets>/<stem>.jpg` (q 92) and `assets.json` (`stem, w, h, sha`, sorted by
    stem), plus one contact sheet at `sheet_path` (default `build/ingest-sheet.jpg`).
    """
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    files = sources(src_dir)
    if not files:
        raise IngestError(f"ingest: no images in {src_dir} (want {', '.join(SUFFIXES)})")
    stems = [f.stem for f in files]
    if len(set(stems)) != len(stems):  # stems name the asset, so a collision is fatal, not a merge
        dupes = sorted({s for s in stems if stems.count(s) > 1})
        raise IngestError(f"ingest: duplicate stems {dupes} (stem = file name without extension)")

    infos: list[AssetInfo] = []
    thumbs: list[tuple[str, Image.Image]] = []
    for src in files:
        with Image.open(src) as raw:
            image = ImageOps.exif_transpose(raw) or raw
            image = image.convert("RGB")
        if max(image.size) > max_side:  # never upscale: small sources pass through
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        out = assets_dir / f"{src.stem}.jpg"
        image.save(out, format="JPEG", quality=quality, optimize=True)
        infos.append(AssetInfo(stem=src.stem, w=image.width, h=image.height, sha=_sha(out)))
        thumbs.append((src.stem, image))

    assets_json_path(assets_dir).write_text(json.dumps([asdict(i) for i in infos], indent=2) + "\n")
    if write_sheet:
        contact_sheet(thumbs, sheet_path or Path("build") / "ingest-sheet.jpg")
    return infos

@dataclass(frozen=True)
class RecInfo:
    """One ingested screen recording. `t0` is the container start_time the events were shifted by."""

    stem: str
    path: Path
    events_path: Path | None
    duration: float
    fps: float
    audio_streams: int
    n_events: int
    t0: float


def _probe_rec(src: Path) -> tuple[float, int]:
    """(container start_time seconds, audio stream count) from ffprobe.

    Input `-ss` is relative to the container start, so that is the clock events use.
    """
    from montaj.encode import FFPROBE, capture

    raw = capture(
        [FFPROBE, "-v", "error", "-print_format", "json", "-show_entries",
         "stream=codec_type:format=start_time", str(src)],
        desc="ffprobe",
    )
    data = json.loads(raw or "{}")
    streams = data.get("streams", [])
    try:
        t0 = float(data.get("format", {}).get("start_time") or 0.0)
    except ValueError:
        t0 = 0.0
    return t0, sum(1 for s in streams if s.get("codec_type") == "audio")


def _same_bytes(a: Path, b: Path) -> bool:
    return a.stat().st_size == b.stat().st_size and _sha(a) == _sha(b)


def ingest_rec(
    video: Path | str, assets_dir: Path | str, *, events: Path | str | None = None, log: Path | str | None = None
) -> RecInfo:
    """Copy a recording into `assets_dir` and write `<stem>.events.json` normalised to video time.

    The input log is seconds from record start; the output `t` is seconds from the file's
    container start (`events.normalise_events`), the clock `clip_in` seeks against.
    """
    from montaj.encode import EncodeError, probe
    from montaj.events import dump_events, load_events_file, normalise_events

    video = Path(video)
    if not video.is_file():
        raise IngestError(f"ingest-rec: {video} is not a file")
    if video.suffix.lower() not in (".mp4", ".mov", ".webm", ".mkv"):
        raise IngestError(f"ingest-rec: {video.name} is not mp4, mov, webm or mkv")
    try:
        info = probe(video, log=log)
        t0, n_audio = _probe_rec(video)
    except EncodeError as exc:
        raise IngestError(f"ingest-rec: cannot probe {video.name}: {exc}") from exc
    rows: list[dict] = []
    if events is not None:
        errors: list[str] = []
        loaded = load_events_file(Path(events), Path(events).name, errors)
        if loaded is None:
            raise IngestError("ingest-rec: " + "; ".join(errors))
        rows = normalise_events(loaded, t0)
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    dest = assets_dir / f"{video.stem}{video.suffix.lower()}"
    events_path = None
    if dest.resolve() != video.resolve():
        # A different recording under the same stem must not inherit the old one's events.
        if dest.exists() and events is None and not _same_bytes(video, dest):
            (assets_dir / f"{video.stem}.events.json").unlink(missing_ok=True)
        shutil.copyfile(video, dest)
    if events is not None:
        events_path = assets_dir / f"{video.stem}.events.json"
        events_path.write_text(dump_events(rows))
    return RecInfo(video.stem, dest, events_path, info.duration, info.fps, n_audio, len(rows), t0)
