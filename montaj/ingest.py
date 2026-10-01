"""Photo ingest: EXIF orientation, size cap, stems, assets.json, contact sheet.

Source folders are messy (rotated phone shots, 12 MP originals). `ingest` normalises
them once into `<assets>/<stem>.jpg` so every later stage can assume RGB, longest
side ≤ 2048 and a stable stem = file name.
"""
from __future__ import annotations

import hashlib
import json
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