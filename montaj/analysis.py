"""Semantic asset analysis: `assets/analysis.json`, keyed by file name, cached by file bytes.

`montaj ingest` fills it (`analyze_assets`); the spec reads it to resolve `focus: face`,
`focus: subject`, `focus: face:N` and `range: speech:N` into numbers (`AnalysisBook`).

One row per photo or clip in the assets folder:
- every row: `kind` (photo|clip), `sha` (sha256 of the file), `size`, `mtime_ns`, `v`, `w`, `h`.
- photo: `faces: [{box, eyes, score}]` largest first, `subject: [x, y]`, `sharpness`, `brightness`,
  `colors` (hex, most-covered first).
- clip: `duration`, `fps`, `cuts: [t]`, `speech` / `silence: [{s, e}]`, `motion` (per second),
  `scenes: [{s, e, faces, subject}]` (one mid-scene frame each).
- `words` is reserved for transcripts (`montaj transcribe`); a re-analysis of the same bytes keeps it.

A row is reused while the file's sha and `ANALYSIS_VERSION` match, so re-ingest of unchanged files
runs no model. Torch loads only when a row needs work (`vision.py`).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

ANALYSIS_VERSION = 1
ANALYSIS_FILE = "analysis.json"
PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".webp")
CLIP_EXTS = (".mp4", ".mov", ".webm", ".mkv")
_MATTES_DIR = "mattes"  # matte.MATTES_DIR (that module imports torch)
KEPT_KEYS = ("words",)  # written by other commands; survive a re-analysis of the same bytes


class Analyzer(Protocol):
    def photo(self, path: Path) -> dict: ...

    def clip(self, path: Path) -> dict: ...


def analysis_path(assets_dir: Path | str) -> Path:
    return Path(assets_dir) / ANALYSIS_FILE


def load_analysis(assets_dir: Path | str) -> dict[str, dict]:
    """Rows by file name; empty when the file is missing. A corrupt file raises ValueError."""
    path = analysis_path(assets_dir)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    return data


def media(assets_dir: Path | str) -> list[tuple[Path, str]]:
    """(file, kind) for every photo and clip in the assets folder, sorted by name.

    Flat files only: subfolders are never analysed. A subfolder of images is a `frames:` layer source,
    listed by `frame_folders` instead (#59)."""
    assets_dir = Path(assets_dir)
    if not assets_dir.is_dir():
        return []
    out = []
    for p in sorted(assets_dir.iterdir(), key=lambda p: p.name):
        if not p.is_file() or p.name.startswith("."):
            continue
        ext = p.suffix.lower()
        if ext in PHOTO_EXTS:
            out.append((p, "photo"))
        elif ext in CLIP_EXTS:
            out.append((p, "clip"))
    return out


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fresh(row: dict | None, path: Path) -> bool:
    """True when `row` describes `path`'s current bytes. A stat match skips hashing."""
    if not row or row.get("v") != ANALYSIS_VERSION:
        return False
    st = path.stat()
    if row.get("size") == st.st_size and row.get("mtime_ns") == st.st_mtime_ns:
        return True
    return row.get("size") == st.st_size and row.get("sha") == _sha(path)


@dataclass(frozen=True)
class AnalyzeResult:
    analyzed: int
    cached: int
    rows: dict[str, dict]


def analyze_assets(
    assets_dir: Path | str,
    make_analyzer: Callable[[], Analyzer],
    *,
    log: Path | str | None = None,
) -> AnalyzeResult:
    """Analyse every photo and clip in `assets_dir` whose bytes changed; rewrite analysis.json.

    `make_analyzer` is called at most once, and only when a row needs work, inside the GPU lock.
    Rows for files that left the folder are dropped.
    """
    from montaj.gpulock import hold

    assets_dir = Path(assets_dir)
    old = load_analysis(assets_dir)
    rows: dict[str, dict] = {}
    todo: list[tuple[Path, str]] = []
    for path, kind in media(assets_dir):
        prev = old.get(path.name)
        if fresh(prev, path):
            st = path.stat()  # a byte-identical rewrite refreshes the stat shortcut
            rows[path.name] = {**prev, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        else:
            todo.append((path, kind))
    if todo:
        with hold(log):
            analyzer = make_analyzer()
            for path, kind in todo:
                st = path.stat()
                sha = _sha(path)
                found = analyzer.photo(path) if kind == "photo" else analyzer.clip(path)
                row = {"kind": kind, "sha": sha, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                       "v": ANALYSIS_VERSION, **found}
                prev = old.get(path.name) or {}
                if prev.get("sha") == sha:
                    row.update({k: prev[k] for k in KEPT_KEYS if k in prev})
                rows[path.name] = row
            _note_peaks(analyzer, log)
    analysis_path(assets_dir).write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    return AnalyzeResult(len(todo), len(rows) - len(todo), rows)


def _note_peaks(analyzer: Analyzer, log: Path | str | None) -> None:
    peak = getattr(analyzer, "peak", None)
    if not peak or log is None:
        return
    path = Path(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("peak vram " + " ".join(f"{k}={v / 2**20:.1f}MiB" for k, v in sorted(peak.items())) + "\n")


def _pt(p: list | None) -> str:
    return "-" if not p else f"[{p[0]:g}, {p[1]:g}]"


def describe(name: str, row: dict | None) -> str:
    """One line per asset for `montaj ingest --list`."""
    if not row:
        return f"{name} not analysed (run montaj ingest)"
    size = f"{row.get('w', 0)}x{row.get('h', 0)}"
    if row.get("kind") == "clip":
        scenes = row.get("scenes", [])
        with_face = sum(1 for s in scenes if s.get("faces"))
        talk = sum(s["e"] - s["s"] for s in row.get("speech", []))
        motion = row.get("motion") or [0.0]
        words = f" words={len(row['words'])}" if row.get("words") else ""
        return (f"{name} clip {size} {row.get('duration', 0):g}s cuts={len(row.get('cuts', []))} "
                f"faces={with_face}/{len(scenes)}scenes speech={len(row.get('speech', []))}({talk:.1f}s) "
                f"motion={sum(motion) / len(motion):.3f}{words}")
    faces = row.get("faces", [])
    centres = " ".join(_pt(face_point(f)) for f in faces[:3])
    shown = f"faces={len(faces)}" + (f" {centres}" if faces else "")
    colors = ",".join(row.get("colors", [])[:3])
    return (f"{name} photo {size} {shown} subject={_pt(row.get('subject'))} "
            f"sharp={row.get('sharpness', 0):g} bright={row.get('brightness', 0):g} colors={colors}")


def frame_folders(assets_dir: Path | str) -> list[tuple[Path, list[Path]]]:
    """(folder, images in order) for each subfolder of the assets folder that holds images: a `frames:`
    layer source (#59). Hidden folders and the matte store (`mattes/`) are skipped."""
    from montaj.project import find_frames

    assets_dir = Path(assets_dir)
    if not assets_dir.is_dir():
        return []
    out = []
    for d in sorted(assets_dir.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or d.name.startswith(".") or d.name == _MATTES_DIR:
            continue
        try:
            out.append((d, find_frames(assets_dir, d.name)))
        except FileNotFoundError:
            continue
    return out


def describe_frames(name: str, files: list[Path]) -> str:
    """One `montaj ingest --list` line for a frames folder: count, first image's size, alpha or opaque."""
    from PIL import Image

    try:
        with Image.open(files[0]) as im:
            w, h = im.size
            alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
    except OSError:
        return f"{name} frames {len(files)} unreadable"
    return f"{name} frames {len(files)} {w}x{h} {'alpha' if alpha else 'opaque'}"


def describe_all(assets_dir: Path | str) -> list[str]:
    """`montaj ingest --list`: one line per analysed photo or clip, and one per frames folder, by name."""
    rows = load_analysis(assets_dir)
    out = []
    for path, _ in media(assets_dir):
        row = rows.get(path.name)
        out.append((path.name, describe(path.name, row if fresh(row, path) else None)))
    out.extend((d.name, describe_frames(d.name, files)) for d, files in frame_folders(assets_dir))
    return [line for _, line in sorted(out, key=lambda x: x[0])]


def face_point(face: dict) -> list[float]:
    """Where `focus: face` aims: the face box centre."""
    x0, y0, x1, y1 = face["box"]
    return [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)]


class AnalysisBook:
    """One read of analysis.json per validate; answers lookups by asset file. Not process-global."""

    def __init__(self, assets_dir: Path | None) -> None:
        self.assets_dir = assets_dir
        self._rows: dict[str, dict] | None = None
        self.problem: str | None = None

    def row(self, path: Path) -> tuple[dict | None, str]:
        """(row, reason): reason is the error text when the row is missing or stale."""
        if self._rows is None:
            try:
                self._rows = load_analysis(self.assets_dir) if self.assets_dir is not None else {}
            except (OSError, ValueError) as exc:
                self._rows = {}
                self.problem = f"unreadable {ANALYSIS_FILE} ({exc})"
        row = self._rows.get(path.name)
        if row is None:
            return None, self.problem or f"no analysis for {path.name}"
        if not fresh(row, path):
            return None, f"analysis of {path.name} is stale"
        return row, ""
