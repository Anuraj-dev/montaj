"""Caption review images (issue #26): `montaj captions --gallery` and `--strip`.

Both draw the project's own spec through the normal `Renderer`, so what the agent sees is what a render
produces. Frames are picked from the words, not from a clock, so every tile shows a caption that is up.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from montaj import gpulock
from montaj.captions import PRESETS, Chunk, expand_captions, plan_chunks
from montaj.render.canvas import Canvas
from montaj.render.frame import Renderer
from montaj.sheet import SHEET_MAX_W, Row, compose_rows, stamp
from montaj.spec import Spec, SpecError, _line, load_spec
from montaj.timeline import resolve

SETTLE = 6  # frames after a chunk starts: the entrance has landed and the word is readable
CANDIDATES = 10  # chunks sampled across the film to find its brightest and darkest moment
STRIP_FRAMES = 8
GALLERY_TILE_W = SHEET_MAX_W // len(PRESETS)


def _load(root: Path) -> tuple[Spec, Path]:
    path = root / "montaj.yaml"
    spec = load_spec(path, expand=False)
    if spec.captions is None:
        raise SpecError([_line("captions", "the spec has no captions block", "add captions: {source: ...}")])
    return spec, path


def _chunks(spec: Spec, root: Path) -> list[Chunk]:
    errors: list[str] = []
    chunks = plan_chunks(spec, root, errors)
    if errors:
        raise SpecError(errors)
    return chunks


def _with_preset(spec: Spec, preset: str) -> Spec:
    assert spec.captions is not None
    return spec.model_copy(update={"captions": spec.captions.model_copy(update={"preset": preset})})


def _shot(spec: Spec, root: Path, frames: list[int], width: int | None) -> dict[int, Image.Image]:
    """Render `frames` of `spec` (captions already expanded or absent) as PIL images, ≤ `width` wide."""
    tl = resolve(spec, root)
    W, H = tl.size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict[int, Image.Image] = {}
    with torch.no_grad():
        renderer = Renderer(tl, root, Canvas(W, H, device=device))
        for f in sorted(set(frames)):
            img = renderer.frame(min(max(0, f), tl.n_frames - 1))
            arr = (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray(arr)
            if width and im.width > width:
                im = im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS)
            out[f] = im
        del renderer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _luma(im: Image.Image) -> float:
    return float(sum(im.convert("L").resize((64, 64)).getdata())) / (64 * 64)


def _pick_times(chunks: list[Chunk], n: int = CANDIDATES) -> list[int]:
    """Frames a caption is up, spread over the film: one per evenly spaced chunk."""
    idx = sorted({round(i * (len(chunks) - 1) / max(1, n - 1)) for i in range(n)}) if len(chunks) > 1 else [0]
    return [min(chunks[i].start + SETTLE, chunks[i].end - 1) for i in idx]


def pick_extremes(luma: dict[int, float]) -> tuple[int, int]:
    """(brightest, darkest) frame of `luma`; the same frame only when there is just one."""
    order = sorted(luma, key=lambda f: (luma[f], f))
    return order[-1], order[0]


def _chunk_at(chunks: list[Chunk], f: int) -> Chunk:
    live = [c for c in chunks if c.start <= f < c.end]
    return live[0] if live else min(chunks, key=lambda c: abs(c.start - f))


def gallery(root: Path, out: Path | None = None) -> tuple[Path, int]:
    """One JPEG: every preset over this project's brightest and darkest caption moments. Returns (path, n presets)."""
    spec, _ = _load(root)
    assert spec.captions is not None
    chunks = _chunks(spec, root)
    out = out or root / "build" / "captions-gallery.jpg"
    with gpulock.hold(root / "build" / "captions.log"):
        bare = spec.model_copy(update={"captions": None})
        cand = _pick_times(chunks)
        shots = _shot(bare, root, cand, 256)
        bright, dark = pick_extremes({f: _luma(im) for f, im in shots.items()})
        images: dict[int, Image.Image] = {}
        rows = [Row(f"{name} {stamp(f / spec.video.fps)}", ()) for name, f in (("bright", bright), ("dark", dark))]
        tiles: list[list[tuple[int, str]]] = [[], []]
        for p, preset in enumerate(PRESETS):
            ex = expand_captions(_with_preset(spec, preset), root)
            pc = _chunks(_with_preset(spec, preset), root)
            frames = []
            for r, base in enumerate((bright, dark)):
                c = _chunk_at(pc, base)
                f = min(max(base, c.start + SETTLE), c.end - 1)
                frames.append((r, f))
            got = _shot(ex, root, [f for _, f in frames], GALLERY_TILE_W)
            for r, f in frames:
                key = r * 100 + p
                images[key] = got[f]
                tiles[r].append((key, f"{preset} {spec.captions.accent}"))
        rows = [Row(rows[r].label, tuple(tiles[r])) for r in range(2)]
        compose_rows(rows, images, out)
    return out, len(PRESETS)


def strip(root: Path, chunk: int, out: Path | None = None) -> tuple[Path, int]:
    """One JPEG: consecutive frames of one chunk's entrance, with the project's own preset. Returns (path, n frames)."""
    spec, _ = _load(root)
    chunks = _chunks(spec, root)
    if not 0 <= chunk < len(chunks):
        raise SpecError([_line("captions --strip", f"chunk {chunk} does not exist", f"have 0 to {len(chunks) - 1}; see captions --print")])
    c = chunks[chunk]
    out = out or root / "build" / "captions-strip.jpg"
    fps = spec.video.fps
    frames = [f for f in range(c.start - 1, c.start - 1 + STRIP_FRAMES)]
    with gpulock.hold(root / "build" / "captions.log"):
        got = _shot(expand_captions(spec, root), root, frames, SHEET_MAX_W // STRIP_FRAMES)
    tiles = tuple((f, f"+{f - c.start}f" if f >= c.start else f"{f - c.start}f") for f in sorted(got))
    head = f"chunk {chunk} \"{c.text}\" {spec.captions.preset} start {stamp(c.start / fps)}"  # type: ignore[union-attr]
    compose_rows([Row(head, tiles)], got, out)
    return out, len(got)
