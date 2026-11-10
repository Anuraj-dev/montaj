"""Caption lints for `montaj check` (#24): reading rate, platform safe zones, contrast, faces.

A caption chunk is the run of consecutive caption blocks (made by `captions`, or authored ones that look
like captions, see `spec.is_caption_block`) that show the same words. Chunk numbers match
`montaj captions --print`. Rate and safe zones come from the resolved layout; contrast and faces are
measured on one frame decoded from the rendered video per chunk. `check.expect: [{caption: [a, b]}]`
silences a designed exception inside a span, as black and frozen spans do.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from montaj.captions import RATE_MAX_CPS, safe_zones
from montaj.render.glyphs import DESIGN_W
from montaj.spec import is_caption_block
from montaj.timeline import ResolvedLine, Timeline

CONTRAST_MIN = 4.5
STROKE_GUARANTEES = 0.04  # a stroke at least this fraction of the size separates the fill from any footage
RING = 0.18  # width of the sampled band around the ink, as a fraction of the size
MAX_SAMPLES = 30  # frames decoded per run; chunks beyond are an even subsample
_NAMED = {"cream": "#fbf1dc", "gold": "#f0c77e", "ink": "#4a2c1a"}


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int  # frames
    end: int
    text: str
    line: ResolvedLine  # the chunk's first line: layout, colour and stroke
    ink: tuple[float, float, float, float] | None  # design px


def _plain(text: str) -> str:
    return text.replace("\\*", "\0").replace("*", "").replace("\0", "*")


def chunks(tl: Timeline) -> list[Chunk]:
    """Caption chunks of the resolved timeline, in order."""
    w, h = tl.size
    design_h = DESIGN_W * h / w
    out: list[Chunk] = []
    last: tuple[str, int] | None = None
    for i, (res, blk) in enumerate(zip(tl.texts, tl.spec.text)):
        if not res.lines or not is_caption_block(tl.spec, i, blk, design_h):
            last = None
            continue
        text = " ".join(_plain(ln.text) for ln in res.lines)
        if last is not None and last[0] == text and last[1] == len(out) - 1:
            prev = out[-1]
            out[-1] = Chunk(prev.index, prev.start, round(res.to), prev.text, prev.line, prev.ink)
            continue
        boxes = [g.ink for ln in res.lines if ln.glyphs is not None for g in ln.glyphs.glyphs if g.ink[2] > g.ink[0]]
        ink = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
               max(b[3] for b in boxes)) if boxes else None
        out.append(Chunk(len(out), round(res.from_), round(res.to), text, res.lines[0], ink))
        last = (text, len(out) - 1)
    return out


def _zones(tl: Timeline) -> dict[str, float]:
    return safe_zones(tl.spec)


def rate_lines(cs: list[Chunk], fps: int) -> list[tuple[Chunk, str]]:
    out = []
    for c in cs:
        secs = (c.end - c.start) / fps
        chars = len(c.text.replace(" ", ""))
        if secs > 0 and chars / secs > RATE_MAX_CPS + 1e-9:
            out.append((c, f"WARN caption-rate chunk={c.index} {chars / secs:.0f}cps"))
    return out


def safe_lines(cs: list[Chunk], tl: Timeline) -> list[tuple[Chunk, str]]:
    w, h = tl.size
    W, H = DESIGN_W, DESIGN_W * h / w
    z = _zones(tl)
    out = []
    for c in cs:
        if c.ink is None:
            continue
        x0, y0, x1, y1 = c.ink
        sides = [name for name, hit in (("top", y0 < z["top"] * H), ("bottom", y1 > H - z["bottom"] * H),
                                        ("left", x0 < z["left"] * W), ("right", x1 > W - z["right"] * W)) if hit]
        if sides:
            out.append((c, f"WARN caption-safe chunk={c.index} {','.join(sides)}"))
    return out


def _rgb(color: str | None) -> tuple[float, float, float]:
    c = _NAMED.get(color or "", color or "#ffffff")
    if not (isinstance(c, str) and len(c) == 7 and c[0] == "#"):
        c = "#ffffff"
    return tuple(int(c[k:k + 2], 16) / 255.0 for k in (1, 3, 5))  # type: ignore[return-value]


def _lum(rgb) -> float:
    def lin(v):
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(float(v)) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a: float, b: float) -> float:
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def contrast(c: Chunk, frame) -> float | None:
    """Worst contrast of the chunk's fill (and accent) against what is behind it; None when unmeasurable.

    `frame` is an HxWx3 uint8 array of the rendered frame. A stroke of at least STROKE_GUARANTEES of the
    size is the backdrop of the glyph edge, so the fill is judged against the stroke colour; otherwise
    against the median luminance of a band around the ink.
    """
    import numpy as np

    ln = c.line
    fills = [_rgb(ln.color)]
    if ln.emphasis is not None and ln.glyphs is not None and ln.glyphs.spans:
        fills.append(_rgb(ln.emphasis.color))
    size = ln.size or 48.0
    if ln.stroke is not None and ln.stroke.width >= STROKE_GUARANTEES * size:
        back = _lum(_rgb(ln.stroke.color))
    else:
        if c.ink is None:
            return None
        k = frame.shape[1] / DESIGN_W
        pad = RING * size
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = c.ink
        ox0, oy0 = max(int((x0 - pad) * k), 0), max(int((y0 - pad) * k), 0)
        ox1, oy1 = min(int((x1 + pad) * k) + 1, W), min(int((y1 + pad) * k) + 1, H)
        if ox1 <= ox0 or oy1 <= oy0:
            return None
        mask = np.ones((oy1 - oy0, ox1 - ox0), dtype=bool)
        ix0, iy0 = max(int(x0 * k) - ox0, 0), max(int(y0 * k) - oy0, 0)
        mask[iy0:max(int(y1 * k) - oy0, iy0), ix0:max(int(x1 * k) - ox0, ix0)] = False
        ring = frame[oy0:oy1, ox0:ox1][mask].astype(np.float64) / 255.0
        if ring.size == 0:
            return None
        lin = np.where(ring <= 0.04045, ring / 12.92, ((ring + 0.055) / 1.055) ** 2.4)
        back = float(np.median(lin @ np.array([0.2126, 0.7152, 0.0722])))
    return min(ratio(_lum(f), back) for f in fills)


def _face_boxes():
    """Callable (image path) -> [(x, y, w, h)] on OpenCV's frontal cascade, or None when cv2 is absent."""
    try:
        import cv2
    except ImportError:
        return None
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))

    def find(image: Path):
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return []
        return [tuple(int(v) for v in r) for r in
                cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(48, 48))]

    return find


def _sample(cs: list[Chunk]) -> list[Chunk]:
    if len(cs) <= MAX_SAMPLES:
        return cs
    return [cs[int(k * len(cs) / MAX_SAMPLES)] for k in range(MAX_SAMPLES)]


def frame_lines(cs: list[Chunk], tl: Timeline, video: Path | str, work: Path, *, log: Path | str | None = None,
                faces=None) -> tuple[list[tuple[Chunk, str]], list[str]]:
    """(findings, INFO lines) for contrast and faces on one decoded frame per sampled chunk.
    `faces` replaces the OpenCV detector (tests): image path -> [(x, y, w, h)] in pixels."""
    import numpy as np
    from PIL import Image

    from montaj.guards import _extract

    notes: list[str] = []
    find = faces if faces is not None else _face_boxes()
    if find is None:
        notes.append("INFO caption-face skipped: uv sync --extra guards")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    out: list[tuple[Chunk, str]] = []
    for c in _sample(cs):
        f = c.start + max(0, min(c.end - c.start - 1, 8))  # past the entry motion
        image = work / f"chunk{c.index}-f{f}.png"
        if not _extract(Path(video), f, tl.fps, image, log):
            continue
        with Image.open(image) as im:
            frame = np.asarray(im.convert("RGB"))
        r = contrast(c, frame)
        if r is not None and r < CONTRAST_MIN:
            out.append((c, f"WARN caption-contrast chunk={c.index} ratio={r:.1f}"))
        if find is not None and c.ink is not None:
            k = frame.shape[1] / DESIGN_W
            x0, y0, x1, y1 = (v * k for v in c.ink)
            if any(fx < x1 and fx + fw > x0 and fy < y1 and fy + fh > y0 for fx, fy, fw, fh in find(image)):
                out.append((c, f"WARN caption-face chunk={c.index}"))
    return out, notes


def _label(a: float, b: float) -> str:
    def one(t: float) -> str:
        return str(int(round(t))) if abs(t - round(t)) < 5e-3 else f"{t:.2f}".rstrip("0")

    return f"{one(a)}-{one(b)}s"


def lines(tl: Timeline, video: Path | str | None = None, work: Path | None = None, *,
          log: Path | str | None = None, faces=None) -> list[str]:
    """WARN lines (and INFO for skipped checks and `check.expect` caption spans); [] when there are no captions.
    Without `video`, only the layout checks (rate, safe zones) run."""
    cs = chunks(tl)
    if not cs:
        return []
    found = rate_lines(cs, tl.fps) + safe_lines(cs, tl)
    notes: list[str] = []
    if video is not None and work is not None:
        more, notes = frame_lines(cs, tl, video, work, log=log, faces=faces)
        found += more
    found.sort(key=lambda p: p[0].index)
    expects = [e for e in tl.check_expects if e.kind == "caption"]
    used = [False] * len(expects)
    out: list[str] = []
    for c, line in found:
        mid = (c.start + c.end) / 2 / tl.fps
        hit = next((k for k, e in enumerate(expects) if e.start <= mid < e.end), None)
        if hit is None:
            out.append(line)
        else:
            used[hit] = True
    for k, e in enumerate(expects):
        label = _label(e.start, e.end)
        if used[k]:
            if e.why:
                out.append(f"INFO caption {label} ignored as intended: {e.why}")
        else:
            out.append(f"INFO expect caption {label} not found")
    return out + notes
