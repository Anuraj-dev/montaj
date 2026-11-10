"""Fonts, shaping and signed-distance glyph cells for the GPU text engine. CPU only, no pixels drawn.

Shaping is HarfBuzz (`uharfbuzz`), so Devanagari conjuncts and matras come out as the font intends.
A line's layout is a `GlyphRun`: glyph ids and design-px positions plus the font identities. It is
plain frozen data, built at resolve time and hashed with the segment, so the renderer never shapes.

A glyph cell is a signed distance field computed from the font's own outline (exact segment
distances, nonzero winding), cached on disk by font bytes + glyph id + field parameters. Distances are
in cell px (`SDF_EM` px per em), negative inside. No module state: a `FontBook` holds loaded faces and
lives on one resolve or one Renderer.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
DESIGN_W = 1080.0
EDGE_INSET = 72.0  # design px; `align: left|right` inset, as text-ext.css `.al-left/.al-right`

# Distance-field parameters. Changing any of them changes `atlas_key`, so old cells are never reused.
# 128 px/em keeps light-serif hairlines (~2 cells wide at 64) and serif corners within 0.07 device px of
# the outline up to 260 px type; at 64 the serif style lost 2-4% of its area (test_glyph_coverage_*).
SDF_EM = 128  # cell px per em
SPREAD = 48  # cell px of distance kept each side of the outline (0.375 em): stroke and soft edges live here
SDF_REV = "1"

# Which engine draws each style when a line does not set `engine`. A style moves to "gpu" only after a
# raw-frame parity test against its Chromium raster (issue #20 step 4).
STYLE_ENGINE = MappingProxyType({s: "chromium" for s in ("serif", "script", "caps", "deva", "sans", "mono")})
ENGINES = ("chromium", "gpu")

# style -> ((weight, italic, files)). Within one face, the first file whose cmap has a character shapes
# it: the same split as text.css `unicode-range` (Tiro ships Devanagari and Latin as separate files).
_CORM = (
    (300, False, ("cormorant-garamond-latin-300-normal.woff2",)),
    (600, False, ("cormorant-garamond-latin-600-normal.woff2",)),
    (300, True, ("cormorant-garamond-latin-300-italic.woff2",)),
    (500, True, ("cormorant-garamond-latin-500-italic.woff2",)),
)
FACES = MappingProxyType({
    "serif": _CORM,
    "caps": _CORM,
    "script": ((400, False, ("great-vibes-latin-400-normal.woff2",)),),
    "deva": ((400, False, ("tiro-devanagari-hindi-devanagari-400-normal.woff2",
                           "tiro-devanagari-hindi-latin-400-normal.woff2")),),
    "sans": tuple((w, False, (f"inter-latin-{w}-normal.woff2",)) for w in (400, 600, 700)),
    "mono": tuple((w, False, (f"jetbrains-mono-latin-{w}-normal.woff2",)) for w in (400, 700)),
})


@dataclass(frozen=True)
class StyleMetrics:
    """text.css defaults per style, in the units the spec uses (design px, em)."""

    size: float
    weight: int
    tracking: float  # em
    line_height: float | None  # multiple of size; None is the font's `normal`
    upper: bool = False


STYLE_METRICS = MappingProxyType({
    "serif": StyleMetrics(84.0, 300, 0.01, 1.1),
    "script": StyleMetrics(84.0, 400, 0.0, None),
    "caps": StyleMetrics(26.0, 600, 0.42, None, upper=True),
    "deva": StyleMetrics(62.0, 400, 0.0, 1.5),
    "sans": StyleMetrics(48.0, 400, 0.0, None),
    "mono": StyleMetrics(32.0, 400, 0.0, None),
})


@dataclass(frozen=True)
class FontRef:
    """One font file as the layout saw it. `sha` is of the file bytes; `atlas` keys its glyph cells."""

    name: str
    sha: str
    atlas: str


@dataclass(frozen=True)
class PlacedGlyph:
    """One shaped glyph. Positions in design px on the 1080-wide stage; y grows down.

    `x, y` is the pen origin on the baseline. `ink` is the outline's box (x0, y0, x1, y1); a space has
    an empty box at its origin. `cluster` is the code-point index the glyph came from; `unit` is the
    reveal unit (word or code point) it belongs to.
    """

    font: int
    gid: int
    x: float
    y: float
    advance: float
    ink: tuple[float, float, float, float]
    cluster: int
    unit: int


@dataclass(frozen=True)
class GlyphRun:
    """A shaped line. `box` is the CSS line box (x, y, w, h) in design px; `units` counts reveal units."""

    text: str
    size: float
    fonts: tuple[FontRef, ...]
    glyphs: tuple[PlacedGlyph, ...]
    box: tuple[float, float, float, float]
    baseline: float
    units: int


def atlas_key(sha: str) -> str:
    return f"{sha[:16]}-sdf{SDF_EM}s{SPREAD}r{SDF_REV}"


def _r(v: float) -> float:
    # Positions reach the segment hash; round off float noise that would not move a pixel.
    return round(float(v), 4)


class Face:
    """One decompressed font file: HarfBuzz font for shaping, fontTools font for outlines."""

    def __init__(self, path: Path) -> None:
        import uharfbuzz as hb
        from fontTools.ttLib import TTFont

        data = Path(path).read_bytes()
        self.name = Path(path).name
        self.sha = hashlib.sha256(data).hexdigest()
        self.ref = FontRef(self.name, self.sha, atlas_key(self.sha))
        tt = TTFont(io.BytesIO(data))
        tt.flavor = None  # woff2 -> plain sfnt bytes for HarfBuzz
        buf = io.BytesIO()
        tt.save(buf)
        self.tt = TTFont(io.BytesIO(buf.getvalue()))
        self.upem = int(self.tt["head"].unitsPerEm)
        hhea = self.tt["hhea"]
        self.ascent = float(hhea.ascent)
        self.descent = float(-hhea.descent)
        self.line_gap = float(hhea.lineGap)
        self.cmap = frozenset(self.tt.getBestCmap() or {})
        self.font = hb.Font(hb.Face(hb.Blob(buf.getvalue())))
        self._order = self.tt.getGlyphOrder()
        self._glyphset = self.tt.getGlyphSet()

    def contours(self, gid: int) -> list[np.ndarray]:
        """Closed polylines (n, 2) in font units, y up; curves flattened to under 0.05 cell px."""
        pen = _FlattenPen(self._glyphset, self.upem)
        self._glyphset[self._order[gid]].draw(pen)
        pen.closePath()
        return [np.asarray(c, dtype=np.float64) for c in pen.contours if len(c) >= 3]


class FontBook:
    """Faces by file name, loaded on first use. One per resolve or Renderer."""

    def __init__(self, root: Path | str = FONT_DIR) -> None:
        self.root = Path(root)
        self._faces: dict[str, Face] = {}

    def face(self, name: str) -> Face:
        if name not in self._faces:
            self._faces[name] = Face(self.root / name)
        return self._faces[name]

    def files(self, style: str, weight: float | None, italic: bool) -> tuple[str, ...]:
        """Font files for a style: the nearest shipped weight in the requested slant."""
        faces = FACES.get(style)
        if faces is None:
            raise ValueError(f"no gpu fonts for style {style!r}")
        want = float(weight if weight is not None else STYLE_METRICS[style].weight)
        slanted = [f for f in faces if f[1] == bool(italic)] or list(faces)
        return min(slanted, key=lambda f: (abs(f[0] - want), -f[0]))[2]

    def missing(self, style: str, weight: float | None, italic: bool, text: str) -> list[str]:
        """Characters no file of the style covers (ignoring ZWJ/ZWNJ, which only steer shaping)."""
        faces = [self.face(n) for n in self.files(style, weight, italic)]
        out: list[str] = []
        for ch in _cased(style, text):
            if ch in "‌‍" or ch in out:
                continue
            if not any(ord(ch) in f.cmap for f in faces):
                out.append(ch)
        return out


class _FlattenPen:
    """fontTools pen protocol into polylines. Quadratic and cubic pieces become short chords."""

    def __init__(self, glyphset, upem: int) -> None:
        from fontTools.pens.basePen import BasePen

        outer = self
        self.contours: list[list[tuple[float, float]]] = []
        self._cur: list[tuple[float, float]] = []
        self._px = SDF_EM / float(upem)  # cell px per font unit, for the flattening step

        class _Pen(BasePen):
            def _moveTo(self, p):
                outer._close()
                outer._cur = [p]

            def _lineTo(self, p):
                outer._cur.append(p)

            def _qCurveToOne(self, p1, p2):
                p0 = self._getCurrentPoint()
                outer._curve((p0, p1, p2))

            def _curveToOne(self, p1, p2, p3):
                p0 = self._getCurrentPoint()
                outer._curve((p0, p1, p2, p3))

            def _closePath(self):
                outer._close()

            def _endPath(self):
                outer._close()

        self._pen = _Pen(glyphset)

    def _curve(self, pts) -> None:
        ctrl = np.asarray(pts, dtype=np.float64)
        length = float(np.sum(np.hypot(*np.diff(ctrl, axis=0).T))) * self._px
        n = int(min(64, max(2, math.ceil(length / 2.0))))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        if len(pts) == 3:
            p0, p1, p2 = ctrl
            seg = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
        else:
            p0, p1, p2, p3 = ctrl
            seg = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
        self._cur.extend(map(tuple, seg))

    def _close(self) -> None:
        if len(self._cur) >= 2:
            self.contours.append(self._cur)
        self._cur = []

    # Pen protocol, forwarded to BasePen (which expands implied on-curve points).
    def moveTo(self, p):
        self._pen.moveTo(p)

    def lineTo(self, p):
        self._pen.lineTo(p)

    def qCurveTo(self, *pts):
        self._pen.qCurveTo(*pts)

    def curveTo(self, *pts):
        self._pen.curveTo(*pts)

    def closePath(self):
        self._pen.closePath()

    def endPath(self):
        self._pen.endPath()

    def addComponent(self, name, transform):
        self._pen.addComponent(name, transform)


def _cased(style: str, text: str) -> str:
    return text.upper() if STYLE_METRICS[style].upper else text


def _units(text: str, by: str) -> tuple[list[int], int]:
    """Reveal unit per code point. Words keep their trailing space, as text.py `_split`."""
    if by == "char":
        return list(range(len(text))), len(text)
    out, word = [], 0
    for ch in text:
        out.append(word)
        if ch == " ":
            word += 1
    return out, word + 1


def _font_runs(faces: list[Face], text: str) -> list[tuple[int, int, int]]:
    """(face index, start, end) code-point runs. A char stays in the current face when it covers it."""
    runs: list[tuple[int, int, int]] = []
    cur = 0
    for i, ch in enumerate(text):
        cp = ord(ch)
        if ch in "‌‍" or cp in faces[cur].cmap:
            pick = cur
        else:
            pick = next((j for j, f in enumerate(faces) if cp in f.cmap), cur)
        if runs and runs[-1][0] == pick:
            runs[-1] = (pick, runs[-1][1], i + 1)
        else:
            runs.append((pick, i, i + 1))
        cur = pick
    return runs


def layout_line(
    book: FontBook,
    text: str,
    *,
    style: str,
    y: float,
    size: float | None = None,
    weight: float | None = None,
    italic: bool = False,
    tracking: float | None = None,
    line_height: float | None = None,
    align: str = "center",
    indent: float = 0.0,
    by: str = "word",
) -> GlyphRun:
    """Shape one line and place it like text.css places a `.txt` line at `top: y`.

    The line box is `line_height × size` tall (the font's ascent + descent + gap when unset) with the
    baseline at half-leading + ascent. Letter-spacing follows every cluster, the last one included,
    and turns optional ligatures off, as Chromium does. Left-to-right scripts only.
    """
    import uharfbuzz as hb

    m = STYLE_METRICS[style]
    size = float(size if size is not None else m.size)
    track = float(tracking if tracking is not None else m.tracking) * size
    text = _cased(style, text)
    faces = [book.face(n) for n in book.files(style, weight, italic)]
    missing = book.missing(style, weight, italic, text)
    if missing:
        raise ValueError(f"{style} fonts have no glyph for {''.join(missing)!r}")
    units, n_units = _units(text, by)
    codepoints = [ord(c) for c in text]
    features = {"liga": False, "clig": False} if track else {}

    shaped: list[tuple[int, int, int, float, float, float]] = []  # face, gid, cluster, adv, dx, dy (px)
    for fi, start, end in _font_runs(faces, text):
        face = faces[fi]
        s = size / face.upem
        buf = hb.Buffer()
        buf.add_codepoints(codepoints, start, end - start)
        buf.guess_segment_properties()
        hb.shape(face.font, buf, features)
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            shaped.append((fi, info.codepoint, info.cluster, pos.x_advance * s, pos.x_offset * s, -pos.y_offset * s))

    head = faces[0]
    hs = size / head.upem
    asc, desc = head.ascent * hs, head.descent * hs
    lh = line_height if line_height is not None else m.line_height
    box_h = float(lh) * size if lh is not None else (head.ascent + head.descent + head.line_gap) * hs
    baseline = y + (box_h - (asc + desc)) / 2.0 + asc

    pens: list[float] = []
    pen = 0.0
    for i, (_fi, _gid, cluster, adv, _dx, _dy) in enumerate(shaped):
        pens.append(pen)
        pen += adv
        last_of_cluster = i + 1 == len(shaped) or shaped[i + 1][2] != cluster
        if last_of_cluster:
            pen += track
    width = pen
    shift = float(indent) * size
    if align == "left":
        x0 = EDGE_INSET + shift
    elif align == "right":
        x0 = DESIGN_W - EDGE_INSET - width + shift
    else:
        x0 = (DESIGN_W - width) / 2.0 + shift

    glyphs: list[PlacedGlyph] = []
    for (fi, gid, cluster, adv, dx, dy), p in zip(shaped, pens):
        face = faces[fi]
        s = size / face.upem
        gx, gy = x0 + p + dx, baseline + dy
        ext = face.font.get_glyph_extents(gid)
        if ext is None or ext.width == 0 or ext.height == 0:
            ink = (gx, gy, gx, gy)
        else:
            ix0 = gx + ext.x_bearing * s
            iy0 = gy - ext.y_bearing * s
            ink = (ix0, iy0, ix0 + ext.width * s, iy0 - ext.height * s)
        glyphs.append(PlacedGlyph(
            font=fi, gid=int(gid), x=_r(gx), y=_r(gy), advance=_r(adv),
            ink=tuple(_r(v) for v in ink), cluster=int(cluster), unit=units[cluster] if cluster < len(units) else 0,
        ))
    return GlyphRun(
        text=text,
        size=_r(size),
        fonts=tuple(f.ref for f in faces),
        glyphs=tuple(glyphs),
        box=(_r(x0), _r(y), _r(width), _r(box_h)),
        baseline=_r(baseline),
        units=n_units,
    )


# --- distance-field cells ------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One glyph's field. `sdf` is (h, w) float32 cell px, negative inside, clamped to ±SPREAD.

    `ox, oy` is the cell's top-left in cell px from the glyph origin (y down).
    """

    sdf: np.ndarray
    ox: int
    oy: int


def build_cell(contours: list[np.ndarray], upem: int, *, device: str = "cpu") -> Cell | None:
    """Exact signed distance to the flattened outline, nonzero winding for inside. None for no ink.

    torch does the pixel × segment work (on `device`, chunked to bound memory); the result is
    returned on the CPU for the disk cache.
    """
    import torch

    if not contours:
        return None
    k = SDF_EM / float(upem)
    segs = []
    for c in contours:
        p = np.asarray(c, dtype=np.float64) * np.array([k, -k])  # cell px, y down
        q = np.roll(p, -1, axis=0)
        segs.append(np.concatenate([p, q], axis=1))
    seg = np.concatenate(segs, axis=0)
    seg = seg[np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) > 1e-9]
    if len(seg) == 0:
        return None
    pad = SPREAD + 1
    ox = int(math.floor(min(seg[:, 0].min(), seg[:, 2].min()))) - pad
    oy = int(math.floor(min(seg[:, 1].min(), seg[:, 3].min()))) - pad
    w = int(math.ceil(max(seg[:, 0].max(), seg[:, 2].max()))) - ox + pad
    h = int(math.ceil(max(seg[:, 1].max(), seg[:, 3].max()))) - oy + pad

    dev = torch.device(device)
    S = torch.as_tensor(seg, dtype=torch.float64, device=dev)  # float64 on every device: one cell per key
    a, b = S[:, :2], S[:, 2:]
    ab = b - a
    ab2 = (ab * ab).sum(-1)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=dev, dtype=S.dtype) + oy + 0.5,
        torch.arange(w, device=dev, dtype=S.dtype) + ox + 0.5,
        indexing="ij",
    )
    P = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)
    out = torch.empty(P.shape[0], dtype=S.dtype, device=dev)
    dy = b[:, 1] - a[:, 1]
    safe_dy = torch.where(dy.abs() < 1e-12, torch.ones_like(dy), dy)
    step = max(1, 2_000_000 // len(seg))
    for i in range(0, P.shape[0], step):
        p = P[i:i + step]
        ap = p[:, None, :] - a[None]
        t = ((ap * ab[None]).sum(-1) / ab2[None]).clamp(0.0, 1.0)
        d2 = ((ap - t[..., None] * ab[None]) ** 2).sum(-1).min(1).values
        py, px = p[:, 1:2], p[:, 0:1]
        up = (a[None, :, 1] <= py) & (b[None, :, 1] > py)
        down = (b[None, :, 1] <= py) & (a[None, :, 1] > py)
        xint = a[None, :, 0] + (py - a[None, :, 1]) * ab[None, :, 0] / safe_dy[None]
        right = px < xint
        wind = ((up & right).to(torch.int32) - (down & right).to(torch.int32)).sum(1)
        d = d2.sqrt()
        out[i:i + step] = torch.where(wind != 0, -d, d)
    sdf = out.clamp(-SPREAD, SPREAD).reshape(h, w).to(torch.float32).cpu().numpy()
    return Cell(sdf=sdf, ox=ox, oy=oy)


def _cell_path(cache_dir: Path, ref: FontRef, gid: int) -> Path:
    return Path(cache_dir) / ref.atlas / f"{gid}.npz"


def load_cell(cache_dir: Path, ref: FontRef, gid: int) -> Cell | None | bool:
    """Cached cell, None for a cached empty glyph, False on a miss."""
    path = _cell_path(cache_dir, ref, gid)
    if not path.is_file():
        return False
    with np.load(path) as z:
        if int(z["empty"]):
            return None
        return Cell(sdf=z["sdf"].astype(np.float32), ox=int(z["ox"]), oy=int(z["oy"]))


def save_cell(cache_dir: Path, ref: FontRef, gid: int, cell: Cell | None) -> None:
    path = _cell_path(cache_dir, ref, gid)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as fh:
            if cell is None:
                np.savez(fh, empty=np.int8(1))
            else:
                np.savez(fh, empty=np.int8(0), sdf=cell.sdf.astype(np.float16), ox=np.int32(cell.ox), oy=np.int32(cell.oy))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def glyph_cell(book: FontBook, cache_dir: Path, ref: FontRef, gid: int, *, device: str = "cpu") -> Cell | None:
    """Disk hit, or build from the outline and store. The face must still have the bytes `ref` names."""
    hit = load_cell(cache_dir, ref, gid)
    if hit is not False:
        return hit
    face = book.face(ref.name)
    if face.sha != ref.sha:
        raise ValueError(f"font {ref.name} changed since layout (sha {face.sha[:12]} != {ref.sha[:12]})")
    cell = build_cell(face.contours(gid), face.upem, device=device)
    save_cell(cache_dir, ref, gid, cell)
    # Read back what later renders will read: float16 on disk, not the float32 just computed.
    hit = load_cell(cache_dir, ref, gid)
    return None if hit is False else hit
