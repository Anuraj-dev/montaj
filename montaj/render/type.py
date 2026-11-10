"""GPU text drawing: shaped `GlyphRun`s from glyphs.py, signed-distance cells, torch compositing.

`draw_run` is the one entry point. It turns a run, a `TextStyle` (fill, stroke, shadows, glow) and an
optional per-glyph `GlyphXform` into premultiplied RGBA on the canvas device. Everything is pure in its
arguments; a `GlyphAtlas` (one per Renderer) only caches cells already on disk or derivable from the
font bytes the run names. No browser, no module state.

Units: run positions, stroke widths, shadow offsets and blur radii are design px (×`Canvas.k`).
Blur radii follow CSS: σ = radius / 2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

import torch
import torch.nn.functional as F

from montaj.render import glyphs as G

_NAMED = MappingProxyType({"cream": "#fbf1dc", "gold": "#f0c77e", "ink": "#4a2c1a", "black": "#000000"})


def rgb(color: str | None, default: str = "cream") -> tuple[float, float, float]:
    """`#rrggbb` or a named colour (cream, gold, ink, black) to 0–1 floats."""
    h = _NAMED.get(color or default, color or _NAMED[default]).lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


@dataclass(frozen=True)
class Shadow:
    """One CSS-like text shadow: offset and blur radius in design px."""

    x: float = 0.0
    y: float = 0.0
    blur: float = 0.0
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    opacity: float = 1.0


@dataclass(frozen=True)
class TextStyle:
    """How a run is painted. `stroke_width` is design px outside the outline (0 = none).

    Shadows paint behind the body in order (first is nearest the body), each from the body's alpha.
    `glow` is the same blur-of-alpha but drawn between the shadows and the body, unshifted.
    """

    fill: tuple[float, float, float] = rgb("cream")
    opacity: float = 1.0
    stroke_width: float = 0.0
    stroke_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    shadows: tuple[Shadow, ...] = ()
    glow: tuple[Shadow, ...] = ()
    emphasis_fill: tuple[float, float, float] | None = None  # glyphs with `emph` > 0; None = `fill`


@dataclass(frozen=True)
class GlyphXform:
    """Per-glyph change from its laid-out pose. Transforms act about the glyph's ink centre.

    `dx, dy` design px; `rotate` degrees clockwise; `blur` is a CSS blur radius applied to the glyph
    (fill and stroke); `fill` overrides the style fill. Glyphs with equal xforms paint as one layer,
    so a whole word sharing one xform blurs as a word.
    """

    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    rotate: float = 0.0
    opacity: float = 1.0
    blur: float = 0.0
    fill: tuple[float, float, float] | None = None
    sy: float = 1.0  # extra vertical scale about the ink centre (a flip), after `scale`


IDENTITY = GlyphXform()

# text.css text-shadow per style (opacity is the rgba alpha). Glow-like 0-offset shadows stay shadows
# here so the paint order matches CSS (first listed is on top).
_BLACK = (0.0, 0.0, 0.0)
CSS_SHADOWS = MappingProxyType({
    "serif": (Shadow(0, 4, 30, _BLACK, 0.65),),
    "script": (Shadow(0, 0, 40, rgb("gold"), 0.45), Shadow(0, 4, 24, _BLACK, 0.6)),
    "caps": (Shadow(0, 2, 16, _BLACK, 0.8),),
    "deva": (Shadow(0, 4, 24, _BLACK, 0.8),),
    "sans": (Shadow(0, 2, 16, _BLACK, 0.55),),
    "mono": (Shadow(0, 2, 16, _BLACK, 0.55),),
})
_CSS_FILL = MappingProxyType({"script": "gold", "caps": "gold"})


def css_style(style: str, color: str | None = None, shadow: bool = True) -> TextStyle:
    """The text.css look of a style as a TextStyle: its colour and its text-shadow list."""
    return TextStyle(
        fill=rgb(color or _CSS_FILL.get(style, "cream")),
        shadows=CSS_SHADOWS.get(style, ()) if shadow else (),
    )


# `glow: g` is one unshifted halo in the fill colour: blur radius GLOW_BLUR × size, opacity g.
GLOW_BLUR = 0.25


def line_style(ln) -> TextStyle:
    """A resolved gpu line's paint (timeline.ResolvedLine): the style's text.css look, then the
    line's own colour, stroke, shadow, glow and emphasis colour on top."""
    base = css_style(ln.style, ln.color, bool(ln.shadow))
    shadows = base.shadows
    sh = ln.shadow
    if not isinstance(sh, bool):  # spec.TextShadow
        shadows = (Shadow(float(sh.x), float(sh.y), float(sh.blur), rgb(sh.color), float(sh.opacity)),)
    stroke = getattr(ln, "stroke", None)
    glow = getattr(ln, "glow", None)
    emph = getattr(ln, "emphasis", None)
    size = ln.glyphs.size if ln.glyphs is not None else 0.0
    return TextStyle(
        fill=base.fill,
        stroke_width=float(stroke.width) if stroke is not None else 0.0,
        stroke_color=rgb(stroke.color) if stroke is not None else (0.0, 0.0, 0.0),
        shadows=shadows,
        glow=(Shadow(0.0, 0.0, GLOW_BLUR * size, base.fill, float(glow)),) if glow else (),
        emphasis_fill=rgb(emph.color) if emph is not None else None,
    )


@dataclass
class GlyphAtlas:
    """Glyph cells on one device, keyed by (atlas key, gid). Disk-backed under `cache_dir`."""

    cache_dir: Path
    device: torch.device | str
    book: G.FontBook = field(default_factory=G.FontBook)
    _cells: dict = field(default_factory=dict, repr=False)

    def cell(self, ref: G.FontRef, gid: int) -> tuple[torch.Tensor, int, int] | None:
        key = (ref.atlas, gid)
        if key not in self._cells:
            dev = torch.device(self.device)
            c = G.glyph_cell(self.book, Path(self.cache_dir), ref, gid, device=str(dev))
            self._cells[key] = None if c is None else (
                torch.as_tensor(c.sdf, device=dev, dtype=torch.float32)[None, None], c.ox, c.oy)
        return self._cells[key]


@dataclass(frozen=True)
class GlyphLine:
    """A gpu-engine line as the renderer holds it: the run, the style it paints with, and the cell cache."""

    run: G.GlyphRun
    style: TextStyle
    atlas: GlyphAtlas = field(compare=False, repr=False)


def _gauss(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian over (C, H, W), zero padded, 3σ support. Same kernel as text.py `_blur`."""
    if sigma < 0.25 or img.numel() == 0:
        return img
    r = math.ceil(3 * sigma)
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    c = img.shape[0]
    h = F.conv2d(F.pad(img[None], (r, r, 0, 0)), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(h, (0, 0, r, r)), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)[0]


def _shift(img: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Translate by a fractional device offset, bilinear, same size, zeros shifted in."""
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return img
    ix, iy = math.floor(dx), math.floor(dy)
    fx, fy = dx - ix, dy - iy
    out = torch.zeros_like(img)
    _c, h, w = img.shape
    for ox, oy, wgt in ((ix, iy, (1 - fx) * (1 - fy)), (ix + 1, iy, fx * (1 - fy)),
                        (ix, iy + 1, (1 - fx) * fy), (ix + 1, iy + 1, fx * fy)):
        if wgt < 1e-6:
            continue
        sx0, sx1 = max(0, -ox), min(w, w - ox)
        sy0, sy1 = max(0, -oy), min(h, h - oy)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        out[:, sy0 + oy:sy1 + oy, sx0 + ox:sx1 + ox] += img[:, sy0:sy1, sx0:sx1] * wgt
    return out


def _over(top: torch.Tensor, below: torch.Tensor) -> torch.Tensor:
    return top + below * (1 - top[3:4])


def _corners(box, cx, cy, s, rot, tx, ty, line=None, sy=1.0):
    """Device-px corners of `box` after the glyph pose, then the line rotation `line` = (px, py, rad)."""
    x0, y0, x1, y1 = box
    c, sn = math.cos(rot), math.sin(rot)
    for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        u, v = (x - cx) * s, (y - cy) * s * sy
        qx, qy = cx + tx + u * c - v * sn, cy + ty + u * sn + v * c
        if line is not None:
            px, py, lr = line
            lc, ls = math.cos(lr), math.sin(lr)
            qx, qy = px + (qx - px) * lc - (qy - py) * ls, py + (qx - px) * ls + (qy - py) * lc
        yield qx, qy


def draw_run_crop(cv, run: G.GlyphRun, style: TextStyle, per_glyph_xform=None, *,
                  atlas: GlyphAtlas) -> tuple[torch.Tensor, int, int]:
    """Paint `run` into the smallest device-px crop that holds it. Returns (premul RGBA, x0, y0).

    `per_glyph_xform` is a sequence parallel to `run.glyphs` (None entries mean identity) or None.
    An empty crop (nothing visible) is (4, 0, 0).
    """
    k = float(cv.k)
    dev, dt = torch.device(cv.device), torch.float32
    xs = list(per_glyph_xform) if per_glyph_xform is not None else [None] * len(run.glyphs)
    if len(xs) != len(run.glyphs):
        raise ValueError(f"{len(xs)} xforms for {len(run.glyphs)} glyphs")
    sw = max(0.0, float(style.stroke_width)) * k
    # The whole line turns about its box centre after every per-glyph pose (so a rise is along the
    # line's own up). None when unrotated: the sampling below then skips the extra mapping.
    line_rot = None
    if abs(run.rotate) > 1e-9:
        pvx, pvy = run.pivot
        line_rot = (pvx * k, pvy * k, math.radians(run.rotate))

    # Visible glyphs with their cells and device-space boxes, grouped by xform.
    groups: dict[GlyphXform, list] = {}
    for g, x in zip(run.glyphs, xs):
        x = x or IDENTITY
        if g.emph and style.emphasis_fill is not None and x.fill is None:
            x = replace(x, fill=style.emphasis_fill)
        if x.opacity <= 1e-4 or x.scale <= 1e-6 or x.sy <= 1e-3:
            continue
        ref = run.fonts[g.font]
        cell = atlas.cell(ref, g.gid)
        if cell is None:
            continue
        sdf, ox, oy = cell
        s = run.size * g.scale * k / G.SDF_EM  # device px per cell px
        gx, gy = g.x * k, g.y * k
        cbox = (gx + ox * s, gy + oy * s, gx + (ox + sdf.shape[-1]) * s, gy + (oy + sdf.shape[-2]) * s)
        ink = g.ink if g.ink[2] > g.ink[0] else (g.x, g.y, g.x, g.y)
        cx, cy = (ink[0] + ink[2]) / 2 * k, (ink[1] + ink[3]) / 2 * k
        groups.setdefault(x, []).append((sdf, s, gx, gy, ox, oy, cbox, cx, cy))
    if not groups:
        return torch.zeros(4, 0, 0, device=dev, dtype=dt), 0, 0

    def grow(box, m):
        return box[0] - m, box[1] - m, box[2] + m, box[3] + m

    # Union box in device px of every group after its transform and blur, then of the shadows.
    gboxes = {}
    for x, items in groups.items():
        rot = math.radians(x.rotate)
        pts = [p for it in items for p in _corners(it[6], it[7], it[8], x.scale, rot, x.dx * k, x.dy * k, line_rot, x.sy)]
        m = sw * x.scale + 3 * (x.blur / 2) * k + 2
        gboxes[x] = grow((min(p[0] for p in pts), min(p[1] for p in pts),
                          max(p[0] for p in pts), max(p[1] for p in pts)), m)
    body = (min(b[0] for b in gboxes.values()), min(b[1] for b in gboxes.values()),
            max(b[2] for b in gboxes.values()), max(b[3] for b in gboxes.values()))
    full = body
    for sh in style.shadows + style.glow:
        m = 3 * (sh.blur / 2) * k + 2
        b = (body[0] + sh.x * k - m, body[1] + sh.y * k - m, body[2] + sh.x * k + m, body[3] + sh.y * k + m)
        full = (min(full[0], b[0]), min(full[1], b[1]), max(full[2], b[2]), max(full[3], b[3]))
    X0, Y0 = max(0, math.floor(full[0])), max(0, math.floor(full[1]))
    X1, Y1 = min(cv.W, math.ceil(full[2])), min(cv.H, math.ceil(full[3]))
    if X1 <= X0 or Y1 <= Y0:
        return torch.zeros(4, 0, 0, device=dev, dtype=dt), 0, 0
    H, W = Y1 - Y0, X1 - X0
    layer = torch.zeros(4, H, W, device=dev, dtype=dt)
    stroke_c = torch.tensor(style.stroke_color, device=dev, dtype=dt).view(3, 1, 1)

    for x, items in groups.items():
        b = gboxes[x]
        bx0, by0 = max(X0, math.floor(b[0])), max(Y0, math.floor(b[1]))
        bx1, by1 = min(X1, math.ceil(b[2])), min(Y1, math.ceil(b[3]))
        if bx1 <= bx0 or by1 <= by0:
            continue
        gh, gw = by1 - by0, bx1 - bx0
        fill_a = torch.zeros(1, gh, gw, device=dev, dtype=dt)
        ring_a = torch.zeros(1, gh, gw, device=dev, dtype=dt)
        rot = math.radians(x.rotate)
        c, sn = math.cos(rot), math.sin(rot)
        py = torch.arange(by0, by1, device=dev, dtype=dt) + 0.5
        px = torch.arange(bx0, bx1, device=dev, dtype=dt) + 0.5
        yy, xx = torch.meshgrid(py, px, indexing="ij")
        if line_rot is not None:  # undo the line rotation first: screen px -> the line's own frame
            lpx, lpy, lr = line_rot
            lc, ls = math.cos(lr), math.sin(lr)
            ex, ey = xx - lpx, yy - lpy
            xx, yy = lpx + ex * lc + ey * ls, lpy - ex * ls + ey * lc
        for sdf, s, gx, gy, ox, oy, _cbox, cx, cy in items:
            # Inverse pose: screen px -> untransformed device px -> cell px.
            u, v = xx - cx - x.dx * k, yy - cy - x.dy * k
            ux = (u * c + v * sn) / x.scale + cx
            uy = (-u * sn + v * c) / (x.scale * x.sy) + cy
            qx = (ux - gx) / s - ox
            qy = (uy - gy) / s - oy
            h, w = sdf.shape[-2:]
            grid = torch.stack((qx / w * 2 - 1, qy / h * 2 - 1), -1)[None]
            d = F.grid_sample(sdf, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
            d = d * (s * x.scale * min(1.0, x.sy))  # device px; a squash sharpens along its short axis
            fa = (0.5 - d).clamp(0, 1)
            fill_a = fa + fill_a * (1 - fa)
            if sw > 0:
                ra = (0.5 - (d - sw * x.scale)).clamp(0, 1)
                ring_a = ra + ring_a * (1 - ra)
        fill_c = torch.tensor(x.fill if x.fill is not None else style.fill, device=dev, dtype=dt).view(3, 1, 1)
        g = torch.cat([fill_c * fill_a, fill_a], 0)
        if sw > 0:
            g = _over(g, torch.cat([stroke_c * ring_a, ring_a], 0))
        if x.blur > 0:
            g = _gauss(g, x.blur / 2 * k)
        op = x.opacity * style.opacity
        if op < 1 - 1e-6:
            g = g * op
        sub = layer[:, by0 - Y0:by1 - Y0, bx0 - X0:bx1 - X0]
        layer[:, by0 - Y0:by1 - Y0, bx0 - X0:bx1 - X0] = _over(g, sub)

    alpha = layer[3:4]
    under = torch.zeros_like(layer)
    for sh in reversed(style.glow + style.shadows):  # last listed is furthest back
        a = _shift(_gauss(alpha, sh.blur / 2 * k), sh.x * k, sh.y * k) * sh.opacity
        col = torch.tensor(sh.color, device=dev, dtype=dt).view(3, 1, 1)
        under = _over(torch.cat([col * a, a], 0), under)
    return _over(layer, under), X0, Y0


def draw_bar(cv, rect: tuple[float, float, float, float], color: str, *, radius: float,
             opacity: float = 1.0) -> tuple[torch.Tensor, int, int]:
    """A rounded bar (premul RGBA crop, x0, y0) over `rect` = (x0, y0, x1, y1) design px (#23 `highlight`).

    Edges are antialiased from the rounded-box distance; `radius` is design px.
    """
    k = float(cv.k)
    x0, y0, x1, y1 = (v * k for v in rect)
    ix0, iy0, ix1, iy1 = math.floor(x0) - 1, math.floor(y0) - 1, math.ceil(x1) + 1, math.ceil(y1) + 1
    w, h = ix1 - ix0, iy1 - iy0
    if w <= 2 or h <= 2:
        return torch.zeros(4, 0, 0, device=cv.device), 0, 0
    dev = torch.device(cv.device)
    ys = torch.arange(h, device=dev, dtype=torch.float32).view(h, 1) + (iy0 + 0.5)
    xs = torch.arange(w, device=dev, dtype=torch.float32).view(1, w) + (ix0 + 0.5)
    cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    r = min(radius * k, hx, hy)
    qx, qy = (xs - cx).abs() - (hx - r), (ys - cy).abs() - (hy - r)
    dist = torch.sqrt(qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2) + torch.maximum(qx, qy).clamp(max=0) - r
    a = (0.5 - dist).clamp(0.0, 1.0) * opacity
    rgb_ = torch.tensor(rgb(color), device=dev, dtype=torch.float32).view(3, 1, 1)
    return torch.cat([rgb_ * a, a.unsqueeze(0)], dim=0), ix0, iy0


def draw_run(cv, run: G.GlyphRun, style: TextStyle, per_glyph_xform=None, *, atlas: GlyphAtlas) -> torch.Tensor:
    """`run` painted onto a transparent canvas: (4, H, W) premultiplied RGBA on `cv.device`."""
    crop, x0, y0 = draw_run_crop(cv, run, style, per_glyph_xform, atlas=atlas)
    out = torch.zeros(4, cv.H, cv.W, device=cv.device, dtype=torch.float32)
    h, w = crop.shape[-2:]
    out[:, y0:y0 + h, x0:x0 + w] = crop
    return out
