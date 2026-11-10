"""`describe(plan, ...)`: what is on screen at one frame, as numbers. No pixel is rendered.

Text geometry comes from the renderer's own Chromium rasters (`Renderer.text_layout()`) and its
reveal/visibility helpers (`render.text.line_extent`), so the numbers and the pixels cannot drift.
Boxes are output pixels `[x0, y0, x1, y1]`. `contrast` and `over_face` are left null here: the first
needs a rendered frame (`montaj.inspector` fills it), the second needs face boxes the engine lacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from montaj.render.core import sine
from montaj.render.text import (
    _NAMED,
    _SIZE,
    DESIGN_H,
    RasterImage,
    RasterLine,
    RasterSet,
    Sub,
    TextBlock,
    TextLine,
    line_extent,
)
from montaj.timeline import Card, FramePlan, Timeline, Trans

# A text box must keep this fraction of the frame width/height free on every side.
SAFE_MARGIN = 0.05


@dataclass(frozen=True)
class TextLayout:
    """Everything `describe` reads of the renderer's text: blocks, subs, rasters, card lines (per shot)."""

    blocks: tuple[TextBlock, ...]
    subs: tuple[Sub, ...]
    rasters: RasterSet
    card_lines: dict[int, tuple[TextLine, ...]]
    card_rasters: dict[int, tuple[RasterLine, ...]]
    k: float
    height: int
    ink: dict = field(default_factory=dict, compare=False)  # per-unit ink rects, filled lazily


def _cards(scene) -> list[Card]:
    if isinstance(scene, Card):
        return [scene]
    if isinstance(scene, Trans):
        return _cards(scene.a) + _cards(scene.b)
    return []


def _hex(line: TextLine) -> str:
    return _NAMED["gold"] if line.gold else _NAMED.get(line.color or "cream", line.color or _NAMED["cream"])


def _element(ident: str, text: str, box_design, k: float, size: float | None, color: str, alpha: float,
             W: int, H: int) -> dict:
    x0, y0, x1, y1 = (round(v * k, 1) for v in box_design)
    off = max(0.0, -x0, -y0, x1 - W, y1 - H)
    mx, my = SAFE_MARGIN * W, SAFE_MARGIN * H
    safe = x0 >= mx and y0 >= my and x1 <= W - mx and y1 <= H - my
    return {
        "id": ident, "text": text, "box": [x0, y0, x1, y1], "size": size, "color": color,
        "alpha": round(alpha, 3), "off_frame_px": round(off, 1), "safe": safe,
        "contrast": None, "over_face": None,
    }


def _line_elements(prefix: str, block: TextBlock, lines, rasters, t: float, fps: float, lay: TextLayout,
                   W: int, H: int) -> list[dict]:
    out = []
    for i, (line, raster) in enumerate(zip(lines, rasters)):
        ext = line_extent(block, line, raster, t, fps, lay.k, lay.ink)
        if ext is None:
            continue
        alpha, box = ext
        size = line.size if line.size is not None else _SIZE.get(line.style)
        out.append(_element(f"{prefix}.{i}", line.text, box, lay.k, size, _hex(line), alpha, W, H))
    return out


def describe(plan: FramePlan, tl: Timeline, lay: TextLayout) -> dict:
    W, H = tl.size
    fps = float(tl.fps)
    f = plan.f
    t = float(f)
    scene = plan.scene
    text: list[dict] = []
    # Overlay blocks own the first len(block.lines) rasters each, in block order.
    cursor = 0
    starts = []
    for block in lay.blocks:
        starts.append(cursor)
        cursor += len(block.lines)
    for bi in plan.texts:
        block = lay.blocks[bi]
        rasters = lay.rasters.lines[starts[bi] : starts[bi] + len(block.lines)]
        text += _line_elements(f"text#{bi}", block, block.lines, rasters, t, fps, lay, W, H)
    for card in _cards(scene):
        lines = lay.card_lines.get(card.shot, ())
        rasters = lay.card_rasters.get(card.shot, ())
        block = TextBlock(card.span[0], card.span[1], 0.0, 0.0, 0.0, lines)
        text += _line_elements(f"card#{card.shot}", block, lines, rasters, t, fps, lay, W, H)
    # draw_text paints only the first sub whose window holds t.
    lead, tail, fade = 0.2 * fps, 0.3 * fps, 0.5 * fps
    for si, (sub, ras) in enumerate(zip(lay.subs, lay.rasters.subs)):
        if t < sub.start - lead or t > sub.end + tail:
            continue
        op = min(sine((t - (sub.start - lead)) / fade), sine(((sub.end + tail) - t) / fade))
        if op > 1e-4 and ras.fill.numel():
            x, y, w, h = ras.box
            y += (H - DESIGN_H * lay.k) / lay.k if ras.bottom else 0.0
            text.append(_element(f"sub#{si}", sub.text, (x, y, x + w, y + h), lay.k, sub.size,
                                 _NAMED.get(sub.color or "cream", sub.color or _NAMED["cream"]), op, W, H))
        break
    return {
        "t": round(f / fps, 4), "frame": f, "shot": _shot(tl, f), "transition": _transition(scene, fps),
        "layers": _layers(tl, scene), "text": text, "fx": _fx(plan),
    }


def _shot(tl: Timeline, f: int) -> int | None:
    for sh in tl.shots:
        if sh.start <= f < sh.end:
            return sh.index
    return None


def _transition(scene, fps: float) -> dict | None:
    if not isinstance(scene, Trans):
        return None
    return {"kind": scene.kind, "start": round(scene.t0, 3), "dur_frames": round(scene.dur, 3)}


def _leaves(scene) -> list:
    return _leaves(scene.a) + _leaves(scene.b) if isinstance(scene, Trans) else [scene]


def _layers(tl: Timeline, scene) -> list[dict]:
    """One entry per scene leaf. Photo/clip placement is camera maths the engine keeps private, so no box."""
    out = []
    for leaf in _leaves(scene):
        index = getattr(leaf, "shot", None)
        if index is None:
            out.append({"id": "blank", "kind": "blank"})
            continue
        sh = tl.shots[index]
        src = sh.spec.photo or sh.spec.clip
        entry = {"id": f"shot:{index}", "kind": sh.kind}
        if src:
            entry["src"] = src
        if sh.spec.layers is not None:
            entry["layers"] = [layer.clip or layer.image for layer in sh.spec.layers]
        out.append(entry)
    return out


def _fx(plan: FramePlan) -> list[str]:
    fx = []
    for name, on in (("flash", plan.flashes), ("leak", plan.leaks), ("hit", plan.hits), ("burst", plan.bursts),
                     ("rgb_split", plan.splits), ("dust", plan.dust), ("glow", plan.glow), ("bars", plan.bars),
                     ("intro", plan.intro), ("outro", plan.outro), ("fade_out", plan.fade_out)):
        if on:
            fx.append(name)
    return fx
