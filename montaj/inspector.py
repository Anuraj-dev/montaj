"""`montaj inspect`: what is on screen at a time, as numbers instead of an image.

Geometry is `render.describe.describe` over the `FramePlan` (no render). `contrast` needs one rendered
frame per time and is skipped with `fast`; `over_face` stays null (no face boxes in the engine yet).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from montaj import gpulock
from montaj.render.canvas import Canvas
from montaj.render.describe import describe
from montaj.render.frame import Renderer
from montaj.render.text import _rgb
from montaj.spec import load_spec
from montaj.timeline import Timeline, resolve

MIN_CONTRAST = 3.0  # WCAG AA for large text; captions are large
RING_PX = 6  # background is sampled in this band just outside the text box


def _linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    lin = _linear(rgb)
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(img: torch.Tensor, box: list[float], color: str) -> float | None:
    """WCAG contrast ratio of the text colour against the frame just around `box` (the box itself
    holds the glyphs). Falls back to the box median when the frame leaves no ring. None if empty."""
    _, H, W = img.shape
    x0, y0 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x1, y1 = min(W, math.ceil(box[2])), min(H, math.ceil(box[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    ox0, oy0, ox1, oy1 = max(0, x0 - RING_PX), max(0, y0 - RING_PX), min(W, x1 + RING_PX), min(H, y1 + RING_PX)
    mask = torch.ones(oy1 - oy0, ox1 - ox0, dtype=torch.bool, device=img.device)
    mask[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0] = False
    region = img[:, oy0:oy1, ox0:ox1]
    if mask.any():
        bg = _luma(region[:, mask]).mean()
    else:
        bg = _luma(region.reshape(3, -1)).median()
    fg = _luma(torch.tensor(_rgb(color), device=img.device, dtype=img.dtype))
    hi, lo = max(float(fg), float(bg)), min(float(fg), float(bg))
    return round((hi + 0.05) / (lo + 0.05), 2)


def _frame_of(tl: Timeline, t: float) -> int:
    return min(max(0, int(round(t * tl.fps))), tl.n_frames - 1)


def _step(raw: str, fps: int) -> int:
    """`10f` frames or `0.5s` seconds, as whole frames >= 1."""
    text = raw.strip()
    try:
        n = float(text[:-1]) if text.endswith("f") else float(text.rstrip("s")) * fps
    except ValueError:
        raise ValueError(f"--every {raw!r}: use frames like 10f or seconds like 0.5s") from None
    if not math.isfinite(n) or n < 1:
        raise ValueError(f"--every {raw!r}: use at least one frame")
    return int(round(n))


OVERLAP_MIN = 5.0  # % of the smaller ink box; less is italic or kerning overhang, not a defect

Rect = tuple[float, float, float, float]


def _turn(r: Rect, pivot: tuple[float, float], deg: float) -> Rect:
    """Bounds of rect `r` turned `deg` degrees about `pivot`."""
    px, py = pivot
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    pts = [(px + (x - px) * c - (y - py) * s, py + (x - px) * s + (y - py) * c)
           for x, y in ((r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3]))]
    return min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)


def unit_boxes(block, line, raster, t: float, fps: float, k: float, memo: dict) -> list[tuple[str, Rect]]:
    """`(text, ink box)` per reveal unit painted at `t`, in paint order, design px. A moving gpu unit's box
    is its posed glyphs' (scale, squash, offset and turn as `motion_xforms` hands them to the painter);
    other units sit at their settled place, as `line_extent` measures them. Empty when the block is hidden."""
    from montaj.render.glyphs import _units
    from montaj.render.text import _ink, _pop, _unit_ks, _vis, motion_xforms
    from montaj.render.type import GlyphLine

    if _vis(t, block, fps) <= 0.0:
        return []
    if not isinstance(raster, GlyphLine):  # chromium raster: one bitmap per unit
        n = len(raster.units)
        ks = [1.0 if _pop(line, block, t) > 1e-4 else 0.0] * n if line.reveal == "pop" else _unit_ks(line, n, t, fps)
        return [(u.text.strip(), r) for u, uk in zip(raster.units, ks) if uk > 1e-4 and (r := _ink(u, k, memo)) is not None]
    run = raster.run
    if line.motion is not None:
        xforms, ks = motion_xforms(line, run, t, fps), None
    else:
        xforms = [None] * len(run.glyphs)
        ks = [1.0 if _pop(line, block, t) > 1e-4 else 0.0] * run.units if line.reveal == "pop" else _unit_ks(line, run.units, t, fps)
    boxes: dict[int, list[float]] = {}
    for g, xf in zip(run.glyphs, xforms):
        if g.ink[2] <= g.ink[0]:
            continue
        if xf is None:
            if ks is not None and g.unit < len(ks) and ks[g.unit] <= 1e-4:
                continue
            r = g.ink
        else:
            if xf.opacity <= 1e-4:
                continue
            cx, cy = (g.ink[0] + g.ink[2]) / 2 + xf.dx, (g.ink[1] + g.ink[3]) / 2 + xf.dy
            hw, hh = (g.ink[2] - g.ink[0]) / 2 * xf.scale, (g.ink[3] - g.ink[1]) / 2 * xf.scale * xf.sy
            r = (cx - hw, cy - hh, cx + hw, cy + hh)
            if xf.rotate:
                r = _turn(r, (cx, cy), xf.rotate)
        b = boxes.setdefault(g.unit, list(r))
        b[:] = min(b[0], r[0]), min(b[1], r[1]), max(b[2], r[2]), max(b[3], r[3])
    of, _ = _units(run.text, line.by or "word")
    out = []
    for u in sorted(boxes):
        r = tuple(boxes[u])
        if run.rotate:
            r = _turn(r, run.pivot, run.rotate)
        out.append(("".join(c for c, w in zip(run.text, of) if w == u).strip(), r))
    return out


def _covered(a: Rect, b: Rect) -> float:
    """Share (%) of the smaller of two boxes that the other covers."""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return 100.0 * w * h / small if small > 0 else 0.0


class Inspector:
    """One resolved timeline plus one Renderer (layout always, pixels only for contrast)."""

    def __init__(self, root: Path, *, fast: bool):
        self.tl = resolve(load_spec(root / "montaj.yaml"), root)
        W, H = self.tl.size
        self.fast = fast
        device = "cpu" if fast or not torch.cuda.is_available() else "cuda"
        self.renderer = Renderer(self.tl, root, Canvas(W, H, device=device))
        self.layout = self.renderer.text_layout()

    def at(self, f: int) -> dict:
        out = describe(self.tl.plan(f), self.tl, self.layout)
        if not self.fast and out["text"]:
            self.add_contrast(f, out["text"])
        if not self.fast:
            out["scopes"] = self.scopes_at(f)
        return out

    def scopes_at(self, f: int) -> dict | None:
        """Scope numbers of the graded picture at `f` (#45); None on a transition or a blank."""
        from montaj import scopes

        pic = self.renderer.picture(f)
        if pic is None:
            return None
        return scopes.as_dict(scopes.measure([pic], self.tl.plan(f).scene.shot))

    def add_contrast(self, f: int, elements: list[dict]) -> None:
        with torch.no_grad():
            img = self.renderer.frame(f)
        for el in elements:
            el["contrast"] = contrast(img, el["box"], el["color"])

    def frames(self, *, at: float | None, shot: int | None, rng: list[str] | None, every: str | None) -> list[int]:
        tl = self.tl
        if at is not None:
            return [_frame_of(tl, at)]
        if shot is not None:
            if not 0 <= shot < len(tl.shots):
                raise ValueError(f"--shot {shot}: film has shots 0..{len(tl.shots) - 1}")
            sh = tl.shots[shot]
            return [(sh.start + sh.end - 1) // 2]
        assert rng is not None
        a, b = _frame_of(tl, float(rng[0].rstrip("s"))), _frame_of(tl, float(rng[1].rstrip("s")))
        return list(range(a, b + 1, _step(every or "10f", tl.fps)))

    def inked(self, plan, subs: list[dict]) -> list[tuple[str, str, list[tuple[str, Rect]]]]:
        """`(id, text, unit boxes)` per text element on screen in `plan`, in describe's order and ids, design px.
        `subs` are describe's `sub#` elements (one box each, output px)."""
        from montaj.render.describe import _cards
        from montaj.render.text import TextBlock

        lay, t, fps = self.layout, float(plan.f), float(self.tl.fps)
        starts, cursor = [], 0
        for block in lay.blocks:
            starts.append(cursor)
            cursor += len(block.lines)
        groups = [(f"text#{bi}", lay.blocks[bi], lay.blocks[bi].lines,
                   lay.rasters.lines[starts[bi]: starts[bi] + len(lay.blocks[bi].lines)]) for bi in plan.texts]
        for card in _cards(plan.scene):
            lines = lay.card_lines.get(card.shot, ())
            groups.append((f"card#{card.shot}", TextBlock(card.span[0], card.span[1], 0.0, 0.0, 0.0, lines),
                           lines, lay.card_rasters.get(card.shot, ())))
        out = []
        for prefix, block, lines, rasters in groups:
            for i, (line, raster) in enumerate(zip(lines, rasters)):
                units = unit_boxes(block, line, raster, t, fps, lay.k, lay.ink)
                if units:
                    out.append((f"{prefix}.{i}", line.text, units))
        for el in subs:
            out.append((el["id"], el["text"], [(el["text"], tuple(v / lay.k for v in el["box"]))]))
        return out

    def overlaps(self, plan, subs: list[dict]) -> list[dict]:
        """Ink over ink at `f`: a word over an earlier word of its line, or any word of an element over a
        word of an element painted before it. Value: % of the smaller ink box covered (>= OVERLAP_MIN)."""
        out, f = [], plan.f
        els = self.inked(plan, subs)
        for k, (ident, text, units) in enumerate(els):
            for j, (over, b) in enumerate(units):  # inside the line: later units paint over earlier ones
                for under, a in units[:j]:
                    pct = _covered(a, b)
                    if pct >= OVERLAP_MIN:
                        out.append({"id": ident, "text": over, "under": under, "frame": f, "issue": "overlap", "value": round(pct)})
            for _, under_text, under_units in els[:k]:  # between elements: the later one paints on top
                pct = max((_covered(a, b) for _, a in under_units for _, b in units), default=0.0)
                if pct >= OVERLAP_MIN:
                    out.append({"id": ident, "text": text, "under": under_text, "frame": f, "issue": "overlap", "value": round(pct)})
        return out

    def problems(self) -> list[dict]:
        """Every text element that is off-frame or outside the safe area on some frame, whose ink covers
        other ink at its animated pose on some frame (`overlap`, worst frame), or low-contrast at its most
        visible frame. One record per (element, issue), and per covered text for overlaps."""
        found: dict[tuple[str, ...], dict] = {}
        peak: dict[str, tuple[float, int]] = {}
        for f in range(self.tl.n_frames):
            plan = self.tl.plan(f)
            els = describe(plan, self.tl, self.layout)["text"]
            for rec in self.overlaps(plan, [e for e in els if e["id"].startswith("sub#")]):
                key = (rec["id"], "overlap", rec["under"])
                if key not in found or rec["value"] > found[key]["value"]:
                    found[key] = rec
            for el in els:
                if el["off_frame_px"] > 0:
                    rec = found.setdefault((el["id"], "off_frame"), {"id": el["id"], "text": el["text"], "frame": f, "issue": "off_frame", "value": 0.0})
                    rec["value"] = max(rec["value"], el["off_frame_px"])
                elif not el["safe"]:
                    found.setdefault((el["id"], "unsafe"), {"id": el["id"], "text": el["text"], "frame": f, "issue": "unsafe", "value": None})
                if el["alpha"] > peak.get(el["id"], (0.0, 0))[0] + 1e-6:
                    peak[el["id"]] = (el["alpha"], f)
        if not self.fast:
            by_frame: dict[int, set[str]] = {}
            for ident, (_, f) in peak.items():
                by_frame.setdefault(f, set()).add(ident)
            for f, ids in sorted(by_frame.items()):
                els = [e for e in describe(self.tl.plan(f), self.tl, self.layout)["text"] if e["id"] in ids]
                self.add_contrast(f, els)
                for el in els:
                    if el["contrast"] is not None and el["contrast"] < MIN_CONTRAST:
                        found[(el["id"], "contrast")] = {"id": el["id"], "text": el["text"], "frame": f, "issue": "low_contrast", "value": el["contrast"]}
        return sorted(found.values(), key=lambda r: (r["frame"], r["id"]))


def inspect(root: Path, *, at: float | None = None, shot: int | None = None, rng: list[str] | None = None,
            every: str | None = None, fast: bool = False, problems: bool = False) -> dict:
    """`{"frames": [...]}` for times, `{"problems": [...]}` for `problems`. Takes the GPU lock unless `fast`."""
    if not fast:
        with gpulock.hold(root / "build" / "inspect.log"):
            return _run(root, at, shot, rng, every, fast, problems)
    return _run(root, at, shot, rng, every, fast, problems)


def _run(root, at, shot, rng, every, fast, problems) -> dict:
    ins = Inspector(root, fast=fast)
    if problems:
        return {"problems": ins.problems()}
    return {"frames": [ins.at(f) for f in ins.frames(at=at, shot=shot, rng=rng, every=every)]}


TRACK_SAMPLES = 12
MAX_SAMPLES = 500


def _spread(first: int, last: int, samples: int) -> list[int]:
    count = min(samples, max(2, last - first + 1))
    return sorted({round(first + (last - first) * i / (count - 1)) for i in range(count)})


def _round(v):
    return [round(x, 4) for x in v] if isinstance(v, tuple) else round(v, 4)


def track(root: Path, ref: str, samples: int = TRACK_SAMPLES) -> dict:
    """The sampled curve of one animated value, no render and no GPU. `ref` is `SHOT.LAYER.PROP` (a keyed or
    `expr` layer property; shot and layer as in `--shot`: zero-based, LAYER may be a layer `id`),
    `SHOT.camera.zoom` / `SHOT.camera.focus` (a clip shot's camera keys; focus in source px) or
    `tracks.dust|glow|bars`. Frames are shot-local, except `tracks.*` (film frames)."""
    from montaj.clipmap import camera_at
    from montaj.render.core import key as eval_key
    from montaj.render.layers import is_keyed, sample
    from montaj.spec import LAYER_KEYED

    if not 2 <= samples <= MAX_SAMPLES:
        raise ValueError(f"--samples {samples}: use 2 to {MAX_SAMPLES}")
    parts = ref.split(".")
    if len(parts) not in (2, 3) or not (parts[0] == "tracks" or parts[0].isdigit()):
        raise ValueError("--track wants SHOT.LAYER.PROP (1.0.scale), SHOT.camera.zoom|focus or tracks.NAME")
    tl = resolve(load_spec(root / "montaj.yaml"), root)
    if parts[0] == "tracks":
        keys = tl._tracks.get(parts[1])
        if keys is None:
            raise ValueError(f"no track {parts[1]!r} (dust, glow, bars)")
        if not keys:
            raise ValueError(f"{ref} has no keys")
        frames = _spread(round(keys[0][0]), round(keys[-1][0]), samples)
        values = [float(eval_key(float(f), keys)) for f in frames]
        return {"track": ref, "keys": [round(k[0]) for k in keys], "frames": frames, "values": [_round(v) for v in values]}
    n = int(parts[0])
    if not 0 <= n < len(tl.shots):
        raise ValueError(f"no shot {n} (0..{len(tl.shots) - 1})")
    shot = tl.shots[n]
    if parts[1] == "camera":
        cam = shot.spec.camera
        what = parts[2] if len(parts) == 3 else ""
        if what not in ("zoom", "focus"):
            raise ValueError("--track camera wants SHOT.camera.zoom or SHOT.camera.focus")
        if not cam or (what == "focus" and not any(k.focus for k in cam)):
            raise ValueError(f"shot {n} has no camera {what} keys")
        keys = [(float(k.t), k.zoom, k.focus, k.ease) for k in cam]
        frames = _spread(cam[0].t, cam[-1].t, samples)
        crop = tuple(shot.spec.crop or (0, 0, 1, 1))
        out = [camera_at(keys, crop, float(f)) for f in frames]
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        values = [v[0] for v in out] if what == "zoom" else [
            (crop[0] + v[1] * cw, crop[1] + v[2] * ch) for v in out]
        return {"track": ref, "keys": [k.t for k in cam], "frames": frames, "values": [_round(v) for v in values]}
    if len(parts) != 3:
        raise ValueError("--track wants SHOT.LAYER.PROP, e.g. 1.0.scale")
    layer_s, prop = parts[1], parts[2]
    layers = shot.spec.layers or []
    idx = next((i for i, l in enumerate(layers) if l.id == layer_s), int(layer_s) if layer_s.isdigit() else -1)
    if not 0 <= idx < len(layers):
        raise ValueError(f"shot {n} has no layer {layer_s}")
    if prop not in LAYER_KEYED:
        raise ValueError(f"{prop} is not keyable (one of {', '.join(LAYER_KEYED)})")
    val = getattr(layers[idx], prop)
    if not is_keyed(val):
        raise ValueError(f"{ref} is not keyed")
    if isinstance(val, list):
        keys = [k.t for k in val]
        frames = _spread(keys[0], keys[-1], samples)
    else:  # an expression has no keys: sample the whole hold
        keys = []
        frames = _spread(0, max(1, shot.end - shot.start - 1), samples)
    return {"track": ref, "keys": keys, "frames": frames, "values": [_round(sample(val, f, None)) for f in frames]}


def lines(payload: dict) -> list[str]:
    if "track" in payload:
        pts = " ".join(f"{f}f={v}" for f, v in zip(payload["frames"], payload["values"]))
        return [f"OK track {payload['track']} keys={payload['keys']} {pts}"]
    """Compact text: one line per frame header and per text element, then the result line."""
    out: list[str] = []
    if "problems" in payload:
        probs = payload["problems"]
        for p in probs:
            if p["issue"] == "overlap":
                out.append(f"PROBLEM {p['id']} f={p['frame']} overlap {p['text']!r} over {p['under']!r} {p['value']}%")
                continue
            val = "" if p["value"] is None else f" {p['value']}"
            out.append(f"PROBLEM {p['id']} f={p['frame']} {p['issue']}{val} {p['text']!r}")
        out.append(f"{'WARN' if probs else 'OK'} inspect {len(probs)} problems")
        return out
    frames = payload["frames"]
    for fr in frames:
        tr = f" transition={fr['transition']['kind']}" if fr["transition"] else ""
        layers = ",".join(layer["id"] for layer in fr["layers"])
        out.append(f"f={fr['frame']} t={fr['t']}s shot={fr['shot']}{tr} layers={layers} fx={','.join(fr['fx']) or '-'}")
        for el in fr["text"]:
            c = "-" if el["contrast"] is None else el["contrast"]
            out.append(f"  {el['id']} {el['text']!r} box={el['box']} size={el['size']} contrast={c} "
                       f"off={el['off_frame_px']} safe={str(el['safe']).lower()}")
    n = sum(len(fr["text"]) for fr in frames)
    out.append(f"OK inspect {len(frames)} frames {n} text elements")
    return out
