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
        return out

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

    def problems(self) -> list[dict]:
        """Every text element that is off-frame or outside the safe area on some frame, or low-contrast
        at its most visible frame. One record per (element, issue)."""
        found: dict[tuple[str, str], dict] = {}
        peak: dict[str, tuple[float, int]] = {}
        for f in range(self.tl.n_frames):
            for el in describe(self.tl.plan(f), self.tl, self.layout)["text"]:
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


def lines(payload: dict) -> list[str]:
    """Compact text: one line per frame header and per text element, then the result line."""
    out: list[str] = []
    if "problems" in payload:
        probs = payload["problems"]
        for p in probs:
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
