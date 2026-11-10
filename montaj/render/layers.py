"""Layer compositing (#34): one source drawn over the layers before it.

A layer is a texture (premultiplied RGBA or opaque RGB, on the device), a destination rect in device
px, a per-frame `LayerState` (scale, rotate, opacity, shift) and a blend mode. `layer_state` is the only
place a layer's time-varying values are computed, so keyframes (#35) replace its body, not its callers.

Geometry follows `shots.card_frame`: a screen pixel centre maps into the layer's local frame about the
rect centre (scale, then clockwise rotation), samples the texture bilinearly, and a rounded-rect
signed distance gives a 1 px anti-aliased edge (coverage = 0.5 − d). Work is confined to the layer's
screen bounding window, so a small sticker costs a small tensor.

Blending is the W3C separable set on an opaque backdrop, in display (sRGB) values like every other
compositor here: `out = (1 − a)·cb + a·B(cb, cs)`, cs the unpremultiplied source; `add` is
`cb + premul` clamped.

PNG layers decode on the device: zlib inflate is byte decompression on the host (like demux); the
row filters are undone on the device in one anti-diagonal wavefront, then the palette / colour key /
alpha expand there too. Interlaced and sub-byte PNGs fall back to PIL through the GPU policy.
"""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from montaj.easing import parse_ease
from montaj.render.core import key as eval_key
from montaj.render.shots import _erf_rect

# Drop shadow in design px (scaled by Canvas.k and the layer's scale): offset down, gaussian sigma.
SHADOW_OFFSET = 18.0
SHADOW_SIGMA = 28.0
# Back easing overshoot for `pop` (easeOutBack); opacity reaches 1 at this fraction of the pop.
_BACK = 1.70158
_POP_FADE = 0.35


@dataclass(frozen=True)
class LayerState:
    """A layer's values at one instant. Everything `composite` animates is here."""

    scale: float
    rotate: float  # degrees, clockwise
    opacity: float
    dx: float = 0.0  # design px shift of the layer's rect from where `pos` first placed it (a keyed `pos`)
    dy: float = 0.0


def _smooth(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def _out_back(x: float) -> float:
    c3 = _BACK + 1.0
    return 1.0 + c3 * (x - 1.0) ** 3 + _BACK * (x - 1.0) ** 2


def _anim(state: LayerState, kind: str, x: float) -> LayerState:
    """Apply an entrance at progress x (0 hidden, 1 settled); an exit passes 1 − progress."""
    x = min(1.0, max(0.0, x))
    if kind == "fade":
        return replace(state, opacity=state.opacity * _smooth(x))
    if kind == "pop":
        return replace(state, scale=state.scale * (0.5 + 0.5 * _out_back(x)),
                       opacity=state.opacity * min(1.0, x / _POP_FADE))
    raise ValueError(f"layer anim: unknown {kind!r}")


def sample(prop, t: float, default):
    """A layer property at shot-clock `t`: a plain value, or the key list held before the first key and
    after the last (`render.core.key`, with `easing.parse_ease` shaping each move)."""
    if prop is None:
        return default
    if not isinstance(prop, list):
        return prop
    keys = [(float(k.t), tuple(map(float, k.v)) if isinstance(k.v, (tuple, list)) else float(k.v),
             parse_ease(k.ease)) for k in prop]
    return eval_key(t, keys)


def base_pos(layer) -> tuple[float, float] | None:
    """Where a `pos` layer is placed (design px): its value, or its first key's. Animation moves it from here."""
    pos = layer.pos
    if pos is None:
        return None
    p = pos[0].v if isinstance(pos, list) else pos
    return float(p[0]), float(p[1])


def base_box(layer) -> tuple[float, float, float, float] | None:
    """A box layer's rect (design px): its value, or its first key's. `src` and `fit` resolve against it."""
    box = layer.box
    if box is None:
        return None
    b = box[0].v if isinstance(box, list) else box
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def box_growth(layer) -> float:
    """How much larger than its first-key box a keyed `box` ever gets (>= 1), for decode sizing."""
    if not isinstance(layer.box, list):
        return 1.0
    b = base_box(layer)
    w, h = max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1])
    return max(1.0, *((float(k.v[2]) - float(k.v[0])) / w for k in layer.box),
               *((float(k.v[3]) - float(k.v[1])) / h for k in layer.box))


def peak_scale(layer) -> float:
    """The largest scale the layer reaches from its keys (a plain scale is itself), for decode sizing."""
    sc = layer.scale
    if sc is None:
        return 1.0
    return max(float(k.v) for k in sc) if isinstance(sc, list) else float(sc)


def layer_state(layer, t_local: float, hold: int) -> LayerState | None:
    """Values at `t_local` frames after the shot start, or None while the layer is off.

    `from`/`to` and the anim durations are whole frames (materialised by `resolve`). An unset `from`
    shows the layer from before the shot (transition and motion-blur samples); an unset `to` keeps it
    through the outgoing transition. `out` ends at `to`, or at the end of the hold.
    """
    start = layer.from_
    if start is not None and t_local < start:
        return None
    if layer.to is not None and t_local >= layer.to:
        return None
    state = LayerState(
        float(sample(layer.scale, t_local, 1.0)),
        float(sample(layer.rotate, t_local, 0.0)),
        min(1.0, max(0.0, float(sample(layer.opacity, t_local, 1.0)))),  # a spring may overshoot past 1
    )
    if isinstance(layer.pos, list):
        x, y = sample(layer.pos, t_local, None)
        bx, by = base_pos(layer)
        state = replace(state, dx=x - bx, dy=y - by)
    if layer.in_ is not None:
        state = _anim(state, layer.in_.type, (t_local - (start or 0)) / float(layer.in_.dur))
    if layer.out is not None:
        end = layer.to if layer.to is not None else hold
        state = _anim(state, layer.out.type, (end - t_local) / float(layer.out.dur))
    return state if state.opacity > 0 else None


def _sat(cs: torch.Tensor) -> torch.Tensor:
    return cs.clamp(0.0, 1.0)


def _soft_light(cb: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
    d = torch.where(cb <= 0.25, ((16 * cb - 12) * cb + 4) * cb, cb.sqrt())
    return torch.where(cs <= 0.5, cb - (1 - 2 * cs) * cb * (1 - cb), cb + (2 * cs - 1) * (d - cb))


_BLENDS = {
    "screen": lambda cb, cs: cb + cs - cb * cs,
    "multiply": lambda cb, cs: cb * cs,
    "difference": lambda cb, cs: (cb - cs).abs(),
    "overlay": lambda cb, cs: torch.where(cb <= 0.5, 2 * cb * cs, 1 - 2 * (1 - cb) * (1 - cs)),
    "soft-light": _soft_light,
}


def blend(cb: torch.Tensor, premul: torch.Tensor, a: torch.Tensor, mode: str) -> torch.Tensor:
    """Composite premultiplied `premul` (3, h, w) with alpha `a` (1, h, w) over opaque `cb`."""
    if mode == "normal":
        return premul + cb * (1 - a)
    if mode == "add":
        return (cb + premul).clamp(max=1.0)
    fn = _BLENDS.get(mode)
    if fn is None:
        raise ValueError(f"layer blend: unknown {mode!r}")
    cs = _sat(premul / a.clamp(min=1e-6))
    return cb * (1 - a) + a * fn(cb, cs)


class LayerTexture:
    """A source texture (C, h, w), C = 3 opaque or 4 premultiplied, plus a lazy mip chain for minification."""

    def __init__(self, tex: torch.Tensor):
        self.levels = [tex]

    @property
    def tex(self) -> torch.Tensor:
        return self.levels[0]

    def level(self, ratio: float) -> torch.Tensor:
        """The level for `ratio` source px per screen px (bilinear only aliases past 2:1)."""
        want = int(math.floor(math.log2(ratio))) if ratio >= 2.0 else 0
        while len(self.levels) <= want:
            top = self.levels[-1]
            h, w = int(top.shape[1]), int(top.shape[2])
            if h < 4 or w < 4:
                break
            self.levels.append(F.interpolate(top[None], size=((h + 1) // 2, (w + 1) // 2), mode="bilinear",
                                             antialias=True, align_corners=False)[0])
        return self.levels[min(want, len(self.levels) - 1)]


def _window(cv, cx: float, cy: float, ex: float, ey: float) -> tuple[int, int, int, int] | None:
    x0 = max(0, math.floor(cx - ex))
    y0 = max(0, math.floor(cy - ey))
    x1 = min(cv.W, math.ceil(cx + ex))
    y1 = min(cv.H, math.ceil(cy + ey))
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)


def composite(cv, dst: torch.Tensor, source: LayerTexture, src_rect: tuple[float, float, float, float],
              dest_rect: tuple[float, float, float, float], state: LayerState, mode: str = "normal",
              radius: float = 0.0, shadow: float = 0.0, post=None, post_pad: float = 0.0) -> None:
    """Draw `src_rect` of `source` (texture px) into `dest_rect` (device px) of `dst`, in place.

    `radius` is device px; `shadow` is the drop shadow's opacity. `dst` must be caller-owned.
    `post(rgba, cx, cy)` filters the layer's premultiplied window (layer centre in window px) before the
    blend: layer effects (#38). `post_pad` device px grows the window so the filter has room to spread.
    """
    x0, y0, x1, y1 = dest_rect
    hw, hh = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    if hw <= 0 or hh <= 0 or state.scale <= 0 or state.opacity <= 0:
        return
    s = state.scale
    cx, cy = (x0 + x1) / 2.0 + state.dx * cv.k, (y0 + y1) / 2.0 + state.dy * cv.k
    th = math.radians(state.rotate)
    cos, sin = math.cos(th), math.sin(th)
    ex = (abs(cos) * hw + abs(sin) * hh) * s
    ey = (abs(sin) * hw + abs(cos) * hh) * s
    pad = 2.0 + max(0.0, post_pad)
    off = sig = 0.0
    if shadow > 0:
        off, sig = SHADOW_OFFSET * cv.k, SHADOW_SIGMA * cv.k
        pad += (off + 3.0 * sig) * s
    win = _window(cv, cx, cy, ex + pad, ey + pad)
    if win is None:
        return
    wx0, wy0, wx1, wy1 = win
    dev = dst.device
    X = torch.arange(wx0, wx1, device=dev, dtype=torch.float32)[None, :] + 0.5
    Y = torch.arange(wy0, wy1, device=dev, dtype=torch.float32)[:, None] + 0.5
    dx, dy = (X - cx) / s, (Y - cy) / s
    lx = cos * dx + sin * dy
    ly = -sin * dx + cos * dy

    sx0, sy0, sx1, sy1 = src_rect
    tex = source.level((sx1 - sx0) / max(1e-6, 2.0 * hw * s))
    tw, tht = float(source.tex.shape[2]), float(source.tex.shape[1])
    gx = (sx0 + (lx + hw) / (2 * hw) * (sx1 - sx0)) / tw * 2 - 1
    gy = (sy0 + (ly + hh) / (2 * hh) * (sy1 - sy0)) / tht * 2 - 1
    grid = torch.stack(torch.broadcast_tensors(gx, gy), dim=-1)[None]
    smp = F.grid_sample(tex[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]

    r = min(max(0.0, radius), hw, hh)
    qx, qy = lx.abs() - (hw - r), ly.abs() - (hh - r)
    sd = torch.hypot(qx.clamp(min=0), qy.clamp(min=0)) + torch.maximum(qx, qy).clamp(max=0) - r
    cov = (0.5 - sd * s).clamp(0.0, 1.0)[None] * state.opacity
    if smp.shape[0] == 4:
        premul, a = smp[:3] * cov, smp[3:4] * cov
    else:
        premul, a = smp * cov, cov
    if post is not None:
        rgba = post(torch.cat((premul, a.expand(1, *premul.shape[1:])), dim=0), cx - wx0, cy - wy0)
        premul, a = rgba[:3], rgba[3:4]

    view = dst[:, wy0:wy1, wx0:wx1]
    cb = view
    if shadow > 0:
        a_sh = (shadow * state.opacity) * _erf_rect(lx, ly, -hw, hw, -hh + off, hh + off, sig)
        cb = cb * (1 - a_sh[None])
    view.copy_(blend(cb, premul, a, mode))


# ---- PNG on the device ----------------------------------------------------------------------------

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


class PngUnsupported(ValueError):
    """A PNG the device decoder does not handle (interlaced, sub-byte, malformed)."""


def _png_chunks(data: bytes):
    if data[:8] != _PNG_SIG:
        raise PngUnsupported("not a PNG")
    pos = 8
    while pos + 8 <= len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        yield kind, data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if kind == b"IEND":
            return


def _unfilter(rows: torch.Tensor, ftype: torch.Tensor) -> torch.Tensor:
    """Undo PNG row filters. `rows` (h, W, bpp) int32 filtered bytes, `ftype` (h,) int32.

    A byte depends on its left, up and up-left neighbours (one pixel = bpp bytes back), so every pixel on
    one anti-diagonal x + y = d is independent: W + h − 1 steps. The buffer is skewed (pixel (y, x) at
    column x + y + 2, row y + 1) so each step reads and writes one column slice.
    """
    h, W, bpp = rows.shape
    D = W + h + 2
    dev = rows.device
    ys = torch.arange(h, device=dev)
    xs = torch.arange(W, device=dev)
    cols = xs[None, :] + ys[:, None] + 2  # (h, W)
    raw = torch.zeros((h, D, bpp), dtype=torch.int32, device=dev)
    raw[ys[:, None], cols] = rows
    valid = torch.zeros((h, D, 1), dtype=torch.int32, device=dev)
    valid[ys[:, None], cols] = 1
    out = torch.zeros((h + 1, D, bpp), dtype=torch.int32, device=dev)
    ft = ftype.to(dev)[:, None]
    for j in range(2, W + h + 1):
        a = out[1:, j - 1]
        b = out[:-1, j - 1]
        c = out[:-1, j - 2]
        pa, pb, pc = (b - c).abs(), (a - c).abs(), (a + b - 2 * c).abs()
        paeth = torch.where((pa <= pb) & (pa <= pc), a, torch.where(pb <= pc, b, c))
        pred = torch.where(ft == 1, a, torch.where(ft == 2, b, torch.where(ft == 3, (a + b) >> 1,
                           torch.where(ft == 4, paeth, torch.zeros_like(a)))))
        out[1:, j] = ((raw[:, j] + pred) & 255) * valid[:, j]
    return out[1 + ys[:, None], cols]


def decode_png(data: bytes, device) -> torch.Tensor:
    """(4, h, w) float32 premultiplied RGBA on `device`. Raises PngUnsupported for what it cannot decode."""
    ihdr = plte = trns = None
    idat: list[bytes] = []
    for kind, body in _png_chunks(data):
        if kind == b"IHDR":
            ihdr = body
        elif kind == b"PLTE":
            plte = body
        elif kind == b"tRNS":
            trns = body
        elif kind == b"IDAT":
            idat.append(body)
    if ihdr is None or len(ihdr) < 13:
        raise PngUnsupported("no IHDR")
    w, h, depth, ctype, _comp, _filt, interlace = struct.unpack(">IIBBBBB", ihdr[:13])
    if ctype not in _PNG_CHANNELS or depth not in (8, 16) or (ctype == 3 and depth != 8):
        raise PngUnsupported(f"bit depth {depth} colour type {ctype}")
    if interlace:
        raise PngUnsupported("interlaced")
    ch = _PNG_CHANNELS[ctype]
    bpp = ch * depth // 8
    try:
        raw = zlib.decompress(b"".join(idat))
    except zlib.error as exc:
        raise PngUnsupported(f"bad image data ({exc})") from exc
    stride = 1 + w * bpp
    if len(raw) < h * stride:
        raise PngUnsupported("truncated image data")
    host = np.frombuffer(raw, np.uint8, count=h * stride).reshape(h, stride)
    ftype = torch.from_numpy(host[:, 0].astype(np.int32))
    if int(ftype.max()) > 4:
        raise PngUnsupported("unknown row filter")
    dev = torch.device(device)
    rows = torch.from_numpy(host[:, 1:].copy()).to(dev).reshape(h, w, bpp).to(torch.int32)
    px = _unfilter(rows, ftype) if int(ftype.max()) > 0 else rows
    if depth == 16:
        px = px.reshape(h, w, ch, 2)
        samples = px[..., 0] * 256 + px[..., 1]
        top = 65535.0
    else:
        samples = px.reshape(h, w, ch)
        top = 255.0
    samples = samples.permute(2, 0, 1)  # (ch, h, w) int32
    if ctype == 3:
        if plte is None:
            raise PngUnsupported("palette image without PLTE")
        pal = torch.frombuffer(bytearray(plte), dtype=torch.uint8).reshape(-1, 3).to(dev).float() / 255
        alpha_tab = torch.ones(pal.shape[0], device=dev)
        if trns:
            n = min(len(trns), pal.shape[0])
            alpha_tab[:n] = torch.frombuffer(bytearray(trns[:n]), dtype=torch.uint8).to(dev).float() / 255
        idx = samples[0].long().clamp(max=pal.shape[0] - 1)
        rgb = pal[idx].permute(2, 0, 1)
        alpha = alpha_tab[idx][None]
    else:
        vals = samples.float() / top
        if ctype in (0, 4):
            rgb = vals[:1].expand(3, h, w)
        else:
            rgb = vals[:3]
        if ctype in (4, 6):
            alpha = vals[-1:]
        elif trns is not None and len(trns) >= 2 * (1 if ctype == 0 else 3):
            key = torch.tensor(struct.unpack(f">{1 if ctype == 0 else 3}H", trns[:2 * (1 if ctype == 0 else 3)]),
                               device=dev, dtype=torch.int32)
            alpha = (samples != key[:, None, None]).any(dim=0, keepdim=True).float()
        else:
            alpha = torch.ones((1, h, w), device=dev)
    return torch.cat((rgb * alpha, alpha), dim=0).contiguous()


def _pil_rgba(path: Path, device) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGBA"), np.float32) / 255
    t = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    return torch.cat((t[:3] * t[3:4], t[3:4]), dim=0).contiguous()


def load_rgba(path, device, policy=None) -> torch.Tensor:
    """A layer image as (4, h, w) premultiplied float32 on `device`, through the GPU policy's `photo` seam.

    PNG decodes on the device; JPEG takes the photo path (`core.decode_rgb`); anything else is PIL.
    """
    from montaj.render.core import _is_jpeg, decode_rgb

    path = Path(path)
    dev = torch.device(device)
    if _is_jpeg(str(path)):
        rgb = torch.from_numpy(decode_rgb(str(path), dev, policy)).permute(2, 0, 1).to(dev)
        return torch.cat((rgb, torch.ones_like(rgb[:1])), dim=0).contiguous()
    data = path.read_bytes()
    reason = None
    if dev.type != "cuda":
        reason = "device is not cuda"
    elif data[:8] != _PNG_SIG:
        reason = f"{path.suffix.lstrip('.').upper() or 'this format'} has no device decoder"
    else:
        try:
            out = decode_png(data, dev)
            if policy is not None:
                policy.gpu("photo")
            return out
        except PngUnsupported as exc:
            reason = f"PNG {exc}"
    if policy is not None:
        policy.cpu("photo", f"decoding layer image {path.name} ({reason})",
                   "save it as a non-interlaced 8-bit PNG (or JPEG)")
    return _pil_rgba(path, dev)
