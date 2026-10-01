"""Shot framing. Ports film.html fitCover / Photo (lines 113–163).

Screen geometry is device px. Arguments named in design px (`px`, `py`, blur radii,
caption sizes) are multiplied by `cv.k`. `fx, fy` are fractions of the crop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .core import cubic

_LUMA = (0.2126, 0.7152, 0.0722)  # W3C saturate()
_CREAM = (246 / 255, 239 / 255, 226 / 255)  # .frame background, film.html line 21


def _clamp(x: float, a: float, b: float) -> float:
    """film.html line 68. `b < a` still returns `b` (JS Math.min/max order)."""
    return min(b, max(a, x))


def sine(x: float) -> float:
    """film.html line 73 `es`. Smoothstep is a different curve."""
    x = max(0.0, min(1.0, x))
    return 0.5 - 0.5 * math.cos(math.pi * x)


def drift(t: float, v0: float, v1: float, zoom: tuple[float, float] = (1.0, 1.0),
          pan: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float, float]:
    """film.html lines 140–141. Zoom eases with `es`; pan is linear in k, about 0."""
    k = 0.0 if v1 == v0 else _clamp((t - v0) / (v1 - v0), 0.0, 1.0)
    z = zoom[0] + (zoom[1] - zoom[0]) * sine(k)
    return z, pan[0] * (k - 0.5), pan[1] * (k - 0.5)


def focus_frac(photo, focus: tuple[float, float] | None = None) -> tuple[float, float]:
    """`(focus − crop0) / crop_size`, default the crop centre."""
    x0, y0, x1, y1 = photo.crop
    if focus is None:
        return 0.5, 0.5
    return (focus[0] - x0) / (x1 - x0), (focus[1] - y0) / (y1 - y0)


def _fit(cv, nw: float, nh: float, fx: float, fy: float, z: float, px: float, py: float):
    """film.html fitCover lines 114–119. Returns scale (device/source), drawn size, translation."""
    s = max(cv.W / nw, cv.H / nh) * z
    w, h = nw * s, nh * s
    ox = _clamp(fx * w - cv.W / 2, 0.0, w - cv.W)
    oy = _clamp(fy * h - cv.H / 2, 0.0, h - cv.H)
    return s, w, h, -ox + px * cv.k, -oy + py * cv.k


def _sample_crop(photo, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Bilinear sample of the crop. `u, v` are full-image left-edge coordinates.

    Dest pixels that fall inside the image clamp the kernel to the edge (Skia);
    the caller zeros pixels whose centre is outside the element.
    """
    x0, y0, x1, y1 = photo.crop
    crop = photo.tex[0, :3, y0:y1, x0:x1].contiguous()
    _c, nh, nw = crop.shape
    ul = u - x0
    vl = v - y0
    grid = torch.stack((ul * 2 / nw - 1, vl * 2 / nh - 1), -1)[None]
    return F.grid_sample(crop[None], grid, mode="bilinear", padding_mode="border",
                         align_corners=False)[0]


def _render_cover(cv, photo, fx, fy, z, px, py, margin: int = 0) -> torch.Tensor:
    x0, y0, x1, y1 = photo.crop
    nw, nh = x1 - x0, y1 - y0
    s, w, h, tx, ty = _fit(cv, nw, nh, fx, fy, z, px, py)
    H, W = cv.H + 2 * margin, cv.W + 2 * margin
    ys = torch.arange(H, device=cv.device, dtype=torch.float32) + 0.5 - margin
    xs = torch.arange(W, device=cv.device, dtype=torch.float32) + 0.5 - margin
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    inside = (gx >= tx) & (gx < tx + w) & (gy >= ty) & (gy < ty + h)
    sampled = _sample_crop(photo, x0 + (gx - tx) / s, y0 + (gy - ty) / s)
    return torch.where(inside[None], sampled, torch.zeros_like(sampled))


def kenburns(cv, photo, fx: float, fy: float, z: float, px: float, py: float) -> torch.Tensor:
    """fitCover onto the canvas. No 1.005 overscan (film.html line 115). (3, H, W)."""
    return _render_cover(cv, photo, fx, fy, z, px, py).clamp(0, 1)


def _clip01(img: torch.Tensor) -> torch.Tensor:
    """Each CSS filter primitive clamps before the next one sees it."""
    return img.clamp(0, 1)


def tone(img: torch.Tensor, brightness: float = 1.0, saturate: float = 1.0,
         contrast: float = 1.0) -> torch.Tensor:
    """CSS filter in fixed order brightness → saturate → contrast, sRGB 0..1.

    `saturate` is the W3C matrix (luma .2126/.7152/.0722). Contrast is `(v−.5)·c+.5`.
    White through brightness 2 then contrast .5 is .75, because brightness clips to 1 first.
    """
    out = _clip01(img * float(brightness))
    s = float(saturate)
    lr, lg, lb = _LUMA
    r, g, b = out[0], out[1], out[2]
    rr, rg, rb = lr * (1 - s) + s, lg * (1 - s), lb * (1 - s)
    gr, gg, gb = lr * (1 - s), lg * (1 - s) + s, lb * (1 - s)
    br, bg, bb = lr * (1 - s), lg * (1 - s), lb * (1 - s) + s
    out = _clip01(torch.stack((rr * r + rg * g + rb * b, gr * r + gg * g + gb * b, br * r + bg * g + bb * b)))
    return _clip01((out - 0.5) * float(contrast) + 0.5)


def _gaussian(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable normalised gaussian, zero outside the tensor. σ in device px."""
    radius = max(1, int(math.ceil(sigma * 3)))
    x = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    c = img.shape[0]
    n = img[None]
    hx = F.conv2d(F.pad(n, (radius, radius, 0, 0)), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(hx, (0, 0, radius, radius)), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)[0]


def css_blur(img: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """`filter: blur(r)` with σ = r device px (film.html leaks line 38, frame bg line 131).

    σ > 8 device px is blurred on a half-resolution pyramid. The kernel is the same
    gaussian either way; the pyramid is only legal because a wide gaussian is smooth.
    """
    sigma = float(sigma_px)
    if sigma < 1e-3:
        return img
    work = img
    while sigma > 8.0 and min(work.shape[-2:]) >= 8:
        _c, h, w = work.shape
        ph, pw = h & 1, w & 1
        if ph or pw:
            work = F.pad(work[None], (0, pw, 0, ph))[0]
        work = F.avg_pool2d(work[None], 2)[0]
        sigma *= 0.5
    work = _gaussian(work, max(sigma, 1e-3))
    while work.shape[-2] < img.shape[-2] or work.shape[-1] < img.shape[-1]:
        work = F.interpolate(work[None], scale_factor=2, mode="bilinear", align_corners=False)[0]
    return work[..., : img.shape[-2], : img.shape[-1]]


def frame_background(cv, photo) -> torch.Tensor:
    """film.html lines 131 and 145: cover at (.5, .5, 1.25), then blur 38, brightness .42, saturate 1.2.

    The image extends past the frame; the blur is applied to that larger layer and cropped,
    so the frame edge does not fade toward transparent.
    """
    margin = int(math.ceil(38 * cv.k * 3))
    big = _render_cover(cv, photo, 0.5, 0.5, 1.25, 0.0, 0.0, margin)
    blurred = css_blur(big, 38 * cv.k)
    view = blurred[:, margin:margin + cv.H, margin:margin + cv.W]
    return tone(view, brightness=0.42, saturate=1.2).clamp(0, 1)


def card_layout(cv, photo, caption) -> dict:
    """film.html lines 146–148. Lengths are device px."""
    x0, y0, x1, y1 = photo.crop
    nw, nh = x1 - x0, y1 - y0
    k = cv.k
    pad = 22 * k
    cap_h = (120 * k) if caption else pad
    s = min(900 * k / nw, 1260 * k / nh)
    iw, ih = nw * s, nh * s
    cw, ch = iw + pad * 2, ih + pad + cap_h
    left = (cv.W - cw) / 2
    top = (cv.H - ch) / 2 - 40 * k
    return {"pad": pad, "cap_h": cap_h, "s": s, "iw": iw, "ih": ih, "cw": cw, "ch": ch,
            "left": left, "top": top, "x0": x0, "y0": y0}


@dataclass(frozen=True)
class Card:
    """Axis-aligned card art in device px, before tilt. Shadow is applied in `card_frame`."""

    rgba: torch.Tensor          # premultiplied RGBA (4, Hc, Wc)
    cw: float
    ch: float
    left: float                 # untransformed screen position of the card's top-left
    top: float


def _paste_straight(base: torch.Tensor, sprite: torch.Tensor, x: float, y: float) -> None:
    """Source-over a straight-RGBA sprite onto a premultiplied card.

    Bilinear runs in premultiplied space. An opaque red texel beside transparency
    is premul red .5 at alpha .5; filtering straight RGB first would store .25.
    """
    _c, sh, sw = sprite.shape
    bh, bw = base.shape[-2:]
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = int(math.ceil(x + sw)), int(math.ceil(y + sh))
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(bw, x1), min(bh, y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return
    premul = torch.cat((sprite[:3] * sprite[3:4], sprite[3:4]), 0)
    ys = torch.arange(iy0, iy1, device=base.device, dtype=torch.float32) + 0.5 - y
    xs = torch.arange(ix0, ix1, device=base.device, dtype=torch.float32) + 0.5 - x
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((gx * 2 / sw - 1, gy * 2 / sh - 1), -1)[None]
    samp = F.grid_sample(premul[None], grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)[0]
    sp, sa = samp[:3], samp[3:4]
    dst = base[:, iy0:iy1, ix0:ix1]
    base[:, iy0:iy1, ix0:ix1] = torch.cat((sp + dst[:3] * (1 - sa), sa + dst[3:4] * (1 - sa)), 0)


def frame_card(cv, photo, caption, rasters) -> Card:
    """Polaroid card. film.html lines 21–23 and 146–151.

    `rasters` is the caption line box as straight RGBA (4, h, w) in device px
    (Great Vibes 64, #5a3b24, centred on the card width), or None. Its bottom
    edge sits 18 design px above the card bottom. The returned buffer is premultiplied.
    """
    L = card_layout(cv, photo, caption)
    cw, ch, pad = L["cw"], L["ch"], L["pad"]
    Wc, Hc = max(1, int(math.ceil(cw - 1e-4))), max(1, int(math.ceil(ch - 1e-4)))
    dev = cv.device
    xs = torch.arange(Wc, device=dev, dtype=torch.float32) + 0.5
    ys = torch.arange(Hc, device=dev, dtype=torch.float32) + 0.5
    ax = (torch.minimum(xs + 0.5, torch.tensor(cw, device=dev)) - torch.maximum(xs - 0.5, torch.tensor(0.0, device=dev))).clamp(0, 1)
    ay = (torch.minimum(ys + 0.5, torch.tensor(ch, device=dev)) - torch.maximum(ys - 0.5, torch.tensor(0.0, device=dev))).clamp(0, 1)
    alpha = ay[:, None] * ax[None, :]
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    iw, ih, s = L["iw"], L["ih"], L["s"]
    inside = (gx >= pad) & (gx < pad + iw) & (gy >= pad) & (gy < pad + ih)
    photo_rgb = _sample_crop(photo, L["x0"] + (gx - pad) / s, L["y0"] + (gy - pad) / s).clamp(0, 1)
    cream = torch.tensor(_CREAM, device=dev, dtype=torch.float32)[:, None, None]
    rgb = torch.where(inside[None], photo_rgb, cream)
    rgba = torch.cat((rgb * alpha, alpha[None]), 0)
    if caption and rasters is not None:
        sprite = rasters if rasters.device == dev else rasters.to(dev)
        _paste_straight(rgba, sprite.float(), (cw - sprite.shape[-1]) / 2, ch - 18 * cv.k - sprite.shape[-2])
    return Card(rgba, cw, ch, L["left"], L["top"])


def _erf_rect(x, y, x0, x1, y0, y1, sigma: float) -> torch.Tensor:
    """Integral of a unit gaussian over an axis-aligned rect. σ = box-shadow radius / 2."""
    s = sigma * math.sqrt(2)
    ax = 0.5 * (torch.erf((x1 - x) / s) - torch.erf((x0 - x) / s))
    ay = 0.5 * (torch.erf((y1 - y) / s) - torch.erf((y0 - y) / s))
    return ax * ay


def card_frame(cv, card: Card, bg: torch.Tensor, z: float, tilt_deg: float,
               px: float, py: float) -> torch.Tensor:
    """film.html line 150: translate(p) scale(z·.97) rotate(tilt) about the card centre.

    Box-shadow `0 40px 120px rgba(0,0,0,.7), 0 0 0 1px rgba(0,0,0,.2)` (line 21)
    is evaluated in local px so it rotates with the card. σ = 120/2.
    """
    scale = z * 0.97
    th = math.radians(tilt_deg)
    cos, sin = math.cos(th), math.sin(th)
    ccx = card.left + card.cw / 2 + px * cv.k
    ccy = card.top + card.ch / 2 + py * cv.k
    sx = cv.XX + 0.5
    sy = cv.YY + 0.5
    dx = (sx - ccx) / scale
    dy = (sy - ccy) / scale
    lx = card.cw / 2 + cos * dx + sin * dy
    ly = card.ch / 2 - sin * dx + cos * dy
    tex = card.rgba
    _c, th_, tw = tex.shape
    grid = torch.stack((lx * 2 / tw - 1, ly * 2 / th_ - 1), -1)[None]
    samp = F.grid_sample(tex[None], grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)[0]
    premul, a = samp[:3], samp[3:4]
    k = cv.k
    # Shadow rect is the card shifted down 40 design px, blurred with σ = 60.
    a_blur = 0.7 * _erf_rect(lx, ly, 0.0, card.cw, 40 * k, card.ch + 40 * k, 60 * k)
    spread = 1 * k
    # 1px spread ring, no blur. Coverage of a 1px-local square.
    half = 0.5 / max(scale, 1e-6)
    def cov(x0, x1, y0, y1):
        ix = (torch.clamp(lx + half, max=x1) - torch.clamp(lx - half, min=x0)).clamp(min=0)
        iy = (torch.clamp(ly + half, max=y1) - torch.clamp(ly - half, min=y0)).clamp(min=0)
        return (ix / (2 * half)) * (iy / (2 * half))
    a_ring = 0.2 * cov(-spread, card.cw + spread, -spread, card.ch + spread)
    a_s = a_ring + a_blur * (1 - a_ring)
    # Black shadows under the card, then the card, all source-over on bg.
    return (premul + (1 - a) * (1 - a_s) * bg).clamp(0, 1)


def morph_mask(cv, x: float, center: tuple[float, float]) -> torch.Tensor:
    """film.html lines 154–156. `x` is raw progress; `center` is a screen fraction.

    `r = inOutCubic(x) · 1900` design px. Solid inside `r−380`, transparent at `r`.
    `r >= 1900` is an unmasked layer (`mask-image: none`).
    """
    k = cubic(float(x))
    r = k * 1900.0 * cv.k
    if r <= 0:
        return torch.zeros(1, cv.H, cv.W, device=cv.device)
    if r >= 1900.0 * cv.k - 1e-3:
        return torch.ones(1, cv.H, cv.W, device=cv.device)
    cx = float(center[0]) * cv.W
    cy = float(center[1]) * cv.H
    dist = torch.sqrt((cv.XX + 0.5 - cx) ** 2 + (cv.YY + 0.5 - cy) ** 2)
    start = max(0.0, r - 380.0 * cv.k)
    span = max(r - start, 1e-6)
    return ((r - dist) / span).clamp(0, 1)[None]


def pulse_heartbeat(phase: float) -> float:
    """film.html line 295. `phase` is the 0..1 position inside the beat."""
    b = phase
    scale = 1.0 + 0.035 * math.exp(-18 * b)
    if b > 0.22:
        scale += 0.02 * math.exp(-18 * (b - 0.22))
    return scale
