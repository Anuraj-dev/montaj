"""Core primitives: easing/keys, Photo, camera, draw, still. Port of film.py / film2.py."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------- easing / keys
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def smooth(x: float) -> float:
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def cubic(x: float) -> float:
    x = clamp01(x)
    return 4 * x**3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def outc(x: float) -> float:
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def inc(x: float) -> float:
    x = clamp01(x)
    return x**3


def expo(x: float) -> float:
    x = clamp01(x)
    if x in (0, 1):
        return x
    return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2


def ramp(x: float, k: float = 0.75) -> float:
    x = clamp01(x)
    return x - k * math.sin(2 * math.pi * x) / (2 * math.pi)


def linear(x: float) -> float:
    return clamp01(x)


EASE = {"linear": linear, "smooth": smooth, "cubic": cubic, "outc": outc,
        "inc": inc, "expo": expo, "ramp": ramp}


def lerp(a: float, b: float, e: float) -> float:
    return a + (b - a) * e


def lerp2(a: tuple, b: tuple, e: float) -> tuple:
    return (lerp(a[0], b[0], e), lerp(a[1], b[1], e))


def lerps(a: float, b: float, e: float) -> float:
    return math.exp(lerp(math.log(a), math.log(b), e))


def key(t: float, keys: list) -> object:
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0, _), (t1, v1, e) in zip(keys, keys[1:]):
        if t <= t1:
            x = (e or smooth)((t - t0) / (t1 - t0))
            return tuple(lerp(a, b, x) for a, b in zip(v0, v1)) if isinstance(v0, tuple) else lerp(v0, v1, x)
    return keys[-1][1]


def keys_s(t: float, keys: list) -> float:
    return math.exp(key(t, [(a, math.log(v), e) for a, v, e in keys]))


# ---------------------------------------------------------------- photo
class Photo:
    """film.Img minus the depth channel (film2 never uses relief)."""

    def __init__(self, cv, path: str, crop: tuple | None = None, card: bool = False, pad: int = 90):
        im = Image.open(path).convert("RGB")
        w, h = im.size
        x0, y0, x1, y1 = crop or (0, 0, w, h)
        self.crop = (x0, y0, x1, y1)
        rgb = np.asarray(im, np.float32) / 255
        if not card:
            a = np.zeros((h, w), np.float32)
            a[y0:y1, x0:x1] = 1
            self.off = (0.0, 0.0)
            tex = np.dstack([rgb * a[..., None], a])
            self.th, self.tw = h, w
        else:  # print: cream border + soft drop shadow around the cropped photo
            cw, ch = x1 - x0, y1 - y0
            b = int(0.035 * min(cw, ch))
            P = pad
            TW, TH = cw + 2 * b + 2 * P, ch + 2 * b + 2 * P
            card_a = np.zeros((TH, TW), np.float32)
            card_a[P:P + ch + 2 * b, P:P + cw + 2 * b] = 1
            sh = Image.new("L", (TW, TH), 0)
            ImageDraw.Draw(sh).rectangle((P + 10, P + 28, P + cw + 2 * b + 10, P + ch + 2 * b + 28), fill=255)
            sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(P / 2.5)), np.float32) / 255 * 0.7
            col = np.zeros((TH, TW, 3), np.float32)
            col[:] = (0.95, 0.92, 0.86)
            col[P + b:P + b + ch, P + b:P + b + cw] = rgb[y0:y1, x0:x1]
            a = card_a + (1 - card_a) * sh
            rgb = col * card_a[..., None]  # shadow is premultiplied black
            self.off = (P + b - x0, P + b - y0)
            tex = np.dstack([rgb, a])
            self.th, self.tw = TH, TW
        self.tex = torch.from_numpy(tex).permute(2, 0, 1)[None].to(cv.device)

    def sample(self, sx, sy, chans, mode):
        gx = (sx + self.off[0]) * 2 / self.tw - 1
        gy = (sy + self.off[1]) * 2 / self.th - 1
        g = torch.stack([gx, gy], -1)[None]
        return F.grid_sample(self.tex[:, chans], g, mode=mode, padding_mode="zeros", align_corners=False)[0]


# ---------------------------------------------------------------- camera / draw
def camera(cv, T: tuple, s: float, yaw: float = 0.0, pitch: float = 0.0):
    h = cv.foc / s
    y, th = math.radians(yaw), math.radians(pitch)
    x_ax = np.array([math.cos(y), math.sin(y), 0.0])
    up = np.array([math.sin(y), -math.cos(y), 0.0])
    z_ax = -math.cos(th) * np.array([0, 0, 1.0]) + math.sin(th) * up
    y_ax = np.cross(x_ax, z_ax)
    C = np.array([T[0], T[1], 0.0]) - h * z_ax
    dirs = [float(x_ax[i]) * cv.GA + float(y_ax[i]) * cv.GB + float(z_ax[i]) for i in range(3)]
    return C, dirs


def draw(cv, cam, layers: list, bg=None):
    C, (dx, dy, dz) = cam
    out = torch.zeros(3, cv.H, cv.W, device=cv.device) if bg is None else bg(cam)
    for L in layers:
        img = L["img"]
        px, py = L.get("pos", (0, 0))
        rot = math.radians(L.get("rot", 0))
        p = L.get("p", 1.0)
        cx, cy = L.get("c", (0, 0))
        cr, sr = math.cos(-rot), math.sin(-rot)
        t = (0.0 - C[2]) / dz
        X = C[0] + t * dx - px
        Y = C[1] + t * dy - py
        sx = cx + (cr * X - sr * Y) / p
        sy = cy + (sr * X + cr * Y) / p
        rgba = img.sample(sx, sy, [0, 1, 2, 3], "bicubic").clamp(0, 1)
        rgba = rgba * (t > 0).float()
        if "light" in L:
            rgba[:3] *= L["light"](C[0] + t * dx, C[1] + t * dy)
        op = L.get("op", 1.0)
        out = out * (1 - rgba[3:4] * op) + rgba[:3] * op
    return out


def fit_s(cv, photo: Photo, T: tuple, yaw: float = 0.0) -> float:
    x0, y0, x1, y1 = photo.crop
    c, s_ = math.cos(math.radians(-yaw)), math.sin(math.radians(-yaw))
    need = 0.0
    for ux, uy in ((-cv.W / 2, -cv.H / 2), (cv.W / 2, -cv.H / 2), (-cv.W / 2, cv.H / 2), (cv.W / 2, cv.H / 2)):
        rx, ry = c * ux - s_ * uy, s_ * ux + c * uy
        for r, lo, hi, t in ((rx, x0, x1, T[0]), (ry, y0, y1, T[1])):
            room = (t - lo) if r < 0 else (hi - t)
            need = max(need, abs(r) / max(room, 1e-3))
    return need


def still(cv, photo: Photo, focus: tuple | None = None, zoom: float | None = None):
    """film2.still; zoom is design px per photo px (scaled by Canvas.k)."""
    x0, y0, x1, y1 = photo.crop
    T = ((x0 + x1) / 2, (y0 + y1) / 2) if focus is None else focus
    s = zoom * cv.k if zoom is not None else fit_s(cv, photo, T, 0) * 1.005
    return draw(cv, camera(cv, T, s), [dict(img=photo)])
