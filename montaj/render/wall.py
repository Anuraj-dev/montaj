"""Wall of prints + recipe camera tracks. Port of film.build_wall/wall_bg + film2.build_wall."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .core import EASE, camera, draw, key, keys_s


def wpt(print: dict, src: tuple) -> tuple:
    r = math.radians(print["rot"])
    dx, dy = (src[0] - print["c"][0]) * print["p"], (src[1] - print["c"][1]) * print["p"]
    return (print["pos"][0] + math.cos(r) * dx - math.sin(r) * dy,
            print["pos"][1] + math.sin(r) * dx + math.cos(r) * dy)


class WallScene:
    def __init__(self, order: list, light):
        self.order = order
        self.light = light
        self.by_name = {L["n"]: L for L in order}

    def __getitem__(self, k: str) -> dict:
        return self.by_name[k]


def build_wall(cv, wall_spec: dict, photo_loader) -> WallScene:
    gx, gy = str(wall_spec.get("grid", "1500x1950")).lower().split("x")
    GX, GY = float(gx), float(gy)
    rng = np.random.default_rng(wall_spec.get("seed", 5))
    j = wall_spec.get("jitter", 70)
    rot = wall_spec.get("rot", 4)
    ph = wall_spec.get("print_height", 1300)
    light_spec = wall_spec.get("light", {})
    lc, lr = light_spec.get("center", [-1500, 0]), light_spec.get("radius", [4200, 3000])
    cx0, cy0, rx, ry = float(lc[0]), float(lc[1]), float(lr[0]), float(lr[1])

    def light(X, Y):
        return (0.5 + 0.5 * torch.exp(-((X - cx0) ** 2 / rx**2 + (Y - cy0) ** 2 / ry**2)))[None]

    layers = []
    for pr in wall_spec["prints"]:
        n = str(pr["photo"])
        cx, cy = pr.get("cell", [0, 0])
        crop = pr.get("crop")
        img = photo_loader(n, tuple(crop) if crop else None, True)
        x0, y0, x1, y1 = img.crop
        if pr.get("scale") is not None:
            p = float(pr["scale"])
        else:
            p = ph / max(y1 - y0, (x1 - x0) * 1.15)
        if pr.get("straight"):  # key prints: no jitter, no rng draws
            jx, jy, r = 0.0, 0.0, 0.0
        else:
            jx, jy = float(rng.uniform(-j, j)), float(rng.uniform(-j, j))
            r = float(rng.uniform(-rot, rot))
        layers.append(dict(n=n, img=img, pos=(cx * GX + jx, cy * GY + jy),
                           rot=r, p=p, c=((x0 + x1) / 2, (y0 + y1) / 2)))
    return WallScene(layers, light)


def wall_bg(cv):
    base = torch.tensor([0.075, 0.058, 0.048], device=cv.device)[:, None, None]

    def bg(cam):
        C, (dx, dy, dz) = cam
        t = (0 - C[2]) / dz
        X, Y = C[0] + t * dx, C[1] + t * dy
        g = torch.stack([torch.remainder(X / 2500, 2) - 1, torch.remainder(Y / 2500, 2) - 1], -1)[None]
        n = F.grid_sample(cv.wall_noise, g, mode="bilinear", padding_mode="reflection",
                          align_corners=False)[0, 0]
        r2 = (X**2 + Y**2) / (3600.0**2)
        return (base * (0.8 + 0.4 * n)[None] * (0.42 + 0.58 * torch.exp(-r2 * 0.55))[None]
                * (t > 0).float()[None])

    return bg


def wall_frame(cv, wall: WallScene, T: tuple, s: float):
    layers = [dict(L, light=wall.light) for L in wall.order]
    return draw(cv, camera(cv, T, s * cv.k), layers, wall_bg(cv))


def _parse_t(t) -> float:
    if isinstance(t, str):
        t = t.strip()
        return float(t[:-1]) if t.endswith("f") else float(t)
    return float(t)


def _resolve(wall: WallScene, keys: list, is_zoom: bool) -> list:
    out, cur = [], None
    for e in keys:
        fn = e.get("ease")
        fn = fn if (fn is None or callable(fn)) else EASE[str(fn)]
        if is_zoom:
            if e.get("value") is not None:
                cur = float(e["value"])
                if e.get("print") is not None:
                    cur /= wall.by_name[str(e["print"])]["p"]
        elif e.get("print") is not None and e.get("at") is not None:
            cur = wpt(wall.by_name[str(e["print"])], tuple(e["at"]))
        assert cur is not None, "first camera key must carry a value"
        out.append((_parse_t(e["t"]), cur, fn))
    return out


def wall_camera(wall: WallScene, camera_spec: dict, t: float) -> tuple:
    T = key(t, _resolve(wall, camera_spec["pos"], False))
    return T, keys_s(t, _resolve(wall, camera_spec["zoom"], True))
