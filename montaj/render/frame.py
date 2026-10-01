"""Evaluate one FramePlan into a finished RGB frame.

The renderer reads only the plan, the shots it names, and asset files. Photos are
loaded once; a still's pixels do not depend on time, so the composited still is
cached too. Wall frames are not: the camera moves.
"""
from __future__ import annotations

from pathlib import Path

import torch

from montaj.project import find_photo
from montaj.render.canvas import Canvas
from montaj.render.core import Photo, fit_s, still
from montaj.render.finish import LOOKS, finish
from montaj.render.transitions import fade, swirl, whip
from montaj.render.wall import build_wall, wall_camera, wall_frame
from montaj.timeline import Scene, Still, Timeline, Trans, Wall


def _as_crop(crop: tuple[float, ...] | None) -> tuple[int, int, int, int] | None:
    """Photo slices need ints. The schema stores whole source pixels as floats."""
    if crop is None:
        return None
    return (int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3]))


class Renderer:
    """`frame(f)` samples the plan's scene across the motion-blur shutter, then finishes it."""

    def __init__(self, tl: Timeline, spec_dir: Path, cv: Canvas):
        self.tl = tl
        self.cv = cv
        self._assets = Path(spec_dir) / tl.spec.assets
        self._photos: dict[tuple, Photo] = {}
        self._walls: dict[int, object] = {}
        self._stills: dict[int, torch.Tensor] = {}

    def _photo(self, stem: str, crop: tuple[int, int, int, int] | None, card: bool) -> Photo:
        # Stills are flat; wall prints are cards. The same file is often both.
        key = (stem, crop, card)
        hit = self._photos.get(key)
        if hit is None:
            hit = Photo(self.cv, str(find_photo(self._assets, stem)), crop=crop, card=card)
            self._photos[key] = hit
        return hit

    def _load_photo(self, stem: str, crop, card: bool) -> Photo:
        return self._photo(str(stem), _as_crop(tuple(crop) if crop else None), bool(card))

    def _wall(self, index: int):
        hit = self._walls.get(index)
        if hit is None:
            spec = self.tl.shots[index].spec.wall
            if spec is None:
                raise ValueError(f"shot {index} has no wall")
            hit = build_wall(self.cv, spec.model_dump(), self._load_photo)
            self._walls[index] = hit
        return hit

    def _still_image(self, index: int) -> torch.Tensor:
        hit = self._stills.get(index)
        if hit is None:
            sh = self.tl.shots[index].spec
            if sh.photo is None:
                raise ValueError(f"shot {index} is not a still")
            photo = self._photo(sh.photo, _as_crop(sh.crop), False)
            hit = still(self.cv, photo, focus=sh.focus, zoom=sh.zoom)
            self._stills[index] = hit
        return hit

    def _screen_scale(self, index: int, photo: Photo, focus: tuple[float, float]) -> float:
        """Device px per photo px. A spec zoom is design px, so it scales with Canvas.k."""
        zoom = self.tl.shots[index].spec.zoom
        if zoom is not None:
            return float(zoom) * self.cv.k
        return fit_s(self.cv, photo, focus, 0.0) * 1.005

    def _swirl_center(self, tr: Trans) -> tuple[float, float]:
        # center is a point in the outgoing photo. That shot's focus sits at screen centre.
        scene: Scene = tr.a
        while isinstance(scene, Trans):
            scene = scene.a
        sh = self.tl.shots[scene.shot].spec
        if sh.photo is None:
            return (self.cv.W / 2, self.cv.H / 2)
        photo = self._photo(sh.photo, _as_crop(sh.crop), False)
        x0, y0, x1, y1 = photo.crop
        focus = sh.focus if sh.focus is not None else ((x0 + x1) / 2, (y0 + y1) / 2)
        center = tr.center if tr.center is not None else focus
        scale = self._screen_scale(scene.shot, photo, focus)
        return (
            self.cv.W / 2 + (center[0] - focus[0]) * scale,
            self.cv.H / 2 + (center[1] - focus[1]) * scale,
        )

    def _eval(self, scene: Scene, t: float) -> torch.Tensor:
        if isinstance(scene, Still):
            return self._still_image(scene.shot)
        if isinstance(scene, Wall):
            spec = self.tl.shots[scene.shot].spec.wall
            if spec is None:
                raise ValueError(f"shot {scene.shot} has no wall")
            wall = self._wall(scene.shot)
            # t0 is the wall shot's start, fade included. Camera keys share that clock.
            T, s = wall_camera(wall, spec.camera.model_dump(), t - scene.t0)
            return wall_frame(self.cv, wall, T, s)
        x = (t - scene.t0) / scene.dur
        a = self._eval(scene.a, t)
        b = self._eval(scene.b, t)
        if scene.kind == "fade":
            return fade(a, b, x)
        if scene.kind == "whip":
            return whip(self.cv, a, b, x, scene.axis)
        if scene.kind == "swirl":
            return swirl(self.cv, a, b, x, self._swirl_center(scene))
        raise ValueError(f"transition: unknown {scene.kind!r}")

    def frame(self, f: int) -> torch.Tensor:
        """(3, H, W) in 0..1. Sub-frame times share the integer frame's scene."""
        plan = self.tl.plan(f)
        n = self.tl.spec.video.motion_blur
        with torch.no_grad():
            if n <= 1:
                # The shutter formula divides by n-1; one sample is the frame time itself.
                acc = self._eval(plan.scene, float(f))
            else:
                acc = None
                for k in range(n):
                    t = f + (k / (n - 1) - 0.5) * 0.5
                    sample = self._eval(plan.scene, t)
                    acc = sample if acc is None else acc + sample
                acc = acc / n
            look = LOOKS.get(self.tl.spec.video.look)
            if look is None:
                have = ", ".join(LOOKS)
                raise ValueError(f"look: unknown {self.tl.spec.video.look!r} (have: {have})")
            return finish(self.cv, acc, f, look, plan.flashes, plan.leaks, plan.intro, plan.outro)
