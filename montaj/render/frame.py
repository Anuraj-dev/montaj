"""Evaluate one FramePlan into a finished RGB frame.

The renderer reads only the plan, the shots it names, and asset files. Photos,
backgrounds, cards, stills, walls, clips and captions are cached on the Renderer
up to a fixed cap and loaded again on a miss; a reload reads the same bytes as
the first load. `begin_segment` drops entries the segment's frames cannot read,
so a wall keeps its prints only while it is on screen. An M1 still (no drift, tone, frame, morph, or pulse) is time-flat,
so its composite is cached. M2 framing is a function of t. Text rasters are built
on the first frame that needs them, one batch per kind, under ``<project>/build/text/``.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from montaj.project import find_clip, find_photo
from montaj.render.canvas import Canvas
from montaj.render.clip import ClipSource, window_key
from montaj.render.core import Photo, fit_s, sine, still
from montaj.render.finish import LOOKS, finish
from montaj.render.shots import (
    card_frame,
    css_blur,
    drift,
    focus_frac,
    frame_background,
    frame_card,
    kenburns,
    morph_mask,
    pulse_heartbeat,
    tone,
)
from montaj.render.text import (
    RasterImage,
    RasterLine,
    RasterSet,
    Sub,
    TextBlock,
    TextLine,
    Unit,
    draw_text,
    raster_caption,
    raster_lines,
    raster_subs,
)
from montaj.render.transitions import fade, swirl, whip
from montaj.render.wall import build_wall, wall_camera, wall_frame
from montaj.spec import _dur_frames
from montaj.timeline import Blank, Clip, Scene, Still, Timeline, Trans, Wall

# film.html 155: an omitted morph centre is (cx ?? .5, cy ?? .45). The spec leaves it unset.
_MORPH_CENTER = (0.5, 0.45)

# A transition composites two shots, and motion blur samples that pair again, so
# the whole set has to stay resident for the frame. A framed morph holds its photo,
# the morph photo, and a background and a card for each: 4 photos, 4 backgrounds,
# 4 cards. Caps are that set plus one more shot — 8, 6, 6 — so a swirl re-read or
# a card-print variant touched in the same frame does not push the pair out.
# Stills are the two M1 endpoints plus the same slack (4). Walls and clip sources
# are the two endpoints (2). Captions are the sprites those cards paste (8).
_PHOTO_CAP = 8
_BG_CAP = 6
_CARD_CAP = 6
_STILL_CAP = 4
_WALL_CAP = 2
_CLIP_CAP = 2
_CAPTION_CAP = 8


class _LRU:
    """Per-renderer bound. Evicted entries are not kept anywhere else in the cache."""

    def __init__(self, cap: int):
        self.cap = cap
        self._data: OrderedDict = OrderedDict()

    def get(self, key, default=None):
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = value
            return
        while len(self._data) >= self.cap:
            self._data.popitem(last=False)
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __delitem__(self, key) -> None:
        del self._data[key]


def _cache_caps(overrides: dict[str, int] | None) -> dict[str, int]:
    caps = {
        "photos": _PHOTO_CAP,
        "bgs": _BG_CAP,
        "cards": _CARD_CAP,
        "stills": _STILL_CAP,
        "walls": _WALL_CAP,
        "clips": _CLIP_CAP,
        "captions": _CAPTION_CAP,
    }
    if not overrides:
        return caps
    unknown = sorted(set(overrides) - set(caps))
    if unknown:
        raise ValueError(f"cache cap: unknown {', '.join(unknown)}")
    for name, cap in overrides.items():
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError(f"cache cap {name}: need an integer ≥ 1, got {cap!r}")
        caps[name] = cap
    return caps


def _rgb(raw: str) -> tuple[float, float, float]:
    text = str(raw).strip()
    if len(text) == 7 and text[0] == "#":
        try:
            n = int(text[1:], 16)
        except ValueError:
            n = -1
        if n >= 0:
            return ((n >> 16) & 255) / 255.0, ((n >> 8) & 255) / 255.0, (n & 255) / 255.0
    raise ValueError(f"background: expected #rrggbb, got {raw!r}")


def _m2(sh) -> bool:
    return sh.drift is not None or sh.tone is not None or sh.frame is not None or sh.morph is not None or sh.pulse is not None


def _scale_about(cv, img: torch.Tensor, scale: float) -> torch.Tensor:
    """CSS `transform: scale(s)` about the stage centre. s > 1 samples a smaller window."""
    inv = 1.0 / scale
    th = torch.tensor([[inv, 0.0, 0.0], [0.0, inv, 0.0]], device=img.device, dtype=img.dtype)[None]
    grid = F.affine_grid(th, (1, img.shape[0], cv.H, cv.W), align_corners=False)
    return F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]


def _straight(premul: torch.Tensor) -> torch.Tensor:
    """Caption rasters are stored premultiplied; the card pastes straight RGBA."""
    alpha = premul[3:4]
    rgb = torch.where(alpha > 1e-4, premul[:3] / alpha.clamp(min=1e-4), torch.zeros_like(premul[:3]))
    return torch.cat((rgb, alpha), 0)


def _move_line(line: RasterLine, device) -> RasterLine:
    def unit(u: Unit) -> Unit:
        return Unit(
            u.text, u.fill.to(device), None if u.shadow is None else u.shadow.to(device),
            u.box, u.shadow_box, u.layout, u.gold,
        )
    shadow = None if line.line_shadow is None else line.line_shadow.to(device)
    return RasterLine(tuple(unit(u) for u in line.units), line.line_box, shadow, line.line_shadow_box, line.scale, line.gold)


def _move_image(image: RasterImage, device) -> RasterImage:
    return RasterImage(image.fill.to(device), image.box, image.bottom)


class Renderer:
    """`frame(f)` samples the plan's scene across the motion-blur shutter, then finishes it."""

    def __init__(self, tl: Timeline, spec_dir: Path, cv: Canvas, *, cache_caps: dict[str, int] | None = None):
        self.tl = tl
        self.cv = cv
        self._dir = Path(spec_dir)
        self._assets = self._dir / tl.spec.assets
        caps = _cache_caps(cache_caps)
        self._photos: _LRU = _LRU(caps["photos"])
        self._walls: _LRU = _LRU(caps["walls"])
        self._stills: _LRU = _LRU(caps["stills"])
        self._clips: _LRU = _LRU(caps["clips"])
        self._bgs: _LRU = _LRU(caps["bgs"])
        self._cards: _LRU = _LRU(caps["cards"])
        self._bg: torch.Tensor | None = None
        self._rasters_ready = False
        self._blocks: tuple = ()
        self._subs: tuple = ()
        self._rasters: RasterSet | None = None
        self._captions: _LRU = _LRU(caps["captions"])
        self._shot: int | None = None
        self._owners: dict[str, dict[tuple, set[int]]] = {"photos": {}, "bgs": {}, "cards": {}}

    def begin_segment(self, shots) -> None:
        """Drop cached resources this segment's frames cannot read.

        `shots` is the segment's own index plus every neighbour those frames
        composite. What remains is still bounded by the LRU caps. A reload later
        reads the same bytes.
        """
        needed = frozenset(int(i) for i in shots)
        self._drop_indexed(self._walls, needed)
        self._drop_indexed(self._stills, needed)
        self._drop_clips(needed)
        self._drop_captions(needed)
        self._drop_owned("photos", self._photos, needed)
        self._drop_owned("bgs", self._bgs, needed)
        self._drop_owned("cards", self._cards, needed)

    def _note(self, kind: str, key: tuple) -> None:
        shot = self._shot
        if shot is None:
            return
        self._owners[kind].setdefault(key, set()).add(shot)

    def _drop_indexed(self, cache: _LRU, needed: frozenset[int]) -> None:
        for key in [key for key in cache if key not in needed]:
            del cache[key]

    def _drop_owned(self, kind: str, cache: _LRU, needed: frozenset[int]) -> None:
        owners = self._owners[kind]
        for key in list(owners):
            if key not in cache:
                del owners[key]
        for key in [key for key in cache if owners.get(key, set()).isdisjoint(needed)]:
            del cache[key]
            owners.pop(key, None)

    def _clip_request(self, index: int) -> tuple[str, float, int] | None:
        shot = self.tl.shots[index]
        sh = shot.spec
        if not sh.clip:
            return None
        extra = self._outgoing_frames(index)
        nframes = (shot.end - shot.start) + int(math.ceil(extra - 1e-6))
        clip_in = 0.0 if sh.clip_in is None else self._frames(sh.clip_in) / self.tl.fps
        return sh.clip, clip_in, nframes

    def _drop_clips(self, needed: frozenset[int]) -> None:
        """Close windows this segment will not read. Same stem, different clip_in, is another window."""
        wanted: dict[str, set[tuple]] = {}
        for index in needed:
            req = self._clip_request(index)
            if req is None:
                continue
            stem, clip_in, nframes = req
            wanted.setdefault(stem, set()).add(window_key(clip_in, self.cv.W, self.cv.H, nframes))
        for stem in [stem for stem in self._clips if stem not in wanted]:
            src = self._clips.get(stem)
            del self._clips[stem]
            if src is not None:
                src.close()
        for stem, keys in wanted.items():
            src = self._clips.get(stem)
            if src is not None:
                src.keep_windows(keys)

    def _drop_captions(self, needed: frozenset[int]) -> None:
        texts: set[str] = set()
        for i in needed:
            frame = self.tl.shots[i].spec.frame
            if frame is not None and frame.caption:
                texts.add(frame.caption)
        for key in [key for key in self._captions if key not in texts]:
            del self._captions[key]

    def _photo(self, stem: str, crop: tuple[int, int, int, int] | None, card: bool) -> Photo:
        # Stills are flat; wall prints are cards. The same file is often both.
        key = (stem, crop, card)
        hit = self._photos.get(key)
        if hit is None:
            hit = Photo(self.cv, str(find_photo(self._assets, stem)), crop=crop, card=card)
            self._photos[key] = hit
        self._note("photos", key)
        return hit

    def _load_photo(self, stem: str, crop: tuple[int, ...] | None, card: bool) -> Photo:
        box = None if not crop else (crop[0], crop[1], crop[2], crop[3])
        return self._photo(str(stem), box, bool(card))

    def _wall(self, index: int):
        self._shot = index
        hit = self._walls.get(index)
        if hit is None:
            spec = self.tl.shots[index].spec.wall
            if spec is None:
                raise ValueError(f"shot {index} has no wall")
            hit = build_wall(self.cv, spec.model_dump(), self._load_photo)
            self._walls[index] = hit
        return hit

    def _still_image(self, index: int) -> torch.Tensor:
        self._shot = index
        hit = self._stills.get(index)
        if hit is None:
            sh = self.tl.shots[index].spec
            if sh.photo is None:
                raise ValueError(f"shot {index} is not a still")
            photo = self._photo(sh.photo, sh.crop, False)
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
        if isinstance(scene, Blank) or not isinstance(scene, Still):
            return (self.cv.W / 2, self.cv.H / 2)
        sh = self.tl.shots[scene.shot].spec
        if sh.photo is None:
            return (self.cv.W / 2, self.cv.H / 2)
        self._shot = scene.shot
        photo = self._photo(sh.photo, sh.crop, False)
        x0, y0, x1, y1 = photo.crop
        focus = sh.focus if sh.focus is not None else ((x0 + x1) / 2, (y0 + y1) / 2)
        center = tr.center if tr.center is not None else focus
        scale = self._screen_scale(scene.shot, photo, focus)
        return (
            self.cv.W / 2 + (center[0] - focus[0]) * scale,
            self.cv.H / 2 + (center[1] - focus[1]) * scale,
        )

    def _frames(self, raw: str) -> float:
        spec = self.tl.spec
        got = _dur_frames(raw, spec.video.fps, spec.video.bpm, "duration", [])
        if got is None:
            raise ValueError(f"bad duration {raw!r}")
        return got

    def _background(self) -> torch.Tensor:
        # film.html 433: the stage colour shows through a fade from nothing.
        if self._bg is None:
            col = torch.tensor(_rgb(self.tl.spec.video.background), device=self.cv.device, dtype=torch.float32)
            self._bg = col[:, None, None].expand(3, self.cv.H, self.cv.W).contiguous()
        # Each eval gets its own tensor so an in-place grade cannot tint the next blank.
        return self._bg.clone()

    def _clip_source(self, stem: str) -> ClipSource:
        hit = self._clips.get(stem)
        if hit is None:
            hit = ClipSource(find_clip(self._assets, stem), self.tl.fps)
            self._clips[stem] = hit
        return hit

    def _outgoing_frames(self, index: int) -> float:
        """Frames of this shot still visible after its hold, while the next transition plays.

        A fade or swirl runs `dur` past the cut. A whip runs `dur / 2`. ClipSource clamps
        an index past the frames ffmpeg actually returned.
        """
        nxt = index + 1
        if nxt >= len(self.tl.shots):
            return 0.0
        tr = self.tl.shots[nxt].spec.in_
        if tr is None or tr.type == "cut" or tr.dur is None:
            return 0.0
        dur = self._frames(tr.dur)
        if tr.type == "whip":
            return dur / 2.0
        if tr.type in ("fade", "swirl"):
            return dur
        return 0.0

    def _clip_frame(self, scene: Clip, t: float) -> torch.Tensor:
        req = self._clip_request(scene.shot)
        if req is None:
            raise ValueError(f"shot {scene.shot} is not a clip")
        stem, clip_in, nframes = req
        return self._clip_source(stem).frame(
            t, float(scene.t0), clip_in, self.cv.W, self.cv.H, nframes=nframes, device=self.cv.device,
        )

    def _frame_bg(self, stem: str, photo: Photo) -> torch.Tensor:
        key = (stem, photo.crop)
        hit = self._bgs.get(key)
        if hit is None:
            hit = frame_background(self.cv, photo)
            self._bgs[key] = hit
        self._note("bgs", key)
        return hit

    def _sprite(self, caption: str) -> torch.Tensor:
        hit = self._captions.get(caption)
        if hit is None:
            premul, _size = raster_caption(
                caption, font_css=None, scale=float(self.cv.k), cache_dir=self._dir / "build" / "text",
            )
            hit = _straight(premul).to(self.cv.device)
            self._captions[caption] = hit
        return hit

    def _frame_card(self, stem: str, photo: Photo, caption: str | None):
        key = (stem, photo.crop, caption)
        hit = self._cards.get(key)
        if hit is None:
            sprite = self._sprite(caption) if caption else None
            hit = frame_card(self.cv, photo, caption, sprite)
            self._cards[key] = hit
        self._note("cards", key)
        return hit

    def _layer(self, sh, stem: str, photo: Photo, fx: float, fy: float, z: float, px: float, py: float,
               tilt: float, tone_on: bool) -> torch.Tensor:
        if sh.frame is not None:
            return card_frame(self.cv, self._frame_card(stem, photo, sh.frame.caption),
                              self._frame_bg(stem, photo), z, tilt, px, py)
        img = kenburns(self.cv, photo, fx, fy, z, px, py)
        # film.html 134: the CSS filter is on the cover image, not the frame-card background.
        if tone_on and sh.tone is not None:
            img = tone(img, sh.tone.brightness, sh.tone.saturate, sh.tone.contrast)
        return img

    def _tilt(self, sh, t: float, v0: float, v1: float) -> float:
        # film.html 150: rotate(lerp(tilt0, tilt1, es(k))). None stays 0; the film's own default is [-2.5, 1.5].
        if sh.frame is None or sh.frame.tilt is None:
            return 0.0
        a, b = sh.frame.tilt
        k = 0.0 if v1 == v0 else min(1.0, max(0.0, (t - v0) / (v1 - v0)))
        return a + (b - a) * sine(k)

    def _pulse(self, img: torch.Tensor, t: float, start: float) -> torch.Tensor:
        # film.html 295: phase is the beat fraction from the shot start. Scale is about the stage centre.
        bpm = self.tl.spec.video.bpm
        if not bpm:
            return img
        beat = self.tl.fps * 60.0 / float(bpm)
        if beat <= 0:
            return img
        scale = pulse_heartbeat(((t - start) % beat) / beat)
        if abs(scale - 1.0) < 1e-3:
            return img
        return _scale_about(self.cv, img, scale)

    def _morph_pulse(self, base: torch.Tensor, arrived: torch.Tensor | None, sh, shot, t: float) -> torch.Tensor:
        if sh.morph is not None:
            at = self._frames(sh.morph.at)
            dur = self._frames(sh.morph.dur)
            # film.html 154: x runs from the shot start, not the visible-span origin.
            x = 1.0 if dur <= 0 else (t - (shot.start + at)) / dur
            center = sh.morph.center if sh.morph.center is not None else _MORPH_CENTER
            mask = morph_mask(self.cv, x, center)
            base = base * (1.0 - mask) + arrived * mask
        if sh.pulse:
            base = self._pulse(base, t, float(shot.start))
        return base

    def _m2_still(self, index: int, span: tuple[int, int] | None, t: float) -> torch.Tensor:
        self._shot = index
        shot = self.tl.shots[index]
        sh = shot.spec
        # Drift and a frame card own the framing. Tone, morph, or pulse alone keep M1 focus/zoom.
        if sh.drift is None and sh.frame is None:
            base = self._still_image(index)
            if sh.tone is not None:
                base = tone(base, sh.tone.brightness, sh.tone.saturate, sh.tone.contrast)
            arrived = None
            if sh.morph is not None:
                other = self._photo(sh.morph.photo, None, False)
                arrived = still(self.cv, other, focus=None, zoom=sh.zoom)
            return self._morph_pulse(base, arrived, sh, shot, t)
        v0, v1 = (float(span[0]), float(span[1])) if span is not None else (float(shot.start), float(shot.end))
        zoom = sh.drift.zoom if sh.drift is not None else (1.0, 1.0)
        pan = sh.drift.pan if sh.drift is not None else (0.0, 0.0)
        z, px, py = drift(t, v0, v1, zoom, pan)
        tilt = self._tilt(sh, t, v0, v1)
        primary = self._photo(sh.photo, sh.crop, False)
        fx, fy = focus_frac(primary, sh.focus)
        base = self._layer(sh, sh.photo, primary, fx, fy, z, px, py, tilt, True)
        arrived = None
        if sh.morph is not None:
            other = self._photo(sh.morph.photo, None, False)
            arrived = self._layer(sh, sh.morph.photo, other, fx, fy, z, px, py, tilt, False)
        return self._morph_pulse(base, arrived, sh, shot, t)

    def _fade(self, a: torch.Tensor, b: torch.Tensor, x: float, blur: float) -> torch.Tensor:
        # blur 0 stays on M1 smoothstep (`fade`). film.html vis() line 411 eases with `es`
        # (line 73, sine). Line 417 blurs each side by (1 − opacity) · blur.
        if blur <= 0:
            return fade(a, b, x)
        e = sine(x)
        sa = e * blur * self.cv.k
        sb = (1.0 - e) * blur * self.cv.k
        if sa > 1e-3:
            a = css_blur(a, sa)
        if sb > 1e-3:
            b = css_blur(b, sb)
        return a * (1.0 - e) + b * e

    def _line(self, ln) -> TextLine:
        weight = None if ln.weight is None else int(ln.weight)
        return TextLine(
            text=ln.text, style=ln.style, y=ln.y, at=ln.at, italic=ln.italic, gold=ln.gold,
            color=ln.color, size=ln.size, weight=weight, tracking=ln.tracking,
            line_height=ln.line_height, by=ln.by, stagger=ln.stagger, dur=ln.dur, rise=ln.rise,
            reveal=ln.reveal, sweep=ln.sweep, shadow=ln.shadow, indent=ln.indent,
        )

    def _ensure_rasters(self) -> None:
        """One pass per Renderer. A disk hit starts no browser; a miss batches each kind."""
        if self._rasters_ready:
            return
        captions = []
        for shot in self.tl.shots:
            frame = shot.spec.frame
            if frame is not None and frame.caption and frame.caption not in captions:
                captions.append(frame.caption)
        if not self.tl.texts and not self.tl.subs and not captions:
            self._rasters_ready = True
            return
        cache = self._dir / "build" / "text"
        blocks = []
        flat: list[TextLine] = []
        for block in self.tl.texts:
            lines = tuple(self._line(ln) for ln in block.lines)
            flat.extend(lines)
            blocks.append(TextBlock(block.from_, block.to, block.fade_in, block.fade_out, block.blur, lines))
        subs = tuple(Sub(s.from_, s.to, s.text, size=s.size, color=s.color) for s in self.tl.subs)
        device = self.cv.device
        scale = float(self.cv.k)
        lines = tuple(_move_line(ln, device) for ln in raster_lines(flat, scale=scale, cache_dir=cache)) if flat else ()
        sub_r = tuple(_move_image(im, device) for im in raster_subs(subs, scale=scale, cache_dir=cache)) if subs else ()
        self._blocks = tuple(blocks)
        self._subs = subs
        # Captions are card art, not the finish overlay. An empty RasterSet would still clone.
        if blocks or subs:
            self._rasters = RasterSet(lines, sub_r)
        for caption in captions:
            self._sprite(caption)
        self._rasters_ready = True

    def _overlay(self, f: int):
        if self._rasters is None:
            return None

        def overlay(img: torch.Tensor) -> torch.Tensor:
            return draw_text(
                self.cv, img, self._blocks, self._subs, float(f), self._rasters, fps=float(self.tl.fps),
            )

        return overlay

    def _eval(self, scene: Scene, t: float) -> torch.Tensor:
        if isinstance(scene, Blank):
            return self._background()
        if isinstance(scene, Still):
            sh = self.tl.shots[scene.shot].spec
            if not _m2(sh):
                return self._still_image(scene.shot)
            return self._m2_still(scene.shot, scene.span, t)
        if isinstance(scene, Clip):
            return self._clip_frame(scene, t)
        if isinstance(scene, Wall):
            spec = self.tl.shots[scene.shot].spec.wall
            if spec is None:
                raise ValueError(f"shot {scene.shot} has no wall")
            wall = self._wall(scene.shot)
            # t0 is the wall shot's start, fade included. Camera keys share that clock.
            T, s = wall_camera(wall, spec.camera.model_dump(), t - scene.t0)
            return wall_frame(self.cv, wall, T, s)
        x = (t - scene.t0) / scene.dur if scene.dur else 1.0
        a = self._eval(scene.a, t)
        b = self._eval(scene.b, t)
        if scene.kind == "fade":
            return self._fade(a, b, x, scene.blur)
        if scene.kind == "whip":
            return whip(self.cv, a, b, x, scene.axis)
        if scene.kind == "swirl":
            return swirl(self.cv, a, b, x, self._swirl_center(scene))
        raise ValueError(f"transition: unknown {scene.kind!r}")

    def _look(self):
        name = self.tl.spec.video.look
        look = LOOKS.get(name)
        if look is None:
            have = ", ".join(LOOKS)
            raise ValueError(f"look: unknown {name!r} (have: {have})")
        if name != "golden-film":
            return look
        # Copy: fps is per film, and LOOKS itself stays immutable.
        return {**look, "fps": self.tl.fps}

    def frame(self, f: int) -> torch.Tensor:
        """(3, H, W) in 0..1. Sub-frame times share the integer frame's scene."""
        plan = self.tl.plan(f)
        self._ensure_rasters()
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
            return finish(self.cv, acc, f, self._look(), plan, self._overlay(f))
