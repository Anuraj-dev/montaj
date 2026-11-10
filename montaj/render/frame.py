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

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image

from montaj.clipmap import camera_at, source_index
from montaj.encode import probe
from montaj.matte import find_matte
from montaj.project import find_photo, matte_source, resolve_clip, shot_clips, shot_mattes
from montaj.render.canvas import Canvas
from montaj.render.clip import ClipSource, window_key
from montaj.render.color import grade, to_display, to_linear
from montaj.render.core import Photo, fit_s, sine, still
from montaj.render.custom import CustomRunner
from montaj.render import effects as fxlib
from montaj.render.finish import LOOKS, finish, lut_texture
from montaj.render.layers import (
    LayerState, LayerTexture, base_box, base_pos, box_growth, composite, load_rgba, peak_scale_world, sample, world_state,
)
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
    TextBackdrop,
    TextBlock,
    TextLine,
    Unit,
    draw_text,
    raster_caption,
    raster_lines,
    raster_subs,
)
from montaj.render.describe import TextLayout
from montaj.render.overlay import draw_overlays
from montaj.render.type import GlyphAtlas, GlyphLine, line_style
from montaj.render.transitions import burn_white, dip, fade, glitch, push, swirl, wall_pullback, whip, zoom
from montaj.render.wall import build_wall, wall_camera, wall_frame
from montaj.spec import _dur_frames, _parse_beat0
from montaj.timeline import (
    Blank,
    Card,
    Clip,
    Custom,
    Layers,
    Scene,
    Sequence,
    Still,
    Timeline,
    Trans,
    Wall,
    clip_read_frames,
    sequence_index,
    transition_overlap,
)

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
# Layer images (stickers, logos) a segment can composite: a transition's two layer shots, with slack.
_LAYER_IMAGE_CAP = 8
# Sub-frame samples for a `text[].motion_blur` block: at least this many, more when video.motion_blur is higher.
TEXT_BLUR_SAMPLES = 5


@dataclass(frozen=True)
class _ClipPlan:
    """How one clip shot is decoded. `crop` None means the plain cover decode at the canvas size."""

    stem: str
    clip_in: float
    count: int
    w: int
    h: int
    crop: tuple[int, int, int, int] | None
    src: tuple[int, ...] | None
    box: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class _LayerPlan:
    """Where one layer draws. `dest` is device px; `src` is texture px (a decoded clip's whole frame).

    `clip` is the decode window for a clip layer, None for an image. `t_start` is the frame the clip's
    first decoded frame shows at. `full` marks a full-frame cover clip decoded like a plain clip shot.
    """

    stem: str
    dest: tuple[float, float, float, float]
    src: tuple[float, float, float, float]
    clip: _ClipPlan | None
    t_start: float
    full: bool


_IDENTITY = LayerState(1.0, 0.0, 1.0)


def _radius(layer, local: float) -> float:
    return max(0.0, float(sample(layer.radius, local, 0.0)))


def _cover_crop(sw: float, sh: float, aspect: float, focus) -> tuple[float, float, float, float]:
    """The largest `aspect` (w/h) window of a sw x sh source, centred on `focus` and kept inside it."""
    if sw / sh > aspect:
        cw, ch = sh * aspect, sh
    else:
        cw, ch = sw, sw / aspect
    fx, fy = focus if focus is not None else (sw / 2, sh / 2)
    x0 = min(max(fx - cw / 2, 0.0), sw - cw)
    y0 = min(max(fy - ch / 2, 0.0), sh - ch)
    return x0, y0, x0 + cw, y0 + ch


def _place(layer, sw: int, sh: int, W: int, H: int, k: float):
    """(dest rect device px, source rect source px) for a layer over a sw x sh source."""
    if layer.box is not None:
        x0, y0, x1, y1 = (v * k for v in base_box(layer))
    elif layer.pos is not None:
        bx, by = base_pos(layer)
        cx, cy = bx * k, by * k
        x0, y0, x1, y1 = cx - sw * k / 2, cy - sh * k / 2, cx + sw * k / 2, cy + sh * k / 2
        return (x0, y0, x1, y1), (0.0, 0.0, float(sw), float(sh))
    else:
        x0, y0, x1, y1 = 0.0, 0.0, float(W), float(H)
    if layer.fit == "contain":
        f = min((x1 - x0) / sw, (y1 - y0) / sh)
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, sw * f / 2, sh * f / 2
        return (cx - hw, cy - hh, cx + hw, cy + hh), (0.0, 0.0, float(sw), float(sh))
    return (x0, y0, x1, y1), _cover_crop(sw, sh, (x1 - x0) / (y1 - y0), layer.focus)


class _ClipView:
    """The decoded crop, shaped like a Photo for `kenburns`: crop is the whole texture."""

    __slots__ = ("crop", "tex")

    def __init__(self, tex: torch.Tensor):
        self.tex = tex
        self.crop = (0, 0, int(tex.shape[3]), int(tex.shape[2]))


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
        "layer_images": _LAYER_IMAGE_CAP,
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

    def __init__(self, tl: Timeline, spec_dir: Path, cv: Canvas, *, cache_caps: dict[str, int] | None = None,
                 log=None, policy=None):
        self.tl = tl
        self._policy = policy  # GpuPolicy or None; each CPU pixel seam asks it before running
        self.cv = cv
        self._log = log  # callable(str) for build/render.log lines, or None
        self._dir = Path(spec_dir)
        self._assets = self._dir / tl.spec.assets
        caps = _cache_caps(cache_caps)
        self._photos: _LRU = _LRU(caps["photos"])
        self._walls: _LRU = _LRU(caps["walls"])
        self._stills: _LRU = _LRU(caps["stills"])
        self._clips: _LRU = _LRU(caps["clips"])
        self._clip_plans: dict[int, _ClipPlan] = {}
        self._clip_sizes: dict[str, tuple[int, int]] = {}
        self._bgs: _LRU = _LRU(caps["bgs"])
        self._cards: _LRU = _LRU(caps["cards"])
        self._bg: torch.Tensor | None = None
        self._rasters_ready = False
        self._blocks: tuple = ()
        self._subs: tuple = ()
        self._rasters: RasterSet | None = None
        self._card_rasters: dict[int, tuple[RasterLine, ...]] = {}
        self._card_lines: dict[int, tuple[TextLine, ...]] = {}
        self._captions: _LRU = _LRU(caps["captions"])
        self._shot: int | None = None
        self._lut = None
        self._layer_images: _LRU = _LRU(caps["layer_images"])
        self._layer_plans_by_shot: dict[int, tuple[_LayerPlan, ...]] = {}
        self._matte_files: dict[tuple[str, str], Path] = {}
        # #36: text blocks drawn under the subject, and the rest; split once in `_ensure_rasters`.
        self._behind: tuple[tuple, RasterSet, tuple, RasterSet] | None = None
        self._owners: dict[str, dict[tuple, set[int]]] = {"photos": {}, "bgs": {}, "cards": {}, "layer_images": {}}
        # A layer shot reads several clips in one frame; keep each one's reader (plus a neighbour's) open.
        # A clip's subject matte (#36) is one more reader beside it, on a plain clip shot too.
        most = max((len(set(shot_clips(s.spec))) + len(shot_mattes(s.spec))
                    for s in tl.shots if s.spec.layers or shot_mattes(s.spec)), default=0)
        if most > 1:
            self._clips.cap = max(self._clips.cap, most + 1)
        video = tl.spec.video
        beat0 = _parse_beat0(video.beat0, tl.fps, []) if video.beat0 is not None else None
        self._custom = CustomRunner(self._dir, cv.W, cv.H, cv.device, tl.fps, video.bpm, beat0 or 0.0, log=log)
        # #43: shots, effects, transitions and the shutter average run in linear light; finish gets display.
        self._linear = video.color is not None and video.color.working == "linear"

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
        self._drop_owned("layer_images", self._layer_images, needed)

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

    def _clip_size(self, stem: str) -> tuple[int, int]:
        hit = self._clip_sizes.get(stem)
        if hit is None:
            info = probe(resolve_clip(self._dir, self._assets, stem))
            if info.width <= 0 or info.height <= 0:
                raise RuntimeError(f"clip {stem} has no video size")
            hit = self._clip_sizes[stem] = (info.width, info.height)
        return hit

    @staticmethod
    def _camera_keys(sh) -> list[tuple]:
        """Shot-level focus/zoom is a one-key camera."""
        if sh.camera:
            return [(float(k.t), k.zoom, k.focus, k.ease) for k in sh.camera]
        return [(0.0, sh.zoom, sh.focus, "linear")]

    def _clip_plan(self, index: int) -> _ClipPlan | None:
        hit = self._clip_plans.get(index)
        if hit is not None:
            return hit
        shot = self.tl.shots[index]
        sh = shot.spec
        if not sh.clip:
            return None
        # A range stops at its end: the outgoing fade holds the last frame instead of reading on.
        extra = 0.0 if sh.range is not None else self._outgoing_frames(index)
        nframes = clip_read_frames(shot.end - shot.start, extra)
        clip_in = self._clip_in_seconds(sh)
        speed = sh.speed
        if isinstance(speed, list):
            speed = [(k.t, k.v, k.ease) for k in speed]
        src = source_index(speed, sh.freeze, nframes)
        count = nframes if src is None else max(src) + 1
        W, H = self.cv.W, self.cv.H
        if sh.crop is None and sh.camera is None and sh.focus is None and sh.zoom is None:
            plan = _ClipPlan(sh.clip, clip_in, count, W, H, None, src, None)
        else:
            sw, shh = self._clip_size(sh.clip)
            box = tuple(sh.crop) if sh.crop is not None else (0, 0, sw, shh)
            cw, ch = box[2] - box[0], box[3] - box[1]
            zmax = max([1.0] + [float(k[1]) for k in self._camera_keys(sh) if k[1] is not None])
            # Decode no finer than the deepest zoom can show; never upscale in ffmpeg.
            d = min(1.0, max(W / cw, H / ch) * zmax)
            plan = _ClipPlan(
                sh.clip, clip_in, count, max(2, round(cw * d)), max(2, round(ch * d)), box, src, box,
            )
        self._clip_plans[index] = plan
        return plan

    def _clip_request(self, index: int) -> tuple[str, float, int] | None:
        plan = self._clip_plan(index)
        return None if plan is None else (plan.stem, plan.clip_in, plan.count)

    def _clip_in_seconds(self, sh) -> float:
        """Source offset. `range` starts at its first time; `clip_in` is the other way in."""
        raw = sh.range[0] if sh.range is not None else sh.clip_in
        if raw is None:
            return 0.0
        return self._frames(raw) / self.tl.fps

    def _drop_clips(self, needed: frozenset[int]) -> None:
        """Close windows this segment will not read. Same stem, different clip_in, is another window."""
        wanted: dict[str | tuple, set[tuple]] = {}
        for index in needed:
            sh = self.tl.shots[index].spec
            layers = sh.layers or ()
            plans = [(lp.clip, layer.mask is not None)
                     for layer, lp in zip(layers, self._layer_plans(index)) if lp.clip is not None]
            plans.append((self._clip_plan(index), sh.matte is not None))
            for plan, matted in plans:
                if plan is None:
                    continue
                key = window_key(plan.clip_in, plan.w, plan.h, plan.count, plan.crop)
                wanted.setdefault(plan.stem, set()).add(key)
                if matted:  # its matte reads the same window of the matte file
                    wanted.setdefault(("matte", plan.stem), set()).add(key)
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
            hit = Photo(self.cv, str(find_photo(self._assets, stem)), crop=crop, card=card, policy=self._policy)
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

    def _solid(self, color: str) -> torch.Tensor:
        col = torch.tensor(_rgb(color), device=self.cv.device, dtype=torch.float32)
        return col[:, None, None].expand(3, self.cv.H, self.cv.W).contiguous().clone()

    def _sequence_frame(self, scene: Sequence, t: float) -> torch.Tensor:
        shot = self.tl.shots[scene.shot]
        seq = shot.spec.sequence
        if seq is None or not seq.items:
            raise ValueError(f"shot {scene.shot} is not a sequence")
        every = seq.every
        if isinstance(every, bool) or not isinstance(every, int):
            raise ValueError(f"shot {scene.shot} sequence.every is not resolved")
        item = seq.items[sequence_index(scene.t0, every, len(seq.items), t)]
        self._shot = scene.shot
        if item.color is not None:
            return self._solid(item.color)
        if item.photo is None:
            raise ValueError(f"shot {scene.shot} sequence item has no photo or color")
        return still(self.cv, self._photo(item.photo, None, False))

    def _card_render_lines(self, card, t0: float, end: float) -> tuple[TextLine, ...]:
        """Same styling as an overlay line. Clocks are durations from the shot start."""
        lines = []
        for line in card.text:
            if line.sweep:
                sweep = (t0 + self._frames(line.sweep[0]), t0 + self._frames(line.sweep[1]))
            else:
                sweep = (t0, end)
            lines.append(
                self._line(
                    SimpleNamespace(
                        text=line.text,
                        y=line.y,
                        at=t0 + self._frames(line.at),
                        style=line.style,
                        italic=line.italic,
                        gold=line.gold,
                        color=line.color,
                        size=line.size,
                        weight=line.weight,
                        tracking=line.tracking,
                        line_height=line.line_height,
                        by=line.by,
                        stagger=None if line.stagger is None else self._frames(line.stagger),
                        dur=None if line.dur is None else self._frames(line.dur),
                        rise=line.rise,
                        reveal=line.reveal,
                        sweep=sweep,
                        shadow=line.shadow,
                        indent=line.indent,
                        align=line.align,
                        box=line.box,
                        box_color=line.box_color,
                        dot=line.dot,
                        cursor=line.cursor,
                    )
                )
            )
        return tuple(lines)

    def _clip_source(self, stem: str) -> ClipSource:
        hit = self._clips.get(stem)
        if hit is None:
            hit = ClipSource(resolve_clip(self._dir, self._assets, stem), self.tl.fps,
                             device=self.cv.device, log=self._log, policy=self._policy)
            self._clips[stem] = hit
        return hit

    def _matte_file(self, kind: str, stem: str) -> Path:
        hit = self._matte_files.get((kind, stem))
        if hit is None:
            src = matte_source(self._dir, self._assets, kind, stem)
            hit = self._matte_files[(kind, stem)] = find_matte(self._assets, src, stem)
        return hit

    def _matte_reader(self, stem: str) -> ClipSource:
        """A clip's matte decodes like the clip: same fps, window, crop and frame map."""
        key = ("matte", stem)
        hit = self._clips.get(key)
        if hit is None:
            hit = ClipSource(self._matte_file("clip", stem), self.tl.fps,
                             device=self.cv.device, log=self._log, policy=self._policy)
            self._clips[key] = hit
        return hit

    def _photo_matte(self, stem: str) -> torch.Tensor:
        """(1, h, w) matte of a still, at the still's own size."""
        key = ("matte", stem)
        hit = self._layer_images.get(key)
        if hit is None:
            hit = LayerTexture(load_rgba(self._matte_file("photo", stem), self.cv.device, self._policy)[:1])
            self._layer_images[key] = hit
        self._note("layer_images", key)
        return hit.tex

    def _clip_matte(self, scene: Clip, t: float) -> torch.Tensor:
        """(1, H, W) subject plane of a `matte:` clip shot, placed exactly like `_clip_frame`."""
        plan = self._clip_plan(scene.shot)
        m = self._matte_reader(plan.stem).frame(
            t, float(scene.t0), plan.clip_in, plan.w, plan.h, nframes=plan.count, device=self.cv.device,
            crop=plan.crop, src=plan.src,
        )
        if plan.box is not None:
            keys = self._camera_keys(self.tl.shots[scene.shot].spec)
            z, fx, fy = camera_at(keys, plan.box, t - scene.t0)
            m = kenburns(self.cv, _ClipView(m[None]), fx, fy, z, 0.0, 0.0)
        return m[:1]

    def _outgoing_frames(self, index: int) -> float:
        """Frames of this shot still visible after its hold, while the next transition plays.

        A fade, swirl, glitch, zoom, dip, push, wall-pullback or burn-white runs `dur` past the cut. A whip runs
        `dur / 2`. ClipSource clamps an index past the frames ffmpeg actually returned.
        """
        nxt = index + 1
        if nxt >= len(self.tl.shots):
            return 0.0
        tr = self.tl.shots[nxt].spec.in_
        if tr is None or tr.type == "cut" or tr.dur is None:
            return 0.0
        return transition_overlap(tr.type, self._frames(tr.dur))

    def _clip_frame(self, scene: Clip, t: float) -> torch.Tensor:
        plan = self._clip_plan(scene.shot)
        if plan is None:
            raise ValueError(f"shot {scene.shot} is not a clip")
        img = self._clip_source(plan.stem).frame(
            t, float(scene.t0), plan.clip_in, plan.w, plan.h, nframes=plan.count, device=self.cv.device,
            crop=plan.crop, src=plan.src,
        )
        if plan.box is None:
            return img
        # Crop, focus and zoom share the photo path: cover-fit the crop, then focus/zoom on it.
        # t is the continuous shot clock, so motion-blur sub-samples stay smooth.
        keys = self._camera_keys(self.tl.shots[scene.shot].spec)
        z, fx, fy = camera_at(keys, plan.box, t - scene.t0)
        return kenburns(self.cv, _ClipView(img[None]), fx, fy, z, 0.0, 0.0)

    def _layer_plans(self, index: int) -> tuple[_LayerPlan, ...]:
        hit = self._layer_plans_by_shot.get(index)
        if hit is not None:
            return hit
        shot = self.tl.shots[index]
        if not shot.spec.layers:
            return ()
        W, H, k = self.cv.W, self.cv.H, self.cv.k
        hold = shot.end - shot.start
        plans = []
        for idx, layer in enumerate(shot.spec.layers):
            start = layer.from_ or 0
            if layer.image is not None:
                with Image.open(find_photo(self._assets, layer.image)) as im:  # header only
                    sw, sh = im.size
                dest, src = _place(layer, sw, sh, W, H, k)
                plans.append(_LayerPlan(layer.image, dest, src, None, float(shot.start + start), False))
                continue
            count = (layer.to - start if layer.to is not None
                     else clip_read_frames(hold - start, self._outgoing_frames(index)))
            clip_in = 0.0 if layer.clip_in is None else self._frames(layer.clip_in) / self.tl.fps
            full = (layer.box is None and layer.pos is None and layer.fit in (None, "cover")
                    and layer.focus is None)
            if full:
                clip = _ClipPlan(layer.clip, clip_in, count, W, H, None, None, None)
                dest, src = (0.0, 0.0, float(W), float(H)), (0.0, 0.0, float(W), float(H))
            else:
                sw, sh = self._clip_size(layer.clip)
                dest, crop = _place(layer, sw, sh, W, H, k)
                crop = tuple(int(round(v)) for v in crop)
                cw, ch = max(1, crop[2] - crop[0]), max(1, crop[3] - crop[1])
                grow = max(1.0, peak_scale_world(shot.spec.layers, idx)) * box_growth(layer) * (1.1 if "pop" in (
                    getattr(layer.in_, "type", None), getattr(layer.out, "type", None)) else 1.0)
                # Decode no finer than the layer is drawn (pop overshoot included); never upscale.
                d = min(1.0, max((dest[2] - dest[0]) / cw, (dest[3] - dest[1]) / ch) * grow)
                w, h = max(2, round(cw * d)), max(2, round(ch * d))
                clip = _ClipPlan(layer.clip, clip_in, count, w, h, crop, None, None)
                src = (0.0, 0.0, float(w), float(h))
            plans.append(_LayerPlan(layer.clip, dest, src, clip, float(shot.start + start), full))
        hit = self._layer_plans_by_shot[index] = tuple(plans)
        return hit

    def _layer_image(self, stem: str) -> LayerTexture:
        key = (stem,)
        hit = self._layer_images.get(key)
        if hit is None:
            hit = LayerTexture(load_rgba(find_photo(self._assets, stem), self.cv.device, self._policy))
            self._layer_images[key] = hit
        self._note("layer_images", key)
        return hit

    def _layer_clip(self, plan: _LayerPlan, t: float) -> torch.Tensor:
        c = plan.clip
        return self._clip_source(c.stem).frame(
            t, plan.t_start, c.clip_in, c.w, c.h, nframes=c.count, device=self.cv.device, crop=c.crop,
        )

    def _layer_matte(self, plan: _LayerPlan, t: float) -> torch.Tensor:
        """(1, h, w) matte in the layer's texture px."""
        c = plan.clip
        if c is None:
            return self._photo_matte(plan.stem)
        return self._matte_reader(c.stem).frame(
            t, plan.t_start, c.clip_in, c.w, c.h, nframes=c.count, device=self.cv.device, crop=c.crop,
        )[:1]

    def _layer_texture(self, layer, plan: _LayerPlan, t: float) -> LayerTexture:
        """The layer's source; `mask: subject` multiplies it (premultiplied) by its matte."""
        if layer.mask is None:
            return LayerTexture(self._layer_clip(plan, t)) if plan.clip is not None else self._layer_image(plan.stem)
        m = self._layer_matte(plan, t)
        if plan.clip is not None:
            rgb = self._layer_clip(plan, t)
            return LayerTexture(torch.cat([rgb * m, m]))
        key = ("cut", plan.stem)
        hit = self._layer_images.get(key)
        if hit is None:
            tex = self._layer_image(plan.stem).tex
            tex = tex if tex.shape[0] == 4 else torch.cat([tex, torch.ones_like(tex[:1])])
            hit = self._layer_images[key] = LayerTexture(tex * m)
        self._note("layer_images", key)
        return hit

    def _layers_subject(self, scene: Layers, t: float) -> torch.Tensor | None:
        """(1, H, W): every `mask:` layer's matte where that layer lands, or None without one."""
        shot = self.tl.shots[scene.shot]
        hold = shot.end - shot.start
        out: torch.Tensor | None = None
        plans = self._layer_plans(scene.shot)
        for n, (layer, plan) in enumerate(zip(shot.spec.layers or (), plans)):
            if layer.mask is None:
                continue
            state = self._state_at(shot, plans, n, t - scene.t0, hold)
            if state is None:
                continue
            if out is None:
                out = torch.zeros(3, self.cv.H, self.cv.W, device=self.cv.device)
            m = self._layer_matte(plan, t)
            local = t - scene.t0
            # Opacity and blend shape how the cutout looks, not where the subject is.
            composite(self.cv, out, LayerTexture(m.expand(4, -1, -1)), plan.src, self._dest_at(layer, plan, local),
                      LayerState(state.scale, state.rotate, 1.0), "normal", _radius(layer, local) * self.cv.k)
        return None if out is None else out[:1]

    def _subject(self, scene: Scene, t: float) -> torch.Tensor | None:
        """(1, H, W) in 0..1: where the on-screen subject is, for `behind:` text. None when no shot
        on screen has a matte. A fade mixes both sides' planes; other transitions have none."""
        if isinstance(scene, Clip):
            return self._clip_matte(scene, t) if self.tl.shots[scene.shot].spec.matte is not None else None
        if isinstance(scene, Layers):
            return self._layers_subject(scene, t)
        if isinstance(scene, Trans) and scene.kind == "fade":
            a, b = self._subject(scene.a, t), self._subject(scene.b, t)
            if a is None and b is None:
                return None
            zero = torch.zeros(1, self.cv.H, self.cv.W, device=self.cv.device)
            x = (t - scene.t0) / scene.dur if scene.dur else 1.0
            a3, b3 = (zero if a is None else a).expand(3, -1, -1), (zero if b is None else b).expand(3, -1, -1)
            return self._fade(a3, b3, x, scene.blur)[:1]
        return None

    def _layers_frame(self, scene: Layers, t: float) -> torch.Tensor:
        """Composite the shot's layers back to front. A bottom full-frame clip at rest is the base as-is,
        so a one-layer film matches the clip shot it replaces bit for bit."""
        shot = self.tl.shots[scene.shot]
        self._shot = scene.shot
        hold = shot.end - shot.start
        k = self.cv.k
        out: torch.Tensor | None = None
        plans = self._layer_plans(scene.shot)
        for n, (layer, plan) in enumerate(zip(shot.spec.layers or (), plans)):
            state = self._state_at(shot, plans, n, t - scene.t0, hold)
            if state is None:
                continue
            mode = layer.blend or "normal"
            local = t - scene.t0
            radius = _radius(layer, local) * k
            shadow = min(1.0, max(0.0, float(sample(layer.shadow, local, 0.0))))
            if (out is None and plan.full and state == _IDENTITY and mode == "normal"
                    and radius == 0 and shadow == 0 and not layer.effects and layer.mask is None):
                out = self._layer_clip(plan, t)
                continue
            if out is None:
                out = self._background()
            fxs = layer.effects or ()
            # Trails: older copies first, fainter with age, each re-drawn at its own time (no frame memory).
            for fx in fxs:
                if fx.type != "trails":
                    continue
                v = min(1.0, fxlib.level(fx, local))
                if v <= 0:
                    continue
                self._seam_effect("trails")
                decay = fxlib.TRAIL_DECAY if fx.decay is None else float(fx.decay)
                for i in range(fx.frames or fxlib.TRAIL_FRAMES, 0, -1):
                    ghost = self._state_at(shot, plans, n, local - i, hold)
                    if ghost is not None:
                        ghost = LayerState(ghost.scale, ghost.rotate, ghost.opacity * v * decay ** i)
                        self._draw_layer(out, layer, plan, ghost, t - i, local - i, mode,
                                         _radius(layer, local - i) * k, shadow)
            self._draw_layer(out, layer, plan, state, t, local, mode, radius, shadow)
        return out if out is not None else self._background()

    def _state_at(self, shot, plans, n: int, local: float, hold: int) -> LayerState | None:
        """Layer `n`'s state on the shot clock, with its parents' transforms folded in."""
        layers, k = shot.spec.layers, self.cv.k

        def centre(j: int) -> tuple[float, float]:
            x0, y0, x1, y1 = self._dest_at(layers[j], plans[j], local)
            return (x0 + x1) / 2 / k, (y0 + y1) / 2 / k

        return world_state(layers, n, local, hold, centre)

    def _dest_at(self, layer, plan, local: float) -> tuple[float, float, float, float]:
        """The layer's rect on the shot clock: fixed, or a keyed `box` stretching the first key's source window."""
        if not isinstance(layer.box, list):
            return plan.dest
        return tuple(v * self.cv.k for v in sample(layer.box, local, None))

    def _draw_layer(self, out, layer, plan, state, t, local, mode, radius, shadow) -> None:
        """One layer at film time `t` (`local` on the shot clock), with its shake and pixel effects."""
        fxs = layer.effects or ()
        dest = self._dest_at(layer, plan, local)
        post = pad = None
        if fxs:
            x0, y0, x1, y1 = dest
            for fx in fxs:
                if fx.type == "shake":
                    amp = fxlib.level(fx, local) * self.cv.k
                    if amp > 0:
                        self._seam_effect("shake")
                        dx, dy = fxlib.shake_offset(fx.seed or 0, local, self.tl.fps, amp)
                        x0, y0, x1, y1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
            dest = (x0, y0, x1, y1)
            ref = math.hypot(x1 - x0, y1 - y0) / 2.0 * state.scale
            post = fxlib.layer_pixels(self.cv, fxs, local, ref)
            if post is not None:
                self._seam_effect("layer")
                pad = fxlib.layer_pad(self.cv, fxs, local)
        tex = self._layer_texture(layer, plan, t)
        composite(self.cv, out, tex, plan.src, dest, state, mode, radius, shadow, post=post, post_pad=pad or 0.0)

    def _seam_effect(self, what: str) -> None:
        """Built-in effects are torch on the canvas device: gpu on CUDA, a CPU fallback otherwise."""
        if self._policy is None:
            return
        if torch.device(self.cv.device).type == "cuda":
            self._policy.gpu("effect")
        else:
            self._policy.cpu("effect", f"{what} effect", "render on a CUDA device")

    def _frame_bg(self, stem: str, photo: Photo) -> torch.Tensor:
        key = (stem, photo.crop)
        hit = self._bgs.get(key)
        if hit is None:
            hit = frame_background(self.cv, photo)
            self._bgs[key] = hit
        self._note("bgs", key)
        return hit

    def _seam_text(self, what: str | None, hint: str) -> None:
        """Report one text element to the GPU policy: `what=None` is a gpu line, else a Chromium one."""
        if self._policy is None:
            return
        if what is None:
            self._policy.gpu("text")
        else:
            self._policy.cpu("text", f"{what} rasterised by Chromium", hint)

    def _sprite(self, caption: str) -> torch.Tensor:
        hit = self._captions.get(caption)
        if hit is None:
            self._seam_text(f"caption {caption!r}", "frame captions are Chromium-only today; drop the caption")
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

    def _word_card(self, scene: Card, t: float) -> torch.Tensor:
        """Colour and text are one picture, so a transition carries both."""
        self._shot = scene.shot
        card = self.tl.shots[scene.shot].spec.card
        if card is None:
            raise ValueError(f"shot {scene.shot} has no card")
        img = self._solid(card.color)
        rasters = self._card_rasters.get(scene.shot)
        if not rasters:
            return img
        lines = self._card_render_lines(card, float(scene.t0), float(scene.end))
        block = TextBlock(scene.span[0], scene.span[1], 0.0, 0.0, 0.0, lines)
        return draw_text(
            self.cv, img, [block], [], t, RasterSet(rasters, ()), fps=float(self.tl.fps),
        )

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
            reveal=ln.reveal, sweep=ln.sweep, shadow=bool(ln.shadow), indent=ln.indent,
            align=ln.align, box=ln.box, box_color=ln.box_color, dot=ln.dot, cursor=ln.cursor,
            motion=getattr(ln, "motion", None), bar=getattr(ln, "bar", None),
        )

    def _backdrop(self, src) -> TextBackdrop | None:
        if src is None:
            return None
        return TextBackdrop(src.type, float(src.strength), src.place)

    def _ensure_rasters(self) -> None:
        """One pass per Renderer. A disk hit starts no browser; a miss batches each kind."""
        if self._rasters_ready:
            return
        captions = []
        for shot in self.tl.shots:
            frame = shot.spec.frame
            if frame is not None and frame.caption and frame.caption not in captions:
                captions.append(frame.caption)
        card_lines: list[TextLine] = []
        card_index: dict[int, tuple[int, int]] = {}
        for shot in self.tl.shots:
            card = shot.spec.card
            if card is None:
                continue
            built = self._card_render_lines(card, float(shot.start), float(shot.end))
            card_index[shot.index] = (len(card_lines), len(card_lines) + len(built))
            self._card_lines[shot.index] = built
            card_lines.extend(built)
        if not self.tl.texts and not self.tl.subs and not captions and not card_lines:
            self._rasters_ready = True
            return
        cache = self._dir / "build" / "text"
        blocks = []
        flat: list[TextLine] = []
        glyph_lines: dict[int, GlyphLine] = {}  # flat index -> gpu-engine line; the rest go to Chromium
        atlas = None
        for block in self.tl.texts:
            lines = tuple(self._line(ln) for ln in block.lines)
            for j, ln in enumerate(block.lines):
                if ln.glyphs is not None:
                    atlas = atlas or GlyphAtlas(self._dir / "build" / "glyphs", self.cv.device)
                    glyph_lines[len(flat) + j] = GlyphLine(ln.glyphs, line_style(ln), atlas)
            flat.extend(lines)
            blocks.append(
                TextBlock(
                    block.from_, block.to, block.fade_in, block.fade_out, block.blur, lines,
                    self._backdrop(block.backdrop),
                    # `motion_blur: true`: the film's sub-frame count, at least TEXT_BLUR_SAMPLES.
                    samples=max(self.tl.spec.video.motion_blur, TEXT_BLUR_SAMPLES) if block.motion_blur else 1,
                )
            )
        subs = tuple(Sub(s.from_, s.to, s.text, size=s.size, color=s.color) for s in self.tl.subs)
        device = self.cv.device
        scale = float(self.cv.k)
        chromium = [ln for i, ln in enumerate(flat) if i not in glyph_lines]
        everything = (*chromium, *card_lines)
        for _ in glyph_lines:
            self._seam_text(None, "")
        for ln in chromium:
            self._seam_text(f"line {ln.text!r} (style {ln.style})", "set `engine: gpu` on the line")
        for ln in card_lines:
            self._seam_text(f"card line {ln.text!r}", "card text is Chromium-only today; use a `text` block with `engine: gpu`")
        for sub in subs:
            self._seam_text(f"sub {sub.text!r}", "subs are Chromium-only today; use a `text` block with `engine: gpu`")
        drawn = iter(tuple(_move_line(ln, device) for ln in raster_lines(everything, scale=scale, cache_dir=cache)) if everything else ())
        # Back in flat order: gpu lines take their GlyphLine slot, so an all-gpu film starts no browser.
        rasters = tuple(glyph_lines[i] if i in glyph_lines else next(drawn) for i in range(len(flat))) + tuple(drawn)
        sub_r = tuple(_move_image(im, device) for im in raster_subs(subs, scale=scale, cache_dir=cache)) if subs else ()
        self._blocks = tuple(blocks)
        self._subs = subs
        n_overlay = len(flat)
        # Captions are card art, not the finish overlay. An empty RasterSet would still clone.
        if blocks or subs:
            self._rasters = RasterSet(rasters[:n_overlay], sub_r)
            self._split_behind(rasters[:n_overlay], sub_r)
        self._card_rasters = {
            index: rasters[n_overlay + lo : n_overlay + hi] for index, (lo, hi) in card_index.items()
        }
        for caption in captions:
            self._sprite(caption)
        self._rasters_ready = True

    def _split_behind(self, lines: tuple, sub_r: tuple) -> None:
        """Partition blocks (and their line rasters) into `behind: subject` and the rest."""
        flags = [block.behind is not None for block in self.tl.spec.text]
        if not any(flags):
            return
        back, back_l, front, front_l = [], [], [], []
        cursor = 0
        for block, behind in zip(self._blocks, flags):
            n = len(block.lines)
            (back if behind else front).append(block)
            (back_l if behind else front_l).extend(lines[cursor:cursor + n])
            cursor += n
        self._behind = (tuple(back), RasterSet(tuple(back_l), ()), tuple(front), RasterSet(tuple(front_l), sub_r))

    def text_layout(self) -> TextLayout:
        """The text this film shows, laid out (Chromium rasters, no frame rendered). Feeds `describe`."""
        self._ensure_rasters()
        return TextLayout(
            self._blocks, self._subs, self._rasters or RasterSet(), self._card_lines, self._card_rasters,
            float(self.cv.k), self.cv.H,
        )

    def _overlay(self, f: int, above: list, scene: Scene | None = None):
        """Hook run inside `finish`: `above_finish` shapes, then text, so text always sits on top.

        With `behind:` blocks (#36) and a subject on screen: those blocks first, then the picture
        again wherever the subject's matte is, then the other blocks and subs on top."""
        if self._rasters is None and not above:
            return None
        subject = None
        if self._behind is not None and scene is not None:
            subject = self._subject(scene, float(f))

        def overlay(img: torch.Tensor) -> torch.Tensor:
            img = draw_overlays(self.cv, img, above, float(f))
            if self._rasters is None:
                return img
            fps = float(self.tl.fps)
            if subject is None:
                return draw_text(self.cv, img, self._blocks, self._subs, float(f), self._rasters, fps=fps)
            back, back_r, front, front_r = self._behind
            under = draw_text(self.cv, img, back, (), float(f), back_r, fps=fps)
            img = under + (img - under) * subject
            return draw_text(self.cv, img, front, self._subs, float(f), front_r, fps=fps)

        return overlay

    def _run_custom(self, rel: str, params: dict | None, *, t: float, p: float, f: float,
                    inputs: list[torch.Tensor]) -> torch.Tensor:
        """Run one custom file. The policy sees a `custom` seam: gpu on CUDA, a CPU fallback otherwise."""
        if self._policy is not None:
            if torch.device(self.cv.device).type == "cuda":
                self._policy.gpu("custom")
            else:
                self._policy.cpu("custom", f"custom {rel}", "render on a CUDA device")
        return self._custom.run(rel, params, t=t, p=p, f=f, inputs=inputs, background=self._background())

    def _custom_shot(self, scene: Custom, t: float) -> torch.Tensor:
        sh = self.tl.shots[scene.shot].spec
        local = t - scene.t0
        return self._run_custom(sh.custom, sh.params, t=local / self.tl.fps,
                                p=local / max(1, scene.end - 1 - scene.t0), f=t, inputs=[])

    def _effects(self, scene: Scene, t: float) -> torch.Tensor:
        """The base scene, then each `effects[]` entry in order, each reading the picture so far."""
        shot = self.tl.shots[scene.shot]
        return self._effect_chain(scene, shot, t, len(shot.spec.effects))

    def _effect_chain(self, scene: Scene, shot, t: float, n: int) -> torch.Tensor:
        """The picture after the first `n` effects at `t`. Trails call it again at earlier times, so a
        frame's trail is the same whether or not its segment rendered the frames before it."""
        if n == 0:
            return self._eval(scene, t, effects=False)
        fx = shot.spec.effects[n - 1]
        local = t - shot.start
        if fx.custom is not None:
            img = self._effect_chain(scene, shot, t, n - 1)
            return self._run_custom(fx.custom, fx.params, t=local / self.tl.fps,
                                    p=local / max(1, shot.end - 1 - shot.start), f=t, inputs=[img])
        v = fxlib.level(fx, local)
        if v <= 0:
            return self._effect_chain(scene, shot, t, n - 1)
        self._seam_effect(fx.type)
        if fx.type == "trails":
            decay = fxlib.TRAIL_DECAY if fx.decay is None else float(fx.decay)
            return fxlib.trails(lambda i: self._effect_chain(scene, shot, t - i, n - 1), v,
                                fx.frames or fxlib.TRAIL_FRAMES, decay)
        return fxlib.frame_effect(self.cv, self._effect_chain(scene, shot, t, n - 1), fx, local, self.tl.fps)

    def _color(self, scene: Scene, img: torch.Tensor) -> torch.Tensor:
        """Input transform and primary grade on one shot's picture, before its effects (#43)."""
        g = None if isinstance(scene, Blank) else self.tl.shots[scene.shot].spec.grade
        if g is None and not self._linear:
            return img  # display mode, no grade: untouched, so existing films stay bit-identical
        self._seam_grade("primary grade" if g is not None else "linear transform")
        lin = to_linear(img) if g is None else grade(to_linear(img), g)
        return lin if self._linear else to_display(lin)

    def _seam_grade(self, what: str) -> None:
        if self._policy is None:
            return
        if torch.device(self.cv.device).type == "cuda":
            self._policy.gpu("grade")
        else:
            self._policy.cpu("grade", what, "render on a CUDA device")

    def _eval(self, scene: Scene, t: float, effects: bool = True, color: bool = True) -> torch.Tensor:
        if color and not isinstance(scene, Trans) and (
                isinstance(scene, Blank) or not (effects and self.tl.shots[scene.shot].spec.effects)):
            return self._color(scene, self._eval(scene, t, effects, color=False))
        if isinstance(scene, Blank):
            return self._background()
        if effects and not isinstance(scene, Trans) and self.tl.shots[scene.shot].spec.effects:
            return self._effects(scene, t)
        if isinstance(scene, Custom):
            return self._custom_shot(scene, t)
        if isinstance(scene, Card):
            return self._word_card(scene, t)
        if isinstance(scene, Still):
            sh = self.tl.shots[scene.shot].spec
            if not _m2(sh):
                return self._still_image(scene.shot)
            return self._m2_still(scene.shot, scene.span, t)
        if isinstance(scene, Sequence):
            return self._sequence_frame(scene, t)
        if isinstance(scene, Clip):
            return self._clip_frame(scene, t)
        if isinstance(scene, Layers):
            return self._layers_frame(scene, t)
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
        if scene.kind == "push":
            return push(self.cv, a, b, x, scene.axis)
        if scene.kind == "dip":
            return dip(self.cv, a, b, x, scene.dip_color, linear=self._linear)
        if scene.kind == "zoom":
            return zoom(self.cv, a, b, x, scene.strength)
        if scene.kind == "glitch":
            # Sub-frame samples share their integer frame's tear pattern.
            return glitch(self.cv, a, b, x, scene.strength, scene.seed, math.floor(t + 0.5))
        if scene.kind == "wall-pullback":
            if scene.zoom is None:
                raise ValueError("wall-pullback: missing zoom")
            return wall_pullback(self.cv, a, b, x, scene.zoom)
        if scene.kind == "burn-white":
            return burn_white(a, b, x)
        if scene.kind == "custom":
            return self._run_custom(scene.src, json.loads(scene.params or "{}"), t=(t - scene.t0) / self.tl.fps,
                                    p=x, f=t, inputs=[a, b])
        raise ValueError(f"transition: unknown {scene.kind!r}")

    def _look(self):
        name = self.tl.spec.video.look
        if not isinstance(name, str):
            # `look: {lut: …}`. The texture is parsed once per Renderer; strength/vignette/grain ride along.
            if self._lut is None:
                from montaj.cube import load_cube

                self._lut = lut_texture(load_cube(self._dir / name.lut), self.cv.device)
            return {"kind": "graded", "lut": self._lut, "strength": name.strength,
                    "vignette": name.vignette, "grain": name.grain}
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
            if self._linear:  # output transform: the look, overlays and text are display-referred
                acc = to_display(acc)
            # Plain overlays are graded with the picture. `above_finish` ones skip the look but
            # still land under text and fade with the picture.
            ovs = [self.tl.overlays[i] for i in plan.overlays]
            acc = draw_overlays(self.cv, acc, [o for o in ovs if not o.above_finish], float(f))
            above = [o for o in ovs if o.above_finish]
            return finish(self.cv, acc, f, self._look(), plan, self._overlay(f, above, plan.scene))
