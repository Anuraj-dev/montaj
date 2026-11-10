"""Colour scopes as numbers, scope sheets and shot matching (#45).

Every number comes from sampled frames of the *graded* picture, rendered by the existing `Renderer`
(`Renderer.picture`: after grade and effects, before overlays, text and the look). The statistics run
as torch ops on the renderer's device; `match` solves a small grade (exposure, temp/tint, saturation)
with `montaj.render.color.grade`, the same operator the renderer applies. No second pipeline.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml

from montaj import gpulock
from montaj.analysis import load_analysis
from montaj.gpu_policy import GpuPolicy, GpuStrictError, load_policy, resolve_device
from montaj.grade import Grade
from montaj.pipeline import _apply_mode
from montaj.render.canvas import Canvas
from montaj.render.color import grade as apply_grade
from montaj.render.color import to_display, to_linear
from montaj.render.core import fit_s
from montaj.render.frame import Renderer, _m2
from montaj.render.nvdec import geometry
from montaj.spec import Spec, _clip_start_s, load_spec
from montaj.timeline import Blank, Timeline, Trans, resolve

SAMPLES = 4  # frames per shot
STAT_SIDE = 320  # long side the statistics run at (area-downsampled)
LUMA = (0.2126, 0.7152, 0.0722)  # Rec.709, on display-encoded values (Y')
CLIP_HI_WARN = 0.02  # fraction of pixels at the top
SKIN_LINE = (10.0, 34.0)  # degrees of HSV hue; the skin line sits near 22
LUMA_JUMP = 0.06  # p50 jump between adjacent shots of one scene that `check` flags
MATCH_FIELDS = ("exposure", "temp", "tint", "saturation")
# Bounds keep a match a small correction: what lies past them is a content difference, not a grade.
_RANGES = {"temp": (-2000.0, 2000.0), "tint": (-40.0, 40.0), "saturation": (0.5, 2.0), "exposure": (-4.0, 4.0)}
_ROUND = {"exposure": 2, "temp": -1, "tint": 1, "saturation": 2}
_SKIP_KINDS = ("blank", "card")


@dataclass
class ShotScope:
    shot: int
    p1: float
    p50: float
    p99: float
    clip_hi: float
    clip_lo: float
    sat_mean: float
    cast: float  # + warm (R over B), - cool
    tint: float  # + green, - magenta
    skin_hue: float | None = None
    frames: int = 0


# --- statistics ---------------------------------------------------------------------------------

def _small(img: torch.Tensor) -> torch.Tensor:
    """(3, H, W) -> area-downsampled so the long side is at most STAT_SIDE."""
    h, w = img.shape[-2:]
    k = max(h, w) / STAT_SIDE
    if k <= 1:
        return img
    return F.interpolate(img[None], size=(max(1, round(h / k)), max(1, round(w / k))), mode="area")[0]


def _neutral_weights(x: torch.Tensor) -> torch.Tensor:
    """Per-pixel weight of near-neutral midtones in (3, N) display values: where white balance reads."""
    y = LUMA[0] * x[0] + LUMA[1] * x[1] + LUMA[2] * x[2]
    hi, lo = x.amax(0), x.amin(0)
    sat = (hi - lo) / hi.clamp(min=1e-6)
    return (1 - sat).clamp(0, 1) ** 2 * (1 - (y - 0.5).abs() / 0.45).clamp(0, 1)


def _numbers(x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """(3, ...) display values -> tensor [p1, p50, p99, clip_hi, clip_lo, sat_mean, cast, tint]."""
    x = x.float().clamp(0, 1).reshape(3, -1)
    r, g, b = x[0], x[1], x[2]
    y = LUMA[0] * r + LUMA[1] * g + LUMA[2] * b
    q = torch.quantile(y, torch.tensor([0.01, 0.5, 0.99], device=x.device))
    hi, lo = x.amax(0), x.amin(0)
    clip_hi = (y >= 0.985).float().mean()  # on luma: a saturated primary at 1.0 is not a blown highlight
    clip_lo = (y <= 0.015).float().mean()
    sat = (hi - lo) / hi.clamp(min=1e-6)
    lit = (hi > 0.05).float()
    sat_mean = (sat * lit).sum() / lit.sum().clamp(min=1)
    # White balance from the near-neutral midtones: saturated colour is content, not cast.
    # `w` pins those pixels (the solver), so a grade cannot move the metric by moving the weights.
    w = _neutral_weights(x) if w is None else w
    wsum = w.sum().clamp(min=1e-6)
    cast = (w * (r - b) / 2).sum() / wsum
    tint = (w * (g - (r + b) / 2)).sum() / wsum
    return torch.stack([q[0], q[1], q[2], clip_hi, clip_lo, sat_mean, cast, tint])


def _flat(pictures: list[torch.Tensor]) -> torch.Tensor:
    """All sampled pixels as one (3, 1, N) image: `color.grade` broadcasts per channel on (3, H, W)."""
    return torch.cat([_small(p).reshape(3, 1, -1) for p in pictures], dim=2)


def hue_deg(rgb: tuple[float, float, float]) -> float:
    """HSV hue of one colour, degrees 0..360."""
    r, g, b = rgb
    hi, lo = max(rgb), min(rgb)
    if hi - lo < 1e-6:
        return 0.0
    if hi == r:
        h = ((g - b) / (hi - lo)) % 6
    elif hi == g:
        h = (b - r) / (hi - lo) + 2
    else:
        h = (r - g) / (hi - lo) + 4
    return 60.0 * h


def _skin_pixels(img: torch.Tensor, box: tuple[float, float, float, float]) -> torch.Tensor | None:
    """(3, N) skin-candidate pixels in the central half of a screen face box, or None."""
    H, W = img.shape[-2:]
    x0, y0, x1, y1 = box
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 4
    a, b = max(0, int(cx - hw)), min(W, int(math.ceil(cx + hw)))
    c, d = max(0, int(cy - hh)), min(H, int(math.ceil(cy + hh)))
    if b - a < 2 or d - c < 2:
        return None
    px = img[:, c:d, a:b].reshape(3, -1).float().clamp(0, 1)
    y = LUMA[0] * px[0] + LUMA[1] * px[1] + LUMA[2] * px[2]
    hi, lo = px.amax(0), px.amin(0)
    keep = (y > 0.15) & (y < 0.95) & ((hi - lo) / hi.clamp(min=1e-6) > 0.1)
    return px[:, keep]


def measure(pictures: list[torch.Tensor], shot: int, faces: list | None = None) -> ShotScope:
    """Scope numbers for one shot from its display-encoded (3, H, W) pictures (any device).
    `faces[k]` is a screen-px face box for `pictures[k]`, or None."""
    vals = _numbers(_flat(pictures)).tolist()
    skin = None
    if faces:
        parts = [_skin_pixels(p, box) for p, box in zip(pictures, faces) if box is not None]
        parts = [p for p in parts if p is not None and p.shape[1]]
        if parts:
            px = torch.cat(parts, dim=1)
            if px.shape[1] >= 50:
                skin = round(hue_deg(tuple(px.mean(1).tolist())), 1)
    p1, p50, p99, chi, clo, sat, cast, tint = (round(v, 4) for v in vals)
    return ShotScope(shot, p1, p50, p99, chi, clo, sat, cast, tint, skin, len(pictures))


def on_skin_line(hue: float) -> bool:
    return SKIN_LINE[0] <= hue <= SKIN_LINE[1]


def line(s: ShotScope) -> str:
    warm = "warm" if s.cast > 0.005 else "cool" if s.cast < -0.005 else "neutral"
    out = (f"shot {s.shot}  luma p1={s.p1:.2f} p50={s.p50:.2f} p99={s.p99:.2f}  "
           f"clip_hi={s.clip_hi * 100:.1f}% clip_lo={s.clip_lo * 100:.1f}%  sat_mean={s.sat_mean:.2f}  "
           f"cast={s.cast:+.3f} {warm}  tint={s.tint:+.3f}")
    if s.skin_hue is not None:
        out += f"  skin_hue={s.skin_hue:.0f}° ({'on line' if on_skin_line(s.skin_hue) else 'off line'})"
    return out


# --- sampling through the renderer ---------------------------------------------------------------

def _device(policy: GpuPolicy) -> str:
    if torch.cuda.is_available():
        policy.gpu("scopes")
        return resolve_device(policy.device)
    policy.cpu("scopes", "scope statistics", "run on a machine with a CUDA device")
    return "cpu"


def _renderer(spec: Spec, root: Path, policy: GpuPolicy, log: Path) -> Renderer:
    tl = resolve(_apply_mode(spec, "preview"), root)
    W, H = tl.size
    return Renderer(tl, root, Canvas(W, H, device=_device(policy)), policy=policy,
                    log=lambda message: _log(log, message))


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def measurable(tl: Timeline, i: int) -> bool:
    return tl.shots[i].kind not in _SKIP_KINDS


def sample_frames(tl: Timeline, i: int, n: int = SAMPLES) -> list[int]:
    """Up to `n` evenly spaced frames where shot `i` is on screen alone (no transition)."""
    sh = tl.shots[i]
    own = [f for f in range(sh.start, sh.end)
           if not isinstance(sc := tl.plan(f).scene, (Trans, Blank)) and getattr(sc, "shot", None) == i]
    if len(own) <= n:
        return own
    return [own[round((k + 0.5) * len(own) / n - 0.5)] for k in range(n)]


def _analysis(root: Path, tl: Timeline) -> dict[str, dict]:
    try:
        rows = load_analysis(root / tl.spec.assets)
    except (OSError, ValueError):
        return {}
    return {Path(name).stem: row for name, row in rows.items()}


def face_box(tl: Timeline, i: int, f: int, rows: dict[str, dict]) -> tuple[float, float, float, float] | None:
    """The largest analysed face of shot `i` at frame `f`, in canvas px. Only plain clips and
    static stills map back to the screen; anything moved by a camera or speed ramp gives None."""
    sh = tl.shots[i].spec
    W, H = tl.size
    if sh.clip:
        row = rows.get(str(sh.clip))
        if (row is None or sh.crop is not None or sh.camera is not None or sh.focus is not None
                or sh.zoom is not None or sh.speed is not None or sh.freeze is not None):
            return None
        start = sh.range[0] if isinstance(sh.range, tuple) else sh.clip_in
        t = _clip_start_s(tl.spec, start) + (f - tl.shots[i].start) / tl.fps
        scenes = row.get("scenes") or []
        scene = next((s for s in scenes if s["s"] <= t < s["e"]), None)
        faces = (scene or {}).get("faces") or []
        if not faces or "w" not in row:
            return None
        g = geometry(int(row["w"]), int(row["h"]), W, H, None)
        x0, y0, x1, y1 = faces[0]["box"]
        return ((x0 - g.x0) * W / g.cw, (y0 - g.y0) * H / g.ch, (x1 - g.x0) * W / g.cw, (y1 - g.y0) * H / g.ch)
    if sh.photo and tl.shots[i].kind == "still" and not _m2(sh) and sh.camera is None:
        row = rows.get(str(sh.photo))
        if row is None or not row.get("faces") or "w" not in row or isinstance(sh.focus, str):
            return None
        crop = tuple(sh.crop) if sh.crop is not None else (0, 0, row["w"], row["h"])
        T = ((crop[0] + crop[2]) / 2, (crop[1] + crop[3]) / 2) if sh.focus is None else tuple(sh.focus)
        k = W / 1080
        s = sh.zoom * k if sh.zoom is not None else fit_s(SimpleNamespace(W=W, H=H), SimpleNamespace(crop=crop), T) * 1.005
        x0, y0, x1, y1 = row["faces"][0]["box"]
        return ((x0 - T[0]) * s + W / 2, (y0 - T[1]) * s + H / 2, (x1 - T[0]) * s + W / 2, (y1 - T[1]) * s + H / 2)
    return None


def _pictures(r: Renderer, i: int) -> tuple[list[torch.Tensor], list[int]]:
    frames = sample_frames(r.tl, i)
    pics, kept = [], []
    for f in frames:
        p = r.picture(f)
        if p is not None:
            pics.append(p)
            kept.append(f)
    return pics, kept


def scope_shots(r: Renderer, root: Path) -> list[ShotScope]:
    rows = _analysis(root, r.tl)
    out = []
    for i in range(len(r.tl.shots)):
        if not measurable(r.tl, i):
            continue
        pics, frames = _pictures(r, i)
        if not pics:
            continue
        out.append(measure(pics, i, [face_box(r.tl, i, f, rows) for f in frames]))
    return out


def scopes(root: Path, *, policy: GpuPolicy | None = None, image: bool = False) -> tuple[list[ShotScope], Path | None]:
    """Scope numbers for every shot (and the waveform/vectorscope sheet with `image`)."""
    policy = policy if policy is not None else load_policy()
    log = root / "build" / "scopes.log"
    spec = load_spec(root / "montaj.yaml")
    with gpulock.hold(log):
        r = _renderer(spec, root, policy, log)
        stats = scope_shots(r, root)
        sheet = scope_sheet(r, stats, root / "build" / "scopes.jpg") if image else None
    _log(log, f"scopes: {len(stats)} shots")
    return stats, sheet


# --- sheet ---------------------------------------------------------------------------------------

def _waveform(pics: list[torch.Tensor], levels: int = 128) -> torch.Tensor:
    cols = min(_small(p).shape[-1] for p in pics)
    hist = None
    for p in pics:
        x = _small(p).float().clamp(0, 1)
        y = LUMA[0] * x[0] + LUMA[1] * x[1] + LUMA[2] * x[2]
        h, w = y.shape
        col = (torch.arange(w, device=y.device) * cols // w).expand(h, w)  # cols <= w: no empty columns
        lev = ((1 - y) * (levels - 1)).round().long()
        idx = (lev * cols + col).reshape(-1)
        part = torch.zeros(levels * cols, device=y.device).scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        hist = part if hist is None else hist + part
    hist = hist.reshape(levels, cols)
    return torch.log1p(hist) / torch.log1p(hist.max()).clamp(min=1e-6)


def _vectorscope(pics: list[torch.Tensor], n: int = 192) -> torch.Tensor:
    hist = torch.zeros(n * n, device=pics[0].device)
    for p in pics:
        x = _small(p).float().clamp(0, 1).reshape(3, -1)
        y = LUMA[0] * x[0] + LUMA[1] * x[1] + LUMA[2] * x[2]
        cb, cr = (x[2] - y) / 1.8556, (x[0] - y) / 1.5748
        u = ((cb + 0.5) * (n - 1)).round().long().clamp(0, n - 1)
        v = ((0.5 - cr) * (n - 1)).round().long().clamp(0, n - 1)
        hist.scatter_add_(0, v * n + u, torch.ones_like(y))
    hist = hist.reshape(n, n)
    return torch.log1p(hist) / torch.log1p(hist.max()).clamp(min=1e-6)


def _to_image(t: torch.Tensor, tint: tuple[int, int, int]):
    from PIL import Image

    a = (t.clamp(0, 1) * 255).round().to(torch.uint8).cpu()  # the finished scope, to PIL for the sheet
    rgb = torch.stack([(a.float() * c / 255).to(torch.uint8) for c in tint], dim=-1).numpy()
    return Image.fromarray(rgb, "RGB")


def scope_sheet(r: Renderer, stats: list[ShotScope], out: Path) -> Path:
    """One sheet, one tile per shot: luma waveform | vectorscope with the skin line."""
    from PIL import Image, ImageDraw

    from montaj.sheet import compose_grid

    tiles, labels = [], []
    for s in stats:
        pics, _ = _pictures(r, s.shot)
        wf = _to_image(_waveform(pics), (120, 255, 140)).resize((384, 256))
        vs = _to_image(_vectorscope(pics), (255, 255, 255)).resize((256, 256))
        d = ImageDraw.Draw(wf)
        for level in (0.0, 0.5, 1.0):
            yy = round((1 - level) * 255)
            d.line([(0, yy), (383, yy)], fill=(70, 70, 70))
        d = ImageDraw.Draw(vs)
        d.ellipse([0, 0, 255, 255], outline=(70, 70, 70))
        # Skin line: the direction of a typical skin tone in the Cb/Cr plane.
        ang = math.atan2(-(0.8 - 0.6311) / 1.5748, (0.5 - 0.6311) / 1.8556)  # sRGB (0.8, 0.6, 0.5)
        d.line([(128, 128), (128 + 127 * math.cos(ang), 128 + 127 * math.sin(ang))], fill=(230, 160, 90))
        tile = Image.new("RGB", (640, 256))
        tile.paste(wf, (0, 0))
        tile.paste(vs, (384, 0))
        tiles.append(tile)
        labels.append(f"shot {s.shot}  p50={s.p50:.2f}  cast={s.cast:+.3f}")
    if not tiles:
        raise ValueError("scopes: no shot to draw")
    return compose_grid(tiles, labels, out, cols=2, tile_ratio=256 / 640)


# --- match ---------------------------------------------------------------------------------------

def _graded(lin: torch.Tensor, g: Grade) -> torch.Tensor:
    return to_display(apply_grade(lin, g))


def solve(lin: torch.Tensor, base: Grade, target: ShotScope, rounds: int = 3, steps: int = 22) -> Grade:
    """Exposure, temp/tint and saturation on top of `base` that bring the (3, 1, N) linear pixels'
    p50, cast, tint and sat_mean to `target`. Coordinate bisection: each field is monotone in its
    number, with exposure last so the luma match is the tightest."""
    want = {"temp": (6, target.cast, 1), "tint": (7, target.tint, -1), "saturation": (5, target.sat_mean, 1),
            "exposure": (1, target.p50, 1)}
    g = base.model_copy(update={"exposure": 0.0, "temp": 0.0, "tint": 0.0, "saturation": 1.0})
    w = _neutral_weights(_graded(lin, g).reshape(3, -1))
    for _ in range(rounds):
        for name in ("temp", "tint", "saturation", "exposure"):
            k, goal, sign = want[name]
            lo, hi = _RANGES[name]
            for _ in range(steps):
                mid = (lo + hi) / 2
                v = float(_numbers(_graded(lin, g.model_copy(update={name: mid})), w)[k])
                if (v - goal) * sign < 0:
                    lo = mid
                else:
                    hi = mid
            val = round((lo + hi) / 2, _ROUND[name]) + 0.0  # + 0.0 drops a negative zero
            g = g.model_copy(update={name: int(val) if name == "temp" else val})
    return g


def _plain(v):
    if isinstance(v, tuple | list):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v + 0.0 if isinstance(v, float) else v


def _flow(g: Grade) -> str:
    """`{exposure: .., temp: .., tint: .., saturation: ..}` plus any other non-default fields."""
    data = {k: getattr(g, k) for k in MATCH_FIELDS}
    data.update(g.model_dump(exclude_defaults=True, exclude=set(MATCH_FIELDS)))
    return yaml.safe_dump(_plain(data), default_flow_style=True, sort_keys=False, width=10_000).strip()


def _scalar(v) -> str:
    return yaml.safe_dump(_plain(v), default_flow_style=True).strip().removesuffix("...").strip()


def write_grades(text: str, grades: dict[int, Grade], count: int | None = None) -> str:
    """Set `grade` on shots by index, editing only those spans so comments and layout survive.
    `count` is the expanded shot count; a file whose list differs (include, components,
    `cuts: auto`) cannot be indexed by shot number and is refused."""
    root = yaml.compose(text)
    shots = None
    if isinstance(root, yaml.MappingNode):
        shots = next((v for k, v in root.value if k.value == "shots"), None)
    if not isinstance(shots, yaml.SequenceNode):
        raise ValueError("match: montaj.yaml has no shots list")
    if count is not None and len(shots.value) != count:
        raise ValueError(f"match: montaj.yaml lists {len(shots.value)} shots but the film has {count} "
                         "(include/components/cuts expand them); add grades by hand")
    edits: list[tuple[int, int, str]] = []
    for i, g in grades.items():
        item = shots.value[i]
        if not isinstance(item, yaml.MappingNode):
            raise ValueError(f"match: shots[{i}] is not a mapping")
        keys = {k.value: (k, v) for k, v in item.value}
        if "use" in keys:
            raise ValueError(f"match: shots[{i}] comes from a component; add its grade by hand")
        if "grade" in keys:
            _, v = keys["grade"]
            if isinstance(v, yaml.MappingNode) and not v.flow_style and v.value:
                edits += _block_edits(text, v, g)
            else:
                edits.append((v.start_mark.index, v.end_mark.index, _flow(g)))
        elif item.flow_style:
            last = item.value[-1][1]
            edits.append((last.end_mark.index, last.end_mark.index, f", grade: {_flow(g)}"))
        else:
            anchor = next((v for _, v in item.value if isinstance(v, yaml.ScalarNode)
                           or getattr(v, "flow_style", False)), None)
            if anchor is None:
                raise ValueError(f"match: shots[{i}]: no line to add grade under")
            col = item.value[0][0].start_mark.column
            eol = text.find("\n", anchor.end_mark.index)
            eol = len(text) if eol < 0 else eol
            edits.append((eol, eol, "\n" + " " * col + f"grade: {_flow(g)}"))
    for a, b, new in sorted(edits, key=lambda e: e[0], reverse=True):
        text = text[:a] + new + text[b:]
    return text


def _block_edits(text: str, node: yaml.MappingNode, g: Grade) -> list[tuple[int, int, str]]:
    have = {k.value: v for k, v in node.value}
    edits, missing = [], []
    for name in MATCH_FIELDS:
        v = have.get(name)
        if isinstance(v, yaml.ScalarNode):
            edits.append((v.start_mark.index, v.end_mark.index, _scalar(getattr(g, name))))
        elif v is None:
            missing.append(name)
        else:
            raise ValueError(f"match: grade.{name} is not a number")
    if missing:
        col = node.value[0][0].start_mark.column
        last = node.value[-1][1]
        eol = text.find("\n", last.end_mark.index if last.end_mark.line == last.start_mark.line
                        else last.end_mark.index - 1)
        eol = len(text) if eol < 0 else eol
        add = "".join("\n" + " " * col + f"{n}: {_scalar(getattr(g, n))}" for n in missing)
        edits.append((eol, eol, add))
    return edits


@dataclass
class MatchRow:
    shot: int
    before: Grade
    after: Grade
    p50_before: float
    p50_after: float


def match(root: Path, to: int, *, policy: GpuPolicy | None = None) -> tuple[list[MatchRow], ShotScope]:
    """Write a matching grade under every measurable shot but `to`; returns per-shot changes and
    the target's numbers. The spec file is restored if the rewrite does not validate."""
    policy = policy if policy is not None else load_policy()
    path = root / "montaj.yaml"
    log = root / "build" / "scopes.log"
    spec = load_spec(path)
    text = path.read_text(encoding="utf-8")
    n = len(spec.shots)
    if not 0 <= to < n:
        raise ValueError(f"--to {to}: film has shots 0..{n - 1}")
    with gpulock.hold(log):
        # Measure ungraded pictures for the others (their old grade is replaced, not stacked).
        bare = [s if i == to else s.model_copy(update={"grade": None}) for i, s in enumerate(spec.shots)]
        r = _renderer(spec.model_copy(update={"shots": bare}), root, policy, log)
        if not measurable(r.tl, to):
            raise ValueError(f"--to {to}: shot {to} is a {r.tl.shots[to].kind}; pick a picture shot")
        pics, _ = _pictures(r, to)
        if not pics:
            raise ValueError(f"--to {to}: shot {to} is never on screen alone")
        target = measure(pics, to)
        solved: dict[int, Grade] = {}
        rows: list[MatchRow] = []
        for i in range(n):
            if i == to or not measurable(r.tl, i):
                continue
            pics, _ = _pictures(r, i)
            if not pics:
                continue
            lin = to_linear(_flat(pics))
            old = spec.shots[i].grade if isinstance(spec.shots[i].grade, Grade) else Grade()
            before = float(_numbers(_graded(lin, old) if spec.shots[i].grade is not None else _flat(pics))[1])
            solved[i] = solve(lin, old, target)
            rows.append(MatchRow(i, old, solved[i], before, 0.0))
            _log(log, f"match: shot {i} -> {solved[i].model_dump(include=set(MATCH_FIELDS))}")
    if not solved:
        raise ValueError("match: no other picture shot to match")
    path.write_text(write_grades(text, solved, n), encoding="utf-8")
    try:
        new = load_spec(path)
    except Exception:
        path.write_text(text, encoding="utf-8")
        raise
    with gpulock.hold(log):
        r = _renderer(new, root, policy, log)
        for row in rows:
            pics, _ = _pictures(r, row.shot)
            row.p50_after = measure(pics, row.shot).p50
    return rows, target


def match_line(row: MatchRow, target: ShotScope) -> str:
    def ch(name: str, fmt: str) -> str:
        return f"{name} {format(getattr(row.before, name), fmt)}→{format(getattr(row.after, name), fmt)}"

    return (f"shot {row.shot}  {ch('exposure', '+.2f')}  {ch('temp', '+.0f')}  {ch('tint', '+.1f')}  "
            f"{ch('saturation', '.2f')}  p50 {row.p50_before:.2f}→{row.p50_after:.2f} (target {target.p50:.2f})")


# --- check -------------------------------------------------------------------------------------

def _scene_key(tl: Timeline, i: int) -> str | None:
    sh = tl.shots[i].spec
    return str(sh.clip or sh.photo) if (sh.clip or sh.photo) else None


def warnings(stats: list[ShotScope], tl: Timeline) -> list[str]:
    out = []
    for s in stats:
        if s.clip_hi > CLIP_HI_WARN:
            out.append(f"WARN scopes shot {s.shot}: clip_hi {s.clip_hi * 100:.1f}% > {CLIP_HI_WARN * 100:.0f}% "
                       "(lower grade.exposure or gain)")
        if s.skin_hue is not None and not on_skin_line(s.skin_hue):
            out.append(f"WARN scopes shot {s.shot}: skin hue {s.skin_hue:.0f}° off the skin line "
                       f"({SKIN_LINE[0]:.0f}–{SKIN_LINE[1]:.0f}°) (adjust grade.temp/tint)")
    by = {s.shot: s for s in stats}
    for i in range(len(tl.shots) - 1):
        a, b = by.get(i), by.get(i + 1)
        key = _scene_key(tl, i)
        if a is None or b is None or key is None or key != _scene_key(tl, i + 1):
            continue
        if abs(a.p50 - b.p50) > LUMA_JUMP:
            out.append(f"WARN scopes shots {i}→{i + 1}: p50 luma jumps {b.p50 - a.p50:+.2f} within one scene "
                       f"(montaj match --to {i})")
    return out


def check_lines(root: Path, spec: Spec, *, policy: GpuPolicy | None = None) -> list[str]:
    policy = policy if policy is not None else load_policy()
    log = root / "build" / "scopes.log"
    try:
        with gpulock.hold(log):
            r = _renderer(spec, root, policy, log)
            stats = scope_shots(r, root)
    except GpuStrictError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        # check reads a finished file; a spec whose pictures no longer render is not a scope finding.
        _log(log, f"scopes: unavailable: {exc}")
        return [f"INFO scopes: unavailable ({' '.join(str(exc).split())[:120]})"]
    return warnings(stats, r.tl)


def as_dict(s: ShotScope) -> dict:
    return asdict(s)
