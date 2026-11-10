"""`custom:` shots, effects and transitions: an agent's torch `draw(ctx)` run on the canvas device.

A spec that names a `custom:` file executes that file in-process with its author's trust, the same
trust `music compose` gives its script. Montaj does not sandbox it.

Contract for the file: a top-level `def draw(ctx)` that is a pure function of `ctx` (no file or
network IO, no module-level state) and returns a float32 tensor on `ctx.device`:
(3, H, W) RGB in 0..1, or (4, H, W) premultiplied RGBA composited over `ctx.inputs[0]`
(or the video background when there are no inputs). Values are clamped to 0..1.

The segment hash covers the file's bytes, its `params` and HELPER_VERSION, so bump HELPER_VERSION
whenever a helper below changes a pixel.
"""
from __future__ import annotations

import copy
import math
import time
import traceback
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from montaj.render.shots import css_blur

# Reaches every segment hash that runs a custom file. Bump when a helper's output changes.
HELPER_VERSION = 1

_SHAPE_HINT = "return (3, H, W) RGB or (4, H, W) premultiplied RGBA, H, W = ctx.H, ctx.W"


class CustomError(RuntimeError):
    """`custom <file>:<line>: <problem> (<hint>)`, one line for `ERR`."""

    def __init__(self, rel: str, line: int | None, problem: str, hint: str):
        self.rel, self.line, self.problem, self.hint = rel, line, problem, hint
        where = rel if line is None else f"{rel}:{line}"
        super().__init__(f"custom {where}: {problem} ({hint})")


def _line_in(exc: BaseException, filename: str) -> int | None:
    """The deepest traceback line inside the custom file."""
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == filename:
            line = frame.lineno
    if line is None and isinstance(exc, SyntaxError) and exc.filename == filename:
        line = exc.lineno
    return line


def load_draw(path: Path, rel: str) -> Callable:
    """Compile and run the file in a fresh namespace; return its `draw`. Errors name file:line."""
    filename = str(path)
    try:
        source = path.read_bytes()
    except OSError as e:
        raise CustomError(rel, None, f"cannot read: {e.strerror or e}", "check the path") from e
    try:
        code = compile(source, filename, "exec")
    except SyntaxError as e:
        raise CustomError(rel, e.lineno, f"syntax error: {e.msg}", "fix the file") from e
    ns: dict = {"__name__": f"montaj_custom_{path.stem}", "__file__": filename}
    try:
        exec(code, ns)
    except Exception as e:
        raise CustomError(rel, _line_in(e, filename), f"{type(e).__name__}: {e}", "fix the module top level") from e
    draw = ns.get("draw")
    if not callable(draw):
        raise CustomError(rel, None, "no top-level draw(ctx)", "define def draw(ctx): ... return rgb")
    return draw


class Ctx:
    """What `draw` sees. Coordinates: `uv[..., 0]` right, `uv[..., 1]` up, the short side spans [-1, 1].

    Fields: uv (H, W, 2), t (seconds into the shot or transition), p (progress 0..1), f (film frame,
    fractional under motion blur), fps, beat (beats since beat0, None without bpm), params (dict),
    inputs (list of (3, H, W): [] for a shot, [layer] for an effect, [outgoing, incoming] for a
    transition), W, H, k (W / 1080), device.
    """

    def __init__(self, *, uv, t, p, f, fps, beat, params, inputs, W, H, k, device):
        self.uv, self.t, self.p, self.f, self.fps, self.beat = uv, t, p, f, fps, beat
        self.params, self.inputs = params, inputs
        self.W, self.H, self.k, self.device = W, H, k, device
        self._extent = (W / min(W, H), H / min(W, H))

    # --- noise -------------------------------------------------------------------------------
    def noise(self, seed: int = 0) -> torch.Tensor:
        """(H, W) white noise in [0, 1), the same for the same seed."""
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        return torch.rand(self.H, self.W, device=self.device, generator=gen)

    def hash(self, p: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """Integer lattice hash of p (..., 2) to [0, 1). Same input, same output."""
        q = torch.floor(p).to(torch.int64)
        h = (q[..., 0] * 374761393 + q[..., 1] * 668265263 + int(seed) * 144665) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        h = h ^ (h >> 16)
        return h.to(torch.float32) / 4294967296.0

    def vnoise(self, p: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """Smooth value noise in [0, 1) at points p (..., 2); one cell per unit."""
        i = torch.floor(p)
        fr = p - i
        u = fr * fr * (3 - 2 * fr)
        one = torch.tensor([1.0, 0.0], device=p.device)
        a = self.hash(i, seed)
        b = self.hash(i + one, seed)
        c = self.hash(i + one.flip(0), seed)
        d = self.hash(i + 1.0, seed)
        return torch.lerp(torch.lerp(a, b, u[..., 0]), torch.lerp(c, d, u[..., 0]), u[..., 1])

    def fbm(self, p: torch.Tensor, octaves: int = 5, seed: int = 0) -> torch.Tensor:
        """Fractal value noise in [0, 1): `octaves` layers, each twice the frequency, half the weight."""
        total, amp, norm = torch.zeros(p.shape[:-1], device=p.device), 0.5, 0.0
        for o in range(int(octaves)):
            total = total + amp * self.vnoise(p * (2.0 ** o), seed + o)
            norm += amp
            amp *= 0.5
        return total / norm

    # --- shapes ------------------------------------------------------------------------------
    def sd_circle(self, p: torch.Tensor, r: float, center=(0.0, 0.0)) -> torch.Tensor:
        """Signed distance to a circle (negative inside), in uv units."""
        c = torch.tensor(center, device=p.device, dtype=p.dtype)
        return (p - c).norm(dim=-1) - r

    def sd_box(self, p: torch.Tensor, half, center=(0.0, 0.0)) -> torch.Tensor:
        """Signed distance to an axis-aligned box with half-size `half` (w, h)."""
        c = torch.tensor(center, device=p.device, dtype=p.dtype)
        hs = torch.tensor(half, device=p.device, dtype=p.dtype)
        d = (p - c).abs() - hs
        return d.clamp(min=0).norm(dim=-1) + d.max(dim=-1).values.clamp(max=0)

    def fill(self, d: torch.Tensor, soft: float | None = None) -> torch.Tensor:
        """Coverage 0..1 of a signed distance: 1 inside, antialiased over `soft` (default 1.5 px)."""
        s = soft if soft is not None else 3.0 / min(self.W, self.H)
        return self.smoothstep(s, -s, d)

    def polar(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(radius, angle in radians) of p (..., 2)."""
        return p.norm(dim=-1), torch.atan2(p[..., 1], p[..., 0])

    # --- colour and images -------------------------------------------------------------------
    def palette(self, t, a=(0.5, 0.5, 0.5), b=(0.5, 0.5, 0.5), c=(1.0, 1.0, 1.0), d=(0.0, 0.33, 0.67)):
        """Cosine palette a + b*cos(2π(c*t + d)): (3, *t.shape)."""
        t = torch.as_tensor(t, device=self.device, dtype=torch.float32)
        shape = (3,) + (1,) * t.dim()
        A, B, C, D = (torch.tensor(v, device=self.device).view(shape) for v in (a, b, c, d))
        return A + B * torch.cos(2 * math.pi * (C * t + D))

    def rgb(self, hexstr: str) -> torch.Tensor:
        """`#rrggbb` as a (3, 1, 1) tensor, broadcastable against (3, H, W)."""
        h = hexstr.lstrip("#")
        vals = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        return torch.tensor(vals, device=self.device).view(3, 1, 1)

    def sample(self, img: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Read image (C, H, W) at uv points p (h, w, 2) with bilinear filtering, edges clamped."""
        ex, ey = self._extent
        grid = torch.stack((p[..., 0] / ex, -p[..., 1] / ey), dim=-1)
        return F.grid_sample(img[None], grid[None], mode="bilinear", padding_mode="border",
                             align_corners=False)[0]

    def blur(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
        """Gaussian blur of (C, H, W); sigma in design px (1080 wide)."""
        return css_blur(img, float(sigma) * self.k)

    @staticmethod
    def smoothstep(e0, e1, x):
        t = ((x - e0) / (e1 - e0)).clamp(0, 1)
        return t * t * (3 - 2 * t)

    @staticmethod
    def mix(a, b, t):
        return a + (b - a) * t


def uv_grid(W: int, H: int, device) -> torch.Tensor:
    """(H, W, 2) pixel-centre coordinates: x right, y up, the short side spans [-1, 1]."""
    half = min(W, H) / 2.0
    x = (torch.arange(W, device=device, dtype=torch.float32) + 0.5 - W / 2) / half
    y = -(torch.arange(H, device=device, dtype=torch.float32) + 0.5 - H / 2) / half
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def check_output(out, rel: str, line: int, W: int, H: int, device) -> torch.Tensor:
    """Reject a non-tensor, the wrong device, dtype or shape, and NaN/inf. Never copies across devices."""
    if not isinstance(out, torch.Tensor):
        raise CustomError(rel, line, f"draw returned {type(out).__name__}, not a tensor", _SHAPE_HINT)
    want = torch.device(device)
    if out.device.type != want.type:
        raise CustomError(rel, line, f"draw returned a {out.device.type} tensor, the canvas is {want.type}",
                          "create tensors on ctx.device or derive them from ctx.uv")
    if out.dtype != torch.float32:
        raise CustomError(rel, line, f"draw returned {str(out.dtype).removeprefix('torch.')}, not float32",
                          "use float32 tensors (.float())")
    if out.dim() != 3 or out.shape[0] not in (3, 4) or tuple(out.shape[1:]) != (H, W):
        raise CustomError(rel, line, f"draw returned shape {tuple(out.shape)}, need ({{3,4}}, {H}, {W})", _SHAPE_HINT)
    if not bool(torch.isfinite(out).all()):
        raise CustomError(rel, line, "draw returned NaN or inf", "guard divisions, or torch.nan_to_num")
    return out


class CustomRunner:
    """Loads custom files once per Renderer and runs them. Holds no state beyond that cache."""

    def __init__(self, spec_dir: Path, W: int, H: int, device, fps: int, bpm: float | None, beat0: float,
                 log=None):
        self._dir = Path(spec_dir)
        self.W, self.H, self.device = int(W), int(H), device
        self.fps, self.bpm, self.beat0 = fps, bpm, beat0
        self._log = log
        self._draws: dict[str, Callable] = {}
        self._uv: torch.Tensor | None = None

    def _draw(self, rel: str) -> Callable:
        fn = self._draws.get(rel)
        if fn is None:
            fn = self._draws[rel] = load_draw(self._dir / rel, rel)
        return fn

    def draw_line(self, rel: str) -> int | None:
        code = getattr(self._draw(rel), "__code__", None)
        return None if code is None else code.co_firstlineno

    def call(self, rel: str, ctx: Ctx) -> torch.Tensor:
        """`draw(ctx)` with its exception turned into a CustomError at the file's line; output checked."""
        fn = self._draw(rel)
        filename = str(self._dir / rel)
        try:
            out = fn(ctx)
        except Exception as e:
            if self._log is not None:
                self._log(f"custom {rel}: frame {ctx.f:g}\n" + "".join(traceback.format_exception(e)))
            raise CustomError(rel, _line_in(e, filename), f"{type(e).__name__}: {_first_line(e)}",
                              "fix draw(ctx); traceback in build/render.log") from e
        return check_output(out, rel, self.draw_line(rel), self.W, self.H, self.device)

    def uv(self) -> torch.Tensor:
        if self._uv is None:
            self._uv = uv_grid(self.W, self.H, self.device)
        return self._uv

    def ctx(self, params: dict | None, *, t: float, p: float, f: float, inputs: list[torch.Tensor]) -> Ctx:
        """A fresh Ctx: params and inputs are copies, so a draw that writes into them changes nothing else."""
        beat = None if self.bpm is None else (f - self.beat0) / self.fps * self.bpm / 60.0
        return Ctx(
            uv=self.uv().clone(), t=t, p=min(1.0, max(0.0, p)), f=f, fps=self.fps, beat=beat,
            params=copy.deepcopy(params or {}), inputs=[x.clone() for x in inputs],
            W=self.W, H=self.H, k=self.W / 1080.0, device=self.device,
        )

    def run(self, rel: str, params: dict | None, *, t: float, p: float, f: float,
            inputs: list[torch.Tensor], background: torch.Tensor) -> torch.Tensor:
        """One call of `draw`. Returns (3, H, W) in 0..1."""
        out = self.call(rel, self.ctx(params, t=t, p=p, f=f, inputs=inputs)).clamp(0, 1)
        if out.shape[0] == 4:
            base = inputs[0] if inputs else background
            out = out[:3] + (1 - out[3:4]) * base
        return out


def probe(spec_dir: Path, rel: str, params: dict | None, *, kind: str, W: int, H: int, device,
          fps: int = 30, frames: int = 6, dur: float = 2.0) -> tuple[list[torch.Tensor], float, int]:
    """Run `draw` on `frames` evenly spaced times; check range and determinism.

    Returns (frames, ms per call, peak VRAM bytes or 0 on CPU). `kind` is shot, effect or
    transition and decides `ctx.inputs` (gradients stand in for real layers).
    """
    runner = CustomRunner(spec_dir, W, H, device, fps, None, 0.0)
    uv = runner.uv()
    ramp = torch.stack((uv[..., 0] * 0.25 + 0.5, uv[..., 1] * 0.25 + 0.5, torch.full_like(uv[..., 0], 0.5)))
    ramp = ramp.clamp(0, 1)
    n_in = {"shot": 0, "effect": 1, "transition": 2}[kind]
    inputs = [ramp, ramp.flip(-1).flip(-2)][:n_in]
    bg = torch.zeros(3, H, W, device=device)
    line = runner.draw_line(rel)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    out: list[torch.Tensor] = []
    total = 0.0
    n = max(2, int(frames))
    for i in range(n):
        p = i / (n - 1)
        kw = dict(t=p * dur, p=p, f=p * dur * fps, inputs=inputs)
        t0 = time.perf_counter()
        raw = runner.call(rel, runner.ctx(params, **kw))
        if cuda:
            torch.cuda.synchronize(device)
        total += time.perf_counter() - t0
        lo, hi = float(raw.min()), float(raw.max())
        if lo < -1e-3 or hi > 1 + 1e-3:
            raise CustomError(rel, line, f"values outside 0..1 at p={p:.2f} (min {lo:.3g}, max {hi:.3g})",
                              "clamp(0, 1) the result")
        if not torch.equal(raw, runner.call(rel, runner.ctx(params, **kw))):
            raise CustomError(rel, line, f"two calls at p={p:.2f} differ", "draw must be a pure function of ctx")
        if raw.shape[0] == 4:
            base = inputs[0] if inputs else bg
            raw = raw[:3] + (1 - raw[3:4]) * base
        out.append(raw)
    peak = torch.cuda.max_memory_allocated(device) if cuda else 0
    return out, total / n * 1000.0, int(peak)


def _first_line(e: BaseException) -> str:
    text = str(e).strip()
    return text.splitlines()[0] if text else "(no message)"
