"""A bar chart that grows in: axis draws, bars pop up one by one, a dashed capacity line slides across.

params: values (0..1 each, chart-height fractions), highlight (bar index drawn in the accent, -1 none),
cap (capacity level 0..1, omit for none), cap_at (s), start (s before the first bar), stagger (s), grow (s).
"""
import math

import torch

BG_TOP, BG_BOTTOM, INK, MUTED, ACCENT = "#16213a", "#0b1020", "#f4efe6", "#4dd6ff", "#ffb84d"
X0, X1, BASE, HEIGHT = -0.55, 1.55, -0.22, 0.82  # chart box in uv (right of centre, above the lower third)


def clamp01(x):
    return min(1.0, max(0.0, x))


def back_out(x):
    """Overshoots ~10% then settles: the bars land with a little bounce."""
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def hline(ctx, y, x0, x1, half):
    """Coverage of a horizontal stroke from x0 to x1 at height y."""
    return ctx.fill(ctx.sd_box(ctx.uv, ((x1 - x0) / 2, half), ((x0 + x1) / 2, y)))


def ground(ctx):
    """Navy gradient under a faint dot grid that drifts slowly, so a finished diagram never freezes."""
    uv = ctx.uv
    out = ctx.mix(ctx.rgb(BG_BOTTOM), ctx.rgb(BG_TOP), ((uv[..., 1] + 1) * 0.5)[None])
    q = (uv + torch.tensor([0.02 * ctx.t, 0.012 * ctx.t], device=ctx.device)) / 0.16
    d = (q - torch.floor(q) - 0.5).norm(dim=-1) * 0.16
    return out + (ctx.fill(d - 0.006) * 0.12)[None] * ctx.rgb("#9fb4d8")


def draw(ctx):
    P = ctx.params
    values = [float(v) for v in P.get("values", [0.3, 0.45, 0.6, 0.8, 0.95])]
    hi = int(P.get("highlight", -1))
    start, stagger, grow = float(P.get("start", 0.4)), float(P.get("stagger", 0.35)), float(P.get("grow", 0.7))
    t, uv = ctx.t, ctx.uv
    out = ground(ctx)

    # Axis and gridlines draw left to right over the first 0.6 s.
    reveal = clamp01(t / 0.6)
    xr = X0 + (X1 - X0) * reveal
    if reveal > 0:
        for k in (0.25, 0.5, 0.75, 1.0):
            out = ctx.mix(out, ctx.rgb(INK), 0.10 * hline(ctx, BASE + HEIGHT * k, X0, xr, 0.0018)[None])
        out = ctx.mix(out, ctx.rgb(INK), hline(ctx, BASE, X0, xr, 0.005)[None])

    n = len(values)
    slot = (X1 - X0) / n
    for i, v in enumerate(values):
        x = clamp01((t - start - i * stagger) / grow)
        h = HEIGHT * v * back_out(x) if x > 0 else 0.0
        if h <= 1e-3:
            continue
        cx = X0 + slot * (i + 0.5)
        bar = ctx.fill(ctx.sd_box(uv, (slot * 0.3, h / 2), (cx, BASE + h / 2)) - 0.012)
        colour = ACCENT if i == hi else MUTED
        lift = 1.0 if i == hi else 0.75
        if i == hi:  # the highlighted bar breathes once it has landed
            lift += 0.15 * math.sin(max(0.0, t - start - i * stagger - grow) * 4.0)
        out = ctx.mix(out, ctx.rgb(colour) * lift, bar[None])

    cap = P.get("cap")
    if cap is not None:
        y = BASE + HEIGHT * float(cap)
        a = clamp01((t - float(P.get("cap_at", 3.0))) / 0.8)
        if a > 0:
            xr = X0 - 0.1 + (X1 - X0 + 0.2) * a
            dash = (torch.remainder(uv[..., 0] * 18.0 - t * 2.0, 1.0) < 0.55).float()
            stroke = hline(ctx, y, X0 - 0.1, xr, 0.006) * dash
            out = ctx.mix(out, ctx.rgb(INK), stroke[None])
    return out.clamp(0, 1)
