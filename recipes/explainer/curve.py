"""A line graph that draws itself: wait time W = rho / (1 - rho) against utilisation rho, with a glowing tip.

params: start (s before the line starts), draw (s to draw it), drawn (0..1 already drawn at t=0),
zone (rho where the danger zone starts, omit for none), zone_at (s).
"""
import math

import torch

BG_TOP, BG_BOTTOM, INK, LINE, ACCENT = "#16213a", "#0b1020", "#f4efe6", "#4dd6ff", "#ffb84d"
X0, X1, Y0, Y1 = -0.55, 1.55, -0.22, 0.62  # plot box in uv
RHO_MAX, W_MAX = 0.95, 10.0                 # x axis spans rho 0..0.95; the y axis tops out at W = 10


def clamp01(x):
    return min(1.0, max(0.0, x))


def smooth(x):
    return x * x * (3 - 2 * x)


def box(ctx, x0, y0, x1, y1):
    return ctx.fill(ctx.sd_box(ctx.uv, ((x1 - x0) / 2, (y1 - y0) / 2), ((x0 + x1) / 2, (y0 + y1) / 2)))


def ground(ctx):
    """Navy gradient under a faint dot grid that drifts slowly, so a finished diagram never freezes."""
    uv = ctx.uv
    out = ctx.mix(ctx.rgb(BG_BOTTOM), ctx.rgb(BG_TOP), ((uv[..., 1] + 1) * 0.5)[None])
    q = (uv + torch.tensor([0.02 * ctx.t, 0.012 * ctx.t], device=ctx.device)) / 0.16
    d = (q - torch.floor(q) - 0.5).norm(dim=-1) * 0.16
    return out + (ctx.fill(d - 0.006) * 0.12)[None] * ctx.rgb("#9fb4d8")


def draw(ctx):
    P = ctx.params
    t, uv = ctx.t, ctx.uv
    out = ground(ctx)
    sx, sy = (X1 - X0) / RHO_MAX, (Y1 - Y0) / W_MAX

    # Axes grow out of the origin.
    a = smooth(clamp01(t / 0.6)) if P.get("drawn", 0) < 1 else 1.0
    out = ctx.mix(out, ctx.rgb(INK), box(ctx, X0 - 0.005, Y0 - 0.005, X0 + (X1 - X0) * a, Y0 + 0.005)[None])
    out = ctx.mix(out, ctx.rgb(INK), box(ctx, X0 - 0.005, Y0 - 0.005, X0 + 0.005, Y0 + (Y1 - Y0) * a)[None])

    zone = P.get("zone")
    if zone is not None:
        za = smooth(clamp01((t - float(P.get("zone_at", 0.5))) / 0.8))
        xz = X0 + float(zone) * sx
        shade = box(ctx, xz, Y0 + 0.006, X1, Y1) * (0.13 + 0.04 * math.sin(t * 3.0)) * za
        out = ctx.mix(out, ctx.rgb(ACCENT), shade[None])
        dash = (torch.remainder(uv[..., 1] * 16.0 + t, 1.0) < 0.5).float()
        out = ctx.mix(out, ctx.rgb(ACCENT), (box(ctx, xz - 0.004, Y0, xz + 0.004, Y0 + (Y1 - Y0) * za) * dash)[None])

    # The curve: vertical distance over the slope gives a cheap, even stroke width.
    drawn = float(P.get("drawn", 0.0))
    prog = clamp01(drawn + (1 - drawn) * clamp01((t - float(P.get("start", 0.5))) / float(P.get("draw", 3.0))))
    if prog > 0:
        rho = ((uv[..., 0] - X0) / sx).clamp(0.0, RHO_MAX)
        w = rho / (1 - rho)
        y = Y0 + w.clamp(max=W_MAX * 1.2) * sy
        slope = sy / sx / (1 - rho) ** 2
        d = (uv[..., 1] - y).abs() / torch.sqrt(1 + slope ** 2) - 0.006
        rho_tip = RHO_MAX * prog
        inside = (uv[..., 0] >= X0) & (uv[..., 0] <= X0 + rho_tip * sx) & (uv[..., 1] <= Y1 + 0.02)
        out = ctx.mix(out, ctx.rgb(LINE), (ctx.fill(d) * inside.float())[None])
        # Glowing tip. Once the line is drawn it becomes a probe sliding along the curve, with guides
        # dropped to both axes, so the frame keeps moving while the narration reads the graph.
        if prog >= 1:
            done = t - float(P.get("start", 0.5)) - float(P.get("draw", 3.0)) * (1 - drawn)
            rho_tip = RHO_MAX * (0.72 - 0.2 * math.cos(max(0.0, done) * 1.4))
        w_tip = min(rho_tip / (1 - rho_tip), W_MAX)
        if prog >= 1:
            gx, gy = X0 + rho_tip * sx, Y0 + w_tip * sy
            guides = box(ctx, gx - 0.003, Y0, gx + 0.003, gy) + box(ctx, X0, gy - 0.003, gx, gy + 0.003)
            out = ctx.mix(out, ctx.rgb(ACCENT), (0.45 * guides.clamp(0, 1))[None])
        tip = (X0 + rho_tip * sx, Y0 + w_tip * sy)
        r = 0.022 + 0.006 * math.sin(t * 5.0)
        out = ctx.mix(out, ctx.rgb(ACCENT), ctx.fill(ctx.sd_circle(uv, r, tip))[None])
        halo = torch.exp(-(ctx.sd_circle(uv, 0.0, tip) / 0.08) ** 2) * 0.35
        out = out + halo[None] * ctx.rgb(ACCENT)
    return out.clamp(0, 1)
