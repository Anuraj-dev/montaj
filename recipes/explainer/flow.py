"""A queue diagram: arrivals -> queue box -> server node(s), with packets flowing along the edges.

params: servers (1 or more), backlog ([at t=0, at the end] items waiting, 0..6), span (s the backlog
change takes), rate (packets per second), start (s before the nodes pop in).
"""
import math

import torch

BG_TOP, BG_BOTTOM, INK, NODE, ACCENT = "#16213a", "#0b1020", "#f4efe6", "#4dd6ff", "#ffb84d"
SRC, QUEUE, SRV_X, MID_Y = (-1.25, 0.2), (0.15, 0.2), 1.25, 0.2


def clamp01(x):
    return min(1.0, max(0.0, x))


def back_out(x):
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def segment(ctx, a, b):
    """Distance from every pixel to the segment a-b."""
    A = torch.tensor(a, device=ctx.device)
    ba = torch.tensor(b, device=ctx.device) - A
    pa = ctx.uv - A
    h = ((pa * ba).sum(-1) / float((ba * ba).sum())).clamp(0, 1)
    return (pa - ba * h[..., None]).norm(dim=-1)


def lerp2(a, b, x):
    return (a[0] + (b[0] - a[0]) * x, a[1] + (b[1] - a[1]) * x)


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
    n = max(1, int(P.get("servers", 1)))
    start, rate = float(P.get("start", 0.3)), float(P.get("rate", 1.4))
    out = ground(ctx)
    servers = [(SRV_X, MID_Y + (i - (n - 1) / 2) * 0.42) for i in range(n)]
    q_in, q_out = (QUEUE[0] - 0.36, QUEUE[1]), (QUEUE[0] + 0.36, QUEUE[1])

    # Edges draw in after the nodes they join.
    edges = [(SRC, q_in, start + 0.3)] + [(q_out, s, start + 0.8 + 0.15 * i) for i, s in enumerate(servers)]
    for a, b, at in edges:
        g = clamp01((t - at) / 0.5)
        if g > 0:
            out = ctx.mix(out, ctx.rgb(INK), (0.55 * ctx.fill(segment(ctx, a, lerp2(a, b, g)) - 0.004))[None])

    # Nodes pop in: source, queue, then the servers.
    pops = [(SRC, 0.0)] + [(s, 0.6 + 0.15 * i) for i, s in enumerate(servers)]
    for c, delay in pops:
        g = clamp01((t - start - delay) / 0.4)
        if g > 0:
            r = 0.13 * back_out(g)
            ring = ctx.fill((ctx.sd_circle(uv, r, c)).abs() - 0.008)
            core = ctx.fill(ctx.sd_circle(uv, r * 0.55, c))
            out = ctx.mix(out, ctx.rgb(NODE), torch.maximum(ring, 0.8 * core)[None])
    g = clamp01((t - start - 0.3) / 0.4)
    if g > 0:
        frame = ctx.fill(ctx.sd_box(uv, (0.36 * back_out(g), 0.14), QUEUE).abs() - 0.007)
        out = ctx.mix(out, ctx.rgb(INK), frame[None])

    # Backlog: items stacked right-to-left in the queue box, count eased between the two params.
    b0, b1 = (float(v) for v in P.get("backlog", [1, 6]))
    span = float(P.get("span", 4.0))
    count = b0 + (b1 - b0) * clamp01((t - start - 0.6) / span) if g > 0 else 0.0
    for k in range(6):
        a = clamp01(count - k)
        if a > 0:
            c = (q_out[0] - 0.07 - k * 0.112, QUEUE[1])
            item = ctx.fill(ctx.sd_box(uv, (0.04, 0.08), c) - 0.008)
            out = ctx.mix(out, ctx.rgb(ACCENT), (a * item)[None])

    # Packets: arrivals on the first edge, departures to each server, steady so the frame keeps moving.
    live = t - start - 1.2
    if live > 0:
        for j in range(3):
            x = (live * rate * 0.5 + j / 3) % 1.0
            dot = ctx.fill(ctx.sd_circle(uv, 0.03, lerp2(SRC, q_in, x)))
            out = ctx.mix(out, ctx.rgb(ACCENT), dot[None])
        for i, s in enumerate(servers):
            x = (live * rate * 0.5 / max(1.0, 2.0 - 0.5 * (n - 1)) + i * 0.37) % 1.0
            dot = ctx.fill(ctx.sd_circle(uv, 0.026, lerp2(q_out, s, x)))
            out = ctx.mix(out, ctx.rgb(INK), dot[None])
        halo = sum(torch.exp(-(ctx.sd_circle(uv, 0.0, s) / 0.2) ** 2) for s in servers)
        out = out + (0.12 + 0.05 * math.sin(t * 4.0)) * halo[None] * ctx.rgb(NODE)
    return out.clamp(0, 1)
