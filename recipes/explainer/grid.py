"""Title ground: a navy gradient with a slowly drifting dot grid and a soft accent glow (never still)."""
import math

import torch


def draw(ctx):
    top, bottom = ctx.params.get("top", "#16213a"), ctx.params.get("bottom", "#0b1020")
    accent = ctx.params.get("accent", "#ffb84d")
    uv = ctx.uv
    ground = ctx.mix(ctx.rgb(bottom), ctx.rgb(top), ((uv[..., 1] + 1) * 0.5)[None])
    # Dot grid, 0.16 uv apart, drifting up and to the left at a constant rate.
    drift = torch.tensor([0.02 * ctx.t, 0.012 * ctx.t], device=ctx.device)
    q = (uv + drift) / 0.16
    d = (q - torch.floor(q) - 0.5).norm(dim=-1) * 0.16
    dots = ctx.fill(d - 0.006) * 0.16
    # A glow that breathes around the upper middle, where the title sits.
    glow = torch.exp(-((uv - torch.tensor([0.0, 0.25], device=ctx.device)).norm(dim=-1) / 0.9) ** 2)
    glow = glow * (0.10 + 0.03 * math.sin(ctx.t * 1.3))
    out = ground + dots[None] * ctx.rgb("#9fb4d8") + glow[None] * ctx.rgb(accent)
    return out.clamp(0, 1)
