"""A ragged portal opens from the centre: the outgoing picture swirls into it, the incoming shows inside."""
import torch


def draw(ctx):
    old, new = ctx.inputs
    uv = ctx.uv
    r, a = ctx.polar(uv)
    reach = float(uv.norm(dim=-1).max()) + 0.3
    radius = reach * (ctx.p * ctx.p * (3 - 2 * ctx.p)) ** 1.6
    edge = ctx.fbm(torch.stack((torch.cos(a) * 2.5, torch.sin(a) * 2.5 + ctx.t * 3), dim=-1), octaves=3)
    d = r - radius * (0.8 + 0.4 * edge)
    inside = ctx.fill(d, soft=0.02)
    # Pull the outgoing picture around the opening, harder near the rim.
    twist = 2.5 * ctx.p * torch.exp(-(d.clamp(min=0) * 3))
    c, s = torch.cos(twist), torch.sin(twist)
    warp = torch.stack((uv[..., 0] * c - uv[..., 1] * s, uv[..., 0] * s + uv[..., 1] * c), dim=-1)
    pulled = ctx.sample(old, warp)
    rim = torch.exp(-(d / 0.035) ** 2) * (1 - (2 * ctx.p - 1) ** 2)
    color = ctx.rgb(ctx.params.get("rim", "#ffffff"))
    return (ctx.mix(pulled, new, inside[None]) + color * rim[None]).clamp(0, 1)
