"""A diagonal wipe with a bright accent edge: the next diagram slides in behind a moving line.

params: accent ('#rrggbb'), angle (degrees of the wipe direction, 0 = left to right).
"""
import math


def draw(ctx):
    out_shot, in_shot = ctx.inputs
    ang = math.radians(float(ctx.params.get("angle", 20.0)))
    s = ctx.uv[..., 0] * math.cos(ang) + ctx.uv[..., 1] * math.sin(ang)
    reach = ctx.W / min(ctx.W, ctx.H) + 0.3  # far enough that p = 0 and p = 1 cover the whole frame
    p = ctx.p * ctx.p * (3 - 2 * ctx.p)
    edge = -reach + 2 * reach * p
    m = ctx.smoothstep(edge + 0.02, edge - 0.02, s)
    line = ctx.fill((s - edge).abs() - 0.012) * math.sin(math.pi * ctx.p)
    mixed = ctx.mix(out_shot, in_shot, m[None])
    return ctx.mix(mixed, ctx.rgb(ctx.params.get("accent", "#ffb84d")), line[None]).clamp(0, 1)
