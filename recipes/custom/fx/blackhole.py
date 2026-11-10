"""A Schwarzschild-style black hole: lensed star background, tilted accretion disk, photon ring."""
import math

import torch


def stars(ctx, p, density, seed):
    q = p * density
    cell = torch.floor(q)
    jitter = torch.stack((ctx.hash(cell, seed), ctx.hash(cell, seed + 1)), dim=-1)
    d = (q - cell - (0.15 + 0.7 * jitter)).norm(dim=-1)
    lit = (ctx.hash(cell, seed + 2) > 0.83).float()
    return lit * torch.exp(-(d / (0.03 + 0.05 * ctx.hash(cell, seed + 3))) ** 2)


def disk(ctx, p, rs, tilt, spin, seed):
    """Emission of a thin disk from 1.6 rs to 4 rs, seen at `tilt` (y squashed), turbulent and rotating."""
    q = torch.stack((p[..., 0], p[..., 1] / tilt), dim=-1)
    r, a = ctx.polar(q)
    band = ctx.smoothstep(1.5 * rs, 1.9 * rs, r) * ctx.smoothstep(4.2 * rs, 2.4 * rs, r)
    swirl = a + spin * ctx.t * (2.0 * rs / r.clamp(min=1e-3)) ** 1.5
    flow = torch.stack((torch.cos(swirl) * 3.0, torch.sin(swirl) * 3.0 + r * 9.0), dim=-1)
    turb = ctx.fbm(flow, octaves=4, seed=seed)
    heat = band * (0.35 + 0.9 * turb) * (rs * 2.2 / r.clamp(min=rs)) ** 1.4
    doppler = 1.0 + 0.55 * torch.cos(a + math.pi)  # the side moving toward us is brighter
    return heat * doppler


def draw(ctx):
    grow = ctx.p * ctx.p * (3 - 2 * ctx.p)  # the hole swells slightly across the shot
    rs = ctx.params.get("mass", 0.2) * (0.85 + 0.15 * grow)
    tilt, spin = ctx.params.get("tilt", 0.3), ctx.params.get("spin", 0.4)
    seed = int(ctx.params.get("seed", 0))
    uv = ctx.uv
    r = uv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    # Lens: light passing at radius r appears bent outward by about rs^2 / r (weak-field Einstein ring).
    bent = uv * (1.0 - (2.6 * rs * rs) / (r * r))
    sky = stars(ctx, bent, 14.0, seed) + 0.6 * stars(ctx, bent * 1.7, 20.0, seed + 7)
    sky = sky * ctx.smoothstep(2.6 * rs, 3.2 * rs, r[..., 0])  # nothing escapes from inside the shadow

    front = uv[..., 1] < 0  # the near half of the disk is drawn over the shadow
    heat_near = disk(ctx, uv, rs, tilt, spin, seed)
    # The far half is lensed up over the top of the hole, so it shows as an arc above the shadow.
    arc = torch.stack((uv[..., 0], (uv[..., 1] - 0.2 * rs) * 0.55), dim=-1)
    heat_far = disk(ctx, arc, rs * 0.92, 1.0, spin, seed + 1) * 0.8

    shadow = ctx.fill(ctx.sd_circle(uv, 2.6 * rs), soft=0.01)
    ring = torch.exp(-((r[..., 0] - 2.62 * rs) / (0.035 * rs)) ** 2)
    heat = torch.where(front, heat_near, heat_near * (1 - shadow)) + heat_far * (1 - shadow) + 1.4 * ring
    fire = ctx.palette(heat.clamp(0, 1.2) * 0.42 + 0.02, a=(0.6, 0.4, 0.25), b=(0.5, 0.4, 0.3),
                       c=(1.0, 1.0, 1.0), d=(0.0, 0.08, 0.2))
    glow = fire * heat.clamp(0, 2)[None]
    glow = glow + 0.35 * ctx.blur(glow.clamp(0, 1), 40.0)
    pic = sky.clamp(0, 1)[None] * ctx.rgb("#d8e2ff") * (1 - shadow)[None] + glow
    return pic.clamp(0, 1)
