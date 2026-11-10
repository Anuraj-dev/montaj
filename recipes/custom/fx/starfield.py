"""Three parallax layers of stars drifting toward the viewer over a faint nebula."""
import torch


def stars(ctx, p, density, seed):
    """Point stars on a jittered grid: one candidate per cell, about 1 in 6 lit."""
    q = p * density
    cell = torch.floor(q)
    jitter = torch.stack((ctx.hash(cell, seed), ctx.hash(cell, seed + 1)), dim=-1)
    d = (q - cell - (0.15 + 0.7 * jitter)).norm(dim=-1)
    lit = (ctx.hash(cell, seed + 2) > 0.83).float()
    size = 0.03 + 0.05 * ctx.hash(cell, seed + 3)
    return lit * torch.exp(-(d / size) ** 2)


def draw(ctx):
    speed, seed = ctx.params.get("speed", 0.5), int(ctx.params.get("seed", 0))
    uv = ctx.uv
    sky = torch.zeros(ctx.H, ctx.W, device=ctx.device)
    for layer in range(3):
        zoom = 1.0 / (1.0 + speed * ctx.t * (0.15 + 0.25 * layer))
        sky = sky + stars(ctx, uv * zoom, 18.0 - 5.0 * layer, seed + 10 * layer) * (0.45 + 0.3 * layer)
    neb = ctx.fbm(uv * 1.6 + torch.tensor([0.05 * ctx.t, 0.0], device=ctx.device), octaves=5, seed=seed)
    neb = ctx.smoothstep(0.45, 0.85, neb)
    tint = ctx.palette(neb * 0.6 + 0.55, d=(0.1, 0.25, 0.55)) * neb * 0.35
    return (tint + sky.clamp(0, 1)[None] * ctx.rgb("#dfe8ff")).clamp(0, 1)
