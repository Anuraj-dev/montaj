"""Radial chromatic split: red pushed out, blue pulled in, stronger toward the edges (a lens effect)."""
import torch


def draw(ctx):
    a = ctx.params.get("amount", 0.01)
    img, uv = ctx.inputs[0], ctx.uv
    r = uv.norm(dim=-1, keepdim=True)
    red = ctx.sample(img, uv * (1 - a * r))[0]
    blue = ctx.sample(img, uv * (1 + a * r))[2]
    return torch.stack((red, img[1], blue))
