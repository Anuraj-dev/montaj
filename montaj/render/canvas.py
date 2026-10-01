"""Canvas: output geometry plus film.py's seeded noise. No module-level state."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class Canvas:
    """Output surface. k maps design px (1080 wide) to device px; noise matches film.py."""

    def __init__(self, W: int, H: int, device: str = "cuda", focal: float = 1400.0):
        self.W, self.H, self.device = int(W), int(H), device
        self.k = self.W / 1080.0
        self.foc = float(focal) * self.k
        # film.py order: seed once, lownoise draw, then wall-noise draw, same bicubic upsamples.
        # A private generator instead of the global one so concurrent Canvases cannot interleave draws.
        gen = torch.Generator(device=device).manual_seed(7)
        u = (torch.arange(self.W, device=device).float() + 0.5 - self.W / 2) / self.foc
        v = (torch.arange(self.H, device=device).float() + 0.5 - self.H / 2) / self.foc
        self.GA = u[None, :].expand(self.H, self.W)
        self.GB = v[:, None].expand(self.H, self.W)
        self.YY, self.XX = torch.meshgrid(
            torch.arange(self.H, device=device).float(),
            torch.arange(self.W, device=device).float(),
            indexing="ij",
        )
        n = F.interpolate(torch.rand(1, 1, 24, 14, device=device, generator=gen), (self.H, self.W),
                          mode="bicubic", align_corners=False)[0, 0]
        self.lownoise = (n - n.mean()) / n.std()
        del n
        self.wall_noise = F.interpolate(torch.rand(1, 1, 64, 64, device=device, generator=gen), (512, 512),
                                        mode="bicubic", align_corners=False)
