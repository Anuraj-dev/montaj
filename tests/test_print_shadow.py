"""#50: the print shadow is built on the canvas device, bit-identical to film2's PIL build."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFilter

from montaj.render.core import print_texture
from montaj.render.pil_blur import gaussian_blur_u8


def _pil_print(photo: np.ndarray, P: int) -> tuple[tuple[int, int], np.ndarray]:
    """The PIL/numpy print build core.Photo used before #50 (film2's), as (H, W, 4)."""
    ch, cw = photo.shape[:2]
    b = int(0.035 * min(cw, ch))
    TW, TH = cw + 2 * b + 2 * P, ch + 2 * b + 2 * P
    card_a = np.zeros((TH, TW), np.float32)
    card_a[P:P + ch + 2 * b, P:P + cw + 2 * b] = 1
    sh = Image.new("L", (TW, TH), 0)
    ImageDraw.Draw(sh).rectangle((P + 10, P + 28, P + cw + 2 * b + 10, P + ch + 2 * b + 28), fill=255)
    sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(P / 2.5)), np.float32) / 255 * 0.7
    col = np.zeros((TH, TW, 3), np.float32)
    col[:] = (0.95, 0.92, 0.86)
    col[P + b:P + b + ch, P + b:P + b + cw] = photo
    a = card_a + (1 - card_a) * sh
    return (P + b, P + b), np.dstack([col * card_a[..., None], a])


def _devices() -> list:
    cuda = pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(
        not torch.cuda.is_available(), reason="no CUDA device")])
    return ["cpu", cuda]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("sigma", [0.3, 1, 2.5, 7.2, 13, 36, 36.4, 80])
def test_blur_matches_pil_bit_exact(device: str, sigma: float) -> None:
    rng = np.random.default_rng(int(sigma * 10))
    for h, w in [(1, 1), (3, 200), (61, 47), (120, 95)]:  # includes boxes wider than the image
        img = rng.integers(0, 256, (h, w), dtype=np.uint8)
        want = np.asarray(Image.fromarray(img, "L").filter(ImageFilter.GaussianBlur(sigma)))
        got = gaussian_blur_u8(torch.from_numpy(img).to(device), sigma).cpu().numpy()
        assert np.array_equal(got, want), (sigma, h, w, int(np.abs(got.astype(int) - want).max()))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("pad", [0, 8, 20, 29, 60, 90, 120])
def test_print_texture_matches_pil_bit_exact(device: str, pad: int) -> None:
    rng = np.random.default_rng(pad)
    for ch, cw in [(40, 30), (97, 151), (300, 220)]:
        photo = rng.random((ch, cw, 3), dtype=np.float32)
        off_want, want = _pil_print(photo, pad)
        off, tex = print_texture(photo, pad, device)
        got = tex[0].permute(1, 2, 0).cpu().numpy()
        assert off == off_want
        assert got.dtype == np.float32 and got.shape == want.shape
        assert np.array_equal(got, want), (pad, ch, cw, float(np.abs(got - want).max()))
