"""Finish look: sharpen kernel scales with Canvas.k; intro lift dies inside dur."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from montaj.render.canvas import Canvas
from montaj.render.core import smooth
from montaj.render.finish import LOOKS, _binomial, _odd, finish

# Isolate the term under test: identity grade, no bloom.
_SHARP = {**LOOKS["warm-film"], "sharpen": 1.0, "scurve": 0.0, "lift": 0.0, "gain": 1.0, "vignette": 0.0}
_LIFT = {**LOOKS["warm-film"], "intro_bloom": 0.0, "sharpen": 0.0, "scurve": 0.0, "lift": 0.0, "gain": 1.0, "vignette": 0.0}


def _unsharp(img: torch.Tensor, row: list[int], padding: int) -> torch.Tensor:
    v = torch.tensor(row, dtype=torch.float32)
    gk = (v[:, None] * v[None]) / float(v.sum() ** 2)
    bl = F.conv2d(img[:, None], gk[None, None], padding=padding)[:, 0]
    return (img + (img - bl)).clamp(0, 1)


def test_sharpen_k1_is_five_tap() -> None:
    n = max(3, _odd(5 * 1.0))
    assert n == 5
    v = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
    assert torch.equal(_binomial(n, "cpu"), (v[:, None] * v[None]) / 256)
    cv = Canvas(1080, 48, device="cpu")
    assert cv.k == 1.0
    img = torch.rand(3, 48, 1080)
    got = finish(cv, img, 100, _SHARP, [], [], None, None)
    assert torch.allclose(got, _unsharp(img, [1, 4, 6, 4, 1], 2), atol=1e-6, rtol=0)


def test_sharpen_k05_is_three_tap() -> None:
    n = max(3, _odd(5 * 0.5))
    assert n == 3
    v = torch.tensor([1, 2, 1], dtype=torch.float32)
    assert torch.equal(_binomial(n, "cpu"), (v[:, None] * v[None]) / 16)
    cv = Canvas(540, 48, device="cpu")
    assert cv.k == 0.5
    img = torch.rand(3, 48, 540)
    got = finish(cv, img, 100, _SHARP, [], [], None, None)
    three = _unsharp(img, [1, 2, 1], 1)
    five = _unsharp(img, [1, 4, 6, 4, 1], 2)
    assert torch.allclose(got, three, atol=1e-6, rtol=0)
    assert not torch.allclose(got, five, atol=1e-5, rtol=0)


def test_intro_lift_recipe_frames_match_24() -> None:
    """intro_lift_frac 0.8 × 30f = 24, the film2 width, so recipe frames 0–30 are unchanged."""
    cv = Canvas(64, 64, device="cpu")
    img = torch.zeros(3, 64, 64)
    intro = ("white-lift", 30)
    for f in range(31):
        got = finish(cv, img, f, _LIFT, [], [], intro, None)
        want = 0.35 * (1 - smooth(f / 24.0))
        assert torch.allclose(got, torch.full_like(img, want), atol=1e-6), f"f={f}"


def test_intro_lift_short_intro_reaches_zero_by_frame_8() -> None:
    cv = Canvas(64, 64, device="cpu")
    img = torch.rand(3, 64, 64)
    intro = ("white-lift", 10)
    open_ = finish(cv, img, 0, _LIFT, [], [], intro, None)
    mid = finish(cv, img, 7, _LIFT, [], [], intro, None)
    end = finish(cv, img, 8, _LIFT, [], [], intro, None)
    none = finish(cv, img, 8, _LIFT, [], [], None, None)
    assert not torch.allclose(open_, none)
    assert not torch.allclose(mid, none)
    assert torch.equal(end, none)
