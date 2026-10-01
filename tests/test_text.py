"""Chromium raster + GPU text compositor. Fidelity is a Chrome screenshot of film.html's text CSS."""
from __future__ import annotations

import math

import pytest
import torch

from montaj.render.canvas import Canvas
from montaj.render.core import outc
from montaj.render import text as text_mod
from montaj.render.text import (
    OX, OY, RasterLine, RasterSet, Sub, TextBlock, TextLine, Unit, draw_text, line_markup,
    raster_caption, raster_line, raster_lines, raster_subs, sine,
)

GREY = 128 / 255


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def test_sine_clamps_like_film():
    # film.html:73 es clamps; film.html:71 eo is outc.
    assert sine(0.5) == pytest.approx(0.5)
    assert sine(-1) == 0.0
    assert sine(2) == 1.0
    assert outc(0.5) == pytest.approx(1 - 0.5 ** 3)


def test_markup_matches_film_classes():
    serif = TextLine("held tight", "serif", 70, 0, italic=True)
    script = TextLine("Joy", "script", 340, 0, by="char", size=110)
    gold = TextLine("4 yrs", "serif", 640, 0, gold=True, size=150, weight=300, line_height=1)
    assert 'class="txt serif it"' in line_markup(serif)
    assert ">held </span>" in line_markup(serif) and ">tight</span>" in line_markup(serif)
    markup = line_markup(script)
    assert 'class="txt script"' in markup
    assert ">J</span>" in markup and ">o</span>" in markup and ">y</span>" in markup
    g = line_markup(gold, bp="50% 0")
    assert "gold" in g and "--bp:50% 0" in g and ">4 </span>" in g


def test_draw_is_pure():
    class CV:
        W, H, k = 32, 32, 32 / 1080

    img = torch.zeros(3, 32, 32)
    out = draw_text(CV(), img, [], [], 0.0, [])
    assert out is not img and torch.equal(out, img)
    img[0, 0, 0] = 1
    assert out[0, 0, 0] == 0


def test_cache_hit_does_not_start_chromium(tmp_path, monkeypatch):
    line = TextLine("Hello", "serif", 120, 0, dur=1, stagger=0)
    first = raster_lines([line], scale=1, cache_dir=tmp_path)

    def boom(*_a, **_k):
        raise AssertionError("chromium started")

    monkeypatch.setattr(text_mod, "_Browser", boom)
    second = raster_lines([line], scale=1, cache_dir=tmp_path)
    one = raster_line(line, scale=1, cache_dir=tmp_path)
    assert torch.equal(first[0].units[0].fill, second[0].units[0].fill)
    assert torch.equal(one.units[0].fill, first[0].units[0].fill)
    assert isinstance(one, RasterLine)
    assert first[0].units[0].fill[3].max() > 0.2


def test_half_scale_is_half_size(tmp_path):
    line = TextLine("Hello", "serif", 120, 0)
    full = raster_lines([line], scale=1, cache_dir=tmp_path / "a")[0].units[0]
    half = raster_lines([line], scale=0.5, cache_dir=tmp_path / "b")[0].units[0]
    assert abs(half.fill.shape[-2] * 2 - full.fill.shape[-2]) <= 2
    assert abs(half.fill.shape[-1] * 2 - full.fill.shape[-1]) <= 2
    assert abs(half.layout[2] - full.layout[2]) <= 1.0
    assert abs(half.layout[3] - full.layout[3]) <= 1.0


def test_caption_is_brown_script(tmp_path):
    sprite, (w, h) = raster_caption("Arshiya", font_css="", scale=1, cache_dir=tmp_path)
    assert sprite.shape[0] == 4 and sprite[3].max() > 0.4
    assert w > 10 and h > 10
    flat = sprite[3].reshape(-1)
    idx = int(flat.argmax())
    alpha = flat[idx]
    rgb = sprite[:3].reshape(3, -1)[:, idx] / alpha
    expect = torch.tensor([0x5A, 0x3B, 0x24], dtype=torch.float32) / 255
    assert (rgb - expect).abs().max() < 12 / 255


def _save(path, img):
    from PIL import Image
    import numpy as np
    arr = (img.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def _best_shift(ref, out, rad=4):
    best = None
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            a = ref[:, max(0, dy):ref.shape[1] + min(0, dy), max(0, dx):ref.shape[2] + min(0, dx)]
            b = out[:, max(0, -dy):out.shape[1] + min(0, -dy), max(0, -dx):out.shape[2] + min(0, -dx)]
            h, w = min(a.shape[1], b.shape[1]), min(a.shape[2], b.shape[2])
            err = (a[:, :h, :w] - b[:, :h, :w]).abs().mean().item()
            if best is None or err < best[0]:
                best = (err, dx, dy)
    return best


# film.html lines 7–35, copied verbatim. The harness below only moves the stage to the
# raster origin and replaces the black plate with the comparison grey.
_FILM_TEXT_CSS = """
  @font-face { font-family: 'Corm'; src: url(fonts/cormorant-garamond-latin-300-normal.woff2) format('woff2'); font-weight: 300; }
  @font-face { font-family: 'Corm'; src: url(fonts/cormorant-garamond-latin-600-normal.woff2) format('woff2'); font-weight: 600; }
  @font-face { font-family: 'Corm'; src: url(fonts/cormorant-garamond-latin-300-italic.woff2) format('woff2'); font-weight: 300; font-style: italic; }
  @font-face { font-family: 'Corm'; src: url(fonts/cormorant-garamond-latin-500-italic.woff2) format('woff2'); font-weight: 500; font-style: italic; }
  @font-face { font-family: 'Vibes'; src: url(fonts/great-vibes-latin-400-normal.woff2) format('woff2'); }
  @font-face { font-family: 'Tiro'; src: url(fonts/tiro-devanagari-hindi-devanagari-400-normal.woff2) format('woff2'); unicode-range: U+0900-097F, U+200C-200D, U+25CC, U+A8E0-A8FF; }
  @font-face { font-family: 'Tiro'; src: url(fonts/tiro-devanagari-hindi-latin-400-normal.woff2) format('woff2'); unicode-range: U+0000-00FF; }
  :root { --cream: #fbf1dc; --gold: #f0c77e; --gold2: #d99a45; --ink: #07060a; }
  html, body { margin: 0; background: #000; overflow: hidden; }
  #stage { position: absolute; left: 0; top: 0; width: 1080px; height: 1920px; overflow: hidden; background: var(--ink);
    transform-origin: 0 0; color: var(--cream); font-family: 'Corm', serif; }
  .L { position: absolute; left: 0; top: 0; width: 1080px; height: 1920px; overflow: hidden; }
  .scene { display: none; will-change: opacity, filter; }
  img { position: absolute; display: block; user-select: none; }
  .frame { position: absolute; background: #f6efe2; box-shadow: 0 40px 120px rgba(0,0,0,.7), 0 0 0 1px rgba(0,0,0,.2); }
  .frame img { position: absolute; }
  .frame .cap { position: absolute; left: 0; right: 0; bottom: 18px; text-align: center; font-family: 'Vibes'; font-size: 64px; color: #5a3b24; }
  .txt { position: absolute; left: 0; width: 1080px; text-align: center; white-space: nowrap; }
  .w { display: inline-block; white-space: pre; }
  .serif { font-weight: 300; font-size: 84px; line-height: 1.1; letter-spacing: .01em; text-shadow: 0 4px 30px rgba(0,0,0,.65); }
  .it { font-style: italic; font-weight: 300; }
  .script { font-family: 'Vibes'; font-weight: 400; color: var(--gold); text-shadow: 0 0 40px rgba(240,199,126,.45), 0 4px 24px rgba(0,0,0,.6); }
  .caps { font-weight: 600; letter-spacing: .42em; text-transform: uppercase; font-size: 26px; color: var(--gold); text-shadow: 0 2px 16px rgba(0,0,0,.8); }
  .deva { font-family: 'Tiro'; font-size: 62px; line-height: 1.5; text-shadow: 0 4px 24px rgba(0,0,0,.8); }
  .goldtext, .gold .w { background: linear-gradient(100deg, #b87a35 0%, #f7dca0 30%, #fff6de 45%, #e9b566 60%, #a86d2c 100%);
    background-size: 300% 100%; background-position: var(--bp, 50% 0); -webkit-background-clip: text; background-clip: text; color: transparent; text-shadow: none; }
  .gold { filter: drop-shadow(0 6px 30px rgba(0,0,0,.55)); }
  #sub { position: absolute; left: 60px; right: 60px; bottom: 190px; text-align: center; font-style: italic; font-weight: 500;
    font-size: 44px; letter-spacing: .02em; color: rgba(251,241,220,.92); text-shadow: 0 2px 18px rgba(0,0,0,.95), 0 0 4px rgba(0,0,0,.8); white-space: nowrap; }
"""


def _film_oracle_page() -> str:
    """Words() markup for the three fidelity lines. Classes and inline overrides are film's, not line_markup."""
    # film.html:170-174. Words() appends a trailing space to every word except the last; chars get none.
    # Sweep midpoint is --bp: 50% 0 (film.html:32 default, and the linear 100→0 lerp at mid-window).
    body = (
        '<div class="txt serif it" style="top:70px">'
        '<span class="w">held </span><span class="w">tight</span></div>'
        '<div class="txt script" style="top:340px;font-size:110px">'
        '<span class="w">J</span><span class="w">o</span><span class="w">y</span></div>'
        '<div class="txt serif gold" style="top:640px;font-size:150px;font-weight:300;line-height:1;--bp:50% 0">'
        '<span class="w">4 </span><span class="w">yrs</span></div>'
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        + _FILM_TEXT_CSS
        + "html, body { background: rgb(128,128,128); }"
        + "#stage { background: transparent; overflow: visible; }"
        + "</style></head><body>"
        + f'<div style="position:absolute;left:{OX}px;top:{OY}px"><div id="stage">{body}</div></div>'
        + "</body></html>"
    )


@pytest.mark.gpu
def test_fidelity_three_lines(tmp_path):
    _cuda()
    # film.html:28 sets no font-size. A copy of text.css's `.script { font-size: 84px }` would fail this.
    script_rule = _FILM_TEXT_CSS.split(".script {")[1].split("}")[0]
    assert "font-size" not in script_rule
    assert "font-size: 84px" in _FILM_TEXT_CSS
    assert "letter-spacing: .01em" in _FILM_TEXT_CSS
    lines = [
        TextLine("held tight", "serif", 70, 0, italic=True, dur=1, stagger=0),
        TextLine("Joy", "script", 340, 0, by="char", size=110, dur=1, stagger=0),
        TextLine("4 yrs", "serif", 640, 0, gold=True, size=150, weight=300, line_height=1,
                 dur=1, stagger=0, sweep=(0, 200)),
    ]
    block = TextBlock(0, 200, 0, 0, 10, lines)
    rasters = raster_lines(lines, scale=1, cache_dir=tmp_path / "c")
    with text_mod._Browser(1, clear=(128, 128, 128, 255)) as br:
        br.show(_film_oracle_page())
        png = br.capture(OX, OY, 1080, 1000, omit_background=False)
    ref = text_mod._png_rgb(png).cpu().float()
    assert ref.shape[-2:] == (1000, 1080), ref.shape
    # Same grey the screenshot actually painted, which is the CSS mid-grey.
    bg = ref[:, 0, 0]
    assert (bg - GREY).abs().max() < 2 / 255, bg.tolist()
    cv = Canvas(1080, 1000, device="cuda")
    img = bg.to("cuda").view(3, 1, 1).expand(3, 1000, 1080).contiguous()
    before = img.clone()
    out = draw_text(cv, img, [block], [], 100.0, rasters, fps=30).cpu().float()
    assert torch.equal(img, before)
    diff = (ref - out).abs()
    mask = (ref - bg.view(3, 1, 1)).abs().amax(0).gt(2 / 255) | (out - bg.view(3, 1, 1)).abs().amax(0).gt(2 / 255)
    assert int(mask.sum()) > 100, "no text pixels"
    mean = diff[:, mask].mean().item()
    mse = (ref - out)[:, mask].pow(2).mean().item()
    psnr = 99.0 if mse <= 0 else 10 * math.log10(1.0 / mse)
    if mean >= 2 / 255 or psnr <= 35:
        _save("/tmp/fid-ref.png", ref)
        _save("/tmp/fid-ours.png", out)
        _save("/tmp/fid-diff.png", (diff / diff.max().clamp(min=1e-6)).clamp(0, 1))
        shift = _best_shift(ref, out)
        raise AssertionError(
            f"mean |Δ|={mean:.6f} ({mean * 255:.2f}/255) PSNR={psnr:.2f} dB "
            f"best shift mean={shift[0]:.6f} dx={shift[1]} dy={shift[2]}"
        )


@pytest.mark.gpu
def test_reveal_opacity_matches_outc(tmp_path):
    _cuda()
    line = TextLine("Hello", "serif", 160, at=10, dur=20, stagger=0)
    block = TextBlock(0, 80, 0, 0, 10, (line,))
    rasters = raster_lines([line], scale=1, cache_dir=tmp_path)
    cv = Canvas(1080, 480, device="cuda")
    black = torch.zeros(3, 480, 1080, device="cuda")
    mid = draw_text(cv, black, [block], [], 20.0, rasters, fps=30)
    full = draw_text(cv, black, [block], [], 30.0, rasters, fps=30)
    assert full.abs().sum() > 1
    ratio = mid.sum().item() / full.sum().item()
    assert ratio == pytest.approx(outc(0.5), abs=0.02), ratio


@pytest.mark.gpu
def test_sweep_is_continuous(tmp_path):
    _cuda()
    line = TextLine("4 yrs", "serif", 70, at=-10, gold=True, size=140, weight=300, line_height=1,
                    dur=1, stagger=0, sweep=(0, 4000))
    block = TextBlock(0, 4000, 0, 0, 10, (line,))
    rasters = raster_lines([line], scale=1, cache_dir=tmp_path)
    cv = Canvas(1080, 420, device="cuda")
    black = torch.zeros(3, 420, 1080, device="cuda")
    frames = [draw_text(cv, black, [block], [], float(t), rasters, fps=30) for t in range(2000, 2031)]
    limit = 1 / 255 + 1e-5
    for a, b in zip(frames, frames[1:]):
        jump = (a - b).abs().max().item()
        assert jump <= limit, jump
    # Whole-frame mean stays under 1/255 because the glyph is a small part of the frame.
    # The sweep still has to move: the pixels that change must move by more than one level.
    ends = draw_text(cv, black, [block], [], 0.0, rasters, fps=30)
    other = draw_text(cv, black, [block], [], 4000.0, rasters, fps=30)
    delta = (ends - other).abs()
    moved = delta.amax(0) > 1 / 255
    assert int(moved.sum()) > 50, int(moved.sum())
    assert delta[:, moved].mean().item() > 4 / 255


@pytest.mark.gpu
def test_subs_fade_and_first_wins(tmp_path):
    _cuda()
    subs = [Sub(100, 160, "alpha line"), Sub(150, 220, "beta line")]
    sprites = raster_subs(subs, scale=1, cache_dir=tmp_path)
    cv = Canvas(1080, 640, device="cuda")
    black = torch.zeros(3, 640, 1080, device="cuda")
    pack = RasterSet((), sprites)

    def at(t, which=None):
        if which is None:
            return draw_text(cv, black, [], subs, t, pack, fps=30)
        i = which
        return draw_text(cv, black, [], [subs[i]], t, RasterSet((), (sprites[i],)), fps=30)

    # Window of the first sub opens at 100 - 0.2s. Opacity there is 0, and it still wins.
    assert at(93).abs().max().item() < 1e-4
    assert at(94).abs().max().item() < 1e-4
    early = at(100)
    full = at(150)
    assert full.abs().sum() > 1
    # film.html:431 — sine((t-(from-.2))/.5) at t = from is sine(0.2/0.5).
    expect = sine(0.4)
    assert early.sum().item() / full.sum().item() == pytest.approx(expect, abs=0.05)
    both = at(150)
    assert torch.allclose(both, at(150, 0), atol=1e-5)
    assert not torch.allclose(both, at(150, 1), atol=1e-3)
    # Past the first window the second sub is the one on screen.
    assert torch.allclose(at(180), at(180, 1), atol=1e-5)


class _CV:
    W, H, k = 80, 80, 1.0


def _cy(img: torch.Tensor) -> float:
    mass = img[0].sum(1)
    total = mass.sum()
    ys = torch.arange(img.shape[1], dtype=img.dtype)
    return float((mass * ys).sum() / total)


def test_reveal_translation_keeps_subpixel_motion():
    # dur=30, rise=26. t=15, 15.25, 15.5 used to round to the same 3px step.
    fill = torch.zeros(4, 8, 8)
    fill[:, 2:6, 2:6] = 1
    unit = Unit("A", fill, None, (20.0, 20.0, 8.0, 8.0), None, (20.0, 20.0, 8.0, 8.0), False)
    raster = RasterLine((unit,), (20.0, 20.0, 8.0, 8.0), None, None, 1.0, False)
    line = TextLine("A", "serif", 20, at=0, dur=30, stagger=0, rise=26, shadow=False)
    block = TextBlock(0, 100, 0, 0, 0, (line,))
    black = torch.zeros(3, 80, 80)

    def dy(t: float) -> float:
        return (1 - outc(t / 30)) * 26

    frames = {t: draw_text(_CV(), black, [block], [], t, [raster], fps=30) for t in (15.0, 15.25, 15.5, 30.0)}
    c15, c25, c50 = _cy(frames[15.0]), _cy(frames[15.25]), _cy(frames[15.5])
    assert c25 - c15 == pytest.approx(dy(15.25) - dy(15.0), abs=0.03)
    assert c50 - c25 == pytest.approx(dy(15.5) - dy(15.25), abs=0.03)
    # Integer rest pose stays a sharp blit: the opaque block occupies rows 22..25.
    assert frames[30.0][0, 21, 23] == pytest.approx(0.0)
    assert frames[30.0][0, 22, 23] == pytest.approx(1.0)
    assert _cy(frames[30.0]) == pytest.approx(23.5)


def test_staggered_gold_shadow_stays_behind_earlier_glyphs():
    # Unit 1's captured shadow covers unit 0. A per-word shadow-then-fill would blacken it.
    fill = torch.ones(4, 6, 6)
    black_shadow = torch.zeros(4, 6, 6)
    black_shadow[3] = 1
    first = Unit("A", fill, None, (10.0, 10.0, 6.0, 6.0), None, (10.0, 10.0, 6.0, 6.0), True)
    second = Unit(
        "B", fill.clone(), black_shadow, (40.0, 10.0, 6.0, 6.0), (10.0, 10.0, 6.0, 6.0),
        (40.0, 10.0, 6.0, 6.0), True,
    )
    line = TextLine("A B", "serif", 10, at=0, gold=True, rise=0, dur=10, stagger=5)
    block = TextBlock(0, 100, 0, 0, 0, (line,))
    white = torch.ones(3, 80, 80)
    duo = RasterLine((first, second), (10.0, 10.0, 36.0, 6.0), None, None, 1.0, True)
    solo = RasterLine((first,), (10.0, 10.0, 6.0, 6.0), None, None, 1.0, True)
    # t=10 → k = 1 and outc(0.5). Non-uniform, so the parent shadow is of the finished layer.
    painted = draw_text(_CV(), white, [block], [], 10.0, [duo], fps=30)
    alone = draw_text(_CV(), white, [block], [], 10.0, [solo], fps=30)
    assert torch.allclose(painted[:, 13, 13], alone[:, 13, 13], atol=1e-5)
    assert float(painted[:, 13, 13].sum()) > 0.5
    # Parent drop-shadow of the finished layer darkens the plate; it does not repaint the glyph.
    assert float(painted[0, 18, 13]) < float(painted[0, 70, 70]) - 0.005


def test_block_blur_keeps_requested_sigma():
    # blur=200 at mid fade, k=1, requests σ=100. A radius-128 kernel measures near 66.
    n = 801
    img = torch.zeros(1, n, n)
    img[0, n // 2, n // 2] = 1
    out = text_mod._blur(img, 100.0)
    p = out[0]
    p = p / p.sum()
    axis = torch.arange(n, dtype=p.dtype)
    cy = (p.sum(1) * axis).sum()
    cx = (p.sum(0) * axis).sum()
    var_y = float((p.sum(1) * (axis - cy) ** 2).sum())
    var_x = float((p.sum(0) * (axis - cx) ** 2).sum())
    assert var_y == pytest.approx(100.0 ** 2, rel=0.05)
    assert var_x == pytest.approx(100.0 ** 2, rel=0.05)


def test_required_font_failure_is_not_cached(tmp_path, monkeypatch):
    line = TextLine("Hello", "serif", 120, 0)
    orig = text_mod._css_text

    def broken() -> str:
        return orig().replace("font-family: 'Corm', serif", "font-family: 'NoSuchMontajFont', serif", 1)

    monkeypatch.setattr(text_mod, "_css_text", broken)
    with pytest.raises(RuntimeError, match="required fonts failed to load"):
        raster_lines([line], scale=1, cache_dir=tmp_path)
    assert list(tmp_path.rglob("meta.json")) == []
