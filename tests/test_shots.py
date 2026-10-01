"""Shot framing: formulas on CPU, Chromium oracle for the pixels."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from PIL import Image

from montaj.doctor import chromium
from montaj.render.canvas import Canvas
from montaj.render.core import Photo, cubic
from montaj.render.shots import (
    card_frame,
    card_layout,
    drift,
    focus_frac,
    _paste_straight,
    frame_background,
    frame_card,
    kenburns,
    morph_mask,
    pulse_heartbeat,
    sine,
    tone,
)
from tests.oracle_html import expect_psnr, render_html, render_page, stage_html

PHOTO = Path("/home/raja/Anuraj-dev/claude-test/bday-video/assets/photos/shy2.jpg")
# Repo root from this file, so the face still resolves after the worktree is removed.
FONT = Path(__file__).resolve().parents[1] / "montaj/fonts/great-vibes-latin-400-normal.woff2"
CAP = "Arshiya"

pytestmark_gpu = pytest.mark.skipif(chromium() is None, reason="chromium missing")


@pytest.fixture(scope="session", autouse=True)
def _psnr_log() -> None:
    Path("/tmp/w5c-psnr.txt").write_text("")


@pytest.fixture(scope="module")
def cv():
    return Canvas(1080, 1920, "cpu")


@pytest.fixture(scope="module")
def photo(cv):
    return Photo(cv, str(PHOTO))


def test_sine_and_drift_endpoints() -> None:
    assert sine(0) == 0 and sine(1) == 1
    assert sine(0.5) == pytest.approx(0.5)
    z, x, y = drift(0, 10, 20, (1.1, 1.3), (40, -20))
    assert (z, x, y) == (1.1, -20, 10)
    z, x, y = drift(20, 10, 20, (1.1, 1.3), (40, -20))
    assert (z, x, y) == (1.3, 20, -10)
    z, x, y = drift(15, 10, 20, (1.1, 1.3), (40, -20))
    assert z == pytest.approx(1.2) and x == 0 and y == 0
    # Parked outside the visible span.
    assert drift(-5, 10, 20, (1.1, 1.3), (40, -20))[0] == 1.1
    assert drift(50, 10, 20, (1.1, 1.3), (40, -20))[0] == 1.3


def test_focus_fraction_defaults_to_crop_centre(tmp_path: Path) -> None:
    path = tmp_path / "p.png"
    Image.new("RGB", (40, 20), (10, 20, 30)).save(path)
    photo = Photo(Canvas(64, 64, "cpu"), str(path), crop=(4, 2, 24, 18))
    assert focus_frac(photo, None) == (0.5, 0.5)
    assert focus_frac(photo, (4, 18)) == (0.0, 1.0)
    assert focus_frac(photo, (14, 10)) == (0.5, 0.5)


def test_pulse_heartbeat_two_lobes() -> None:
    assert pulse_heartbeat(0) == pytest.approx(1.035)
    # The second lobe starts strictly after 0.22 (film.html line 295).
    assert pulse_heartbeat(0.22) == pytest.approx(1 + 0.035 * math.exp(-18 * 0.22))
    b = 0.5
    assert pulse_heartbeat(b) == pytest.approx(
        1 + 0.035 * math.exp(-18 * b) + 0.02 * math.exp(-18 * (b - 0.22))
    )


def test_tone_matches_css_swatches() -> None:
    def px(rgb):
        img = torch.tensor(rgb, dtype=torch.float32).view(3, 1, 1) / 255
        return img

    grey = tone(px((64, 64, 64)), brightness=0.45, saturate=0.9)
    assert torch.allclose(grey, torch.full_like(grey, 64 / 255 * 0.45))
    orange = tone(px((255, 128, 0)), saturate=0.9)[:, 0, 0]
    assert torch.allclose(orange, torch.tensor([244 / 255, 130 / 255, 15 / 255]), atol=0.5 / 255)
    ink = tone(px((51, 85, 119)), brightness=0.6, contrast=1.05)[:, 0, 0]
    assert torch.allclose(ink, torch.tensor([26 / 255, 47 / 255, 69 / 255]), atol=0.5 / 255)


def test_tone_clamps_each_filter() -> None:
    """CSS clips every primitive. Clamping only the finished image leaves white at 1, not .75."""
    white = torch.ones(3, 1, 1)
    assert torch.allclose(tone(white, brightness=2, contrast=0.5), torch.full_like(white, 0.75))
    # saturate(2) on red is 1.787 before the clip; contrast .5 then sees (1, 0, 0).
    red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    assert torch.allclose(
        tone(red, saturate=2, contrast=0.5), torch.tensor([0.75, 0.25, 0.25]).view(3, 1, 1)
    )
    assert torch.allclose(tone(white, contrast=2), torch.ones_like(white))


def test_paste_filters_premultiplied_rgb() -> None:
    """Opaque red beside transparency is premul .5 at alpha .5, not straight-filtered .25."""
    sprite = torch.zeros(4, 1, 2)
    sprite[:, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 1.0])
    base = torch.zeros(4, 1, 2)
    _paste_straight(base, sprite, -0.5, 0.0)
    assert torch.allclose(base[:, 0, 0], torch.tensor([0.5, 0.0, 0.0, 0.5]))
    assert torch.allclose(base[:, 0, 1], torch.zeros(4))


def test_kenburns_clamps_focus_inside_the_image(tmp_path: Path) -> None:
    im = Image.new("RGB", (40, 20))
    pix = im.load()
    for x in range(40):
        for y in range(20):
            pix[x, y] = (int(x * 255 / 39), int(y * 255 / 19), 80)
    path = tmp_path / "p.png"
    im.save(path)
    cv = Canvas(1080, 1920, "cpu")
    out = kenburns(cv, Photo(cv, str(path)), 0.0, 0.5, 1.0, 0.0, 0.0)
    assert out.shape == (3, 1920, 1080)
    # s = 1920/20, window clamped to the left edge, so screen x=0 is source x≈0.
    assert float(out[0, 960, 0]) < 0.05
    assert 0.2 < float(out[0, 960, 1079]) < 0.4


def test_card_layout_shy2(photo, cv) -> None:
    L = card_layout(cv, photo, CAP)
    assert L["iw"] == pytest.approx(900)
    assert L["ih"] == pytest.approx(675)
    assert L["cw"] == pytest.approx(944)
    assert L["ch"] == pytest.approx(817)
    assert L["left"] == pytest.approx(68)
    assert L["top"] == pytest.approx(511.5)


def test_morph_endpoints_and_feather(cv) -> None:
    assert float(morph_mask(cv, 0, (0.5, 0.5)).max()) == 0
    # film.html line 156: r >= 1900 drops the mask.
    assert float(morph_mask(cv, 1, (0.55, 0.5)).min()) == 1
    m = morph_mask(cv, 0.3, (0.55, 0.5))
    r = cubic(0.3) * 1900
    assert r == pytest.approx(205.2)
    cx, cy = int(0.55 * 1080), int(0.50 * 1920)
    assert float(m[0, cy, cx]) > 0.99
    assert float(m[0, 0, 0]) == 0


def _pattern() -> torch.Tensor:
    bands = (
        (255, 128, 0), (64, 64, 64), (51, 85, 119), (255, 0, 0), (0, 255, 0),
        (0, 0, 255), (128, 64, 200), (20, 20, 20), (240, 230, 210), (10, 80, 40),
    )
    img = torch.zeros(3, 1920, 1080)
    for i, rgb in enumerate(bands):
        img[:, i * 192:(i + 1) * 192] = torch.tensor(rgb).view(3, 1, 1) / 255
    return img


def _draw_pattern_js() -> str:
    bands = [
        [255, 128, 0], [64, 64, 64], [51, 85, 119], [255, 0, 0], [0, 255, 0],
        [0, 0, 255], [128, 64, 200], [20, 20, 20], [240, 230, 210], [10, 80, 40],
    ]
    return f"""
      window.__oracleReady = false;
      const g = document.getElementById('c').getContext('2d');
      const bands = {json.dumps(bands)};
      bands.forEach((rgb, i) => {{
        g.fillStyle = `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
        g.fillRect(0, i * 192, 1080, 192);
      }});
      window.__oracleReady = true;
    """


@pytest.mark.gpu
@pytestmark_gpu
@pytest.mark.parametrize("filt,kwargs", [
    ("brightness(.45) saturate(.9)", {"brightness": 0.45, "saturate": 0.9}),
    ("brightness(.6) contrast(1.05)", {"brightness": 0.6, "contrast": 1.05}),
])
def test_tone_matches_chromium(filt, kwargs) -> None:
    html = stage_html(
        body=f'<canvas id="c" class="L" width="1080" height="1920" style="filter: {filt}"></canvas>',
        script=_draw_pattern_js(),
    )
    chrome = render_html(html)
    expect_psnr(f"tone {filt}", chrome, tone(_pattern(), **kwargs).numpy(), 38)


@pytest.mark.gpu
@pytestmark_gpu
@pytest.mark.parametrize("fx,fy,z,px,py", [
    (0.0, 0.5, 1.0, 0.0, 0.0),       # fx clamps the window to the left edge
    (0.5, 0.3, 1.2, 12.0, -18.0),    # fy clamps, then the pan still covers
])
def test_kenburns_matches_chromium(cv, photo, fx, fy, z, px, py) -> None:
    html = stage_html(
        body=f'<img id="im" src="{PHOTO.as_uri()}">',
        script=f"""
          window.__oracleReady = false;
          const img = document.getElementById('im');
          const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
          img.decode().then(() => {{
            const nw = img.naturalWidth, nh = img.naturalHeight;
            const s = Math.max(1080 / nw, 1920 / nh) * {z};
            const w = nw * s, h = nh * s;
            const ox = clamp({fx} * w - 540, 0, w - 1080);
            const oy = clamp({fy} * h - 960, 0, h - 1920);
            img.style.left = '0px'; img.style.top = '0px';
            img.style.width = w + 'px'; img.style.height = h + 'px';
            img.style.transform = `translate(${{-ox + {px}}}px, ${{-oy + {py}}}px)`;
            window.__oracleReady = true;
          }});
        """,
    )
    chrome = render_html(html)
    pred = kenburns(cv, photo, fx, fy, z, px, py)
    expect_psnr(f"kenburns z={z} fx={fx}", chrome, pred.numpy(), 38)


def _caption_raster(cw: float) -> torch.Tensor:
    html = stage_html(
        css=f"""
          @font-face {{ font-family: 'Vibes'; src: url('{FONT.as_uri()}') format('woff2'); }}
          .cap {{ position: absolute; left: 0; top: 0; width: {cw}px; text-align: center;
                  font-family: 'Vibes'; font-size: 64px; color: #5a3b24; }}
        """,
        body=f'<div id="cap" class="cap">{CAP}</div>',
        script=f"""
          window.__oracleReady = false;
          document.fonts.load('64px Vibes', {json.dumps(CAP)}).then(() => {{
            const r = document.getElementById('cap').getBoundingClientRect();
            window.__oracleResult = {{x: r.x, y: r.y, width: r.width, height: r.height}};
            window.__oracleReady = true;
          }});
        """,
        # No painted backdrop: omitBackground then keeps the glyph coverage only.
        bg="transparent",
    )
    shot = render_page(html)
    assert shot.rgba is not None and shot.rgba.shape[-1] == 4
    return torch.from_numpy(shot.rgba.astype("float32") / 255).permute(2, 0, 1).contiguous()


@pytest.mark.gpu
@pytestmark_gpu
@pytest.mark.parametrize("z,tilt,px,py", [(1.0, 2.0, 0.0, 0.0), (1.04, -1.5, 8.0, -6.0)])
def test_frame_card_matches_chromium(cv, photo, z, tilt, px, py) -> None:
    L = card_layout(cv, photo, CAP)
    rasters = _caption_raster(L["cw"])
    card = frame_card(cv, photo, CAP, rasters)
    bg = frame_background(cv, photo)
    pred = card_frame(cv, card, bg, z, tilt, px, py)
    html = stage_html(
        css=f"""
          @font-face {{ font-family: 'Vibes'; src: url('{FONT.as_uri()}') format('woff2'); }}
          .frame {{ position: absolute; background: #f6efe2;
            box-shadow: 0 40px 120px rgba(0,0,0,.7), 0 0 0 1px rgba(0,0,0,.2); }}
          .frame img {{ position: absolute; }}
          .frame .cap {{ position: absolute; left: 0; right: 0; bottom: 18px; text-align: center;
            font-family: 'Vibes'; font-size: 64px; color: #5a3b24; }}
        """,
        body=f"""
          <div class="L">
            <img id="bg" src="{PHOTO.as_uri()}">
            <div class="frame" id="card"><img id="fg" src="{PHOTO.as_uri()}"><div class="cap">{CAP}</div></div>
          </div>
        """,
        script=f"""
          window.__oracleReady = false;
          const bg = document.getElementById('bg'), fg = document.getElementById('fg');
          const card = document.getElementById('card');
          const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
          Promise.all([bg.decode(), fg.decode(), document.fonts.load('64px Vibes', {json.dumps(CAP)})]).then(() => {{
            const nw = fg.naturalWidth, nh = fg.naturalHeight;
            bg.style.filter = 'blur(38px) brightness(.42) saturate(1.2)';
            const s = Math.max(1080 / nw, 1920 / nh) * 1.25;
            const w = nw * s, h = nh * s;
            const ox = clamp(.5 * w - 540, 0, w - 1080), oy = clamp(.5 * h - 960, 0, h - 1920);
            Object.assign(bg.style, {{ left: '0px', top: '0px', width: w + 'px', height: h + 'px',
              transform: `translate(${{-ox}}px, ${{-oy}}px)` }});
            const pad = 22, capH = 120;
            const cs = Math.min(900 / nw, 1260 / nh);
            const iw = nw * cs, ih = nh * cs, cw = iw + pad * 2, ch = ih + pad + capH;
            Object.assign(card.style, {{
              width: cw + 'px', height: ch + 'px',
              left: (1080 - cw) / 2 + 'px', top: (1920 - ch) / 2 - 40 + 'px',
              transform: `translate({px}px, {py}px) scale({z * 0.97}) rotate({tilt}deg)`
            }});
            Object.assign(fg.style, {{ left: pad + 'px', top: pad + 'px', width: iw + 'px', height: ih + 'px' }});
            window.__oracleReady = true;
          }});
        """,
    )
    chrome = render_html(html)
    # Background blur dominates the frame; the card interior is sharper.
    expect_psnr(f"frame tilt={tilt}", chrome, pred.numpy(), 32)


@pytest.mark.gpu
@pytestmark_gpu
@pytest.mark.parametrize("x", [0.3, 0.7])
def test_morph_mask_matches_chromium(cv, x) -> None:
    center = (0.55, 0.50)
    html = stage_html(
        body='<div id="m" class="L" style="background:#fff"></div>',
        script=f"""
          const eio = k => k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
          const r = eio({x}) * 1900;
          const g = `radial-gradient(circle at 55% 50%, #000 ${{Math.max(0, r - 380)}}px, transparent ${{r}}px)`;
          const el = document.getElementById('m');
          el.style.webkitMaskImage = el.style.maskImage = g;
          window.__oracleReady = true;
        """,
    )
    chrome = render_html(html)
    pred = morph_mask(cv, x, center).expand(3, -1, -1)
    expect_psnr(f"morph x={x}", chrome, pred.numpy(), 38)
