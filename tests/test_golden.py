"""Golden-film layers. mulberry32 is checked against node; pixels against Chromium."""
from __future__ import annotations

import json
import math
import os
import subprocess

import numpy as np
import pytest
import torch

from montaj.doctor import chromium
from montaj.render.canvas import Canvas
from montaj.render.golden import (
    bars,
    burst_parts,
    dust,
    fade_out,
    grain,
    hits,
    leaks,
    mote_params,
    mulberry32,
    vignette,
)
from tests.oracle_html import expect_psnr, psnr, render_html, stage_html

needs_chrome = pytest.mark.skipif(chromium() is None, reason="chromium missing")

_RNG = (
    "function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; "
    "let t = Math.imul(seed ^ seed >>> 15, 1 | seed); "
    "t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; "
    "return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }"
)


@pytest.fixture(scope="module")
def cv():
    return Canvas(1080, 1920, "cpu")


@pytest.fixture(scope="module")
def grey():
    return torch.full((3, 1920, 1080), 128 / 255)


def test_mulberry32_matches_node() -> None:
    script = (
        _RNG + "\nfor (const s of [22, 7, 100, 107]) {\n"
        "  const r = rng(s); const xs = [];\n"
        "  for (let i = 0; i < 8; i++) xs.push(r());\n"
        "  console.log(s + ' ' + xs.join(' '));\n}\n"
    )
    env = os.environ.copy()
    env.pop("FORCE_COLOR", None)
    env["NO_COLOR"] = "1"
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    for line in proc.stdout.splitlines():
        if not line.strip() or line[0].isalpha():
            continue
        head, *nums = line.split()
        draw = mulberry32(int(head))
        for got in nums:
            assert abs(draw() - float(got)) < 1e-12


def test_mote_and_burst_consume_rng_in_film_order() -> None:
    draw = mulberry32(22)
    motes = mote_params()
    assert len(motes) == 70
    for x, y, s, v, ph, dr in motes:
        assert x == draw() * 1080 and y == draw() * 1920
        assert s == 1.5 + draw() * draw() * 9
        assert v == 8 + draw() * 30 and ph == draw() * 6.28 and dr == draw() * 40
    parts = burst_parts(3, 4)
    draw = mulberry32(103)
    for vx, vy, s, life, tw in parts:
        a = draw() * 6.283
        sp = 250 + draw() * 900
        assert vx == math.cos(a) * sp and vy == math.sin(a) * sp - 200
        assert s == 1.5 + draw() * 4 and life == 1.4 + draw() * 1.8 and tw == draw() * 6


def test_fade_out_ramps_across_frames() -> None:
    img = torch.ones(3, 2, 2)
    assert torch.equal(fade_out(img, 10, 10, 20), img)
    assert torch.equal(fade_out(img, 30, 10, 20), torch.zeros_like(img))
    assert torch.allclose(fade_out(img, 20, 10, 20), img * 0.5)
    # A zero duration leaves the frame alone.
    assert fade_out(img, 5, 0, 0) is img


def test_hit_opacity_ramps_then_decays() -> None:
    from montaj.render.golden import _hit_opacity
    assert _hit_opacity(10, [(10.04, 0.7)]) == pytest.approx(0.35)
    assert _hit_opacity(10, [(10, 0.7)]) == pytest.approx(0.7)
    assert _hit_opacity(11, [(10, 0.7)]) == pytest.approx(0.7 * math.exp(-3.2))
    # The brightest hit wins, and a huge k still clamps at 1.
    assert _hit_opacity(10, [(10, 0.2), (10, 4.0)]) == 1


def test_bars_cover_partial_rows() -> None:
    cv = Canvas(1080, 32, "cpu")
    img = torch.ones(3, 32, 1080)
    out = bars(cv, img, 2.5)
    assert torch.allclose(out[:, 0], torch.zeros(3, 1080))
    assert torch.allclose(out[:, 2], torch.full((3, 1080), 0.5))
    assert torch.allclose(out[:, 3], torch.ones(3, 1080))
    assert torch.allclose(out[:, 29], torch.full((3, 1080), 0.5))
    assert torch.allclose(out[:, 31], torch.zeros(3, 1080))


def _leak_html(t: float, glow: float = 1.0, dur: float = 165.0) -> str:
    lk = max(0.0, min(1.0, glow)) * (1 - max(0.0, min(1.0, (t - (dur - 2)) / 2)))
    discs = (
        (900, -300 + 180 * math.sin(t * 0.23), 200 + 300 * math.sin(t * 0.17 + 1), 255, 140, 60, 0.55 * lk),
        (1000, 500 + 160 * math.cos(t * 0.19), 1000 + 280 * math.sin(t * 0.21), 255, 70, 110, 0.35 * lk),
    )
    boxes = "".join(
        f'<div class="leak" style="width:{s}px;height:{s}px;left:{left}px;top:{top}px;'
        f'background:radial-gradient(circle, rgba({r},{g},{b},{a}), transparent 65%)"></div>'
        for s, left, top, r, g, b, a in discs
    )
    return stage_html(
        css=".leak { position: absolute; border-radius: 50%; mix-blend-mode: screen; filter: blur(60px); }",
        body=f'<div class="L">{boxes}</div>',
        bg="#808080",
    )


def _dust_html(t: float, level: float, bursts: list) -> str:
    payload = json.dumps([{"i": i, "t": at, "x": x, "y": y, "n": n} for i, at, x, y, n in bursts])
    return stage_html(
        body='<canvas id="dust" class="L" width="1080" height="1920"></canvas>',
        bg="#808080",
        script=f"""
          window.__oracleReady = false;
          {_RNG}
          const W = 1080, H = 1920, t = {t}, level = {level};
          const bursts = {payload};
          const dust = document.getElementById('dust').getContext('2d');
          const MOTES = (() => {{ const r = rng(22); return Array.from({{ length: 70 }}, () => ({{
            x: r() * W, y: r() * H, s: 1.5 + r() * r() * 9, v: 8 + r() * 30, ph: r() * 6.28, dr: r() * 40 }})); }})();
          dust.globalCompositeOperation = 'lighter';
          const k = Math.min(1.2, Math.max(0, level)) * Math.min(1, Math.max(0, t / 2));
          for (const m of MOTES) {{
            const y = ((m.y - m.v * t) % H + H) % H, x = m.x + Math.sin(t * .4 + m.ph) * m.dr;
            const a = k * (.35 + .35 * Math.sin(t * 1.7 + m.ph * 3));
            if (a <= 0) continue;
            const g = dust.createRadialGradient(x, y, 0, x, y, m.s * 3);
            g.addColorStop(0, `rgba(255,226,160,${{a}})`);
            g.addColorStop(.35, `rgba(240,190,110,${{a * .45}})`);
            g.addColorStop(1, 'rgba(240,190,110,0)');
            dust.fillStyle = g; dust.beginPath(); dust.arc(x, y, m.s * 3, 0, 6.283); dust.fill();
          }}
          for (const b of bursts) {{
            const dt = t - b.t; if (dt < 0 || dt > 3.4) continue;
            const r = rng(100 + b.i);
            for (let n = 0; n < b.n; n++) {{
              const ang = r() * 6.283, sp = 250 + r() * 900;
              const p = {{ vx: Math.cos(ang) * sp, vy: Math.sin(ang) * sp - 200, s: 1.5 + r() * 4,
                           life: 1.4 + r() * 1.8, tw: r() * 6 }};
              if (dt > p.life) continue;
              const drag = (1 - Math.exp(-dt * 1.6)) / 1.6;
              const x = b.x + p.vx * drag, y = b.y + p.vy * drag + 90 * dt * dt;
              const al = (1 - dt / p.life) * (.6 + .4 * Math.sin(dt * 20 + p.tw));
              const g = dust.createRadialGradient(x, y, 0, x, y, p.s * 4);
              g.addColorStop(0, `rgba(255,248,225,${{al}})`);
              g.addColorStop(.3, `rgba(255,205,120,${{al * .6}})`);
              g.addColorStop(1, 'rgba(255,190,100,0)');
              dust.fillStyle = g; dust.beginPath(); dust.arc(x, y, p.s * 4, 0, 6.283); dust.fill();
            }}
          }}
          window.__oracleReady = true;
        """,
    )


@pytest.mark.gpu
@needs_chrome
def test_vignette_matches_chromium(cv, grey) -> None:
    html = stage_html(
        body='<div class="L" style="background:radial-gradient(ellipse 75% 60% at 50% 48%, transparent 45%, rgba(0,0,0,.55) 85%, rgba(0,0,0,.85) 100%)"></div>',
        bg="#808080",
    )
    expect_psnr("vignette", render_html(html), vignette(cv, grey).numpy(), 38)


@pytest.mark.gpu
@needs_chrome
@pytest.mark.parametrize("t", [0.4, 8.0])
def test_leaks_match_chromium(cv, grey, t) -> None:
    pred = leaks(cv, grey, t, 1.0, 165.0)
    expect_psnr(f"leaks t={t}", render_html(_leak_html(t)), pred.numpy(), 32)


def _rgb_u8(pred: np.ndarray) -> np.ndarray:
    got = np.asarray(pred)
    if got.dtype != np.uint8:
        if got.ndim == 3 and got.shape[0] in (1, 3, 4) and got.shape[-1] not in (1, 3, 4):
            got = np.transpose(got, (1, 2, 0))
        got = np.clip(np.rint(np.clip(got, 0, 1) * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(got[..., :3])


def _assert_effect_region(name: str, chrome: np.ndarray, pred: np.ndarray, floor: float = 32) -> None:
    """Match the motes, not the grey plate. A no-op stays under the floor on that region.

    Max-channel delta >= 8 means the plate's MSE there is at least 64, so its PSNR
    cannot clear 32 dB. Full-frame PSNR still passes for a no-op (~44 dB).
    """
    got = _rgb_u8(pred)
    plate = np.full(chrome.shape, 128, np.uint8)
    moved = np.max(np.abs(chrome.astype(np.int16) - plate.astype(np.int16)), axis=-1) >= 8
    n = int(moved.sum())
    assert n >= 1000, f"{name} effect region is {n} px"
    region = psnr(chrome[moved], got[moved])
    plate_region = psnr(chrome[moved], plate[moved])
    assert region >= floor, f"{name} effect PSNR {region:.2f} < {floor}"
    assert plate_region < floor, f"{name} no-op still clears the effect ({plate_region:.2f} dB)"
    expect_psnr(name, chrome, pred, floor)


@pytest.mark.gpu
@needs_chrome
@pytest.mark.parametrize("t", [2.5, 6.0])
def test_dust_matches_chromium(cv, grey, t) -> None:
    pred = dust(cv, grey, t, 1.0, ())
    _assert_effect_region(f"dust t={t}", render_html(_dust_html(t, 1.0, [])), pred.numpy())


@pytest.mark.gpu
@needs_chrome
def test_burst_midflight_matches_chromium(cv, grey) -> None:
    bursts = ((2, 5.0, 540.0, 1000.0, 16),)
    t = 5.7
    pred = dust(cv, grey, t, 1.0, bursts)
    _assert_effect_region("burst", render_html(_dust_html(t, 1.0, list(bursts))), pred.numpy())


@pytest.mark.gpu
@needs_chrome
def test_grain_tile_and_offset(cv, grey) -> None:
    f = 5  # tile 1, offset ((5*97)%256, (5*57)%256) = (229, 29)
    html = stage_html(
        css="#grain { mix-blend-mode: overlay; opacity: .16; background-repeat: repeat; }",
        body='<div id="grain" class="L"></div>',
        bg="#808080",
        script=f"""
          window.__oracleReady = false;
          {_RNG}
          const c = document.createElement('canvas'); c.width = c.height = 256;
          const x = c.getContext('2d'), d = x.createImageData(256, 256), r = rng(7 + {f % 4});
          for (let j = 0; j < d.data.length; j += 4) {{
            const v = r() * 255; d.data[j] = d.data[j + 1] = d.data[j + 2] = v; d.data[j + 3] = 255;
          }}
          x.putImageData(d, 0, 0);
          const g = document.getElementById('grain');
          g.style.backgroundImage = `url(${{c.toDataURL()}})`;
          g.style.backgroundPosition = '{(f * 97) % 256}px {(f * 57) % 256}px';
          window.__oracleReady = true;
        """,
    )
    expect_psnr("grain", render_html(html), grain(cv, grey, f).numpy(), 32)


@pytest.mark.gpu
@needs_chrome
def test_hit_flash_matches_chromium(cv, grey) -> None:
    html = stage_html(
        css="#flash { background: radial-gradient(circle at 50% 45%, #fff8e6, #f3c47a 60%, #b0662a); mix-blend-mode: screen; }",
        body='<div id="flash" class="L" style="opacity:.7"></div>',
        bg="#808080",
    )
    pred = hits(cv, grey, 10.0, [(10.0, 0.7)])
    expect_psnr("hit", render_html(html), pred.numpy(), 38)


@pytest.mark.gpu
@needs_chrome
def test_bars_match_chromium(cv, grey) -> None:
    html = stage_html(
        body='<div class="L"><div style="position:absolute;left:0;top:0;width:1080px;height:80px;background:#000"></div>'
             '<div style="position:absolute;left:0;bottom:0;width:1080px;height:80px;background:#000"></div></div>',
        bg="#808080",
    )
    expect_psnr("bars", render_html(html), bars(cv, grey, 80).numpy(), 38)
