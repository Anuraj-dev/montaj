"""GPU parity vs film2.py (the M1 oracle). At 1080x1920 every primitive must match
within mean |d| < 1e-3 and max |d| < 2/255.

film/film2 allocate CUDA tensors at import time, so the oracle is imported lazily by the
CUDA-guarded `oracle` fixture; the CPU-only tests compare against inline formulas."""
import importlib
import math
import os
import sys

import pytest
import torch
import yaml

from montaj.render import core as mc
from montaj.render import finish as mf
from montaj.render import transitions as mt
from montaj.render import wall as mw
from montaj.render.canvas import Canvas

V2 = os.path.expanduser("~/Anuraj-dev/montaj-projects/_oracle/v2")
W, H = 1080, 1920
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(V2, "src")


def load(cv, stem, crop=None, card=False):
    return mc.Photo(cv, os.path.join(SRC, f"{stem}.jpg"),
                    crop=tuple(crop) if crop else None, card=card)


def close(a, b, what):
    d = (a - b).abs()
    mean, mx = d.mean().item(), d.max().item()
    assert mean < 1e-3 and mx < 2 / 255, f"{what}: mean {mean:.2e} max {mx:.2e}"


def wall_spec():
    with open(os.path.join(ROOT, "recipes", "birthday-short.yaml")) as fh:
        return yaml.safe_load(fh)["shots"][-1]["wall"]


@pytest.fixture(scope="module")
def oracle():
    """The film.py / film2.py pair, imported only where a GPU exists to hold their tensors."""
    torch.cuda.is_available() or pytest.skip("needs CUDA")
    cwd, sys.path[:] = os.getcwd(), [V2] + sys.path
    try:
        f2 = importlib.import_module("film2")
        os.chdir(V2)  # film loads src/<n>.jpg relatively
        f2.build_wall()
        yield importlib.import_module("film"), f2
    finally:
        os.chdir(cwd)
        sys.path.remove(V2)


@pytest.fixture(scope="module")
def cv():
    torch.cuda.is_available() or pytest.skip("needs CUDA")
    with torch.no_grad():
        yield Canvas(W, H, device="cuda")


@pytest.fixture(scope="module")
def walls(cv, oracle):
    with torch.no_grad():
        yield mw.build_wall(cv, wall_spec(), lambda stem, crop, card: load(cv, stem, crop, card))
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- easing, no GPU needed
def ref_ease(name, x):
    """film.py's easings, restated so the CPU test needs no oracle."""
    x = max(0.0, min(1.0, x))
    if name == "smooth":
        return x * x * (3 - 2 * x)
    if name == "cubic":
        return 4 * x**3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
    if name == "outc":
        return 1 - (1 - x) ** 3
    if name == "inc":
        return x**3
    if name == "expo":
        if x in (0, 1):
            return x
        return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2
    if name == "ramp":
        return x - 0.75 * math.sin(2 * math.pi * x) / (2 * math.pi)
    if name == "linear":
        return x
    raise AssertionError(name)


def ref_key(t, keys):
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0, _), (t1, v1, e) in zip(keys, keys[1:]):
        if t <= t1:
            x = ref_ease(e or "smooth", (t - t0) / (t1 - t0))
            return tuple(a + (b - a) * x for a, b in zip(v0, v1)) if isinstance(v0, tuple) else v0 + (v1 - v0) * x
    return keys[-1][1]


def ref_keys_s(t, keys):
    return math.exp(ref_key(t, [(a, math.log(v), e) for a, v, e in keys]))


def callable_keys(keys):
    """Key lists as mc.key/keys_s want them: ease names already resolved through EASE."""
    return [(t, v, mc.EASE[e] if e else None) for t, v, e in keys]


XS = (0.0, 0.13, 0.5, 0.77, 1.0)
KEYS = [(0, (1.0, 2.0), None), (10, (3.0, 4.0), "cubic"), (20, (5.0, 6.0), None)]
SKEYS = [(0, 1.5, None), (10, 2.5, "cubic")]


def test_easing_cpu():
    for name in ("smooth", "cubic", "outc", "inc", "expo", "ramp"):
        for x in XS:
            assert getattr(mc, name)(x) == pytest.approx(ref_ease(name, x))
    for x in XS + (-0.2, 1.5):
        assert mc.linear(x) == pytest.approx(ref_ease("linear", x))
    for t in (0, 3, 10, 15, 20, 25):
        assert mc.key(t, callable_keys(KEYS)) == pytest.approx(ref_key(t, KEYS))
    for t in (0, 5, 10, 15):
        assert mc.keys_s(t, callable_keys(SKEYS)) == pytest.approx(ref_keys_s(t, SKEYS))


@pytest.mark.gpu
def test_easing_vs_oracle(oracle):
    fm, _ = oracle
    for name in ("smooth", "cubic", "outc", "inc", "expo", "ramp"):
        for x in XS:
            assert getattr(mc, name)(x) == pytest.approx(getattr(fm, name)(x))
    for x in XS + (-0.2, 1.5):
        assert mc.linear(x) == pytest.approx(fm.clamp01(x))
    keys = [(0, (1.0, 2.0), None), (10, (3.0, 4.0), fm.cubic), (20, (5.0, 6.0), None)]
    for t in (0, 3, 10, 15, 20, 25):
        assert mc.key(t, keys) == pytest.approx(fm.key(t, keys))
    skeys = [(0, 1.5, None), (10, 2.5, fm.cubic)]
    for t in (0, 5, 10, 15):
        assert mc.keys_s(t, skeys) == pytest.approx(fm.keys_s(t, skeys))


# ---------------------------------------------------------------- stills
@pytest.mark.gpu
def test_still_sky(cv, oracle):
    _, f2 = oracle
    with torch.no_grad():
        close(mc.still(cv, load(cv, "16"), focus=(470, 745), zoom=1.3), f2.shot("sky"), "sky")


@pytest.mark.gpu
def test_still_crop_default_zoom(cv, oracle):
    _, f2 = oracle
    with torch.no_grad():
        mine = mc.still(cv, load(cv, "22", (60, 180, 865, 1330)))
        close(mine, f2.shot("22"), "22")


# ---------------------------------------------------------------- transitions, off the midpoint
@pytest.mark.gpu
def test_swirl(cv, oracle):
    _, f2 = oracle
    with torch.no_grad():
        a, b = f2.shot("sky"), f2.shot("gogh")
        c = (W / 2 + (431 - 470) * 1.3, H / 2 + (578 - 745) * 1.3)
        for x in (0.2, 0.5, 0.7):
            close(mt.swirl(cv, a, b, x, c), f2.swirl(a, b, x, c), f"swirl x={x}")


@pytest.mark.gpu
def test_whip(cv, oracle):
    _, f2 = oracle
    with torch.no_grad():
        a, b = f2.shot("19"), f2.shot("09")
        for x in (0.2, 0.5, 0.7):
            t = 225 - 7 + 14 * x  # film2 counts progress from the cut: (t - (tc - 7)) / 14
            for axis in ("x", "y"):
                close(mt.whip(cv, a, b, x, axis), f2.whip(t, 225, a, b, axis), f"whip-{axis} x={x}")


@pytest.mark.gpu
def test_fade(cv, oracle):
    _, f2 = oracle
    with torch.no_grad():
        a, b = f2.shot("38"), f2.shot("29")
        for x in (0.2, 0.5, 0.7):
            close(mt.fade(a, b, x), f2.fade(a, b, x), f"fade x={x}")


# ---------------------------------------------------------------- wall
def _film2_wall_ts(fm, f2, t):
    A = fm.wpt(f2.WALL["34"], (420, 640))
    sA = 1.606 / f2.WALL["34"]["p"]
    D = fm.wpt(f2.WALL["10"], (512, 740))
    sD = 1.36
    T = fm.key(t, [(0, A, None), (30, A, None), (95, A, None), (205, D, f2.cubic), (375, D, None)])
    s = fm.keys_s(t, [(0, sA, None), (30, sA, None), (95, 0.42, f2.cubic),
                      (205, 0.42, None), (275, sD, f2.cubic), (375, sD, None)])
    return T, s


@pytest.mark.gpu
def test_wall(cv, walls, oracle):
    fm, f2 = oracle
    with torch.no_grad():
        for t in (0, 150, 300):
            T, s = _film2_wall_ts(fm, f2, t)
            close(mw.wall_frame(cv, walls, T, s), f2.s_wall(t), f"wall t={t}")
            torch.cuda.empty_cache()


@pytest.mark.gpu
def test_wall_camera_tracks_recipe(cv, walls):
    with torch.no_grad():
        cam = wall_spec()["camera"]
        A = mw.wpt(walls["34"], (420, 640))
        T, s = mw.wall_camera(walls, cam, 0)
        assert T == pytest.approx(A) and s == pytest.approx(1.606 / walls["34"]["p"])
        T, s = mw.wall_camera(walls, cam, 110)  # bare keys hold
        assert T == pytest.approx(A) and s == pytest.approx(0.42)


@pytest.mark.gpu
def test_wall_end_to_end_recipe_camera(cv, walls, oracle):
    """The recipe's wall keys are film2's shifted +15f: recipe wall starts at 450, film2's at 465."""
    fm, f2 = oracle
    with torch.no_grad():
        cam = wall_spec()["camera"]
        for t in (0, 60, 100, 150, 240, 300):
            T, s = mw.wall_camera(walls, cam, t + 15)
            want_T, want_s = _film2_wall_ts(fm, f2, t)
            assert T == pytest.approx(want_T) and s == pytest.approx(want_s), f"wall camera t={t}"
            close(mw.wall_frame(cv, walls, T, s), f2.s_wall(t), f"wall end-to-end t={t}")
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- finish
@pytest.mark.gpu
def test_finish(cv, oracle):
    _, f2 = oracle
    with torch.no_grad():
        img = f2.shot("sky")
        for f in (0, 10, 70, 120, 760):
            torch.manual_seed(999)
            want = f2.finish(img, f)
            torch.manual_seed(999)
            got = mf.finish(cv, img, f, mf.LOOKS["warm-film"], f2.FLASH, f2.LEAKS,
                            ("white-lift", 30), ("glow", 110, 730))
            close(got, want, f"finish f={f}")
            del want, got
        torch.cuda.empty_cache()