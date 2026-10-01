"""Resolve recipes/birthday-song-film.yaml against film.html cues; birthday-short still loads."""

from __future__ import annotations

import math
from pathlib import Path
from shutil import copyfile

import pytest
from PIL import Image

from montaj.cli import main
from montaj.spec import load_spec, round_frame
from montaj.timeline import resolve

ROOT = Path(__file__).resolve().parents[1]
FILM_ASSETS = Path.home() / "Anuraj-dev/claude-test/bday-video/assets"
FPS = 30
BEAT = 60 / 76
DUR = 165.0

# film.html 79–86 C — independent of the recipe
C = {
    "hum": 5.2, "v1": 12.0, "masjid": 17.9, "mehendi": 23.74, "chaar": 29.88,
    "pre": 34.9, "pre2": 41.4,
    "ch1": 47.96, "ch1b": 52.82, "ch1c": 58.8, "ch1d": 64.58, "ch1e": 68.1,
    "ch1end": 72, "v2": 72, "pahad": 77.78, "gaadi": 83.42, "so": 88.46,
    "muskaan": 94.22, "br": 99.98, "br2": 105.98, "br3": 112.38, "build": 118.4,
    "ch2": 124.88, "ch2b": 129.84, "ch2c": 136.84, "janam": 141.6, "jaan": 145.96,
    "arshiya": 148.04, "hb": 151.06, "hamesha": 154.36, "samar": 157.3,
}

BIRTHDAY_SHORT_STEMS = [
    "09", "10", "12", "16", "19", "20", "21", "22",
    "28", "29", "32", "33", "34", "35", "37", "38",
]

# 20 times spanning every dustK/leakK/barH piece
SAMPLE_TIMES = (
    0.0, 20.0, 46.5, 47.2, 47.6, 47.96, 55.0,
    72.0, 73.0, 74.0, 80.0,
    99.5, 100.0, 106.0,
    112.38, 118.4, 124.88,
    140.0, 151.5, 160.0,
)


def _P(t: float, a: float, b: float) -> float:
    return min(1.0, max(0.0, (t - a) / (b - a)))


def _es(k: float) -> float:
    k = min(1.0, max(0.0, k))
    return 0.5 - 0.5 * math.cos(math.pi * k)


def dustK(t: float) -> float:
    """film.html 372."""
    return (
        0.35
        + 0.5 * _P(t, C["ch1"] - 1, C["ch1"])
        - 0.35 * _P(t, C["ch1end"], C["ch1end"] + 3)
        + 0.7 * _P(t, C["br3"], C["ch2"])
        - 0.6 * _P(t, C["hb"], C["hb"] + 1.5)
    )


def leakK(t: float) -> float:
    """film.html 404."""
    return (
        0.25
        + 0.35 * _P(t, C["ch1"] - 0.5, C["ch1"])
        - 0.35 * _P(t, C["v2"], C["v2"] + 3)
        + 0.45 * _P(t, C["br3"], C["ch2"])
        - 0.5 * _P(t, C["hb"], C["hb"] + 1)
    )


def barH(t: float) -> float:
    """film.html 406."""
    return (
        150 * (1 - _es(_P(t, C["ch1"] - 0.6, C["ch1"] + 0.4)))
        + 150 * _es(_P(t, C["v2"], C["v2"] + 2)) * (1 - _es(_P(t, C["br"] - 0.5, C["br"] + 0.5)))
    )


def film_photo_cues() -> list[float]:
    """film.html Photo t0 list, 193–346, independent of the recipe."""
    cues = [
        0.0, 6.8, C["v1"], C["v1"] + 2.8, C["masjid"], C["masjid"] + 2.5,
        C["mehendi"], C["mehendi"] + 2.7, C["chaar"], C["pre"], C["pre2"],
        C["pre2"] + 2.9, C["ch1"], C["ch1b"], C["ch1c"], C["ch1d"], C["ch1e"],
        C["v2"], C["pahad"], C["gaadi"], C["muskaan"], C["br"],
    ]
    n_mont = math.floor((C["br2"] - C["br"]) / BEAT) + 1
    cues.extend(C["br"] + i * BEAT for i in range(n_mont))
    cues.append(C["br2"])
    cues.append(C["br3"])
    a = C["build"] + 2
    d = BEAT
    while a < C["ch2"] - 0.05:
        cues.append(a)
        b = min(a + d, C["ch2"])
        a = b
        d = max(BEAT * 0.25, d * 0.8)
    step = BEAT * 2
    n_pick = math.floor((C["ch2c"] - C["ch2"]) / step)
    cues.extend(C["ch2"] + i * step for i in range(n_pick))
    cues.extend([C["ch2c"], C["janam"], C["hb"]])
    return cues


def _jpeg(folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(folder / f"{stem}.jpg", "JPEG")


def _film_project(tmp: Path) -> Path:
    proj = tmp / "film"
    assets = proj / "assets"
    assets.mkdir(parents=True)
    for src in (FILM_ASSETS / "photos").glob("*.jpg"):
        (assets / src.name).symlink_to(src.resolve())
    for src in (FILM_ASSETS / "gen").glob("*.png"):
        (assets / src.name).symlink_to(src.resolve())
    (proj / "music").mkdir()
    (proj / "music" / "song.wav").write_bytes(b"RIFF")
    copyfile(ROOT / "recipes" / "birthday-song-film.yaml", proj / "montaj.yaml")
    return proj


def _short_project(tmp: Path) -> Path:
    proj = tmp / "short"
    assets = proj / "assets"
    assets.mkdir(parents=True)
    for stem in BIRTHDAY_SHORT_STEMS:
        _jpeg(assets, stem)
    copyfile(ROOT / "recipes" / "birthday-short.yaml", proj / "montaj.yaml")
    return proj


def test_birthday_song_film_resolves(tmp_path: Path) -> None:
    proj = _film_project(tmp_path)
    spec = load_spec(proj / "montaj.yaml")
    tl = resolve(spec, proj)
    assert tl.n_frames == 4950
    assert tl.n_frames == int(DUR * FPS)
    blank = [s for s in tl.shots if s.kind == "blank"]
    assert len(blank) == 1
    assert blank[0].start == round_frame(161.8 * FPS)
    assert blank[0].end == tl.n_frames
    indents = [
        line.indent
        for block in tl.texts
        for line in block.lines
        if line.text == "ARSHIYA" and line.gold
    ]
    assert indents == [0.16, 0.18]
    starts = [s.start for s in tl.shots]
    for t in film_photo_cues():
        cue = round_frame(t * FPS)
        assert any(abs(st - cue) <= 1 for st in starts), (t, cue, starts)


def test_birthday_song_film_tracks(tmp_path: Path) -> None:
    proj = _film_project(tmp_path)
    tl = resolve(load_spec(proj / "montaj.yaml"), proj)
    assert len(SAMPLE_TIMES) == 20
    for t in SAMPLE_TIMES:
        f = min(max(round_frame(t * FPS), 0), tl.n_frames - 1)
        ts = f / FPS
        plan = tl.plan(f)
        assert plan.dust == pytest.approx(dustK(ts), abs=1e-6)
        assert plan.glow == pytest.approx(leakK(ts), abs=1e-6)
        assert plan.bars == pytest.approx(barH(ts), abs=1e-6)


def test_birthday_short_still_resolves(tmp_path: Path) -> None:
    proj = _short_project(tmp_path)
    spec = load_spec(proj / "montaj.yaml")
    tl = resolve(spec, proj)
    assert tl.n_frames == 840


def test_birthday_song_film_validate(tmp_path: Path, capsys) -> None:
    proj = _film_project(tmp_path)
    code = main(["-C", str(proj), "validate"])
    out = capsys.readouterr().out.strip()
    assert code == 0
    assert out == "OK montaj.yaml 4950f 165.00s 56 shots 56 segments"
