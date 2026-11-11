"""#58 End on black: a textless colour card, and `video.outro.hold` (black held after the fade)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from montaj.cli import main
from montaj.project import segment_hash
from montaj.spec import IntroOutro, SpecError, load_spec
from montaj.timeline import Card, resolve

ROOT = Path(__file__).resolve().parents[1]

# A bright last shot with a text block on it to the very end: a fade alone never holds black here.
_FILM = (
    "video:\n  size: 270x480\n  fps: 30\n  look: clean\n  outro: {type: fade, dur: 15f, hold: 10f}\n"
    "assets: assets\nshots:\n"
    '  - card: {color: "#000"}\n    hold: 30f\n'
    '  - card: {color: "#ffffff", text: []}\n    hold: 30f\n'
    "text:\n  - from: 1s\n    to: 2s\n"
    '    lines: [{text: "STAY", size: 120, y: 900, at: 1s, color: "#ff2200"}]\n'
    "check:\n  expect:\n    - {black: [1.67s, 2s], why: held black}\n"
)


def _write(tmp: Path, text: str) -> Path:
    (tmp / "assets").mkdir(exist_ok=True)
    p = tmp / "montaj.yaml"
    p.write_text(text)
    return p


def test_textless_card_validates_and_short_hex_reads_long(tmp_path: Path) -> None:
    spec = load_spec(_write(tmp_path, _FILM))
    assert spec.shots[0].card.color == "#000000" and spec.shots[0].card.text == []
    assert spec.shots[1].card.text == []
    tl = resolve(spec, tmp_path)
    assert isinstance(tl.plan(0).scene, Card)


def test_textless_card_renders_flat(tmp_path: Path) -> None:
    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer

    tl = resolve(load_spec(_write(tmp_path, _FILM)), tmp_path)
    r = Renderer(tl, tmp_path, Canvas(tl.size[0], tl.size[1], device="cpu"))
    r._ensure_rasters()
    black = r._eval(tl.plan(10).scene, 10.0)
    white = r._eval(tl.plan(40).scene, 40.0)
    assert black.shape == (3, 480, 270)
    assert black.abs().max() < 1e-6
    assert (white - 1.0).abs().max() < 1e-6


def test_outro_hold_moves_the_fade_and_holds_black_over_text(tmp_path: Path) -> None:
    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer

    tl = resolve(load_spec(_write(tmp_path, _FILM)), tmp_path)
    assert tl.n_frames == 60
    assert tl.plan(59).fade_out == (15, 60 - 15 - 10)
    assert tl.plan(34).fade_out is None
    r = Renderer(tl, tmp_path, Canvas(tl.size[0], tl.size[1], device="cpu"))
    assert r.frame(45).abs().max() > 0.2  # text and white card before the fade ends
    for f in (50, 55, 59):  # the held frames: card and text both black
        assert r.frame(f).abs().max() == 0.0


def test_outro_hold_validation(tmp_path: Path) -> None:
    def errors(text: str) -> list[str]:
        with pytest.raises(SpecError) as ei:
            load_spec(_write(tmp_path, text))
        return ei.value.errors

    glow = errors(_FILM.replace("type: fade, dur: 15f, hold: 10f", "type: glow, dur: 15f, hold: 10f"))
    assert any("video.outro.hold" in e and "fade" in e for e in glow)
    intro = errors(_FILM.replace("  outro:", "  intro: {type: white-lift, dur: 10f, hold: 5f}\n  outro:"))
    assert any("video.intro.hold" in e for e in intro)
    bad = errors(_FILM.replace("hold: 10f}", "hold: soon}"))
    assert any("video.outro.hold" in e for e in bad)
    for zero in ("0f", "0.01s"):
        short = errors(_FILM.replace("hold: 10f}", f"hold: {zero}}}"))
        assert any("video.outro.hold" in e and "at least 1 frame" in e for e in short), short
    no_fade = errors(_FILM.replace("dur: 15f, hold: 10f", "dur: 0f, hold: 10f"))
    assert any("video.outro.hold" in e and "needs a fade" in e for e in no_fade), no_fade


def test_outro_hold_past_the_film_is_a_resolve_error(tmp_path: Path) -> None:
    text = _FILM.replace("dur: 15f, hold: 10f", "dur: 15f, hold: 50f")  # 65f on a 60f film
    spec = load_spec(_write(tmp_path, text))
    with pytest.raises(SpecError) as ei:
        resolve(spec, tmp_path)
    assert any("video.outro.hold" in e and "15f" in e and "50f" in e and "60f" in e for e in ei.value.errors)
    tl = resolve(load_spec(_write(tmp_path, _FILM.replace("hold: 10f", "hold: 45f"))), tmp_path)
    assert tl.plan(59).fade_out == (15, 0)  # exactly the film length is fine


def test_hold_reaches_the_hash_and_films_without_it_keep_theirs(tmp_path: Path) -> None:
    """Hashes pinned from 877aef8 (before `hold` existed) for the same film with `outro: {fade, 15f}`."""
    base_text = (
        'video:\n  size: 270x480\n  fps: 30\n  look: clean\n  outro: {type: fade, dur: 15f}\n'
        'assets: assets\nshots:\n  - card: {color: "#ffffff", text: [{text: "Hi", y: 900, at: 0s}]}\n'
        "    hold: 30f\n  - blank: true\n    hold: 30f\n"
    )

    def hashes(text: str) -> list[str]:
        tl = resolve(load_spec(_write(tmp_path, text)), tmp_path)
        return [segment_hash(tl, seg, {}, "e1") for seg in tl.segments]

    base = hashes(base_text)
    assert base == [
        "78afa964005142215b9befcc02e6e874795522d3bd7c3bcdc8e2871c508ffacf",
        "5f03bba3fa71eefa38f7312ce9a3e8f442023c13ded8c593466b64034cb780b8",
    ]
    held = hashes(base_text.replace("dur: 15f}", "dur: 15f, hold: 5f}"))
    assert held[1] != base[1]  # the tail segment, where the fade now ends early
    assert IntroOutro(type="fade", dur="15f").model_dump() == {"type": "fade", "dur": "15f"}


def test_held_black_is_an_intended_freeze(tmp_path: Path) -> None:
    from montaj.cli import _intended_freeze

    from PIL import Image

    # A drifting photo last: its own frames are not a designed hold, only the held black is.
    film = _FILM.replace('  - card: {color: "#ffffff", text: []}\n    hold: 30f\n',
                         "  - photo: p1\n    hold: 30f\n    drift: {zoom: [1.0, 1.2]}\n")
    _write(tmp_path, film.replace(", hold: 10f}", "}"))
    Image.new("RGB", (270, 480), "white").save(tmp_path / "assets" / "p1.jpg")
    assert _intended_freeze(tmp_path) == [(0.0, 1.0)]  # no hold: just the black card
    _write(tmp_path, film.replace("hold: 10f}", "hold: 20f}"))
    assert _intended_freeze(tmp_path) == [(0.0, 1.0), (40 / 30, 2.0)]


@pytest.mark.parametrize(
    ("recipe", "end", "hold"), [("tech-launch", 14.0, 0.25), ("gaming-highlight", 30.0, 0.5)]
)
def test_recipes_expect_the_held_black(recipe: str, end: float, hold: float) -> None:
    spec = yaml.safe_load((ROOT / "recipes" / f"{recipe}.yaml").read_text())
    outro = spec["video"]["outro"]
    assert outro["type"] == "fade" and "hold" in outro
    black = [e["black"] for e in spec["check"]["expect"] if "black" in e]
    assert len(black) == 1
    lo, hi = (float(str(v).rstrip("s")) for v in black[0])
    assert hi == end and end - hold <= lo  # the span sits inside the held black, not the fade


@pytest.mark.gpu
def test_held_black_is_detected_by_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path, _FILM.replace("270x480", "540x960"))  # NVENC rejects the tiny preview size
    assert main(["-C", str(tmp_path), "render"]) == 0
    assert main(["-C", str(tmp_path), "check"]) == 0
    out = capsys.readouterr().out
    assert "ignored as intended: held black" in out, out
    assert "expect black" not in out, out
    assert "WARN frozen" not in out, out


def _demo_with_text_over_the_end(tmp: Path, recipe: str, frm: str, to: str) -> Path:
    proj = tmp / "demo"
    assert main(["new", str(proj), "--recipe", recipe]) == 0
    spec_file = proj / "montaj.yaml"
    spec = yaml.safe_load(spec_file.read_text())
    spec.setdefault("text", []).append({
        "from": frm, "to": to, "fade_in": "0s", "fade_out": "0s",
        "lines": [{"text": "STILL HERE", "size": 160, "y": 500, "at": frm, "color": "#ffffff"}],
    })
    spec_file.write_text(yaml.safe_dump(spec, sort_keys=False))
    return proj


@pytest.mark.gpu
@pytest.mark.parametrize(("recipe", "frm", "to"), [("tech-launch", "12s", "14s"), ("gaming-highlight", "28s", "30s")])
def test_recipes_end_on_black_under_added_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], recipe: str, frm: str, to: str
) -> None:
    proj = _demo_with_text_over_the_end(tmp_path, recipe, frm, to)
    assert main(["-C", str(proj), "render"]) == 0
    assert main(["-C", str(proj), "check"]) == 0
    out = capsys.readouterr().out
    assert "ignored as intended: fade to black" in out, out
    assert "expect black" not in out, out
    assert out.rstrip().splitlines()[-1] == "OK check", out  # the held black is no frozen WARN either
