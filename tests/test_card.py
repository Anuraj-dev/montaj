"""Card shot: a solid colour plus text as one shot picture."""
from __future__ import annotations

from pathlib import Path

import pytest

from montaj.project import segment_hash
from montaj.spec import SpecError, load_spec
from montaj.timeline import Card, resolve

_CARD = (
    '  - card: {color: "#223344", text: [{text: "Hello", y: 900, at: 0s}]}\n'
    "    hold: 20f\n"
)


def _write(tmp: Path, body: str) -> Path:
    (tmp / "assets").mkdir(exist_ok=True)
    p = tmp / "montaj.yaml"
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 76\n  look: golden-film\n"
        f"assets: assets\nshots:\n{body}"
    )
    return p


def _errors(tmp: Path, body: str) -> list[str]:
    with pytest.raises(SpecError) as ei:
        load_spec(_write(tmp, body))
    return ei.value.errors


def test_card_resolves_to_card_scene(tmp_path: Path) -> None:
    tl = resolve(load_spec(_write(tmp_path, _CARD)), tmp_path)
    scene = tl.plan(0).scene
    assert isinstance(scene, Card) and scene.shot == 0


def test_card_validation_errors(tmp_path: Path) -> None:
    bad_color = _errors(tmp_path, _CARD.replace("#223344", "navy"))
    assert any("card.color" in e and "#rrggbb" in e for e in bad_color)
    bare = _errors(tmp_path, _CARD.replace("at: 0s", "at: 2"))
    assert any("card.text[0].at" in e for e in bare)
    marker = _errors(tmp_path, _CARD.replace("at: 0s", "at: v1"))
    assert any("card.text[0].at" in e for e in marker)
    empty = _errors(tmp_path, '  - card: {color: "#223344", text: []}\n    hold: 5f\n')
    assert any("card.text" in e and "at least one line" in e for e in empty)
    both = _errors(tmp_path, _CARD.replace("hold: 20f", "hold: 20f\n    blank: true"))
    assert any("mutually exclusive" in e for e in both)
    style = _errors(tmp_path, _CARD.replace("at: 0s", "at: 0s, style: bogus"))
    assert any("card.text[0].style" in e for e in style)


def test_card_edits_reach_segment_hash(tmp_path: Path) -> None:
    def hashes(body: str) -> list[str]:
        tl = resolve(load_spec(_write(tmp_path, body)), tmp_path)
        return [segment_hash(tl, seg, {}, "e1") for seg in tl.segments]

    base = hashes(_CARD)
    assert hashes(_CARD) == base
    assert hashes(_CARD.replace("#223344", "#223345")) != base
    assert hashes(_CARD.replace("Hello", "Hullo")) != base


@pytest.mark.gpu
def test_card_renders_colour_and_text(tmp_path: Path) -> None:
    """One card shot paints its colour and, once revealed, glyphs over it. Needs CUDA and Chromium."""
    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer

    p = _write(tmp_path, _CARD)
    p.write_text(p.read_text().replace("1080x1920", "270x480"))
    tl = resolve(load_spec(p), tmp_path)
    cv = Canvas(tl.size[0], tl.size[1], device="cuda")
    img = Renderer(tl, tmp_path, cv).frame(15)
    assert img.shape == (3, 480, 270)
    corner = img[:, 0, 0]
    assert (img - corner[:, None, None]).abs().amax() > 0.2  # glyphs differ from the colour
    assert (img[:, :40, :] - corner[:, None, None]).abs().amax() < 0.05  # top is plain colour
