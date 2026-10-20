"""Card shot: a solid colour plus text as one shot picture."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

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


def test_card_renders_colour_and_text(tmp_path: Path) -> None:
    """The card scene itself (before the look's vignette and grain) is the colour with glyphs on it."""
    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer

    p = _write(tmp_path, _CARD)
    p.write_text(p.read_text().replace("1080x1920", "270x480"))
    tl = resolve(load_spec(p), tmp_path)
    r = Renderer(tl, tmp_path, Canvas(tl.size[0], tl.size[1], device="cpu"))
    r._ensure_rasters()
    img = r._eval(tl.plan(15).scene, 15.0)
    assert img.shape == (3, 480, 270)
    flat = torch.tensor([0x22, 0x33, 0x44], dtype=torch.float32) / 255.0
    diff = (img - flat[:, None, None]).abs().amax(dim=0)
    assert diff[:40, :].max() < 1e-4  # the top is the plain colour
    assert diff.max() > 0.2  # glyphs sit on it
    assert diff[int(900 / 1920 * 480) - 30 : int(900 / 1920 * 480) + 30, :].max() > 0.2  # near y: 900


def test_card_nonfinite_numeric_is_validation_error(tmp_path: Path) -> None:
    errs = _errors(tmp_path, _CARD.replace("at: 0s", "at: 0s, weight: .nan"))
    assert any("card.text[0].weight" in e and "finite" in e for e in errs)
    errs = _errors(tmp_path, _CARD.replace("at: 0s", "at: 0s, size: -4"))
    assert any("card.text[0].size" in e for e in errs)


def test_card_field_does_not_change_existing_hashes(tmp_path: Path) -> None:
    """A film with no card shot hashes exactly as it did before the card field existed."""
    from PIL import Image

    (tmp_path / "assets").mkdir()
    Image.new("RGB", (40, 60)).save(tmp_path / "assets" / "p1.jpg")
    p = _write(tmp_path, "  - photo: p1\n    hold: 20f\n  - blank: true\n    hold: 10f\n")
    tl = resolve(load_spec(p), tmp_path)
    got = [segment_hash(tl, seg, {"p1": "abc"}, "e1") for seg in tl.segments]
    assert got == [
        "f3275515065843b5dcf908991483c46a0ce59a18825e792e0c966d14bf83be76",
        "cdda609de59fa0089e05b129c226170783a281a9ebc7a1bbb061a43036b35d5f",
    ]


def test_card_lines_take_the_new_text_styles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Card text uses the same line fields as a text block: it renders, and its ext fonts reach the hash."""
    import montaj.project as project
    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer

    styled = _CARD.replace("at: 0s", "at: 0s, style: sans, box: chip, align: left")
    p = _write(tmp_path, styled)
    p.write_text(p.read_text().replace("1080x1920", "270x480"))
    tl = resolve(load_spec(p), tmp_path)
    r = Renderer(tl, tmp_path, Canvas(tl.size[0], tl.size[1], device="cpu"))
    r._ensure_rasters()
    assert r._eval(tl.plan(15).scene, 15.0).shape == (3, 480, 270)

    seen: list[frozenset[str] | None] = []
    real = project._text_asset_shas
    monkeypatch.setattr(project, "_text_asset_shas", lambda ext=None: seen.append(ext) or real(ext))
    segment_hash(tl, tl.segments[0], {}, "e1")
    assert seen == [frozenset({"sans"})]
    assert any("card.text[0].box" in e and "need a box" in e for e in _errors(tmp_path, _CARD.replace("at: 0s", 'at: 0s, dot: "#ff0000"')))
