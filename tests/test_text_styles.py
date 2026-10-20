"""Sans / mono styles, typewriter reveal, chips and lower-thirds."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from montaj.render import text as text_mod
from montaj.render.text import TextBlock, TextLine, _font_digest, _key, _unit_ks, line_markup, raster_line
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

from tests.test_spec import _write


def _spec(tmp: Path, text: str) -> Path:
    return _write(tmp, "  - {photo: a, hold: 5s}\n", ["a"], extra=text)


def _errors(tmp: Path, text: str) -> list[str]:
    with pytest.raises(SpecError) as ei:
        load_spec(_spec(tmp, text))
    return ei.value.errors


LINE = "text:\n  - from: 0s\n    to: 3s\n    lines:\n      - {text: hi, y: 100, at: 0s, %s}\n"


def test_markup_per_new_style():
    assert 'class="txt sans"' in line_markup(TextLine("Theme", "sans", 120, 0, weight=600))
    assert "font-weight:600" in line_markup(TextLine("Theme", "sans", 120, 0, weight=600))
    assert 'class="txt mono"' in line_markup(TextLine("ls", "mono", 120, 0))
    typed = line_markup(TextLine("abc", "mono", 120, 0, reveal="type"))
    assert typed.count('class="w"') == 3  # typewriter splits by char


def test_chip_markup_and_old_markup_unchanged():
    chip = line_markup(TextLine("ok", "sans", 10, 0, box="chip", dot="gold", box_color="#112233"))
    assert '<span class="chip" style="--chip:#112233">' in chip and "--dot:#f0c77e" in chip
    plain = line_markup(TextLine("ok", "serif", 10, 0))
    assert "chip" not in plain and "al-" not in plain


def test_font_added_changes_cache_key(tmp_path, monkeypatch):
    base = _key("line", "x", 1.0, b"css", _font_digest())
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    for p in text_mod._FONTS.glob("*.woff2"):
        (fonts / p.name).write_bytes(p.read_bytes())
    monkeypatch.setattr(text_mod, "_FONTS", fonts)
    assert _key("line", "x", 1.0, b"css", _font_digest()) == base
    (fonts / "new-400.woff2").write_bytes(b"x")
    assert _key("line", "x", 1.0, b"css", _font_digest()) != base


def test_typewriter_timing_is_one_instant_char_per_stagger():
    line = TextLine("abcd", "mono", 0, 10, reveal="type")
    assert line.by == "char" and line.rise == 0
    fps = 30.0
    stagger = 0.05 * fps
    assert _unit_ks(line, 4, 10 - 0.01, fps) == [0.0] * 4
    assert _unit_ks(line, 4, 10, fps) == [1.0, 0.0, 0.0, 0.0]
    assert _unit_ks(line, 4, 10 + 2 * stagger, fps) == [1.0, 1.0, 1.0, 0.0]
    assert _unit_ks(line, 4, 10 + 3 * stagger, fps) == [1.0] * 4


def test_cursor_solid_while_typing_then_blinks():
    fps = 30.0
    for n, stagger in ((4, None), (2, 30.0)):  # the second case pins "blink starts at the last char, not one stagger later"
        line = TextLine("ab" * (n // 2), "mono", 0, 0, reveal="type", cursor=True, stagger=stagger)
        # the actual moment the last character is revealed, found by scanning the reveal, not by formula
        t = 0.0
        while not all(k == 1.0 for k in _unit_ks(line, n, t, fps)):
            t += 0.01
        last = t
        assert not text_mod._cursor_on(line, n, -1, fps)
        assert text_mod._cursor_on(line, n, last - 0.1, fps)
        assert text_mod._cursor_on(line, n, last + 0.53 * fps - 0.2, fps)  # still solid half a period after the last char
        assert not text_mod._cursor_on(line, n, last + 0.53 * fps + 0.2, fps)
        assert text_mod._cursor_on(line, n, last + 1.06 * fps + 0.2, fps)


def test_bad_values_are_one_line_errors(tmp_path):
    for field, hint in (("style: nope", "mono"), ("box: pill", "chip"), ("reveal: slide", "type"),
                        ("align: top", "left")):
        errs = _errors(tmp_path, LINE % field)
        key = field.split(":")[0]
        assert any(f".{key}" in e and hint in e for e in errs), errs
        assert all("\n" not in e for e in errs)


def test_field_dependencies_are_validated(tmp_path):
    assert any("cursor" in e and "reveal: type" in e for e in _errors(tmp_path, LINE % "cursor: true"))
    assert any("dot" in e and "box: chip" in e for e in _errors(tmp_path, LINE % "dot: gold"))
    assert any("weight" in e and "400, 600, 700" in e for e in _errors(tmp_path, LINE % "style: sans, weight: 300"))
    assert any(".dot" in e and "#rrggbb" in e for e in _errors(tmp_path, LINE % "box: chip, dot: blue"))


def test_new_fields_reach_the_timeline(tmp_path):
    p = _spec(tmp_path, LINE % "style: mono, reveal: type, cursor: true, box: chip, dot: '#4cd6a0'")
    tl = resolve(load_spec(p), p.parent)
    ln = tl.texts[0].lines[0]
    assert (ln.reveal, ln.by, ln.rise, ln.dur, ln.cursor, ln.box, ln.dot) == ("type", "char", 0.0, 0.0, True, "chip", "#4cd6a0")
    assert ln.stagger == pytest.approx(0.05 * 30)


def test_lower_third_expands_to_title_chip_and_subtitle(tmp_path):
    p = _spec(tmp_path, "text:\n  - from: 8s\n    to: 10s\n    lower_third: {title: Voice, subtitle: talks back, side: right}\n")
    spec = load_spec(p)
    lines = spec.text[0].lines
    assert [ln.text for ln in lines] == ["Voice", "talks back"]
    assert lines[0].box == "chip" and lines[0].align == "right" and lines[0].at == "8s"
    assert lines[1].y > lines[0].y
    assert resolve(spec, p.parent).texts[0].fade_in == pytest.approx(0.4 * 30)


def test_lower_third_errors(tmp_path):
    bad = "text:\n  - from: 0s\n    to: 1s\n    lower_third: {title: A, side: up}\n"
    assert any("side" in e for e in _errors(tmp_path, bad))
    both = "text:\n  - from: 0s\n    to: 1s\n    lower_third: {title: A}\n    lines: [{text: b, y: 1, at: 0s}]\n"
    assert any("lower_third" in e for e in _errors(tmp_path, both))


def test_raster_sans_mono_and_chip(tmp_path):
    for style in ("sans", "mono"):
        r = raster_line(TextLine("Theme synced", style, 200, 0, weight=700), scale=1, cache_dir=tmp_path / style)
        assert r.units[0].fill[3].max() > 0.2
    chip = raster_line(
        TextLine("Theme synced", "sans", 200, 0, weight=600, box="chip", dot="#4cd6a0", align="left"),
        scale=1, cache_dir=tmp_path / "chip",
    )
    assert chip.chip is not None and chip.chip[3].max() > 0.2
    # the chip background stays out of the glyph units
    plain = raster_line(TextLine("Theme synced", "sans", 200, 0, weight=600, align="left"), scale=1, cache_dir=tmp_path / "plain")
    assert chip.units[0].fill[3].sum() == pytest.approx(float(plain.units[0].fill[3].sum()), rel=0.02)
    x, _y, w, _h = chip.chip_box
    assert x < chip.units[0].layout[0] and x + w > chip.units[-1].layout[0]
    assert chip.units[0].layout[0] < 540  # left-aligned, not centred


def test_draw_typewriter_with_cursor_and_chip(tmp_path):
    from montaj.render.canvas import Canvas
    from montaj.render.text import draw_text

    line = TextLine("typed", "mono", 200, 0, reveal="type", cursor=True, box="chip", color="#ffffff")
    block = TextBlock(0, 100, 0, 0, 0, (line,))
    r = raster_line(line, scale=1, cache_dir=tmp_path)
    cv = Canvas(1080, 1920, device="cpu")
    img = torch.zeros(4, 1920, 1080)
    # t=0: only the first char (and chip) and a solid cursor
    a = draw_text(cv, img, [block], [], 0.0, [r], fps=30.0)
    b = draw_text(cv, img, [block], [], 5 * 0.05 * 30 + 1, [r], fps=30.0)
    assert 0 < a[3].sum() < b[3].sum()
    # after typing, one blink half-period later the cursor is off: fewer lit pixels
    on = draw_text(cv, img, [block], [], 5 * 0.05 * 30 + 1, [r], fps=30.0)[3].sum()
    off = draw_text(cv, img, [block], [], 5 * 0.05 * 30 + 0.53 * 30 + 1, [r], fps=30.0)[3].sum()
    assert off < on


def test_typewriter_rejects_dur_and_lower_third_rejects_nonfinite_y(tmp_path):
    assert any(".dur" in e and "instantly" in e for e in _errors(tmp_path, LINE % "reveal: type, dur: 2s"))
    assert any(".y" in e and "finite" in e for e in _errors(tmp_path, "text:\n  - from: 0s\n    to: 1s\n    lines:\n      - {text: a, y: .nan, at: 0s}\n"))
    bad = "text:\n  - from: 0s\n    to: 1s\n    lower_third: {title: A, y: .nan}\n"
    assert any("lower_third.y" in e and "finite" in e for e in _errors(tmp_path, bad))


# Segment hashes of a film using only pre-existing text features, computed from `main` before the
# sans/mono/typewriter/chip patch (spec: serif+italic, script+gold+char, caps+pop, one sub, one caption).
_LEGACY_TEXT = """text:
  - from: 0s
    to: 2s
    lines:
      - {text: Hello, y: 600, at: 0s, style: serif, italic: true}
      - {text: World, y: 800, at: 0.2s, style: script, gold: true, by: char}
  - from: 2s
    to: 4s
    lines:
      - {text: CAPS, y: 400, at: 2s, style: caps, reveal: pop}
subs:
  - {from: 1s, to: 3s, text: sub line}
"""
_LEGACY_HASHES = [
    "55fd4439767e06e5d47f97404cba5e148afad9efa18480713194d5c4329752a7",
    "e5e5f70659169fa0909d6779e8e4be1ee66fe48223898232ef168a990efa4289",
]


def _hashes(tmp: Path, text: str) -> list[str]:
    from montaj.project import segment_hash

    p = _write(tmp, "  - {photo: a, hold: 2s, frame: {caption: hi}}\n  - {photo: a, hold: 3s}\n", ["a"], extra=text)
    tl = resolve(load_spec(p), tmp)
    return [segment_hash(tl, seg, {"photo:a": "aa", "a": "aa"}, "test") for seg in tl.segments]


def test_legacy_text_subtitle_and_caption_segment_hashes_are_unchanged(tmp_path):
    assert _hashes(tmp_path, _LEGACY_TEXT) == _LEGACY_HASHES


def test_using_a_new_text_feature_changes_the_hash(tmp_path):
    mono = _LEGACY_TEXT.replace("y: 400, at: 2s, style: caps, reveal: pop", "y: 400, at: 2s, style: mono, reveal: type, cursor: true")
    got = _hashes(tmp_path, mono)
    assert got[0] != _LEGACY_HASHES[0] and got[1] != _LEGACY_HASHES[1]  # both segments overlap the typed block
    (tmp_path / "b").mkdir()
    only = _hashes(tmp_path / "b", _LEGACY_TEXT.replace("style: caps, reveal: pop", "style: caps, align: left"))
    assert only != _LEGACY_HASHES


def test_added_font_files_reach_only_the_hash_of_segments_using_that_style():
    from montaj.project import _text_asset_shas

    base = _text_asset_shas()
    assert not any(k.startswith(("inter-latin-6", "inter-latin-7", "jetbrains", "OFL")) or k == "text-ext.css" for k in base)
    sans = _text_asset_shas(frozenset({"sans"}))
    assert "inter-latin-700-normal.woff2" in sans and not any(k.startswith("jetbrains") for k in sans)
    assert "text-ext.css" in sans
