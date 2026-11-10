"""Text fit and wrap (issue #25): shaped measurement, `fit`, `wrap: balance`, and the validate ERR."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from montaj.project import segment_hash
from montaj.render import glyphs as G
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

from tests.test_gpu_text import _Shas, _text_spec

LONG = "text: Happy Birthday to the most wonderful person I know, style: script, size: 100, y: 800, at: 0s"


def _lines(tmp: Path, *lines: str):
    p = _text_spec(tmp, *lines)
    return resolve(load_spec(p), p.parent).texts[0].lines


def test_overlong_line_is_an_err_with_the_fitting_size(tmp_path: Path) -> None:
    with pytest.raises(SpecError) as e:
        _lines(tmp_path, LONG)
    msg = str(e.value)
    assert "text[0].lines[0]: 627px wider than the frame at size 100 (fits at size 54" in msg
    size = float(msg.split("fits at size ")[1].split(";")[0])
    book = G.FontBook()
    assert G.measure(book, "Happy Birthday to the most wonderful person I know", style="script", size=size)[1] <= 936
    assert G.measure(book, "Happy Birthday to the most wonderful person I know", style="script", size=size + 1)[1] > 936


def test_fit_materialises_a_size_that_fits_the_margins(tmp_path: Path) -> None:
    (line,) = _lines(tmp_path, LONG + ", fit: true")
    assert line.size == 54.0
    x, w, _ = G.measure(G.FontBook(), line.text, style="script", size=line.size)
    assert 72 <= x and x + w <= 1008


def test_short_line_keeps_its_size_and_default_hash(tmp_path: Path) -> None:
    short = "text: Joy, style: script, size: 100, y: 800, at: 0s"
    (a,) = _lines(tmp_path, short)
    (b,) = _lines(tmp_path, short + ", fit: true")
    assert (a.size, a.text, a.y) == (b.size, b.text, b.y)
    p1, p2 = _text_spec(tmp_path, short), _text_spec(tmp_path, short + ", fit: true")
    h = lambda p: [segment_hash(t, s, _Shas(), "e1") for t in [resolve(load_spec(p), p.parent)] for s in t.segments]
    assert h(p1) == h(p2)


def test_fit_and_wrap_reach_the_hash(tmp_path: Path) -> None:
    def hashes(extra: str):
        p = _text_spec(tmp_path, LONG + extra)
        tl = resolve(load_spec(p), p.parent)
        return [segment_hash(tl, s, _Shas(), "e1") for s in tl.segments]

    assert hashes(", fit: true") != hashes(", wrap: balance")
    assert hashes(", fit: true") != hashes(", fit: true, size: 80")


def test_wrap_balance_breaks_without_orphans_and_stacks_lines(tmp_path: Path) -> None:
    lines = _lines(tmp_path, LONG + ", wrap: balance")
    assert [l.text for l in lines] == ["Happy Birthday to the", "most wonderful person I know"]
    assert lines[1].y == pytest.approx(800 + G.measure(G.FontBook(), "x", style="script", size=100)[2], abs=0.01)
    assert lines[1].at > lines[0].at  # the reveal continues down the stack


@pytest.mark.parametrize("text", [
    "I love you so very much Anuraj today",
    "Congratulations on turning forty two years young",
    "Wishing you a day full of joy laughter and cake",
])
def test_balance_never_orphans_a_short_word(text: str) -> None:
    book = G.FontBook()
    plan = G.plan_text(book, text, style="serif", size=84, balance=True)
    assert len(plan.lines) == 2 and plan.fits
    w = [G.measure(book, ln, style="serif", size=84)[1] for ln in plan.lines]
    assert not (len(plan.lines[-1].split()) == 1 and w[-1] < 0.4 * max(w))
    assert max(w) / min(w) < 1.6


def test_wrap_then_fit_shrinks_when_two_lines_are_not_enough(tmp_path: Path) -> None:
    long = "text: " + " ".join(["wonderful"] * 8) + ", style: serif, size: 120, y: 700, at: 0s"
    lines = _lines(tmp_path, long + ", wrap: balance, fit: true, max_lines: 3")
    assert len(lines) == 3 and lines[0].size < 120
    with pytest.raises(SpecError, match="wider than the frame"):
        _lines(tmp_path, long + ", wrap: balance")


def test_max_width_and_field_errors(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="wider than max_width 300"):
        _lines(tmp_path, "text: Happy Birthday, style: serif, y: 700, at: 0s, max_width: 300")
    for bad, frag in ((", wrap: knuth", "unknown 'knuth'"), (", max_lines: 2", "needs wrap"), (", max_width: -1", "> 0")):
        with pytest.raises(SpecError, match=frag):
            _lines(tmp_path, "text: Hi, style: serif, y: 700, at: 0s" + bad)


def test_unmeasurable_glyphs_are_skipped_unless_fit_asks(tmp_path: Path) -> None:
    _lines(tmp_path, "text: Hi 😀, style: serif, y: 700, at: 0s")  # Chromium falls back; no verdict
    with pytest.raises(SpecError, match="fit: true|no glyph"):
        _lines(tmp_path, "text: Hi 😀, style: serif, y: 700, at: 0s, fit: true")


def test_chromium_style_widths_equal_shaped_widths() -> None:
    # Chromium's layout boxes for these lines were measured equal to the pixel (see glyphs.py note).
    x, w, _ = G.measure(G.FontBook(), "Happy Birthday Anu, my love", style="script", size=100)
    assert (round(x, 1), round(w, 1)) == (23.0, 1034.0)


@pytest.mark.gpu
def test_fitted_gpu_line_inks_inside_the_margins(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from montaj.render.canvas import Canvas
    from montaj.render.type import GlyphAtlas, TextStyle, draw_run

    book = G.FontBook()
    cv = Canvas(1080, 1920, device="cuda")
    atlas = GlyphAtlas(tmp_path, "cuda", book)

    def inked(run) -> tuple[float, float]:
        a = draw_run(cv, run, TextStyle(fill=(1.0, 1.0, 1.0)), atlas=atlas)[3]
        xs = torch.nonzero((a > 0.5).any(dim=0), as_tuple=True)[0].float() / cv.k
        return float(xs.min()), float(xs.max()) + 1

    text = "Happy Birthday to the most wonderful person I know"
    (fitted,) = _lines(tmp_path, LONG + ", fit: true, engine: gpu")
    lo, hi = inked(fitted.glyphs)
    assert lo >= 72 - 1 and hi <= 1008 + 1
    raw = G.layout_line(book, text, style="script", y=800, size=100)
    assert inked(raw)[1] > 1080 - 72  # without fit it spills past the margin
    for ln in _lines(tmp_path, LONG + ", wrap: balance, engine: gpu"):
        lo, hi = inked(ln.glyphs)
        assert lo >= 72 and hi <= 1008
