"""Text styling (issue #21): fonts, case, *emphasis*, stroke, shadow, glow, pos and rotate on the gpu engine."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
import torch

from montaj import cli
from montaj.project import _run_payload
from montaj.render import glyphs as G
from montaj.spec import SpecError, load_spec, text_warnings
from montaj.timeline import resolve

from tests.test_gpu_text import _cuda, _hashes, _no_browser, _text_spec

EXAMPLE = (
    'text: "I tried this for *30 days*", font: montserrat-black, size: 96, case: upper, color: "#ffffff", '
    'emphasis: {color: "#ffe600", scale: 1.18}, stroke: {width: 8, color: "#000000"}, '
    'shadow: {x: 0, y: 6, blur: 12, color: "#000000", opacity: 0.6}, glow: 0.4, pos: [540, 1240], rotate: -3, at: 0s'
)


def _line(tmp: Path, *lines: str):
    p = _text_spec(tmp, *lines)
    return resolve(load_spec(p), p.parent).texts[0].lines


def _errors(tmp: Path, *lines: str) -> list[str]:
    with pytest.raises(SpecError) as e:
        load_spec(_text_spec(tmp, *lines))
    return e.value.errors


def test_parse_emphasis_spans_and_escape() -> None:
    assert G.parse_emphasis("I tried *30 days*") == ("I tried 30 days", [0] * 8 + [1] * 7)
    text, flags = G.parse_emphasis(r"*a* b \* *c*")
    assert text == "a b * c" and flags == [1, 0, 0, 0, 0, 0, 2]
    with pytest.raises(ValueError, match="unclosed"):
        G.parse_emphasis("one *two")
    # Case follows code points: ß -> SS stays inside its span.
    assert G.apply_case("*straße*"[1:-1], "serif", "upper", [1] * 6) == ("STRASSE", [1] * 7)
    assert G.apply_case("Hi", "caps", "as-is") == "Hi" and G.apply_case("Hi", "serif", "lower") == "hi"


def test_layout_emphasis_scale_pos_rotate_and_font() -> None:
    book = G.FontBook()
    plain = G.layout_line(book, "I tried 30 days", style="sans", y=100, font="anton", size=96)
    run = G.layout_line(book, "I tried *30 days*", style="sans", y=100, font="anton", size=96, emphasis_scale=1.2)
    assert run.text == plain.text and run.spans == 1 and plain.spans == 0
    emph = [g for g in run.glyphs if g.emph]
    assert {g.scale for g in emph} == {1.2} and all(g.scale == 1.0 for g in run.glyphs if not g.emph)
    assert "".join(run.text[g.cluster] for g in emph) == "30 days"
    assert run.box[2] > plain.box[2] and run.baseline == plain.baseline  # bigger words, shared baseline
    assert run.fonts[0].name == "anton-latin-400-normal.woff2"
    # Font bytes are the cache key: the run names the file's sha.
    assert run.fonts[0].sha == hashlib.sha256((G.FONT_DIR / run.fonts[0].name).read_bytes()).hexdigest()

    placed = G.layout_line(book, "Hi", style="sans", pos=(300, 1240), rotate=-3)
    x, y, w, h = placed.box
    assert placed.pivot == pytest.approx((300, 1240), abs=1e-3) and placed.rotate == -3.0
    with pytest.raises(ValueError, match="unknown font 'comic'; available: montserrat-black"):
        G.layout_line(book, "Hi", style="sans", y=0, font="comic")


def test_every_bundled_font_shapes_and_has_a_licence() -> None:
    book = G.FontBook()
    for name, files in G.FONTS.items():
        run = G.layout_line(book, "Happy 30", style="sans", y=0, font=name)
        assert len(run.glyphs) == 8, name
        for f in files:
            assert (G.FONT_DIR / f).is_file()
    for lic in ("Montserrat", "Anton", "BebasNeue", "Poppins", "Bangers", "ArchivoBlack", "Mukta"):
        assert "SIL Open Font License" in (G.FONT_DIR / f"OFL-{lic}.txt").read_text()
    assert G.layout_line(book, "नमस्ते", style="sans", y=0, font="mukta-extrabold").glyphs


def test_plain_gpu_run_hashes_without_the_new_fields() -> None:
    run = G.layout_line(G.FontBook(), "Happy Birthday", style="caps", y=900)
    raw = _run_payload(run)
    assert "rotate" not in raw and all("emph" not in g and "scale" not in g for g in raw["glyphs"])
    styled = _run_payload(G.layout_line(G.FontBook(), "Happy *Birthday*", style="caps", y=900, rotate=2))
    assert styled["rotate"] == 2.0 and any(g.get("emph") == 1 for g in styled["glyphs"])


def test_styling_fields_select_the_gpu_engine(tmp_path: Path) -> None:
    legacy = _line(tmp_path, "text: Happy Birthday, style: caps, y: 900, at: 0s")[0]
    assert legacy.glyphs is None  # Chromium line, untouched
    for extra in ("font: anton", "case: lower", "stroke: {width: 4}", "glow: 0.3", "rotate: 2",
                  "shadow: {y: 4}", "emphasis: {scale: 1.1}"):
        text = "Happy *Birthday*" if "emphasis" in extra else "Happy Birthday"
        ln = _line(tmp_path, f"text: {text}, style: caps, y: 900, at: 0s, {extra}")[0]
        assert ln.glyphs is not None, extra
    marked = _line(tmp_path, 'text: "Happy *Birthday*", style: caps, y: 900, at: 0s')[0]
    assert marked.glyphs.spans == 1 and marked.emphasis is not None and marked.emphasis.color == "#ffe600"
    placed = _line(tmp_path, "text: Hi, pos: [540, 1000], at: 0s")[0]
    assert placed.glyphs is not None and placed.y == pytest.approx(placed.glyphs.box[1])
    assert placed.glyphs.pivot == pytest.approx((540, 1000), abs=1e-3)


def test_styling_reaches_the_segment_hash(tmp_path: Path) -> None:
    base = "text: Happy Birthday, style: caps, y: 900, at: 0s, engine: gpu"
    plain = _hashes(_text_spec(tmp_path, base))
    for extra in ("stroke: {width: 4}", "stroke: {width: 5}", "glow: 0.3", "shadow: {y: 4}", "rotate: 2", "font: anton"):
        assert _hashes(_text_spec(tmp_path, f"{base}, {extra}")) != plain, extra
    assert _hashes(_text_spec(tmp_path, base)) == plain


def test_validate_text_styling(tmp_path: Path) -> None:
    errs = _errors(tmp_path, "text: Hi, y: 900, at: 0s, font: comic")
    assert any(".font: unknown 'comic'" in e and "montserrat-black" in e and "mukta-extrabold" in e for e in errs)
    assert any(".stroke: stroke is drawn by the gpu engine" in e
               for e in _errors(tmp_path, "text: Hi, y: 900, at: 0s, engine: chromium, stroke: {width: 4}"))
    assert any("unclosed *" in e for e in _errors(tmp_path, 'text: "a *b", y: 900, at: 0s'))
    assert any(".emphasis: no *marked* words" in e for e in _errors(tmp_path, "text: Hi, y: 900, at: 0s, emphasis: {scale: 1.2}"))
    assert any(".y: missing" in e for e in _errors(tmp_path, "text: Hi, at: 0s"))
    assert any("pos replaces y" in e for e in _errors(tmp_path, "text: Hi, y: 9, pos: [1, 2], at: 0s"))
    assert any(".case: unknown 'title'" in e for e in _errors(tmp_path, "text: Hi, y: 9, at: 0s, case: title"))
    assert any(".glow: must be from 0 to 1" in e for e in _errors(tmp_path, "text: Hi, y: 9, at: 0s, glow: 2"))
    assert any(".stroke.width:" in e and "over 38%" in e
               for e in _errors(tmp_path, "text: Hi, y: 9, at: 0s, size: 40, stroke: {width: 20}"))
    assert any(".gold: gold is not drawn by the gpu engine yet" in e and "drop stroke" in e
               for e in _errors(tmp_path, "text: Hi, y: 9, at: 0s, gold: true, stroke: {width: 2}"))
    from tests.test_spec import _write

    card = _write(tmp_path, '  - card: {color: "#112233", text: [{text: hi, y: 100, at: 0s, stroke: {width: 3}}]}\n    hold: 2s\n', [])
    with pytest.raises(SpecError) as e:
        load_spec(card)
    assert any("card.text[0].stroke: card lines are drawn by Chromium" in x for x in e.value.errors)


def test_validate_warns_on_a_wide_stroke(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    p = _text_spec(tmp_path, "text: Hi, y: 900, at: 0s, size: 40, stroke: {width: 10}")
    assert text_warnings(load_spec(p)) == [
        "WARN text[0].lines[0].stroke.width: 10 px is over 20% of size 40; the outline will swallow the letters"]
    assert text_warnings(load_spec(_text_spec(tmp_path, "text: Hi, y: 900, at: 0s, size: 40, stroke: {width: 8}"))) == []
    p = _text_spec(tmp_path, "text: Hi, y: 900, at: 0s, size: 40, stroke: {width: 10}")
    assert cli.main(["-C", str(p.parent), "validate"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("WARN text[0].lines[0].stroke.width") and out[-1].startswith("OK montaj.yaml")


def test_overflow_is_measured_with_the_styled_line(tmp_path: Path) -> None:
    """validate's overflow ERR (#25) sees the font, emphasis scale, stroke, rotation and pos of a gpu line."""
    errs = _errors(tmp_path, EXAMPLE)  # the issue's example is wider than the frame at size 96
    assert any("wider than the frame at size 96" in e for e in errs), errs
    (fitted,) = _line(tmp_path, EXAMPLE + ", fit: true")
    assert fitted.size < 96 and fitted.stroke is not None and fitted.glyphs.rotate == -3
    book = G.FontBook()
    x, w, _ = G.measure(book, fitted.text, style="serif", size=fitted.size, font="montserrat-black", case="upper",
                        emphasis_scale=1.18, stroke=8, rotate=-3, pos=(540, 0))
    assert 72 - 1e-6 <= x and x + w <= 1008 + 1e-6
    base = "text: Happy Birthday Ana, size: 128, y: 900, at: 0s, font: anton"
    _line(tmp_path, base)  # fits
    for extra in ("pos: [200, 900]", "stroke: {width: 40}"):
        line = base.replace(", y: 900", "") if extra.startswith("pos") else base
        assert any("wider than the frame" in e for e in _errors(tmp_path, f"{line}, {extra}")), extra
    flat = G.measure(book, "Happy Birthday Ana", style="serif", size=120, font="anton")
    assert G.measure(book, "Happy Birthday Ana", style="serif", size=120)[1] != flat[1]  # the font is measured
    turned = G.measure(book, "Happy Birthday Ana", style="serif", size=120, font="anton", rotate=-3)
    assert turned[1] == pytest.approx(flat[1] * math.cos(math.radians(3)) + flat[2] * math.sin(math.radians(3)))


def test_wrap_keeps_styling_and_emphasis_on_every_piece(tmp_path: Path) -> None:
    text = 'text: "I tried this for *thirty whole days* straight", font: anton, size: 120, stroke: {width: 6}'
    pieces = _line(tmp_path, f"{text}, y: 700, at: 0s, wrap: balance")
    assert len(pieces) == 2
    assert all(p.stroke is not None and p.glyphs is not None and p.glyphs.fonts[0].name.startswith("anton")
               for p in pieces)
    assert all(p.glyphs.spans == 1 and p.emphasis is not None for p in pieces)  # the span was cut and closed
    assert "".join(p.glyphs.text for p in pieces).replace(" ", "") == "Itriedthisforthirtywholedaysstraight"
    # pos: the stack is centred on it; rotate: the stack turns as one block.
    placed = _line(tmp_path, f"{text}, pos: [540, 1000], rotate: 10, at: 0s, wrap: balance")
    (ax, ay), (bx, by) = (p.glyphs.pivot for p in placed)
    assert ((ax + bx) / 2, (ay + by) / 2) == pytest.approx((540, 1000), abs=0.5)
    assert math.degrees(math.atan2(-(bx - ax), by - ay)) == pytest.approx(10, abs=0.05)
    assert all(p.glyphs.rotate == 10 for p in placed)


# --- GPU --------------------------------------------------------------------------------------------


def _frame(tmp: Path, size: str, line: str) -> tuple[torch.Tensor, object]:
    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer

    tmp.mkdir(parents=True, exist_ok=True)
    p = _text_spec(tmp, line)
    p.write_text(p.read_text().replace("size: 1080x1920", f"size: {size}"))
    tl = resolve(load_spec(p), p.parent)
    cv = Canvas(*tl.size, device="cuda")
    with torch.no_grad():
        return Renderer(tl, p.parent, cv).frame(45), tl


def _masks(img: torch.Tensor, line_img: torch.Tensor):
    """Where the text changed the frame, and the white / black / yellow parts of it."""
    d = (img - line_img).abs().sum(0) > 0.15
    r, g, b = img[0], img[1], img[2]
    white = d & (r > 0.9) & (g > 0.9) & (b > 0.9)
    black = d & (r < 0.08) & (g < 0.08) & (b < 0.08)
    yellow = d & (r > 0.85) & (g > 0.75) & (b < 0.25)
    return d, white, black, yellow


@pytest.mark.gpu
def test_caps_stroke_and_emphasis_match_between_final_and_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Done-when of #21: a white caps line, black stroke, one yellow emphasis word, at 1080×1920 and the
    540×960 preview, framed the same."""
    _cuda()
    started = _no_browser(monkeypatch)
    line = ('text: "Happy *Birthday* Ana", case: upper, font: anton, size: 110, color: "#ffffff", '
            'stroke: {width: 8}, emphasis: {color: "#ffe600"}, shadow: false, y: 860, at: 0s')
    blank = "text: x, y: 3000, at: 0s, engine: gpu"  # same film, the text drawn off-frame
    boxes = {}
    for size in ("1080x1920", "540x960"):
        img, tl = _frame(tmp_path / size, size, line)
        ref, _ = _frame(tmp_path / (size + "-ref"), size, blank)
        d, white, black, yellow = _masks(img, ref)
        assert white.sum() > 200 and black.sum() > 200 and yellow.sum() > 200, size
        k = img.shape[-1] / 1080
        run = tl.texts[0].lines[0].glyphs
        emph_x0 = min(g.ink[0] for g in run.glyphs if g.emph) * k
        emph_x1 = max(g.ink[2] for g in run.glyphs if g.emph) * k
        yx = torch.nonzero(yellow, as_tuple=True)[1].float()
        assert float(yx.min()) >= emph_x0 - 2 and float(yx.max()) <= emph_x1 + 2, size  # only the marked word
        wx = torch.nonzero(white, as_tuple=True)[1].float()
        assert float(((wx >= emph_x0) & (wx <= emph_x1)).float().mean()) < 0.02, size
        ys, xs = torch.nonzero(d, as_tuple=True)
        boxes[size] = [float(v) / k for v in (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)]
        # The stroke rings the fill: the drawn ink reaches past the outline boxes by about the width.
        x0 = min(g.ink[0] for g in run.glyphs if g.ink[2] > g.ink[0])
        assert boxes[size][0] < x0 - 4, size
    full, half = boxes["1080x1920"], boxes["540x960"]
    assert all(abs(a - b) <= 3.0 for a, b in zip(full, half)), (full, half)  # design px
    assert not started


@pytest.mark.gpu
def test_line_rotation_and_shadow_glow_paint(tmp_path: Path) -> None:
    _cuda()
    from montaj.render.canvas import Canvas
    from montaj.render.type import GlyphAtlas, TextStyle, draw_run, Shadow

    cv = Canvas(1080, 1920, device="cuda")
    book = G.FontBook()
    atlas = GlyphAtlas(tmp_path, "cuda", book)
    flat = G.layout_line(book, "WIDE LINE", style="sans", font="anton", size=120, pos=(540, 960))
    turned = G.layout_line(book, "WIDE LINE", style="sans", font="anton", size=120, pos=(540, 960), rotate=-10)

    def box(img):
        ys, xs = torch.nonzero(img[3] > 0.5, as_tuple=True)
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), float(img[3].sum())

    a, b = box(draw_run(cv, flat, TextStyle(), atlas=atlas)), box(draw_run(cv, turned, TextStyle(), atlas=atlas))
    assert b[3] - b[1] > (a[3] - a[1]) + 0.1 * (a[2] - a[0])  # turned: taller by about w·sin 10°
    assert b[4] == pytest.approx(a[4], rel=0.01)  # same ink
    left = torch.nonzero(draw_run(cv, turned, TextStyle(), atlas=atlas)[3, :, : int(a[0]) + 40] > 0.5, as_tuple=True)[0]
    assert float(left.float().mean()) > 960  # counter-clockwise: the left end sits lower
    glow = draw_run(cv, flat, TextStyle(glow=(Shadow(0, 0, 30, (1, 1, 1), 0.4),)), atlas=atlas)
    plain = draw_run(cv, flat, TextStyle(), atlas=atlas)
    assert float(glow[3].sum()) > float(plain[3].sum()) * 1.1  # a halo outside the ink
