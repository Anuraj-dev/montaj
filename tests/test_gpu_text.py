"""GPU text engine (issue #20): HarfBuzz layout, distance-field cells, draw_run, `engine: gpu` lines."""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
import torch
from PIL import Image

from montaj import doctor
from montaj.project import segment_hash
from montaj.render import glyphs as G
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

from tests.test_spec import _write

ROOT = Path(__file__).resolve().parents[1]
SHORT_STEMS = ["09", "10", "12", "16", "19", "20", "21", "22", "28", "29", "32", "33", "34", "35", "37", "38"]


class _Shas(dict):
    def __missing__(self, key):
        return "sha-" + key


def _jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path, "JPEG")


def _digest(proj: Path) -> tuple[int, str]:
    tl = resolve(load_spec(proj / "montaj.yaml"), proj)
    hashes = [segment_hash(tl, seg, _Shas(), "e1") for seg in tl.segments]
    return len(hashes), hashlib.sha256("".join(hashes).encode()).hexdigest()


# Captured on the tree before the gpu engine existed (placeholder photos, fake shas, engine "e1").
_RECIPE_DIGESTS = {
    "birthday-short.yaml": (13, "24265e9d45fcae5f6ee51b281e6d2f19c8b2e341c3f2c53968e184ed40c70986"),
    "birthday-song-film.yaml": (56, "1e656b235585ac601b14aafb9ff8e7aa85878399bdc8e61a4ef5e023c61b0ca6"),
}


@pytest.mark.parametrize("recipe", sorted(_RECIPE_DIGESTS))
def test_existing_recipes_keep_segment_hashes(tmp_path: Path, recipe: str) -> None:
    src = (ROOT / "recipes" / recipe).read_text()
    stems = SHORT_STEMS if recipe == "birthday-short.yaml" else set(re.findall(r"photo: ([\w-]+)", src))
    for stem in stems:
        _jpeg(tmp_path / "assets" / f"{stem}.jpg")
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "song.wav").write_bytes(b"RIFF")
    copyfile(ROOT / "recipes" / recipe, tmp_path / "montaj.yaml")
    assert _digest(tmp_path) == _RECIPE_DIGESTS[recipe]


_TEXT = "text:\n  - from: 0s\n    to: 3s\n    lines:\n{lines}"


def _text_spec(tmp: Path, *lines: str) -> Path:
    body = "".join(f"      - {{{ln}}}\n" for ln in lines)
    return _write(tmp, "  - {photo: a, hold: 3s}\n", ["a"], extra=_TEXT.format(lines=body))


def _hashes(p: Path) -> list[str]:
    tl = resolve(load_spec(p), p.parent)
    return [segment_hash(tl, seg, _Shas(), "e1") for seg in tl.segments]


def test_gpu_line_materialises_layout_into_the_hash(tmp_path: Path) -> None:
    base = "text: Happy Birthday, style: caps, y: 900, at: 0s"
    plain = _hashes(_text_spec(tmp_path, base))
    chromium = _hashes(_text_spec(tmp_path, base + ", engine: chromium"))
    gpu = _hashes(_text_spec(tmp_path, base + ", engine: gpu"))
    assert chromium != plain  # explicit field reaches the hash; only unset is omitted
    assert gpu != plain and gpu != chromium
    p = _text_spec(tmp_path, base + ", engine: gpu")
    tl = resolve(load_spec(p), p.parent)
    run = tl.texts[0].lines[0].glyphs
    assert run is not None and run.text == "HAPPY BIRTHDAY" and run.units == 2
    assert tl.texts[0].lines[0].glyphs.fonts[0].name.startswith("cormorant-garamond-latin-600")
    assert run == resolve(load_spec(p), p.parent).texts[0].lines[0].glyphs  # deterministic layout
    tracked = _hashes(_text_spec(tmp_path, base + ", engine: gpu, tracking: 0.2"))
    assert tracked != gpu


def test_glyph_runs_payload_follows_the_font_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _text_spec(tmp_path, "text: Hi, style: sans, y: 900, at: 0s, engine: gpu")
    before = _hashes(p)
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    for f in G.FONT_DIR.glob("*.woff2"):
        (fonts / f.name).write_bytes(f.read_bytes())
    # Same glyphs, different file: the run's font sha (and so the hash) must change.
    src = fonts / "inter-latin-400-normal.woff2"
    from fontTools.ttLib import TTFont

    tt = TTFont(src)
    tt["head"].fontRevision += 1
    tt.save(src)
    monkeypatch.setattr(G, "FONT_DIR", fonts)
    monkeypatch.setattr(G.FontBook.__init__, "__defaults__", (fonts,))
    assert _hashes(p) != before


def test_devanagari_shapes_into_conjuncts() -> None:
    book = G.FontBook()
    text = "क्षत्रिय जन्मदिन"
    run = G.layout_line(book, text, style="deva", y=100)
    assert len(run.glyphs) < len(text)  # conjuncts and matras merge clusters
    assert [g.cluster for g in run.glyphs] == sorted(g.cluster for g in run.glyphs)
    assert {g.unit for g in run.glyphs} == {0, 1} and run.units == 2
    assert run.fonts[0].name.startswith("tiro-devanagari-hindi-devanagari")
    # The pre-base matra (ि) is reordered before its consonant but keeps the cluster.
    xs = [g.x for g in run.glyphs]
    assert xs == sorted(xs)


def test_layout_alignment_tracking_and_box() -> None:
    book = G.FontBook()
    c = G.layout_line(book, "Launch", style="sans", y=200)
    x, y, w, h = c.box
    assert x == pytest.approx((1080 - w) / 2, abs=1e-3) and y == 200
    left = G.layout_line(book, "Launch", style="sans", y=200, align="left", indent=0.5)
    assert left.box[0] == pytest.approx(72 + 0.5 * 48, abs=1e-3)
    right = G.layout_line(book, "Launch", style="sans", y=200, align="right")
    assert right.box[0] + right.box[2] == pytest.approx(1080 - 72, abs=1e-3)
    tracked = G.layout_line(book, "Launch", style="sans", y=200, tracking=0.1)
    assert tracked.box[2] == pytest.approx(w + 6 * 0.1 * 48, abs=1e-3)  # after every cluster, the last too
    deva = G.layout_line(book, "नमस्ते", style="deva", y=0)
    assert deva.box[3] == pytest.approx(1.5 * 62)  # text.css line-height
    assert 0 < deva.baseline < deva.box[3]
    caps = G.layout_line(book, "hello world", style="caps", y=0, by="char")
    assert caps.text == "HELLO WORLD" and caps.units == 11
    with pytest.raises(ValueError, match="no glyph"):
        G.layout_line(book, "party \U0001f389", style="sans", y=0)


def test_gpu_line_validation(tmp_path: Path) -> None:
    def errors(*lines: str) -> list[str]:
        with pytest.raises(SpecError) as ei:
            load_spec(_text_spec(tmp_path, *lines))
        return ei.value.errors

    base = "text: hi, y: 100, at: 0s, engine: gpu"
    assert any("engine: unknown 'webgl'" in e for e in errors("text: hi, y: 100, at: 0s, engine: webgl"))
    assert any(".gold:" in e and "gpu engine" in e for e in errors(base + ", gold: true"))
    assert any(".box:" in e for e in errors(base + ", style: sans, box: chip"))
    assert any(".cursor:" in e for e in errors(base + ", style: mono, reveal: type, cursor: true"))
    assert any("no glyph for" in e and "\U0001f389" in e for e in errors("text: \"yay \U0001f389\", y: 100, at: 0s, engine: gpu"))
    load_spec(_text_spec(tmp_path, base + ", style: deva, text: नमस्ते"))  # ok
    card = _write(tmp_path, '  - card: {color: "#112233", text: [{text: hi, y: 100, at: 0s, engine: gpu}]}\n    hold: 2s\n', [])
    with pytest.raises(SpecError) as ei:
        load_spec(card)
    assert any("card lines are drawn by Chromium" in e for e in ei.value.errors)


def test_doctor_reports_text_engine_per_style() -> None:
    line = next(ln for ln in doctor.doctor() if "text-engine" in ln)
    assert line.startswith("OK text-engine: ")
    for style in ("serif", "script", "caps", "deva", "sans", "mono"):
        assert f"{style} chromium" in line
    assert "gpu opt-in" in line and "harfbuzz" in line


def test_cell_distances_are_exact_and_signed() -> None:
    upem = G.SDF_EM  # 1 font unit = 1 cell px
    outer = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], float)  # counter-clockwise (y up)
    hole = outer[::-1] * 0.5 + 10  # clockwise: a hole under nonzero winding
    cell = G.build_cell([outer, hole], upem)
    # Cell px centre (i + .5) maps to point ox + i + .5; font y is negated (y down).
    def d(x: float, y: float) -> float:
        return float(cell.sdf[int(-y - cell.oy - 0.5), int(x - cell.ox - 0.5)])

    assert d(3.5, 20.5) == pytest.approx(-3.5, abs=1e-4)  # in the ring: 3.5 from the outer edge, 6.5 from the hole
    assert d(20.5, 20.5) == pytest.approx(9.5, abs=1e-4)  # in the hole: outside the shape
    assert d(-6.5, 20.5) == pytest.approx(6.5, abs=1e-4)
    assert cell.ox == -(G.SPREAD + 1) and float(cell.sdf.max()) == G.SPREAD  # clamped far field


def test_cell_disk_cache_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    book = G.FontBook()
    run = G.layout_line(book, "A ", style="sans", y=0)
    ref = run.fonts[0]
    a, space = run.glyphs
    cell = G.glyph_cell(book, tmp_path, ref, a.gid)
    assert cell is not None and cell.sdf.dtype == np.float32
    assert G.glyph_cell(book, tmp_path, ref, space.gid) is None  # empty glyph, cached as empty
    assert (tmp_path / ref.atlas / f"{a.gid}.npz").is_file()

    def boom(*_a, **_k):
        raise AssertionError("cell rebuilt despite a disk hit")

    monkeypatch.setattr(G, "build_cell", boom)
    again = G.glyph_cell(G.FontBook(), tmp_path, ref, a.gid)
    assert np.array_equal(again.sdf, cell.sdf) and (again.ox, again.oy) == (cell.ox, cell.oy)
    assert G.glyph_cell(G.FontBook(), tmp_path, ref, space.gid) is None


# --- GPU ----------------------------------------------------------------------------------------


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _outline_area(face: G.Face, gid: int) -> float:
    """Signed shoelace over the flattened contours: holes (opposite winding) subtract. Font units²."""
    total = 0.0
    for c in face.contours(gid):
        x, y = c[:, 0], c[:, 1]
        total += 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return abs(total)


def _outline_perimeter(face: G.Face, gid: int) -> float:
    """Total length of the flattened contours, font units."""
    return sum(float(np.sum(np.hypot(*(np.roll(c, -1, 0) - c).T))) for c in face.contours(gid))


@pytest.mark.gpu
@pytest.mark.parametrize("size", [24.0, 84.0, 260.0])
def test_glyph_coverage_matches_outline_area(tmp_path: Path, size: float) -> None:
    _cuda()
    from montaj.render.canvas import Canvas
    from montaj.render.type import GlyphAtlas, TextStyle, draw_run

    cv = Canvas(1080, 1080, device="cuda")
    book = G.FontBook()
    atlas = GlyphAtlas(tmp_path, "cuda", book)
    # Light-serif hairlines (O) and serif corners (H) are the hard cases for a distance field.
    for style, ch in (("serif", "O"), ("serif", "H"), ("sans", "H"), ("deva", "क")):
        run = G.layout_line(book, ch, style=style, y=300, size=size)
        g = run.glyphs[0]
        face = book.face(run.fonts[g.font].name)
        want = _outline_area(face, g.gid) * (size / face.upem) ** 2  # design px² = device px² at k=1
        img = draw_run(cv, run, TextStyle(), atlas=atlas)
        a = img[3]
        got = float(a.sum())
        # Area error over perimeter = how far the drawn edge sits from the true outline on average: one
        # sub-pixel bound for every face, where a relative-area bound is loose for bold faces and harsh
        # on hairlines.
        drift = (got - want) / (_outline_perimeter(face, g.gid) * size / face.upem)
        assert abs(drift) <= 0.07, (style, ch, size, got, want, drift)
        ys, xs = torch.nonzero(a > 0.02, as_tuple=True)  # any visible coverage: thin serif feet stay under 0.5
        x0, y0, x1, y1 = g.ink
        assert abs(float(xs.min()) - x0) <= 1.0 and abs(float(xs.max()) + 1 - x1) <= 1.0, (style, size)
        assert abs(float(ys.min()) - y0) <= 1.0 and abs(float(ys.max()) + 1 - y1) <= 1.0, (style, size)
        assert float(a.max()) == pytest.approx(1.0)


def _no_browser(monkeypatch: pytest.MonkeyPatch) -> list:
    from montaj.render import text as text_mod

    started: list = []

    def refuse(*a, **k):
        started.append(a)
        raise AssertionError("Chromium started for a gpu-engine line")

    real_popen = subprocess.Popen

    def popen(args, *a, **k):
        head = str(args[0] if isinstance(args, (list, tuple)) else args)
        if "chrom" in head.lower():
            refuse(args)
        return real_popen(args, *a, **k)

    monkeypatch.setattr(text_mod._Browser, "__init__", refuse)
    monkeypatch.setattr(subprocess, "Popen", popen)
    return started


@pytest.mark.gpu
def test_caps_and_devanagari_fill_stroke_shadow_without_chromium(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cuda()
    started = _no_browser(monkeypatch)
    from montaj.render.canvas import Canvas
    from montaj.render.type import GlyphAtlas, Shadow, TextStyle, draw_run

    cv = Canvas(1080, 1920, device="cuda")
    book = G.FontBook()
    atlas = GlyphAtlas(tmp_path, "cuda", book)
    red, green, blue = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    style = TextStyle(fill=red, stroke_width=3.0, stroke_color=green, shadows=(Shadow(0, 24, 8, blue, 1.0),))
    for st, text in (("caps", "Happy Birthday"), ("deva", "जन्मदिन मुबारक")):
        run = G.layout_line(book, text, style=st, y=800, size=72 if st == "caps" else None)
        img = draw_run(cv, run, style, atlas=atlas)
        a = img[3]
        rgb = img[:3] / a.clamp_min(1e-6)
        solid = a > 0.99
        fill = solid & (rgb[0] > 0.99)
        stroke = solid & (rgb[1] > 0.99)
        shadow = (a > 0.2) & (rgb[2] > 0.99)
        assert fill.sum() > 500 and stroke.sum() > 500 and shadow.sum() > 500, st
        # Stroke rings the fill; the shadow sits below the ink.
        ys_fill = torch.nonzero(fill, as_tuple=True)[0].float().mean()
        ys_shadow = torch.nonzero(shadow, as_tuple=True)[0].float().mean()
        assert ys_shadow > ys_fill + 10, st
        x0 = min(g.ink[0] for g in run.glyphs if g.ink[2] > g.ink[0])
        xs_stroke = torch.nonzero(stroke, as_tuple=True)[1].float()
        assert float(xs_stroke.min()) < x0 - 1.0, st
    assert not started


@pytest.mark.gpu
def test_renderer_draws_gpu_lines_without_chromium(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cuda()
    started = _no_browser(monkeypatch)
    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer

    lines = (
        "text: Happy Birthday, style: caps, y: 700, at: 0s, engine: gpu",
        "text: जन्मदिन मुबारक, style: deva, y: 800, at: 0s, engine: gpu, reveal: pop",
    )
    p = _text_spec(tmp_path, *lines)
    tl = resolve(load_spec(p), p.parent)
    cv = Canvas(*tl.size, device="cuda")
    r = Renderer(tl, p.parent, cv)
    with torch.no_grad():
        early, done = r.frame(1), r.frame(60)
    band = slice(int(690 * cv.k), int(920 * cv.k))
    assert float((done[:, band] - early[:, band]).abs().mean()) > 1e-3  # text revealed over time
    assert (tmp_path / "build" / "glyphs").is_dir() and any((tmp_path / "build" / "glyphs").rglob("*.npz"))
    assert not started
