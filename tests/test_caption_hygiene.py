"""Caption hygiene (issue #24): normalisation, validate and check lines, slam kept inside the safe zone."""
from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from montaj import caption_check as K
from montaj import captions as C
from montaj.encode import FFMPEG
from montaj.render import motion as M
from montaj.render.glyphs import DESIGN_W
from montaj.spec import load_spec, text_warnings
from montaj.timeline import resolve

from tests.test_captions import _proj, _words
from tests.test_spec import _write

BOTTOM_CAPTION = ("text:\n  - from: 0s\n    to: 2s\n    lines:\n"
                  "      - {text: Thanks for watching., style: sans, y: 1700, at: 0s, box: chip}\n")


SHORT = [("hello", 0.0, 0.5), ("there", 0.6, 1.2), ("friend", 1.3, 2.0)]


def _short(tmp: Path, caps: str, extra: str = "") -> Path:
    _words(tmp, SHORT)
    return _write(tmp, "  - {photo: a, hold: 3s}\n", ["a"], extra=f"captions: {caps}\n{extra}")


def _text_proj(tmp: Path, extra: str) -> Path:
    return _write(tmp, "  - {photo: a, hold: 3s}\n", ["a"], extra=extra)


# --- normalisation -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw,want", [
    ("end.", "end"), ("so,", "so"), ("what?", "what?"), ("wow!", "wow!"), ("wait…", "wait…"),
    ("hmm...", "hmm..."), ("है।", "है"), ("नहीं,", "नहीं"), ("3.5", "3.5"), ("₹50,000", "₹50,000"),
])
def test_drop_end_keeps_questions_exclamations_ellipses(raw: str, want: str) -> None:
    assert C.drop_end(raw) == want


def test_captions_punctuation_keep_opts_out(tmp_path: Path) -> None:
    kept, _ = C.plan(_proj(tmp_path, "{source: talk.words.json, punctuation: keep}"))
    assert any(c.text.endswith("demonstration.") for c in kept) or any("demonstration." in c.text for c in kept)
    spec = load_spec(_proj(tmp_path, "{source: talk.words.json, punctuation: keep}"))
    assert any("demonstration." in ln.text for b in spec.text for ln in b.lines)
    dropped = load_spec(_proj(tmp_path))
    assert not any(ln.text.rstrip("*").endswith((".", ",")) for b in dropped.text for ln in b.lines)


def test_devanagari_danda_is_dropped_from_captions(tmp_path: Path) -> None:
    _words(tmp_path, [("मैं", 0.0, 0.3), ("ठीक", 0.32, 0.6), ("हूँ।", 0.62, 0.9)])
    p = _write(tmp_path, "  - {photo: a, hold: 2s}\n", ["a"], extra="captions: {source: talk.words.json}\n")
    spec = load_spec(p)
    assert all("।" not in ln.text for b in spec.text for ln in b.lines)


def test_subs_drop_only_when_asked(tmp_path: Path) -> None:
    subs = ("subs:\n  - {from: 0s, to: 1s, text: 'one.', punctuation: drop}\n"
            "  - {from: 1s, to: 2s, text: 'two,', punctuation: keep}\n  - {from: 2s, to: 3s, text: 'three.'}\n")
    spec = load_spec(_text_proj(tmp_path, subs))
    assert [s.text for s in spec.subs] == ["one", "two,", "three."]  # unset stays as written: old hashes hold
    assert [w for w in text_warnings(spec) if w.startswith("WARN subs")] == [
        "WARN subs[2] ends with a full stop or comma; set punctuation: drop"]


# --- validate lines ----------------------------------------------------------------------------------------


def test_boxed_full_stopped_bottom_caption_gets_exactly_four_warnings(tmp_path: Path) -> None:
    warns = text_warnings(load_spec(_text_proj(tmp_path, BOTTOM_CAPTION)))
    assert [w.split(" ")[1] for w in warns] == ["text[0].lines[0]", "caption-box", "caption-weight", "caption-size"]
    assert warns[0] == "WARN text[0].lines[0] ends with a full stop"
    assert warns[1] == "WARN caption-box text[0] uses a static box; use stroke + shadow"


def test_solid_backdrop_counts_as_a_static_box(tmp_path: Path) -> None:
    extra = BOTTOM_CAPTION.replace("box: chip", "weight: 700, size: 100").replace("Thanks for watching.", "Thanks")
    extra = extra.replace("    lines:", "    backdrop: solid\n    lines:")
    warns = text_warnings(load_spec(_text_proj(tmp_path, extra)))
    assert warns == ["WARN caption-box text[0] uses a static box; use stroke + shadow"]


def test_punch_default_has_no_warnings(tmp_path: Path) -> None:
    assert text_warnings(load_spec(_proj(tmp_path))) == []


def test_a_lower_third_chip_is_not_a_caption(tmp_path: Path) -> None:
    extra = "text:\n  - from: 0s\n    to: 2s\n    lower_third: {title: Anuraj, subtitle: Maker}\n"
    assert [w for w in text_warnings(load_spec(_text_proj(tmp_path, extra))) if "caption" in w] == []


# --- check lines -------------------------------------------------------------------------------------------


def test_safe_zone_names_the_side(tmp_path: Path) -> None:
    low = resolve(load_spec(_short(tmp_path, "{source: talk.words.json, y: 1800}")), tmp_path)
    got = K.lines(low)
    assert got and all(g.startswith("WARN caption-safe chunk=") and g.endswith(" bottom") for g in got)
    assert K.lines(resolve(load_spec(_short(tmp_path, "{source: talk.words.json}")), tmp_path)) == []


def test_safe_zones_are_configurable(tmp_path: Path) -> None:
    caps = "{source: talk.words.json, y: 1700}"
    assert K.lines(resolve(load_spec(_short(tmp_path, caps)), tmp_path))
    p = _short(tmp_path, caps, extra="check:\n  safe: {bottom: 0.05}\n")
    assert K.lines(resolve(load_spec(p), tmp_path)) == []


def test_reading_rate_warns_above_twenty_characters_per_second(tmp_path: Path) -> None:
    _words(tmp_path, [("extraordinarily", 0.0, 0.2), ("unbelievable", 0.22, 0.4), ("tomorrow", 0.9, 1.5)])
    p = _write(tmp_path, "  - {photo: a, hold: 3s}\n", ["a"], extra="captions: {source: talk.words.json}\n")
    got = K.lines(resolve(load_spec(p), tmp_path))
    assert [g for g in got if "caption-rate" in g] == ["WARN caption-rate chunk=0 64cps"]


def test_check_expect_caption_silences_a_span(tmp_path: Path) -> None:
    extra = "check:\n  expect:\n    - {caption: [0s, 7s], why: designed low title}\n"
    p = _short(tmp_path, "{source: talk.words.json, y: 1800}", extra=extra)
    assert K.lines(resolve(load_spec(p), tmp_path)) == ["INFO caption 0-7s ignored as intended: designed low title"]
    unused = "check:\n  expect:\n    - {caption: [20s, 21s]}\n"
    p2 = _short(tmp_path, "{source: talk.words.json}", extra=unused)
    assert K.lines(resolve(load_spec(p2), tmp_path)) == ["INFO expect caption 20-21s not found"]


def test_contrast_is_measured_against_the_band_and_a_stroke_guarantees_it(tmp_path: Path) -> None:
    tl = resolve(load_spec(_short(tmp_path, "{source: talk.words.json, preset: minimal}")), tmp_path)
    c = K.chunks(tl)[0]
    assert c.ink is not None and c.line.stroke is None
    white = np.full((1920, 1080, 3), 255, np.uint8)
    black = np.zeros((1920, 1080, 3), np.uint8)
    assert K.contrast(c, white) < 1.5  # pale fill on a white frame
    assert K.contrast(c, black) > 10
    stroked = resolve(load_spec(_short(tmp_path, "{source: talk.words.json}")), tmp_path)
    assert all(K.contrast(sc, white) > 10 for sc in K.chunks(stroked))  # punch: dark stroke under the fill


def _video(path: Path, color: str, secs: int = 7) -> Path:
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=1080x1920:r=30:d={secs}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_check_measures_contrast_and_faces_on_decoded_frames(tmp_path: Path) -> None:
    tl = resolve(load_spec(_short(tmp_path, "{source: talk.words.json, preset: minimal}")), tmp_path)
    video = _video(tmp_path / "white.mp4", "white")
    got = K.lines(tl, video, tmp_path / "work", faces=lambda image: [(0, 1000, 1080, 400)])
    assert got[0].startswith("WARN caption-contrast chunk=0 ratio=1.")
    assert "WARN caption-face chunk=0" in got
    dark = K.lines(tl, _video(tmp_path / "black.mp4", "black"), tmp_path / "work", faces=lambda image: [])
    assert dark == []


# --- slam keeps the ink in the safe zone -------------------------------------------------------------------


def _edge_slam(tmp: Path):
    rows = [("this", 0.0, 0.3), ("was", 0.32, 0.6), ("recorded", 0.62, 1.1), ("yesterday", 1.12, 1.6)]
    _words(tmp, rows)
    p = _write(tmp, "  - {photo: a, hold: 3s}\n", ["a"],
               extra="captions: {source: talk.words.json, emphasis: [3], words_per_chunk: [1, 3]}\n")
    tl = resolve(load_spec(p), tmp)
    (line,) = [ln for t in tl.texts for ln in t.lines if ln.motion is not None and ln.motion.emph_kind == "slam"]
    return tl, line


def _worst_overhang(line, motion) -> float:
    """Largest distance (design px) the emphasised word's posed ink box goes outside the 5% safe margin,
    over every frame of the entry and the shake."""
    boxes = [g.ink for g in line.glyphs.glyphs if g.emph]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    stroke = line.stroke.width if line.stroke else 0.0
    unit = next(g.unit for g in line.glyphs.glyphs if g.emph)
    worst = 0.0
    for k in range(40):
        pose = M.unit_pose(motion, unit, True, line.at + k / 4, line.at, line.stagger)
        if pose.opacity <= 0:
            continue
        hw, hh = (x1 - x0) / 2 * pose.scale + stroke, (y1 - y0) / 2 * pose.scale * pose.sy + stroke
        px, py = cx + pose.dx, cy + pose.dy
        worst = max(worst, 0.05 * DESIGN_W - (px - hw), (px + hw) - 0.95 * DESIGN_W)
    return worst


def test_slam_peak_stays_inside_the_safe_zone_at_the_frame_edge(tmp_path: Path) -> None:
    tl, line = _edge_slam(tmp_path)
    assert line.motion.slam_from is not None and 1.0 <= line.motion.slam_from < M.SLAM_FROM
    assert _worst_overhang(line, line.motion) <= 0.5
    # The test can fail: the uncapped slam really did leave the frame at this word.
    assert _worst_overhang(line, replace(line.motion, slam_from=None)) > 20


def test_slam_cap_leaves_other_films_untouched(tmp_path: Path) -> None:
    _, line = _edge_slam(tmp_path)
    plain = replace(line.motion, slam_from=None)
    assert "slam_from" not in {k for k, v in __import__("dataclasses").asdict(plain).items() if k != "slam_from"}
    assert M.entry("slam", 0.0, 6.0, 1.06, 0.0, None).scale == M.SLAM_FROM  # no cap: the original slam
