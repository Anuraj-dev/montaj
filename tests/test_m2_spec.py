from pathlib import Path

import pytest
from PIL import Image

from montaj.spec import Spec, SpecError, load_spec, parse_time
from montaj.timeline import resolve

_LINE = __import__("re").compile(r"^[^:]+: .+ \(.+\)$")


def _jpeg(folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (180, 90, 40)).save(folder / f"{stem}.jpg", "JPEG")


def _write(tmp: Path, body: str, stems: list[str] = ("a",), *, extra: str = "") -> Path:
    assets = tmp / "assets"
    assets.mkdir(exist_ok=True)
    for s in stems:
        _jpeg(assets, s)
    p = tmp / "montaj.yaml"
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 76\n  look: golden-film\n"
        f"assets: assets\nshots:\n{body}{extra}\n"
    )
    return p


def _err(tmp: Path, body: str, needle: str, stems: list[str] = ("a",), extra: str = "") -> None:
    p = _write(tmp, body, stems, extra=extra)
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert ei.value.errors
    assert all(_LINE.match(e) for e in ei.value.errors)
    assert any(needle in e for e in ei.value.errors), ei.value.errors


def test_round_trip_new_fields(tmp_path: Path):
    extra = """
markers:
  v1: 12.0s
  later: v1+0.3s
audio:
  track: music/song.wav
  markers: music/markers.json
  fade_out: 3s
  loudnorm: true
text:
  - from: 0s
    to: 7.2s
    fade_out: 0.9s
    lines:
      - {text: "a little film for", style: caps, y: 700, at: 1.2s, stagger: 0.12s}
      - {text: Arshiya, style: script, size: 230, y: 760, at: v1-0.3s, by: char}
subs:
  - {from: 12.0s, to: 17.7s, text: "Jaipur ki dhoop mein"}
fx:
  - {hit: 0.7, at: v1}
  - {burst: 70, at: v1, pos: [540, 1100]}
tracks:
  dust: [{t: 0s, v: .35}, {t: v1, v: .85, ease: linear}]
  glow: [{t: 0s, v: .25}]
  bars: [{t: 0s, v: 150}, {t: v1+0.4s, v: 0}]
"""
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "song.wav").write_bytes(b"RIFF")
    (tmp_path / "music" / "markers.json").write_text(
        '{"words": [{"i": 1, "w": "Jaipur", "s": 12.0, "e": 13.0}]}'
    )
    shots = (
        "  - {photo: a, until: v1+0.3s, drift: {zoom: [1.25, 1.04], pan: [0, 40]}, "
        "tone: {brightness: .5}, pulse: heartbeat}\n"
        "  - {photo: a, hold: 1s, in: {type: fade, dur: 0.5s, blur: 14}, "
        "frame: {caption: hi, tilt: [2, -1.5]}, morph: {photo: a, at: 0.2s, dur: 0.4s, center: [.55, .5]}}\n"
    )
    p = _write(tmp_path, shots, extra=extra)
    spec = load_spec(p)
    dumped = spec.model_dump(mode="json", by_alias=True)
    assert Spec.model_validate(dumped) == spec
    assert spec.video.background == "#000000"
    assert spec.shots[0].until == "v1+0.3s"
    assert spec.shots[0].hold is None
    assert spec.markers["later"] == "v1+0.3s"
    assert spec.text[0].lines[0].style == "caps"
    assert spec.tracks is not None and spec.tracks.bars[0].v == 150


def test_hold_xor_until(tmp_path: Path):
    _err(tmp_path, "  - {photo: a, hold: 10f, until: 20f}\n", "hold and until")
    _err(tmp_path, "  - {photo: a}\n", "need hold or until")


def test_until_before_start(tmp_path: Path):
    _err(
        tmp_path,
        "  - {photo: a, hold: 30f}\n  - {photo: a, until: 10f}\n",
        "not after start 30f",
    )


def test_unknown_marker(tmp_path: Path):
    extra = "\nmarkers:\n  ch1: 1s\n  ch1b: 2s\ntext:\n  - from: 0s\n    to: 3s\n    lines:\n      - {text: x, y: 10, at: ch9}\n"
    _err(tmp_path, "  - {photo: a, hold: 10f}\n", "unknown marker 'ch9'", extra=extra)


def test_photo_wall_clip_exclusive(tmp_path: Path):
    (tmp_path / "assets").mkdir(exist_ok=True)
    _jpeg(tmp_path / "assets", "a")
    (tmp_path / "assets" / "intro.mp4").write_bytes(b"x")
    _err(tmp_path, "  - {photo: a, clip: intro, hold: 10f}\n", "mutually exclusive", stems=["a"])
    _err(tmp_path, "  - {hold: 10f}\n", "need photo, wall, clip or blank")


def test_zoom_and_drift(tmp_path: Path):
    _err(
        tmp_path,
        "  - {photo: a, hold: 10f, zoom: 1.2, drift: {zoom: [1, 1.1]}}\n",
        "zoom and drift",
    )


def test_shot0_transition(tmp_path: Path):
    _err(tmp_path, "  - {photo: a, hold: 30f, in: {type: swirl, dur: 10f}}\n", "previous shot")
    p = _write(tmp_path, "  - {photo: a, hold: 30f, in: {type: fade, dur: 10f, blur: 14}}\n")
    spec = load_spec(p)
    assert spec.shots[0].in_ is not None and spec.shots[0].in_.type == "fade"
    assert spec.shots[0].in_.blur == 14


def test_burst_without_pos(tmp_path: Path):
    _err(
        tmp_path,
        "  - {photo: a, hold: 10f}\n",
        "required for burst",
        extra="\nfx:\n  - {burst: 70, at: 1s}\n",
    )


def test_pulse_without_bpm(tmp_path: Path):
    p = tmp_path / "montaj.yaml"
    (tmp_path / "assets").mkdir()
    _jpeg(tmp_path / "assets", "a")
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\n"
        "assets: assets\nshots:\n  - {photo: a, hold: 10f, pulse: heartbeat}\n"
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert all(_LINE.match(e) for e in ei.value.errors)
    assert any("pulse" in e and "bpm" in e for e in ei.value.errors)


def test_parse_time_unrounded():
    assert parse_time("1.51s", {}, 30, None) == pytest.approx(45.3)
    assert parse_time("12.0s", {}, 30, 76) == pytest.approx(360.0)
    markers = {"ch1": 47.96 * 30}
    assert parse_time("ch1+2.4s", markers, 30, None) == pytest.approx((47.96 + 2.4) * 30)
    assert parse_time("ch1-6f", markers, 30, None) == pytest.approx(47.96 * 30 - 6)


def test_word_marker(tmp_path: Path):
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "song.wav").write_bytes(b"RIFF")
    (tmp_path / "music" / "markers.json").write_text(
        '{"words": [{"i": 1, "w": "Jaipur", "s": 12.0, "e": 13.08}]}'
    )
    extra = """
markers:
  v1: word:1
audio:
  track: music/song.wav
  markers: music/markers.json
"""
    p = _write(tmp_path, "  - {photo: a, until: v1}\n", extra=extra)
    spec = load_spec(p)
    tl = resolve(spec, p.parent)
    assert tl.markers["v1"] == pytest.approx(360.0)
    assert tl.shots[0].end == 360


def test_marker_name_ending_in_b_without_bpm(tmp_path: Path):
    p = tmp_path / "montaj.yaml"
    (tmp_path / "assets").mkdir()
    _jpeg(tmp_path / "assets", "a")
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\n"
        "assets: assets\nmarkers:\n  ch1b: 1s\n"
        "shots:\n  - {photo: a, hold: 40f}\n"
        "fx:\n  - {hit: 0.5, at: ch1b}\n"
    )
    spec = load_spec(p)
    tl = resolve(spec, p.parent)
    assert tl.markers["ch1b"] == pytest.approx(30.0)


def test_word_marker_bad_s_is_spec_error(tmp_path: Path):
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "song.wav").write_bytes(b"RIFF")
    extra = """
markers:
  v1: word:1
audio:
  track: music/song.wav
  markers: music/markers.json
"""
    cases = (
        ('{"words": [{"i": 1, "w": "x", "s": "nope", "e": 1}]}', "not a number"),
        ('{"words": [{"i": 1, "w": "x", "s": NaN, "e": 1}]}', "finite"),
        ('{"words": [{"i": 1, "w": "x", "s": Infinity, "e": 1}]}', "finite"),
        ('{"words": [{"i": 1, "w": "x", "s": null, "e": 1}]}', "not a number"),
    )
    for payload, needle in cases:
        (tmp_path / "music" / "markers.json").write_text(payload)
        p = _write(tmp_path, "  - {photo: a, until: v1}\n", extra=extra)
        with pytest.raises(SpecError) as ei:
            load_spec(p)
        assert all(_LINE.match(e) for e in ei.value.errors)
        assert any("audio.markers.words[0].s" in e and needle in e for e in ei.value.errors), ei.value.errors
