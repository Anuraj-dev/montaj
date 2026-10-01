from pathlib import Path

import pytest
from PIL import Image

from montaj.spec import SpecError, load_spec

_LINE = __import__("re").compile(r"^[^:]+: .+ \(.+\)$")


def _jpeg(folder: Path, stem: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{stem}.jpg"
    Image.new("RGB", (8, 8), (180, 90, 40)).save(p, "JPEG")
    return p


def _spec_text(shots: str, *, video: str | None = None, fx: str = "", extra: str = "") -> str:
    video = video or "size: 1080x1920\n  fps: 30\n  bpm: 120\n  look: warm-film"
    fx_block = f"\nfx:\n{fx}" if fx else ""
    return f"video:\n  {video}\nassets: assets\nshots:\n{shots}{fx_block}\n{extra}"


def _write(tmp: Path, shots: str, stems: list[str], **kw) -> Path:
    (tmp / "assets").mkdir(exist_ok=True)
    for s in stems:
        _jpeg(tmp / "assets", s)
    p = tmp / "montaj.yaml"
    p.write_text(_spec_text(shots, **kw))
    return p


def test_load_ok_and_int_photo_coerced(tmp_path: Path):
    p = _write(tmp_path, '  - {photo: 16, hold: 4b}\n  - {photo: "09", hold: 1s}\n', ["16", "09"])
    spec = load_spec(p)
    assert spec.shots[0].photo == "16"
    assert spec.shots[1].photo == "09"
    assert spec.shots[1].hold == "1s"


def test_bare_number_hold_is_spec_error(tmp_path: Path):
    p = _write(tmp_path, "  - {photo: a, hold: 30}\n", ["a"])
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert ei.value.errors
    assert all(_LINE.match(e) for e in ei.value.errors)
    assert any("shots[0].hold" in e and "22f, 4b or 1.5s" in e for e in ei.value.errors)


def test_beats_need_bpm(tmp_path: Path):
    p = _write(
        tmp_path,
        "  - {photo: a, hold: 4b}\n",
        ["a"],
        video="size: 1080x1920\n  fps: 30\n  look: warm-film",
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert any("video.bpm" in e and "beats" in e for e in ei.value.errors)
    assert all(_LINE.match(e) for e in ei.value.errors)


def test_odd_size(tmp_path: Path):
    p = _write(
        tmp_path,
        "  - {photo: a, hold: 10f}\n",
        ["a"],
        video="size: 1081x1920\n  fps: 30\n  look: warm-film",
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert any("video.size" in e and "even" in e for e in ei.value.errors)


def test_missing_photos_all_reported(tmp_path: Path):
    p = _write(
        tmp_path,
        '  - {photo: "16", hold: 10f}\n  - {photo: "99", hold: 10f}\n',
        ["16"],
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    errs = ei.value.errors
    assert any("shots[1].photo" in e and "99" in e and "have:" in e for e in errs)
    assert all(_LINE.match(e) for e in errs)


def test_two_missing_photos_and_extra_field(tmp_path: Path):
    p = _write(
        tmp_path,
        '  - {photo: "aa", hold: 10f, nope: 1}\n  - {photo: "bb", hold: 10f}\n',
        ["zz"],
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    errs = ei.value.errors
    assert any("nope" in e for e in errs)
    assert any("shots[0].photo" in e and "aa" in e for e in errs)
    assert any("shots[1].photo" in e and "bb" in e for e in errs)
    assert all(_LINE.match(e) for e in errs)


def test_fade_longer_than_hold(tmp_path: Path):
    p = _write(
        tmp_path,
        '  - {photo: a, hold: 10f}\n  - {photo: b, hold: 8f, in: {type: fade, dur: 12f}}\n',
        ["a", "b"],
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert any("shots[1].in.dur" in e and "fade" in e for e in ei.value.errors)


def test_whip_half_longer_than_previous(tmp_path: Path):
    p = _write(
        tmp_path,
        '  - {photo: a, hold: 4f}\n  - {photo: b, hold: 20f, in: {type: whip, dur: 14f}}\n',
        ["a", "b"],
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert any("whip half" in e and "previous" in e for e in ei.value.errors)


def test_swirl_on_first_shot(tmp_path: Path):
    p = _write(
        tmp_path,
        '  - {photo: a, hold: 30f, in: {type: swirl, dur: 10f}}\n',
        ["a"],
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert any("shots[0].in" in e and "previous" in e for e in ei.value.errors)


def test_overlapping_transition_windows(tmp_path: Path):
    p = _write(
        tmp_path,
        "  - {photo: a, hold: 20f}\n"
        "  - {photo: b, hold: 20f, in: {type: fade, dur: 20f}}\n"
        "  - {photo: c, hold: 20f, in: {type: whip, dur: 10f}}\n",
        ["a", "b", "c"],
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    errs = ei.value.errors
    assert any(
        "shots[2].in.dur" in e and "overlaps" in e and "lengthen shots[1]" in e for e in errs
    )
    assert all(_LINE.match(e) for e in errs)
    ok = tmp_path / "ok.yaml"
    ok.write_text(
        _spec_text(
            "  - {photo: a, hold: 20f}\n"
            "  - {photo: b, hold: 20f, in: {type: fade, dur: 15f}}\n"
            "  - {photo: c, hold: 20f, in: {type: whip, dur: 10f}}\n"
        )
    )
    load_spec(ok)


def _wall(prints: str, pos: str, zoom: str, extra_print: str = "") -> str:
    return f"""
  - hold: 30f
    wall:
      grid: 100x100
      seed: 1
      jitter: 0
      rot: 0
      print_height: 10
      prints:
        - {{photo: a, cell: [0, 0], straight: true{extra_print}}}
{prints}      light: {{center: [0, 0], radius: [1, 1]}}
      camera:
        pos: [{pos}]
        zoom: [{zoom}]
"""


def test_wall_camera_unknown_print(tmp_path: Path):
    p = _write(
        tmp_path,
        _wall(
            '        - {photo: b, cell: [1, 0], straight: true}\n',
            '{t: 0f, print: zz, at: [1, 1]}',
            '{t: 0f, value: 1, print: zz}',
        ),
        ["a", "b"],
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    errs = ei.value.errors
    assert any("camera.pos[0].print" in e and "zz" in e and "have:" in e and "a" in e for e in errs)
    assert any("camera.zoom[0].print" in e and "zz" in e and "have:" in e for e in errs)
    assert all(_LINE.match(e) for e in errs)


def test_wall_zoom_value_and_print_scale_must_be_positive_finite(tmp_path: Path):
    cases = [
        (", scale: 0", "{t: 0f}", "{t: 0f, value: 1}", "scale"),
        (", scale: .nan", "{t: 0f}", "{t: 0f, value: 1}", "scale"),
        ("", "{t: 0f}", "{t: 0f, value: 0}", "value"),
        ("", "{t: 0f}", "{t: 0f, value: .inf}", "value"),
        ("", "{t: 0f}", "{t: 0f, value: -.inf}", "value"),
        ("", "{t: 0f}", "{t: 0f, value: .nan}", "value"),
    ]
    for extra, pos, zoom, needle in cases:
        p = _write(tmp_path, _wall("", pos, zoom, extra_print=extra), ["a"])
        with pytest.raises(SpecError) as ei:
            load_spec(p)
        assert any(needle in e and "positive" in e for e in ei.value.errors), ei.value.errors
        assert all(_LINE.match(e) for e in ei.value.errors)


def test_bpm_nan_and_inf_are_spec_error(tmp_path: Path):
    for bpm in (".nan", ".inf", "-.inf"):
        p = _write(
            tmp_path,
            "  - {photo: a, hold: 4b}\n",
            ["a"],
            video=f"size: 1080x1920\n  fps: 30\n  bpm: {bpm}\n  look: warm-film",
        )
        with pytest.raises(SpecError) as ei:
            load_spec(p)
        assert any("video.bpm" in e and "finite" in e for e in ei.value.errors)
        assert all(_LINE.match(e) for e in ei.value.errors)


def test_wall_print_missing(tmp_path: Path):
    wall = """
  - hold: 10f
    wall:
      grid: 100x100
      seed: 1
      jitter: 0
      rot: 0
      print_height: 10
      prints:
        - {photo: "zz", cell: [0, 0], straight: true}
      light: {center: [0, 0], radius: [1, 1]}
      camera:
        pos: [{t: 0f, print: "zz", at: [1, 1]}]
        zoom: [{t: 0f, value: 1}]
"""
    p = _write(tmp_path, wall, ["aa"])
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert any("prints[0].photo" in e and "zz" in e for e in ei.value.errors)
