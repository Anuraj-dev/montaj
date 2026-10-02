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


def test_crop_is_integer_source_px(tmp_path: Path):
    p = _write(
        tmp_path,
        '  - {photo: a, hold: 10f, crop: [60, 180, 865, 1330]}\n'
        "  - hold: 10f\n"
        "    wall:\n"
        "      grid: 100x100\n"
        "      seed: 1\n"
        "      jitter: 0\n"
        "      rot: 0\n"
        "      print_height: 10\n"
        "      prints:\n"
        "        - {photo: a, cell: [0, 0], crop: [0.0, 0, 8, 8]}\n"
        "      light: {center: [0, 0], radius: [1, 1]}\n"
        "      camera:\n"
        "        pos: [{t: 0f}]\n"
        "        zoom: [{t: 0f, value: 1}]\n",
        ["a"],
    )
    spec = load_spec(p)
    assert spec.shots[0].crop == (60, 180, 865, 1330)
    assert all(type(x) is int for x in spec.shots[0].crop)
    wall = spec.shots[1].wall
    assert wall is not None and wall.prints[0].crop == (0, 0, 8, 8)
    assert all(type(x) is int for x in wall.prints[0].crop)

    bad = _write(tmp_path, "  - {photo: a, hold: 10f, crop: [60.5, 180, 865, 1330]}\n", ["a"])
    with pytest.raises(SpecError) as ei:
        load_spec(bad)
    assert any("crop" in e and "60.5" in e and "integer source-px" in e for e in ei.value.errors)
    assert all(_LINE.match(e) for e in ei.value.errors)

    wall_bad = _write(
        tmp_path,
        "  - hold: 10f\n"
        "    wall:\n"
        "      grid: 100x100\n"
        "      seed: 1\n"
        "      jitter: 0\n"
        "      rot: 0\n"
        "      print_height: 10\n"
        "      prints:\n"
        "        - {photo: a, cell: [0, 0], crop: [1, 2, 3.25, 4]}\n"
        "      light: {center: [0, 0], radius: [1, 1]}\n"
        "      camera:\n"
        "        pos: [{t: 0f}]\n"
        "        zoom: [{t: 0f, value: 1}]\n",
        ["a"],
    )
    with pytest.raises(SpecError) as ei:
        load_spec(wall_bad)
    assert any("prints[0].crop" in e and "3.25" in e for e in ei.value.errors)
    assert all(_LINE.match(e) for e in ei.value.errors)


def test_birthday_recipe_crops_load(tmp_path: Path):
    src = Path(__file__).resolve().parents[1] / "recipes" / "birthday-short.yaml"
    stems = ["09", "10", "12", "16", "19", "20", "21", "22", "28", "29", "32", "33", "34", "35", "37", "38"]
    for stem in stems:
        _jpeg(tmp_path / "assets", stem)
    dest = tmp_path / "montaj.yaml"
    dest.write_text(src.read_text())
    spec = load_spec(dest)
    assert spec.shots[2].crop == (60, 180, 865, 1330)
    wall = spec.shots[-1].wall
    assert wall is not None
    assert wall.prints[7].crop == (60, 180, 865, 1330)
    assert wall.prints[15].crop == (0, 0, 1024, 890)


def _errors(tmp_path: Path, shots: str, *, video: str | None = None, extra: str = "") -> list[str]:
    p = _write(tmp_path, shots, ["a"], video=video, extra=extra)
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert ei.value.errors
    assert all(_LINE.match(e) for e in ei.value.errors)
    return ei.value.errors


def test_unknown_look_names_the_presets(tmp_path: Path):
    errs = _errors(tmp_path, "  - {photo: a, hold: 10f}\n", video="size: 1080x1920\n  fps: 30\n  look: noir")
    assert "video.look: unknown 'noir' (use warm-film or golden-film)" in errs


def test_look_names_match_finish_presets():
    from montaj.render.finish import LOOKS
    from montaj.spec import LOOK_NAMES

    assert set(LOOK_NAMES) == set(LOOKS)


def test_intro_and_outro_types_are_closed(tmp_path: Path):
    intro = _errors(
        tmp_path,
        "  - {photo: a, hold: 10f}\n",
        video="size: 1080x1920\n  fps: 30\n  look: warm-film\n  intro: {type: spin, dur: 10f}",
    )
    assert "video.intro.type: unknown 'spin' (use white-lift)" in intro
    outro = _errors(
        tmp_path,
        "  - {photo: a, hold: 10f}\n",
        video="size: 1080x1920\n  fps: 30\n  look: warm-film\n  outro: {type: wipe, dur: 10f}",
    )
    assert "video.outro.type: unknown 'wipe' (use glow or fade)" in outro
    ok = _write(
        tmp_path,
        "  - {photo: a, hold: 30f}\n",
        ["a"],
        video=(
            "size: 1080x1920\n  fps: 30\n  look: warm-film\n"
            "  intro: {type: white-lift, dur: 8f}\n  outro: {type: fade, dur: 8f}"
        ),
    )
    spec = load_spec(ok)
    assert spec.video.intro is not None and spec.video.intro.type == "white-lift"
    assert spec.video.outro is not None and spec.video.outro.type == "fade"


def test_bool_burst_and_sub_size_are_rejected(tmp_path: Path):
    burst_true = _errors(
        tmp_path,
        "  - {photo: a, hold: 10f}\n",
        extra="fx:\n  - {burst: true, at: 1s, pos: [1, 2]}\n",
    )
    assert any("fx[0].burst" in e and "boolean" in e for e in burst_true)
    burst_false = _errors(
        tmp_path,
        "  - {photo: a, hold: 10f}\n",
        extra="fx:\n  - {burst: false, at: 1s, pos: [1, 2]}\n",
    )
    assert any("fx[0].burst" in e and "boolean" in e for e in burst_false)
    size_true = _errors(
        tmp_path,
        "  - {photo: a, hold: 30f}\n",
        extra="subs:\n  - {from: 0s, to: 1s, text: hi, size: true}\n",
    )
    assert any("subs[0].size" in e and "boolean" in e for e in size_true)
    size_false = _errors(
        tmp_path,
        "  - {photo: a, hold: 30f}\n",
        extra="subs:\n  - {from: 0s, to: 1s, text: hi, size: false}\n",
    )
    assert any("subs[0].size" in e and "boolean" in e for e in size_false)


def test_burst_must_be_a_whole_number(tmp_path: Path):
    errs = _errors(
        tmp_path,
        "  - {photo: a, hold: 10f}\n",
        extra="fx:\n  - {burst: 1.5, at: 1s, pos: [1, 2]}\n",
    )
    assert any(e.startswith("fx[0].burst: need a whole number") for e in errs)
    p = _write(
        tmp_path,
        "  - {photo: a, hold: 10f}\n",
        ["a"],
        extra="fx:\n  - {burst: 3.0, at: 1s, pos: [10, 20]}\n",
    )
    spec = load_spec(p)
    assert spec.fx[0].burst == 3


def test_shot0_non_fade_hint_and_missing_source(tmp_path: Path):
    errs = _errors(tmp_path, "  - {photo: a, hold: 30f, in: {type: swirl, dur: 10f}}\n")
    assert any("remove in or use type: fade" in e and "use type: cut" not in e for e in errs)
    missing = _errors(tmp_path, "  - {hold: 10f}\n")
    assert any("need photo, wall, clip or blank" in e for e in missing)
    blank = _write(tmp_path, "  - {blank: true, hold: 10f}\n", ["a"])
    assert load_spec(blank).shots[0].blank is True
    fade = _write(tmp_path, "  - {photo: a, hold: 30f, in: {type: fade, dur: 10f}}\n", ["a"])
    assert load_spec(fade).shots[0].in_.type == "fade"


def test_sub_size_and_color_validate(tmp_path: Path):
    bad_size = _errors(
        tmp_path,
        "  - {photo: a, hold: 30f}\n",
        extra="subs:\n  - {from: 0s, to: 1s, text: hi, size: 0}\n",
    )
    assert any("subs[0].size" in e and "finite" in e and "> 0" in e for e in bad_size)
    bad_color = _errors(
        tmp_path,
        "  - {photo: a, hold: 30f}\n",
        extra="subs:\n  - {from: 0s, to: 1s, text: hi, color: blue}\n",
    )
    assert any("subs[0].color" in e and "#rrggbb" in e for e in bad_color)
    p = _write(
        tmp_path,
        "  - {photo: a, hold: 30f}\n",
        ["a"],
        extra="subs:\n  - {from: 0s, to: 1s, text: hi, size: 60, color: cream}\n",
    )
    spec = load_spec(p)
    assert spec.subs[0].size == 60
    assert spec.subs[0].color == "cream"


def test_sub_size_changes_segment_hash_and_reaches_the_timeline(tmp_path: Path):
    from montaj.project import file_sha, segment_hash
    from montaj.timeline import resolve

    p = _write(
        tmp_path,
        "  - {photo: a, hold: 30f}\n",
        ["a"],
        extra="subs:\n  - {from: 0s, to: 1s, text: hello, size: 44, color: '#112233'}\n",
    )
    spec = load_spec(p)
    tl = resolve(spec, p.parent)
    assert tl.subs[0].size == 44
    assert tl.subs[0].color == "#112233"
    shas = {"a": file_sha(p.parent / "assets" / "a.jpg")}
    h0 = segment_hash(tl, tl.segments[0], shas, "e1")
    edited = spec.model_copy(deep=True)
    edited.subs[0].size = 80
    h1 = segment_hash(resolve(edited, p.parent), tl.segments[0], shas, "e1")
    assert h0 != h1


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
