from pathlib import Path
from shutil import copyfile

import pytest
from PIL import Image

from montaj.project import file_sha, segment_hash
from montaj.spec import load_spec
from montaj.timeline import Blank, Clip, Still, Trans, Wall, resolve, wall_static_spans

BIRTHDAY_STEMS = [
    "09", "10", "12", "16", "19", "20", "21", "22",
    "28", "29", "32", "33", "34", "35", "37", "38",
]
BIRTHDAY_STARTS = [0, 60, 120, 150, 165, 180, 195, 210, 225, 300, 345, 391, 450]


def _jpeg(folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(folder / f"{stem}.jpg", "JPEG")


def _load(tmp: Path, yaml_text: str, stems: list[str]):
    assets = tmp / "assets"
    assets.mkdir()
    for s in stems:
        _jpeg(assets, s)
    p = tmp / "montaj.yaml"
    p.write_text(yaml_text)
    return resolve(load_spec(p), p.parent)


def _mini(tmp: Path, body: str, stems: list[str], *, intro="", outro="", fx=""):
    intro_l = f"\n  intro: {intro}" if intro else ""
    outro_l = f"\n  outro: {outro}" if outro else ""
    fx_b = f"\nfx:\n{fx}" if fx else ""
    text = (
        "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 120\n  look: warm-film"
        f"{intro_l}{outro_l}\nassets: assets\nshots:\n{body}{fx_b}\n"
    )
    return _load(tmp, text, stems)


def test_birthday_short_frames_and_starts(tmp_path: Path):
    src = Path("recipes/birthday-short.yaml")
    dst = tmp_path / "birthday-short.yaml"
    copyfile(src, dst)
    assets = tmp_path / "assets"
    assets.mkdir()
    for s in BIRTHDAY_STEMS:
        _jpeg(assets, s)
    tl = resolve(load_spec(dst), dst.parent)
    assert tl.n_frames == 840
    assert tl.fps == 30
    assert tl.size == (1080, 1920)
    assert [s.start for s in tl.shots] == BIRTHDAY_STARTS
    assert tl.shots[-1].end == 840
    assert [s.kind for s in tl.shots] == ["still"] * 12 + ["wall"]
    assert len(tl.segments) == 13
    assert tl.segments[0].start == 0 and tl.segments[-1].end == 840
    for i, seg in enumerate(tl.segments):
        assert seg.index == i
        assert seg.start == tl.shots[i].start
        assert seg.end == tl.shots[i].end
        if i:
            assert tl.segments[i - 1].end == seg.start


def test_durations_f_b_s(tmp_path: Path):
    tl = _mini(
        tmp_path,
        "  - {photo: a, hold: 4b}\n  - {photo: b, hold: 1.5s}\n  - {photo: c, hold: 10f}\n",
        ["a", "b", "c"],
    )
    assert [s.start for s in tl.shots] == [0, 60, 105]
    assert tl.n_frames == 115


def test_fade_window_edges(tmp_path: Path):
    tl = _mini(
        tmp_path,
        "  - {photo: a, hold: 20f}\n  - {photo: b, hold: 20f, in: {type: fade, dur: 15f}}\n",
        ["a", "b"],
    )
    assert isinstance(tl.plan(19).scene, Still) and tl.plan(19).scene.shot == 0
    sc = tl.plan(20).scene
    assert isinstance(sc, Trans) and sc.kind == "fade"
    assert sc.t0 == 20.0 and sc.dur == 15.0
    assert isinstance(sc.a, Still) and sc.a.shot == 0
    assert isinstance(sc.b, Still) and sc.b.shot == 1
    assert isinstance(tl.plan(34).scene, Trans) and tl.plan(34).scene.kind == "fade"
    assert isinstance(tl.plan(35).scene, Still) and tl.plan(35).scene.shot == 1


def test_swirl_window_and_center(tmp_path: Path):
    tl = _mini(
        tmp_path,
        "  - {photo: a, hold: 60f, focus: [470, 745]}\n"
        "  - {photo: b, hold: 60f, in: {type: swirl, dur: 22f, center: [431, 578]}}\n",
        ["a", "b"],
    )
    assert isinstance(tl.plan(59).scene, Still)
    sc = tl.plan(60).scene
    assert isinstance(sc, Trans) and sc.kind == "swirl"
    assert sc.center == (431.0, 578.0) and sc.t0 == 60.0 and sc.dur == 22.0
    assert isinstance(tl.plan(81).scene, Trans)
    assert isinstance(tl.plan(82).scene, Still) and tl.plan(82).scene.shot == 1


def test_swirl_center_defaults_to_outgoing_focus(tmp_path: Path):
    tl = _mini(
        tmp_path,
        "  - {photo: a, hold: 20f, focus: [10, 20]}\n"
        "  - {photo: b, hold: 20f, in: {type: swirl, dur: 8f}}\n",
        ["a", "b"],
    )
    sc = tl.plan(20).scene
    assert isinstance(sc, Trans) and sc.center == (10.0, 20.0)


def test_whip_window_x_and_y(tmp_path: Path):
    tl = _mini(
        tmp_path,
        "  - {photo: a, hold: 30f}\n"
        "  - {photo: b, hold: 30f, in: {type: whip, dur: 14f, axis: x}}\n"
        "  - {photo: c, hold: 30f, in: {type: whip, dur: 14f, axis: y}}\n",
        ["a", "b", "c"],
    )
    assert isinstance(tl.plan(22).scene, Still) and tl.plan(22).scene.shot == 0
    sc = tl.plan(23).scene
    assert isinstance(sc, Trans) and sc.kind == "whip" and sc.axis == "x"
    assert sc.t0 == 23.0 and sc.dur == 14.0
    assert isinstance(tl.plan(36).scene, Trans)
    assert isinstance(tl.plan(37).scene, Still) and tl.plan(37).scene.shot == 1
    scy = tl.plan(53).scene
    assert isinstance(scy, Trans) and scy.kind == "whip" and scy.axis == "y"
    assert scy.t0 == 53.0
    assert isinstance(tl.plan(52).scene, Still)
    assert isinstance(tl.plan(67).scene, Still) and tl.plan(67).scene.shot == 2


def test_flash_and_leak_windows(tmp_path: Path):
    tl = _mini(
        tmp_path,
        "  - {photo: a, hold: 40f}\n  - {photo: b, hold: 80f, flash: 0.3}\n",
        ["a", "b"],
        fx="  - {leak: 0.35, at: 70f, dir: 1}\n  - {flash: 0.2, at: 10f}\n",
    )
    assert tl.plan(39).flashes == ()
    assert tl.plan(40).flashes == ((40, 0.3),)
    assert tl.plan(45).flashes == ((40, 0.3),)
    assert tl.plan(46).flashes == ()
    assert tl.plan(9).flashes == ()
    assert tl.plan(10).flashes == ((10, 0.2),)
    assert tl.plan(15).flashes == ((10, 0.2),)
    assert tl.plan(16).flashes == ()
    assert tl.plan(45).leaks == ()
    assert tl.plan(46).leaks == ((70, 0.35, 1),)
    assert tl.plan(70).leaks == ((70, 0.35, 1),)
    assert tl.plan(94).leaks == ((70, 0.35, 1),)
    assert tl.plan(95).leaks == ()


def test_intro_outro(tmp_path: Path):
    tl = _mini(
        tmp_path,
        "  - {photo: a, hold: 100f}\n",
        ["a"],
        intro="{type: white-lift, dur: 30f}",
        outro="{type: glow, dur: 20f}",
    )
    assert tl.plan(0).intro == ("white-lift", 30)
    assert tl.plan(29).intro == ("white-lift", 30)
    assert tl.plan(30).intro is None
    assert tl.plan(79).outro is None
    assert tl.plan(80).outro == ("glow", 20, 80)
    assert tl.plan(99).outro == ("glow", 20, 80)


def test_wall_scene_and_fade_into_wall(tmp_path: Path):
    wall = """
  - {photo: a, hold: 20f}
  - hold: 40f
    in: {type: fade, dur: 15f}
    wall:
      grid: 1500x1950
      seed: 5
      jitter: 0
      rot: 0
      print_height: 10
      prints:
        - {photo: a, cell: [0, 0], straight: true}
      light: {center: [0, 0], radius: [1, 1]}
      camera:
        pos: [{t: 0f, at: [1, 1]}]
        zoom: [{t: 0f, value: 1}]
"""
    tl = _mini(tmp_path, wall, ["a"])
    sc = tl.plan(20).scene
    assert isinstance(sc, Trans) and sc.kind == "fade"
    assert isinstance(sc.b, Wall) and sc.b.t0 == 20 and sc.b.shot == 1
    w = tl.plan(35).scene
    assert isinstance(w, Wall) and w.t0 == 20
    assert tl.n_frames == 60


def test_wall_camera_key_t_is_integer_frames(tmp_path: Path):
    def body(t_pos, t_zoom):
        return f"""
  - {{photo: a, hold: 8f}}
  - hold: 90f
    wall:
      grid: 10x10
      seed: 1
      jitter: 0
      rot: 0
      print_height: 8
      prints:
        - {{photo: a, cell: [0, 0], straight: true}}
      light: {{center: [0, 0], radius: [1, 1]}}
      camera:
        pos:
          - {{t: {t_pos[0]}}}
          - {{t: {t_pos[1]}, print: a, at: [1, 1]}}
          - {{t: {t_pos[2]}}}
        zoom:
          - {{t: {t_zoom[0]}, value: 1.0}}
          - {{t: {t_zoom[1]}, value: 0.5}}
"""

    tl = _mini(tmp_path, body(("0f", "1s", "4b"), ("0f", "1.5s")), ["a"])
    wall = tl.shots[1].spec.wall
    assert wall is not None
    assert [k.t for k in wall.camera.pos] == [0, 30, 60]
    assert [k.t for k in wall.camera.zoom] == [0, 45]
    dumped = tl.shots[1].spec.model_dump(mode="json")
    assert dumped["wall"]["camera"]["pos"][1]["t"] == 30
    assert dumped["wall"]["camera"]["zoom"][1]["t"] == 45
    shas = {s: file_sha(tmp_path / "assets" / f"{s}.jpg") for s in ["a"]}
    h_units = segment_hash(tl, tl.segments[1], shas, "e1")
    other = tmp_path / "frames"
    other.mkdir()
    tl_f = _mini(other, body(("0f", "30f", "60f"), ("0f", "45f")), ["a"])
    shas_f = {s: file_sha(other / "assets" / f"{s}.jpg") for s in ["a"]}
    # same bytes as the first jpegs so photo sha matches
    shas_f["a"] = shas["a"]
    assert segment_hash(tl_f, tl_f.segments[1], shas_f, "e1") == h_units
    assert tl.spec.shots[1].wall.camera.pos[1].t == "1s"


def test_birthday_transition_edges(tmp_path: Path):
    src = Path("recipes/birthday-short.yaml")
    dst = tmp_path / "birthday-short.yaml"
    copyfile(src, dst)
    assets = tmp_path / "assets"
    assets.mkdir()
    for s in BIRTHDAY_STEMS:
        _jpeg(assets, s)
    tl = resolve(load_spec(dst), dst.parent)
    swirl = tl.plan(60).scene
    assert isinstance(swirl, Trans) and swirl.kind == "swirl" and swirl.center == (431.0, 578.0)
    assert isinstance(tl.plan(59).scene, Still)
    assert isinstance(tl.plan(81).scene, Trans)
    assert isinstance(tl.plan(82).scene, Still)
    whip = tl.plan(218).scene
    assert isinstance(whip, Trans) and whip.kind == "whip" and whip.axis == "x" and whip.t0 == 218.0
    assert isinstance(tl.plan(217).scene, Still)
    assert isinstance(tl.plan(231).scene, Trans)
    assert isinstance(tl.plan(232).scene, Still)
    fade = tl.plan(345).scene
    assert isinstance(fade, Trans) and fade.kind == "fade"
    assert isinstance(tl.plan(344).scene, Still)
    assert isinstance(tl.plan(359).scene, Trans)
    assert isinstance(tl.plan(360).scene, Still)
    whip_y = tl.plan(384).scene
    assert isinstance(whip_y, Trans) and whip_y.kind == "whip" and whip_y.axis == "y"
    assert isinstance(tl.plan(383).scene, Still)
    assert isinstance(tl.plan(397).scene, Trans)
    assert isinstance(tl.plan(398).scene, Still)
    wall_fade = tl.plan(450).scene
    assert isinstance(wall_fade, Trans) and isinstance(wall_fade.b, Wall)
    assert isinstance(tl.plan(465).scene, Wall)
    assert tl.plan(0).intro == ("white-lift", 30)
    assert tl.plan(730).outro == ("glow", 110, 730)
    assert tl.plan(120).flashes == ((120, 0.3),)
    assert tl.plan(70).leaks == ((70, 0.35, 1),)


def test_birthday_wall_static_spans(tmp_path: Path):
    """Camera holds on print 34 at the wall start, then after the last key through the outro.

    Decided from spec keys only: pos 0–110 hold print 34, zoom 0–45 hold 1.606; both
    constant on [0, 45]. Last zoom key is 290, so the tail is local t ≥ 290.
    """
    src = Path("recipes/birthday-short.yaml")
    dst = tmp_path / "birthday-short.yaml"
    copyfile(src, dst)
    assets = tmp_path / "assets"
    assets.mkdir()
    for s in BIRTHDAY_STEMS:
        _jpeg(assets, s)
    tl = resolve(load_spec(dst), dst.parent)
    wall = tl.shots[-1]
    assert wall.kind == "wall" and wall.start == 450 and wall.end == 840
    spans = wall_static_spans(wall)
    assert (450, 496) in spans  # local [0, 45] inclusive
    assert (740, 840) in spans  # after last zoom key t=290
    covered = {f for lo, hi in spans for f in range(lo, hi)}
    assert 450 in covered and 495 in covered and 496 not in covered
    assert 740 in covered and 839 in covered
    # zoom-out, pan, zoom-in must not count as holds
    for f in (496, 560, 600, 669, 670, 671, 700, 739):
        assert f not in covered, f"moving wall frame {f} marked static: {spans}"


def test_until_rounding(tmp_path: Path):
    tl = _mini(tmp_path, "  - {photo: a, until: 1.51s}\n", ["a"])
    assert tl.n_frames == 45
    assert tl.shots[0].end == 45


def test_visible_span_fade_and_whip(tmp_path: Path):
    tl = _mini(
        tmp_path,
        "  - {photo: a, hold: 20f, drift: {zoom: [1.2, 1.0]}}\n"
        "  - {photo: b, hold: 20f, in: {type: fade, dur: 10f}}\n",
        ["a", "b"],
    )
    sc0 = tl.plan(0).scene
    assert isinstance(sc0, Still) and sc0.span == (0, 30)
    assert isinstance(tl.plan(19).scene, Still) and tl.plan(19).scene.span == (0, 30)
    assert isinstance(tl.plan(20).scene, Trans)
    assert tl.plan(20).scene.a.span == (0, 30)

    other = tmp_path / "whip"
    other.mkdir()
    tlw = _mini(
        other,
        "  - {photo: a, hold: 30f, drift: {zoom: [1.0, 1.1]}}\n"
        "  - {photo: b, hold: 30f, drift: {zoom: [1.1, 1.0]}, in: {type: whip, dur: 14f}}\n",
        ["a", "b"],
    )
    assert isinstance(tlw.plan(0).scene, Still) and tlw.plan(0).scene.span == (0, 37)
    sc1 = tlw.plan(40).scene
    assert isinstance(sc1, Still) and sc1.span == (23, 60)
    whip = tlw.plan(23).scene
    assert isinstance(whip, Trans) and whip.kind == "whip"
    assert whip.a.span == (0, 37) and whip.b.span == (23, 60)


def test_text_and_sub_windows(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    _jpeg(assets, "a")
    p = tmp_path / "montaj.yaml"
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 120\n  look: warm-film\n"
        "assets: assets\nshots:\n  - {photo: a, hold: 70f}\n"
        "text:\n  - from: 1s\n    to: 2s\n    lines:\n      - {text: hello, y: 100, at: 1s}\n"
        "subs:\n  - {from: 0.5s, to: 1.5s, text: lyric}\n"
    )
    tl = resolve(load_spec(p), p.parent)
    assert tl.plan(28).texts == ()
    assert tl.plan(29).texts == (0,)
    assert tl.plan(60).texts == (0,)
    assert tl.plan(61).texts == (0,)
    assert tl.plan(62).texts == ()
    # from 0.5s=15f, to 1.5s=45f; opacity [from−0.2s, to+0.3s] ±1 frame → [8, 55]
    assert tl.plan(7).subs == ()
    assert tl.plan(8).subs == (0,)
    assert tl.plan(55).subs == (0,)
    assert tl.plan(56).subs == ()


def test_hit_and_burst_windows(tmp_path: Path):
    extra = (
        "\nfx:\n"
        "  - {hit: 0.7, at: 10s}\n"
        "  - {burst: 70, at: 10s, pos: [540, 1100]}\n"
    )
    tl = _mini(tmp_path, "  - {photo: a, hold: 410f}\n", ["a"], fx=extra.split("\nfx:\n", 1)[1])
    assert tl.plan(296).hits == ()
    assert tl.plan(297).hits == ((300.0, 0.7),)
    assert tl.plan(335).hits == ((300.0, 0.7),)
    assert tl.plan(336).hits == ()
    assert tl.plan(299).bursts == ()
    b = tl.plan(300).bursts
    assert b == ((0, 300.0, 540.0, 1100.0, 70),)
    assert tl.plan(402).bursts == b
    assert tl.plan(403).bursts == ()


def test_track_values_linear_sine_before_after(tmp_path: Path):
    extra = """
tracks:
  dust:
    - {t: 1s, v: 0.35}
    - {t: 2s, v: 0.85, ease: linear}
  glow:
    - {t: 0s, v: 0.25}
  bars:
    - {t: 0s, v: 150}
    - {t: 1s, v: 0}
"""
    p = tmp_path / "t.yaml"
    assets = tmp_path / "assets"
    assets.mkdir()
    _jpeg(assets, "a")
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 120\n  look: warm-film\n"
        "assets: assets\nshots:\n  - {photo: a, hold: 90f}\n" + extra
    )
    tl = resolve(load_spec(p), p.parent)
    assert tl.plan(0).dust == pytest.approx(0.35)
    assert tl.plan(45).dust == pytest.approx(0.6)
    assert tl.plan(89).dust == pytest.approx(0.85)
    assert tl.plan(10).glow == pytest.approx(0.25)
    import math
    k = 8 / 30.0
    sine = 0.5 - 0.5 * math.cos(math.pi * k)
    assert tl.plan(8).bars == pytest.approx(150 * (1 - sine), abs=1e-6)
    assert tl.plan(0).bars == pytest.approx(150)
    assert tl.plan(30).bars == pytest.approx(0)


def test_fade_out_plan(tmp_path: Path):
    tl = _mini(
        tmp_path,
        "  - {photo: a, hold: 100f}\n",
        ["a"],
        outro="{type: fade, dur: 1.2s}",
    )
    assert tl.plan(63).fade_out is None
    assert tl.plan(64).fade_out == (36, 64)
    assert tl.plan(99).fade_out == (36, 64)
    assert tl.plan(64).outro is None


def test_clip_scene(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    _jpeg(assets, "a")
    (assets / "intro.mp4").write_bytes(b"x")
    p = tmp_path / "montaj.yaml"
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\n"
        "assets: assets\nshots:\n  - {clip: intro, hold: 10f, clip_in: 1.5s}\n"
        "  - {photo: a, hold: 10f}\n"
    )
    tl = resolve(load_spec(p), p.parent)
    sc = tl.plan(0).scene
    assert isinstance(sc, Clip) and sc.shot == 0 and sc.t0 == 0
    assert isinstance(tl.plan(10).scene, Still)
    assert tl.shots[0].kind == "clip"


def test_m1_still_has_no_span(tmp_path: Path):
    tl = _mini(tmp_path, "  - {photo: a, hold: 10f}\n", ["a"])
    sc = tl.plan(0).scene
    assert isinstance(sc, Still) and sc.span is None
    assert sc.shot == 0


def test_shot0_fade_from_blank(tmp_path: Path):
    tl = _mini(tmp_path, "  - {photo: a, hold: 30f, in: {type: fade, dur: 10f}}\n", ["a"])
    sc0 = tl.plan(0).scene
    assert isinstance(sc0, Trans) and sc0.kind == "fade"
    assert isinstance(sc0.a, Blank)
    assert isinstance(sc0.b, Still) and sc0.b.shot == 0
    assert isinstance(tl.plan(9).scene, Trans) and tl.plan(9).scene.kind == "fade"
    end = tl.plan(10).scene
    assert isinstance(end, Still) and end.shot == 0


def test_frame_card_has_span(tmp_path: Path):
    tl = _mini(
        tmp_path,
        "  - {photo: a, hold: 20f, frame: {tilt: [2, -1.5]}}\n"
        "  - {photo: b, hold: 20f, in: {type: fade, dur: 10f}}\n",
        ["a", "b"],
    )
    sc0 = tl.plan(0).scene
    assert isinstance(sc0, Still) and sc0.span == (0, 30)


def test_ten_one_beat_holds_at_76bpm_do_not_drift(tmp_path: Path):
    """10 × 1b at 76 bpm / 30 fps is 237 f, not 10 × round(23.684) = 240 f."""
    from montaj.spec import round_frame, validate_spec

    shots = "\n".join(f"  - {{photo: a, hold: 1b}}" for _ in range(10))
    tl = _load(
        tmp_path,
        "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 76\n  look: warm-film\n"
        f"assets: assets\nshots:\n{shots}\n",
        ["a"],
    )
    period = 60.0 / 76.0 * 30.0
    ends = [round_frame(k * period) for k in range(1, 11)]
    assert ends[-1] == 237
    assert [s.end for s in tl.shots] == ends
    assert [s.start for s in tl.shots] == [0] + ends[:-1]
    assert tl.n_frames == 237
    parsed, *_ = validate_spec(tl.spec, [], spec_dir=tmp_path)
    holds = [parsed[f"shots[{i}].hold"] for i in range(10)]
    assert holds == [e - s for s, e in zip([0] + ends[:-1], ends)]
    assert sum(holds) == 237


def test_markers_beat_grid_phases_four_bar_holds(tmp_path: Path):
    """No video.bpm: 4b holds lock to markers beats[0] + n·4 beats."""
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "song.wav").write_bytes(b"RIFF")
    (tmp_path / "music" / "markers.json").write_text(
        '{"bpm": 80, "beats": [0.41, 1.16, 1.91, 2.66, 3.41]}'
    )
    shots = "\n".join(f"  - {{photo: a, hold: 4b}}" for _ in range(4))
    tl = _load(
        tmp_path,
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\n"
        "assets: assets\n"
        "audio:\n  track: music/song.wav\n  markers: music/markers.json\n"
        f"shots:\n{shots}\n",
        ["a"],
    )
    # round(30 · (0.41 + 3k)) for k = 1..4
    assert [s.end for s in tl.shots] == [102, 192, 282, 372]
    assert tl.n_frames == 372


def test_video_beat0_overrides_markers_beats(tmp_path: Path):
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "song.wav").write_bytes(b"RIFF")
    (tmp_path / "music" / "markers.json").write_text(
        '{"bpm": 80, "beats": [0.41, 1.16, 1.91]}'
    )
    shots = "\n".join(f"  - {{photo: a, hold: 4b}}" for _ in range(4))
    tl = _load(
        tmp_path,
        "video:\n  size: 1080x1920\n  fps: 30\n  beat0: 0.5s\n  look: warm-film\n"
        "assets: assets\n"
        "audio:\n  track: music/song.wav\n  markers: music/markers.json\n"
        f"shots:\n{shots}\n",
        ["a"],
    )
    assert [s.end for s in tl.shots] == [105, 195, 285, 375]


def test_whole_beat_hold_snaps_forward_after_off_grid_seconds(tmp_path: Path):
    tl = _load(
        tmp_path,
        "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 76\n  look: warm-film\n"
        "assets: assets\nshots:\n"
        "  - {photo: a, hold: 1.5s}\n"
        "  - {photo: a, hold: 1b}\n",
        ["a"],
    )
    from montaj.spec import round_frame

    period = 60.0 / 76.0 * 30.0
    # 1.5s = 45 f, off the 76 bpm grid; 1b then snaps forward to 3 beats.
    assert tl.shots[0].end == 45
    assert tl.shots[1].end == round_frame(3 * period)


def test_fractional_beat_hold_does_not_snap(tmp_path: Path):
    tl = _load(
        tmp_path,
        "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 80\n  beat0: 0.41s\n"
        "  look: warm-film\nassets: assets\nshots:\n  - {photo: a, hold: 1.5b}\n",
        ["a"],
    )
    from montaj.spec import round_frame

    # 1.5 · 22.5 f, no snap to beat0 + 2 beats (34.8 f).
    assert tl.shots[0].end == round_frame(1.5 * 60.0 / 80.0 * 30.0)
    assert tl.shots[0].end == 34


def test_changing_beat0_changes_affected_segment_hashes(tmp_path: Path):
    def _film(folder: Path, beat0: str | None, holds: str):
        assets = folder / "assets"
        assets.mkdir(parents=True)
        _jpeg(assets, "a")
        _jpeg(assets, "b")
        beat0_l = f"\n  beat0: {beat0}" if beat0 else ""
        p = folder / "montaj.yaml"
        p.write_text(
            "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 80\n  look: warm-film"
            f"{beat0_l}\nassets: assets\nshots:\n{holds}\n"
        )
        tl = resolve(load_spec(p), p.parent)
        shas = {"a": file_sha(assets / "a.jpg"), "b": file_sha(assets / "b.jpg")}
        return tl, [segment_hash(tl, seg, shas, "e1") for seg in tl.segments]

    beat_holds = "  - {photo: a, hold: 4b}\n  - {photo: b, hold: 4b}\n"
    a, ha = _film(tmp_path / "a", None, beat_holds)
    b, hb = _film(tmp_path / "b", "0.41s", beat_holds)
    assert a.shots[0].end != b.shots[0].end
    assert ha[0] != hb[0]
    assert ha[1] != hb[1]

    frame_holds = "  - {photo: a, hold: 10f}\n  - {photo: b, hold: 10f}\n"
    c, hc = _film(tmp_path / "c", None, frame_holds)
    d, hd = _film(tmp_path / "d", "0.41s", frame_holds)
    assert c.n_frames == d.n_frames
    assert hc != hd


def test_beat0_change_with_fixed_shot_bounds_changes_text_hash(tmp_path: Path):
    """text at: 8b moves 180 → 192.3 f; shot bounds stay 300 f; hash must change."""
    extra = (
        "text:\n"
        "  - from: 0s\n"
        "    to: 10s\n"
        "    lines:\n"
        "      - {text: x, y: 100, at: 8b}\n"
    )

    def _film(folder: Path, beat0: str | None):
        assets = folder / "assets"
        assets.mkdir(parents=True)
        _jpeg(assets, "a")
        beat0_l = f"\n  beat0: {beat0}" if beat0 else ""
        p = folder / "montaj.yaml"
        p.write_text(
            "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 80\n  look: warm-film"
            f"{beat0_l}\nassets: assets\nshots:\n  - {{photo: a, hold: 10s}}\n{extra}"
        )
        tl = resolve(load_spec(p), p.parent)
        shas = {"a": file_sha(assets / "a.jpg")}
        return tl, segment_hash(tl, tl.segments[0], shas, "e1")

    a, ha = _film(tmp_path / "a", None)
    b, hb = _film(tmp_path / "b", "0.41s")
    assert a.n_frames == b.n_frames == 300
    assert a.texts[0].lines[0].at == pytest.approx(180.0)
    assert b.texts[0].lines[0].at == pytest.approx(192.3)
    assert ha != hb
