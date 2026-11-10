"""#63: the luma-jump warning knows a scene from a take, and a designed jump can be declared."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from montaj.project import _shot_payload
from montaj.scopes import ShotScope, _recordings, warnings
from montaj.spec import Shot, SpecError, load_spec
from montaj.timeline import ResolvedExpect, resolve

FPS = 30


def _s(i: int, p50: float) -> ShotScope:
    return ShotScope(i, p1=0.02, p50=p50, p99=0.9, clip_hi=0.0, clip_lo=0.0, sat_mean=0.3, cast=0.0, tint=0.0)


def _tl(*shots: Shot, expects: list[ResolvedExpect] | None = None) -> SimpleNamespace:
    """Back-to-back 2 s shots built from real `Shot` models."""
    spec = SimpleNamespace(video=SimpleNamespace(fps=FPS, bpm=None))
    rs = [SimpleNamespace(start=2 * FPS * k, end=2 * FPS * (k + 1), spec=sh) for k, sh in enumerate(shots)]
    return SimpleNamespace(shots=rs, fps=FPS, spec=spec, check_expects=expects or [])


def _clip(stem: str, at: str, **kw) -> Shot:
    return Shot(clip=stem, clip_in=at, hold="2s", **kw)


LIGHT_DARK = [_s(0, 0.30), _s(1, 0.75)]  # +0.45: far past LUMA_JUMP


def test_screen_recording_brightness_is_content_not_exposure() -> None:
    tl = _tl(_clip("takeA", "10s"), _clip("takeA", "12s"))
    assert warnings(LIGHT_DARK, tl, {}, frozenset({"takeA"})) == []


def test_recordings_are_the_stems_with_an_events_log(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "takeA.events.json").write_text("[]")
    (tmp_path / "assets" / "cam.mp4").write_bytes(b"")
    tl = SimpleNamespace(spec=SimpleNamespace(assets="assets"))
    assert _recordings(tmp_path, tl) == frozenset({"takeA"})


def test_one_camera_take_seconds_apart_still_warns_with_the_cut_time() -> None:
    tl = _tl(_clip("cam", "10s"), _clip("cam", "15s"))
    assert warnings(LIGHT_DARK, tl, {"cam": {"cuts": []}}) == [
        "WARN scopes shots 0→1 (2.0s): p50 luma jumps +0.45 within one scene "
        "(montaj match --to 0; designed? check.expect luma or match: false)"
    ]


def test_a_detected_cut_between_the_windows_is_a_new_scene() -> None:
    tl = _tl(_clip("cam", "10s"), _clip("cam", "15s"))
    assert warnings(LIGHT_DARK, tl, {"cam": {"cuts": [13.5]}}) == []
    assert warnings(LIGHT_DARK, tl, {"cam": {"cuts": [40.0]}}) != []  # a cut elsewhere in the file


def test_far_apart_moments_of_one_take_are_different_scenes() -> None:
    tl = _tl(_clip("cam", "10s"), _clip("cam", "40s"))  # 28 s between windows > SCENE_GAP_S
    assert warnings(LIGHT_DARK, tl, {}) == []


def test_match_false_accepts_the_jump() -> None:
    tl = _tl(_clip("cam", "10s"), _clip("cam", "15s", match=False))
    assert warnings(LIGHT_DARK, tl, {}) == [
        "INFO scopes shots 0→1 (2.0s) luma jump ignored as intended: match: false on shot 1"
    ]


def test_expect_luma_accepts_the_jump_and_reports_a_stale_span() -> None:
    expects = [ResolvedExpect("luma", 1.5, 2.5, "theme flips dark"), ResolvedExpect("luma", 8.0, 9.0, None),
               ResolvedExpect("caption", 0.0, 9.0, "other kind")]
    tl = _tl(_clip("cam", "10s"), _clip("cam", "15s"), expects=expects)
    assert warnings(LIGHT_DARK, tl, {}) == [
        "INFO scopes shots 0→1 (2.0s) luma jump ignored as intended: theme flips dark",
        "INFO expect luma 8-9s not found",
    ]


def _project(tmp: Path, extra: str) -> Path:
    (tmp / "assets").mkdir()
    Image.new("RGB", (8, 8), (180, 90, 40)).save(tmp / "assets" / "a.jpg", "JPEG")
    p = tmp / "montaj.yaml"
    p.write_text("video:\n  size: 1080x1920\n  fps: 30\n  look: clean\nassets: assets\nshots:\n"
                 "  - {photo: a, hold: 2s}\n  - {photo: a, hold: 2s}\n" + extra)
    return p


def test_check_expect_luma_resolves_through_the_spec(tmp_path: Path) -> None:
    p = _project(tmp_path, "check:\n  expect:\n    - {luma: [1.5s, 2.5s], why: lamp switched on}\n")
    tl = resolve(load_spec(p), p.parent)
    (e,) = tl.check_expects
    assert (e.kind, e.start, e.end, e.why) == ("luma", 1.5, 2.5, "lamp switched on")
    assert warnings(LIGHT_DARK, tl) == [
        "INFO scopes shots 0→1 (2.0s) luma jump ignored as intended: lamp switched on"
    ]


@pytest.mark.parametrize("item", ["{luma: [1s, 2s], black: [1s, 2s]}", "{luma: [2s, 1s]}"])
def test_check_expect_luma_rejects_bad_items(tmp_path: Path, item: str) -> None:
    with pytest.raises(SpecError):
        load_spec(_project(tmp_path, f"check:\n  expect:\n    - {item}\n"))


def test_match_field_is_false_only_and_never_reaches_the_segment_hash() -> None:
    with pytest.raises(ValueError):
        Shot(photo="a", hold="2s", match=True)
    plain, marked = Shot(photo="a", hold="2s"), Shot(photo="a", hold="2s", match=False)
    assert "match" not in plain.model_dump()
    assert _shot_payload(marked) == _shot_payload(plain)
