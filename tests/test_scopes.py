"""#45: scope numbers, the comment-preserving grade writer, the match solver and check warnings."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from montaj import scopes
from montaj.cli import main
from montaj.grade import Grade
from montaj.render.color import grade as apply_grade
from montaj.render.color import to_display, to_linear
from montaj.scopes import ShotScope, hue_deg, measure, on_skin_line, solve, warnings, write_grades


def _flat(rgb: tuple[float, float, float], h: int = 40, w: int = 30) -> torch.Tensor:
    return torch.tensor(rgb).view(3, 1, 1).expand(3, h, w).clone()


def _ramp(h: int = 64, w: int = 96) -> torch.Tensor:
    """A neutral grey ramp (white balance) with coloured bands (saturation), so each reads apart."""
    x = torch.linspace(0.05, 0.95, w).view(1, 1, w).expand(3, h, w).clone()
    for k, rgb in enumerate(((0.7, 0.2, 0.15), (0.2, 0.6, 0.25), (0.15, 0.3, 0.75))):
        x[:, 8 * k:8 * k + 6] = torch.tensor(rgb).view(3, 1, 1)
    return x.clamp(0, 1)


def test_numbers_on_flat_fields() -> None:
    grey = measure([_flat((0.5, 0.5, 0.5))], 0)
    assert grey.p1 == grey.p50 == grey.p99 == pytest.approx(0.5, abs=1e-3)
    assert grey.clip_hi == grey.clip_lo == 0 and grey.sat_mean == pytest.approx(0, abs=1e-4)
    assert grey.cast == pytest.approx(0, abs=1e-4) and grey.tint == pytest.approx(0, abs=1e-4)
    assert measure([_flat((1.0, 1.0, 1.0))], 0).clip_hi == 1.0
    assert measure([_flat((0.0, 0.0, 0.0))], 0).clip_lo == 1.0
    assert measure([_flat((0.6, 0.5, 0.45))], 0).cast > 0  # warm
    assert measure([_flat((0.45, 0.5, 0.6))], 0).cast < 0  # cool
    assert measure([_flat((0.5, 0.58, 0.5))], 0).tint > 0  # green
    # A saturated primary is colour, not a blown highlight.
    assert measure([_flat((1.0, 0.0, 0.0))], 0).clip_hi == 0


def test_skin_hue_from_a_face_box() -> None:
    img = _flat((0.2, 0.2, 0.2), 100, 100)
    img[:, 20:80, 20:80] = torch.tensor([0.80, 0.60, 0.48]).view(3, 1, 1)
    s = measure([img], 3, [(20, 20, 80, 80)])
    assert s.skin_hue == pytest.approx(hue_deg((0.80, 0.60, 0.48)), abs=0.5) and on_skin_line(s.skin_hue)
    green = img.clone()
    green[:, 20:80, 20:80] = torch.tensor([0.5, 0.75, 0.45]).view(3, 1, 1)
    assert not on_skin_line(measure([green], 3, [(20, 20, 80, 80)]).skin_hue)
    assert measure([img], 3, [None]).skin_hue is None
    assert measure([img], 3, [(0, 0, 4, 4)]).skin_hue is None  # too few pixels to say


def test_solver_matches_p50_cast_and_saturation() -> None:
    ref = _ramp()
    target = measure([ref], 0)
    off = to_display(apply_grade(to_linear(ref), Grade(exposure=-1.3, temp=900, saturation=0.6)))
    lin = to_linear(off.reshape(3, 1, -1))
    g = solve(lin, Grade(), target)
    got = measure([to_display(apply_grade(lin, g)).reshape(3, 64, 96)], 0)
    assert abs(got.p50 - target.p50) <= 0.02
    assert abs(got.cast - target.cast) <= 0.01
    assert abs(got.sat_mean - target.sat_mean) <= 0.03
    assert g.exposure > 1 and g.temp < 0 and g.saturation > 1
    assert set(Grade().model_dump()) >= set(scopes.MATCH_FIELDS)


SPEC = """\
video: {size: 360x640, fps: 25}   # keep
assets: assets
grades:
  night: {temp: -500}
# shots below
shots:
  - {clip: c, hold: 1s}   # the reference
  - {clip: c, hold: 1s}   # flow, no grade
  - {clip: c, hold: 1s, grade: night}   # named
  - clip: c   # block, no grade
    hold: 1s
  - clip: c
    grade:
      exposure: -1.0   # too dark
      contrast: 1.1
    hold: 1s
  - {clip: c, hold: 1s, grade: {exposure: 2, contrast: 1.2}}
"""


def test_write_grades_keeps_comments_and_layout() -> None:
    g = Grade(exposure=0.25, temp=120, tint=-2.0, saturation=1.05)
    named = Grade(temp=-500).model_copy(update={"exposure": 0.25, "temp": 120, "tint": -2.0, "saturation": 1.05})
    keep = Grade(contrast=1.2).model_copy(update={"exposure": 0.25, "temp": 120, "tint": -2.0, "saturation": 1.05})
    out = write_grades(SPEC, {1: g, 2: named, 3: g, 4: keep.model_copy(update={"contrast": 1.1}), 5: keep}, 6)
    for comment in ("# keep", "# shots below", "# the reference", "# flow, no grade", "# named",
                    "# block, no grade", "# too dark"):
        assert comment in out
    assert out.splitlines()[0] == "video: {size: 360x640, fps: 25}   # keep"
    data = yaml.safe_load(out)
    want = {"exposure": 0.25, "temp": 120, "tint": -2.0, "saturation": 1.05}
    shots = data["shots"]
    assert "grade" not in shots[0]
    assert shots[1]["grade"] == want and shots[3]["grade"] == want
    assert shots[2]["grade"] == want  # named becomes inline; temp -500 is replaced by the solved temp
    assert shots[4]["grade"] == {**want, "contrast": 1.1} and shots[4]["hold"] == "1s"
    assert shots[5]["grade"] == {**want, "contrast": 1.2}
    assert "      exposure: 0.25   # too dark\n" in out  # block value edited in place


def test_write_grades_refuses_what_it_cannot_index() -> None:
    with pytest.raises(ValueError, match="lists 6 shots but the film has 7"):
        write_grades(SPEC, {1: Grade()}, 7)
    with pytest.raises(ValueError, match="component"):
        write_grades("shots:\n  - {use: intro}\n", {0: Grade()})


def _tl(*stems: str | None) -> SimpleNamespace:
    """Back-to-back 1 s shots that read each clip from its start (one take, one moment)."""
    spec = SimpleNamespace(video=SimpleNamespace(fps=25, bpm=None))
    shots = [SimpleNamespace(start=25 * k, end=25 * (k + 1), spec=SimpleNamespace(
        clip=s, photo=None, range=None, clip_in=None, speed=None)) for k, s in enumerate(stems)]
    return SimpleNamespace(shots=shots, fps=25, spec=spec, check_expects=[])


def _s(i: int, p50: float, **kw) -> ShotScope:
    base = dict(p1=0.02, p99=0.9, clip_hi=0.0, clip_lo=0.0, sat_mean=0.3, cast=0.0, tint=0.0)
    return ShotScope(i, p50=p50, **{**base, **kw})


def test_check_warnings() -> None:
    tl = _tl("a", "a", "b")
    assert warnings([_s(0, 0.4), _s(1, 0.42), _s(2, 0.8)], tl) == []  # the big jump is a new scene
    lines = warnings([_s(0, 0.4, clip_hi=0.05), _s(1, 0.55, skin_hue=50.0), _s(2, 0.5, skin_hue=22.0)], tl)
    assert lines == [
        "WARN scopes shot 0: clip_hi 5.0% > 2% (lower grade.exposure or gain)",
        "WARN scopes shot 1: skin hue 50° off the skin line (10–34°) (adjust grade.temp/tint)",
        "WARN scopes shots 0→1 (1.0s): p50 luma jumps +0.15 within one scene "
        "(montaj match --to 0; designed? check.expect luma or match: false)",
    ]


def test_match_needs_a_target(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-C", str(tmp_path), "match"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("ERR") and "--to" in out


# --- GPU: the real renderer ----------------------------------------------------------------------

def _project(tmp: Path) -> Path:
    (tmp / "assets").mkdir(parents=True)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc2=size=320x240:rate=25:duration=2", "-pix_fmt", "yuv420p",
                    str(tmp / "assets" / "c.mp4")], check=True)
    (tmp / "montaj.yaml").write_text(
        "video:\n  size: 360x640\n  fps: 25\n  look: clean\nassets: assets\n"
        "# one clip, three exposures\nshots:\n"
        "  - {clip: c, hold: 1s}   # reference\n"
        "  - {clip: c, hold: 1s, grade: {exposure: 1.2, temp: 600}}   # hot\n"
        "  - clip: c   # dark\n    hold: 1s\n    grade:\n      exposure: -1.1\n      saturation: 0.7\n")
    return tmp


def _scopes(root: Path, capsys) -> dict:
    assert main(["-C", str(root), "--json", "scopes"]) == 0
    return {s["shot"]: s for s in json.loads(capsys.readouterr().out)["shots"]}


@pytest.mark.gpu
def test_match_brings_mis_exposed_copies_within_002(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("MONTAJ_GPU_STRICT", "1")
    root = _project(tmp_path)
    before = _scopes(root, capsys)
    assert abs(before[1]["p50"] - before[0]["p50"]) > 0.1 and abs(before[2]["p50"] - before[0]["p50"]) > 0.1
    assert main(["-C", str(root), "match", "--to", "0"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "OK montaj.yaml: matched 2 shots to shot 0" and out[0].startswith("shot 1  exposure +1.20→")
    text = (root / "montaj.yaml").read_text()
    assert "# reference" in text and "# hot" in text and "# dark" in text
    after = _scopes(root, capsys)
    for i in (1, 2):
        assert abs(after[i]["p50"] - after[0]["p50"]) <= 0.02, after[i]
        assert abs(after[i]["cast"] - after[0]["cast"]) <= 0.01
    assert main(["-C", str(root), "scopes", "--image"]) == 0
    assert capsys.readouterr().out.strip().endswith(f"sheet {root / 'build' / 'scopes.jpg'}")
    assert (root / "build" / "scopes.jpg").is_file()


@pytest.mark.gpu
def test_inspect_json_carries_scopes(tmp_path: Path, capsys) -> None:
    root = _project(tmp_path)
    assert main(["-C", str(root), "--json", "inspect", "--at", "0.5s"]) == 0
    frame = json.loads(capsys.readouterr().out)
    assert frame["scopes"]["shot"] == 0 and 0 < frame["scopes"]["p50"] < 1
