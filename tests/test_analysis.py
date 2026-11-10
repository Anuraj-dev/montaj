"""Semantic ingest: analysis.json caching and `focus: face|subject|face:N` / `range: speech:N` sugar.

CPU only: a fake analyser stands in for the GPU models, so these pin the cache and resolution
contracts. Model behaviour lives in test_vision.py (gpu).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from PIL import Image

from montaj import cli
from montaj.analysis import analysis_path, analyze_assets, describe_all, load_analysis
from montaj.pipeline import _asset_shas
from montaj.project import segment_hash
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

A_FACES = [
    {"box": [100.0, 200.0, 300.0, 440.0], "eyes": [[160, 290], [240, 290]], "score": 0.99},
    {"box": [400.0, 100.0, 460.0, 180.0], "eyes": [[415, 135], [445, 135]], "score": 0.9},
]
CLIP_ROW = {
    "w": 64, "h": 48, "duration": 4.0, "fps": 30.0, "cuts": [2.0],
    "speech": [{"s": 0.25, "e": 1.0}, {"s": 2.5, "e": 3.25}],
    "silence": [{"s": 0.0, "e": 0.25}, {"s": 1.0, "e": 2.5}, {"s": 3.25, "e": 4.0}],
    "motion": [0.0, 0.1, 0.0, 0.0],
    "scenes": [
        {"s": 0.0, "e": 2.0, "faces": [{"box": [10, 10, 20, 30], "eyes": [], "score": 0.9}], "subject": [30, 20]},
        {"s": 2.0, "e": 4.0, "faces": [{"box": [40, 10, 60, 30], "eyes": [], "score": 0.9}], "subject": [50, 20]},
    ],
}


class FakeAnalyzer:
    """Counts model calls; answers from fixed rows keyed by stem."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def photo(self, path: Path) -> dict:
        self.calls.append(path.name)
        faces = A_FACES if path.stem == "a" else []
        return {"w": 600, "h": 800, "faces": faces, "subject": [310.0, 405.0], "sharpness": 12.0,
                "brightness": 0.5, "colors": ["#a0a0a0"]}

    def clip(self, path: Path) -> dict:
        self.calls.append(path.name)
        return dict(CLIP_ROW)


def _jpeg(path: Path, color: tuple[int, int, int] = (90, 120, 150)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (600, 800), color).save(path, quality=90)


def _video(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:size=64x48:rate=30:duration=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def _project(tmp: Path, shots: list[dict], *, analyse: bool = True, clip: bool = False) -> Path:
    assets = tmp / "assets"
    _jpeg(assets / "a.jpg")
    _jpeg(assets / "b.jpg", (20, 30, 40))
    if clip:
        _video(assets / "talk.mp4")
    if analyse:
        analyze_assets(assets, lambda: FakeAnalyzer([]))
    spec = {"video": {"size": "1080x1920", "fps": 30, "look": "warm-film"}, "assets": "assets", "shots": shots}
    p = tmp / "montaj.yaml"
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    return p


def _errors(p: Path) -> list[str]:
    with pytest.raises(SpecError) as exc:
        resolve(load_spec(p), p.parent)
    return exc.value.errors


def test_focus_sugar_resolves_to_face_and_subject_points(tmp_path: Path) -> None:
    p = _project(tmp_path, [
        {"photo": "a", "hold": "1s", "focus": "face", "zoom": 1.3},
        {"photo": "a", "hold": "1s", "focus": "face:2"},
        {"photo": "b", "hold": "1s", "focus": "subject"},
        {"clip": "talk", "hold": "1s", "camera": [{"t": "0s", "zoom": 1.0, "focus": "subject"},
                                                  {"t": "1s", "zoom": 1.4, "focus": "face"}]},
    ], clip=True)
    tl = resolve(load_spec(p), tmp_path)
    assert tl.shots[0].spec.focus == (200.0, 320.0)
    assert tl.shots[1].spec.focus == (430.0, 140.0)
    assert tl.shots[2].spec.focus == (310.0, 405.0)
    assert [k.focus for k in tl.shots[3].spec.camera] == [(30.0, 20.0), (15.0, 20.0)]


def test_sugar_hashes_like_the_numbers_it_resolves_to(tmp_path: Path) -> None:
    sugar = _project(tmp_path, [{"photo": "a", "hold": "1s", "focus": "face", "zoom": 1.3}])
    spec = load_spec(sugar)
    tl = resolve(spec, tmp_path)
    h_sugar = segment_hash(tl, tl.segments[0], _asset_shas(spec, tmp_path), "e1")
    sugar.write_text(sugar.read_text().replace("focus: face", "focus: [200, 320]"))
    spec = load_spec(sugar)
    tl = resolve(spec, tmp_path)
    assert segment_hash(tl, tl.segments[0], _asset_shas(spec, tmp_path), "e1") == h_sugar


def test_missing_face_is_one_err_naming_the_stem(tmp_path: Path) -> None:
    p = _project(tmp_path, [{"photo": "a", "hold": "1s"}, {"photo": "b", "hold": "1s", "focus": "face"}])
    assert _errors(p) == ["shots[1].focus: no face in b (use subject or [x, y])"]


def test_face_n_past_the_last_face_lists_what_exists(tmp_path: Path) -> None:
    p = _project(tmp_path, [{"photo": "a", "hold": "1s", "focus": "face:3"}])
    assert _errors(p) == ["shots[0].focus: no face 3 in a (have: face:1, face:2)"]


def test_missing_or_stale_analysis_asks_for_ingest(tmp_path: Path) -> None:
    p = _project(tmp_path, [{"photo": "a", "hold": "1s", "focus": "face"}], analyse=False)
    assert _errors(p) == ["shots[0].focus: no analysis for a.jpg (run montaj ingest)"]
    analyze_assets(tmp_path / "assets", lambda: FakeAnalyzer([]))
    _jpeg(tmp_path / "assets" / "a.jpg", (1, 2, 3))  # new bytes, same name
    assert _errors(p) == ["shots[0].focus: analysis of a.jpg is stale (run montaj ingest)"]


def test_unknown_sugar_is_a_plain_field_error(tmp_path: Path) -> None:
    p = _project(tmp_path, [{"photo": "a", "hold": "1s", "focus": "nose"}])
    with pytest.raises(SpecError) as exc:
        load_spec(p)
    assert len(exc.value.errors) == 1
    assert exc.value.errors[0].startswith("shots[0].focus: ") and "face:N" in exc.value.errors[0]


def test_range_speech_and_clip_face_follow_the_scene(tmp_path: Path) -> None:
    p = _project(tmp_path, [{"clip": "talk", "range": "speech:2", "focus": "face"},
                            {"clip": "talk", "range": "speech:1", "focus": "subject"}], clip=True)
    tl = resolve(load_spec(p), tmp_path)
    assert tl.shots[0].spec.range == ("2.5s", "3.25s")
    assert tl.shots[0].spec.focus == (50.0, 20.0)  # speech 2 starts in the second scene
    assert (tl.shots[0].start, tl.shots[0].end) == (0, 23)  # 0.75 s at 30 fps rounds half up
    assert tl.shots[1].spec.range == ("0.25s", "1s")
    assert tl.shots[1].spec.focus == (30.0, 20.0)


def test_range_speech_past_the_last_span(tmp_path: Path) -> None:
    p = _project(tmp_path, [{"clip": "talk", "range": "speech:3"}], clip=True)
    assert _errors(p) == ["shots[0].range: no speech span 3 in talk (have: 2)"]


def test_reingest_of_unchanged_files_runs_no_model(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _jpeg(assets / "a.jpg")
    _video(assets / "talk.mp4")
    calls: list[str] = []
    made: list[int] = []

    def make() -> FakeAnalyzer:
        made.append(1)
        return FakeAnalyzer(calls)

    first = analyze_assets(assets, make)
    assert (first.analyzed, first.cached, sorted(calls), len(made)) == (2, 0, ["a.jpg", "talk.mp4"], 1)
    rows = load_analysis(assets)
    rows["talk.mp4"]["words"] = [{"w": "hi", "s": 0.3, "e": 0.5}]  # written later by transcribe
    analysis_path(assets).write_text(json.dumps(rows))

    _jpeg(assets / "a.jpg")  # same bytes rewritten: new mtime, same sha
    again = analyze_assets(assets, make)
    assert (again.analyzed, again.cached, len(calls), len(made)) == (0, 2, 2, 1)
    assert load_analysis(assets)["talk.mp4"]["words"][0]["w"] == "hi"

    _jpeg(assets / "a.jpg", (9, 9, 9))
    third = analyze_assets(assets, make)
    assert (third.analyzed, third.cached, calls[-1], len(made)) == (1, 1, "a.jpg", 2)
    (assets / "a.jpg").unlink()
    assert list(analyze_assets(assets, make).rows) == ["talk.mp4"]


def test_cli_ingest_analyses_then_caches_and_lists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "montaj.yaml").write_text("assets: assets\n")
    photos = tmp_path / "photos"
    _jpeg(photos / "a.jpg")
    _jpeg(photos / "b.png", (20, 30, 40))
    calls: list[str] = []
    monkeypatch.setattr(cli, "_make_analyzer", lambda: FakeAnalyzer(calls))

    assert cli.main(["-C", str(proj), "ingest", "--photos", str(photos)]) == 0
    assert capsys.readouterr().out.strip() == (
        "OK 2 photos sheet=build/ingest-sheet.jpg analyzed=2 cached=0 -> assets/analysis.json")
    assert cli.main(["-C", str(proj), "ingest", "--photos", str(photos)]) == 0
    assert capsys.readouterr().out.strip().endswith("analyzed=0 cached=2 -> assets/analysis.json")
    assert cli.main(["-C", str(proj), "ingest"]) == 0
    assert capsys.readouterr().out.strip() == "OK analyzed=0 cached=2 -> assets/analysis.json"
    assert len(calls) == 2

    assert cli.main(["-C", str(proj), "ingest", "--list"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("a.jpg photo 600x800 faces=2 [200, 320] [430, 140] subject=[310, 405]")
    assert lines[1].startswith("b.jpg photo 600x800 faces=0 subject=")
    assert lines[2] == "OK 2 assets"
    assert describe_all(proj / "assets") == lines[:2]
