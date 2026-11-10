"""Motion sheets: frame-index selection for strips, transitions and diffs, and the size bound."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from montaj.cli import main
from montaj.sheet import (
    SHEET_MAX_W, STRIP_MAX_FRAMES, TRANS_MAX_ROWS, Row, compose_rows, step_frames, strip_indices,
    strip_row, transition_rows,
)
from montaj.spec import load_spec
from montaj.timeline import resolve
from tests.test_recipes import ROOT, _short_project


def test_strip_indices_centre_clamp_and_step() -> None:
    assert strip_indices(100, 8, 1, 1000) == [97, 98, 99, 100, 101, 102, 103, 104]
    assert strip_indices(100, 4, 3, 1000) == [96, 99, 102, 105]
    assert strip_indices(0, 8, 1, 1000) == list(range(8))  # shifted in, still 8 distinct frames
    assert strip_indices(999, 8, 1, 1000) == list(range(992, 1000))
    assert len(strip_indices(50, 99, 1, 1000)) == STRIP_MAX_FRAMES


def test_step_frames_units() -> None:
    assert step_frames("1f", 30) == 1
    assert step_frames("2f", 30) == 2
    assert step_frames("0.1s", 30) == 3
    assert step_frames("0.001", 30) == 1
    with pytest.raises(ValueError):
        step_frames("0f", 30)
    with pytest.raises(ValueError):
        step_frames("abc", 30)


@pytest.fixture
def short_tl(tmp_path: Path):
    proj = _short_project(tmp_path)
    return resolve(load_spec(proj / "montaj.yaml"), proj)


def test_transition_rows_one_per_transition_with_five_stops(short_tl) -> None:
    rows, total = transition_rows(short_tl, 30)
    kinds = [w[2].kind for w in short_tl._windows]
    assert total == len(kinds) and len(rows) == min(total, TRANS_MAX_ROWS)
    for row in rows:
        idx = [i for i, _ in row.tiles]
        assert idx == sorted(idx) and len(idx) == 5
        assert all(0 <= i < short_tl.n_frames for i in idx)
        assert row.label.split()[1] in kinds
    starts = [r.tiles[0][0] for r in rows]
    assert starts == sorted(starts)
    # the first row is the swirl into shot 1: starts at that shot, ends inside its window
    swirl = rows[0]
    assert swirl.label.startswith("#1 swirl") and swirl.tiles[0][0] == short_tl.shots[1].start
    assert swirl.tiles[-1][0] == swirl.tiles[0][0] + 21  # dur 22f, last frame inside


def test_strip_row_labels_shot_and_transition(short_tl) -> None:
    f = short_tl.shots[1].start + 5
    rows = strip_row(short_tl, f, 8, 1, short_tl.n_frames, 30)
    assert len(rows) == 1 and len(rows[0].tiles) == 8
    assert "shot #1" in rows[0].label and "trans swirl" in rows[0].label


def test_compose_rows_bounded_and_diff() -> None:
    rows = [Row(f"#{k} fade", tuple((k * 10 + j, f"c{j}") for j in range(5))) for k in range(TRANS_MAX_ROWS)]
    images = {i: Image.new("RGB", (1080, 1920), (i * 3 % 255, 40, 90)) for r in rows for i, _ in r.tiles}
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = compose_rows(rows, images, Path(d) / "a.jpg")
        with Image.open(out) as im:
            assert max(im.size) <= SHEET_MAX_W
        base = {i: Image.new("RGB", (1080, 1920), (i * 3 % 255, 40, 90)) for i in images}
        base[0] = Image.new("RGB", (1080, 1920), (255, 255, 255))
        out2 = compose_rows(rows, images, Path(d) / "b.jpg", base=base)
        with Image.open(out2) as im:
            assert max(im.size) <= SHEET_MAX_W


def _cli(argv, capsys):
    code = main(argv)
    c = capsys.readouterr()
    return code, c.out, c.err


def test_cli_transitions_and_diff(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    proj = _short_project(tmp_path)
    tl = resolve(load_spec(proj / "montaj.yaml"), proj)
    video = proj / "out" / "preview.mp4"
    video.parent.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=270x480:rate=30:duration={tl.n_frames / 30}", "-pix_fmt", "yuv420p", str(video)],
                   check=True)
    code, out, err = _cli(["-C", str(proj), "sheet", "--diff"], capsys)
    assert code != 0 and "baseline" in out  # no baseline yet
    code, out, err = _cli(["-C", str(proj), "sheet", "--transitions"], capsys)
    assert code == 0 and out.startswith("OK build/sheet-transitions.jpg 5 rows") and err == "", out
    with Image.open(proj / "build" / "sheet-transitions.jpg") as im:
        assert max(im.size) <= SHEET_MAX_W
    code, out, _ = _cli(["-C", str(proj), "sheet", "--diff"], capsys)
    assert code == 0 and out.startswith("OK build/sheet-diff.jpg"), out
    code, out, _ = _cli(["-C", str(proj), "sheet", "--strip", "1.2s", "--frames", "6", "--step", "2f"], capsys)
    assert code == 0 and out.strip() == "OK build/sheet-strip.jpg 1 rows 6 frames", out
