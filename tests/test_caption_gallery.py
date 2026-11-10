"""Caption gallery and `ref` (issue #26): frame picking, cut counting on a clip with known cuts, CLI wiring."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from montaj.capgallery import GALLERY_TILE_W, pick_extremes
from montaj.cli import main
from montaj.encode import FFMPEG
from montaj.ref import RefStats, detect_cuts, sample_times
from montaj.sheet import SHEET_MAX_W


def test_pick_extremes_brightest_and_darkest() -> None:
    assert pick_extremes({10: 80.0, 20: 200.0, 30: 15.0}) == (20, 30)
    assert pick_extremes({7: 50.0}) == (7, 7)


def test_gallery_is_no_wider_than_a_sheet() -> None:
    assert GALLERY_TILE_W * 6 <= SHEET_MAX_W


def test_ref_stats_rate_and_median() -> None:
    s = RefStats(duration=10.0, cuts=(1.0, 2.0, 5.0))
    assert s.shots == 4
    assert s.per_10s == pytest.approx(3.0)
    assert s.median_shot == pytest.approx(2.0)  # shots 1, 1, 3, 5
    assert RefStats(4.0, ()).median_shot == 4.0


def test_sample_times_one_per_shot_then_thinned() -> None:
    few = sample_times(RefStats(4.0, (1.0, 2.0, 3.0)))
    assert few == pytest.approx([0.1, 1.1, 2.1, 3.1])
    many = sample_times(RefStats(100.0, tuple(float(i) for i in range(1, 40))))
    assert len(many) == 12 and many[0] == pytest.approx(0.1)


@pytest.fixture(scope="module")
def four_colour_clip(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("ref")
    parts = []
    for c in ("red", "blue", "green", "yellow"):
        p = d / f"{c}.mp4"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color={c}:s=160x120:r=30:d=1",
                        "-pix_fmt", "yuv420p", str(p)], check=True)
        parts.append(f"file {p.name}\n")
    (d / "l.txt").write_text("".join(parts))
    out = d / "four.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-i", str(d / "l.txt"), "-c", "copy", str(out)],
                   check=True)
    return out


def test_detect_cuts_counts_known_cuts(four_colour_clip: Path) -> None:
    cuts = detect_cuts(four_colour_clip, 4.0)
    assert len(cuts) == 3
    assert cuts == pytest.approx((1.0, 2.0, 3.0), abs=0.1)


def test_ref_command_reports_count_and_writes_sheet(four_colour_clip: Path, tmp_path: Path, capsys) -> None:
    assert main(["-C", str(tmp_path), "ref", str(four_colour_clip)]) == 0
    out = capsys.readouterr().out
    assert "4 shots 3 cuts cut_rate=7.5/10s" in out
    assert (tmp_path / "build" / "ref-four.jpg").is_file()


def test_ref_missing_file_is_one_err_line(tmp_path: Path, capsys) -> None:
    assert main(["-C", str(tmp_path), "ref", "nope.mp4"]) == 1
    assert capsys.readouterr().out.startswith("ERR ref:")
