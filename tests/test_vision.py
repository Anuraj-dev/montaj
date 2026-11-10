"""Models and measurements behind `montaj ingest` analysis (vision.py)."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from montaj.analysis import analyze_assets
from montaj.vision import AUDIO_RATE, speech_spans

ORACLE_SRC = Path.home() / "Projects/media/montaj-projects/_oracle/v2/src"


def test_energy_vad_finds_loud_spans_and_quiet_gaps() -> None:
    t = np.arange(int(AUDIO_RATE * 4)) / AUDIO_RATE
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    on = ((t >= 0.5) & (t < 1.5)) | ((t >= 2.5) & (t < 3.0))
    x = np.where(on, tone, 0.0005 * np.sin(2 * np.pi * 50 * t))
    speech, silence = speech_spans(x.astype(np.float32), AUDIO_RATE, 4.0)
    assert speech == [{"s": 0.5, "e": 1.5}, {"s": 2.5, "e": 3.0}]
    assert silence == [{"s": 0.0, "e": 0.5}, {"s": 1.5, "e": 2.5}, {"s": 3.0, "e": 4.0}]


def _two_scene_clip(path: Path) -> None:
    """Red 2 s then blue 2 s; a 440 Hz tone from 0.5 s to 1.5 s, silence otherwise."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=red:size=320x240:rate=30:duration=2",
         "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=30:duration=2",
         "-f", "lavfi", "-i", "aevalsrc='if(between(t,0.5,1.5),0.5*sin(2*PI*440*t),0)':s=48000:d=4",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True,
    )


@pytest.mark.gpu
def test_clip_analysis_finds_cut_speech_and_scenes(tmp_path: Path) -> None:
    from montaj.vision import GpuAnalyzer

    clip = tmp_path / "talk.mp4"
    _two_scene_clip(clip)
    row = GpuAnalyzer().clip(clip)
    assert (row["w"], row["h"]) == (320, 240)
    assert row["cuts"] == [2.0]
    assert [(s["s"], s["e"]) for s in row["scenes"]] == [(0.0, 2.0), (2.0, row["duration"])]
    assert all(s["faces"] == [] for s in row["scenes"])
    assert len(row["speech"]) == 1
    span = row["speech"][0]
    assert abs(span["s"] - 0.5) <= 0.06 and abs(span["e"] - 1.5) <= 0.06
    assert len(row["motion"]) == 4 and max(row["motion"]) > 0  # the cut second moves


@pytest.mark.gpu
def test_flat_photo_has_no_face(tmp_path: Path) -> None:
    from montaj.vision import GpuAnalyzer

    p = tmp_path / "flat.jpg"
    Image.new("RGB", (640, 480), (128, 128, 128)).save(p)
    row = GpuAnalyzer().photo(p)
    assert row["faces"] == [] and row["sharpness"] == 0
    assert row["colors"][0] == "#808080"


@pytest.mark.gpu
@pytest.mark.skipif(not (ORACLE_SRC / "09.jpg").is_file(), reason="oracle photos not on this machine")
def test_portrait_face_and_cached_reingest(tmp_path: Path) -> None:
    from montaj.vision import GpuAnalyzer

    assets = tmp_path / "assets"
    assets.mkdir()
    for stem in ("09", "37", "16"):  # 16 is a hand-covered face: no confident detection
        shutil.copy(ORACLE_SRC / f"{stem}.jpg", assets / f"{stem}.jpg")
    made: list[GpuAnalyzer] = []

    def make() -> GpuAnalyzer:
        made.append(GpuAnalyzer())
        return made[-1]

    first = analyze_assets(assets, make)
    rows = first.rows
    assert first.analyzed == 3
    for stem in ("09", "37"):
        face = rows[f"{stem}.jpg"]["faces"][0]
        x0, y0, x1, y1 = face["box"]
        assert face["score"] >= 0.8 and 0 <= x0 < x1 <= rows[f"{stem}.jpg"]["w"] and 0 <= y0 < y1
        assert len(face["eyes"]) == 2
    assert rows["16.jpg"]["faces"] == []
    assert made[0].peak["faces"] < 512 * 2**20  # small model: well under the 8 GB card

    t0 = time.perf_counter()
    again = analyze_assets(assets, make)
    assert (again.analyzed, again.cached, len(made)) == (0, 3, 1)
    assert time.perf_counter() - t0 < 0.5
