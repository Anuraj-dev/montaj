"""#64: the clipped-mix line names when, which sources and a gain that clears it; a composed bed leaves headroom."""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from montaj.audio import mix
from montaj.audio.mix import SR, MixPlan, Placed
from montaj.cli import main
from montaj.emphasis import DEFAULT_SFX_GAIN
from montaj.spec import load_spec
from montaj.timeline import resolve

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

FIX = re.compile(r"lower (.+?) by ([\d.]+) dB\)$")


def tone(sec: float, amp: float, hz: float = 220.0) -> np.ndarray:
    x = amp * np.sin(2 * np.pi * hz * np.arange(round(sec * SR)) / SR)
    return np.stack([x, x], axis=1).astype(np.float32)


def float_wav(path: Path, x: np.ndarray) -> str:
    mix.write_wav(path, x)  # float32: decode reads back exactly what was written
    return str(path)


def peak_line(stats: mix.MixStats) -> str:
    lines = mix.check_lines(json.loads(json.dumps(dataclasses.asdict(stats))))  # as `montaj check` reads it
    assert len(lines) == 2 and lines[1].startswith("INFO clip "), lines
    return lines[1]


SPEC = """\
video: {{size: 64x64, fps: 30, bpm: 120, look: warm-film}}
assets: assets
audio:
  track: music/song.wav
  clips:
    - {{src: "synth:hit", at: [1s, 2.5s], gain: {hit}}}
    - {{src: sfx/whoosh, at: 1s, gain: -6dB}}
shots:
  - {{blank: true, hold: 4s}}
"""


def spec_mix(root: Path, hit_gain: str) -> mix.MixStats:
    (root / "montaj.yaml").write_text(SPEC.format(hit=hit_gain))
    spec = load_spec(root / "montaj.yaml")
    tl = resolve(spec, root)
    plan = mix.plan_audio(tl.spec, root, tl.markers, 30, 120.0, 0.0, [])
    return mix.mix_audio(plan)[1]


def test_clipped_line_names_time_sources_and_a_clip_gain_that_clears_it_through_the_spec(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    float_wav(tmp_path / "music" / "song.wav", tone(4.0, 0.5))
    float_wav(tmp_path / "sfx" / "whoosh.wav", tone(0.3, 0.6, 800.0))
    stats = spec_mix(tmp_path, "0dB")
    assert stats.clipped
    line = peak_line(stats)
    assert re.search(rf" at {stats.peak_s:.2f}s = ", line) and 1.0 <= stats.peak_s < 1.3
    labels = [n for n, _ in stats.peak_parts]
    assert {"music", "synth:hit"} <= set(labels), line  # every source sounding at the peak sample
    assert "whole mix scaled" in line and "clean" in line
    who, cut = FIX.search(line).groups()
    assert who == "synth:hit gain", line
    # One round: write the printed cut into the spec's `gain`, mix again, nothing left to scale.
    again = spec_mix(tmp_path, f"-{cut}dB")
    assert not again.clipped and again.peak_dbfs <= 20 * np.log10(mix.PEAK_CEILING)
    assert mix.check_lines(dataclasses.asdict(again))[1:] == []


def test_a_bed_driven_peak_asks_for_a_quieter_music_bed_and_that_clears_it(tmp_path: Path) -> None:
    music = tone(2.0, 0.96)
    tick = Placed(None, float_wav(tmp_path / "t.wav", tone(0.2, 0.1)), 0, 0.0, None, 0.5, 0.0, 0.0)
    stats = mix.mix_audio(MixPlan(float_wav(tmp_path / "m.wav", music), clips=(tick,)))[1]
    who, cut = FIX.search(peak_line(stats)).groups()
    assert who == "the music bed"  # cutting the tick alone would take ~20 dB; the bed needs < 1 dB
    quieter = float_wav(tmp_path / "m2.wav", music * np.float32(10 ** (-float(cut) / 20)))
    assert not mix.mix_audio(MixPlan(quieter, clips=(tick,)))[1].clipped


def test_no_single_source_can_clear_it_so_every_source_is_lowered(tmp_path: Path) -> None:
    a = float_wav(tmp_path / "a.wav", tone(1.0, 0.6))

    def plan(db: float) -> MixPlan:
        bed = float_wav(tmp_path / f"m{db}.wav", tone(1.0, 0.6) * np.float32(10 ** (db / 20)))
        return MixPlan(bed, clips=(Placed("x", a, 0, 0.0, None, 0.0, db, 0.0),
                                   Placed("y", a, 0, 0.0, None, 0.0, db, 0.0)))

    stats = mix.mix_audio(plan(0.0))[1]
    who, cut = FIX.search(peak_line(stats)).groups()
    assert who == "music and every clip"  # any two of the three already clip
    assert not mix.mix_audio(plan(-float(cut)))[1].clipped


def test_stats_from_before_the_peak_detail_still_report_and_are_mixed_again(tmp_path: Path) -> None:
    old = {"clips": 1, "duck_db": 0.0, "peak_dbfs": 2.0, "clipped": True}
    assert mix.check_lines(old)[1] == "INFO clip mix peak +2.0 dBFS; whole mix scaled -2.2 dB, clean"
    music = float_wav(tmp_path / "m.wav", tone(1.0, 0.9))
    plan = MixPlan(music, clips=(Placed(None, music, 0, 0.0, None, 0.0, 0.0, 0.0),))
    wav, stats = mix.render_mix(plan, tmp_path / "build")
    meta = wav.with_suffix(".json")
    meta.write_text(json.dumps({**old, "duration": 1.0, "hash": stats.hash}))  # a pre-#64 cache entry
    _, again = mix.render_mix(plan, tmp_path / "build")
    assert again.peak_s is not None and again.fix and FIX.search(peak_line(again))


# --- music compose ---------------------------------------------------------------------------------

def _hot_bed_script(path: Path, peak_db: float = -1.0) -> None:
    """A 16-bit stereo 440 Hz bed peaking at `peak_db`."""
    path.write_text(
        "import numpy as np, sys, wave\n"
        f"x = 10 ** ({peak_db} / 20) * np.sin(2 * np.pi * 440 * np.arange(2 * 48000) / 48000)\n"
        "pcm = (np.stack([x, x], 1) * 32767).astype('<i2')\n"
        "with wave.open(sys.argv[1], 'wb') as w:\n"
        "    w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000); w.writeframes(pcm.tobytes())\n"
    )


def _compose(project: Path, capsys: pytest.CaptureFixture[str]) -> str:
    _hot_bed_script(project / "compose.py")
    code = main(["-C", str(project), "music", "compose", "compose.py"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0, out
    return out[-1]


def test_composed_bed_under_spec_clips_gets_headroom_and_sfx_on_top_do_not_clip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    project = tmp_path / "p"
    (project / "build").mkdir(parents=True)
    # The spec names the track compose is about to write, so it does not validate yet: still counts.
    (project / "montaj.yaml").write_text(SPEC.format(hit="0dB"))
    line = _compose(project, capsys)
    assert " peak -6.0dBFS " in line and line.endswith("(scaled -5.0 dB: headroom for the spec's voice/SFX)"), line
    with wave.open(str(project / "music" / "song.wav"), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth()) == (2, 2)  # same format, just quieter
    # SFX at the gain Montaj gives the sounds it places, every 37 ms so some land on the bed's crests.
    hits = tuple(Placed(None, "synth:hit", 0, 0.0, None, t, float(DEFAULT_SFX_GAIN[:-2]), 0.0)
                 for t in np.arange(0.0, 1.8, 0.037))
    assert not mix.mix_audio(MixPlan(str(project / "music" / "song.wav"), clips=hits))[1].clipped
    # Control: the same SFX over the bed as composed (-1 dBFS) do clip.
    hot = tmp_path / "hot"
    (hot / "build").mkdir(parents=True)
    assert " peak -1.0dBFS " in _compose(hot, capsys)
    assert mix.mix_audio(MixPlan(str(hot / "music" / "song.wav"), clips=hits))[1].clipped


def test_compose_without_clips_in_the_spec_keeps_the_bed_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    project = tmp_path / "p"
    (project / "build").mkdir(parents=True)
    (project / "montaj.yaml").write_text("video: {size: 64x64, fps: 30}\naudio: {track: music/song.wav}\n"
                                         "shots:\n  - {blank: true, hold: 2s}\n")
    line = _compose(project, capsys)
    assert " peak -1.0dBFS " in line and "scaled" not in line
