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
    assert {"music (track)", "synth:hit"} <= set(labels), line  # every source sounding at the peak sample
    assert "whole mix scaled" in line and "clean" in line
    who, cut = FIX.search(line).groups()
    assert who == "gain on every synth:hit clip", line  # one spec entry, two placements, named by its src
    # One round: write the printed cut into the spec's `gain`, mix again, nothing left to scale.
    again = spec_mix(tmp_path, f"-{cut}dB")
    assert not again.clipped and again.peak_dbfs <= 20 * np.log10(mix.PEAK_CEILING)
    assert mix.check_lines(dataclasses.asdict(again))[1:] == []


def test_a_clip_wins_over_the_track_whenever_one_clears_it(tmp_path: Path) -> None:
    tick = Placed(None, float_wav(tmp_path / "t.wav", tone(0.2, 0.1)), 0, 0.0, None, 0.5, 0.0, 0.0)
    plan = MixPlan(float_wav(tmp_path / "m.wav", tone(2.0, 0.96)), clips=(tick,))
    who, cut = FIX.search(peak_line(mix.mix_audio(plan)[1])).groups()
    assert who == "gain on the t.wav clip"  # costlier than < 1 dB on the bed, but the spec can set it
    quiet = dataclasses.replace(tick, gain_db=-float(cut))
    assert not mix.mix_audio(dataclasses.replace(plan, clips=(quiet,)))[1].clipped


def test_a_bed_only_peak_names_the_track_file_not_a_spec_field(tmp_path: Path) -> None:
    music = tone(2.0, 1.0)  # the bed alone is over the ceiling: no clip cut can clear it
    tick = Placed(None, float_wav(tmp_path / "t.wav", tone(0.2, 0.1)), 0, 0.0, None, 0.5, 0.0, 0.0)
    line = peak_line(mix.mix_audio(MixPlan(float_wav(tmp_path / "m.wav", music), clips=(tick,)))[1])
    m = re.search(r"\(to skip the scale re-compose m\.wav or turn the file down by ([\d.]+) dB "
                  r"\(no spec field sets the track level\)\)$", line)
    assert m, line
    cut = m.group(1)
    assert "music (track) " in line
    quieter = float_wav(tmp_path / "m2.wav", music * np.float32(10 ** (-float(cut) / 20)))
    assert not mix.mix_audio(MixPlan(quieter, clips=(tick,)))[1].clipped


def test_the_every_source_fallback_spares_duck_sources(tmp_path: Path) -> None:
    a = float_wav(tmp_path / "a.wav", tone(1.0, 0.6))
    v = float_wav(tmp_path / "v.wav", tone(1.0, 0.1))

    def plan(db: float) -> MixPlan:
        bed = float_wav(tmp_path / f"m{db}.wav", tone(1.0, 0.6) * np.float32(10 ** (db / 20)))
        return MixPlan(bed, clips=(Placed("x", a, 0, 0.0, None, 0.0, db, 0.0),
                                   Placed("y", a, 0, 0.0, None, 0.0, db, 0.0),
                                   Placed("v", v, 0, 0.0, None, 0.0, 0.0, 0.0)),  # the voice keeps its gain
                       duck=mix.DuckPlan(("v",), -1.0, 0.01, 0.05))

    _, stats = mix.mix_audio(plan(0.0))
    who, cut = FIX.search(peak_line(stats)).groups()
    assert who == "the track and every clip except duck sources (v)"  # any two of x, y, bed already clip
    _, again = mix.mix_audio(plan(-float(cut)))
    assert not again.clipped and again.duck_db == stats.duck_db  # the duck still triggers as before


def test_emphasis_sounds_are_named_by_sfx_gain_and_a_clip_called_music_is_not_the_bed(tmp_path: Path) -> None:
    from montaj.emphasis import SFX_ID

    a = float_wav(tmp_path / "a.wav", tone(1.0, 0.7))
    plan = MixPlan(float_wav(tmp_path / "m.wav", tone(1.0, 0.2)),
                   clips=(Placed(SFX_ID, a, 0, 0.0, None, 0.0, 0.0, 0.0),
                          Placed("music", float_wav(tmp_path / "b.wav", tone(1.0, 0.3)), 0, 0.0, None, 0.0, 0.0, 0.0)))
    stats = mix.mix_audio(plan)[1]
    line = peak_line(stats)
    assert [n for n, _ in stats.peak_parts] == ["audio.sfx_gain", "music", "music (track)"], line
    who, cut = FIX.search(line).groups()
    assert who == "audio.sfx_gain", line
    quiet = dataclasses.replace(plan.clips[0], gain_db=-float(cut))
    assert not mix.mix_audio(dataclasses.replace(plan, clips=(quiet, plan.clips[1])))[1].clipped


def test_with_loudnorm_off_the_scale_is_a_warn() -> None:
    stats = {"clips": 1, "duck_db": 0.0, "peak_dbfs": 2.0, "clipped": True}
    assert mix.check_lines(stats)[1].startswith("INFO clip mix peak +2.0 dBFS")
    off = mix.check_lines(stats, loudnorm=False)[1]
    assert off.startswith("WARN clip mix peak +2.0 dBFS") and "loudnorm is off" in off


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


def _compose(project: Path, capsys: pytest.CaptureFixture[str], out: str = "music/song.wav") -> list[str]:
    _hot_bed_script(project / "compose.py")
    code = main(["-C", str(project), "music", "compose", "compose.py", "--out", out])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0, lines
    return lines


def test_composed_bed_gets_headroom_and_sfx_on_top_do_not_clip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    project = tmp_path / "p"
    (project / "build").mkdir(parents=True)
    lines = _compose(project, capsys)  # no spec yet: the skill composes before writing it
    assert lines[0].startswith("INFO music compose: bed turned down 5.0 dB to -6 dBFS peak"), lines
    assert " peak -6.0dBFS " in lines[-1] and lines[-1].endswith("LUFS"), lines
    with wave.open(str(project / "music" / "song.wav"), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth()) == (2, 2)  # same format, just quieter
    # SFX at the gain Montaj gives the sounds it places, every 37 ms so some land on the bed's crests.
    hits = tuple(Placed(None, "synth:hit", 0, 0.0, None, t, float(DEFAULT_SFX_GAIN[:-2]), 0.0)
                 for t in np.arange(0.0, 1.8, 0.037))
    assert not mix.mix_audio(MixPlan(str(project / "music" / "song.wav"), clips=hits))[1].clipped
    # Control: the same SFX over the bed at the -1 dBFS the script wrote do clip.
    hot = tmp_path / "hot.wav"
    _hot_bed_script(tmp_path / "hot.py")
    import subprocess, sys
    subprocess.run([sys.executable, str(tmp_path / "hot.py"), str(hot)], check=True)
    assert mix.mix_audio(MixPlan(str(hot), clips=hits))[1].clipped


@pytest.mark.parametrize(("spec", "out", "scaled"), [
    ("audio: {track: music/song.wav}\n", "music/song.wav", True),  # the track, no clips: still the bed
    ("video: {fps: 30}\n", "music/song.wav", True),  # a spec without a track yet
    ("audio: {track: music/other.wav}\n", "music/song.wav", False),  # plainly not the track
    ("audio:\n  clips:\n    - {src: sfx/sting, at: 1s}\n", "sfx/sting.wav", False),  # a clip src
])
def test_compose_headroom_applies_to_the_bed_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
    spec: str, out: str, scaled: bool,
) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    project = tmp_path / "p"
    (project / "build").mkdir(parents=True)
    (project / "montaj.yaml").write_text(spec)
    lines = _compose(project, capsys, out)
    assert (" peak -6.0dBFS " in lines[-1]) == scaled and (" peak -1.0dBFS " in lines[-1]) != scaled, lines
    assert any(line.startswith("INFO music compose") for line in lines) == scaled


# --- re-review of 1359cd0 ---------------------------------------------------------------------


def test_sfx_gain_names_its_default_when_the_sounds_sit_at_it(tmp_path: Path) -> None:
    from montaj.emphasis import SFX_ID

    default_db = float(DEFAULT_SFX_GAIN.removesuffix("dB"))
    plan = MixPlan(float_wav(tmp_path / "m.wav", tone(1.0, 0.2)),
                   clips=(Placed(SFX_ID, float_wav(tmp_path / "a.wav", tone(1.0, 1.5)), 0, 0.0, None, 0.0,
                                 default_db, 0.0),
                          Placed("music", float_wav(tmp_path / "b.wav", tone(1.0, 0.3)), 0, 0.0, None, 0.0, 0.0, 0.0)))
    who, cut = FIX.search(peak_line(mix.mix_audio(plan)[1])).groups()
    assert who == f"audio.sfx_gain (default {DEFAULT_SFX_GAIN})"
    quiet = dataclasses.replace(plan.clips[0], gain_db=default_db - float(cut))
    assert not mix.mix_audio(dataclasses.replace(plan, clips=(quiet, plan.clips[1])))[1].clipped


def test_an_audio_layer_is_never_the_clip_cut_and_does_not_move_the_hash(tmp_path: Path) -> None:
    src = float_wav(tmp_path / "c.wav", tone(1.0, 0.1))
    layer = mix.Linked("pad", src, 30, 0, 0, SR, 0.0, None, None, 0.0, fixed=True)
    b = Placed("b", src, 0, 0.0, None, 0.0, 0.0, 0.0)
    xl, xb = tone(1.0, 0.6), tone(1.0, 0.3)
    bed = tone(1.0, 0.2)
    full = bed + xl + xb  # the layer alone is the cheapest cut, but no spec field sets its gain
    plan = MixPlan(None, clips=(b,), links=(layer,))
    kind, name, n, cut = mix._peak_fix(full, [(layer, 0, xl), (b, 0, xb)], plan, [("id", "pad"), ("id", "b")])
    assert (kind, name, n) == ("id", "b", 1)
    assert float(np.max(np.abs(bed + xl + xb * np.float32(10 ** (-cut / 20))))) <= mix.PEAK_CEILING
    unfixed = dataclasses.replace(plan, links=(dataclasses.replace(layer, fixed=False),))
    assert mix.plan_hash(unfixed) == mix.plan_hash(plan)  # existing films keep their cached mix


def test_unnamed_clips_of_the_same_file_name_in_two_dirs_are_two_sources(tmp_path: Path) -> None:
    one = float_wav(tmp_path / "x" / "t.wav", tone(1.0, 0.6))
    two = float_wav(tmp_path / "y" / "t.wav", tone(1.0, 0.1))
    plan = MixPlan(float_wav(tmp_path / "m.wav", tone(1.0, 0.5)),
                   clips=(Placed(None, one, 0, 0.0, None, 0.0, 0.0, 0.0),
                          Placed(None, two, 0, 0.0, None, 0.0, 0.0, 0.0)))
    stats = mix.mix_audio(plan)[1]
    assert [n for n, _ in stats.peak_parts] == ["t.wav", "music (track)", "t.wav"]  # shown by name, keyed by path
    who, cut = FIX.search(peak_line(stats)).groups()
    assert who == "gain on the t.wav clip"  # one placement, not "every"
    quiet = dataclasses.replace(plan.clips[0], gain_db=-float(cut))
    assert not mix.mix_audio(dataclasses.replace(plan, clips=(quiet, plan.clips[1])))[1].clipped


def test_a_user_clip_cannot_take_the_sfx_id(tmp_path: Path) -> None:
    from montaj.spec import SpecError

    (tmp_path / "sfx").mkdir()
    spec = SPEC.format(hit="0dB").replace('- {src: "synth:hit"', '- {id: sfx, src: "synth:hit"')
    (tmp_path / "montaj.yaml").write_text(spec)
    with pytest.raises(SpecError) as e:
        load_spec(tmp_path / "montaj.yaml")
    assert any(line.startswith("audio.clips[0].id: 'sfx' is reserved") for line in e.value.errors), e.value.errors


@pytest.mark.parametrize(("spec", "out", "bed"), [
    ("audio:\n  clips:\n    - {src: boom, at: 1s}\n", "assets/boom.wav", False),  # found in the assets dir
    ("assets: media\naudio:\n  clips:\n    - {src: boom, at: 1s}\n", "media/boom.wav", False),
    ("audio:\n  clips:\n    - {src: 'synth:boom', at: 1s}\n", "sfx/boom.wav", False),  # a project sfx/ synth
    ("audio:\n  clips:\n    - {src: 'synth:boom', at: 1s}\n", "music/song.wav", True),
])
def test_is_bed_finds_clip_srcs_where_the_mixer_does(tmp_path: Path, spec: str, out: str, bed: bool) -> None:
    from montaj.cli import _is_bed

    (tmp_path / "montaj.yaml").write_text(spec)
    assert _is_bed(tmp_path, (tmp_path / out).resolve()) is bed
