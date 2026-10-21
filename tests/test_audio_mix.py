"""Audio mixer: edit splices, placed clips, ducking, synth, spec validation. CPU only, tiny wavs."""
from __future__ import annotations

import shutil
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from montaj.audio import mix
from montaj.audio.mix import SR, DuckPlan, EditOp, MixPlan, Placed
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def tone(sec: float, hz: float = 220.0, amp: float = 0.3) -> np.ndarray:
    return amp * np.sin(2 * np.pi * hz * np.arange(round(sec * SR)) / SR)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def db(x: float) -> float:
    return 20 * np.log10(x)


def test_decode_mono_is_not_attenuated(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "m.wav", tone(0.5))
    out = mix.decode(str(p))
    assert out.shape[1] == 2 and abs(db(rms(out[:, 0])) - db(rms(tone(0.5)))) < 0.1
    assert np.allclose(out[:, 0], out[:, 1])


def test_decode_picks_the_audio_stream_of_a_video(tmp_path: Path) -> None:
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    write_wav(a, tone(0.5, amp=0.1))
    write_wav(b, tone(0.5, amp=0.4))
    mp4 = tmp_path / "rec.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.5:r=10",
         "-i", str(a), "-i", str(b), "-map", "0:v", "-map", "1:a", "-map", "2:a",
         "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(mp4)], check=True)
    s0, s1 = mix.decode(str(mp4), 0), mix.decode(str(mp4), 1)
    assert rms(s1) > 3 * rms(s0)
    with pytest.raises(Exception, match="stream 2"):
        mix.decode(str(mp4), 2)


def test_cut_keeps_length_minus_cut_and_crossfades_the_join() -> None:
    n = 4 * SR
    music = np.stack([tone(4.0), tone(4.0, 330.0)], axis=1).astype(np.float32)
    out = mix.apply_edits(music, [EditOp("cut", 1.0, 2.5, 1)])
    assert len(out) == n - (round(2.5 * SR) - round(1.0 * SR))
    j = round(1.0 * SR)
    assert np.array_equal(out[: j - 400], music[: j - 400])
    assert np.array_equal(out[j + 400 :], music[round(2.5 * SR) + 400 :])
    assert np.max(np.abs(np.diff(out[j - 400 : j + 400, 0]))) < 0.1  # no click at the join


def section_hz(x: np.ndarray, sec: int) -> float:
    """Dominant frequency of the middle half of output second `sec` (clear of any crossfade)."""
    seg = x[sec * SR + SR // 4 : sec * SR + 3 * SR // 4, 0]
    return float(np.argmax(np.abs(np.fft.rfft(seg))) * SR / len(seg))


def test_loop_plays_the_section_times_total_in_order() -> None:
    hz = [200.0, 300.0, 400.0, 500.0]  # one distinguishable tone per second
    left = np.concatenate([tone(1.0, f) for f in hz])
    music = np.stack([left, left], axis=1).astype(np.float32)
    out = mix.apply_edits(music, [EditOp("loop", 1.0, 2.0, 3)])
    assert len(out) == len(music) + 2 * SR
    assert [round(section_hz(out, i)) for i in range(6)] == [200, 300, 300, 300, 400, 500]


def test_edits_apply_sequentially_to_the_finite_track() -> None:
    music = np.stack([tone(4.0)] * 2, axis=1).astype(np.float32)
    assert mix.edit_range_error([EditOp("loop", 3.0, 6.0, 2)], 4.0) is not None
    ops = [EditOp("loop", 3.0, 4.0, 2), EditOp("cut", 4.0, 5.0, 1)]  # 4s -> 5s -> 4s
    assert mix.edit_range_error(ops, 4.0) is None
    assert len(mix.apply_edits(music, ops)) == 4 * SR
    with pytest.raises(Exception, match=r"audio\.edit\[0\]"):
        mix.apply_edits(music, [EditOp("loop", 3.0, 6.0, 2)])
    with pytest.raises(Exception, match=r"audio\.edit\[1\]"):
        mix.apply_edits(music, [EditOp("loop", 3.0, 4.0, 2), EditOp("cut", 5.5, 6.0, 1)])


def test_edit_ops_resolve_beats_on_the_grid_and_reject_bad_ranges() -> None:
    class E:
        def __init__(self, **k):
            self.cut, self.loop, self.times = k.get("cut"), k.get("loop"), k.get("times")

    errors: list[str] = []
    ops = mix.edit_ops([E(cut=("4b", "8b"))], 30, 120.0, 0.5 * 30, errors)
    assert not errors and ops == [EditOp("cut", 2.5, 4.5, 1)]  # beat0 .5s + 4 beats of .5s
    bad = mix.edit_ops([E(cut=("8b", "4b")), E(cut=("1b", "2b"), times=2), E(loop=("1b", "2b"), times=1)],
                       30, 120.0, 0.0, errors)
    assert bad == [] and len(errors) == 3


def test_shift_words_follow_the_edit() -> None:
    class A:
        class audio:
            class _E:
                cut, loop, times = ("2s", "3s"), None, None
            edit = [_E()]

    out = mix.shift_words(A, {1: 1.5, 2: 2.5, 3: 4.0}, 30, None, 0.0)
    assert out == {1: 1.5, 3: 3.0}  # word 2 was cut


def test_duck_reduces_music_by_depth_under_voice() -> None:
    n = 6 * SR
    voice = np.zeros((n, 2), dtype=np.float32)
    voice[2 * SR : 4 * SR] = tone(2.0, 500.0, 0.2)[:, None]
    g = mix.duck_gain(voice, DuckPlan(("v",), -6.0, 0.04, 0.2), n)
    assert abs(db(g[3 * SR]) - -6.0) < 0.1
    assert db(g[SR]) > -0.1 and db(g[5 * SR + SR // 2]) > -0.1
    assert g[2 * SR - 1500] < 1.0  # leads the voice a little
    assert db(g[4 * SR + 2400]) < -3.0 and db(g[4 * SR + SR // 5 * 2]) > -3.0  # release is eased


def test_mix_places_clips_with_gain_pan_and_ducks(tmp_path: Path) -> None:
    music = write_wav(tmp_path / "m.wav", tone(4.0, 220.0, 0.3))
    voice = write_wav(tmp_path / "v.wav", tone(1.0, 500.0, 0.2))
    plan = MixPlan(
        str(music),
        clips=(Placed("voice", str(voice), 0, 0.0, None, 1.0, -6.0, 0.0),
               Placed(None, "synth:tick", 0, 0.0, None, 3.5, 0.0, 1.0)),
        duck=DuckPlan(("voice",), -6.0, 0.04, 0.2),
    )
    audio, stats = mix.mix_audio(plan)
    assert len(audio) == 4 * SR and stats.clips == 2
    assert stats.duck_db < -5.9
    seg = slice(round(1.3 * SR), round(1.7 * SR))
    # music (-6 dB duck) + voice (-6 dB gain): both halves of the mix are on, the bed is quieter
    bed = tone(4.0, 220.0, 0.3)[seg] * 10 ** (-6 / 20)
    assert abs(db(rms(audio[seg, 0] - tone(1.0, 500.0, 0.2)[round(0.3 * SR):round(0.7 * SR)] * 10 ** (-6 / 20)))
               - db(rms(bed))) < 0.3
    tk = slice(round(3.5 * SR), round(3.55 * SR))
    extra = audio[tk] - tone(4.0, 220.0, 0.3)[tk, None]  # what the tick added on top of the music
    assert np.max(np.abs(extra[:, 0])) < 1e-3 < np.max(np.abs(extra[:, 1]))  # hard right pan


def test_clip_past_the_music_extends_and_hot_mix_is_scaled(tmp_path: Path) -> None:
    music = write_wav(tmp_path / "m.wav", tone(1.0, 220.0, 0.9))
    loud = write_wav(tmp_path / "l.wav", tone(1.0, 220.0, 0.9))
    plan = MixPlan(str(music), clips=(Placed(None, str(loud), 0, 0.0, 0.5, 0.5, 0.0, 0.0),))
    audio, stats = mix.mix_audio(plan)
    assert len(audio) == round(1.0 * SR) and stats.clipped and stats.peak_dbfs > 0
    assert np.max(np.abs(audio)) <= mix.PEAK_CEILING + 1e-6
    assert any(line.startswith("WARN clip clipped") for line in mix.check_lines(stats.__dict__))


def test_synth_is_deterministic_and_all_names_exist() -> None:
    for name in mix.SYNTH:
        a, b = mix.synth(name), mix.synth(name)
        assert np.array_equal(a, b) and a.dtype == np.float32 and 0.5 < np.max(np.abs(a)) <= 0.9 + 1e-6


def test_render_mix_is_cached_by_input_bytes(tmp_path: Path) -> None:
    music = write_wav(tmp_path / "m.wav", tone(1.0))
    plan = MixPlan(str(music), clips=(Placed(None, "synth:tick", 0, 0.0, None, 0.2, 0.0, 0.0),))
    wav, stats = mix.render_mix(plan, tmp_path / "build")
    stamp = wav.stat().st_mtime_ns
    wav2, stats2 = mix.render_mix(plan, tmp_path / "build")
    assert wav2 == wav and wav.stat().st_mtime_ns == stamp and stats2 == stats
    assert len(mix.decode(str(wav))) == SR  # float wav reads back through ffmpeg
    write_wav(music, tone(1.0, 330.0))
    assert mix.render_mix(plan, tmp_path / "build")[0] != wav


# --- spec --------------------------------------------------------------------------------------

SPEC = """\
video: {{size: 64x64, fps: 30, bpm: 120, look: warm-film}}
assets: assets
markers: {{logo: 2s}}
audio:
  track: music/song.wav
{audio}shots:
  - {{blank: true, hold: 4s}}
"""


def write_spec(tmp_path: Path, audio: str) -> Path:
    (tmp_path / "assets").mkdir(exist_ok=True)
    write_wav(tmp_path / "music" / "song.wav", tone(4.0))
    write_wav(tmp_path / "sfx" / "whoosh.wav", tone(0.3, 800.0))
    write_wav(tmp_path / "clips" / "line.wav", tone(1.0, 500.0))
    spec = tmp_path / "montaj.yaml"
    spec.write_text(SPEC.format(audio=audio))
    return spec


GOOD = """\
  edit:
    - {cut: [4b, 6b]}
    - {loop: [1b, 2b], times: 2}
  clips:
    - {id: voice, src: clips/line, in: 100ms, out: 0.9s, at: 0.8s, gain: -1dB, pan: -0.2}
    - {src: sfx/whoosh, at: [2b, 4b, logo], offset: -0.16s, gain: -10dB}
    - {src: "synth:boom", at: logo}
  duck: {under: [voice], depth: -6dB, attack: 40ms, release: 200ms}
"""


def test_spec_with_clips_edit_and_duck_validates_and_plans(tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path, GOOD)
    spec = load_spec(spec_path)
    tl = resolve(spec, tmp_path)
    plan = mix.plan_audio(tl.spec, tmp_path, tl.markers, 30, 120.0, 0.0, [])
    assert plan.active and [e.kind for e in plan.edits] == ["cut", "loop"]
    whoosh = [c for c in plan.clips if c.src.endswith("whoosh.wav")]
    assert [round(c.at_s, 3) for c in whoosh] == [0.84, 1.84, 1.84]  # 2b=1s, 4b=2s, logo=2s, -0.16s
    assert plan.clips[0].in_s == 0.1 and plan.duck.attack_s == 0.04


def test_plain_audio_block_is_not_a_mix(tmp_path: Path) -> None:
    tl = resolve(load_spec(write_spec(tmp_path, "")), tmp_path)
    assert not mix.plan_audio(tl.spec, tmp_path, tl.markers, 30, 120.0, 0.0, []).active


@pytest.mark.parametrize("audio, needle", [
    ("  clips:\n    - {src: sfx/nope, at: 1s}\n", "audio.clips[0].src: 'sfx/nope' not found"),
    ("  clips:\n    - {src: 'synth:zap', at: 1s}\n", "unknown synth 'zap'"),
    ("  clips:\n    - {src: sfx/whoosh, at: 1}\n", "bare number"),
    ("  clips:\n    - {src: sfx/whoosh, at: nowhere}\n", "unknown marker 'nowhere'"),
    ("  clips:\n    - {src: sfx/whoosh, at: 1s, gain: loud}\n", "invalid level"),
    ("  clips:\n    - {src: sfx/whoosh, at: 1s, pan: 2}\n", "must be in [-1, 1]"),
    ("  clips:\n    - {src: sfx/whoosh, at: 1s, in: 2s, out: 1s}\n", "must be after in"),
    ("  clips:\n    - {src: sfx/whoosh, at: 1s, Id: x}\n", "audio.clips[0].Id"),
    ("  duck: {under: [voice]}\n", "unknown clip id 'voice'"),
    ("  clips:\n    - {id: voice, src: sfx/whoosh, at: 1s}\n  duck: {under: [voice], depth: 3dB}\n",
     "must be below 0 dB"),
    ("  edit:\n    - {cut: [4b, 2b]}\n", "need 0 <= start < end"),
    ("  edit:\n    - {loop: [3s, 6s]}\n    - {cut: [4s, 5s]}\n", "audio.edit[0]: loop ends at 6.000s"),
    ("  edit:\n    - {loop: [1s, 2s]}\n    - {cut: [5s, 6s]}\n", "audio.edit[1]: cut ends at 6.000s"),
    ("  clips:\n    - {src: sfx/whoosh, at: 1s, gain: 10000dB}\n", "out of range"),
    ("  clips:\n    - {src: sfx/whoosh, at: 1s, gain: " + "9" * 400 + "dB}\n", "out of range"),
    ("  edit:\n    - {cut: [intro, 2b]}\n", "invalid music time"),
    ("  edit:\n    - {loop: [1b, 2b], times: 1}\n", "whole number >= 2"),
    ("  edit:\n    - {cut: [1b, 2b], times: 2}\n", "times needs loop"),
    ("  edit:\n    - {}\n", "exactly one of cut or loop"),
])
def test_spec_rejects_bad_audio(tmp_path: Path, audio: str, needle: str) -> None:
    spec_path = write_spec(tmp_path, audio)
    with pytest.raises(SpecError) as exc:
        load_spec(spec_path)
    assert needle in str(exc.value)


def _planned_srcs(tmp_path: Path) -> list[str]:
    spec = load_spec(tmp_path / "montaj.yaml")
    tl = resolve(spec, tmp_path)
    plan = mix.plan_audio(tl.spec, tmp_path, tl.markers, 30, 120.0, 0.0, [])
    return [c.src for c in plan.clips]


def test_clip_src_bare_asset_name_resolves_from_assets(tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path, "  clips:\n    - {src: line, at: 1s}\n    - {src: line.wav, at: 2s}\n")
    spec_path.write_text(spec_path.read_text().replace("assets: assets\n", "assets: media\n", 1))
    write_wav(tmp_path / "media" / "line.wav", tone(0.4, 500.0))
    want = str((tmp_path / "media" / "line.wav").resolve())
    assert _planned_srcs(tmp_path) == [want, want]


def test_clip_src_spec_relative_wins_over_assets(tmp_path: Path) -> None:
    write_spec(tmp_path, "  clips:\n    - {src: tone, at: 1s}\n    - {src: tone.wav, at: 2s}\n")
    write_wav(tmp_path / "tone.wav", tone(0.2, 440.0))
    write_wav(tmp_path / "assets" / "tone.wav", tone(0.2, 880.0))
    want = str((tmp_path / "tone.wav").resolve())
    assert _planned_srcs(tmp_path) == [want, want]


def test_clip_src_missing_names_the_assets_dir(tmp_path: Path) -> None:
    write_spec(tmp_path, "  clips:\n    - {src: nope, at: 1s}\n")
    with pytest.raises(SpecError) as exc:
        load_spec(tmp_path / "montaj.yaml")
    line = next(err for err in exc.value.errors if err.startswith("audio.clips[0].src:"))
    assert "'nope' not found" in line
    hint = line.split("(", 1)[1]
    assert "spec file" in hint and str(tmp_path / "assets") in hint


def test_beat_units_in_clips_need_a_grid(tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path, "  clips:\n    - {src: sfx/whoosh, at: 2b}\n")
    spec_path.write_text(spec_path.read_text().replace("bpm: 120, ", ""))
    with pytest.raises(SpecError, match="video.bpm: required when a duration uses beats"):
        load_spec(spec_path)


def test_check_lines_report_the_mix() -> None:
    lines = mix.check_lines({"clips": 3, "duck_db": -6.0, "peak_dbfs": -1.3, "clipped": False})
    assert lines == ["OK audio-mix 3 clips, duck -6.0 dB, peak -1.3 dBFS"]


def test_render_muxes_the_mixed_wav_and_check_reports_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline wiring with the GPU stubbed out: the mix replaces the track handed to mux."""
    import json
    from contextlib import contextmanager

    import torch
    from PIL import Image

    from montaj import pipeline
    from montaj.cli import _audio_mix_lines

    (tmp_path / "assets").mkdir()
    Image.new("RGB", (8, 8), (20, 40, 60)).save(tmp_path / "assets" / "a.jpg")
    write_wav(tmp_path / "music" / "song.wav", tone(2.0))
    (tmp_path / "montaj.yaml").write_text(
        "video: {size: 64x64, fps: 30, look: warm-film}\nassets: assets\n"
        "audio:\n  track: music/song.wav\n  clips:\n    - {src: 'synth:tick', at: 0.5s}\n"
        "shots:\n  - {photo: a, hold: 30f}\n"
    )

    @contextmanager
    def hold(log=None):
        yield

    class CV:
        def __init__(self, W, H, device="cuda"):
            self.W, self.H = W, H

    class R:
        def __init__(self, tl, spec_dir, cv):
            self.cv = cv

        def begin_segment(self, shots):
            pass

        def frame(self, f):
            return torch.zeros(3, self.cv.H, self.cv.W)

    class Enc:
        def __init__(self, path, *a, **k):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.path.write_bytes(b"seg")

        def write(self, frame):
            pass

    muxed: dict = {}

    def fake_mux(video, audio, out, loudnorm=True, fade_out=0.0, *, log=None):
        muxed["audio"] = Path(audio)
        Path(out).write_bytes(b"muxed")

    def fake_concat(chunks, out, **_k):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"cat")

    for name, fake in (("gpu_hold", hold), ("Canvas", CV), ("Renderer", R), ("Encoder", Enc),
                       ("concat", fake_concat), ("mux", fake_mux)):
        monkeypatch.setattr(f"montaj.pipeline.{name}", fake)
    pipeline.render(tmp_path / "montaj.yaml", "preview")
    assert muxed["audio"].parent == tmp_path / "build" / "audio" and muxed["audio"].suffix == ".wav"
    assert len(mix.decode(str(muxed["audio"]))) == 2 * SR
    stats = json.loads((tmp_path / "build" / "audio-mix.json").read_text())
    assert stats["clips"] == 1
    assert _audio_mix_lines(tmp_path)[0].startswith("OK audio-mix 1 clips")


def test_mixed_wav_goes_through_the_two_pass_mux(tmp_path: Path) -> None:
    from montaj.encode import mux, probe

    music = write_wav(tmp_path / "m.wav", tone(2.0))
    wav, _ = mix.render_mix(
        MixPlan(str(music), clips=(Placed(None, "synth:boom", 0, 0.0, None, 0.5, 0.0, 0.0),)), tmp_path / "build")
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2:r=10",
                    "-c:v", "mpeg4", str(video)], check=True)
    out = mux(video, wav, tmp_path / "out.mp4", log=tmp_path / "mux.log")
    info = probe(out)
    assert info.has_audio and abs(info.audio_duration - 2.0) < 0.1


def test_zero_fps_reports_an_error_not_a_traceback(tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path, "  edit:\n    - {cut: [0f, 1f]}\n")
    spec_path.write_text(spec_path.read_text().replace("fps: 30", "fps: 0"))
    with pytest.raises(SpecError) as exc:
        load_spec(spec_path)
    assert "video.fps" in str(exc.value)


def test_render_mix_publishes_metadata_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second renderer arriving while the first is mid-publish must not read a half-written file."""
    music = write_wav(tmp_path / "m.wav", tone(1.0))
    plan = MixPlan(str(music), clips=(Placed(None, "synth:tick", 0, 0.0, None, 0.2, 0.0, 0.0),))
    build = tmp_path / "build"
    gate, release = threading.Event(), threading.Event()
    real_write = Path.write_text

    def stall(self: Path) -> None:
        # Whatever the implementation does to a final metadata path, stall right after it is created.
        if self.parent.name == "audio" and self.suffix == ".json" and not self.name.startswith("."):
            gate.set()
            release.wait(timeout=10)

    def slow_write(self: Path, data: str, *a: object, **k: object) -> int:
        if self.parent.name == "audio" and self.suffix == ".json" and not self.name.startswith("."):
            self.write_bytes(b"")  # truncated, visible, not yet filled
            stall(self)
        return real_write(self, data, *a, **k)

    monkeypatch.setattr(Path, "write_text", slow_write)
    errors: list[BaseException] = []

    def first() -> None:
        try:
            mix.render_mix(plan, build)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            gate.set()  # the first renderer finished without ever stalling

    t = threading.Thread(target=first)
    t.start()
    gate.wait(timeout=30)
    try:
        wav, stats = mix.render_mix(plan, build)
    finally:
        release.set()
        t.join(timeout=30)
    assert not errors and stats.clips == 1 and wav.is_file()
    assert not [p for p in (build / "audio").iterdir() if p.name.endswith(".partial")]


def test_concurrent_renders_share_one_mix(tmp_path: Path) -> None:
    music = write_wav(tmp_path / "m.wav", tone(1.0))
    plan = MixPlan(str(music), clips=(Placed(None, "synth:tick", 0, 0.0, None, 0.2, 0.0, 0.0),))
    barrier = threading.Barrier(4)
    results: list[object] = []

    def run() -> None:
        barrier.wait(timeout=30)
        try:
            results.append(mix.render_mix(plan, tmp_path / "build")[1])
        except BaseException as exc:  # noqa: BLE001
            results.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert len(results) == 4 and all(isinstance(r, mix.MixStats) for r in results)


def test_audio_mix_fields_do_not_change_segment_hashes(tmp_path: Path) -> None:
    from montaj.project import segment_hash

    def hashes(audio: str) -> list[str]:
        root = tmp_path / ("a" if audio else "b")
        root.mkdir()
        tl = resolve(load_spec(write_spec(root, audio)), root)
        return [segment_hash(tl, seg, {}, "e1") for seg in tl.segments]

    assert hashes("") == hashes("  edit:\n    - {cut: [1b, 2b]}\n  clips:\n    - {src: sfx/whoosh, at: 1s}\n")
