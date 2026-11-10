"""Linked clip audio (#40): a clip shot's own sound follows its range, speed and freeze; J and L cuts."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from montaj.audio import mix
from montaj.audio.mix import SR
from montaj.project import segment_hash
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve
from tests.test_audio_mix import tone, write_wav

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

FPS = 30
SPF = SR // FPS  # 1600 samples a frame
CLIP_S = 8.0
# Irregular single-sample clicks, so any offset shows and no two windows look alike.
CLICKS = [round((0.21 + 0.137 * k + 0.011 * (k % 5)) * SR) for k in range(52)]
FADE = round(mix.CLIP_FADE_S * SR)


def click_track() -> np.ndarray:
    x = np.zeros(round(CLIP_S * SR))
    x[CLICKS] = 0.5
    return x


def make_clip(path: Path, audio: bool = True, sound: np.ndarray | None = None) -> Path:
    """An 8 s 64x64 clip with the click track (or `sound`) as 16-bit PCM (no codec priming, so no lag)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=gray:s=64x64:d={CLIP_S}:r={FPS}"]
    if audio:
        wav = write_wav(path.with_suffix(".wav"), click_track() if sound is None else sound)
        cmd += ["-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:a", "pcm_s16le"]
    subprocess.run(cmd + ["-c:v", "mpeg4", str(path)], check=True)
    return path


def film(tmp: Path, shots: list[dict], audio: dict | None = None, overlays: list | None = None) -> Path:
    if not (tmp / "assets" / "talk.mov").is_file():
        make_clip(tmp / "assets" / "talk.mov")
    if not (tmp / "music" / "bed.wav").is_file():
        write_wav(tmp / "music" / "bed.wav", np.zeros(round(10 * SR)))
    spec: dict = {"video": {"size": "64x64", "fps": FPS, "look": "warm-film"}, "assets": "assets",
                  "audio": {"track": "music/bed.wav", **(audio or {})}, "shots": shots}
    if overlays:
        spec["overlays"] = overlays
    p = tmp / "montaj.yaml"
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    return p


def mixed(p: Path) -> np.ndarray:
    tl = resolve(load_spec(p), p.parent)
    plan = mix.plan_audio(tl.spec, p.parent, tl.markers, FPS, None, 0.0, [], shots=tl.shots)
    assert plan is not None and plan.active
    out, stats = mix.mix_audio(plan)
    assert not stats.clipped
    return out[:, 0]


def src() -> np.ndarray:
    return click_track().astype(np.float32)


def assert_plays(out: np.ndarray, film_a: int, film_b: int, src_a: int) -> None:
    """Film samples `[film_a, film_b)` are the source from `src_a`, sample for sample, inside the edge fades."""
    lo, hi = film_a + FADE, film_b - FADE
    s = src()
    want = s[src_a + FADE : src_a + (hi - film_a)]
    assert len(want) and np.any(want)
    np.testing.assert_allclose(out[lo:hi], want, atol=4e-5)  # one 16-bit PCM step


def test_linked_shots_play_their_range_sample_exact(tmp_path: Path) -> None:
    out = mixed(film(tmp_path, [
        {"clip": "talk", "range": ["1s", "2s"], "audio": True},
        {"clip": "talk", "range": ["3s", "4.5s"], "audio": True},
    ]))
    assert_plays(out, 0, SR, SR)
    assert_plays(out, SR, round(2.5 * SR), 3 * SR)
    # Cross-correlation against the source agrees: lag 0 for each shot's window.
    for a, b, s0 in ((0, SR, SR), (SR, round(2.5 * SR), 3 * SR)):
        seg = out[a + FADE : b - FADE]
        ref = src()[s0 + FADE - 400 : s0 + (b - a) - FADE + 400]
        lag = int(np.argmax(np.correlate(ref, seg, mode="valid"))) - 400
        assert lag == 0


def test_moving_a_linked_shot_keeps_sync(tmp_path: Path) -> None:
    out = mixed(film(tmp_path, [
        {"blank": True, "hold": "17f"},
        {"clip": "talk", "range": ["2s", "3s"], "audio": True},
    ]))
    assert not np.any(out[: 17 * SPF])
    assert_plays(out, 17 * SPF, 17 * SPF + SR, 2 * SR)


def test_j_cut_starts_the_next_sound_early_and_ends_the_last(tmp_path: Path) -> None:
    out = mixed(film(tmp_path, [
        {"clip": "talk", "range": ["1s", "2s"], "audio": True},
        {"clip": "talk", "range": ["4s", "5s"], "audio": True, "audio_lead": "8f"},
    ]))
    cut = SR - 8 * SPF
    assert_plays(out, 0, cut, SR)
    assert_plays(out, cut, 2 * SR, 4 * SR - 8 * SPF)  # the lead reads the 8 frames before the range


def test_l_cut_carries_the_sound_under_the_next_shot(tmp_path: Path) -> None:
    out = mixed(film(tmp_path, [
        {"clip": "talk", "range": ["1s", "2s"], "audio": True, "audio_tail": "12f"},
        {"clip": "talk", "range": ["4s", "5s"], "audio": True},
    ]))
    cut = SR + 12 * SPF
    assert_plays(out, 0, cut, SR)  # 1s .. 2s + 12f of source
    assert_plays(out, cut, 2 * SR, 4 * SR + 12 * SPF)


def test_lead_under_an_unlinked_shot_plays_over_it(tmp_path: Path) -> None:
    out = mixed(film(tmp_path, [
        {"blank": True, "hold": "1s"},
        {"clip": "talk", "range": ["3s", "4s"], "audio": True, "audio_lead": "250ms"},
    ]))
    a = SR - SR // 4
    assert not np.any(out[: a])
    assert_plays(out, a, 2 * SR, 3 * SR - SR // 4)


def peaks(x: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.abs(x) > 0.2)]


def test_speed_reads_the_source_at_that_rate(tmp_path: Path) -> None:
    out = mixed(film(tmp_path, [{"clip": "talk", "clip_in": "1s", "hold": "1s", "speed": 2, "audio": True}]))
    # Film sample s plays source 1s + 2s; a click at source c lands at (c - 1s) / 2 (linear interp splits odd ones).
    got = peaks(out[FADE : SR - FADE])
    want = [(c - SR) // 2 - FADE for c in CLICKS if SR + 2 * FADE <= c < SR + 2 * (SR - FADE) and (c - SR) % 2 == 0]
    assert want and set(want) <= set(got)


def test_speed_ramp_matches_the_picture_clock_within_a_frame(tmp_path: Path) -> None:
    from montaj.clipmap import source_index

    keys = [{"t": "0f", "v": 1}, {"t": "30f", "v": 2, "ease": "linear"}]
    p = film(tmp_path, [{"clip": "talk", "clip_in": "1s", "hold": "1s", "speed": keys, "audio": True}])
    tl = resolve(load_spec(p), tmp_path)
    plan = mix.plan_audio(tl.spec, tmp_path, tl.markers, FPS, None, 0.0, [], shots=tl.shots)
    link = plan.links[0]
    table = source_index(link.speed, None, 30)
    pos, live = mix._source_frames(link.speed, None, np.arange(30, dtype=float))
    assert live.all()
    assert all(int(np.floor(v + 1e-6)) == t for v, t in zip(pos, table))  # the picture floors the same integral


def test_freeze_mutes_the_sound_from_the_freeze(tmp_path: Path) -> None:
    out = mixed(film(tmp_path, [{"clip": "talk", "clip_in": "1s", "hold": "2s", "freeze": "1s", "audio": True}]))
    assert np.any(out[FADE:SR - FADE])
    assert not np.any(out[SR:2 * SR])


def test_gain_and_duck_under_a_linked_id(tmp_path: Path) -> None:
    write_wav(tmp_path / "music" / "bed.wav", tone(4.0, 220.0, 0.2))
    make_clip(tmp_path / "assets" / "hum.mov", sound=tone(CLIP_S, 330.0, 0.3))  # steady "speech" for the duck
    out = mixed(film(tmp_path, [
        {"clip": "talk", "range": ["1s", "2s"], "audio": True, "id": "cam", "gain": "-6dB"},
        {"clip": "hum", "range": ["1s", "2s"], "audio": True, "id": "voice"},
    ], audio={"duck": {"under": ["voice"], "depth": "-12dB"}}))
    tl = resolve(load_spec(tmp_path / "montaj.yaml"), tmp_path)
    plan = mix.plan_audio(tl.spec, tmp_path, tl.markers, FPS, None, 0.0, [], shots=tl.shots)
    cam = mix.source_window(plan, "cam", 0, SR)[:, 0]
    k = [c - SR for c in CLICKS if SR + FADE < c < 2 * SR - FADE][0]
    assert abs(cam[k] - 0.5 * 10 ** (-6 / 20)) < 1e-3
    music = mix.source_window(plan, "music", 0, 4 * SR)[:, 0]
    loud, under = np.max(np.abs(music[3 * SR:])), np.max(np.abs(music[13 * SR // 10: 18 * SR // 10]))
    assert loud > 3.5 * under  # -12 dB while the linked "voice" sounds
    assert len(out) >= 2 * SR


def test_waveform_follows_the_linked_sound(tmp_path: Path) -> None:
    p = film(tmp_path, [
        {"clip": "talk", "range": ["1s", "2s"], "audio": True, "id": "voice", "audio_tail": "6f"},
        {"blank": True, "hold": "1s"},
    ], overlays=[{"shape": "waveform", "from": "0s", "to": "2s", "box": [0, 0, 64, 64], "source": "voice"}])
    tl = resolve(load_spec(p), tmp_path)
    levels = tl.overlays[0].levels
    plan = mix.plan_audio(tl.spec, tmp_path, tl.markers, FPS, None, 0.0, [], shots=tl.shots)
    stem = mix.source_window(plan, "voice", 0, 2 * SR)
    bars = len(levels[0])
    for f in (3, 31, 34, 40):
        assert levels[f] == mix.band_levels(stem[f * SPF:(f + 1) * SPF], bars)
    assert any(any(r) for r in levels[30:36])  # the tail sounds under the blank
    assert not any(any(r) for r in levels[37:])


def test_audio_fields_do_not_change_segment_hashes(tmp_path: Path) -> None:
    def hashes(root: Path, extra: dict) -> list[str]:
        p = film(root, [{"clip": "talk", "range": ["1s", "2s"], **extra},
                        {"clip": "talk", "range": ["3s", "4s"]}])
        tl = resolve(load_spec(p), root)
        return [segment_hash(tl, seg, {"clip:talk": "x"}, "e1") for seg in tl.segments]

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert hashes(tmp_path / "a", {}) == hashes(
        tmp_path / "b", {"audio": True, "gain": "-2dB", "audio_tail": "4f"})


def test_unset_fields_stay_out_of_the_dump(tmp_path: Path) -> None:
    spec = load_spec(film(tmp_path, [{"clip": "talk", "range": ["1s", "2s"]}]))
    dumped = spec.shots[0].model_dump(by_alias=True)
    assert not {"audio", "gain", "audio_lead", "audio_tail", "id"} & set(dumped)


@pytest.mark.parametrize("shots, needle", [
    ([{"clip": "talk", "range": ["1s", "2s"], "audio": True, "audio_lead": "4f"}], "no shot before"),
    ([{"clip": "talk", "range": ["1s", "2s"], "audio": True, "audio_tail": "4f"}], "no shot after"),
    ([{"blank": True, "hold": "1s", "audio": True}], "audio needs clip"),
    ([{"clip": "talk", "range": ["1s", "2s"], "gain": "-2dB"}], "gain needs audio: true"),
    ([{"clip": "talk", "range": ["1s", "2s"], "audio_tail": "2f"}, {"blank": True, "hold": "1s"}],
     "audio_tail needs audio: true"),
    ([{"blank": True, "hold": "5f"}, {"clip": "talk", "range": ["1s", "2s"], "audio": True, "audio_lead": "8f"}],
     "longer than shots[0]"),
    ([{"clip": "talk", "range": ["1s", "2s"], "audio": True, "audio_tail": "2f"},
      {"clip": "talk", "range": ["3s", "4s"], "audio": True, "audio_lead": "2f"}], "already moves this audio cut"),
    ([{"clip": "talk", "range": ["1s", "2s"], "audio": True, "id": "Voice"}], "invalid id"),
    ([{"clip": "talk", "range": ["1s", "2s"], "audio": True, "gain": "loud"}], "invalid level"),
    ([{"clip": "talk", "range": ["1s", "2s"], "audio": True, "audio_tail": "8"}, {"blank": True, "hold": "1s"}],
     "invalid duration"),
    ([{"clip": "mute", "range": ["1s", "2s"], "audio": True}], "has no audio stream"),
])
def test_bad_linked_audio_is_rejected(tmp_path: Path, shots: list, needle: str) -> None:
    make_clip(tmp_path / "assets" / "mute.mov", audio=False)
    with pytest.raises(SpecError, match=needle.replace("[", r"\[").replace("]", r"\]")):
        resolve(load_spec(film(tmp_path, shots)), tmp_path)


def test_linked_audio_needs_an_audio_block(tmp_path: Path) -> None:
    p = film(tmp_path, [{"clip": "talk", "range": ["1s", "2s"], "audio": True}])
    raw = yaml.safe_load(p.read_text())
    del raw["audio"]
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(SpecError, match="needs an audio block"):
        resolve(load_spec(p), tmp_path)


def test_duck_under_accepts_a_linked_id(tmp_path: Path) -> None:
    p = film(tmp_path, [{"clip": "talk", "range": ["1s", "2s"], "audio": True, "id": "voice"}],
             audio={"duck": {"under": ["voice"]}})
    resolve(load_spec(p), tmp_path)  # no "unknown clip id"


def test_plan_without_links_keeps_its_hash() -> None:
    """Pinned from main before links existed: films without linked shots reuse their cached mix wav."""
    plan = mix.MixPlan("synth:x", clips=(mix.Placed(None, "synth:tick", 0, 0.0, None, 1.0, 0.0, 0.0),))
    assert mix.plan_hash(plan) == "43cde5899d8d7a6b"
