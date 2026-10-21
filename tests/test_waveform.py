"""Waveform overlay: levels from the named audio, validation, hash, raster."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from montaj.audio.mix import SR, EditOp, MixPlan, Placed, apply_edits, decode, write_wav
from montaj.pipeline import engine_version
from montaj.project import segment_hash
from montaj.render.canvas import Canvas
from montaj.render.frame import Renderer
from montaj.render.overlay import draw_overlays
from montaj.spec import SpecError, load_spec
from montaj.timeline import ResolvedOverlay, resolve

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg"
)

# Pinned at main 819c988, before waveform `levels` existed. A ring film must keep them.
_RING_HASH_0 = "016bd06592ef4ef33ff6de66768cc088dd3bf4c3baef8356ca19b22c638dff3b"
_RING_HASH_1 = "643aaf7e69bde8603441b43e4ab2e66d06a2011f5d959b2e1dbfde73c2270fb2"
_PLAIN_HASH = "3a9f1914c3134710b09cd30a23977d048904cdc2d3d65fff065c66614c1ef87a"


def _photo(tmp: Path) -> None:
    assets = tmp / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (30, 60, 90)).save(assets / "a.jpg", "JPEG")


def _tone(path: Path, sec: float, hz: float, amp: float) -> Path:
    n = round(sec * SR)
    x = (amp * np.sin(2 * np.pi * hz * np.arange(n) / SR)).astype(np.float32)
    write_wav(path, np.stack([x, x], axis=1))
    return path


def _mono_wav(path: Path, x: np.ndarray) -> Path:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def _write(tmp: Path, body: str, *, size: str = "270x480", hold: str = "15f") -> Path:
    _photo(tmp)
    p = tmp / "montaj.yaml"
    p.write_text(
        f"video:\n  size: {size}\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
        f"assets: assets\nshots:\n  - {{photo: a, hold: {hold}}}\n{body}"
    )
    return p


def _hashes(tl) -> list[str]:
    return [segment_hash(tl, s, {"a": "x"}, engine_version()) for s in tl.segments]


def _wave(tmp: Path, *, source: str = "voice", amp: float = 0.5, bars: int | None = None,
          extra: str = "", hold: str = "15f", shots: int = 1) -> Path:
    _tone(tmp / "music.wav", 1.0, 110.0, 0.2)
    _tone(tmp / "voice.wav", 1.0, 30.0, amp)
    bars_f = "" if bars is None else f", bars: {bars}"
    shot = "  - {photo: a, hold: " + hold + "}\n"
    body = (
        "audio:\n  track: music.wav\n  clips:\n"
        "    - {id: voice, src: voice.wav, at: 0s}\n"
        "overlays:\n"
        f"  - {{shape: waveform, from: 0f, to: 6f, source: {source}, "
        f"box: [20, 200, 250, 280], width: 6{bars_f}}}\n{extra}"
    )
    p = _write(tmp, body, hold=hold)
    if shots == 2:
        p.write_text(p.read_text().replace(shot, shot + shot, 1))
    return p


def _ov(**kw) -> ResolvedOverlay:
    base = dict(
        shape="waveform", from_=0.0, to=1.0, center=None, radius=None, sweep=(360.0, 360.0),
        box=(100.0, 40.0, 164.0, 160.0), arm=28.0, points=None, draw=None, width=8.0,
        color="#ffffff", opacity=1.0, glow=0.0, ease="linear", above_finish=False,
        levels=((1.0, 0.0, 0.5, 0.0),),
    )
    base.update(kw)
    return ResolvedOverlay(**base)


# --- levels -------------------------------------------------------------------


@needs_ffmpeg
def test_levels_follow_a_sine_silence_and_loudness(tmp_path: Path) -> None:
    def rows(amp: float, name: str):
        d = tmp_path / name
        d.mkdir()
        tl = resolve(load_spec(_wave(d, amp=amp)), d)
        assert tl.overlays[0].levels is not None
        return tl.overlays[0].levels

    # 0.7 voice + 0.2 music stays under the mix ceiling, so the bars read the sine's own peak.
    loud, again, quiet, silent = rows(0.7, "loud"), rows(0.7, "again"), rows(0.2, "quiet"), None
    assert loud == again  # same samples, same quantised rows
    # Frame 1 is past the 15 ms placement fade. One 30 Hz cycle fills the frame, so the
    # bars are not a flat line and the tallest one is the sine's peak.
    assert max(loud[1]) > min(loud[1])
    assert max(loud[1]) > max(quiet[1]) > 0.0
    assert max(loud[1]) == pytest.approx(0.7, abs=2 / 255)

    d = tmp_path / "silent"
    d.mkdir()
    _tone(d / "music.wav", 1.0, 110.0, 0.0)
    write_wav(d / "voice.wav", np.zeros((SR, 2), np.float32))
    _photo(d)
    p = d / "montaj.yaml"
    p.write_text(
        "video:\n  size: 270x480\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
        "assets: assets\nshots:\n  - {photo: a, hold: 15f}\n"
        "audio:\n  track: music.wav\n  clips:\n    - {id: voice, src: voice.wav, at: 0s}\n"
        "overlays:\n  - {shape: waveform, from: 0f, to: 4f, source: voice, box: [20, 200, 250, 280]}\n"
    )
    silent = resolve(load_spec(p), d).overlays[0].levels
    assert silent is not None and silent == ((0.0,) * 32,) * 4  # default 32 bars, silence is zero


@needs_ffmpeg
def test_music_source_and_bar_count(tmp_path: Path) -> None:
    d = tmp_path / "music"
    d.mkdir()
    _tone(d / "music.wav", 1.0, 30.0, 0.6)
    p = _write(d, (
        "audio:\n  track: music.wav\n"
        "overlays:\n  - {shape: waveform, from: 0f, to: 4f, source: music, "
        "box: [20, 200, 250, 280], bars: 8}\n"
    ))
    tl = resolve(load_spec(p), d)
    levels = tl.overlays[0].levels
    assert levels is not None and len(levels) == 4 and len(levels[1]) == 8
    assert max(levels[1]) > 0.4


# --- validate -----------------------------------------------------------------


def test_unknown_source_and_missing_audio_are_errors(tmp_path: Path) -> None:
    _photo(tmp_path)
    (tmp_path / "music.wav").write_bytes(b"not really audio")
    missing = _write(tmp_path / "no-audio", (
        "overlays:\n  - {shape: waveform, from: 0f, to: 4f, source: voice, box: [0, 0, 10, 10]}\n"
    ))
    (tmp_path / "no-audio").mkdir(exist_ok=True)
    with pytest.raises(SpecError) as ei:
        load_spec(missing)
    assert any("no audio block" in e for e in ei.value.errors), ei.value.errors

    unknown = tmp_path / "bad" / "montaj.yaml"
    unknown.parent.mkdir()
    _photo(unknown.parent)
    (unknown.parent / "music.wav").write_bytes(b"x")
    unknown.write_text(
        "video:\n  size: 270x480\n  fps: 30\n  look: warm-film\n"
        "assets: assets\nshots:\n  - {photo: a, hold: 15f}\n"
        "audio:\n  track: music.wav\n  clips:\n    - {id: voice, src: music.wav, at: 0s}\n"
        "overlays:\n  - {shape: waveform, from: 0f, to: 4f, source: nope, box: [0, 0, 10, 10]}\n"
    )
    with pytest.raises(SpecError) as ei:
        load_spec(unknown)
    assert any("unknown source 'nope'" in e for e in ei.value.errors), ei.value.errors
    assert any("have: voice" in e for e in ei.value.errors)


@needs_ffmpeg
def test_waveform_with_music_source_validates(tmp_path: Path) -> None:
    d = tmp_path / "ok"
    d.mkdir()
    _tone(d / "music.wav", 0.5, 220.0, 0.3)
    p = _write(d, (
        "audio:\n  track: music.wav\n"
        "overlays:\n  - {shape: waveform, from: 1f, to: 5f, source: music, box: [10, 20, 100, 80]}\n"
    ))
    spec = load_spec(p)
    assert spec.overlays[0].source == "music" and spec.overlays[0].bars is None
    assert "source" not in spec.model_copy().shots[0].model_dump()
    dumped = spec.overlays[0].model_dump()
    assert dumped["source"] == "music" and "bars" not in dumped
    assert resolve(spec, d).overlays[0].levels is not None


@pytest.mark.parametrize("body, needle", [
    ("{shape: waveform, from: 0f, to: 4f, source: music}", "box: required for waveform"),
    ("{shape: waveform, from: 0f, to: 4f, source: music, box: [0, 0, 10, 10], center: [1, 1]}",
     "center: not used by waveform"),
    ("{shape: waveform, from: 0f, to: 4f, source: music, box: [0, 0, 10, 10], bars: 0}", "bars"),
    ("{shape: waveform, from: 0f, to: 4f, source: music, box: [0, 0, 10, 10], bars: true}", "bars"),
    ("{shape: ring, from: 0f, to: 4f, center: [1, 1], radius: 5, source: music}", "source: not used by ring"),
])
def test_waveform_field_rules(tmp_path: Path, body: str, needle: str) -> None:
    d = tmp_path / needle
    d.mkdir()
    _photo(d)
    (d / "music.wav").write_bytes(b"x")
    p = d / "montaj.yaml"
    p.write_text(
        "video:\n  size: 270x480\n  fps: 30\n  look: warm-film\n"
        "assets: assets\nshots:\n  - {photo: a, hold: 15f}\n"
        "audio:\n  track: music.wav\n"
        f"overlays:\n  - {body}\n"
    )
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert any(needle in e for e in ei.value.errors), ei.value.errors


# --- hash ---------------------------------------------------------------------


def test_films_without_a_waveform_keep_their_segment_hash(tmp_path: Path) -> None:
    _photo(tmp_path)
    p = tmp_path / "montaj.yaml"
    p.write_text(
        "video:\n  size: 270x480\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
        "assets: assets\nshots:\n  - {photo: a, hold: 30f}\n  - {photo: a, hold: 30f}\n"
        "overlays:\n  - {shape: ring, from: 0f, to: 20f, center: [90, 160], radius: 40, width: 4}\n"
    )
    tl = resolve(load_spec(p), tmp_path)
    assert tl.overlays[0].levels is None
    assert "source" not in tl.spec.overlays[0].model_dump() and "bars" not in tl.spec.overlays[0].model_dump()
    got = _hashes(tl)
    assert got == [_RING_HASH_0, _RING_HASH_1]
    plain = tmp_path / "plain.yaml"
    plain.write_text(
        "video:\n  size: 270x480\n  fps: 30\n  look: warm-film\n  motion_blur: 1\n"
        "assets: assets\nshots:\n  - {photo: a, hold: 30f}\n"
    )
    tlp = resolve(load_spec(plain), tmp_path)
    assert segment_hash(tlp, tlp.segments[0], {"a": "x"}, engine_version()) == _PLAIN_HASH


@needs_ffmpeg
def test_changing_the_source_audio_changes_the_hash_and_the_frames(tmp_path: Path) -> None:
    def film(name: str, amp: float):
        d = tmp_path / name
        d.mkdir()
        p = _wave(d, amp=amp, shots=2)
        tl = resolve(load_spec(p), d)
        frame = Renderer(tl, d, Canvas(*tl.size, device="cpu")).frame(2)
        return _hashes(tl), frame

    quiet_h, quiet_f = film("quiet", 0.15)
    loud_h, loud_f = film("loud", 0.85)
    assert quiet_h[0] != loud_h[0]          # the waveform's levels are in the segment hash
    assert quiet_h[1] == loud_h[1]          # the later shot never shows it
    assert not torch.equal(quiet_f, loud_f)


# --- the window matches the mixer --------------------------------------------


@needs_ffmpeg
def test_source_window_matches_placement_and_edits(tmp_path: Path) -> None:
    from montaj.audio.mix import mix_audio, source_window

    music = _tone(tmp_path / "m.wav", 2.0, 80.0, 0.2)
    voice = _tone(tmp_path / "v.wav", 1.0, 440.0, 0.15)
    full_music = decode(str(music))
    plan = MixPlan(
        str(music),
        edits=(EditOp("cut", 0.4, 0.7),),
        clips=(
            Placed("voice", str(voice), 0, 0.0, None, 0.2, -6.0, -0.5),
            Placed("voice", str(voice), 0, 0.1, 0.4, 0.5, 3.0, 0.0),
            Placed("blip", "synth:tick", 0, 0.0, None, 0.0, -12.0, 1.0),
        ),
    )
    assert not mix_audio(plan)[1].clipped  # no ceiling scale: the windows are the raw stems
    edited = apply_edits(full_music, plan.edits)
    s0, s1 = 10_000, min(len(edited), 60_000)
    got = source_window(plan, "music", s0, s1)
    assert got.shape == (s1 - s0, 2)
    assert np.allclose(got, edited[s0:s1], atol=1e-6)
    tail = source_window(plan, "music", len(edited) - 100, len(edited) + 400)
    assert np.allclose(tail[:100], edited[-100:], atol=1e-6) and np.all(tail[100:] == 0)

    n = SR  # covers both voice placements; the tick is a different id
    acc = _mixer_sum(plan, "voice", n)
    voiced = source_window(plan, "voice", 0, n)
    assert np.allclose(voiced, acc, atol=1e-6)
    assert float(np.max(np.abs(voiced))) > 0.05  # the sum is actually in the window


def _mixer_sum(plan: MixPlan, source: str, n: int) -> np.ndarray:
    """What `mix_audio` adds to the bed for `source`, before ducking, over `[0, n)`."""
    from montaj.audio.mix import _render_clip

    cache: dict = {}
    acc = np.zeros((n, 2), np.float32)
    for c in plan.clips:
        if c.id != source:
            continue
        start, x = _render_clip(c, cache, None)
        lo, hi = max(start, 0), min(start + len(x), n)
        if lo < hi:
            acc[lo:hi] += x[lo - start:lo - start + (hi - lo)]
    return acc


@needs_ffmpeg
def test_compressed_sources_match_the_mixer_to_the_last_sample(tmp_path: Path) -> None:
    """Vorbis pads its container duration; the bars must still end where the mixed clip ends."""
    from montaj.audio.mix import mix_audio, source_window

    wav = _tone(tmp_path / "v.wav", 1.0, 440.0, 0.5)
    ogg = tmp_path / "v.ogg"
    subprocess.check_call(["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-c:a", "libvorbis", str(ogg)])
    plan = MixPlan(
        str(ogg),
        edits=(EditOp("cut", 0.3, 0.5),),
        clips=(Placed("voice", str(ogg), 0, 0.0, None, 0.25, -9.0, 0.3),),  # `out` omitted
    )
    assert not mix_audio(plan)[1].clipped  # no ceiling scale: the windows are the raw stems
    n = 2 * SR
    assert np.array_equal(source_window(plan, "voice", 0, n), _mixer_sum(plan, "voice", n))
    edited = apply_edits(decode(str(ogg)), plan.edits)
    got = source_window(plan, "music", 0, len(edited) + 100)
    assert np.array_equal(got[:len(edited)], edited) and not got[len(edited):].any()


@needs_ffmpeg
def test_sources_sum_to_the_mix_with_duck_and_ceiling(tmp_path: Path) -> None:
    """Every source's window, added up, is exactly what `mix_audio` writes: duck and peak scale included."""
    from montaj.audio.mix import DuckPlan, mix_audio, source_window

    music = _tone(tmp_path / "m.wav", 2.0, 80.0, 0.9)
    voice = _tone(tmp_path / "v.wav", 0.6, 440.0, 0.8)
    plan = MixPlan(
        str(music),
        edits=(EditOp("cut", 0.4, 0.7),),
        clips=(Placed("voice", str(voice), 0, 0.0, None, 0.3, 0.0, 0.0),),
        duck=DuckPlan(("voice",), -12.0, 0.02, 0.2),
    )
    mixed, stats = mix_audio(plan)
    assert stats.clipped  # the ceiling scale is exercised
    cache: dict = {}
    music_w = source_window(plan, "music", 0, len(mixed), cache=cache)
    voice_w = source_window(plan, "voice", 0, len(mixed), cache=cache)
    assert np.allclose(music_w + voice_w, mixed, atol=1e-6)
    under = slice(round(0.4 * SR), round(0.8 * SR))  # voice speaking, duck fully down
    assert np.abs(music_w[under]).max() < 0.35 * np.abs(music_w[: round(0.2 * SR)]).max()


@needs_ffmpeg
def test_bars_follow_the_audio_fade_out(tmp_path: Path) -> None:
    def rows(fade: str, name: str):
        d = tmp_path / name
        d.mkdir()
        p = _wave(d, amp=0.8)
        text = p.read_text().replace("from: 0f, to: 6f", "from: 9f, to: 15f")
        p.write_text(text.replace("  track: music.wav\n", f"  track: music.wav\n  fade_out: {fade}\n"))
        return resolve(load_spec(p), d).overlays[0].levels

    plain, faded = rows("0s", "plain"), rows("6f", "faded")  # the fade covers frames 9..15
    assert len(plain) == len(faded) == 6
    assert all(max(f) <= max(q) for f, q in zip(faded, plain))
    assert max(faded[-1]) < 0.25 * max(plain[-1])  # the last frame is near silent
    assert max(faded[1]) > 0.7 * max(plain[1])  # early in the fade, mostly intact


# --- raster -------------------------------------------------------------------


def _box_mask(cv: Canvas, box: tuple[float, float, float, float]) -> torch.Tensor:
    x0, y0, x1, y1 = (v * cv.k for v in box)
    ys = torch.arange(cv.H, device=cv.device) + 0.5
    xs = torch.arange(cv.W, device=cv.device) + 0.5
    return (ys[:, None] >= y0) & (ys[:, None] < y1) & (xs[None, :] >= x0) & (xs[None, :] < x1)


def test_waveform_bars_are_crisp_mirrored_and_scale() -> None:
    cv = Canvas(1080, 200, device="cpu")  # k = 1, design px are device px
    box = (100.0, 40.0, 164.0, 160.0)
    out = draw_overlays(cv, torch.zeros(3, cv.H, cv.W), [_ov(box=box)], 0.0)
    assert out[:, 100, 108].tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)  # full bar, centre
    assert float(out[0, 100, 124]) == 0.0                                          # a zero bar
    assert float(out[0, 100, 140]) == pytest.approx(1.0, abs=1e-4)                 # half-height bar, still solid
    assert float(out[0, 50, 140]) == 0.0 and float(out[0, 150, 140]) == 0.0        # mirrored, short of the box
    assert float(out[0, 30, 108]) == 0.0 and float(out[0, 100, 90]) == 0.0         # outside the box
    # Silence draws nothing. A wider stroke lays more ink until the slot is full.
    assert float(draw_overlays(cv, torch.zeros(3, cv.H, cv.W), [_ov(levels=((0.0,) * 4,))], 0.0).sum()) == 0.0
    thin = draw_overlays(cv, torch.zeros(3, cv.H, cv.W), [_ov(width=2.0)], 0.0)
    thick = draw_overlays(cv, torch.zeros(3, cv.H, cv.W), [_ov(width=6.0)], 0.0)
    assert float(thick.sum()) > float(thin.sum()) * 2

    half = Canvas(540, 100, device="cpu")
    a = draw_overlays(half, torch.zeros(3, half.H, half.W), [_ov(box=box)], 0.0)
    b = draw_overlays(cv, torch.zeros(3, cv.H, cv.W), [_ov(box=box)], 0.0)
    assert float(b.sum()) == pytest.approx(4.0 * float(a.sum()), rel=0.08)


def _assert_bars_stay_inside(device: str) -> None:
    cv = Canvas(360, 640, device=device)  # both sides >= 320; NVENC rejects anything under ~145
    box = (90.0, 120.0, 990.0, 900.0)
    row = tuple(0.0 if i % 2 == 0 else 1.0 for i in range(16))
    ov = _ov(box=box, width=12.0, glow=0.6, levels=(row,))
    img = torch.zeros(3, cv.H, cv.W, device=device)
    out = draw_overlays(cv, img, [ov], 0.0)
    ink = out.abs().sum(0) > 1e-4
    inside = _box_mask(cv, box)
    assert bool(ink.any())
    assert not bool((ink & ~inside).any())
    assert float(img.abs().sum()) == 0.0
    assert float(draw_overlays(cv, img, [ov], 5.0).abs().sum()) == 0.0  # outside the window


def test_bars_stay_inside_the_box_on_cpu() -> None:
    _assert_bars_stay_inside("cpu")


@pytest.mark.gpu
def test_bars_stay_inside_the_box_on_gpu() -> None:
    _assert_bars_stay_inside("cuda")
