"""IO module tests: encode/ingest/qa/sheet/doctor against tiny lavfi-generated media."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from montaj import doctor, encode, ingest, qa, sheet

W, H, FPS = 320, 240, 25


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def lavfi(src: str, duration: float | None = None) -> str:
    """`testsrc` or `color=c=black` -> a lavfi description at the test size and rate."""
    opts = f"size={W}x{H}:rate={FPS}" + (f":duration={duration}" if duration else "")
    return f"{src}:{opts}" if "=" in src else f"{src}={opts}"


def clip(out: Path, *, seconds: float = 1.0, src: str = "testsrc") -> Path:
    """Synthetic clip from a lavfi source; every log goes to tmp_path so nothing lands in the repo."""
    ff("-f", "lavfi", "-i", lavfi(src, seconds), "-pix_fmt", "yuv420p", str(out))
    assert out.exists(), out
    return out


def tone(out: Path, *, seconds: float = 1.0) -> Path:
    ff("-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}", str(out))
    return out


def frames(count: int, seed: int = 0) -> list[np.ndarray]:
    """`count` distinct uint8 frames so a freeze is never accidental."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(count)]


def encode_frames(out: Path, count: int, log: Path, codec: str = "libx264") -> Path:
    with encode.Encoder(out, W, H, FPS, encoder=codec, log=log) as enc:
        for frame in frames(count):
            enc.write(frame)
    return out


def failing_ffmpeg(tmp_path: Path) -> str:
    """An 'ffmpeg' that exits non-zero with no output — the failed-analysis case, for real."""
    stub = tmp_path / "ffmpeg-stub"
    stub.write_text("#!/bin/sh\nexit 3\n")
    stub.chmod(0o755)
    return str(stub)


def stream_md5(path: Path, selector: str) -> str:
    """MD5 over one stream's packet payloads. Identical MD5 = the stream was copied, not re-encoded."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", selector,
         "-c", "copy", "-f", "md5", "-"],
        stdout=subprocess.PIPE, check=True,
    ).stdout.decode()
    return out.strip().rsplit("=", 1)[-1]


# --- encode ---------------------------------------------------------------------------------

def test_encoder_roundtrip_frame_count(tmp_path: Path) -> None:
    out = encode_frames(tmp_path / "seg.mp4", 30, tmp_path / "encode.log")
    assert encode.probe(out, log=tmp_path / "p.log").size == (W, H)
    assert encode.frame_count(out, log=tmp_path / "p.log") == 30


def test_encoder_rejects_wrong_shape(tmp_path: Path) -> None:
    with encode.Encoder(tmp_path / "bad.mp4", W, H, FPS, encoder="libx264", log=tmp_path / "e.log") as enc:
        with pytest.raises(encode.EncodeError):
            enc.write(np.zeros((H, W, 4), dtype=np.uint8))


def test_encoder_failure_raises_with_stderr(tmp_path: Path) -> None:
    log = tmp_path / "encode.log"
    with pytest.raises(encode.EncodeError):
        with encode.Encoder(tmp_path / "out.mp4", W, H, FPS, encoder="not_a_codec", log=log) as enc:
            for frame in frames(3):
                enc.write(frame)
    assert log.stat().st_size > 0 and "Unknown encoder" in log.read_text()


def test_concat_frame_count_is_sum_of_chunks(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    a = encode_frames(tmp_path / "a.mp4", 10, log)
    b = encode_frames(tmp_path / "b.mp4", 15, log)
    out = encode.concat([a, b], tmp_path / "joined.mp4", log=log)
    assert encode.frame_count(out, log=log) == 25


def test_concat_rejects_missing_chunk(tmp_path: Path) -> None:
    with pytest.raises(encode.EncodeError):
        encode.concat([tmp_path / "nope.mp4"], tmp_path / "out.mp4", log=tmp_path / "io.log")


def test_mux_loudnorm(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    video = clip(tmp_path / "v.mp4", seconds=2.0, src="testsrc")
    audio = tone(tmp_path / "a.wav", seconds=2.0)
    out = encode.mux(video, audio, tmp_path / "m.mp4", log=log)
    info = encode.probe(out, log=log)
    assert info.has_audio and info.audio_codec == "aac" and info.sample_rate == 48000
    assert qa.loudness(out, log=log) == pytest.approx(-14.0, abs=1.0)
    assert any(line.startswith("OK loudness") for line in qa.check(out, log=log))


def test_mux_copies_the_video_stream(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    video = clip(tmp_path / "v.mp4", seconds=2.0, src="testsrc")
    out = encode.mux(video, tone(tmp_path / "a.wav", seconds=2.0), tmp_path / "m.mp4", log=log)
    assert stream_md5(out, "0:v") == stream_md5(video, "0:v")  # not re-encoded
    plain = encode.mux(video, tone(tmp_path / "b.wav", seconds=2.0), tmp_path / "p.mp4", False, log=log)
    assert stream_md5(plain, "0:v") == stream_md5(video, "0:v")


def test_mux_pads_audio_shorter_than_the_film(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    video = clip(tmp_path / "v.mp4", seconds=10.0, src="testsrc")
    out = encode.mux(video, tone(tmp_path / "a.wav", seconds=2.0), tmp_path / "m.mp4", log=log)
    info = encode.probe(out, log=log)
    assert encode.frame_count(out, log=log) == 250  # 10 s at 25 fps survives a 2 s track
    assert info.duration == pytest.approx(10.0, abs=0.1)
    assert info.audio_duration == pytest.approx(10.0, abs=0.2)  # padded, not left short
    assert any(line.startswith("OK av-offset") for line in qa.check(out, log=log))


def test_mux_cuts_audio_longer_than_the_film(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    video = clip(tmp_path / "v.mp4", seconds=2.0, src="testsrc")
    out = encode.mux(video, tone(tmp_path / "a.wav", seconds=6.0), tmp_path / "m.mp4", log=log)
    info = encode.probe(out, log=log)
    assert info.duration == pytest.approx(2.0, abs=0.1)
    assert info.audio_duration == pytest.approx(2.0, abs=0.2)


def test_probe_reports_video_and_audio_durations_separately(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    video = clip(tmp_path / "v.mp4", seconds=1.0, src="testsrc")
    ff("-i", str(video), "-i", str(tone(tmp_path / "a.wav", seconds=3.0)), "-c:v", "copy",
       "-c:a", "aac", "-pix_fmt", "yuv420p", str(tmp_path / "mismatch.mp4"))
    info = encode.probe(tmp_path / "mismatch.mp4", log=log)
    assert info.duration == pytest.approx(1.0, abs=0.05)  # the film is 1 s, not the container's 3
    assert info.av_offset == pytest.approx(2.0, abs=0.05)
    assert any(line.startswith("WARN av-offset") for line in qa.check(tmp_path / "mismatch.mp4", log=log))


def test_export_master_copies_both_streams(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    src = encode.mux(clip(tmp_path / "v.mp4", seconds=2.0, src="testsrc"),
                     tone(tmp_path / "a.wav", seconds=2.0), tmp_path / "m.mp4", log=log)
    master = encode.export(src, tmp_path / "master.mp4", "master", log=log)
    assert stream_md5(master, "0:v") == stream_md5(src, "0:v")  # audio-bearing input, not a silent one
    assert stream_md5(master, "0:a") == stream_md5(src, "0:a")
    assert encode.probe(master, log=log).audio_codec == "aac"


def test_export_whatsapp_width(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    src = encode.concat(
        [encode_frames(tmp_path / f"c{i}.mp4", 5, log) for i in range(2)], tmp_path / "master.mp4", log=log
    )
    wa = encode.export(src, tmp_path / "wa.mp4", "whatsapp", log=log)
    info = encode.probe(wa, log=log)
    assert info.width == 720
    assert info.height % 2 == 0 and info.height < H * 4
    assert stream_md5(wa, "0:v") != stream_md5(src, "0:v")  # whatsapp really re-encodes


def test_prefer_encoder_falls_back_when_nvenc_cannot_encode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(encode, "nvenc_usable", lambda: False)
    assert encode.prefer_encoder() == "libx264"
    with encode.Encoder(tmp_path / "auto.mp4", W, H, FPS, log=tmp_path / "e.log") as enc:
        assert enc.codec == "libx264"
        for frame in frames(3):
            enc.write(frame)
    assert encode.frame_count(tmp_path / "auto.mp4", log=tmp_path / "p.log") == 3


def test_export_rejects_unknown_preset(tmp_path: Path) -> None:
    with pytest.raises(encode.EncodeError):
        encode.export(tmp_path / "x.mp4", tmp_path / "y.mp4", "reels", log=tmp_path / "io.log")


@pytest.mark.gpu
def test_encoder_uses_nvenc_when_present(tmp_path: Path) -> None:
    assert encode.has_encoder("h264_nvenc"), "ffmpeg lists no h264_nvenc"
    out = encode_frames(tmp_path / "gpu.mp4", 12, tmp_path / "encode.log", codec=encode.prefer_encoder())
    assert encode.frame_count(out, log=tmp_path / "p.log") == 12


# --- ingest ---------------------------------------------------------------------------------

def _rotated_jpeg(path: Path, size: tuple[int, int] = (3000, 2000)) -> None:
    """Landscape pixels tagged orientation 6 = "rotate 90 CW" -> viewers show portrait."""
    image = Image.new("RGB", size, (10, 120, 200))
    exif = image.getexif()
    exif[0x0112] = 6
    image.save(path, exif=exif, quality=80)


def test_ingest_exif_rotation_downscale_and_assets_json(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _rotated_jpeg(src / "16.jpg")
    Image.new("RGB", (640, 480), (200, 30, 30)).save(src / "09.png")
    assets = tmp_path / "assets"

    infos = ingest.ingest(src, assets, sheet_path=tmp_path / "ingest-sheet.jpg")

    assert [i.stem for i in infos] == ["09", "16"]  # sorted by stem
    tall = next(i for i in infos if i.stem == "16")
    assert (tall.w, tall.h) == (1365, 2048)  # 2000x3000 after EXIF, longest side capped
    assert max(tall.w, tall.h) <= ingest.MAX_SIDE
    with Image.open(assets / "16.jpg") as out_img:
        assert out_img.size == (1365, 2048) and out_img.mode == "RGB"
    small = next(i for i in infos if i.stem == "09")
    assert (small.w, small.h) == (640, 480)  # never upscaled
    rows = json.loads((assets / "assets.json").read_text())
    assert rows == [{"stem": i.stem, "w": i.w, "h": i.h, "sha": i.sha} for i in infos]
    assert all(len(i.sha) == 64 for i in infos)
    with Image.open(tmp_path / "ingest-sheet.jpg") as sheet_img:
        assert sheet_img.width <= 1536 and sheet_img.width > 0


def test_ingest_rejects_empty_source(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ingest.IngestError):
        ingest.ingest(tmp_path / "empty", tmp_path / "assets", write_sheet=False)


def test_ingest_rejects_duplicate_stems(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (32, 32), (1, 2, 3)).save(src / "07.jpg")
    Image.new("RGB", (32, 32), (3, 2, 1)).save(src / "07.png")
    with pytest.raises(ingest.IngestError):
        ingest.ingest(src, tmp_path / "assets", write_sheet=False)


# --- sheet ----------------------------------------------------------------------------------

def test_sheet_output_size_and_labels(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    video = clip(tmp_path / "v.mp4", seconds=2.0, src="testsrc")
    out = sheet.sheet(video, tmp_path / "sheet.jpg", n=4, log=log)
    with Image.open(out) as img:
        assert img.width <= 1536 and img.width > 0
        assert img.height > 0
    assert [sheet.stamp(t) for t in (0.25, 1.75)] == ["00:00.25", "00:01.75"]
    times = sheet.pick_times(video, n=4, info=encode.probe(video, log=log))
    assert times == pytest.approx([0.25, 0.75, 1.25, 1.75])  # midpoints
    assert sheet.sheet(video, tmp_path / "explicit.jpg", at=[1.5], log=log).exists()


def test_sheet_dedupes_times_inside_one_frame(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    video = clip(tmp_path / "v.mp4", seconds=2.0, src="testsrc")
    info = encode.probe(video, log=log)
    # 0.081 s and 0.087 s are both inside frame 2 at 25 fps
    same_frame = [0.081, 0.087]
    assert [int(round(t * FPS)) for t in same_frame] == [2, 2]

    frames = sheet.extract(video, same_frame, tmp_path / "f", info=info, log=log)
    assert [index for index, _ in frames] == [2]
    assert frames[0][1].exists()

    one = sheet.sheet(video, tmp_path / "one.jpg", at=same_frame, log=log)
    two = sheet.sheet(video, tmp_path / "two.jpg", at=[0.2, 0.6], log=log)
    with Image.open(one) as one_img, Image.open(two) as two_img:
        assert one_img.height > two_img.height  # 1 tile -> one full-width column, 2 -> two


def test_sheet_caps_tile_count(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    video = clip(tmp_path / "v.mp4", seconds=1.0, src="testsrc")
    times = sheet.pick_times(video, n=40, info=encode.probe(video, log=log))
    assert len(times) <= sheet.MAX_TILES


# --- qa -------------------------------------------------------------------------------------

def _spans(*sections: str) -> Path:
    """One silent clip of 1 s sections: a lavfi source name ("testsrc", "color=black") or "still:<path>"."""
    out = Path(tempfile.mkdtemp(prefix="montaj-io-")) / "spans.mp4"
    args: list[str] = []
    for section in sections:
        if section.startswith("still:"):
            args += ["-loop", "1", "-t", "1", "-i", section.removeprefix("still:")]
        else:
            args += ["-f", "lavfi", "-t", "1", "-i", lavfi(section)]
    streams = "".join(f"[{i}:v]" for i in range(len(sections)))
    ff(*args, "-filter_complex", f"{streams}concat=n={len(sections)}:v=1:a=0[v]", "-map", "[v]",
       "-pix_fmt", "yuv420p", str(out))
    return out


def test_check_reports_black_span(tmp_path: Path) -> None:
    log = tmp_path / "qa.log"
    video = _spans("color=c=black", "color=c=white")
    lines = qa.check(video, log=log)
    blacks = [line for line in lines if "black" in line]
    assert blacks and blacks[0].startswith("WARN black 0.00-1.00s")
    assert any(line.startswith("OK duration") for line in lines)
    assert "INFO no audio" in lines


def test_check_detects_frozen_span(tmp_path: Path) -> None:
    log = tmp_path / "qa.log"
    still = tmp_path / "still.png"
    Image.new("RGB", (W, H), (240, 20, 20)).save(still)
    # moving, held still for a second, moving again: the hold is the only frozen span
    video = _spans("testsrc", f"still:{still}", "testsrc")

    warned = [line for line in qa.check(video, log=log) if "frozen" in line]
    assert len(warned) == 1 and warned[0].startswith("WARN frozen"), warned
    start, end = (float(v) for v in re.search(r"([\d.]+)-([\d.]+)s", warned[0]).groups())
    assert start == pytest.approx(1.0, abs=0.2) and end == pytest.approx(2.0, abs=0.2)

    intended = qa.check(video, [(0.9, 2.1)], log=log)
    assert not any("frozen" in line and line.startswith("WARN") for line in intended)
    assert any(line.startswith("INFO frozen 1 span(s) ignored") for line in intended)

    # only the first 0.5 s of the hold was intended: the rest is still a defect
    partial = [line for line in qa.check(video, [(0.9, 1.5)], log=log) if "frozen" in line]
    assert len(partial) == 1 and partial[0].startswith("WARN frozen"), partial
    left, right = (float(v) for v in re.search(r"([\d.]+)-([\d.]+)s", partial[0]).groups())
    assert left == pytest.approx(1.5, abs=0.2) and right == pytest.approx(2.0, abs=0.2)


def test_subtract_keeps_the_part_nobody_claimed() -> None:
    assert qa.subtract((1.0, 10.0), [(1.0, 6.0)]) == [(6.0, 10.0)]
    assert qa.subtract((1.0, 10.0), []) == [(1.0, 10.0)]
    assert qa.subtract((1.0, 10.0), [(2.0, 3.0)]) == [(1.0, 2.0), (3.0, 10.0)]
    assert qa.subtract((1.0, 10.0), [(1.0, 10.0)]) == []  # fully intended


def test_subtract_counts_overlapping_intents_once() -> None:
    # [1,6] twice is 5 s of intent, not 10: the leftover must still be reported
    assert qa.subtract((1.0, 10.0), [(1.0, 6.0), (1.0, 6.0)]) == [(6.0, 10.0)]
    assert qa.subtract((1.0, 10.0), [(1.0, 6.0), (5.0, 10.0)]) == []
    assert qa.union([(1.0, 6.0), (5.0, 10.0), (20.0, 21.0)]) == [(1.0, 10.0), (20.0, 21.0)]


def test_check_raises_when_analysis_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed blackdetect must raise, never look like a clean file."""
    log = tmp_path / "qa.log"
    video = clip(tmp_path / "v.mp4", seconds=1.0, src="testsrc")
    monkeypatch.setattr(qa, "FFMPEG", failing_ffmpeg(tmp_path))
    with pytest.raises(encode.EncodeError):
        qa.check(video, log=log)


def test_check_raises_when_loudness_analysis_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "qa.log"
    video = clip(tmp_path / "v.mp4", seconds=1.0, src="testsrc")
    with_audio = encode.mux(video, tone(tmp_path / "a.wav", seconds=1.0), tmp_path / "m.mp4", log=log)
    monkeypatch.setattr(qa, "FFMPEG", failing_ffmpeg(tmp_path))
    with pytest.raises(encode.EncodeError):
        qa.check(with_audio, log=log)


def test_check_loudness_warning_without_normalisation(tmp_path: Path) -> None:
    log = tmp_path / "io.log"
    video = clip(tmp_path / "v.mp4", seconds=2.0, src="testsrc")
    loud = encode.mux(video, tone(tmp_path / "a.wav", seconds=2.0), tmp_path / "m.mp4",
                      loudnorm=False, log=log)
    assert any(line.startswith("WARN loudness") for line in qa.check(loud, log=log))  # full-scale sine is ~-3 LUFS


# --- doctor ---------------------------------------------------------------------------------

def test_doctor_lines_and_traps() -> None:
    lines = doctor.doctor()
    checks = [line.split(":", 1)[0].removeprefix("TRAP ") for line in lines]
    for name in ("cuda", "nvenc", "ffmpeg", "ffprobe", "chromium", "vram"):
        assert any(name in check for check in checks), f"doctor is missing the {name} check: {lines}"
    assert all(line.split(":", 1)[0].split(" ", 1)[0] in {"OK", "WARN", "FAIL", "TRAP"} for line in lines)
    traps = [line for line in lines if line.startswith("TRAP ")]
    assert len(traps) == len(doctor.TRAPS) == 5
    assert any("chromium" in t for t in traps) and any("vram" in t for t in traps)


def test_doctor_reports_present_tools() -> None:
    lines = doctor.doctor()
    assert any(line.startswith("OK ffmpeg:") for line in lines)
    assert any(line.startswith("OK ffprobe:") for line in lines)


@pytest.mark.gpu
def test_doctor_reports_cuda() -> None:
    assert doctor.cuda_line().startswith("OK cuda:"), doctor.cuda_line()
    vram = doctor.free_vram()
    assert vram is not None and vram[0] > 0


def test_doctor_warns_when_nvenc_is_listed_but_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "has_encoder", lambda name: True)
    monkeypatch.setattr(doctor, "nvenc_usable", lambda: False)
    line = doctor.nvenc_line()
    assert line.startswith("WARN nvenc:") and "cannot encode" in line and "libx264" in line


def test_doctor_reports_nvenc_when_it_encodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "has_encoder", lambda name: True)
    monkeypatch.setattr(doctor, "nvenc_usable", lambda: True)
    assert doctor.nvenc_line().startswith("OK nvenc:")


def test_doctor_warns_when_nvenc_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "has_encoder", lambda name: False)
    assert doctor.nvenc_line().startswith("WARN nvenc: h264_nvenc missing")


@pytest.mark.gpu
def test_nvenc_is_usable_here() -> None:
    assert encode.has_encoder("h264_nvenc")
    assert encode.nvenc_usable(), "h264_nvenc is listed but cannot encode on this machine"
    assert encode.prefer_encoder() == "h264_nvenc"


def test_free_vram_handles_missing_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(doctor.subprocess, "run", boom)
    assert doctor.free_vram() is None