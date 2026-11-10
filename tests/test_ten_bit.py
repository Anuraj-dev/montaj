"""10-bit export (#46): P010 on the GPU, NVENC Main10 / AV1, dithered 8-bit, presets, `check` tags."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from montaj import encode, qa
from montaj.cli import _parser
from montaj.encode import TARGETS, MediaInfo, VideoSettings, rgb_to_nv12, rgb_to_p010
from montaj.gpu_policy import GpuPolicy

W, H, FPS = 320, 240, 25


def _gradient(device: str = "cpu") -> torch.Tensor:
    """A slow horizontal grey ramp across 0.2..0.8: 8-bit limited range gives it ~131 levels."""
    ramp = torch.linspace(0.2, 0.8, W, device=device).expand(H, W)
    return ramp.unsqueeze(0).repeat(3, 1, 1).contiguous()


# ---- conversion (CPU) ---------------------------------------------------------------------


def test_p010_layout_and_bt709_limited_levels() -> None:
    frame = torch.zeros(3, 4, 4)
    frame[:, :, 2:] = 1.0
    words = rgb_to_p010(frame)
    assert words.dtype == torch.uint16 and words.numel() == 4 * 4 + 2 * 2 * 2
    luma = (words[:16].to(torch.int32) >> 6).view(4, 4)
    assert (words[:16].to(torch.int32) & 0x3F).eq(0).all(), "10 bits must sit in the high bits"
    assert luma[:, :2].eq(64).all() and luma[:, 2:].eq(940).all()
    chroma = words[16:].to(torch.int32) >> 6
    assert chroma.eq(512).all(), "grey has neutral chroma"


def test_p010_gradient_keeps_more_than_256_levels() -> None:
    frame = _gradient()
    luma10 = (rgb_to_p010(frame)[: W * H].to(torch.int32) >> 6).unique().numel()
    luma8 = rgb_to_nv12(frame)[: W * H].unique().numel()
    # 320 columns over 0.6 of the range: 8-bit has ~131 codes for them, 10-bit one per column.
    assert luma8 < 256 < luma10, (luma8, luma10)


def test_dither_keeps_the_mean_and_breaks_bands() -> None:
    """A flat field at luma code 100.3: rounding flattens it to 100, the ordered dither mixes 100
    and 101 so the mean (what the eye sees) stays 100.3, identically every frame."""
    level = (100.3 - 16.0) / 219.0
    frame = torch.full((3, 64, 64), level)
    plain = rgb_to_nv12(frame)[: 64 * 64].float()
    dithered = rgb_to_nv12(frame, dither=True)[: 64 * 64].float()
    assert plain.unique().tolist() == [100.0]
    assert set(dithered.unique().tolist()) == {100.0, 101.0}
    assert abs(float(dithered.mean()) - 100.3) < 0.02
    assert torch.equal(rgb_to_nv12(frame, dither=True), rgb_to_nv12(frame, dither=True)), "deterministic"


# ---- settings, hashes, presets (CPU) ------------------------------------------------------


def test_default_video_settings_leave_the_hash_payload_alone() -> None:
    assert VideoSettings().hash_key() is None
    assert VideoSettings("h264").hash_key() is None
    assert VideoSettings("hevc10").hash_key() == {"codec": "hevc10", "cq": 16, "maxrate": None, "dither": False}
    assert VideoSettings("hevc10").depth == 10 and VideoSettings("av1").ten_bit
    with pytest.raises(encode.EncodeError, match="unknown"):
        VideoSettings("vp9")


def test_codec_changes_the_segment_hash_only_when_set(tmp_path: Path) -> None:
    from montaj.project import segment_hash
    from montaj.spec import load_spec
    from montaj.timeline import resolve

    from tests.test_spec import _write

    spec = _write(tmp_path, "  - {photo: a, hold: 1s}\n", ["a"])
    tl = resolve(load_spec(spec), spec.parent)
    seg = tl.segments[0]
    shas = {"a": "x"}
    base = segment_hash(tl, seg, shas, "e1")
    assert segment_hash(tl, seg, shas, "e1", VideoSettings().hash_key()) == base
    assert segment_hash(tl, seg, shas, "e1", VideoSettings("hevc10").hash_key()) != base


def test_targets_table_covers_every_platform_and_tags_bt709() -> None:
    assert set(TARGETS) == {"youtube", "shorts", "reels", "whatsapp", "master"}
    for name, t in TARGETS.items():
        assert t.name == name and t.tags == "bt709"
        line = t.describe()
        assert line.startswith(f"target={name} ") and "LUFS" in line and "bt709" in line
    assert TARGETS["youtube"].video.codec == "hevc10"
    # 8-bit deliveries are dithered; 10-bit ones do not need it.
    assert all(t.video.dither for t in TARGETS.values() if not t.video.ten_bit)


def test_cli_accepts_codec_and_every_target() -> None:
    ns = _parser().parse_args(["render", "--final", "--codec", "hevc10"])
    assert ns.codec == "hevc10"
    assert _parser().parse_args(["render", "--final"]).codec is None
    for name in TARGETS:
        assert _parser().parse_args(["export", "--target", name]).target == name
    with pytest.raises(Exception, match="invalid choice"):
        _parser().parse_args(["render", "--codec", "vp9"])


def _info(**kw) -> MediaInfo:
    base = dict(path=Path("x.mp4"), duration=1.0, width=W, height=H, fps=25.0, has_audio=False, audio_codec=None,
                audio_duration=None, sample_rate=None, vcodec="hevc", pix_fmt="yuv420p10le",
                color_space="bt709", color_primaries="bt709", color_transfer="bt709", color_range="tv")
    base.update(kw)
    return MediaInfo(**base)


def test_bit_depth_from_pix_fmt() -> None:
    assert _info(pix_fmt="yuv420p10le").bit_depth == 10
    assert _info(pix_fmt="p010le").bit_depth == 10
    assert _info(pix_fmt="yuv420p").bit_depth == 8
    assert _info(pix_fmt=None).bit_depth is None


def test_check_video_line_verifies_codec_depth_and_tags() -> None:
    want = TARGETS["youtube"].video
    assert qa.video_line(_info(), want) == "OK video hevc 10-bit yuv420p10le bt709"
    assert qa.video_line(_info(vcodec="h264", pix_fmt="yuv420p"), want).startswith("WARN video h264 8-bit")
    assert qa.video_line(_info(color_transfer=None)).startswith("WARN video hevc 10-bit yuv420p10le colour tags")
    assert "range=pc" in qa.video_line(_info(color_range="pc"))
    assert qa.video_line(_info(vcodec="h264", pix_fmt="yuv420p")).startswith("OK video h264 8-bit")


# ---- NVENC (GPU) --------------------------------------------------------------------------


def _ffprobe(path: Path) -> dict:
    out = subprocess.run([encode.FFPROBE, "-v", "error", "-show_streams", "-select_streams", "v:0",
                          "-of", "json", str(path)], stdout=subprocess.PIPE, check=True).stdout
    return json.loads(out)["streams"][0]


def _luma_levels(path: Path) -> int:
    """Distinct 10-bit luma codes in the first decoded frame."""
    raw = subprocess.run([encode.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    y = np.frombuffer(raw, "<u2")[: W * H]
    return int(np.unique(y).size)


def _need(codec: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    error = encode.ten_bit_error(codec)
    if error is not None:
        pytest.skip(error)


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cuda", None])
@pytest.mark.parametrize("codec,name,profile", [("hevc10", "hevc", "Main 10"), ("av1", "av1", "Main")])
def test_ten_bit_encode_tags_and_does_not_band(tmp_path: Path, codec: str, name: str, profile: str,
                                               device: str | None) -> None:
    """Both routes (zero-copy from CUDA, ffmpeg pipe) write a bt709-tagged 10-bit stream whose
    gradient keeps more than 256 luma levels; strict policy counts the encode as GPU."""
    _need(codec)
    out = tmp_path / "g.mp4"
    policy = GpuPolicy(strict=True)
    frame = _gradient("cuda")
    with encode.Encoder(out, W, H, FPS, log=tmp_path / "e.log", device=device, policy=policy,
                        video=VideoSettings(codec)) as enc:
        for _ in range(10):
            enc.write(frame)
    assert enc.codec == encode.TEN_BIT_ENCODERS[codec]
    direct = encode.zero_copy_error(torch.cuda.current_device(), name) is None
    assert enc.route.zero_copy == (device is not None and direct), enc.route
    assert policy.summary() == "all"
    s = _ffprobe(out)
    assert (s["codec_name"], s["profile"], s["pix_fmt"]) == (name, profile, "yuv420p10le")
    assert (s["color_range"], s["color_space"], s["color_transfer"], s["color_primaries"]) == ("tv",) + ("bt709",) * 3
    if codec == "hevc10":
        assert s["codec_tag_string"] == "hvc1"
    assert int(s["nb_frames"]) == 10
    levels = _luma_levels(out)
    assert levels > 256, levels
    assert qa.video_line(encode.probe(out), VideoSettings(codec)).startswith(f"OK video {name} 10-bit")


@pytest.mark.gpu
def test_dithered_h264_stays_8_bit_and_tagged(tmp_path: Path) -> None:
    if not torch.cuda.is_available() or encode.zero_copy_error(torch.cuda.current_device()) is not None:
        pytest.skip("no zero-copy NVENC")
    out = tmp_path / "d.mp4"
    with encode.Encoder(out, W, H, FPS, log=tmp_path / "e.log", device="cuda",
                        video=VideoSettings("h264", cq=19, maxrate=25, dither=True)) as enc:
        for _ in range(5):
            enc.write(_gradient("cuda"))
    s = _ffprobe(out)
    assert (s["codec_name"], s["pix_fmt"]) == ("h264", "yuv420p")
    assert (s["color_space"], s["color_transfer"], s["color_primaries"]) == ("bt709",) * 3


@pytest.mark.gpu
def test_hevc_concat_and_mux_keep_hvc1(tmp_path: Path) -> None:
    _need("hevc10")
    chunks = []
    for i in range(2):
        p = tmp_path / f"c{i}.mp4"
        with encode.Encoder(p, W, H, FPS, log=tmp_path / "e.log", device="cuda",
                            video=VideoSettings("hevc10")) as enc:
            for _ in range(5):
                enc.write(_gradient("cuda"))
        chunks.append(p)
    out = encode.concat(chunks, tmp_path / "all.mp4", fps=FPS, frames=[5, 5], log=tmp_path / "c.log",
                        vcodec="hevc10")
    s = _ffprobe(out)
    assert s["codec_tag_string"] == "hvc1" and s["pix_fmt"] == "yuv420p10le" and int(s["nb_frames"]) == 10
    wav = tmp_path / "a.wav"
    subprocess.run([encode.FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=d=1", str(wav)], check=True)
    muxed = encode.mux(out, wav, tmp_path / "m.mp4", log=tmp_path / "m.log", lufs=-16.0, audio_kbps=192)
    assert _ffprobe(muxed)["codec_tag_string"] == "hvc1"
