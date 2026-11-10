"""Strict dGPU mode (issue #29): the policy, its seams, and the strict renders."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from montaj import cli, encode
from montaj.encode import Encoder
from montaj.gpu_policy import GpuPolicy, GpuStrictError, load_policy
from montaj.pipeline import render
from montaj.render.core import decode_rgb

ORACLE = Path.home() / "Projects/media/montaj-projects/_oracle/v2/src"
JPEG_PSNR_FLOOR = 42.0  # measured 43.7 .. 81 dB over the 19 oracle photos (6 checked here when present)


# ------------------------------------------------------------------ policy (CPU)
def test_load_policy_defaults_file_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONTAJ_GPU_STRICT", raising=False)
    assert load_policy(tmp_path / "none.toml") == GpuPolicy(strict=False, device="cuda")
    cfg = tmp_path / "c.toml"
    cfg.write_text('[gpu]\nstrict = true\ndevice = "cuda:0"\n')
    assert load_policy(cfg) == GpuPolicy(strict=True, device="cuda:0")
    monkeypatch.setenv("MONTAJ_GPU_STRICT", "0")
    assert not load_policy(cfg).strict
    cfg.write_text("[gpu]\nstrict = 1\n")
    with pytest.raises(RuntimeError, match="strict must be true/false"):
        load_policy(cfg)
    cfg.write_text("[gpu]\nfast = true\n")
    with pytest.raises(RuntimeError, match="takes only strict"):
        load_policy(cfg)


def test_machine_config_load_tolerates_the_gpu_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from montaj import config

    cfg = tmp_path / "c.toml"
    cfg.write_text('whisper_python = "w"\n[gpu]\nstrict = true\n')
    monkeypatch.setenv("MONTAJ_CONFIG", str(cfg))
    assert config.load(required=("whisper_python",)).whisper_python == "w"


def test_seams_count_and_strict_raises() -> None:
    lax = GpuPolicy()
    lax.gpu("decode")
    assert lax.summary() == "all"
    lax.cpu("text", "line 'x'", "set engine: gpu")
    lax.cpu("decode", "clip a.mp4", "re-encode")
    assert lax.summary() == "partial(decode,text)"
    assert lax.seams.n("text", "cpu") == 1 and lax.seams.n("decode", "gpu") == 1
    with pytest.raises(GpuStrictError) as ei:
        GpuPolicy(strict=True).cpu("text", "line 'x'", "set engine: gpu")
    assert str(ei.value) == "gpu-strict: line 'x' would run on the CPU: set engine: gpu"


def test_strict_refuses_libx264_before_writing_anything(tmp_path: Path) -> None:
    out = tmp_path / "o.mp4"
    with pytest.raises(GpuStrictError, match=r"libx264 encode of o\.mp4 would run on the CPU"):
        Encoder(out, 320, 320, 30, encoder="libx264", policy=GpuPolicy(strict=True))
    assert not out.exists()


def test_non_strict_photo_stays_on_pil_for_film2_parity(tmp_path: Path) -> None:
    jpg = tmp_path / "p.jpg"
    Image.new("RGB", (8, 8), (9, 8, 7)).save(jpg)
    pol = GpuPolicy()
    decode_rgb(str(jpg), "cuda" if torch.cuda.is_available() else "cpu", pol)
    assert pol.seams.n("photo", "cpu") == 1 and pol.seams.n("photo", "gpu") == 0


def test_photo_decode_cpu_paths(tmp_path: Path) -> None:
    png = tmp_path / "p.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(png)
    lax = GpuPolicy()
    assert decode_rgb(str(png), "cpu", lax).shape == (8, 8, 3)
    assert lax.seams.n("photo", "cpu") == 1 and lax.summary() == "partial(photo)"
    with pytest.raises(GpuStrictError, match="photo p.png"):
        decode_rgb(str(png), "cpu", GpuPolicy(strict=True))


def test_cli_prints_one_gpu_strict_err_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    (tmp_path / "montaj.yaml").write_text("x")

    def boom(*_a, **_k):
        raise GpuStrictError("line 'hi'", "set engine: gpu")

    monkeypatch.setattr("montaj.pipeline.render", boom)
    assert cli.main(["-C", str(tmp_path), "render"]) == 1
    assert capsys.readouterr().out.strip() == "ERR gpu-strict: line 'hi' would run on the CPU: set engine: gpu"


# ------------------------------------------------------------------ GPU
def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean()) * 255**2
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


@pytest.mark.gpu
def test_nvjpeg_matches_pil(tmp_path: Path) -> None:
    # Photo-like content: gradients and soft edges (white noise is a worst case for chroma upsampling).
    y, x = np.mgrid[0:96, 0:128]
    soft = np.stack([x * 2, y * 2.5, 128 + 100 * np.sin(x / 9) * np.cos(y / 11)], -1).clip(0, 255)
    synth = tmp_path / "s.jpg"
    Image.fromarray(soft.astype(np.uint8)).save(synth, quality=90)
    paths = [synth, *sorted(ORACLE.glob("*.jpg"))[:6]]
    for p in paths:
        pol = GpuPolicy(strict=True)
        got = decode_rgb(str(p), "cuda", pol)
        want = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
        assert pol.seams.n("photo", "gpu") == 1 and pol.seams.n("photo", "cpu") == 0
        floor = JPEG_PSNR_FLOOR if p != synth else 35.0  # clipped saturated gradient: harsher on chroma
        assert got.shape == want.shape and _psnr(got, want) >= floor, p.name


def _film(tmp: Path, *, text_line: str) -> Path:
    assets = tmp / "assets"
    assets.mkdir(exist_ok=True)
    Image.new("RGB", (160, 240), (40, 90, 150)).save(assets / "a.jpg", quality=95)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-colorspace", "bt709", str(assets / "rec.mp4")],
        check=True,
    )
    (tmp / "montaj.yaml").write_text(
        "video: {size: 360x640, fps: 30, look: warm-film, motion_blur: 1}\nassets: assets\nshots:\n"
        "  - {clip: rec, hold: 12f}\n  - {photo: a, hold: 12f, in: {type: fade, dur: 4f}}\n"
        "text:\n  - from: 0s\n    to: 1s\n    lines:\n      - {" + text_line + "}\n"
    )
    return tmp / "montaj.yaml"


GPU_LINE = "text: Hello, style: sans, y: 300, at: 0s, engine: gpu"


@pytest.mark.gpu
def test_strict_film_with_clip_text_and_photo_is_all_gpu(tmp_path: Path) -> None:
    spec = _film(tmp_path, text_line=GPU_LINE)
    pol = GpuPolicy(strict=True)
    result = render(spec, "final", pol)
    assert result.gpu == "all" and result.rendered > 0
    for stage in ("decode", "photo", "text", "encode"):  # every seam ran, on the GPU
        assert pol.seams.n(stage, "gpu") > 0 and pol.seams.n(stage, "cpu") == 0, stage
    # The policy is not in the hash: a lax re-render finds every segment cached.
    again = render(spec, "final", GpuPolicy())
    assert again.rendered == 0 and again.cached == result.rendered


@pytest.mark.gpu
def test_strict_with_broken_nvenc_errors_instead_of_x264(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _film(tmp_path, text_line=GPU_LINE)
    monkeypatch.setattr(encode, "nvenc_usable", lambda: False)
    with pytest.raises(GpuStrictError) as ei:
        render(spec, "final", GpuPolicy(strict=True))
    assert str(ei.value).startswith("gpu-strict: libx264 encode of ")
    assert "would run on the CPU" in str(ei.value)
    assert not list((tmp_path / "build" / "segments").glob("*.mp4")) and not (tmp_path / "out" / "final.mp4").exists()
    lax = GpuPolicy()
    assert render(spec, "final", lax).gpu == "partial(photo,encode)"  # the same break, not strict: x264 falls back


@pytest.mark.gpu
def test_strict_names_the_chromium_text_line(tmp_path: Path) -> None:
    spec = _film(tmp_path, text_line="text: Hello, style: serif, y: 300, at: 0s")
    with pytest.raises(GpuStrictError, match=r"line 'Hello' \(style serif\) rasterised by Chromium.*engine: gpu"):
        render(spec, "final", GpuPolicy(strict=True))
    lax = GpuPolicy()
    assert render(spec, "final", lax).gpu == "partial(photo,text)"
    assert lax.seams.n("text", "cpu") == 1


@pytest.mark.gpu
def test_device_pinned_by_name_and_uuid() -> None:
    from montaj.gpu_policy import resolve_device

    props = torch.cuda.get_device_properties(0)
    assert resolve_device("cuda:0") == "cuda:0"
    assert resolve_device(str(props.uuid)) == "cuda:0"
    with pytest.raises(ValueError, match="no CUDA device with that name"):
        resolve_device("Intel UHD Graphics")
    with pytest.raises(ValueError, match="only"):
        resolve_device("cuda:9")
