"""Subject mattes (#36): ingest writes a matte keyed by source bytes; layers cut out by it; text sits behind it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from montaj.gpu_policy import GpuPolicy, GpuStrictError
from montaj.matte import find_matte, make_mattes, matte_path, refine, steady
from montaj.pipeline import _asset_shas
from montaj.project import segment_hash
from montaj.render.canvas import Canvas
from montaj.render.clip import ClipSource
from montaj.render.frame import Renderer
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve
from tests.test_layers import _ffmpeg, _project

# rec.mkv is 320x240; a 108x192 frame covers it at 0.8x, showing source x 92.5..227.5, so the matte's
# edge at source x = 160 lands at frame x = 54.
EDGE = 54


def _matte(tmp: Path, stem: str = "rec", ext: str = "mkv") -> Path:
    """A hand-made matte at the path ingest would use: the clip's left half is the subject."""
    _project(tmp, "  - {clip: rec, hold: 1s}\n")  # makes the assets
    assets = tmp / "assets"
    out = matte_path(assets, assets / f"{stem}.{ext}")
    out.parent.mkdir(exist_ok=True)
    _ffmpeg("-f", "lavfi", "-i", "color=black:size=320x240:rate=30:duration=3,"
            "drawbox=x=0:y=0:w=160:h=240:color=white:t=fill,format=yuv420p",
            "-c:v", "libx264", "-qp", "0", str(out))
    return out


def _load(tmp: Path, shots: str, extra: str = "") -> Path:
    path = _project(tmp, shots)
    if extra:
        path.write_text(path.read_text() + extra)
    return path


def _errors(path: Path) -> list[str]:
    with pytest.raises(SpecError) as ei:
        load_spec(path)
    return ei.value.errors


_BEHIND = """text:
  - from: 0s
    to: 3s
    {behind}lines:
      - {{text: MMMM, style: sans, size: 220, y: 960, at: 0s, engine: gpu}}
"""


# ---------------------------------------------------------------- ingest


def _half(rgb: torch.Tensor) -> torch.Tensor:
    """Fake segmenter: the left half of the frame is the subject (at a coarse 1/4 resolution)."""
    h, w = int(rgb.shape[1]) // 4, int(rgb.shape[2]) // 4
    out = torch.zeros(1, h, w, device=rgb.device)
    out[:, :, : w // 2] = 1.0
    return out


def test_ingest_writes_a_matte_keyed_by_source_bytes_and_reuses_it(tmp_path: Path) -> None:
    _project(tmp_path, "  - {clip: rec, hold: 1s}\n")
    assets = tmp_path / "assets"
    src = assets / "rec.mkv"
    calls = []

    def build(device):
        calls.append(device)
        return _half

    first = make_mattes(assets, [("rec", src)], device="cpu", segmenter=build)
    assert [r.made for r in first] == [True] and first[0].frames == 90 and calls
    path = find_matte(assets, src, "rec")
    assert path == first[0].path and path.parent.name == "mattes"
    index = json.loads((assets / "mattes" / "index.json").read_text())
    assert index["rec.mkv"]["size"] == src.stat().st_size
    # It decodes through the clip path: left half subject, right half not.
    reader = ClipSource(path, 30.0)
    m = reader.frame(10.0, 0.0, 0.0, 320, 240, nframes=90)
    reader.close()
    assert float(m[0, :, :120].mean()) > 0.95 and float(m[0, :, 200:].mean()) < 0.05
    # Same bytes: nothing recomputed, the model is never built.
    calls.clear()
    again = make_mattes(assets, [("rec", src)], device="cpu", segmenter=build)
    assert [r.made for r in again] == [False] and not calls


def test_strict_policy_refuses_cpu_matte_inference(tmp_path: Path) -> None:
    _project(tmp_path, "  - {clip: rec, hold: 1s}\n")
    assets = tmp_path / "assets"
    pol = GpuPolicy(strict=True)
    with pytest.raises(GpuStrictError, match="matte"):
        make_mattes(assets, [("rec", assets / "rec.mkv")], device="cpu", policy=pol,
                    segmenter=lambda d: pytest.fail("model built before the strict check"))
    assert not (assets / "mattes").exists() or not list((assets / "mattes").glob("*.mp4"))
    # Not strict: the CPU run is counted as a CPU matte seam.
    loose = GpuPolicy()
    make_mattes(assets, [("rec", assets / "rec.mkv")], device="cpu", policy=loose, segmenter=lambda d: _half)
    assert "matte" in loose.cpu_stages()


def test_refine_snaps_to_the_frame_and_steady_keeps_motion() -> None:
    rgb = torch.zeros(3, 64, 64)
    rgb[:, :, :32] = 1.0  # a hard edge at x = 32
    coarse = torch.zeros(1, 8, 8)
    coarse[:, :, :4] = 1.0
    m = refine(coarse, rgb)
    assert m.shape == (1, 64, 64)
    assert float(m[0, :, :28].min()) > 0.9 and float(m[0, :, 36:].max()) < 0.1
    prev = torch.full((1, 4, 4), 0.5)
    cur = prev.clone()
    cur[0, 0, 0] = 0.6  # shimmer: halfway back to the previous value
    cur[0, 3, 3] = 1.0  # motion: taken as is
    out = steady(cur, prev)
    assert float(out[0, 0, 0]) == pytest.approx(0.55) and float(out[0, 3, 3]) == 1.0


# ---------------------------------------------------------------- validate + hash


def test_validate_names_the_ingest_command_and_bad_values(tmp_path: Path) -> None:
    errs = _errors(_load(tmp_path, "  - {clip: rec, hold: 1s, matte: subject}\n"))
    assert any("montaj ingest --matte rec" in e for e in errs), errs
    errs = _errors(_load(tmp_path, "  - {clip: rec, hold: 1s}\n", _BEHIND.format(behind="behind: subject\n    ")))
    assert any("text[0].behind" in e and "no shot has a subject matte" in e for e in errs), errs
    _matte(tmp_path)
    errs = _errors(_load(tmp_path, "  - {clip: rec, hold: 1s, matte: person}\n"))
    assert any("shots[0].matte" in e and "unknown 'person'" in e for e in errs), errs
    errs = _errors(_load(tmp_path, "  - {photo: tag, hold: 1s, matte: subject}\n"))
    assert any("matte needs a clip shot" in e for e in errs), errs
    errs = _errors(_load(tmp_path, "  - {hold: 1s, layers: [{clip: rec, mask: subject, box: [0, 0, 50, 50], shadow: 0.5}]}\n"))
    assert any("layers[0].shadow" in e for e in errs), errs
    load_spec(_load(tmp_path, "  - {clip: rec, hold: 1s, matte: subject}\n",
                    _BEHIND.format(behind="behind: subject\n    ")))


def test_matte_bytes_reach_the_segment_hash_and_unset_stays_out(tmp_path: Path) -> None:
    _matte(tmp_path)

    def hashes(shots: str) -> list[str]:
        path = _load(tmp_path, shots)
        spec = load_spec(path)
        tl = resolve(spec, tmp_path)
        shas = _asset_shas(spec, tmp_path)
        return [segment_hash(tl, seg, shas, "e1") for seg in tl.segments], shas

    plain, plain_shas = hashes("  - {clip: rec, hold: 1s}\n")
    assert not any(k.startswith("matte:") for k in plain_shas)
    matted, shas = hashes("  - {clip: rec, hold: 1s, matte: subject}\n")
    assert "matte:clip:rec" in shas and matted != plain
    # A different matte for the same source bytes is a different picture.
    path = find_matte(tmp_path / "assets", tmp_path / "assets" / "rec.mkv", "rec")
    path.write_bytes(path.read_bytes() + b"\0")
    assert hashes("  - {clip: rec, hold: 1s, matte: subject}\n")[0] != matted
    assert hashes("  - {clip: rec, hold: 1s}\n")[0] == plain


# ---------------------------------------------------------------- render


def _render(tmp: Path, shots: str, extra: str = "", f: int = 30) -> torch.Tensor:
    tl = resolve(load_spec(_load(tmp, shots, extra)), tmp)
    return Renderer(tl, tmp, Canvas(tl.size[0], tl.size[1], device="cpu")).frame(f)


def test_text_behind_draws_under_the_subject_and_over_the_rest(tmp_path: Path) -> None:
    _matte(tmp_path)
    shot = "  - {clip: rec, hold: 2s, matte: subject}\n"
    bare = _render(tmp_path, shot)
    front = _render(tmp_path, shot, _BEHIND.format(behind=""))
    behind = _render(tmp_path, shot, _BEHIND.format(behind="behind: subject\n    "))
    text = (front - bare).abs().amax(0) > 0.2
    left, right = text[:, : EDGE - 4], text[:, EDGE + 4:]
    assert left.sum() > 20 and right.sum() > 20  # the word spans both sides of the matte edge
    # Over the subject the picture wins; elsewhere the text is drawn as before.
    assert (behind - bare)[:, :, : EDGE - 4].abs().amax() < 0.02
    assert torch.allclose(behind[:, :, EDGE + 4:], front[:, :, EDGE + 4:], atol=0.02)


def test_mask_subject_cuts_a_layer_out_over_the_background(tmp_path: Path) -> None:
    _matte(tmp_path)
    out = {}
    for name, layer in (("plain", "{clip: rec}"), ("cut", "{clip: rec, mask: subject}")):
        tl = resolve(load_spec(_load(tmp_path, f"  - {{hold: 1s, layers: [{layer}]}}\n")), tmp_path)
        rend = Renderer(tl, tmp_path, Canvas(tl.size[0], tl.size[1], device="cpu"))
        out[name] = rend._eval(tl.plan(10).scene, 10.0)
        out["bg"] = rend._background().expand_as(out[name])
        if name == "cut":
            subject = rend._subject(tl.plan(10).scene, 10.0)
    assert torch.allclose(out["cut"][:, :, : EDGE - 4], out["plain"][:, :, : EDGE - 4], atol=0.02)
    assert torch.allclose(out["cut"][:, :, EDGE + 4:], out["bg"][:, :, EDGE + 4:], atol=0.02)
    assert float(subject[0, :, : EDGE - 4].min()) > 0.95 and float(subject[0, :, EDGE + 4:].max()) < 0.05


@pytest.mark.gpu
def test_strict_matte_ingest_and_text_behind_render_are_all_gpu(tmp_path: Path) -> None:
    from montaj.pipeline import render

    # NVENC rejects frames under ~145 px a side, so this canvas is larger than the CPU tests'.
    path = _project(tmp_path, "  - {clip: talk, hold: 1s, matte: subject}\n", size="180x320")
    assets = tmp_path / "assets"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", str(assets / "talk.mp4"))
    pol = GpuPolicy(strict=True)
    made = make_mattes(assets, [("talk", assets / "talk.mp4")], device="cuda", policy=pol,
                       segmenter=lambda d: _half)
    assert made[0].made and pol.seams.n("matte", "gpu") > 0 and not pol.cpu_stages()
    path.write_text(path.read_text() + _BEHIND.format(behind="behind: subject\n    "))
    result = render(path, "final", GpuPolicy(strict=True))
    assert result.gpu == "all" and result.rendered > 0
