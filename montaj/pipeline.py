"""Render a spec to out/<mode>.mp4, skipping segments whose hash is already encoded."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

import torch

from montaj import __version__
from montaj.encode import Encoder, concat
from montaj.project import file_sha, find_photo, segment_hash
from montaj.render.canvas import Canvas
from montaj.render.frame import Renderer
from montaj.spec import Spec, load_spec
from montaj.timeline import Segment, resolve

# Bump when a pixel changes. Concatenated onto __version__ inside every segment hash.
RENDER_REV = "+1"


def engine_version() -> str:
    return __version__ + RENDER_REV


@dataclass(frozen=True)
class RenderResult:
    path: Path
    n_frames: int
    seconds: float
    size: tuple[int, int]
    rendered: int
    cached: int


def _apply_mode(spec: Spec, mode: str) -> Spec:
    """Preview halves the frame (down to an even size) and caps motion blur at 2.

    Applied before resolve, so a preview hash cannot hit a final segment.
    """
    if mode == "final":
        return spec
    if mode != "preview":
        raise ValueError(f"mode: must be preview or final, got {mode!r}")
    w_s, _, h_s = spec.video.size.lower().partition("x")
    w = (int(w_s) // 2) & ~1
    h = (int(h_s) // 2) & ~1
    video = spec.video.model_copy(update={
        "size": f"{w}x{h}",
        "motion_blur": min(2, spec.video.motion_blur),
    })
    return spec.model_copy(update={"video": video})


def _asset_shas(spec: Spec, spec_dir: Path) -> dict[str, str]:
    assets = spec_dir / spec.assets
    ids: list[str] = []
    for shot in spec.shots:
        if shot.photo:
            ids.append(shot.photo)
        if shot.wall:
            ids.extend(p.photo for p in shot.wall.prints)
    out: dict[str, str] = {}
    for pid in ids:
        if pid not in out:
            out[pid] = file_sha(find_photo(assets, pid))
    return out


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def _u8(frame: torch.Tensor):
    return (frame * 255.0 + 0.5).byte().permute(1, 2, 0).contiguous().cpu().numpy()


def _encode_segment(renderer: Renderer, seg: Segment, dest: Path, fps: float, log: Path) -> None:
    """Encode to a per-attempt temp file, then `os.replace` it onto the cache file.

    The name is `.<hash>.<pid>-<random>.partial.mp4`. A shared `.<hash>.partial.mp4`
    lets one render rename the file while another is still writing it. The `.mp4`
    suffix stays so Encoder writes faststart. A crash unlinks only this attempt.
    """
    tmp = dest.with_name(f".{dest.stem}.{os.getpid()}-{secrets.token_hex(4)}.partial.mp4")
    try:
        with Encoder(tmp, renderer.cv.W, renderer.cv.H, fps, log=log) as enc:
            for f in range(seg.start, seg.end):
                enc.write(_u8(renderer.frame(f)))
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def render(spec_path: Path, mode: str = "preview") -> RenderResult:
    spec_path = Path(spec_path)
    spec_dir = spec_path.parent
    spec = _apply_mode(load_spec(spec_path), mode)
    tl = resolve(spec, spec_dir)
    shas = _asset_shas(spec, spec_dir)
    version = engine_version()
    log = spec_dir / "build" / "render.log"
    seg_dir = spec_dir / "build" / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    out = spec_dir / "out" / f"{mode}.mp4"
    W, H = tl.size
    _log(log, f"render {mode} {W}x{H} motion_blur {spec.video.motion_blur} frames {tl.n_frames}")

    rendered = 0
    cached = 0
    chunks: list[Path] = []
    with torch.no_grad():
        cv = Canvas(W, H, device="cuda")
        renderer = Renderer(tl, spec_dir, cv)
        try:
            for seg in tl.segments:
                digest = segment_hash(tl, seg, shas, version)
                dest = seg_dir / f"{digest}.mp4"
                chunks.append(dest)
                if dest.is_file() and dest.stat().st_size > 0:
                    cached += 1
                    _log(log, f"segment {seg.index} {seg.start}-{seg.end} hit {digest}")
                    continue
                _log(log, f"segment {seg.index} {seg.start}-{seg.end} miss {digest}")
                _encode_segment(renderer, seg, dest, tl.fps, log)
                rendered += 1
        finally:
            del renderer, cv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    concat(
        chunks,
        out,
        fps=tl.fps,
        frames=[seg.end - seg.start for seg in tl.segments],
        log=log,
    )
    _log(log, f"concat {out.name} rendered {rendered} cached {cached}")
    return RenderResult(
        path=out.resolve(),
        n_frames=tl.n_frames,
        seconds=tl.n_frames / tl.fps,
        size=(W, H),
        rendered=rendered,
        cached=cached,
    )
