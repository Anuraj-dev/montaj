"""NVDEC clip decode into CUDA tensors, the GPU twin of `clip.py`'s ffmpeg reader.

`choose(path, device)` is the one place that decides which decoder a clip source uses.
It returns a `Choice`; every caller (ClipSource, its log line, a future strict mode)
reads that, never re-derives it.

`NvdecReader` reproduces the ffmpeg chain `ClipSource._spawn` runs, frame for frame:
- `-ss clip_in` before `-i`: timestamps shift by `-(format start_time + clip_in)`; frames
  that land before 0 are dropped (accurate seek).
- `fps=<film fps>` (round=near): output frame n holds the newest input frame whose
  rescaled pts is <= `round(first pts) + n`; at EOF the last frame repeats up to its end.
- scale/crop: luma uses swscale bicubic (B=0, C=0.6), widened when shrinking. Chroma is
  centre-sited whatever the stream tags (vf_scale's default) and follows swscale's three
  writers: unscaled (output == cropped source) takes the nearest 2x2 sample; a scaled even
  output width shares one bicubic chroma sample per pixel pair; an odd width interpolates
  chroma per pixel. Then YUV -> RGB with the stream's matrix (bt709 when tagged, bt601
  otherwise, as swscale does), its range (limited unless tagged full), and an empirical
  bias for swscale's fixed-point rounding (`_BIAS`, `_PATH_BIAS`).
- seeking: `PyNvDemuxer.Seek` takes a frame index, not seconds; a seek that lands after the
  target falls back to decoding from the start.
Pixels match the software path within a PSNR pinned in `tests/test_nvdec.py`; the frame
index matches exactly.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import torch

from montaj.encode import _pynvc

try:  # The wheel is a hard dependency; a broken driver still has to fall back.
    _nvc = _pynvc()  # quiets its stdout logger before import
    _IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - depends on the machine
    _nvc = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

FFPROBE = "ffprobe"
_CODECS = {"h264": "H264", "hevc": "HEVC", "av1": "AV1", "vp9": "VP9", "vp8": "VP8", "mpeg2video": "MPEG2"}
# swscale's default bicubic: B = 0, C = 0.6.
_CUBIC_C = 0.6
# swscale's fixed-point YUV -> RGB runs darker than the formula, by an amount that depends
# on the matrix (fitted on unscaled frames of real clips) and on the code path: the scaled
# half-chroma and full-chroma writers round differently from the unscaled tables (measured
# on a neutral-chroma clip). Applied before rounding, in RGB levels.
_PATH_BIAS = {"unscaled": 0.0, "even": -0.94, "odd": 1.23}
_BIAS = {("bt709", False): (-0.95, -1.56, -1.10), ("bt601", False): (-0.57, -0.58, -0.57)}


@dataclass(frozen=True)
class Stream:
    codec: str
    width: int
    height: int
    tb: Fraction
    start: Fraction  # format start_time, seconds (ffmpeg offsets -ss by it)
    matrix: str  # "bt709" or "bt601"
    full_range: bool
    pix_fmt: str


@dataclass(frozen=True)
class Choice:
    """Which decoder a clip uses, and why. `name` is "nvdec" or "ffmpeg"."""

    name: str
    reason: str
    stream: Stream | None = None

    @property
    def gpu(self) -> bool:
        return self.name == "nvdec"


def _frac(value, default: str = "0") -> Fraction:
    try:
        return Fraction(str(value if value not in (None, "N/A") else default))
    except (ValueError, ZeroDivisionError):
        return Fraction(default)


def probe(path) -> Stream | None:
    """Container facts NVDEC does not report the way ffmpeg uses them. No decode runs."""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "format=start_time:stream=codec_name,width,height,time_base,pix_fmt,color_space,color_range",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return None
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        return None
    st = streams[0]
    space = st.get("color_space", "unknown")
    return Stream(
        codec=st.get("codec_name", ""),
        width=int(st.get("width", 0)),
        height=int(st.get("height", 0)),
        tb=_frac(st.get("time_base"), "1/90000"),
        start=_frac((data.get("format") or {}).get("start_time")),
        matrix="bt709" if space == "bt709" else "bt601",
        full_range=st.get("color_range") == "pc",
        pix_fmt=st.get("pix_fmt", ""),
    )


def choose(path, device) -> Choice:
    """The single decoder decision for one clip file on one device."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return Choice("ffmpeg", "device is not cuda")
    if _nvc is None:
        return Choice("ffmpeg", f"PyNvVideoCodec unavailable ({_IMPORT_ERROR})")
    if not torch.cuda.is_available():
        return Choice("ffmpeg", "cuda unavailable")
    stream = probe(path)
    if stream is None:
        return Choice("ffmpeg", "ffprobe found no video stream")
    name = _CODECS.get(stream.codec)
    if name is None:
        return Choice("ffmpeg", f"codec {stream.codec} has no NVDEC path")
    if stream.pix_fmt not in ("yuv420p", "yuvj420p"):
        return Choice("ffmpeg", f"pixel format {stream.pix_fmt} is not 8-bit 4:2:0")
    try:
        caps = _nvc.GetDecoderCaps(
            gpuid=dev.index or 0, codec=getattr(_nvc.cudaVideoCodec, name),
            chromaformat=_nvc.cudaVideoChromaFormat.__members__["420"], bitdepth=8,
        )
    except Exception as exc:
        return Choice("ffmpeg", f"NVDEC caps query failed ({exc})")
    if not caps.get("supported"):
        return Choice("ffmpeg", f"NVDEC does not support {stream.codec} on this GPU")
    if not (caps["width_min"] <= stream.width <= caps["width_max"]
            and caps["height_min"] <= stream.height <= caps["height_max"]):
        return Choice("ffmpeg", f"{stream.width}x{stream.height} outside NVDEC limits")
    if stream.pix_fmt == "yuvj420p":
        stream = Stream(**{**stream.__dict__, "full_range": True})
    return Choice("nvdec", f"{stream.codec} {stream.width}x{stream.height} {stream.matrix}", stream)


def _rescale_near(a: int, b: Fraction) -> int:
    """`av_rescale_q_rnd(a, ..., AV_ROUND_NEAR_INF)`: a * b, halves away from zero."""
    q = a * b
    n, d = q.numerator, q.denominator
    return (2 * abs(n) + d) // (2 * d) * (1 if n >= 0 else -1)


def _cubic(x: torch.Tensor) -> torch.Tensor:
    # Mitchell-Netravali with B = 0 is Keys' cubic, a = -C.
    a = -_CUBIC_C
    x = x.abs()
    near = ((a + 2) * x - (a + 3)) * x * x + 1
    far = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return torch.where(x < 1, near, torch.where(x < 2, far, torch.zeros_like(x)))


def _onehot(index: torch.Tensor, n_in: int, device) -> torch.Tensor:
    out = torch.zeros(len(index), n_in, device=device)
    out[torch.arange(len(index), device=device), index.clamp(0, n_in - 1)] = 1.0
    return out


def _weights(n_out: int, span: float, origin: float, n_in: int, device) -> torch.Tensor:
    """(n_out, n_in) resampling matrix. Output pixel i centre maps to input
    coordinate `origin + (i + .5) * span / n_out - .5`; edges replicate."""
    scale = span / n_out
    pos = origin + (torch.arange(n_out, device=device, dtype=torch.float64) + 0.5) * scale - 0.5
    width = max(1.0, scale)
    taps = int(math.ceil(2 * width)) * 2 + 1
    base = torch.floor(pos) - taps // 2 + 1
    idx = base[:, None] + torch.arange(taps, device=device, dtype=torch.float64)[None, :]
    w = _cubic((pos[:, None] - idx) / width)
    w = w / w.sum(dim=1, keepdim=True)
    out = torch.zeros(n_out, n_in, device=device, dtype=torch.float64)
    out.scatter_add_(1, idx.clamp(0, n_in - 1).long(), w)
    return out.float()


@dataclass(frozen=True)
class _Geometry:
    """Where the output samples the luma plane: `(x0, y0, cw, ch)` in source px."""

    x0: float
    y0: float
    cw: float
    ch: float
    # Integer crop origin, for the chroma plane pointer (ffmpeg crops chroma at x0 >> 1).
    ix: int
    iy: int
    # Width swscale writes, and where the output starts in it (the cover crop's x).
    stage_w: int
    stage_x: int


def geometry(src_w: int, src_h: int, w: int, h: int, crop: tuple | None) -> _Geometry:
    if crop is not None:
        x0, y0, x1, y1 = (int(c) for c in crop)
        return _Geometry(float(x0), float(y0), float(x1 - x0), float(y1 - y0), x0, y0, w, 0)
    # scale=w:h:force_original_aspect_ratio=increase, then crop=w:h centred (on rgb24, exact).
    sw = max(_rescale_near(h, Fraction(src_w, src_h)), w)
    sh = max(_rescale_near(w, Fraction(src_h, src_w)), h)
    # crop=w:h centres at (in - out) / 2 and rounds a half to even (lrint): 291.5 -> 292.
    cx, cy = round((sw - w) / 2), round((sh - h) / 2)
    fx, fy = src_w / sw, src_h / sh
    return _Geometry(cx * fx, cy * fy, w * fx, h * fy, 0, 0, sw, cx)


class _Converter:
    """NV12 (H*3/2, W) uint8 -> RGB (3, h, w) uint8 on the GPU, swscale-style."""

    def __init__(self, stream: Stream, w: int, h: int, crop: tuple | None, device):
        W, H = stream.width, stream.height
        g = geometry(W, H, w, h, crop)
        self._W, self._H = W, H
        self._ly = _weights(h, g.ch, g.y0, H, device)
        self._lx = _weights(w, g.cw, g.x0, W, device)
        # vf_scale leaves swscale's chroma position at its default, centre, whatever the
        # stream tags (checked: `in_h_chr_pos=128` gives the same bytes as the default).
        unscaled = g.cw == w and g.ch == h
        if unscaled:
            # Same size in and out: swscale's unscaled yuv2rgb repeats each chroma sample over
            # its 2x2 block. A crop moves the chroma pointer to (ix >> 1, iy >> 1) first; the
            # cover crop cuts the converted rgb24 frame instead.
            ry = torch.arange(h, device=device)
            rx = torch.arange(w, device=device)
            if crop is not None:
                iy, ix = (g.iy >> 1) + ry // 2, (g.ix >> 1) + rx // 2
            else:
                iy, ix = (int(g.y0) + ry) // 2, (int(g.x0) + rx) // 2
            self._cy = _onehot(iy, H // 2, device)
            self._cx = _onehot(ix, W // 2, device)
        elif crop is None:
            # Scaling the whole frame: chroma sits at its own siting in luma px.
            self._cy = _weights(h, g.ch / 2, (g.y0 - 0.5) / 2 + 0.25, H // 2, device)
            cspan, cbase = W / 2, 0.0  # the whole stage row; stage_x picks the columns
        else:
            # ffmpeg's crop moves the chroma pointer to (ix >> 1, iy >> 1); swscale then
            # scales a ceil(cw / 2) chroma plane over the same output.
            self._cy = _weights(h, -(-int(g.ch) // 2), float(g.iy >> 1), H // 2, device)
            cspan, cbase = -(-int(g.cw) // 2), float(g.ix >> 1)
        if not unscaled:
            # Packed RGB shares one chroma sample per output pixel pair unless the width
            # swscale writes is odd (utils.c forces full chroma interpolation then).
            cols = torch.arange(g.stage_x, g.stage_x + w, device=device)
            if g.stage_w % 2 == 0:
                self._cx = _weights(g.stage_w // 2, cspan, cbase, W // 2, device)[cols // 2]
            else:
                self._cx = _weights(g.stage_w, cspan, cbase, W // 2, device)[cols]
        path = "unscaled" if unscaled else ("even" if g.stage_w % 2 == 0 else "odd")
        kr, kb = (0.2126, 0.0722) if stream.matrix == "bt709" else (0.299, 0.114)
        kg = 1 - kr - kb
        if stream.full_range:
            ys, yo, cs = 1.0, 0.0, 1.0
        else:
            ys, yo, cs = 255 / 219, 16.0, 255 / 224
        self._y = (ys, yo)
        bias = _BIAS.get((stream.matrix, stream.full_range), (0.0, 0.0, 0.0))
        self._bias = (torch.tensor(bias, device=device, dtype=torch.float32) + _PATH_BIAS[path])[:, None, None]
        self._m = torch.tensor(
            [[0.0, 2 * (1 - kr)],
             [-2 * kb * (1 - kb) / kg, -2 * kr * (1 - kr) / kg],
             [2 * (1 - kb), 0.0]],
            device=device, dtype=torch.float32,
        ) * cs

    def __call__(self, nv12: torch.Tensor) -> torch.Tensor:
        W, H = self._W, self._H
        y = nv12[:H, :W].float()
        uv = nv12[H:H + H // 2, :W].reshape(H // 2, W // 2, 2).float()
        ys, yo = self._y
        luma = (self._ly @ y @ self._lx.T - yo) * ys
        u = self._cy @ uv[..., 0] @ self._cx.T - 128
        v = self._cy @ uv[..., 1] @ self._cx.T - 128
        rgb = luma[None] + torch.einsum("ck,khw->chw", self._m, torch.stack((u, v)))
        rgb += self._bias
        return rgb.round_().clamp_(0, 255).to(torch.uint8)


def _identity(frame: torch.Tensor) -> torch.Tensor:
    return frame


class NvdecReader:
    """Forward reader for one window of `fps=…,scale…` output: `read()` returns the next
    frame in its compact form, or None at the end (after `nframes` or EOF); `expand(frame)`
    turns that into (3, h, w) uint8 RGB on the GPU.

    The compact form is whichever is smaller: the decoded NV12 surface (1.5 bytes per source
    px, converted on every expand) or the converted RGB (3 bytes per output px). A repeated
    frame (fps up-conversion) is the same tensor, so a cache holding both pays once.
    """

    def __init__(self, path, choice: Choice, clip_in: float, fps: float, w: int, h: int,
                 nframes: int, crop: tuple | None, device):
        if not choice.gpu or choice.stream is None:
            raise ValueError("NvdecReader needs an nvdec Choice")
        self._stream = choice.stream
        self._device = torch.device(device)
        self._fps = Fraction(fps).limit_denominator(1001)
        self._nframes = int(nframes)
        self._emitted = 0
        self._convert = _Converter(self._stream, int(w), int(h), crop, self._device)
        self.compact = self._stream.width * self._stream.height * 3 // 2 < int(w) * int(h) * 3
        self.expand = self._convert if self.compact else _identity
        tb = self._stream.tb
        # -ss before -i: ts_offset is microseconds, added to pts in the stream's time base.
        offset_us = _rescale_near(1, self._stream.start * 1_000_000) + _rescale_near(1, Fraction(clip_in) * 1_000_000)
        self._offset = _rescale_near(-offset_us, Fraction(1, 1_000_000) / tb)
        self._to_out = tb * self._fps  # pts (stream tb) -> output frame slots
        self._demux = _nvc.CreateDemuxer(str(path))
        with torch.cuda.device(self._device):
            self._cuda_stream = torch.cuda.current_stream(self._device)
            self._dec = _nvc.CreateDecoder(
                gpuid=self._device.index or 0, codec=self._demux.GetNvCodecId(), cudacontext=0,
                cudastream=self._cuda_stream.cuda_stream, usedevicememory=True,
            )
        self._pending: list[tuple[int, torch.Tensor]] = []  # decoded, display order, shifted pts
        self._eof = False
        self._cur: tuple[int, torch.Tensor] | None = None  # frame on screen, rounded slot
        self._next: int | None = None  # next output slot
        self._end: int | None = None  # EOF slot (exclusive)
        self._last_pts: int | None = None
        self._step: int | None = None
        self._seek(path, clip_in)

    def _seek(self, path, clip_in: float) -> None:
        # `PyNvDemuxer.Seek` takes a frame index and returns the first packet of a keyframe
        # near it - measured: index 30 at 30 fps lands on 0 s, 45 on 1 s - so its landing is
        # checked: a keyframe after the first frame we need means decode from the start.
        self._first_pkt = None
        if clip_in <= 0:
            return
        target = -self._offset  # raw pts of the first frame ffmpeg keeps
        idx = int(math.floor(clip_in * float(self._demux.FrameRate())))
        try:
            pkt = self._demux.Seek(idx)
        except Exception:  # index past the end or no index entry
            pkt = None
        if pkt is not None and pkt.bsl > 0 and pkt.pts <= target:
            self._first_pkt = pkt
        else:
            self._demux = _nvc.CreateDemuxer(str(path))

    def live(self) -> bool:
        return self._dec is not None

    def close(self) -> None:
        self._pending.clear()
        self._cur = None
        self._dec = None
        self._demux = None

    def _pull(self) -> bool:
        """Decode one packet's worth of frames into `_pending`. False at EOF."""
        if self._eof:
            return False
        if self._first_pkt is not None:
            pkt, self._first_pkt = self._first_pkt, None
        else:
            pkt = self._demux.Demux()
        frames = self._dec.Decode(pkt)
        for f in frames:
            pts = int(f.getPTS()) + self._offset
            if self._last_pts is not None and self._step is None:
                self._step = pts - self._last_pts
            self._last_pts = pts
            if pts < 0:  # accurate seek drops frames before clip_in
                continue
            # The decoder reuses its surfaces; copy before the next Decode.
            nv12 = torch.from_dlpack(f).clone()
            self._pending.append((pts, nv12))
        if pkt.bsl == 0:
            self._eof = True
        return True

    def _slot(self, pts: int) -> int:
        return _rescale_near(pts, self._to_out)

    def read(self) -> torch.Tensor | None:
        if self._dec is None or self._emitted >= self._nframes:
            return None
        with torch.cuda.device(self._device):
            while True:
                if self._next is None:
                    while not self._pending and self._pull():
                        pass
                    if not self._pending:
                        return None
                    pts, nv12 = self._pending.pop(0)
                    self._cur = (pts, nv12)
                    self._next = self._slot(pts)
                # Advance to the newest frame whose slot is <= next.
                while True:
                    while not self._pending and self._pull():
                        pass
                    if not self._pending:
                        break
                    if self._slot(self._pending[0][0]) > self._next:
                        break
                    self._cur = self._pending.pop(0)
                if not self._pending and self._eof:
                    if self._end is None:
                        step = self._step or 1
                        self._end = self._slot(self._cur[0] + step)
                    if self._next >= self._end:
                        return None
                self._next += 1
                self._emitted += 1
                return self._cur[1] if self.compact else self._convert(self._cur[1])
