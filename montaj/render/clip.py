"""Video clips. One lazy ffmpeg decode per shot window. Contract: cover, centred, no drift.

The decoded span is `[clip_in, clip_in + hold)` — pass `hold` in seconds or `nframes`.
Frames stay uint8 on CPU. `frame` converts that one frame to float on the caller's device.
`clip_in` is `-ss` before `-i`, so decoded frame 0 is that far into the file.
Timeline frame `t` is `floor(t − shot_start)`, clamped to the window.
"""
from __future__ import annotations

import math
import os
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

FFMPEG = os.environ.get("MONTAJ_FFMPEG", "ffmpeg")
# Older windows are dropped past this many frames. The window just decoded is kept
# even when the hold itself is longer — the bound is the hold, not the source.
_CACHE_FRAMES = 96


class ClipSource:
    def __init__(self, path, fps: float):
        self.path = Path(path)
        self.fps = float(fps)
        self._cache: OrderedDict[tuple, tuple[torch.Tensor, ...]] = OrderedDict()

    def _count(self, nframes: int | None, hold: float | None) -> int:
        if nframes is None and hold is None:
            raise TypeError("pass hold (seconds) or nframes for [clip_in, clip_in + hold)")
        count = int(nframes) if nframes is not None else int(round(float(hold) * self.fps))
        if hold is not None and nframes is not None and count != int(round(float(hold) * self.fps)):
            raise ValueError(f"hold {hold}s is {int(round(float(hold) * self.fps))} frames, not {nframes}")
        if count <= 0:
            raise ValueError(f"clip window is empty (nframes={nframes}, hold={hold})")
        return count

    def frames(self, clip_in: float, w: int, h: int, nframes: int | None = None, *,
               hold: float | None = None) -> tuple[torch.Tensor, ...]:
        """uint8 CPU tensors (3, H, W) for the hold. Decoded once per window."""
        count = self._count(nframes, hold)
        key = (round(float(clip_in), 6), int(w), int(h), count)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        decoded = self._decode(float(clip_in), int(w), int(h), count)
        self._cache[key] = decoded
        self._cache.move_to_end(key)
        self._trim()
        return decoded

    def _trim(self) -> None:
        total = sum(len(v) for v in self._cache.values())
        while len(self._cache) > 1 and total > _CACHE_FRAMES:
            _key, old = self._cache.popitem(last=False)
            total -= len(old)

    def _decode(self, clip_in: float, w: int, h: int, nframes: int) -> tuple[torch.Tensor, ...]:
        """Read at most `nframes` from the pipe. `-frames:v` stops ffmpeg at the hold."""
        fps = int(self.fps) if self.fps == int(self.fps) else self.fps
        vf = f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
        cmd = [FFMPEG, "-v", "error", "-ss", f"{clip_in:.6f}", "-i", str(self.path),
               "-an", "-vf", vf, "-frames:v", str(nframes),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        err: list[bytes] = []

        def _drain() -> None:
            assert proc.stderr is not None
            err.append(proc.stderr.read())

        threading.Thread(target=_drain, daemon=True).start()
        assert proc.stdout is not None
        frame_bytes = w * h * 3
        frames: list[torch.Tensor] = []
        while len(frames) < nframes:
            buf = bytearray()
            while len(buf) < frame_bytes:
                chunk = proc.stdout.read(frame_bytes - len(buf))
                if not chunk:
                    break
                buf += chunk
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(bytes(buf), np.uint8).reshape(h, w, 3)
            frames.append(torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))))
        proc.stdout.close()
        code = proc.wait()
        if code != 0:
            tail = b"".join(err).decode(errors="replace")[-400:]
            raise RuntimeError(f"ffmpeg clip decode failed: {tail}")
        if not frames:
            raise RuntimeError(f"clip {self.path} decoded to 0 frames")
        return tuple(frames)

    def frame(self, t: float, shot_start: float, clip_in: float, w: int, h: int,
              nframes: int | None = None, *, hold: float | None = None, device="cpu") -> torch.Tensor:
        """`floor(t − shot_start)` clamped to the hold, float on `device`."""
        frames = self.frames(clip_in, w, h, nframes, hold=hold)
        i = math.floor(t - shot_start)
        i = max(0, min(i, len(frames) - 1))
        # uint8 crosses to the device; the float conversion happens there.
        return frames[i].to(device=device).to(dtype=torch.float32) / 255
