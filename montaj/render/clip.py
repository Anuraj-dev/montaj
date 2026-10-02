"""Video clips. One lazy ffmpeg decode per shot window. Contract: cover, centred, no drift.

The decoded span is `[clip_in, clip_in + hold)` — pass `hold` in seconds or `nframes`.
Frames stay uint8 on CPU. `frame` converts that one frame to float on the caller's device.
`clip_in` is `-ss` before `-i`, so decoded frame 0 is that far into the file.
Timeline frame `t` is `floor(t − shot_start)`, clamped to the window.

A hold longer than `_CACHE_FRAMES` is read forward from that same seek. The source
keeps at most that many uint8 frames across every reader; a fade trims the oldest
frame of the fuller window so both stay open. A hold that would fit on its own is
streamed the same way when another reader already fills the budget. An index that
was trimmed is decoded again from the same seek. `frames()` still returns the whole
hold — the caller owns that tuple.
"""
from __future__ import annotations

import math
import os
import subprocess
import threading
import weakref
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

FFMPEG = os.environ.get("MONTAJ_FFMPEG", "ffmpeg")
# Resident decoded frames across every window this source has open. A single
# hold longer than this is not kept whole; the reader retains only the tail
# it still needs.
_CACHE_FRAMES = 96


def _read_one(stdout, w: int, h: int) -> torch.Tensor | None:
    frame_bytes = w * h * 3
    buf = bytearray()
    while len(buf) < frame_bytes:
        chunk = stdout.read(frame_bytes - len(buf))
        if not chunk:
            return None
        buf += chunk
    arr = np.frombuffer(bytes(buf), np.uint8).reshape(h, w, 3)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


class _ClipWindow:
    """Forward reader for one hold. At most `cap` uint8 frames stay in `_frames`."""

    def __init__(self, source: ClipSource, clip_in: float, w: int, h: int, count: int):
        self._source = weakref.ref(source)
        self._clip_in = float(clip_in)
        self._w = int(w)
        self._h = int(h)
        self._count = int(count)
        self._cap = source._cap
        self._frames: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._proc: subprocess.Popen | None = None
        self._err: list[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._filled = 0
        self._produced: int | None = None

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self):
        return iter(self._frames.values())

    def close(self) -> None:
        self._frames.clear()
        self._close_proc(check=False)

    def __del__(self) -> None:
        try:
            self._close_proc(check=False)
        except Exception:
            pass

    def _spawn(self):
        source = self._source()
        if source is None:
            raise RuntimeError("clip source is gone")
        return source._spawn(self._clip_in, self._w, self._h, self._count)

    def _open(self) -> None:
        self._close_proc(check=False)
        self._frames.clear()
        self._filled = 0
        self._produced = None
        self._proc, self._err, self._thread = self._spawn()

    def _close_proc(self, check: bool) -> None:
        proc = self._proc
        thread = self._thread
        err = self._err
        self._proc = None
        self._thread = None
        self._err = None
        if proc is None:
            return
        if proc.stdout is not None and not proc.stdout.closed:
            proc.stdout.close()
        if not check and proc.poll() is None:
            proc.kill()
        code = proc.wait()
        if thread is not None:
            thread.join(timeout=5)
        if check and code != 0:
            tail = b"".join(err or []).decode(errors="replace")[-400:]
            raise RuntimeError(f"ffmpeg clip decode failed: {tail}")

    def drop_oldest(self) -> None:
        """Forget one retained frame. The decoder stays where it is."""
        if self._frames:
            self._frames.popitem(last=False)

    def _keep(self, index: int, frame: torch.Tensor) -> None:
        while len(self._frames) >= self._cap:
            self._frames.popitem(last=False)
        self._frames[index] = frame
        source = self._source()
        if source is not None:
            source._trim()

    def _name(self) -> str:
        source = self._source()
        return str(source.path) if source is not None else "clip"

    def at(self, index: int) -> torch.Tensor:
        """uint8 frame `index`, or the last frame ffmpeg actually returned."""
        if index < 0:
            index = 0
        if self._produced is not None and index >= self._produced:
            index = self._produced - 1
        hit = self._frames.get(index)
        if hit is not None:
            self._frames.move_to_end(index)
            return hit
        if self._proc is None or index < self._filled:
            self._open()
        assert self._proc is not None and self._proc.stdout is not None
        while self._filled <= index:
            raw = _read_one(self._proc.stdout, self._w, self._h)
            if raw is None:
                self._produced = self._filled
                self._close_proc(check=True)
                if self._produced <= 0:
                    raise RuntimeError(f"clip {self._name()} decoded to 0 frames")
                index = self._produced - 1
                break
            self._keep(self._filled, raw)
            self._filled += 1
        if self._filled >= self._count and self._proc is not None:
            # The hold is in hand. Reap ffmpeg instead of leaving it a zombie.
            self._produced = self._filled
            self._close_proc(check=True)
        got = self._frames.get(index)
        if got is None:
            raise RuntimeError(f"clip {self._name()} frame {index} was not retained")
        return got


class ClipSource:
    def __init__(self, path, fps: float):
        self.path = Path(path)
        self.fps = float(fps)
        self._cap = _CACHE_FRAMES
        self._cache: OrderedDict[tuple, tuple[torch.Tensor, ...] | _ClipWindow] = OrderedDict()

    def close(self) -> None:
        for entry in list(self._cache.values()):
            closer = getattr(entry, "close", None)
            if callable(closer):
                closer()
        self._cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _count(self, nframes: int | None, hold: float | None) -> int:
        if nframes is None and hold is None:
            raise TypeError("pass hold (seconds) or nframes for [clip_in, clip_in + hold)")
        count = int(nframes) if nframes is not None else int(round(float(hold) * self.fps))
        if hold is not None and nframes is not None and count != int(round(float(hold) * self.fps)):
            raise ValueError(f"hold {hold}s is {int(round(float(hold) * self.fps))} frames, not {nframes}")
        if count <= 0:
            raise ValueError(f"clip window is empty (nframes={nframes}, hold={hold})")
        return count

    def _key(self, clip_in: float, w: int, h: int, count: int) -> tuple:
        return (round(float(clip_in), 6), int(w), int(h), count)

    def frames(self, clip_in: float, w: int, h: int, nframes: int | None = None, *,
               hold: float | None = None) -> tuple[torch.Tensor, ...]:
        """uint8 CPU tensors (3, H, W) for the hold. A hold within the cap is cached whole."""
        count = self._count(nframes, hold)
        key = self._key(clip_in, w, h, count)
        if count <= self._cap:
            hit = self._cache.get(key)
            if isinstance(hit, tuple):
                self._cache.move_to_end(key)
                return hit
            decoded = self._decode(float(clip_in), int(w), int(h), count)
            self._cache[key] = decoded
            self._cache.move_to_end(key)
            self._trim()
            return decoded
        window = self._window(key, float(clip_in), int(w), int(h), count)
        out: list[torch.Tensor] = []
        for i in range(count):
            self._cache.move_to_end(key)
            if window._produced is not None and i >= window._produced:
                break
            out.append(window.at(i))
            if window._produced is not None and i >= window._produced:
                out.pop()
                break
        self._cache.move_to_end(key)
        self._trim()
        return tuple(out)

    def _window(self, key: tuple, clip_in: float, w: int, h: int, count: int) -> _ClipWindow:
        hit = self._cache.get(key)
        if isinstance(hit, _ClipWindow):
            self._cache.move_to_end(key)
            return hit
        window = _ClipWindow(self, clip_in, w, h, count)
        self._cache[key] = window
        self._cache.move_to_end(key)
        self._trim()
        return window

    def _resident_count(self) -> int:
        return sum(len(entry) for entry in self._cache.values())

    def _release(self, old) -> None:
        closer = getattr(old, "close", None)
        if callable(closer):
            closer()

    def _trim(self) -> None:
        """Share `_cap` across windows by dropping oldest frames, not whole readers.

        A short hold is one tuple. Those still leave as a unit, oldest first, so a
        stack of small holds keeps the newest. Two long windows of one fade each
        keep the frames they just read.
        """
        while self._resident_count() > self._cap:
            windows = [entry for entry in self._cache.values()
                       if isinstance(entry, _ClipWindow) and len(entry) > 1]
            if windows:
                max(windows, key=len).drop_oldest()
                continue
            dropped = False
            for key, entry in list(self._cache.items()):
                if isinstance(entry, tuple):
                    self._cache.pop(key)
                    dropped = True
                    break
            if dropped:
                continue
            if len(self._cache) > 1:
                _key, old = self._cache.popitem(last=False)
                self._release(old)
                continue
            break

    def _spawn(self, clip_in: float, w: int, h: int, nframes: int):
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

        thread = threading.Thread(target=_drain, daemon=True)
        thread.start()
        return proc, err, thread

    def _decode(self, clip_in: float, w: int, h: int, nframes: int) -> tuple[torch.Tensor, ...]:
        """Read at most `nframes` from the pipe. `-frames:v` stops ffmpeg at the hold."""
        proc, err, thread = self._spawn(clip_in, w, h, nframes)
        frames: list[torch.Tensor] = []
        try:
            assert proc.stdout is not None
            while len(frames) < nframes:
                frame = _read_one(proc.stdout, w, h)
                if frame is None:
                    break
                frames.append(frame)
        finally:
            if proc.stdout is not None and not proc.stdout.closed:
                proc.stdout.close()
            code = proc.wait()
            thread.join(timeout=5)
        if code != 0:
            tail = b"".join(err).decode(errors="replace")[-400:]
            raise RuntimeError(f"ffmpeg clip decode failed: {tail}")
        if not frames:
            raise RuntimeError(f"clip {self.path} decoded to 0 frames")
        return tuple(frames)

    def _stream(self, count: int, key: tuple) -> bool:
        """A hold that fits the cap is still streamed when caching it whole would push another reader out."""
        cached = self._cache.get(key)
        if isinstance(cached, (tuple, _ClipWindow)):
            return isinstance(cached, _ClipWindow)
        if count > self._cap:
            return True
        resident = self._resident_count()
        return resident > 0 and resident + count > self._cap

    def frame(self, t: float, shot_start: float, clip_in: float, w: int, h: int,
              nframes: int | None = None, *, hold: float | None = None, device="cpu") -> torch.Tensor:
        """`floor(t − shot_start)` clamped to the hold, float on `device`."""
        count = self._count(nframes, hold)
        key = self._key(clip_in, w, h, count)
        i = math.floor(t - shot_start)
        i = max(0, min(i, count - 1))
        if self._stream(count, key):
            window = self._window(key, float(clip_in), int(w), int(h), count)
            self._cache.move_to_end(key)
            tensor = window.at(i)
            self._cache.move_to_end(key)
            self._trim()
            return tensor.to(device=device).to(dtype=torch.float32) / 255
        frames = self.frames(clip_in, w, h, nframes=count)
        i = max(0, min(i, len(frames) - 1))
        # uint8 crosses to the device; the float conversion happens there.
        return frames[i].to(device=device).to(dtype=torch.float32) / 255
