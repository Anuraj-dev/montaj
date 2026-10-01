"""Headless Chromium oracle for one film.html layer.

Renders a 1080×1920 HTML document the way `bday-video/export.mjs` does: a private
user-data-dir, deviceScaleFactor 1, PNG clip of the viewport. Never a user profile.
"""
from __future__ import annotations

import base64
import json
import os
import signal
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from dataclasses import dataclass

import numpy as np
from PIL import Image

from montaj.doctor import chromium

W, H = 1080, 1920


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak 255, inputs uint8 or float arrays broadcast to a common shape."""
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    if x.max() <= 1.0 and y.max() <= 1.0:
        x, y = x * 255.0, y * 255.0
    mse = np.mean((x - y) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


class _WS:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = b""

    def send(self, text: str) -> None:
        data = text.encode()
        mask = os.urandom(4)
        n = len(data)
        hdr = bytearray([0x81])
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr.append(0x80 | 126)
            hdr += struct.pack(">H", n)
        else:
            hdr.append(0x80 | 127)
            hdr += struct.pack(">Q", n)
        hdr += mask
        self.sock.sendall(bytes(hdr) + bytes(b ^ mask[i & 3] for i, b in enumerate(data)))

    def _read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise EOFError("chromium socket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self) -> str:
        parts: list[bytes] = []
        while True:
            b0, b1 = self._read(2)
            opcode, ln = b0 & 0x0F, b1 & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._read(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._read(8))[0]
            data = self._read(ln)
            if b1 & 0x80:  # server frames are unmasked; tolerate it anyway
                m = data[:4]
                data = bytes(c ^ m[i & 3] for i, c in enumerate(data[4:]))
            if opcode == 8:
                raise EOFError("websocket closed")
            if opcode == 9:
                continue
            if opcode in (0, 1, 2):
                parts.append(data)
                if b0 & 0x80:
                    return b"".join(parts).decode()


def stage_html(*, css: str = "", body: str = "", script: str = "window.__oracleReady = true;",
               bg: str = "#000") -> str:
    """One 1080×1920 stage. Copied layer CSS is passed in `css`."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
html, body {{ margin: 0; background: {bg}; overflow: hidden; }}
#stage {{ position: absolute; left: 0; top: 0; width: 1080px; height: 1920px;
  overflow: hidden; background: {bg}; }}
.L {{ position: absolute; left: 0; top: 0; width: 1080px; height: 1920px; overflow: hidden; }}
img {{ position: absolute; display: block; user-select: none; }}
{css}
</style></head>
<body><div id="stage">{body}</div>
<script>
{script}
</script></body></html>
"""


@dataclass
class PageShot:
    rgb: np.ndarray
    result: object = None
    rgba: np.ndarray | None = None


def render_html(html: str, *, chrome: Path | None = None) -> np.ndarray:
    """Return the viewport as uint8 RGB (H, W, 3). `html` may set `window.__oracleReady`."""
    return render_page(html, chrome=chrome).rgb


def render_page(html: str, *, chrome: Path | None = None) -> PageShot:
    """Viewport RGB, `window.__oracleResult`, and an RGBA clip when that result is a rect."""
    binary = chrome or chromium()
    if binary is None:
        raise RuntimeError("chromium missing")
    with tempfile.TemporaryDirectory(prefix="montaj-oracle-", ignore_cleanup_errors=True) as td:
        root = Path(td)
        (root / "page.html").write_text(html, encoding="utf-8")
        udd = root / "profile"
        proc = subprocess.Popen(
            [str(binary), "--headless=new", "--remote-debugging-port=0",
             f"--user-data-dir={udd}", "--no-first-run", "--disable-extensions",
             "--hide-scrollbars", "--mute-audio", "--allow-file-access-from-files",
             "--force-device-scale-factor=1", f"--window-size={W},{H}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            port = _dev_port(udd, proc)
            ws = _ws(port)
            try:
                return _shoot(ws, (root / "page.html").as_uri())
            finally:
                ws.sock.close()
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)


def _dev_port(udd: Path, proc: subprocess.Popen) -> int:
    deadline = time.monotonic() + 20
    port_file = udd / "DevToolsActivePort"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"chromium exited {proc.returncode}")
        if port_file.exists():
            line = port_file.read_text().splitlines()
            if line and line[0].isdigit():
                return int(line[0])
        time.sleep(0.05)
    raise RuntimeError("chromium did not open a devtools port")


def _ws(port: int) -> _WS:
    deadline = time.monotonic() + 20
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as resp:
                targets = json.loads(resp.read().decode())
            page = next(t for t in targets if t.get("type") == "page")
            path = urlparse(page["webSocketDebuggerUrl"]).path
            sock = socket.create_connection(("127.0.0.1", port), timeout=30)
            key = base64.b64encode(os.urandom(16)).decode()
            sock.sendall(
                f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode()
            )
            shake = b""
            while b"\r\n\r\n" not in shake:
                shake += sock.recv(4096)
            if b" 101 " not in shake.split(b"\r\n", 1)[0]:
                raise RuntimeError(shake[:200].decode(errors="replace"))
            sock.settimeout(60)
            ws = _WS(sock)
            ws.buf = shake.split(b"\r\n\r\n", 1)[1]
            return ws
        except (OSError, StopIteration, json.JSONDecodeError, KeyError, RuntimeError) as exc:
            last = str(exc)
            time.sleep(0.05)
    raise RuntimeError(f"devtools websocket failed: {last}")


def _png(data_b64: str, mode: str) -> np.ndarray:
    raw = base64.b64decode(data_b64)
    return np.ascontiguousarray(np.asarray(Image.open(__import__("io").BytesIO(raw)).convert(mode)))


def _shoot(ws: _WS, url: str) -> PageShot:
    n = 0

    def send(method: str, params: dict | None = None, *, wait: bool = True):
        nonlocal n
        n += 1
        ws.send(json.dumps({"id": n, "method": method, "params": params or {}}))
        if not wait:
            return None
        want = n
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == want:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    send("Emulation.setDeviceMetricsOverride",
         {"width": W, "height": H, "deviceScaleFactor": 1, "mobile": False})
    send("Page.enable")
    nav = send("Page.navigate", {"url": url})
    if nav.get("errorText"):
        raise RuntimeError(f"navigation failed: {nav['errorText']}")
    # about:blank is already complete, and `__oracleReady !== false` is true there
    # because the name is unset. Wait until this document is the one we asked for.
    ready = (
        "document.readyState === 'complete' && window.__oracleReady === true && "
        f"location.href === {json.dumps(url)}"
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        result = send("Runtime.evaluate", {"expression": ready, "returnByValue": True})
        if result.get("exceptionDetails"):
            time.sleep(0.02)
            continue
        if result.get("result", {}).get("value") is True:
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("oracle page did not become ready")
    ev = send("Runtime.evaluate", {
        "expression": "window.__oracleResult === undefined ? null : window.__oracleResult",
        "returnByValue": True,
    })
    value = ev.get("result", {}).get("value")
    rgba = None
    if isinstance(value, dict) and {"x", "y", "width", "height"} <= value.keys():
        # omitBackground alone still flattens onto white in headless; the override is what
        # leaves the glyph coverage unpainted.
        send("Emulation.setDefaultBackgroundColorOverride",
             {"color": {"r": 0, "g": 0, "b": 0, "a": 0}})
        clip = {k: float(value[k]) for k in ("x", "y", "width", "height")}
        clip["scale"] = 1
        extra = send("Page.captureScreenshot", {
            "format": "png", "omitBackground": True, "clip": clip, "fromSurface": True,
        })
        rgba = _png(extra["data"], "RGBA")
    shot = send("Page.captureScreenshot", {
        "format": "png",
        "clip": {"x": 0, "y": 0, "width": W, "height": H, "scale": 1},
        "fromSurface": True,
    })
    arr = _png(shot["data"], "RGB")
    if arr.shape[0] < H or arr.shape[1] < W:
        raise RuntimeError(f"screenshot {arr.shape} smaller than {W}x{H}")
    return PageShot(np.ascontiguousarray(arr[:H, :W]), value, rgba)


def save_png(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb if rgb.dtype == np.uint8 else np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)).save(path)


def expect_psnr(name: str, chrome: np.ndarray, pred: np.ndarray, floor: float) -> float:
    """Quantise `pred` (HWC or CHW, float 0..1 or uint8) and record the PSNR."""
    got = pred
    if got.dtype != np.uint8:
        if got.ndim == 3 and got.shape[0] in (1, 3, 4) and got.shape[-1] not in (1, 3, 4):
            got = np.transpose(got, (1, 2, 0))
        got = np.clip(np.rint(np.clip(got, 0, 1) * 255.0), 0, 255).astype(np.uint8)
    if got.ndim == 2:
        got = got[..., None]
    if got.shape[-1] == 1:
        got = np.repeat(got, 3, axis=2)
    score = psnr(chrome, got[..., :3])
    with Path("/tmp/w5c-psnr.txt").open("a") as fh:
        fh.write(f"{name} {score:.2f}\n")
    if score < floor:
        safe = "".join(c if c.isalnum() else "-" for c in name)
        save_png(Path(f"/tmp/w5c-{safe}-chrome.png"), chrome)
        save_png(Path(f"/tmp/w5c-{safe}-ours.png"), got[..., :3])
        diff = np.clip(np.abs(chrome.astype(np.int16) - got[..., :3].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        save_png(Path(f"/tmp/w5c-{safe}-diff.png"), diff)
    assert score >= floor, f"{name} PSNR {score:.2f} < {floor}"
    return score
