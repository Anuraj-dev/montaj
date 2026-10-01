"""Chromium text raster and the GPU compositor for text cards.

Raster `scale` is `Canvas.k`: one design px is one CSS px on the 1080-wide stage.
`draw_text` blits 1:1 when the raster was made at that scale. No module state —
the segment cache depends on that. Gold drop-shadows are captured from Chromium;
a gaussian with σ = R/2 does not match Chrome's drop-shadow kernel.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from montaj.doctor import chromium as find_chromium
from montaj.render.core import outc

# Stage sits at (OX, OY) so shadow padding stays inside the viewport and the
# fidelity crop of that rect has the same subpixel phase as the rasters.
OX, OY = 200, 200
DESIGN_W, DESIGN_H = 1080, 1920
_VIEW = (1800, 4200)
_REV = "1"
_HERE = Path(__file__).resolve().parent
_CSS = _HERE / "text.css"
_FONTS = _HERE.parent / "fonts"

# film.html:31–33 stops. Opaque, so premul interpolation is the straight sRGB lerp.
_STOPS = (
    (0.00, (0xB8 / 255, 0x7A / 255, 0x35 / 255)),
    (0.30, (0xF7 / 255, 0xDC / 255, 0xA0 / 255)),
    (0.45, (0xFF / 255, 0xF6 / 255, 0xDE / 255)),
    (0.60, (0xE9 / 255, 0xB5 / 255, 0x66 / 255)),
    (1.00, (0xA8 / 255, 0x6D / 255, 0x2C / 255)),
)
_NAMED = {"cream": "#fbf1dc", "gold": "#f0c77e", "ink": "#4a2c1a"}
_SIZE = {"serif": 84.0, "script": 84.0, "caps": 26.0, "deva": 62.0}
_SHADOW_R = {"serif": 30.0, "script": 40.0, "caps": 16.0, "deva": 24.0}

# Faces the page actually uses. A failed load must reject: swallowing it lets a fallback
# face paint real pixels, pass the nonblank check, and get cached as the requested font.
_FONT_JS = """
(async () => {
  const els = [...document.querySelectorAll('.txt, .w, .sub, .cap')];
  const specs = [];
  const seen = new Set();
  for (const el of els) {
    const cs = getComputedStyle(el);
    const fam = (cs.fontFamily || '').split(',')[0].replace(/["']/g, '').trim();
    if (!fam) continue;
    const spec = cs.fontStyle + ' ' + cs.fontWeight + ' ' + cs.fontSize + ' "' + fam + '"';
    const sample = (el.textContent || '').trim() || 'Aa';
    const key = spec + '\\n' + sample;
    if (seen.has(key)) continue;
    seen.add(key);
    specs.push([spec, sample]);
  }
  const missing = [];
  for (const [spec, sample] of specs) {
    let faces = [];
    try {
      faces = await document.fonts.load(spec, sample);
    } catch (err) {
      missing.push(spec + ' (' + (err && err.message ? err.message : 'load failed') + ')');
      continue;
    }
    if (!faces || !faces.length || !document.fonts.check(spec, sample)) missing.push(spec);
  }
  await document.fonts.ready;
  return {ok: missing.length === 0, missing};
})()
"""


@dataclass(frozen=True)
class TextLine:
    text: str
    style: str
    y: float
    at: float
    italic: bool = False
    gold: bool = False
    color: str | None = None
    size: float | None = None
    weight: int | None = None
    tracking: float | None = None
    line_height: float | None = None
    by: str = "word"
    stagger: float | None = None
    dur: float | None = None
    rise: float = 26
    reveal: str = "rise"
    sweep: tuple[float, float] | None = None
    shadow: bool = True

    def __post_init__(self) -> None:
        if self.by not in ("word", "char"):
            raise ValueError(f"by must be 'word' or 'char', got {self.by!r}")
        if self.reveal not in ("rise", "pop"):
            raise ValueError(f"reveal must be 'rise' or 'pop', got {self.reveal!r}")
        if self.sweep is not None and not isinstance(self.sweep, tuple):
            object.__setattr__(self, "sweep", (float(self.sweep[0]), float(self.sweep[1])))


@dataclass(frozen=True)
class TextBlock:
    start: float
    end: float
    fade_in: float
    fade_out: float
    blur: float
    lines: tuple[TextLine, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.lines, tuple):
            object.__setattr__(self, "lines", tuple(self.lines))


@dataclass(frozen=True)
class Sub:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Unit:
    """One word (with its trailing space) or one character.

    `box` is the captured bitmap in design px. `layout` is the CSS border box
    (fractional) the gold gradient is sized to. `shadow_box` may differ: gold
    shadows are a shifted drop-shadow crop, not the glyph rect.
    """

    text: str
    fill: torch.Tensor
    shadow: torch.Tensor | None
    box: tuple[float, float, float, float]
    shadow_box: tuple[float, float, float, float] | None
    layout: tuple[float, float, float, float]
    gold: bool


@dataclass(frozen=True)
class RasterLine:
    units: tuple[Unit, ...]
    line_box: tuple[float, float, float, float]
    line_shadow: torch.Tensor | None
    line_shadow_box: tuple[float, float, float, float] | None
    scale: float
    gold: bool


@dataclass(frozen=True)
class RasterImage:
    """A single premul RGBA sprite. `bottom` keeps a sub 190px above the canvas bottom."""

    fill: torch.Tensor
    box: tuple[float, float, float, float]
    bottom: bool = False


@dataclass(frozen=True)
class RasterSet:
    lines: tuple[RasterLine, ...] = ()
    subs: tuple[RasterImage, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.lines, tuple):
            object.__setattr__(self, "lines", tuple(self.lines))
        if not isinstance(self.subs, tuple):
            object.__setattr__(self, "subs", tuple(self.subs))


def sine(x: float) -> float:
    """film.html:73 `es`. The spec omits the clamp; the film clamps."""
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return 0.5 - 0.5 * math.cos(math.pi * x)


def _num(v: float) -> str:
    return f"{float(v):.6g}"


def _split(text: str, by: str) -> list[str]:
    # film.html:172-174. JS spread is UTF-16 code units; Python walks code points (same for BMP).
    if by == "char":
        return list(text)
    parts = text.split(" ")
    return [w if i == len(parts) - 1 else w + " " for i, w in enumerate(parts)]


def _pad(line: TextLine) -> int:
    size = float(line.size if line.size is not None else _SIZE.get(line.style, 84.0))
    blur = 0.0 if line.gold or not line.shadow else _SHADOW_R.get(line.style, 30.0)
    return int(max(120, round(size * 0.85 + blur)))


def line_markup(line: TextLine, *, bp: str | None = None, idx: int = 0) -> str:
    """One `.txt` line in film.html's class vocabulary. `bp` is reference-only (`--bp`)."""
    classes = ["txt", line.style]
    if line.italic and line.style == "serif":
        classes.append("it")
    if line.gold:
        classes.append("gold")
    style = [f"top:{_num(line.y)}px"]
    if line.size is not None:
        style.append(f"font-size:{_num(line.size)}px")
    if line.weight is not None:
        style.append(f"font-weight:{int(line.weight)}")
    if line.tracking is not None:
        style.append(f"letter-spacing:{_num(line.tracking)}em")
    if line.line_height is not None:
        style.append(f"line-height:{_num(line.line_height)}")
    if line.color:
        style.append(f"color:{_NAMED.get(line.color, line.color)}")
    if not line.shadow:
        style.append("text-shadow:none;filter:none")
    if bp is not None:
        style.append(f"--bp:{bp}")
    spans = "".join(
        f'<span class="w" id="u{i}">{html.escape(tok)}</span>'
        for i, tok in enumerate(_split(line.text, line.by))
    )
    return f'<div class="{" ".join(classes)}" id="L{idx}" style="{";".join(style)}">{spans}</div>'


def _canon(line: TextLine) -> str:
    # Animation fields do not change pixels. y does: it sets the subpixel phase.
    return line_markup(line, idx=0)


def _css_bytes() -> bytes:
    return _CSS.read_bytes()


def _css_text() -> str:
    return _css_bytes().decode("utf-8").replace("../fonts/", "/fonts/")


def _font_digest() -> bytes:
    h = hashlib.sha256()
    for path in sorted(_FONTS.glob("*.woff2")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.digest()


def _key(kind: str, payload: str, scale: float, css: bytes, fonts: bytes) -> str:
    h = hashlib.sha256()
    h.update(_REV.encode())
    h.update(kind.encode())
    h.update(f"{scale:.8f}|{OX}|{OY}|{DESIGN_W}|{DESIGN_H}".encode())
    h.update(css)
    h.update(fonts)
    h.update(payload.encode())
    return h.hexdigest()


def _page(inner: str, *, bg: str, extra_css: str = "") -> str:
    css = _css_text() + extra_css
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        f"html,body{{margin:0;background:{bg};overflow:hidden;}}"
        f"#stage{{position:absolute;left:{OX}px;top:{OY}px;width:{DESIGN_W}px;"
        f"height:{DESIGN_H}px;overflow:visible;}}"
        f"{css}</style></head><body><div id=\"stage\">{inner}</div>"
        '<style id="mode"></style></body></html>'
    )


def _png_premul(data: bytes) -> torch.Tensor:
    """Chrome's omitBackground PNG is straight alpha. Store premultiplied RGBA."""
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = arr.copy()
    arr[..., :3] *= arr[..., 3:4]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _png_rgb(data: bytes) -> torch.Tensor:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).contiguous()


def _box_list(box: tuple[float, float, float, float] | None) -> list[float] | None:
    return None if box is None else [float(v) for v in box]


def _as_box(raw) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/fonts/"):
            name = urllib.parse.unquote(path[len("/fonts/"):])
            fp = (_FONTS / name).resolve()
            if "/" in name or not str(fp).startswith(str(_FONTS.resolve())) or not fp.is_file():
                self.send_error(404)
                return
            data = fp.read_bytes()
            mime = "font/woff2"
        else:
            data = getattr(self.server, "html", "").encode("utf-8")
            mime = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass

    def log_message(self, fmt: str, *args) -> None:
        return


class _Browser:
    """One headless Chromium. Constructing this is the 'browser started' event tests patch."""

    def __init__(self, scale: float, binary: str | Path | None = None,
                 clear: tuple[int, int, int, int] = (0, 0, 0, 0)) -> None:
        self.scale = float(scale)
        self.clear = clear
        found = binary or find_chromium()
        if not found:
            raise RuntimeError("no Chromium (doctor.chromium() is None)")
        self.binary = str(found)
        self.proc: subprocess.Popen | None = None
        self.ws = None
        self.httpd: ThreadingHTTPServer | None = None
        self.udd: str | None = None
        self._log = None
        self._id = 0

    def __enter__(self) -> _Browser:
        try:
            self._start()
            return self
        except Exception:
            self.close()
            raise

    def __exit__(self, *exc) -> None:
        self.close()

    def _start(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.html = ""
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.udd = tempfile.mkdtemp(prefix="montaj-text-")
        log_path = Path(self.udd) / "chrome.log"
        self._log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [
                self.binary, "--headless=new", "--remote-debugging-port=0",
                f"--user-data-dir={self.udd}", "--no-first-run", "--no-default-browser-check",
                "--disable-extensions", "--hide-scrollbars", "--mute-audio", "--disable-gpu",
                "--force-color-profile=srgb", "--remote-allow-origins=*",
                f"--window-size={_VIEW[0]},{_VIEW[1]}", "about:blank",
            ],
            stdout=subprocess.DEVNULL, stderr=self._log, start_new_session=True,
        )
        port = self._debug_port(log_path)
        ws_url = self._ws_url(port)
        import websocket
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.call("Page.enable")
        self.call("Runtime.enable")
        self.call("Emulation.setDeviceMetricsOverride", {
            "width": _VIEW[0], "height": _VIEW[1],
            "deviceScaleFactor": self.scale, "mobile": False,
        })
        r, g, b, a = self.clear
        self.call("Emulation.setDefaultBackgroundColorOverride", {
            "color": {"r": int(r), "g": int(g), "b": int(b), "a": int(a)},
        })

    def _debug_port(self, log_path: Path) -> int:
        port_file = Path(self.udd) / "DevToolsActivePort"
        deadline = time.time() + 20
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = log_path.read_text()[-1500:] if log_path.exists() else ""
                raise RuntimeError(f"chrome exited {self.proc.returncode}: {tail}")
            if port_file.exists():
                lines = port_file.read_text().splitlines()
                if lines and lines[0].isdigit():
                    return int(lines[0])
            time.sleep(0.05)
        tail = log_path.read_text()[-1500:] if log_path.exists() else ""
        raise TimeoutError(f"DevToolsActivePort missing\n{tail}")

    def _ws_url(self, port: int) -> str:
        deadline = time.time() + 15
        url = f"http://127.0.0.1:{port}/json/list"
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    data = json.loads(resp.read().decode())
                for item in data:
                    if item.get("type") == "page" and item.get("webSocketDebuggerUrl"):
                        return item["webSocketDebuggerUrl"]
            except (OSError, json.JSONDecodeError, urllib.error.URLError):
                time.sleep(0.1)
        raise TimeoutError("CDP json/list had no page target")

    def call(self, method: str, params: dict | None = None, timeout: float = 60) -> dict:
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.ws.settimeout(max(0.1, deadline - time.time()))
            msg = json.loads(self.ws.recv())
            if msg.get("id") != mid:
                continue
            if "error" in msg:
                raise RuntimeError(f"CDP {method}: {msg['error']}")
            return msg.get("result", {})
        raise TimeoutError(method)

    def eval(self, expression: str, await_promise: bool = False):
        result = self.call("Runtime.evaluate", {
            "expression": expression, "returnByValue": True, "awaitPromise": await_promise,
        })
        if result.get("exceptionDetails"):
            detail = result["exceptionDetails"]
            raise RuntimeError(detail.get("text") or str(detail)[:800])
        return result.get("result", {}).get("value")

    def show(self, page: str) -> None:
        self.httpd.html = page
        self.call("Page.navigate", {"url": f"http://127.0.0.1:{self.httpd.server_address[1]}/?n={time.time_ns()}"})
        for _ in range(200):
            if self.eval("document.readyState") == "complete":
                break
            time.sleep(0.02)
        loaded = self.eval(_FONT_JS, await_promise=True) or {}
        if not loaded.get("ok"):
            missing = ", ".join(loaded.get("missing") or ["unknown"])
            raise RuntimeError(f"required fonts failed to load: {missing}")
        self.eval("void document.body.offsetHeight")

    def set_mode(self, css: str) -> None:
        # Style and filter updates are in the next composite. Headless rAF does not tick, so don't wait on it.
        self.eval("document.getElementById('mode').textContent = " + json.dumps(css) + "; void document.body.offsetHeight")

    def capture(self, x: float, y: float, w: float, h: float, *, omit_background: bool) -> bytes:
        params = {
            "format": "png",
            "omitBackground": omit_background,
            "captureBeyondViewport": True,
            "clip": {"x": x, "y": y, "width": w, "height": h, "scale": 1},
        }
        try:
            result = self.call("Page.captureScreenshot", params)
        except RuntimeError:
            params.pop("captureBeyondViewport", None)
            result = self.call("Page.captureScreenshot", params)
        return base64.b64decode(result["data"])

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    self.proc.kill()
                self.proc.wait(timeout=3)
        self.proc = None
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self.udd is not None:
            shutil.rmtree(self.udd, ignore_errors=True)
            self.udd = None


def _measure(br: _Browser) -> list[dict]:
    return br.eval("""(() => {
      const els = [...document.querySelectorAll('#stage .txt, #stage .sub, #stage .cap')];
      return els.map(el => {
        const r = el.getBoundingClientRect();
        const units = [...el.querySelectorAll(':scope > .w')].map(u => {
          const b = u.getBoundingClientRect();
          return {x: b.x, y: b.y, w: b.width, h: b.height, t: u.textContent};
        });
        return {x: r.x, y: r.y, w: r.width, h: r.height, units};
      });
    })()""") or []


def _clip(x: float, y: float, w: float, h: float, pad: int) -> tuple[int, int, int, int]:
    # Integer CSS-pixel clip. Placing the bitmap at this origin reconstructs the stage crop.
    x0 = math.floor(x) - pad
    y0 = math.floor(y) - pad
    x1 = math.ceil(x + w) + pad
    y1 = math.ceil(y + h) + pad
    return x0, y0, x1 - x0, y1 - y0


def _grab(br: _Browser, rect: tuple[int, int, int, int]) -> tuple[torch.Tensor, tuple[float, float, float, float], bytes]:
    x, y, w, h = rect
    png = br.capture(x, y, w, h, omit_background=True)
    sp = _png_premul(png)
    box = (x - OX, y - OY, sp.shape[-1] / br.scale, sp.shape[-2] / br.scale)
    return sp, box, png


def _empty(rect: dict) -> tuple[torch.Tensor, tuple[float, float, float, float]]:
    box = (rect["x"] - OX, rect["y"] - OY, rect["w"], rect["h"])
    return torch.zeros(4, 0, 0), box


def _drop_clip(rects: list[dict], pad_s: int) -> tuple[int, tuple[int, int, int, int]]:
    """Shift the drop-shadow below the glyphs, crop it, and remember how to put it back at y+6.

    Chrome's filter is `drop-shadow(0 6px 30px rgba(0,0,0,.55))` (film.html:33). The kernel is
    not a gaussian of σ = 15, so the pixels are captured. A large Y offset separates the shadow
    from the opaque glyphs; stored design y is `(clip_y - OY) - shift + 6`.
    """
    min_x = min(r["x"] for r in rects)
    max_x = max(r["x"] + r["w"] for r in rects)
    min_y = min(r["y"] for r in rects)
    max_y = max(r["y"] + r["h"] for r in rects)
    shift = math.ceil((max_y - min_y) + pad_s + 30)
    vx = math.floor(min_x) - pad_s
    vw = math.ceil(max_x) - math.floor(min_x) + 2 * pad_s
    vy = math.floor(min_y + shift) - pad_s
    vh = math.ceil(max_y + shift) + pad_s - vy
    if vy < max_y + 20:
        raise RuntimeError(f"drop-shadow clip overlaps glyphs (vy={vy}, max_y={max_y})")
    return shift, (vx, vy, vw, vh)


def _shift_box(box: tuple[float, float, float, float], shift: int) -> tuple[float, float, float, float]:
    return (box[0], box[1] - shift + 6, box[2], box[3])


_FILL_CSS = """
.txt,.gold{text-shadow:none !important;filter:none !important;}
#L0 .w{visibility:hidden;}
#L0 #u%d{visibility:visible !important;text-shadow:none !important;filter:none !important;}
"""
_SHADOW_CSS = """
.txt,.gold{filter:none !important;}
#L0 .w{visibility:hidden;}
#L0 #u%d{visibility:visible !important;color:transparent !important;-webkit-text-fill-color:transparent !important;}
"""
_COVER_CSS = """
.txt,.gold{filter:none !important;text-shadow:none !important;}
#L0 .w{visibility:hidden;}
#L0 #u%d{visibility:visible !important;background-image:linear-gradient(#fff,#fff) !important;
  background-size:100%% 100%% !important;-webkit-background-clip:text !important;background-clip:text !important;
  color:transparent !important;-webkit-text-fill-color:transparent !important;text-shadow:none !important;}
"""


def _drop_css(shift: int, only: int | None) -> str:
    hide = "" if only is None else "#L0 .w{visibility:hidden;} #L0 #u%d{visibility:visible !important;}" % only
    return (
        f"#L0.gold{{filter:drop-shadow(0 {shift}px 30px rgba(0,0,0,.55)) !important;}}"
        "#L0 .w{background:none !important;-webkit-background-clip:border-box !important;"
        "background-clip:border-box !important;-webkit-text-fill-color:#000 !important;"
        "color:#000 !important;text-shadow:none !important;}"
        + hide
    )


def _raster_one(br: _Browser, line: TextLine, scale: float, dest: Path) -> RasterLine:
    br.show(_page(line_markup(line, idx=0), bg="transparent"))
    measured = _measure(br)
    if not measured:
        raise RuntimeError(f"no layout for {line.text!r}")
    host = measured[0]
    pad = _pad(line)
    files: dict[str, bytes] = {}
    units_meta = []
    max_a = 0.0
    for i, rect in enumerate(host["units"]):
        layout = (rect["x"] - OX, rect["y"] - OY, rect["w"], rect["h"])
        if rect["w"] < 0.5 or rect["h"] < 0.5:
            units_meta.append({
                "text": rect.get("t") or "", "fill": None, "shadow": None,
                "box": _box_list(layout), "shadow_box": None, "layout": _box_list(layout),
                "gold": line.gold,
            })
            continue
        clip = _clip(rect["x"], rect["y"], rect["w"], rect["h"], pad)
        br.set_mode((_COVER_CSS if line.gold else _FILL_CSS) % i)
        fill, box, png = _grab(br, clip)
        max_a = max(max_a, float(fill[3].max()) if fill.numel() else 0.0)
        fill_name = f"u{i}_fill.png"
        files[fill_name] = png
        shadow_name = None
        shadow_box = None
        if line.shadow and line.gold:
            shift, sclip = _drop_clip([rect], max(120, pad))
            br.set_mode(_drop_css(shift, i))
            _sp, sbox, spng = _grab(br, sclip)
            shadow_box = _shift_box(sbox, shift)
            shadow_name = f"u{i}_shadow.png"
            files[shadow_name] = spng
        elif line.shadow:
            br.set_mode(_SHADOW_CSS % i)
            _sp, sbox, spng = _grab(br, clip)
            shadow_box = sbox
            shadow_name = f"u{i}_shadow.png"
            files[shadow_name] = spng
        units_meta.append({
            "text": rect.get("t") or "", "fill": fill_name, "shadow": shadow_name,
            "box": _box_list(box), "shadow_box": _box_list(shadow_box),
            "layout": _box_list(layout), "gold": line.gold,
        })
    if line.text.strip() and max_a < 0.01:
        raise RuntimeError(f"blank raster for {line.text!r} (fonts did not paint)")
    line_shadow_name = None
    line_shadow_box = None
    if line.gold and line.shadow and any(u["fill"] for u in units_meta):
        rects = [r for r in host["units"] if r["w"] >= 0.5 and r["h"] >= 0.5]
        shift, sclip = _drop_clip(rects, max(120, pad))
        br.set_mode(_drop_css(shift, None))
        _sp, sbox, spng = _grab(br, sclip)
        line_shadow_box = _shift_box(sbox, shift)
        line_shadow_name = "line_shadow.png"
        files[line_shadow_name] = spng
    line_box = (host["x"] - OX, host["y"] - OY, host["w"], host["h"])
    meta = {
        "rev": _REV, "kind": "line", "scale": scale, "gold": line.gold,
        "line_box": _box_list(line_box), "line_shadow": line_shadow_name,
        "line_shadow_box": _box_list(line_shadow_box), "units": units_meta,
    }
    _write_cache(dest, meta, files)
    loaded = _load_line(dest)
    if loaded is None:
        raise RuntimeError(f"cache write unreadable: {dest}")
    return loaded


def _write_cache(dest: Path, meta: dict, files: dict[str, bytes]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (dest / name).write_bytes(data)
    tmp = dest / "meta.json.tmp"
    tmp.write_text(json.dumps(meta))
    tmp.replace(dest / "meta.json")


def _load_png(dest: Path, name: str | None) -> torch.Tensor | None:
    if not name:
        return None
    return _png_premul((dest / name).read_bytes())


def _load_line(dest: Path) -> RasterLine | None:
    meta_path = dest / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("rev") != _REV or meta.get("kind") != "line":
            return None
        units = []
        for u in meta["units"]:
            fill = _load_png(dest, u["fill"])
            if fill is None:
                fill = torch.zeros(4, 0, 0)
            units.append(Unit(
                text=u.get("text") or "", fill=fill, shadow=_load_png(dest, u.get("shadow")),
                box=_as_box(u["box"]), shadow_box=_as_box(u.get("shadow_box")),
                layout=_as_box(u["layout"]), gold=bool(u["gold"]),
            ))
        return RasterLine(
            units=tuple(units), line_box=_as_box(meta["line_box"]),
            line_shadow=_load_png(dest, meta.get("line_shadow")),
            line_shadow_box=_as_box(meta.get("line_shadow_box")),
            scale=float(meta["scale"]), gold=bool(meta["gold"]),
        )
    except (OSError, KeyError, json.JSONDecodeError, TypeError):
        return None


def raster_lines(lines, *, scale: float, cache_dir, chromium=None) -> list[RasterLine]:
    """Lay out each line in one Chromium, capture every unit, and cache by content hash.

    A full-cache hit does not construct a browser. A partial miss launches one session.
    """
    lines = tuple(lines)
    cache_dir = Path(cache_dir)
    css, fonts = _css_bytes(), _font_digest()
    keys = [_key("line", _canon(ln), scale, css, fonts) for ln in lines]
    out: list[RasterLine | None] = [_load_line(cache_dir / key) for key in keys]
    missing = [i for i, hit in enumerate(out) if hit is None]
    if not missing:
        return out
    with _Browser(scale, chromium) as br:
        for i in missing:
            out[i] = _raster_one(br, lines[i], scale, cache_dir / keys[i])
    return out


def raster_line(line, *, scale: float, cache_dir, chromium=None) -> RasterLine:
    """Single-line contract entry. Same cache key and browser session as `raster_lines`."""
    return raster_lines([line], scale=scale, cache_dir=cache_dir, chromium=chromium)[0]


def _load_image(dest: Path) -> RasterImage | None:
    meta_path = dest / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("rev") != _REV or meta.get("kind") != "image":
            return None
        return RasterImage(_load_png(dest, meta["fill"]), _as_box(meta["box"]), bottom=bool(meta.get("bottom")))
    except (OSError, KeyError, json.JSONDecodeError, TypeError):
        return None


def raster_subs(subs, *, scale: float, cache_dir, chromium=None) -> list[RasterImage]:
    """One sprite per subtitle: fill and shadow together, bottom-anchored to a 1920-tall stage.

    The contract's `draw_text` names subs but not where their sprites come from.
    """
    subs = tuple(subs)
    cache_dir = Path(cache_dir)
    css, fonts = _css_bytes(), _font_digest()
    keys = [_key("sub", s.text, scale, css, fonts) for s in subs]
    out: list[RasterImage | None] = [_load_image(cache_dir / key) for key in keys]
    missing = [i for i, hit in enumerate(out) if hit is None]
    if not missing:
        return out
    with _Browser(scale, chromium) as br:
        for i in missing:
            sub = subs[i]
            inner = f'<div class="sub" id="s">{html.escape(sub.text)}</div>'
            br.show(_page(inner, bg="transparent"))
            measured = _measure(br)
            if not measured:
                raise RuntimeError(f"no layout for sub {sub.text!r}")
            host = measured[0]
            clip = _clip(host["x"], host["y"], host["w"], host["h"], 80)
            _sp, box, png = _grab(br, clip)
            if sub.text.strip() and float(_sp[3].max()) < 0.01:
                raise RuntimeError(f"blank sub raster for {sub.text!r}")
            dest = cache_dir / keys[i]
            _write_cache(dest, {
                "rev": _REV, "kind": "image", "scale": scale, "fill": "fill.png",
                "box": _box_list(box), "bottom": True,
            }, {"fill.png": png})
            loaded = _load_image(dest)
            if loaded is None:
                raise RuntimeError(f"cache write unreadable: {dest}")
            out[i] = loaded
    return out


def _extra_css(font_css) -> str:
    if font_css is None:
        return ""
    path = Path(font_css)
    if str(font_css) and path.is_file():
        return path.read_text()
    return str(font_css)


def raster_caption(text: str, *, font_css, scale: float, cache_dir, chromium=None):
    """Frame-card caption: Great Vibes 64px `#5a3b24`, no shadow (film.html:23). Returns premul RGBA and design size."""
    cache_dir = Path(cache_dir)
    extra = _extra_css(font_css)
    css, fonts = _css_bytes(), _font_digest()
    key = _key("cap", text + "\n" + extra, scale, css, fonts)
    dest = cache_dir / key
    hit = dest / "meta.json"
    if hit.is_file():
        meta = json.loads(hit.read_text())
        if meta.get("rev") == _REV and meta.get("kind") == "caption":
            return _png_premul((dest / "fill.png").read_bytes()), (float(meta["w"]), float(meta["h"]))
    with _Browser(scale, chromium) as br:
        inner = (
            f'<span class="cap" id="c" style="position:absolute;left:80px;top:80px">{html.escape(text)}</span>'
        )
        br.show(_page(inner, bg="transparent", extra_css=extra))
        measured = _measure(br)
        if not measured:
            raise RuntimeError(f"no layout for caption {text!r}")
        host = measured[0]
        clip = _clip(host["x"], host["y"], host["w"], host["h"], 160)
        png = br.capture(*clip, omit_background=True)
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    alpha = np.asarray(im)[..., 3]
    ys, xs = np.where(alpha > 0)
    if len(ys) == 0:
        raise RuntimeError(f"blank caption raster for {text!r}")
    crop = im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    buf = io.BytesIO()
    crop.save(buf, "PNG")
    raw = buf.getvalue()
    sp = _png_premul(raw)
    size = (sp.shape[-1] / float(scale), sp.shape[-2] / float(scale))
    _write_cache(dest, {"rev": _REV, "kind": "caption", "w": size[0], "h": size[1]}, {"fill.png": raw})
    return sp, size


def screenshot(lines, *, scale: float, width: int, height: int,
               background: tuple[int, int, int] = (128, 128, 128), bps: dict | None = None,
               chromium=None) -> torch.Tensor:
    """Straight RGB of the stage crop, composited by Chromium over `background`.

    Same CSS, same stage origin as `raster_lines`, so design (0, 0) is pixel (0, 0).
    `bps` maps a line index to a `--bp` value (gold reference frames).
    """
    bps = bps or {}
    inner = "".join(line_markup(ln, bp=bps.get(i), idx=i) for i, ln in enumerate(lines))
    bg = f"rgb({background[0]},{background[1]},{background[2]})"
    with _Browser(scale, chromium, clear=(*background, 255)) as br:
        br.show(_page(inner, bg=bg))
        png = br.capture(OX, OY, width, height, omit_background=False)
    return _png_rgb(png)


def _vis(t: float, block: TextBlock, fps: float) -> float:
    # film.html:408-411. t == end stays inside. The 0.02s test is in seconds there.
    if t < block.start or t > block.end:
        return 0.0
    thr = 0.02 * fps
    head = sine((t - block.start) / block.fade_in) if block.fade_in > thr else 1.0
    tail = sine((block.end - t) / block.fade_out) if block.fade_out > thr else 1.0
    return min(head, tail)


def _timing(line: TextLine, fps: float) -> tuple[float, float]:
    stagger = line.stagger if line.stagger is not None else (0.07 if line.by == "char" else 0.16) * fps
    dur = line.dur if line.dur is not None else 0.9 * fps
    return stagger, dur


def _unit_ks(line: TextLine, n: int, t: float, fps: float) -> list[float]:
    # film.html:181 `k = eo(P(t, at, at+dur))` with at_i = at + i*stagger (174).
    stagger, dur = _timing(line, fps)
    out = []
    for i in range(n):
        a = line.at + i * stagger
        out.append(1.0 if dur <= 0 and t >= a else 0.0 if dur <= 0 else outc((t - a) / dur))
    return out


def _bp(line: TextLine, block: TextBlock, t: float) -> float:
    # 100% → 0% linearly across sweep (default: the block window). film.html:291 is the same lerp.
    if line.sweep is not None:
        a, b = float(line.sweep[0]), float(line.sweep[1])
    else:
        a, b = block.start, block.end
    k = 1.0 if b <= a else min(1.0, max(0.0, (t - a) / (b - a)))
    return 100.0 * (1.0 - k)


def _gold(fill: torch.Tensor, layout: tuple[float, float, float, float],
          box: tuple[float, float, float, float], bp: float, scale: float,
          device, dtype) -> torch.Tensor:
    """Coverage × the 100° gold gradient, sampled at device-pixel centres.

    background-size is 300%×100% of the unit box; offset_x = (w−3w)·bp/100 (spec).
    CSS angle 0° points up and turns clockwise. Repeat is the CSS default.
    `box` origin is the bitmap; pixel centres are already continuous, so t does not add another ½.
    """
    _c, h, w = fill.shape
    if h == 0 or w == 0 or layout[2] <= 1e-3 or layout[3] <= 1e-3:
        return fill.to(device=device, dtype=dtype)
    ys = box[1] + (torch.arange(h, device=device, dtype=dtype) + 0.5) / scale
    xs = box[0] + (torch.arange(w, device=device, dtype=dtype) + 0.5) / scale
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    bw, bh = float(layout[2]), float(layout[3])
    img_w, img_h = 3.0 * bw, bh
    ix = torch.remainder((gx - layout[0]) - (bw - img_w) * bp / 100.0, img_w)
    iy = torch.remainder(gy - layout[1], img_h)
    rad = math.radians(100.0)
    dx, dy = math.sin(rad), -math.cos(rad)
    length = abs(img_w * dx) + abs(img_h * dy)
    t = ((ix - img_w / 2) * dx + (iy - img_h / 2) * dy) / length + 0.5
    t = t.clamp(0, 1)
    cols = torch.zeros(h, w, 3, device=device, dtype=dtype)
    for (a0, c0), (a1, c1) in zip(_STOPS, _STOPS[1:]):
        mask = (t >= a0) & (t <= a1)
        u = ((t - a0) / (a1 - a0)).unsqueeze(-1)
        c = (1 - u) * torch.tensor(c0, device=device, dtype=dtype) + u * torch.tensor(c1, device=device, dtype=dtype)
        cols = torch.where(mask.unsqueeze(-1), c, cols)
    alpha = fill[3].to(device=device, dtype=dtype)
    return torch.cat([cols.permute(2, 0, 1) * alpha, alpha.unsqueeze(0)], 0)


def _fit(sp: torch.Tensor, box: tuple[float, float, float, float], k: float) -> torch.Tensor:
    th, tw = sp.shape[-2:]
    eh, ew = int(round(box[3] * k)), int(round(box[2] * k))
    if (th, tw) == (eh, ew) or eh <= 0 or ew <= 0 or th == 0 or tw == 0:
        return sp
    return F.interpolate(sp[None], size=(eh, ew), mode="bilinear", align_corners=False)[0]


def _blur_kernel(img: torch.Tensor, sigma: float, radius: int) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    c = img.shape[0]
    horiz = F.conv2d(F.pad(img[None], (radius, radius, 0, 0)), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    vert = F.conv2d(F.pad(horiz, (0, 0, radius, radius)), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return vert[0]


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    # Separable gaussian with a full 3σ kernel. A hard radius cap shrinks the effective σ
    # (σ=100 truncated at 128 lands near 66). Past a kernel we can allocate, downsample so
    # the reduced σ still has 3σ support, blur there, and scale back.
    if sigma < 0.25 or img.numel() == 0:
        return img
    radius = math.ceil(3 * sigma)
    if radius <= 768:
        return _blur_kernel(img, sigma, radius)
    _c, h, w = img.shape
    factor = radius / 768
    nh, nw = max(1, int(round(h / factor))), max(1, int(round(w / factor)))
    small = F.interpolate(img[None], size=(nh, nw), mode="area")
    sigma_s = sigma * (nh / h)
    radius_s = max(1, math.ceil(3 * sigma_s))
    blurred = _blur_kernel(small[0], sigma_s, radius_s)
    return F.interpolate(blurred[None], size=(h, w), mode="bilinear", align_corners=False)[0]


def _stamp(dst: torch.Tensor, sp: torch.Tensor, x: int, y: int) -> None:
    """Premul source-over. RGB dst: `below*(1−a)+rgb`. RGBA dst: `src+dst*(1−a)`."""
    if sp.numel() == 0 or sp.shape[-1] == 0 or sp.shape[-2] == 0:
        return
    _c, H, W = dst.shape
    sh, sw = sp.shape[-2:]
    x0, y0, sx0, sy0 = int(x), int(y), 0, 0
    if x0 < 0:
        sx0, x0 = -x0, 0
    if y0 < 0:
        sy0, y0 = -y0, 0
    x1, y1 = min(W, x0 + sw - sx0), min(H, y0 + sh - sy0)
    if x0 >= x1 or y0 >= y1:
        return
    sp = sp[:, sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    below = dst[:, y0:y1, x0:x1]
    if dst.shape[0] == 3:
        a = sp[3:4]
        dst[:, y0:y1, x0:x1] = below * (1 - a) + sp[:3]
    else:
        dst[:, y0:y1, x0:x1] = sp + below * (1 - sp[3:4])


def _stamp_at(dst: torch.Tensor, sp: torch.Tensor, x: float, y: float) -> None:
    """Place a premul sprite at a fractional device position. Integer positions stay a sharp blit."""
    if sp.numel() == 0 or sp.shape[-1] == 0 or sp.shape[-2] == 0:
        return
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    if fx < 1e-4 and fy < 1e-4:
        _stamp(dst, sp, int(x0), int(y0))
        return
    _c, h, w = sp.shape
    shifted = torch.zeros(_c, h + 1, w + 1, device=sp.device, dtype=sp.dtype)
    w00, w10 = (1 - fx) * (1 - fy), fx * (1 - fy)
    w01, w11 = (1 - fx) * fy, fx * fy
    if w00:
        shifted[:, :h, :w] = shifted[:, :h, :w] + sp * w00
    if w10:
        shifted[:, :h, 1:] = shifted[:, :h, 1:] + sp * w10
    if w01:
        shifted[:, 1:, :w] = shifted[:, 1:, :w] + sp * w01
    if w11:
        shifted[:, 1:, 1:] = shifted[:, 1:, 1:] + sp * w11
    _stamp(dst, shifted, int(x0), int(y0))


def _blit(dst: torch.Tensor, k: float, sprite: torch.Tensor, box, sigma: float,
          opacity: float, dy: float) -> None:
    if sprite is None or sprite.numel() == 0 or opacity <= 1e-4:
        return
    sp = _fit(sprite.to(device=dst.device, dtype=dst.dtype), box, k)
    if sigma >= 0.25:
        sp = _blur(sp, sigma)
    if opacity < 1 - 1e-6:
        sp = sp * opacity
    # Sub-frame reveal is a fractional translateY (film.html:183). Rounding collapses
    # distinct times onto one pixel.
    _stamp_at(dst, sp, box[0] * k, box[1] * k + dy)


def _over_pair(shadow: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
    return fill + shadow.to(device=fill.device, dtype=fill.dtype) * (1 - fill[3:4])


def _scale_about(layer: torch.Tensor, cx: float, cy: float, s: float) -> torch.Tensor:
    # Pixel centres, align_corners=False: grid = pos/size*2−1. Scale about the line box centre.
    _c, H, W = layer.shape
    ys = cy + ((torch.arange(H, device=layer.device, dtype=layer.dtype) + 0.5) - cy) / s
    xs = cx + ((torch.arange(W, device=layer.device, dtype=layer.dtype) + 0.5) - cx) / s
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((gx / W * 2 - 1, gy / H * 2 - 1), dim=-1)
    return F.grid_sample(layer[None], grid[None], mode="bilinear", padding_mode="zeros", align_corners=False)[0]


def _layer_drop_shadow(layer: torch.Tensor, k: float) -> torch.Tensor:
    """Parent `drop-shadow(0 6px 30px rgba(0,0,0,.55))` of an already-animated gold layer.

    film.html:33. Blur radius 30 → σ = 15 design px. The shadow is of the completed line,
    then drawn behind it — not under each word as that word is painted.
    """
    sigma = 15.0 * k
    oy = 6.0 * k
    blurred = _blur(layer[3:4], sigma) * 0.55
    shadow = torch.zeros_like(layer)
    shadow[3:4] = blurred
    moved = torch.zeros_like(layer)
    _stamp_at(moved, shadow, 0.0, oy)
    return moved


def _paint_gold(dst: torch.Tensor, cv, block: TextBlock, line: TextLine, raster: RasterLine,
                t: float, ks: list[float]) -> None:
    # Glyphs land in one layer first. The parent drop-shadow (film.html:33) is then a single
    # shadow of that layer, behind it. Painting each word's shadow before its fill lets a
    # later word's shadow darken earlier glyphs.
    bp = _bp(line, block, t)
    layer = torch.zeros(4, dst.shape[-2], dst.shape[-1], device=dst.device, dtype=dst.dtype)
    any_ink = False
    for unit, k in zip(raster.units, ks):
        if k <= 1e-4:
            continue
        any_ink = True
        sigma = (1 - k) * 12 * cv.k
        dy = (1 - k) * line.rise * cv.k
        fill = _gold(unit.fill, unit.layout, unit.box, bp, raster.scale, dst.device, dst.dtype)
        _blit(layer, cv.k, fill, unit.box, sigma, k, dy)
    if not any_ink:
        return
    uniform = (max(ks) - min(ks)) < 1e-4 if ks else True
    if line.shadow and uniform and raster.line_shadow is not None:
        # Shared reveal: the captured group shadow is that parent filter (Chrome's kernel).
        # blur(k·α) = k·blur(α), so scaling it by the one k matches.
        k = ks[0]
        _blit(dst, cv.k, raster.line_shadow, raster.line_shadow_box,
              (1 - k) * 12 * cv.k, k, (1 - k) * line.rise * cv.k)
    elif line.shadow:
        _stamp(dst, _layer_drop_shadow(layer, cv.k), 0, 0)
    _stamp(dst, layer, 0, 0)


def _paint(dst: torch.Tensor, cv, block: TextBlock, line: TextLine, raster: RasterLine,
           t: float, fps: float, ks: list[float]) -> None:
    if line.gold:
        _paint_gold(dst, cv, block, line, raster, t, ks)
        return
    for unit, k in zip(raster.units, ks):
        if k <= 1e-4:
            continue
        sigma = (1 - k) * 12 * cv.k
        dy = (1 - k) * line.rise * cv.k
        sprite = unit.fill if unit.shadow is None else _over_pair(unit.shadow, unit.fill)
        _blit(dst, cv.k, sprite, unit.box, sigma, k, dy)


def _paint_line(dst: torch.Tensor, cv, block: TextBlock, line: TextLine, raster: RasterLine,
                t: float, fps: float) -> None:
    n = len(raster.units)
    if line.reveal == "pop":
        # film.html:289-291, per line. Spec span is the block window, not one counter step.
        span = block.end - block.start
        pop = 1.0 if span <= 0 else outc((t - line.at) * 5.0 / span)
        if pop <= 1e-4:
            return
        if abs(pop - 1.0) < 1e-4:
            _paint(dst, cv, block, line, raster, t, fps, [1.0] * n)
            return
        layer = torch.zeros(4, dst.shape[-2], dst.shape[-1], device=dst.device, dtype=dst.dtype)
        _paint(layer, cv, block, line, raster, t, fps, [1.0] * n)
        layer = _blur(layer, (1 - pop) * 14 * cv.k)
        scale = 1.25 + (1.0 - 1.25) * pop
        if abs(scale - 1.0) >= 1e-4:
            cx = (raster.line_box[0] + raster.line_box[2] / 2) * cv.k
            cy = (raster.line_box[1] + raster.line_box[3] / 2) * cv.k
            layer = _scale_about(layer, cx, cy, scale)
        _stamp(dst, layer * pop, 0, 0)
        return
    _paint(dst, cv, block, line, raster, t, fps, _unit_ks(line, n, t, fps))


def _rasters_of(rasters) -> tuple[tuple[RasterLine, ...], tuple[RasterImage, ...]]:
    if isinstance(rasters, RasterSet):
        return tuple(rasters.lines), tuple(rasters.subs)
    return tuple(rasters), ()


def draw_text(cv, img: torch.Tensor, blocks, subs, t: float, rasters, *, fps: float = 30.0) -> torch.Tensor:
    """Composite text at sub-frame `t`. Pure in its arguments.

    `subs` is in the task signature; the spec snippet omits it. `fps` (default 30) turns the
    0.2s/0.3s/0.5s sub ramps and the 0.02s vis threshold into frames. `rasters` is a `RasterSet`
    or a sequence of `RasterLine` parallel to the blocks' lines.
    """
    with torch.no_grad():
        out = img.clone()
        blocks = tuple(blocks)
        subs = tuple(subs)
        flat = [(block, line) for block in blocks for line in block.lines]
        lines, sub_rasters = _rasters_of(rasters)
        if len(lines) != len(flat):
            raise ValueError(f"{len(lines)} line rasters for {len(flat)} lines")
        if len(subs) > len(sub_rasters):
            raise ValueError(f"{len(sub_rasters)} sub rasters for {len(subs)} subs")
        cursor = 0
        for block in blocks:
            group = [(line, lines[cursor + i]) for i, line in enumerate(block.lines)]
            cursor += len(group)
            a = _vis(t, block, fps)
            if a <= 0.0:
                continue
            if a >= 1.0 - 1e-6:
                for line, raster in group:
                    _paint_line(out, cv, block, line, raster, t, fps)
                continue
            # Child filters first, then the block blur, then opacity (film.html:416-417).
            layer = torch.zeros(4, out.shape[-2], out.shape[-1], device=out.device, dtype=out.dtype)
            for line, raster in group:
                _paint_line(layer, cv, block, line, raster, t, fps)
            layer = _blur(layer, (1.0 - a) * block.blur * cv.k)
            _stamp(out, layer * a, 0, 0)
        # film.html:431-432. First window that contains t wins, even at opacity 0.
        lead, tail, fade = 0.2 * fps, 0.3 * fps, 0.5 * fps
        for sub, ras in zip(subs, sub_rasters):
            if t < sub.start - lead or t > sub.end + tail:
                continue
            op = min(sine((t - (sub.start - lead)) / fade), sine(((sub.end + tail) - t) / fade))
            if op > 1e-4 and ras.fill.numel():
                y_shift = (cv.H - DESIGN_H * cv.k) if ras.bottom else 0.0
                _blit(out, cv.k, ras.fill, ras.box, 0.0, op, y_shift)
            break
        return out
