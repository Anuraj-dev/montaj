"""Repeatable page capture. A YAML script becomes one clip in the assets folder.

The script compiles to a frame-accurate list of clicks and scrolls. Loading the
same file twice yields the same list, and the recorder plays that list. Page
timers and network timing are not frozen: the clock that repeats is the script.

Encoding is libx264. `Encoder` would otherwise probe NVENC by encoding a frame,
and this command must not touch the shared GPU.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, field_validator

from montaj.doctor import chromium
from montaj.encode import Encoder, probe
from montaj.spec import SpecError, _line, parse_wh, round_frame

_DUR = re.compile(r"^[+]?(?:\d+(?:\.\d*)?|\.\d+)[fs]$")
_BARE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_OUT_EXTS = {".mp4", ".mov"}
# Source plates are encoded again on export. veryfast stays off NVENC.
_CAPTURE_CRF = 18
_CAPTURE_PRESET = "veryfast"


class CaptureError(RuntimeError):
    """The script was valid and the recording failed."""


def _bare_duration(v: object) -> object:
    if isinstance(v, bool) or isinstance(v, (int, float)):
        raise ValueError("bare number")
    return v


def _size_text(v: object) -> object:
    if isinstance(v, bool) or isinstance(v, (int, float)):
        raise ValueError("expected WxH")
    return v


Dur = Annotated[str, BeforeValidator(_bare_duration)]
SizeStr = Annotated[str, BeforeValidator(_size_text)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScrollTo(_Model):
    x: float = 0.0
    y: float = 0.0
    dur: Dur | None = None

    @field_validator("x", "y", mode="before")
    @classmethod
    def _number(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("need a number, not a boolean")
        return v


def _coerce_scroll(v: object) -> object:
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError("need a number or {x, y}")
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("scroll is not finite")
        return {"y": float(v)}
    if isinstance(v, dict):
        return v
    raise ValueError("need a number or {x, y}")


class Action(_Model):
    at: Dur
    click: str | None = None
    scroll: Annotated[ScrollTo | None, BeforeValidator(_coerce_scroll)] = None


class CaptureScript(_Model):
    url: str
    dur: Dur
    size: SizeStr = "3840x2160"
    fps: int = 60
    css: str | None = None
    out: str | None = None
    actions: list[Action] = Field(default_factory=list)

    @field_validator("fps", mode="before")
    @classmethod
    def _fps_before(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("need an integer")
        return v

    @field_validator("css", "out", "url", mode="before")
    @classmethod
    def _text(cls, v: object) -> object:
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("need a string")
        return v


@dataclass(frozen=True)
class Event:
    """One click or one scroll sample, on one frame.

    `action` is the index in `actions`. `sample` is 0 for a click and counts
    the frames of a scroll tween. Playback order is `(frame, action, sample)`.
    """

    frame: int
    action: int
    sample: int
    kind: str
    selector: str = ""
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class CapturePlan:
    url: str
    width: int
    height: int
    fps: int
    n_frames: int
    css: str
    out: Path
    events: tuple[Event, ...]

    @property
    def seconds(self) -> float:
        return self.n_frames / self.fps

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class CaptureResult:
    path: Path
    frames: int
    seconds: float
    size: tuple[int, int]


def _yaml_path(loc: tuple) -> str:
    parts: list[str] = []
    for piece in loc:
        if isinstance(piece, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{piece}]"
            else:
                parts.append(f"[{piece}]")
        else:
            parts.append(str(piece))
    return ".".join(parts) or "."


def _from_pydantic(exc: ValidationError) -> list[str]:
    hints = {
        "string_type": "need a string",
        "int_type": "need an integer",
        "int_parsing": "need an integer",
        "float_type": "need a number",
        "bool_type": "need true or false",
        "list_type": "need a list",
        "model_type": "need a mapping",
    }
    lines: list[str] = []
    for err in exc.errors():
        path = _yaml_path(tuple(err["loc"]))
        kind = err["type"]
        if kind == "missing":
            lines.append(_line(path, "missing", "required"))
            continue
        if kind == "extra_forbidden":
            lines.append(_line(path, "unknown field", "remove this field"))
            continue
        msg = err["msg"]
        if msg.lower().startswith("value error, "):
            msg = msg[13:]
        hint = hints.get(kind, "check the script")
        if msg == "bare number":
            hint = "use 22f or 1.5s"
        elif msg == "expected WxH":
            hint = "use 3840x2160"
        elif msg == "need an integer":
            hint = "use 60"
        elif "not a boolean" in msg:
            hint = "use a number"
        elif msg.startswith("need a number or"):
            hint = "use 800 or {y: 800}"
        lines.append(_line(path, msg, hint))
    return lines


def _frames(raw: object, fps: int, path: str, errors: list[str], *, positive: bool) -> int | None:
    """Frame count at `fps`. `positive` rejects 0 (a capture length, not an instant scroll)."""
    if isinstance(raw, bool) or isinstance(raw, (int, float)) or (
        isinstance(raw, str) and _BARE.match(raw.strip())
    ):
        errors.append(_line(path, "bare number", "use 22f or 1.5s"))
        return None
    if not isinstance(raw, str):
        errors.append(_line(path, f"invalid duration {raw!r}", "use 22f or 1.5s"))
        return None
    text = raw.strip()
    if text.endswith("b"):
        errors.append(_line(path, "beats have no grid here", "use 22f or 1.5s"))
        return None
    if text.startswith("-"):
        errors.append(_line(path, "negative duration", "use a positive 22f or 1.5s"))
        return None
    if not _DUR.match(text):
        errors.append(_line(path, f"invalid duration {text!r}", "use 22f or 1.5s"))
        return None
    n = float(text[:-1])
    unit = text[-1]
    if not math.isfinite(n):
        errors.append(_line(path, "duration is not finite", "use a smaller duration"))
        return None
    frames = n if unit == "f" else n * fps
    if not math.isfinite(frames):
        errors.append(_line(path, "duration is not finite", "use a smaller duration"))
        return None
    got = round_frame(frames)
    if positive and got <= 0:
        errors.append(_line(path, "need a positive duration", "use 22f or 1.5s"))
        return None
    return got


def _assets_dirname(root: Path) -> str:
    """Film `assets:` folder, else `assets`. A broken montaj.yaml does not block capture."""
    spec = root / "montaj.yaml"
    if not spec.is_file():
        return "assets"
    try:
        data = yaml.safe_load(spec.read_text()) or {}
    except yaml.YAMLError:
        return "assets"
    if isinstance(data, dict) and isinstance(data.get("assets"), str) and data["assets"].strip():
        return data["assets"]
    return "assets"


def _output(root: Path, script: Path, raw: str | None, errors: list[str]) -> Path | None:
    if raw is None or not raw.strip():
        path = root / _assets_dirname(root) / f"{script.stem}.mp4"
    else:
        chosen = Path(raw.strip()).expanduser()
        path = chosen if chosen.is_absolute() else root / chosen
    if path.suffix.lower() not in _OUT_EXTS:
        errors.append(_line("out", f"expected a .mp4 or .mov clip, got {path.name!r}", "use assets/name.mp4"))
        return None
    return path.resolve()


def _url(raw: str, errors: list[str]) -> str | None:
    text = raw.strip()
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in {"http", "https", "file"}:
        shown = parts.scheme or text
        errors.append(_line("url", f"unsupported scheme {shown!r}", "use http, https or file"))
        return None
    if parts.scheme in {"http", "https"} and not parts.netloc:
        errors.append(_line("url", f"missing host in {text!r}", "use https://host/path"))
        return None
    if parts.scheme == "file" and not parts.path:
        errors.append(_line("url", "missing path", "use file:///path/to/page.html"))
        return None
    return text


def _compile_actions(
    actions: list[Action], fps: int, n_frames: int | None, errors: list[str],
) -> tuple[Event, ...]:
    """Each scroll is its own straight line. A later action does not rewrite an earlier tween."""
    events: list[Event] = []
    for i, action in enumerate(actions):
        path = f"actions[{i}]"
        click = action.click
        scroll = action.scroll
        if click is None and scroll is None:
            errors.append(_line(path, "need click or scroll", "add one"))
            continue
        if click is not None and scroll is not None:
            errors.append(_line(path, "click and scroll are mutually exclusive", "keep one"))
            continue
        start = _frames(action.at, fps, f"{path}.at", errors, positive=False)
        if start is None:
            continue
        if click is not None:
            selector = click.strip()
            if not selector:
                errors.append(_line(f"{path}.click", "empty selector", "use a CSS selector"))
                continue
            if n_frames is not None and start >= n_frames:
                errors.append(_line(f"{path}.at", "past the capture", "move at before dur"))
                continue
            events.append(Event(start, i, 0, "click", selector))
            continue
        assert scroll is not None
        span = _frames(scroll.dur, fps, f"{path}.scroll.dur", errors, positive=False) if scroll.dur else 1
        if span is None:
            continue
        span = max(span, 1)
        if not math.isfinite(scroll.x) or not math.isfinite(scroll.y):
            errors.append(_line(f"{path}.scroll", "scroll is not finite", "use finite x and y"))
            continue
        if n_frames is not None and start >= n_frames:
            errors.append(_line(f"{path}.at", "past the capture", "move at before dur"))
            continue
        if n_frames is not None and start + span - 1 >= n_frames:
            errors.append(_line(f"{path}.scroll.dur", "runs past the capture", "shorten dur or move at"))
            continue
        origin = _position_before(events, start, i)
        target = (float(scroll.x), float(scroll.y))
        for sample in range(span):
            u = (sample + 1) / span
            x = origin[0] + (target[0] - origin[0]) * u
            y = origin[1] + (target[1] - origin[1]) * u
            events.append(Event(start + sample, i, sample, "scroll", "", x, y))
    events.sort(key=lambda ev: (ev.frame, ev.action, ev.sample))
    return tuple(events)


def _position_before(events: list[Event], frame: int, action: int) -> tuple[float, float]:
    pos = (0.0, 0.0)
    for ev in events:
        if ev.kind != "scroll":
            continue
        if ev.frame < frame or (ev.frame == frame and ev.action < action):
            pos = (ev.x, ev.y)
    return pos


def load_capture(path: Path | str, root: Path | str) -> CapturePlan:
    """Compile `path` against the project at `root`. Raises SpecError on a bad script."""
    path = Path(path)
    root = Path(root)
    if not path.is_file():
        raise FileNotFoundError(f"capture script {path} does not exist")
    data = yaml.safe_load(path.read_text())
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SpecError([_line(".", "need a mapping", "set url and dur")])
    try:
        model = CaptureScript.model_validate(data)
    except ValidationError as exc:
        raise SpecError(_from_pydantic(exc)) from None

    errors: list[str] = []
    size = parse_wh(model.size, "size", errors, even=True)
    fps: int | None = model.fps
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        errors.append(_line("fps", "need an integer >= 1", "use 60"))
        fps = None
    url = _url(model.url, errors)
    out = _output(root, path, model.out, errors)
    n_frames: int | None = None
    if fps is not None:
        n_frames = _frames(model.dur, fps, "dur", errors, positive=True)
    syntax_fps = fps or 60
    events = _compile_actions(model.actions, syntax_fps, n_frames, errors)
    if errors or size is None or fps is None or n_frames is None or url is None or out is None:
        raise SpecError(errors or [_line(".", "invalid capture script", "check url, size, fps and dur")])
    return CapturePlan(
        url=url,
        width=size[0],
        height=size[1],
        fps=fps,
        n_frames=n_frames,
        css=model.css or "",
        out=out,
        events=events,
    )


def event_line(ev: Event, fps: int) -> str:
    """One log line. The same plan always prints the same lines."""
    t = ev.frame / fps
    if ev.kind == "click":
        return f"f{ev.frame} {t:.6f}s click {ev.selector}"
    return f"f{ev.frame} {t:.6f}s scroll {ev.x:.3f} {ev.y:.3f}"


def _note(log: Path, line: str) -> None:
    with log.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _close_encoder(enc: object, *, failed: bool) -> None:
    proc = getattr(enc, "_proc", None)
    if failed and proc is not None:
        if proc.poll() is None:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            proc.kill()
            proc.wait(timeout=5)
        fh = getattr(enc, "_fh", None)
        if fh is not None and not getattr(fh, "closed", True):
            fh.close()
        if hasattr(enc, "_closed"):
            enc._closed = True
        return
    close = getattr(enc, "close", None)
    if close is not None:
        close()


def _apply(page: Any, ev: Event) -> None:
    try:
        if ev.kind == "click":
            page.click(ev.selector)
        elif ev.kind == "scroll":
            page.scroll(ev.x, ev.y)
        else:
            raise CaptureError(f"unknown action {ev.kind}")
    except CaptureError as exc:
        field = "click" if ev.kind == "click" else "scroll"
        raise CaptureError(f"actions[{ev.action}].{field}: {exc} (check the selector)") from exc


def _play(plan: CapturePlan, page: Any, enc: Any, log: Path) -> None:
    page.open(plan.url, plan.width, plan.height)
    if plan.css:
        page.inject_css(plan.css)
    by_frame: dict[int, list[Event]] = {}
    for ev in plan.events:
        by_frame.setdefault(ev.frame, []).append(ev)
    for frame in range(plan.n_frames):
        for ev in by_frame.get(frame, ()):
            _apply(page, ev)
        enc.write(page.screenshot())
        if frame % plan.fps == 0 or frame + 1 == plan.n_frames:
            _note(log, f"frame {frame + 1}/{plan.n_frames}")


def _check_output(plan: CapturePlan, log: Path) -> None:
    info = probe(plan.out, log=log)
    if (info.width, info.height) != (plan.width, plan.height) or abs(info.fps - plan.fps) > 0.05:
        raise CaptureError(
            f"wrote {info.width}x{info.height} at {info.fps:g}fps, "
            f"expected {plan.width}x{plan.height} at {plan.fps}fps"
        )


def record(
    plan: CapturePlan,
    *,
    log: Path,
    page: Any = None,
    encoder: Any = None,
) -> CaptureResult:
    """Play `plan` into `plan.out`. Pass `page` / `encoder` only from tests."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    _note(
        log,
        f"capture url={plan.url} size={plan.width}x{plan.height} fps={plan.fps} "
        f"frames={plan.n_frames} out={plan.out.as_posix()}",
    )
    if plan.css:
        _note(log, f"css {len(plan.css)} bytes")
    for ev in plan.events:
        _note(log, event_line(ev, plan.fps))
    plan.out.parent.mkdir(parents=True, exist_ok=True)

    own_page = page is None
    own_encoder = encoder is None
    page = page if page is not None else ChromiumPage()
    enc = encoder if encoder is not None else Encoder(
        plan.out, plan.width, plan.height, plan.fps,
        encoder="libx264", preset=_CAPTURE_PRESET, crf=_CAPTURE_CRF, log=log,
    )
    try:
        _play(plan, page, enc, log)
        _close_encoder(enc, failed=False)
    except Exception:
        _close_encoder(enc, failed=True)
        raise
    finally:
        if own_page:
            page.close()
    if own_encoder:
        _check_output(plan, log)
    return CaptureResult(plan.out, plan.n_frames, plan.seconds, plan.size)


def run(
    script: Path | str,
    root: Path | str,
    *,
    log: Path,
    page: Any = None,
    encoder: Any = None,
) -> CaptureResult:
    """Load `script` and record it. `root` is the project directory (`-C`)."""
    plan = load_capture(script, root)
    return record(plan, log=log, page=page, encoder=encoder)


class ChromiumPage:
    """One headless Chromium. No process is started until `open`."""

    def __init__(self) -> None:
        self.w = 0
        self.h = 0
        self.proc: subprocess.Popen | None = None
        self.ws: Any = None
        self.udd: str | None = None
        self._chrome_log: Path | None = None
        self._log_fh: Any = None
        self._id = 0

    def open(self, url: str, width: int, height: int) -> None:
        self.close()
        self.w, self.h = int(width), int(height)
        try:
            self._launch()
            self._navigate(url)
        except Exception:
            self.close()
            raise

    def inject_css(self, css: str) -> None:
        expression = (
            "(() => {"
            f"const css = {json.dumps(css)};"
            "let el = document.getElementById('montaj-capture-css');"
            "if (!el) {"
            "  el = document.createElement('style');"
            "  el.id = 'montaj-capture-css';"
            "  (document.head || document.documentElement).appendChild(el);"
            "}"
            "el.textContent = css;"
            "if (document.body) void document.body.offsetHeight;"
            "return true;"
            "})()"
        )
        self._eval(expression)

    def click(self, selector: str) -> None:
        expression = (
            "(() => {"
            f"const sel = {json.dumps(selector)};"
            "let el;"
            "try { el = document.querySelector(sel); }"
            "catch (err) { return {ok: false, error: String(err)}; }"
            "if (!el) return {ok: false, error: 'no element ' + sel};"
            "el.click();"
            "if (document.body) void document.body.offsetHeight;"
            "return {ok: true};"
            "})()"
        )
        got = self._eval(expression) or {}
        if not got.get("ok"):
            raise CaptureError(got.get("error") or f"no element {selector!r}")

    def scroll(self, x: float, y: float) -> None:
        expression = (
            "(() => {"
            f"window.scrollTo({json.dumps(float(x))}, {json.dumps(float(y))});"
            "if (document.body) void document.body.offsetHeight;"
            "return true;"
            "})()"
        )
        self._eval(expression)

    def screenshot(self) -> np.ndarray:
        result = self._call("Page.captureScreenshot", {
            "format": "png",
        })
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(base64.b64decode(result["data"]))).convert("RGB")
        if image.size != (self.w, self.h):
            raise CaptureError(
                f"screenshot is {image.size[0]}x{image.size[1]}, expected {self.w}x{self.h}"
            )
        return np.asarray(image, dtype=np.uint8)

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        proc = self.proc
        self.proc = None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait(timeout=3)
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        if self.udd is not None:
            shutil.rmtree(self.udd, ignore_errors=True)
            self.udd = None

    def _launch(self) -> None:
        binary = chromium()
        if binary is None:
            raise CaptureError("chromium not found (install google-chrome or chromium)")
        exe = shutil.which(str(binary)) or str(binary)
        self.udd = tempfile.mkdtemp(prefix="montaj-capture-")
        self._chrome_log = Path(self.udd) / "chrome.log"
        self._log_fh = open(self._chrome_log, "w")
        self.proc = subprocess.Popen(
            [
                exe, "--headless=new", "--remote-debugging-port=0",
                f"--user-data-dir={self.udd}", "--no-first-run", "--no-default-browser-check",
                "--disable-extensions", "--hide-scrollbars", "--mute-audio", "--disable-gpu",
                "--force-color-profile=srgb", "--force-device-scale-factor=1",
                "--allow-file-access-from-files", "--remote-allow-origins=*",
                f"--window-size={max(self.w, 800)},{max(self.h, 600)}", "about:blank",
            ],
            stdout=subprocess.DEVNULL, stderr=self._log_fh, start_new_session=True,
        )
        port = self._debug_port()
        import websocket

        self.ws = websocket.create_connection(self._ws_url(port), timeout=30, suppress_origin=True)
        self._call("Page.enable")
        self._call("Runtime.enable")
        # Headless Chromium stalls captureScreenshot on windows below its minimum size, so the
        # window stays large and the viewport is overridden to the exact recording size.
        self._call("Emulation.setDeviceMetricsOverride", {
            "width": self.w, "height": self.h, "deviceScaleFactor": 1, "mobile": False,
        })

    def _debug_port(self) -> int:
        assert self.udd is not None
        port_file = Path(self.udd) / "DevToolsActivePort"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise CaptureError(f"chromium exited {self.proc.returncode}: {self._tail()}")
            if port_file.is_file():
                lines = port_file.read_text().splitlines()
                if lines and lines[0].isdigit():
                    return int(lines[0])
            time.sleep(0.05)
        raise CaptureError(f"chromium did not open a debug port: {self._tail()}")

    def _ws_url(self, port: int) -> str:
        deadline = time.monotonic() + 15
        url = f"http://127.0.0.1:{port}/json/list"
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    data = json.loads(resp.read().decode())
                for item in data:
                    if item.get("type") == "page" and item.get("webSocketDebuggerUrl"):
                        return item["webSocketDebuggerUrl"]
            except (OSError, json.JSONDecodeError, urllib.error.URLError):
                time.sleep(0.05)
        raise CaptureError(f"chromium has no page target: {self._tail()}")

    def _navigate(self, url: str) -> None:
        self._call("Page.navigate", {"url": url})
        deadline = time.monotonic() + 30
        last = ""
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise CaptureError(f"chromium exited {self.proc.returncode}: {self._tail()}")
            try:
                last = str(self._eval("document.readyState") or "")
            except CaptureError:
                last = ""
            if last == "complete":
                return
            time.sleep(0.05)
        raise CaptureError(f"page stayed {last or 'loading'} after 30s (did not finish loading)")

    def _call(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        if self.ws is None:
            raise CaptureError("chromium is not open")
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.ws.settimeout(max(0.1, deadline - time.monotonic()))
            try:
                raw = self.ws.recv()
            except Exception as exc:
                raise CaptureError(f"CDP {method} failed: {exc}") from exc
            msg = json.loads(raw)
            if msg.get("id") != mid:
                continue
            if "error" in msg:
                raise CaptureError(f"CDP {method}: {msg['error']}")
            return msg.get("result") or {}
        raise CaptureError(f"CDP {method} timed out")

    def _eval(self, expression: str) -> Any:
        result = self._call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        details = result.get("exceptionDetails")
        if details:
            text = details.get("text") or str(details.get("exception") or details)[:400]
            raise CaptureError(text)
        return result.get("result", {}).get("value")

    def _tail(self) -> str:
        if self._chrome_log is None or not self._chrome_log.is_file():
            return ""
        return " ".join(self._chrome_log.read_text(errors="replace").split())[-400:]
