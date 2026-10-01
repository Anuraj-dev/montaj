"""127.0.0.1 review page + JSON API. Stdlib http.server, Range-capable video."""
from __future__ import annotations

import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from montaj.review.summary import (
    ReviewLoadError,
    load_raw,
    load_timeline,
    merge_review,
    save_review,
    spec_sha,
    timeline_shots,
)

PAGE = Path(__file__).with_name("page.html")
CHUNK = 64 * 1024
MAX_BODY = 1_048_576


def _pick_video(root: Path, given: Path | str | None) -> Path | None:
    if given is not None:
        p = Path(given)
        return p if p.is_absolute() else (root / p)
    cands = [p for p in (root / "out" / "preview.mp4", root / "out" / "final.mp4") if p.is_file()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _rel_video(root: Path, video: Path | None) -> str:
    if video is None:
        return ""
    try:
        return video.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return video.as_posix()


def _mtime(path: Path | None) -> float:
    if path is None or not path.is_file():
        return 0.0
    return path.stat().st_mtime


def _next_id(rev: dict) -> int:
    ids = [0]
    for sh in rev.get("shots") or []:
        for c in sh.get("comments") or []:
            if isinstance(c, dict) and isinstance(c.get("id"), int):
                ids.append(c["id"])
    for c in rev.get("general") or []:
        if isinstance(c, dict) and isinstance(c.get("id"), int):
            ids.append(c["id"])
    return max(ids) + 1


def _shot_index_at(tl, t: float) -> int | None:
    f = float(t) * float(tl.fps)
    for sh in tl.shots:
        if sh.start <= f < sh.end:
            return sh.index
    return None


def _strict_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}: integer required")
    return value


def _finite_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name}: finite number required")
    t = float(value)
    if not math.isfinite(t):
        raise ValueError(f"{name}: finite number required")
    return t


class _Ctx:
    def __init__(self, root: Path, given: Path | None, explicit: bool, log: Path, start_sha: str):
        self.root = root
        self.given = given
        self.explicit = explicit
        self.log = log
        self.lock = threading.Lock()
        self.start_sha = start_sha


def _current_video(ctx: _Ctx) -> Path | None:
    if ctx.explicit:
        return ctx.given if ctx.given is not None and ctx.given.is_file() else None
    return _pick_video(ctx.root, None)


def _state(ctx: _Ctx, tl=None) -> dict:
    if tl is None:
        tl = load_timeline(ctx.root)
    video = _current_video(ctx)
    rel = _rel_video(ctx.root, video)
    shots = timeline_shots(tl)
    rev = merge_review(load_raw(ctx.root), shots, ctx.start_sha, rel)
    return {
        "video": rel,
        "video_mtime": _mtime(video),
        "fps": tl.fps,
        "n_frames": tl.n_frames,
        "shots": shots,
        "review": rev,
    }


def _mutate(ctx: _Ctx, fn) -> dict:
    with ctx.lock:
        tl = load_timeline(ctx.root)
        state = _state(ctx, tl)
        rev = fn(state, tl)
        save_review(ctx.root, rev)
        state["review"] = rev
        return state


def _parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Single byte range, or None for the whole file. Raises ValueError if unsatisfiable."""
    if not header or not header.startswith("bytes="):
        return None
    spec = header[6:].strip()
    if "," in spec:
        spec = spec.split(",", 1)[0].strip()
    if spec.startswith("-"):
        suffix = int(spec)
        if suffix == 0:
            raise ValueError("empty range")
        start = max(0, size + suffix)
        end = size - 1
    else:
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
    if start >= size or start < 0 or end < start:
        raise ValueError("unsatisfiable")
    return start, min(end, size - 1)


def _send_json(handler: BaseHTTPRequestHandler, code: int, payload) -> None:
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(body)


def _reject_constant(name: str):
    raise ValueError(f"non-finite: {name}")


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    raw_len = handler.headers.get("Content-Length")
    if raw_len is None:
        raise ValueError("Content-Length required")
    try:
        length = int(raw_len)
    except (TypeError, ValueError) as exc:
        raise ValueError("Content-Length") from exc
    if length < 0 or length > MAX_BODY:
        raise ValueError("Content-Length")
    raw = handler.rfile.read(length) if length else b"{}"
    data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise ValueError("json object required")
    return data


def _send_video(handler: BaseHTTPRequestHandler, path: Path) -> None:
    try:
        fh = path.open("rb")
    except FileNotFoundError:
        handler.send_error(404, "no video")
        return
    try:
        size = os.fstat(fh.fileno()).st_size
        try:
            span = _parse_range(handler.headers.get("Range"), size)
        except ValueError:
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{size}")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        if span is None:
            start, end, code = 0, size - 1 if size else 0, 200
            length = size
        else:
            start, end = span
            code, length = 206, end - start + 1
        handler.send_response(code)
        handler.send_header("Content-Type", "video/mp4")
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Length", str(length))
        if code == 206:
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        handler.end_headers()
        if handler.command == "HEAD" or length <= 0:
            return
        fh.seek(start)
        left = length
        while left > 0:
            chunk = fh.read(min(CHUNK, left))
            if not chunk:
                break
            handler.wfile.write(chunk)
            left -= len(chunk)
    finally:
        fh.close()


def _make_handler(ctx: _Ctx):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            ctx.log.parent.mkdir(parents=True, exist_ok=True)
            with ctx.log.open("a", encoding="utf-8") as fh:
                fh.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def do_HEAD(self) -> None:
            self._dispatch()

        def do_GET(self) -> None:
            self._dispatch()

        def do_POST(self) -> None:
            self._dispatch()

        def do_DELETE(self) -> None:
            self._dispatch()

        def _dispatch(self) -> None:
            path = urlparse(self.path).path
            try:
                if self.command in ("GET", "HEAD") and path in ("/", "/index.html"):
                    body = PAGE.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(body)
                    return
                if self.command in ("GET", "HEAD") and path == "/video":
                    video = _current_video(ctx)
                    if video is None:
                        self.send_error(404, "no video")
                        return
                    _send_video(self, video)
                    return
                if self.command in ("GET", "HEAD") and path == "/api/state":
                    _send_json(self, 200, _state(ctx))
                    return
                if self.command == "POST" and path == "/api/status":
                    body = _read_json(self)
                    _send_json(self, 200, _post_status(ctx, body))
                    return
                if self.command == "POST" and path == "/api/comment":
                    body = _read_json(self)
                    _send_json(self, 200, _post_comment(ctx, body))
                    return
                if self.command == "DELETE" and path.startswith("/api/comment/"):
                    token = path.rsplit("/", 1)[-1]
                    if not token.isdigit():
                        raise ValueError("comment id: integer required")
                    state = _delete_comment(ctx, int(token))
                    if state is None:
                        _send_json(self, 404, {"error": "comment not found"})
                        return
                    _send_json(self, 200, state)
                    return
                self.send_error(404, "not found")
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                _send_json(self, 400, {"error": str(exc)})
            except ReviewLoadError as exc:
                _send_json(self, 400, {"error": str(exc)})
            except FileNotFoundError as exc:
                _send_json(self, 404, {"error": str(exc)})
            except Exception as exc:
                _send_json(self, 500, {"error": str(exc)})

    return Handler


def _post_status(ctx: _Ctx, body: dict) -> dict:
    shot = _strict_int(body["shot"], "shot")
    status = body.get("status")
    if status not in ("approved", "rejected", None):
        raise ValueError("status: approved, rejected, or null")

    def apply(state: dict, tl) -> dict:
        rev = state["review"]
        for sh in rev["shots"]:
            if sh["index"] == shot:
                sh["status"] = status
                return rev
        raise ValueError(f"shot {shot}")

    return _mutate(ctx, apply)


def _post_comment(ctx: _Ctx, body: dict) -> dict:
    raw_t = _finite_float(body["t"], "t")
    t = round(raw_t, 2)
    text = str(body.get("text") or "").strip()
    if not text:
        raise ValueError("text: required")

    def apply(state: dict, tl) -> dict:
        rev = state["review"]
        idx = _shot_index_at(tl, raw_t)
        item = {"id": _next_id(rev), "t": t, "text": text}
        if idx is None:
            rev.setdefault("general", []).append(item)
            return rev
        for sh in rev["shots"]:
            if sh["index"] == idx:
                sh.setdefault("comments", []).append(item)
                return rev
        raise ValueError(f"shot {idx}")

    return _mutate(ctx, apply)


def _delete_comment(ctx: _Ctx, cid: int) -> dict | None:
    found = {"n": 0}

    def apply(state: dict, tl) -> dict:
        rev = state["review"]
        for sh in rev["shots"]:
            before = len(sh.get("comments") or [])
            sh["comments"] = [c for c in (sh.get("comments") or []) if c.get("id") != cid]
            found["n"] += before - len(sh["comments"])
        before = len(rev.get("general") or [])
        rev["general"] = [c for c in (rev.get("general") or []) if c.get("id") != cid]
        found["n"] += before - len(rev["general"])
        return rev

    state = _mutate(ctx, apply)
    return state if found["n"] else None


def make_server(
    project_dir: Path | str,
    port: int = 0,
    video: Path | str | None = None,
    log: Path | str | None = None,
) -> ThreadingHTTPServer:
    root = Path(project_dir).expanduser().resolve()
    log_path = Path(log) if log is not None else root / "build" / "review.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    explicit = video is not None
    given = _pick_video(root, video) if explicit else None
    ctx = _Ctx(root, given, explicit, log_path, spec_sha(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", int(port)), _make_handler(ctx))
    httpd.ctx = ctx  # type: ignore[attr-defined]
    return httpd


def serve(
    project_dir: Path | str,
    port: int,
    video: Path | str | None = None,
    log: Path | str | None = None,
) -> None:
    httpd = make_server(project_dir, port, video=video, log=log)
    host, bound = httpd.server_address[:2]
    ctx = httpd.ctx  # type: ignore[attr-defined]
    vid = _current_video(ctx)
    shown = vid.as_posix() if vid is not None else (ctx.given.as_posix() if ctx.given is not None else "")
    print(f"OK review http://{host}:{bound} video={shown}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
