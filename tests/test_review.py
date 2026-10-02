"""Review server, review.json, Range, shot assignment, summary budget. CPU only."""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from montaj.review.server import make_server
from montaj.review.summary import summary

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "recipes" / "birthday-short.yaml"
SRC = Path.home() / "Anuraj-dev/montaj-projects/_oracle/v2/src"


def _jpeg(folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (40, 80, 120)).save(folder / f"{stem}.jpg", "JPEG")


def _stems(yaml_text: str) -> list[str]:
    return sorted(set(re.findall(r"photo:\s*\"?([A-Za-z0-9_]+)\"?", yaml_text)))


def _mp4(path: Path, seconds: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25:duration={seconds}",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


def _project(tmp: Path, extra_text: str = "") -> Path:
    proj = tmp / "proj"
    assets = proj / "assets"
    assets.mkdir(parents=True)
    yaml_text = RECIPE.read_text()
    if extra_text:
        yaml_text = yaml_text.rstrip() + "\n" + extra_text + "\n"
    (proj / "montaj.yaml").write_text(yaml_text)
    for stem in _stems(yaml_text):
        src = SRC / f"{stem}.jpg"
        if src.is_file():
            os.symlink(src, assets / f"{stem}.jpg")
        else:
            _jpeg(assets, stem)
    _mp4(proj / "out" / "preview.mp4")
    return proj


def _tiny(tmp: Path, shots: str, stems: list[str], extra: str = "") -> Path:
    proj = tmp / "tiny"
    assets = proj / "assets"
    for stem in stems:
        _jpeg(assets, stem)
    (proj / "montaj.yaml").write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\n"
        f"assets: assets\nshots:\n{shots}{extra}\n"
    )
    _mp4(proj / "out" / "preview.mp4")
    return proj


class _Srv:
    def __init__(self, proj: Path, video: Path | None = None):
        self.httpd = make_server(proj, port=0, video=video)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.httpd.server_address[:2]
        self.base = f"http://{host}:{port}"
        self.proj = proj

    def close(self) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=2)
        self.httpd.server_close()


def _req(url: str, method: str = "GET", body=None, headers=None, raw: bool = False, data=None):
    hdrs = dict(headers or {})
    if data is None and body is not None:
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    elif data is not None:
        hdrs.setdefault("Content-Type", "application/json")
    req = Request(url, data=data, method=method, headers=hdrs)
    try:
        with urlopen(req, timeout=5) as resp:
            payload = resp.read()
            info = dict(resp.headers)
            if raw:
                return resp.status, payload, info
            if payload and "json" in (info.get("Content-Type") or ""):
                return resp.status, json.loads(payload), info
            return resp.status, payload.decode(), info
    except HTTPError as exc:
        payload = exc.read()
        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = payload.decode()
        return exc.code, parsed, dict(exc.headers)


@pytest.fixture
def srv(tmp_path: Path):
    proj = _project(tmp_path)
    s = _Srv(proj)
    try:
        yield s
    finally:
        s.close()


def test_binds_localhost_only(srv: _Srv) -> None:
    assert srv.httpd.server_address[0] == "127.0.0.1"


def test_state_and_page(srv: _Srv) -> None:
    code, data, _ = _req(srv.base + "/api/state")
    assert code == 200
    assert data["fps"] == 30
    assert data["n_frames"] > 0
    assert data["video"].endswith("preview.mp4")
    assert data["video_mtime"] > 0
    shots = data["shots"]
    assert shots[0]["index"] == 0
    assert shots[0]["label"] == "16"
    assert shots[-1]["label"] == "wall"
    assert shots[0]["start"] == 0.0
    assert shots[0]["start_f"] == 0
    code, html, hdrs = _req(srv.base + "/")
    assert code == 200
    assert "montaj" in html.lower()
    assert "<video" in html
    assert "cdn" not in html.lower()
    assert "loadedMtime" in html and "start_f" in html
    assert "text/html" in hdrs.get("Content-Type", "")


def test_status_comment_delete_roundtrip_and_atomic(srv: _Srv) -> None:
    code, data, _ = _req(srv.base + "/api/status", "POST", {"shot": 0, "status": "approved"})
    assert code == 200
    assert data["review"]["shots"][0]["status"] == "approved"
    code, data, _ = _req(srv.base + "/api/status", "POST", {"shot": 0, "status": "rejected"})
    assert data["review"]["shots"][0]["status"] == "rejected"
    code, data, _ = _req(srv.base + "/api/status", "POST", {"shot": 0, "status": None})
    assert data["review"]["shots"][0]["status"] is None

    code, data, _ = _req(srv.base + "/api/comment", "POST", {"t": 0.5, "text": "too dark"})
    assert code == 200
    comments = data["review"]["shots"][0]["comments"]
    assert comments[0]["text"] == "too dark"
    assert comments[0]["t"] == 0.5
    cid = comments[0]["id"]

    disk = json.loads((srv.proj / "review.json").read_text())
    assert disk["version"] == 1
    assert disk["shots"][0]["comments"][0]["text"] == "too dark"
    assert len(disk["spec_sha"]) == 64
    leftovers = list(srv.proj.glob(".review.json*.tmp"))
    assert leftovers == []

    code, data, _ = _req(srv.base + f"/api/comment/{cid}", "DELETE")
    assert code == 200
    assert data["review"]["shots"][0]["comments"] == []
    disk = json.loads((srv.proj / "review.json").read_text())
    assert disk["shots"][0]["comments"] == []


def test_range_returns_206_and_bytes(srv: _Srv) -> None:
    video = srv.proj / "out" / "preview.mp4"
    blob = video.read_bytes()
    code, payload, hdrs = _req(srv.base + "/video", headers={"Range": "bytes=0-15"}, raw=True)
    assert code == 206
    assert payload == blob[:16]
    assert hdrs.get("Content-Range") == f"bytes 0-15/{len(blob)}"
    assert hdrs.get("Content-Length") == "16"
    code, payload, hdrs = _req(srv.base + "/video", headers={"Range": "bytes=10-19"}, raw=True)
    assert code == 206 and payload == blob[10:20]


def test_comment_shot_assignment_at_boundaries(tmp_path: Path) -> None:
    proj = _tiny(tmp_path, "  - {photo: a, hold: 30f}\n  - {photo: b, hold: 30f}\n", ["a", "b"])
    s = _Srv(proj)
    try:
        _, data, _ = _req(s.base + "/api/comment", "POST", {"t": 0.0, "text": "start"})
        assert data["review"]["shots"][0]["comments"][0]["text"] == "start"
        _, data, _ = _req(s.base + "/api/comment", "POST", {"t": 1.0, "text": "cut"})
        assert data["review"]["shots"][1]["comments"][0]["text"] == "cut"
        _, data, _ = _req(s.base + "/api/comment", "POST", {"t": 0.999, "text": "almost"})
        texts = [c["text"] for c in data["review"]["shots"][0]["comments"]]
        assert "almost" in texts
        _, data, _ = _req(s.base + "/api/comment", "POST", {"t": 2.0, "text": "after"})
        assert data["review"]["general"][0]["text"] == "after"
    finally:
        s.close()


def test_stale_when_label_changes(tmp_path: Path) -> None:
    proj = _tiny(tmp_path, "  - {photo: a, hold: 30f}\n  - {photo: b, hold: 30f}\n", ["a", "b", "c"])
    s = _Srv(proj)
    try:
        _req(s.base + "/api/comment", "POST", {"t": 0.2, "text": "keep me"})
        _req(s.base + "/api/status", "POST", {"shot": 0, "status": "approved"})
        yaml = (proj / "montaj.yaml").read_text().replace("photo: a", "photo: c")
        (proj / "montaj.yaml").write_text(yaml)
        _, data, _ = _req(s.base + "/api/state")
        sh = data["review"]["shots"][0]
        assert sh["label"] == "c"
        assert sh["stale"] is True
        assert sh["status"] == "approved"
        assert sh["comments"][0]["text"] == "keep me"
        assert data["shots"][0]["label"] == "c"
    finally:
        s.close()


def test_first_text_line_on_strip(tmp_path: Path) -> None:
    extra = (
        "text:\n"
        "  - from: 0f\n"
        "    to: 30f\n"
        "    lines:\n"
        "      - {text: \"Hello Vidyut\", y: 700, at: 0f}\n"
    )
    proj = _tiny(tmp_path, "  - {photo: a, hold: 30f}\n  - {photo: b, hold: 30f}\n", ["a", "b"], extra=extra)
    s = _Srv(proj)
    try:
        _, data, _ = _req(s.base + "/api/state")
        assert data["shots"][0]["text"] == "Hello Vidyut"
        assert data["shots"][1]["text"] == ""
    finally:
        s.close()


def test_summary_format_and_token_budget(tmp_path: Path) -> None:
    shots = []
    for i in range(55):
        shots.append({
            "index": i,
            "start": round(i * 0.5, 2),
            "end": round((i + 1) * 0.5, 2),
            "label": f"p{i:02d}",
            "status": None,
            "comments": [],
        })
    notes = [
        (12, "rejected", ["too dark", "hold longer"]),
        (3, "approved", []),
        (20, None, ["cut sooner"]),
        (40, "rejected", ["flash too hot"]),
        (7, None, ["crop the ceiling"]),
        (33, "approved", ["nice"]),
        (1, None, ["wait"]),
        (50, "rejected", ["end sooner"]),
    ]
    for i, status, comments in notes:
        shots[i]["status"] = status
        shots[i]["comments"] = [{"id": i, "t": shots[i]["start"], "text": t} for t in comments]
    review = {
        "version": 1,
        "spec_sha": "abc",
        "video": "out/preview.mp4",
        "updated": "2026-10-02T00:00:00Z",
        "shots": shots,
        "general": [
            {"id": 100, "t": 1.5, "text": "music late"},
            {"id": 101, "t": 8.0, "text": "title bigger"},
        ],
    }
    (tmp_path / "review.json").write_text(json.dumps(review))
    lines, data = summary(tmp_path)
    body = "\n".join(lines)
    assert len(body) <= 1200
    assert any(line.startswith("#12 ") and "REJECT" in line and "too dark" in line for line in lines)
    assert lines[-1].startswith("OK review ")
    assert "2 approved" in lines[-1]
    assert "3 rejected" in lines[-1]
    assert lines[-1].endswith("comments")
    assert any(line.startswith("general ") for line in lines)
    empty_shots = [ln for ln in lines[:-1] if ln.startswith("#") is False and not ln.startswith("general")]
    assert empty_shots == [] or all(not ln.startswith("#") for ln in empty_shots)
    named = [ln for ln in lines if ln.startswith("#")]
    assert len(named) == len(notes)
    assert data["approved"] == 2
    assert data["rejected"] == 3
    assert data["comments"] == 8 + 2


def test_shot_seek_uses_frame_not_rounded_seconds(tmp_path: Path) -> None:
    proj = _tiny(tmp_path, "  - {photo: a, hold: 31f}\n  - {photo: b, hold: 30f}\n", ["a", "b"])
    s = _Srv(proj)
    try:
        _, data, _ = _req(s.base + "/api/state")
        s1 = data["shots"][1]
        assert s1["start_f"] == 31 and s1["end_f"] == 61
        assert s1["start"] == 1.03
        _, data, _ = _req(s.base + "/api/comment", "POST", {"t": 31 / 30, "text": "cut"})
        assert data["review"]["shots"][1]["comments"][0]["text"] == "cut"
        _, data, _ = _req(s.base + "/api/comment", "POST", {"t": 1.03, "text": "rounded"})
        assert any(c["text"] == "rounded" for c in data["review"]["shots"][0]["comments"])
    finally:
        s.close()


def test_comment_uses_one_timeline_and_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj = _tiny(tmp_path, "  - {photo: a, hold: 30f}\n  - {photo: b, hold: 30f}\n", ["a", "b", "c"])
    from montaj.review import server as srvmod

    n = {"i": 0}
    real = srvmod.load_timeline

    def wrapped(project_dir):
        n["i"] += 1
        tl = real(project_dir)
        if n["i"] == 1:
            p = Path(project_dir) / "montaj.yaml"
            p.write_text(p.read_text() + "  - {photo: c, hold: 30f}\n")
        return tl

    monkeypatch.setattr(srvmod, "load_timeline", wrapped)
    s = _Srv(proj)
    try:
        code, data, _ = _req(s.base + "/api/comment", "POST", {"t": 2.1, "text": "keep"})
        assert code == 200
        assert n["i"] == 1
        texts = [c["text"] for sh in data["review"]["shots"] for c in sh.get("comments") or []]
        texts += [c["text"] for c in data["review"]["general"]]
        assert "keep" in texts
    finally:
        s.close()


def test_corrupt_review_json_rejects_mutation(tmp_path: Path) -> None:
    proj = _tiny(tmp_path, "  - {photo: a, hold: 30f}\n", ["a"])
    garbage = "{not json"
    (proj / "review.json").write_text(garbage)
    s = _Srv(proj)
    try:
        code, data, _ = _req(s.base + "/api/comment", "POST", {"t": 0.1, "text": "note"})
        assert code == 400
        assert "review.json" in str(data.get("error", data))
        assert (proj / "review.json").read_text() == garbage
        code, _, _ = _req(s.base + "/api/status", "POST", {"shot": 0, "status": "approved"})
        assert code == 400
        assert (proj / "review.json").read_text() == garbage
    finally:
        s.close()


def test_nan_timestamp_rejected(tmp_path: Path) -> None:
    proj = _tiny(tmp_path, "  - {photo: a, hold: 30f}\n", ["a"])
    s = _Srv(proj)
    try:
        code, _, _ = _req(s.base + "/api/comment", "POST", {"t": "NaN", "text": "note"})
        assert code == 400
        code, _, _ = _req(s.base + "/api/comment", "POST", data=b'{"t":NaN,"text":"note"}')
        assert code == 400
        code, _, _ = _req(s.base + "/api/comment", "POST", {"t": float("inf"), "text": "note"})
        assert code == 400
        if (proj / "review.json").is_file():
            raw = (proj / "review.json").read_text()
            assert "NaN" not in raw and "Infinity" not in raw
        code, data, _ = _req(s.base + "/api/state")
        assert code == 200
        json.dumps(data, allow_nan=False)
    finally:
        s.close()


def test_negative_content_length_does_not_hang(srv: _Srv) -> None:
    host, port = srv.httpd.server_address[:2]
    sock = socket.create_connection((host, port), timeout=2)
    try:
        sock.settimeout(2)
        sock.sendall(
            b"POST /api/comment HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Length: -1\r\n"
            b"Content-Type: application/json\r\n\r\n"
        )
        chunks = b""
        while True:
            bit = sock.recv(4096)
            if not bit:
                break
            chunks += bit
            if b"\r\n\r\n" in chunks:
                break
        assert b" 400 " in chunks.split(b"\r\n", 1)[0]
    finally:
        sock.close()


def test_excessive_content_length_rejected(srv: _Srv) -> None:
    host, port = srv.httpd.server_address[:2]
    sock = socket.create_connection((host, port), timeout=2)
    try:
        sock.settimeout(2)
        sock.sendall(
            b"POST /api/comment HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Length: 99999999\r\n"
            b"Content-Type: application/json\r\n\r\n"
        )
        chunks = b""
        while True:
            bit = sock.recv(4096)
            if not bit:
                break
            chunks += bit
            if b"\r\n\r\n" in chunks:
                break
        assert b" 400 " in chunks.split(b"\r\n", 1)[0]
    finally:
        sock.close()


def test_video_content_length_matches_open_file(srv: _Srv) -> None:
    blob = (srv.proj / "out" / "preview.mp4").read_bytes()
    code, payload, hdrs = _req(srv.base + "/video", raw=True)
    assert code == 200
    assert payload == blob
    assert hdrs.get("Content-Length") == str(len(blob))


def test_default_video_picks_newest_each_request(tmp_path: Path) -> None:
    proj = _tiny(tmp_path, "  - {photo: a, hold: 30f}\n", ["a"])
    preview = proj / "out" / "preview.mp4"
    final = _mp4(proj / "out" / "final.mp4")
    os.utime(preview, (1_000, 1_000))
    os.utime(final, (2_000, 2_000))
    s = _Srv(proj)
    try:
        _, data, _ = _req(s.base + "/api/state")
        assert data["video"].endswith("final.mp4")
        os.utime(preview, (3_000, 3_000))
        _, data, _ = _req(s.base + "/api/state")
        assert data["video"].endswith("preview.mp4")
    finally:
        s.close()


def test_explicit_video_does_not_fall_back(tmp_path: Path) -> None:
    proj = _tiny(tmp_path, "  - {photo: a, hold: 30f}\n", ["a"])
    preview = proj / "out" / "preview.mp4"
    _mp4(proj / "out" / "final.mp4")
    os.utime(preview, (1, 1))
    os.utime(proj / "out" / "final.mp4", (9, 9))
    s = _Srv(proj, video=preview)
    try:
        _, data, _ = _req(s.base + "/api/state")
        assert data["video"].endswith("preview.mp4")
        preview.unlink()
        _, data, _ = _req(s.base + "/api/state")
        assert data["video"] == ""
        _mp4(preview)
        _, data, _ = _req(s.base + "/api/state")
        assert data["video"].endswith("preview.mp4")
    finally:
        s.close()


def test_spec_sha_frozen_at_server_start(tmp_path: Path) -> None:
    proj = _tiny(tmp_path, "  - {photo: a, hold: 30f}\n", ["a"])
    sha_a = hashlib.sha256((proj / "montaj.yaml").read_bytes()).hexdigest()
    s = _Srv(proj)
    try:
        (proj / "montaj.yaml").write_text((proj / "montaj.yaml").read_text() + "\n# edit\n")
        assert hashlib.sha256((proj / "montaj.yaml").read_bytes()).hexdigest() != sha_a
        _, data, _ = _req(s.base + "/api/comment", "POST", {"t": 0.1, "text": "note"})
        assert data["review"]["spec_sha"] == sha_a
        disk = json.loads((proj / "review.json").read_text())
        assert disk["spec_sha"] == sha_a
    finally:
        s.close()


def test_first_text_uses_line_appearance(tmp_path: Path) -> None:
    extra = (
        "text:\n"
        "  - from: 0f\n"
        "    to: 90f\n"
        "    lines:\n"
        "      - {text: \"Late\", y: 700, at: 60f}\n"
    )
    proj = _tiny(
        tmp_path,
        "  - {photo: a, hold: 30f}\n  - {photo: b, hold: 60f}\n",
        ["a", "b"],
        extra=extra,
    )
    s = _Srv(proj)
    try:
        _, data, _ = _req(s.base + "/api/state")
        assert data["shots"][0]["text"] == ""
        assert data["shots"][1]["text"] == "Late"
    finally:
        s.close()


def test_status_requires_integer_known_shot(srv: _Srv) -> None:
    code, _, _ = _req(srv.base + "/api/status", "POST", {"shot": 1.5, "status": "approved"})
    assert code == 400
    code, _, _ = _req(srv.base + "/api/status", "POST", {"shot": True, "status": "approved"})
    assert code == 400
    code, data, _ = _req(srv.base + "/api/status", "POST", {"shot": 9999, "status": "approved"})
    assert code == 400
    assert isinstance(data, dict) and "error" in data


def test_bad_json_missing_fields_unknown_comment(srv: _Srv) -> None:
    code, data, _ = _req(srv.base + "/api/comment", "POST", data=b"{")
    assert code == 400 and isinstance(data, dict)
    code, _, _ = _req(srv.base + "/api/comment", "POST", {"text": "x"})
    assert code == 400
    code, data, _ = _req(srv.base + "/api/comment/99999", "DELETE")
    assert code == 404 and isinstance(data, dict) and "error" in data
    code, data, _ = _req(srv.base + "/api/comment/abc", "DELETE")
    assert code == 400 and isinstance(data, dict)
