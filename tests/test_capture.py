"""montaj capture: plan compilation, playback order, and one real Chromium recording."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from montaj.capture import CaptureError, load_capture, record
from montaj.cli import main
from montaj.doctor import chromium
from montaj.encode import probe
from montaj.spec import SpecError

PAGE = """<!doctype html><meta charset=utf-8>
<style>body{margin:0;background:#fff;height:3000px}
#b{position:fixed;top:0;left:0;width:100%;height:100%;background:#e22}
#go{position:absolute;top:10px;left:10px;width:40px;height:40px;background:#00f;z-index:2}</style>
<div id=b></div><div id=go onclick="document.getElementById('b').style.background='#2e2'"></div>
"""


def _script(tmp: Path, body: str) -> Path:
    page = tmp / "page.html"
    page.write_text(PAGE)
    s = tmp / "demo.yaml"
    s.write_text(f"url: {page.as_uri()}\n{body}")
    return s


def test_plan_is_deterministic_and_frame_exact(tmp_path: Path) -> None:
    s = _script(tmp_path, "size: 64x48\nfps: 10\ndur: 2s\nactions:\n"
                          "  - {at: 0.5s, click: '#go'}\n  - {at: 1s, scroll: {y: 100, dur: 4f}}\n")
    a, b = load_capture(s, tmp_path), load_capture(s, tmp_path)
    assert a == b
    assert a.n_frames == 20 and a.out == (tmp_path / "assets" / "demo.mp4").resolve()
    assert [(e.frame, e.kind) for e in a.events] == [(5, "click")] + [(10 + i, "scroll") for i in range(4)]
    assert [e.y for e in a.events[1:]] == [25.0, 50.0, 75.0, 100.0]


@pytest.mark.parametrize("body,needle", [
    ("dur: 1s\nactions:\n  - {at: 2s, click: a}\n", "past the capture"),
    ("dur: 1s\nactions:\n  - {at: 1b, click: a}\n", "beats"),
    ("dur: 1\n", "bare number"),
    ("dur: 1s\nsize: 63x48\n", "size"),
    ("dur: 1s\nactions:\n  - {at: 0s}\n", "need click or scroll"),
    ("dur: 1s\nout: x.gif\n", ".mp4"),
])
def test_bad_scripts(tmp_path: Path, body: str, needle: str) -> None:
    with pytest.raises(SpecError) as ei:
        load_capture(_script(tmp_path, body), tmp_path)
    assert needle in "\n".join(ei.value.errors if hasattr(ei.value, "errors") else [str(ei.value)])


def test_bad_url_scheme(tmp_path: Path) -> None:
    s = tmp_path / "x.yaml"
    s.write_text("url: ftp://x/y\ndur: 1s\n")
    with pytest.raises(SpecError):
        load_capture(s, tmp_path)


class FakePage:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def open(self, url, w, h): self.calls.append(("open", w, h))
    def inject_css(self, css): self.calls.append(("css", css))
    def click(self, sel): self.calls.append(("click", sel))
    def scroll(self, x, y): self.calls.append(("scroll", x, y))
    def close(self): pass

    def screenshot(self):
        self.calls.append(("shot",))
        return np.zeros((2, 2, 3), np.uint8)


class FakeEnc:
    def __init__(self) -> None:
        self.n = 0
        self.closed = False

    def write(self, f): self.n += 1
    def close(self): self.closed = True


def test_playback_order_repeats(tmp_path: Path) -> None:
    s = _script(tmp_path, "size: 64x48\nfps: 4\ndur: 1s\ncss: 'a{b:c}'\nactions:\n"
                          "  - {at: 2f, click: '#go'}\n  - {at: 2f, scroll: 50}\n")
    plan = load_capture(s, tmp_path)
    runs = []
    for i in range(2):
        p, e = FakePage(), FakeEnc()
        record(plan, log=tmp_path / f"b{i}.log", page=p, encoder=e)
        assert e.n == 4 and e.closed
        runs.append(p.calls)
    assert runs[0] == runs[1]
    assert runs[0] == [("open", 64, 48), ("css", "a{b:c}"), ("shot",), ("shot",),
                       ("click", "#go"), ("scroll", 0.0, 50.0), ("shot",), ("shot",)]
    assert (tmp_path / "b0.log").read_text() == (tmp_path / "b0.log").read_text()


def test_missing_selector_is_error(tmp_path: Path) -> None:
    class Bad(FakePage):
        def click(self, sel): raise CaptureError("no element " + sel)
    plan = load_capture(_script(tmp_path, "size: 64x48\nfps: 4\ndur: 1s\nactions:\n  - {at: 0f, click: '#nope'}\n"), tmp_path)
    with pytest.raises(CaptureError, match=r"actions\[0\]\.click"):
        record(plan, log=tmp_path / "l.log", page=Bad(), encoder=FakeEnc())


@pytest.mark.skipif(chromium() is None, reason="no chromium")
def test_real_recording_cli_and_repeatable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = _script(tmp_path, "size: 64x48\nfps: 10\ndur: 1s\ncss: '#b{opacity:1}'\nactions:\n"
                          "  - {at: 0.5s, click: '#go'}\n  - {at: 0.7s, scroll: 200}\n")
    for _ in range(2):
        assert main(["-C", str(tmp_path), "capture", str(s)]) == 0
        out = capsys.readouterr().out.strip().splitlines()
        assert len(out) == 1 and out[0].startswith("OK assets/demo.mp4 1.00s 64x48")
        clip = tmp_path / "assets" / "demo.mp4"
        info = probe(clip)
        assert (info.width, info.height) == (64, 48) and abs(info.fps - 10) < 0.05
    assert (tmp_path / "build" / "capture.log").is_file()
    import subprocess
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(tmp_path / "assets" / "demo.mp4"),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(10, 48, 64, 3)
    assert frames[0, 40, 60, 0] > 150 and frames[0, 40, 60, 1] < 100   # red before click
    assert frames[9, 40, 60, 1] > 150 and frames[9, 40, 60, 0] < 100   # green after click
