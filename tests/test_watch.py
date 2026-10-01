"""Watch debounce, ERR recovery, stop event. `pipeline.render` is monkeypatched."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from montaj import pipeline, watch as watch_mod


def _jpeg(folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(folder / f"{stem}.jpg", "JPEG")


def _spec(proj: Path, body: str = "  - {photo: a, hold: 6f}\n") -> None:
    (proj / "montaj.yaml").write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\n"
        f"assets: assets\nshots:\n{body}"
    )


def _project(tmp: Path) -> Path:
    proj = tmp / "proj"
    _jpeg(proj / "assets", "a")
    _spec(proj)
    return proj


def _wait(pred, timeout: float = 6.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError("timeout")


def test_one_render_per_burst_err_then_recover_and_stop(tmp_path: Path, monkeypatch) -> None:
    proj = _project(tmp_path)
    calls: list[tuple[Path, str]] = []

    def fake_render(spec_path, mode="preview"):
        calls.append((Path(spec_path), mode))
        out = Path(spec_path).parent / "out" / f"{mode}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00\x00")
        return SimpleNamespace(
            path=out, n_frames=6, seconds=0.2, size=(540, 960), rendered=1, cached=0
        )

    monkeypatch.setattr(pipeline, "render", fake_render)
    lines: list[str] = []
    stop = threading.Event()
    thread = threading.Thread(
        target=watch_mod.watch,
        kwargs={"project_dir": proj, "mode": "preview", "out": lines.append, "stop": stop},
        daemon=True,
    )
    thread.start()
    try:
        time.sleep(0.6)
        assert calls == []

        for _ in range(4):
            _spec(proj, "  - {photo: a, hold: 8f}\n")
            time.sleep(0.05)
        _wait(lambda: len(calls) >= 1)
        time.sleep(0.9)
        assert len(calls) == 1
        assert calls[0][1] == "preview"
        assert any(ln.startswith("OK ") for ln in lines)

        (proj / "montaj.yaml").write_text("this is not yaml: [")
        _wait(lambda: any(ln.startswith("ERR ") for ln in lines))
        n = len(calls)
        time.sleep(0.9)
        assert len(calls) == n
        n_ok = sum(1 for ln in lines if ln.startswith("OK "))
        err_at = max(i for i, ln in enumerate(lines) if ln.startswith("ERR "))

        _spec(proj, "  - {photo: a, hold: 10f}\n")
        _wait(lambda: any(i > err_at and ln.startswith("OK ") for i, ln in enumerate(lines)))
        recovered = [ln for i, ln in enumerate(lines) if i > err_at and ln.startswith("OK ")]
        assert recovered, lines
        assert "out/preview.mp4" in recovered[0]
        assert sum(1 for ln in lines if ln.startswith("OK ")) == n_ok + 1
    finally:
        stop.set()
        thread.join(timeout=3)
    assert lines[-1] == "OK watch stopped"
    assert (proj / "build" / "watch.log").is_file()


def test_fingerprint_tolerates_vanishing_files(tmp_path: Path, monkeypatch) -> None:
    proj = _project(tmp_path)
    orig = Path.stat

    def flaky(self, *a, **k):
        if self.name == "a.jpg":
            raise FileNotFoundError("gone")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "stat", flaky)
    fp = watch_mod._fingerprint(proj)
    assert "a.jpg" not in [name for name, *_ in fp]
    assert any(name == "montaj.yaml" for name, *_ in fp)
    monkeypatch.setattr(Path, "stat", orig)
    (proj / "assets").rename(proj / "assets-gone")
    watch_mod._fingerprint(proj)


def test_watch_keeps_running_after_poll_error(tmp_path: Path, monkeypatch) -> None:
    proj = _project(tmp_path)
    calls: list = []
    n = {"i": 0}
    orig_fp = watch_mod._fingerprint

    def flaky(root):
        n["i"] += 1
        if n["i"] == 2:
            raise RuntimeError("poll boom")
        return orig_fp(root)

    def fake_render(spec_path, mode="preview"):
        calls.append(mode)
        out = Path(spec_path).parent / "out" / f"{mode}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00\x00")
        return SimpleNamespace(
            path=out, n_frames=6, seconds=0.2, size=(540, 960), rendered=1, cached=0
        )

    monkeypatch.setattr(watch_mod, "_fingerprint", flaky)
    monkeypatch.setattr(pipeline, "render", fake_render)
    lines: list[str] = []
    stop = threading.Event()
    thread = threading.Thread(
        target=watch_mod.watch,
        kwargs={"project_dir": proj, "mode": "preview", "out": lines.append, "stop": stop},
        daemon=True,
    )
    thread.start()
    try:
        _wait(lambda: any("poll boom" in ln for ln in lines))
        _spec(proj, "  - {photo: a, hold: 8f}\n")
        _wait(lambda: any(ln.startswith("OK ") and "preview.mp4" in ln for ln in lines))
        assert calls
    finally:
        stop.set()
        thread.join(timeout=3)
    assert lines[-1] == "OK watch stopped"
