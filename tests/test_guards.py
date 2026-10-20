"""`check:` content guards: digits (OCR) and faces, sampled per shot. CPU only."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from montaj import guards
from montaj.cli import main
from montaj.spec import SpecError, load_spec

SPEC = (
    "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 120\n  look: warm-film\n"
    "assets: assets\nshots:\n  - {{photo: a, hold: 30f}}\n  - {{photo: b, hold: 30f}}\n{extra}"
)


def _project(tmp: Path, extra: str = "") -> Path:
    proj = tmp / "proj"
    (proj / "assets").mkdir(parents=True)
    for stem in "ab":
        Image.new("RGB", (8, 8), (180, 90, 40)).save(proj / "assets" / f"{stem}.jpg")
    (proj / "montaj.yaml").write_text(SPEC.format(extra=extra))
    return proj


def _video(path: Path, digits: str | None) -> None:
    """Two 1 s shots: shot 0 shows `digits` (or plain white), shot 1 is plain white."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (640, 480), "white")
    if digits:
        draw = ImageDraw.Draw(img)
        draw.text((60, 140), digits, fill="black", font=ImageFont.load_default(size=200))
    png, white = path.with_name("d.png"), path.with_name("w.png")
    img.save(png)
    Image.new("RGB", (640, 480), "white").save(white)
    for name, src, out in (("a", png, "a.mp4"), ("b", white, "b.mp4")):
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-loop", "1", "-framerate", "30",
             "-i", str(src), "-t", "1", "-pix_fmt", "yuv420p", str(path.with_name(out))],
            check=True,
        )
    lst = path.with_name("l.txt")
    lst.write_text(f"file '{path.with_name('a.mp4')}'\nfile '{path.with_name('b.mp4')}'\n")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(path)], check=True)


def _check(proj: Path, capsys) -> tuple[int, list[str]]:
    code = main(["-C", str(proj), "check"])
    return code, capsys.readouterr().out.splitlines()


def test_sample_frames_spread_and_distinct() -> None:
    assert guards.sample_frames(0, 30, 3) == [5, 15, 25]
    assert guards.sample_frames(10, 12, 5) == [10, 11]
    assert guards.sample_frames(7, 7, 2) == [7]


def test_guards_off_by_default(tmp_path: Path, capsys) -> None:
    proj = _project(tmp_path)
    _video(proj / "out" / "preview.mp4", "4217")
    _, lines = _check(proj, capsys)
    assert lines[-1] == "OK check" or lines[-1].startswith("WARN check")
    assert not any("digits" in line or "faces" in line for line in lines)


@pytest.mark.parametrize("bad", [
    "check: {digits: all}", "check: {faces: yes_please}", "check: {sample: 0}", "check: {sample: 99}",
    "check: {nope: 1}",
])
def test_validate_rejects_bad_guard_values(tmp_path: Path, bad: str) -> None:
    proj = _project(tmp_path, bad + "\n")
    with pytest.raises(SpecError):
        load_spec(proj / "montaj.yaml")


def test_fake_backends_warn_with_shot_and_frame(tmp_path: Path, capsys, monkeypatch) -> None:
    proj = _project(tmp_path, "check: {digits: none, faces: none, sample: 2}\n")
    _video(proj / "out" / "preview.mp4", None)
    monkeypatch.setattr(guards, "tesseract_ocr", lambda img: ["42"] if "shot0-f7" in img.name else [])
    monkeypatch.setattr(guards, "opencv_faces", lambda: (lambda img: 1 if "shot1-f" in img.name else 0))
    monkeypatch.setattr(guards.shutil, "which", lambda name: "/usr/bin/" + name)
    code, lines = _check(proj, capsys)
    assert code == 0
    assert 'WARN digits shot=0 frame=7 "42"' in lines
    assert "WARN faces shot=1 frame=37 (1)" in lines and "WARN faces shot=1 frame=52 (1)" in lines
    assert lines[-1] == "WARN check: 3 warnings"


def test_clean_video_is_ok(tmp_path: Path, capsys, monkeypatch) -> None:
    proj = _project(tmp_path, "check: {digits: none, faces: none}\n")
    _video(proj / "out" / "preview.mp4", None)
    monkeypatch.setattr(guards, "tesseract_ocr", lambda img: [])
    monkeypatch.setattr(guards, "opencv_faces", lambda: (lambda img: 0))
    monkeypatch.setattr(guards.shutil, "which", lambda name: "/usr/bin/" + name)
    _, lines = _check(proj, capsys)
    assert not [line for line in lines if line.startswith(("WARN digits", "WARN faces", "ERR"))]


def test_digits_spec_mode_allows_spec_text(tmp_path: Path) -> None:
    proj = _project(
        tmp_path,
        "check: {digits: spec}\ntext:\n  - from: 0s\n    to: 1s\n    lines:\n"
        "      - {text: 'Since 1999', y: 100, at: 0s}\n",
    )
    from montaj.timeline import resolve
    tl = resolve(load_spec(proj / "montaj.yaml"), proj)
    assert guards.spec_digits(tl) == {"1999"}
    _video(proj / "out" / "preview.mp4", None)
    out = guards.content_guards(tl, proj / "out" / "preview.mp4", proj / "build" / "g",
                                ocr=lambda img: ["1999", "7"])
    assert out and all('"7"' in line for line in out)


def test_missing_backends_are_err_lines(tmp_path: Path, capsys, monkeypatch) -> None:
    proj = _project(tmp_path, "check: {digits: none, faces: none}\n")
    _video(proj / "out" / "preview.mp4", None)
    monkeypatch.setattr(guards.shutil, "which", lambda name: None)
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)
    code, lines = _check(proj, capsys)
    assert code == 1
    assert any(line.startswith("ERR digits: install tesseract") for line in lines)
    assert any(line.startswith("ERR faces: install OpenCV") for line in lines)
    assert lines[-1].startswith("ERR check:")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs tesseract")
def test_real_tesseract_finds_digits_on_screen(tmp_path: Path, capsys) -> None:
    proj = _project(tmp_path, "check: {digits: none, sample: 1}\n")
    _video(proj / "out" / "preview.mp4", "4217")
    _, lines = _check(proj, capsys)
    assert any(line.startswith("WARN digits shot=0 frame=15") and "4217" in line for line in lines), lines
    assert not any(line.startswith("WARN digits shot=1") for line in lines)
