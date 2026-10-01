"""CLI text, JSON, -C and exit codes. CPU only; no render."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from montaj.cli import main


def _jpeg(folder: Path, stem: str, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(folder / f"{stem}.jpg")


def _project(tmp: Path, shots: str, stems: list[str]) -> Path:
    proj = tmp / "proj"
    for stem in stems:
        _jpeg(proj / "assets", stem)
    (proj / "montaj.yaml").write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  bpm: 120\n  look: warm-film\n"
        f"assets: assets\nshots:\n{shots}"
    )
    return proj


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _clip(path: Path, *, black: bool) -> None:
    src = "color=c=black:size=320x240:rate=25:duration=1" if black else "testsrc=size=320x240:rate=25:duration=1"
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", src, "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def test_validate_ok_err_json_and_chdir(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    proj = _project(tmp_path, "  - {photo: a, hold: 4b}\n  - {photo: b, hold: 15f}\n", ["a", "b"])

    code, out, err = _run(["-C", str(proj), "validate"], capsys)
    assert code == 0 and err == ""
    assert out.strip() == "OK montaj.yaml 75f 2.50s 2 shots 2 segments"

    code, out, err = _run(["validate", "-C", str(proj), "--json"], capsys)
    assert code == 0 and err == ""
    payload = json.loads(out)
    assert set(payload) == {"status", "lines"}
    assert payload == {"status": "OK", "lines": ["OK montaj.yaml 75f 2.50s 2 shots 2 segments"]}

    monkeypatch.chdir(tmp_path)
    code, out, err = _run(["-C", "proj", "validate"], capsys)
    assert code == 0 and out.strip() == "OK montaj.yaml 75f 2.50s 2 shots 2 segments"

    (proj / "montaj.yaml").write_text(
        "video:\n  size: 1080x1920\n  fps: 30\n  look: warm-film\n"
        "assets: assets\nshots:\n  - {photo: a, hold: 30}\n  - {photo: missing, hold: 10f}\n"
    )
    code, out, err = _run(["-C", str(proj), "validate"], capsys)
    assert code == 1 and err == ""
    lines = out.splitlines()
    assert lines and all(line.startswith("ERR ") for line in lines)
    assert any("shots[0].hold" in line and "22f, 4b or 1.5s" in line for line in lines)
    assert any("missing" in line for line in lines)

    code, out, err = _run(["--json", "-C", str(proj), "validate"], capsys)
    assert code == 1 and err == ""
    payload = json.loads(out)
    assert payload == {"status": "ERR", "lines": lines}


def test_missing_spec_and_bad_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(["-C", str(tmp_path), "validate"], capsys)
    assert code == 1 and out.startswith("ERR ") and err == ""

    code, out, err = _run(["--json", "-C"], capsys)
    assert code == 1 and err == ""
    payload = json.loads(out)
    assert payload["status"] == "ERR"
    assert "needs a directory" in payload["lines"][0]


def test_new_and_ingest(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    photos = tmp_path / "photos"
    _jpeg(photos, "16", (1, 2, 3))
    _jpeg(photos, "09", (4, 5, 6))
    dest = tmp_path / "film"

    code, out, err = _run(
        ["new", str(dest), "--recipe", "birthday-short", "--photos", "photos"], capsys,
    )
    assert code == 0 and err == "", out
    shown = dest.resolve().as_posix()
    assert out.strip() == f"OK {shown}/montaj.yaml sheet={shown}/build/ingest-sheet.jpg"
    text = (dest / "montaj.yaml").read_text()
    assert "warm-film" in text and "birthday" in text.lower()
    assert (dest / "assets" / "16.jpg").is_file()
    assert (dest / "assets" / "09.jpg").is_file()
    assert (dest / "build" / "ingest-sheet.jpg").is_file()

    code, out, err = _run(["new", "nope", "--recipe", "no-such-recipe", "--photos", "photos"], capsys)
    assert code == 1 and "no-such-recipe" in out and err == ""

    more = tmp_path / "more"
    _jpeg(more, "10", (7, 8, 9))
    code, out, err = _run(["-C", str(dest), "ingest", "--photos", str(more)], capsys)
    assert code == 0 and err == "", out
    assert out.strip() == "OK 1 photos sheet=build/ingest-sheet.jpg"
    assert (dest / "assets" / "10.jpg").is_file()
    assert (dest / "build" / "ingest-sheet.jpg").is_file()


def test_sheet_check_export_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    proj = tmp_path / "proj"
    preview = proj / "out" / "preview.mp4"
    final = proj / "out" / "final.mp4"
    _clip(preview, black=False)
    _clip(final, black=True)
    os.utime(preview, (1_000, 1_000))
    os.utime(final, (2_000, 2_000))

    code, out, err = _run(["-C", str(proj), "sheet", "--at", "0.2,0.6"], capsys)
    assert code == 0 and err == "", out
    assert out.strip() == "OK build/sheet.jpg 2 frames"
    assert (proj / "build" / "sheet.jpg").is_file()
    assert not (elsewhere / "build").exists()

    code, out, err = _run(["-C", str(proj), "check"], capsys)
    assert code == 0 and err == "", out
    body, summary = out.splitlines()[:-1], out.splitlines()[-1]
    assert any(line.startswith("WARN black") for line in body)
    assert summary.startswith("WARN check:")

    code, out, err = _run(["-C", str(proj), "--json", "check", "--video", "out/preview.mp4"], capsys)
    assert code == 0 and err == ""
    payload = json.loads(out)
    assert payload["status"] in {"OK", "WARN"}
    assert payload["lines"][-1].startswith(("OK check", "WARN check"))

    code, out, err = _run(["-C", str(proj), "export", "--target", "master"], capsys)
    assert code == 0 and err == "", out
    assert out.startswith("OK out/master.mp4 ") and out.strip().endswith("MB")
    assert (proj / "out" / "master.mp4").is_file()

    code, out, err = _run(["-C", str(proj), "export", "--target", "whatsapp"], capsys)
    assert code == 0 and err == "", out
    assert (proj / "out" / "whatsapp.mp4").is_file()

    empty = tmp_path / "empty"
    empty.mkdir()
    code, out, err = _run(["-C", str(empty), "check"], capsys)
    assert code == 1 and out.startswith("ERR ") and err == ""


def test_doctor_summary_matches_exit(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(["doctor", "--json"], capsys)
    assert err == ""
    payload = json.loads(out)
    lines = payload["lines"]
    failed = sum(line.startswith("FAIL") for line in lines)
    if failed:
        assert code == 1
        assert lines[-1] == f"ERR doctor: {failed} failed"
        assert payload["status"] == "ERR"
    else:
        assert code == 0
        assert lines[-1] == "OK doctor"
        assert payload["status"] == "OK"
    assert any(line.startswith("TRAP ") for line in lines)


def test_new_refuses_existing_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    photos = tmp_path / "photos"
    _jpeg(photos, "16")
    dest = tmp_path / "film"
    dest.mkdir()
    spec = dest / "montaj.yaml"
    spec.write_text("MARKER\n")

    code, out, err = _run(
        ["new", str(dest), "--recipe", "birthday-short", "--photos", "photos"], capsys,
    )
    assert code == 1 and err == ""
    line = out.strip()
    assert line.startswith("ERR new:") and f"{spec.resolve().as_posix()} exists" in line
    assert spec.read_text() == "MARKER\n"
    assert not (dest / "assets").exists()
    assert not (dest / "build").exists()


def test_argument_error_is_err_on_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(["validate", "--bad"], capsys)
    assert code == 1 and err == ""
    assert out.splitlines() == [out.strip()]
    assert out.strip().startswith("ERR validate:") and "--bad" in out

    code, out, err = _run(["validate", "--json", "--bad"], capsys)
    assert code == 1 and err == ""
    payload = json.loads(out)
    assert payload["status"] == "ERR"
    assert len(payload["lines"]) == 1
    assert payload["lines"][0].startswith("ERR validate:") and "--bad" in payload["lines"][0]


def test_expected_failures_are_one_err_line(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "montaj.yaml").write_text("assets: [broken\n")

    code, out, err = _run(["-C", str(proj), "ingest", "--photos", str(tmp_path)], capsys)
    assert code == 1 and err == "" and "Traceback" not in out
    assert out.splitlines() == [out.strip()]
    assert out.strip().startswith("ERR ingest:")
    ingest_log = (proj / "build" / "ingest.log").read_text()
    assert "\n" in ingest_log.strip()
    assert " ".join(ingest_log.split()) in out

    def boom(*_a, **_k):
        raise RuntimeError("cuda exploded\nno memory")

    monkeypatch.setattr("montaj.pipeline.render", boom)
    code, out, err = _run(["-C", str(proj), "render", "--preview"], capsys)
    assert code == 1 and err == "" and "Traceback" not in out
    assert out.splitlines() == [out.strip()]
    assert out.strip() == "ERR render: cuda exploded no memory"
    render_log = (proj / "build" / "render.log").read_text()
    assert "cuda exploded\nno memory" in render_log

    def deny(*_a, **_k):
        raise PermissionError("denied\nread")

    monkeypatch.setattr("montaj.cli.load_spec", deny)
    code, out, err = _run(["-C", str(proj), "--json", "validate"], capsys)
    assert code == 1 and err == ""
    payload = json.loads(out)
    assert payload["status"] == "ERR" and payload["lines"] == ["ERR validate: denied read"]
    validate_log = (proj / "build" / "validate.log").read_text()
    assert "denied\nread" in validate_log


def test_sheet_at_caps_count_at_twelve(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    proj = tmp_path / "proj"
    _clip(proj / "out" / "preview.mp4", black=False)
    times = ",".join(f"{i * 0.04:.2f}" for i in range(13))
    code, out, err = _run(["-C", str(proj), "sheet", "--at", times], capsys)
    assert code == 0 and err == "", out
    assert out.strip() == "OK build/sheet.jpg 12 frames"
