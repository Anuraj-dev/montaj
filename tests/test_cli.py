"""CLI text, JSON, -C and exit codes. CPU only; no render."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from shutil import copyfile

import pytest
from PIL import Image

from montaj.cli import _intended_freeze, main


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


def _solid(path: Path, seconds: float, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=red:size=320x240:rate={fps}:duration={seconds}",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def test_check_ignores_still_holds_but_warns_on_transition_freeze(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Still hold is intended freeze; a freeze inside a fade or a moving wall is not."""
    monkeypatch.chdir(tmp_path)
    wall = (
        "      grid: 100x100\n"
        "      seed: 1\n"
        "      jitter: 0\n"
        "      rot: 0\n"
        "      print_height: 10\n"
        "      prints:\n"
        "        - {photo: a, cell: [0, 0], straight: true}\n"
        "      light: {center: [0, 0], radius: [1, 1]}\n"
        "      camera:\n"
        "        pos:\n"
        "          - {t: 0f, print: a, at: [0, 0]}\n"
        "          - {t: 60f, print: a, at: [80, 80]}\n"
        "        zoom:\n"
        "          - {t: 0f, value: 1}\n"
        "          - {t: 60f, value: 2}\n"
    )
    proj = _project(
        tmp_path,
        "  - {photo: a, hold: 60f}\n"
        "  - {photo: b, hold: 60f, in: {type: fade, dur: 30f}}\n"
        "  - hold: 60f\n"
        "    wall:\n" + wall,
        ["a", "b"],
    )
    video = proj / "out" / "preview.mp4"
    _solid(video, 6)

    code, out, err = _run(["-C", str(proj), "check"], capsys)
    assert code == 0 and err == "", out
    frozen = [line for line in out.splitlines() if line.startswith("WARN frozen")]
    assert frozen, out
    spans = [tuple(float(v) for v in re.search(r"([\d.]+)-([\d.]+)s", line).groups()) for line in frozen]

    def overlaps(span: tuple[float, float], lo: float, hi: float) -> bool:
        return max(span[0], lo) < min(span[1], hi)

    # stills: [0, 2)s and [3, 4)s — those holds must not WARN
    assert not any(overlaps(s, 0.0, 1.9) or overlaps(s, 3.1, 3.9) for s in spans), frozen
    # fade [2, 3)s and moving wall [4, 6)s — a freeze there is a defect
    assert any(overlaps(s, 2.0, 3.0) for s in spans), frozen
    assert any(overlaps(s, 4.0, 6.0) for s in spans), frozen
    assert out.splitlines()[-1].startswith("WARN check:")


def test_check_ignores_wall_camera_holds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wall freeze is intended while both camera tracks hold; a freeze while they move is not."""
    monkeypatch.chdir(tmp_path)
    wall = (
        "      grid: 100x100\n"
        "      seed: 1\n"
        "      jitter: 0\n"
        "      rot: 0\n"
        "      print_height: 10\n"
        "      prints:\n"
        "        - {photo: a, cell: [0, 0], straight: true}\n"
        "      light: {center: [0, 0], radius: [1, 1]}\n"
        "      camera:\n"
        "        pos:\n"
        "          - {t: 0f, print: a, at: [1, 1]}\n"
        "          - {t: 30f}\n"
        "          - {t: 60f, print: a, at: [50, 50]}\n"
        "        zoom:\n"
        "          - {t: 0f, value: 1}\n"
        "          - {t: 30f}\n"
        "          - {t: 60f, value: 2}\n"
    )
    proj = _project(
        tmp_path,
        "  - {photo: a, hold: 30f}\n"
        "  - hold: 90f\n"
        "    wall:\n" + wall,
        ["a"],
    )
    _solid(proj / "out" / "preview.mp4", 4)

    intended = _intended_freeze(proj)
    def covers(lo: float, hi: float) -> bool:
        return any(a - 1e-6 <= lo and b + 1e-6 >= hi for a, b in intended)
    # still [0, 1); wall hold [1, 2]; moving (2, 3); wall hold after last key [3, 4)
    assert covers(0.0, 1.0) and covers(1.0, 2.0) and covers(3.0, 4.0)
    assert not any(max(a, 2.05) < min(b, 2.95) for a, b in intended)

    code, out, err = _run(["-C", str(proj), "check"], capsys)
    assert code == 0 and err == "", out
    frozen = [line for line in out.splitlines() if line.startswith("WARN frozen")]
    assert frozen, out
    spans = [tuple(float(v) for v in re.search(r"([\d.]+)-([\d.]+)s", line).groups()) for line in frozen]

    def overlaps(span: tuple[float, float], lo: float, hi: float) -> bool:
        return max(span[0], lo) < min(span[1], hi)

    assert not any(overlaps(s, 0.0, 1.9) or overlaps(s, 3.1, 3.9) for s in spans), frozen
    assert any(overlaps(s, 2.0, 3.0) for s in spans), frozen
    assert out.splitlines()[-1].startswith("WARN check:")


def test_intended_freeze_birthday_wall_holds(tmp_path: Path) -> None:
    """CLI intended spans cover the wall start hold (after the fade) and the tail after the last key."""
    stems = [
        "09", "10", "12", "16", "19", "20", "21", "22",
        "28", "29", "32", "33", "34", "35", "37", "38",
    ]
    copyfile(Path("recipes/birthday-short.yaml"), tmp_path / "montaj.yaml")
    for stem in stems:
        _jpeg(tmp_path / "assets", stem)
    spans = _intended_freeze(tmp_path)
    def covers(lo: float, hi: float) -> bool:
        return any(a - 1e-6 <= lo and b + 1e-6 >= hi for a, b in spans)
    # fade occupies [450, 465) = [15.0, 15.5); Wall hold then runs to local t=45 → 16.533s
    assert covers(15.5, 496 / 30)
    assert covers(740 / 30, 28.0)
    assert not any(max(a, 16.6) < min(b, 24.5) for a, b in spans)


def test_sheet_at_caps_count_at_twelve(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    proj = tmp_path / "proj"
    _clip(proj / "out" / "preview.mp4", black=False)
    times = ",".join(f"{i * 0.04:.2f}" for i in range(13))
    code, out, err = _run(["-C", str(proj), "sheet", "--at", times], capsys)
    assert code == 0 and err == "", out
    assert out.strip() == "OK build/sheet.jpg 12 frames"


def test_music_group_names_itself_in_errors_and_has_help(capsys: pytest.CaptureFixture[str]) -> None:
    """`montaj music …` failures must say `music`, not `cli`: the agent acts on that word."""
    code, out, err = _run(["music", "gen", "--caption", "x"], capsys)
    assert code == 1 and err == "" and len(out.splitlines()) == 1
    assert out.startswith("ERR music: ") and "--lyrics" in out

    code, out, err = _run(["--json", "music", "analyze", "song.wav", "--bpm", "fast"], capsys)
    assert code == 1 and err == ""
    payload = json.loads(out)
    assert payload["status"] == "ERR" and payload["lines"][0].startswith("ERR music:")

    assert main(["music", "--help"]) == 0
    usage = capsys.readouterr().out
    assert "gen" in usage and "analyze" in usage
