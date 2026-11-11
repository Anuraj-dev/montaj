"""Issue #65: commands are quiet. Native libav stderr goes to build/*.log, designed title text escapes the
full-stop lint, a stale `check.expect` is a WARN naming its spec path, one mistake prints one ERR."""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from montaj import caption_check as K
from montaj import cli, qa
from montaj.spec import SpecError, load_spec, text_warnings
from montaj.timeline import resolve

from tests.test_i57_caption_findings import LANDSCAPE
from tests.test_spec import _write

UDTA = "[mov,mp4,m4a,3gp,3g2,mj2 @ 0x1033a340] UDTA parsing failed retrying raw"


# 1. libav stderr -------------------------------------------------------------------------------

def _montaj(project: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", "import sys; from montaj.cli import main; sys.exit(main())", *args],
                          cwd=project, capture_output=True, text=True, check=False, timeout=300)


def test_native_stderr_goes_to_the_log_and_python_stderr_stays(tmp_path: Path) -> None:
    """A C library writing to fd 2 mid-command reaches build/<cmd>.log, not the terminal; stdout stays clean."""
    (tmp_path / "montaj.yaml").write_text("video: {size: 320x240, fps: 30}\nshots: []\n")
    code = textwrap.dedent(f"""
        import os, sys
        from pathlib import Path
        from montaj import cli
        def noisy(ns, root, as_json):
            os.write(2, b"{UDTA}\\n")
            print("python-stderr-line", file=sys.stderr)
            return cli._emit(["OK noisy"], False)
        cli._run = noisy
        sys.exit(cli.main(["validate"]))
    """)
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert proc.stdout == "OK noisy\n", proc
    assert "UDTA" not in proc.stderr and proc.stderr == "python-stderr-line\n", proc.stderr
    assert UDTA in (tmp_path / "build" / "validate.log").read_text()


def _udta_clip(path: Path) -> None:
    """A tiny h264 mp4 whose video trak carries a `udta` with a bare `name` child: the shape of a screen
    recorder's file, which libav's mov demuxer reports as `UDTA parsing failed retrying raw`."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    buf = bytearray(path.read_bytes())

    def box(start: int, end: int, kind: bytes) -> tuple[int, int]:
        i = start
        while i < end:
            size, name = struct.unpack(">I4s", buf[i:i + 8])
            if name == kind:
                return i, size
            i += size
        raise AssertionError(kind)

    moov, msize = box(0, len(buf), b"moov")
    trak, tsize = box(moov + 8, moov + msize, b"trak")
    payload = b"Applications: pw-cat"
    child = struct.pack(">I4s", 8 + len(payload), b"name") + payload
    udta = struct.pack(">I4s", 8 + len(child), b"udta") + child
    buf[trak + tsize:trak + tsize] = udta
    struct.pack_into(">I", buf, trak, tsize + len(udta))
    struct.pack_into(">I", buf, moov, msize + len(udta))
    path.write_bytes(buf)


@pytest.mark.gpu
def test_render_on_a_udta_clip_prints_result_lines_only(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    _udta_clip(tmp_path / "assets" / "take.mp4")
    (tmp_path / "montaj.yaml").write_text(
        "video: {size: 640x360, fps: 30, look: clean}\nassets: assets\nshots:\n  - {clip: take, hold: 1s}\n")
    for cmd in (["render", "--preview"], ["check"], ["inspect", "--at", "0.5"]):
        proc = _montaj(tmp_path, *cmd)
        text = proc.stdout + proc.stderr
        assert "UDTA" not in text and "@ 0x" not in text, (cmd, text)
        assert proc.stdout.strip().splitlines()[-1].split(" ", 1)[0] in {"OK", "WARN"}, (cmd, proc.stdout)
    assert "UDTA parsing failed" in (tmp_path / "build" / "render.log").read_text()  # the fixture does trigger it


def test_a_native_crash_still_shows_on_the_terminal(tmp_path: Path) -> None:
    code = textwrap.dedent("""
        import faulthandler, sys
        from montaj import cli
        def crash(ns, root, as_json):
            faulthandler._sigsegv()
        cli._run = crash
        sys.exit(cli.main(["validate"]))
    """)
    for project in (True, False):
        d = tmp_path / str(project)
        d.mkdir()
        if project:
            (d / "montaj.yaml").write_text("video: {size: 320x240, fps: 30}\nshots: []\n")
        env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")}
        proc = subprocess.run([sys.executable, "-c", code], cwd=d, capture_output=True, text=True, check=False, env=env)
        assert "Segmentation fault" in proc.stderr, (project, proc.stderr)
    assert (tmp_path / "cache" / "montaj" / "logs").is_dir()  # outside a project the native log goes to the cache


# 2. full-stop lint follows the caption decision --------------------------------------------------

def _stops(tmp: Path, style_role: str, y: int, text: str) -> list[str]:
    extra = (f"styles:\n  slam:\n    size: 120\n    in: {{type: slam, by: word, stagger: 0.06s}}\n{style_role}"
             f"text:\n  - from: 0s\n    to: 2s\n    lines: [{{text: \"{text}\", style: slam, pos: [540, {y}], at: 0s}}]\n")
    tmp.mkdir()
    p = _write(tmp, "  - {photo: a, hold: 3s}\n", ["a"], video=LANDSCAPE, extra=extra)
    return [w for w in text_warnings(load_spec(p)) if "full stop" in w]


def test_full_stop_lint_skips_designed_titles_and_keeps_captions(tmp_path: Path) -> None:
    assert _stops(tmp_path / "a", "    role: title\n", 500, "PAUSED.") == []  # slam style marked a title
    assert _stops(tmp_path / "b", "", 200, "PAUSED.") == []  # mid-frame: not a caption by the check rule
    assert _stops(tmp_path / "c", "", 500, "Hello there.") == ["WARN text[0].lines[0] ends with a full stop"]


# 3. stale expect ---------------------------------------------------------------------------------

def test_stale_caption_expect_is_a_warn_with_path_and_the_spec_expression(tmp_path: Path) -> None:
    extra = ("markers: {mid: 2s}\ncheck:\n  expect:\n    - {black: [0s, 0.5s]}\n    - {caption: [mid, mid+1s]}\n"
             "text:\n  - from: 0s\n    to: 1s\n    lines: [{text: \"HI\", size: 80, pos: [540, 900], at: 0s}]\n")
    p = _write(tmp_path, "  - {photo: a, hold: 6s}\n", ["a"], video=LANDSCAPE, extra=extra)
    got = [g for g in K.lines(resolve(load_spec(p), tmp_path)) if "expect" in g]
    assert len(got) == 1 and got[0].startswith("WARN expect check.expect[1] caption [mid, mid+1s] (2-3s) not found ("), got
    assert "nearest" in got[0]


def test_a_stale_expect_counts_in_the_check_status() -> None:
    lines = ["OK duration 1s", "WARN expect check.expect[0] black [0s, 1s] (0–1s) not found (no black frames anywhere)"]
    assert cli._findings(lines + ["WARN check: 1 warnings"], [])[0]["level"] == "WARN"
    assert cli._status(lines + ["WARN check: 1 warnings"]) == "WARN"


# 4. one mistake, three lines, one ERR ------------------------------------------------------------

def test_a_bad_style_used_by_three_lines_prints_one_err(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = "".join(f"  - from: {i}s\n    to: {i + 1}s\n    lines: [{{text: \"WORD *{i}*\", style: bad, pos: [540, 300], "
                   f"at: {i}s}}]\n" for i in range(3))
    extra = "styles:\n  bad:\n    emphasis: {color: nope, scale: 9}\n    color: \"#12\"\ntext:\n" + rows
    p = _write(tmp_path, "  - {photo: a, hold: 4s}\n", ["a"], video=LANDSCAPE, extra=extra)
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert len(ei.value.errors) == 9  # the loader reports each style mistake once per line that uses it
    code = cli.main(["-C", str(tmp_path), "validate"])
    out = capsys.readouterr().out.splitlines()
    assert code == 1 and len(out) == 3 and all(ln.startswith("ERR styles.bad.") for ln in out), out


def test_every_missed_expect_kind_names_its_path_through_the_real_check(tmp_path: Path) -> None:
    from montaj.scopes import ShotScope, warnings as scope_warnings

    extra = "check:\n  expect:\n    - {black: [0s, 1s]}\n    - {frozen: [2s, 3s]}\n    - {luma: [4s, 5s]}\n"
    p = _write(tmp_path, "  - {photo: a, hold: 3s}\n  - {photo: a, hold: 3s}\n", ["a"], video=LANDSCAPE, extra=extra)
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=4",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    expects, _, _ = cli._check_overrides(tmp_path)  # the CLI's own resolution of check.expect
    lines = qa.check(video, log=tmp_path / "qa.log", expect=expects)
    tl = resolve(load_spec(p), tmp_path)
    rows = [ShotScope(i, p1=0.02, p50=0.4, p99=0.9, clip_hi=0.0, clip_lo=0.0, sat_mean=0.3, cast=0.0, tint=0.0)
            for i in range(len(tl.shots))]
    lines += scope_warnings(rows, tl, {})
    stale = [ln for ln in lines if ln.startswith("WARN expect")]
    assert [ln.split(" ")[2:5] for ln in stale] == [["check.expect[0]", "black", "[0s,"], ["check.expect[1]", "frozen", "[2s,"],
                                                    ["check.expect[2]", "luma", "[4s,"]], stale
    found = cli._findings(stale + ["WARN check: 3 warnings"], [])
    assert [f["path"] for f in found] == ["check.expect[0]", "check.expect[1]", "check.expect[2]"]
