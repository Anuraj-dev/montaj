"""Issue #65: commands are quiet. Native libav stderr goes to build/*.log, designed title text escapes the
full-stop lint, a stale `check.expect` is a WARN naming its spec path, one mistake prints one ERR."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from montaj import caption_check as K
from montaj import cli
from montaj.spec import SpecError, load_spec, text_warnings
from montaj.timeline import resolve

from tests.test_i57_caption_findings import LANDSCAPE
from tests.test_spec import _write

UDTA = "[mov,mp4,m4a,3gp,3g2,mj2 @ 0x1033a340] UDTA parsing failed retrying raw"
TAKE_A = Path.home() / "Videos" / "auren-launch" / "raw" / "takeA.mp4"


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


@pytest.mark.gpu
@pytest.mark.skipif(not TAKE_A.is_file(), reason="needs the Auren take with a udta atom")
def test_render_on_a_udta_clip_prints_result_lines_only(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    os.symlink(TAKE_A, tmp_path / "assets" / "take.mp4")
    (tmp_path / "montaj.yaml").write_text(
        "video: {size: 640x360, fps: 30, look: clean}\nassets: assets\nshots:\n  - {clip: take, hold: 1s}\n")
    for cmd in (["render", "--preview"], ["check"], ["inspect", "--at", "0.5"]):
        proc = _montaj(tmp_path, *cmd)
        text = proc.stdout + proc.stderr
        assert "UDTA" not in text and "@ 0x" not in text, (cmd, text)
        assert proc.stdout.strip().splitlines()[-1].split(" ", 1)[0] in {"OK", "WARN"}, (cmd, proc.stdout)


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
