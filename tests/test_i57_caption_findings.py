"""Issue #57: caption findings name the spec path, time span, text and fix; a reading window never bridges a
gap; `role: title` takes designed text out of the caption lints; `check --json` findings; message kinds in
`montaj spec --search`; an unmet `check.expect` says what was nearest."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from montaj import caption_check as K
from montaj import cli, qa, specref
from montaj.captions import window_cps
from montaj.encode import FFMPEG
from montaj.spec import SpecError, load_spec, text_warnings
from montaj.timeline import resolve

from tests.test_spec import _write

LANDSCAPE = "size: 1920x1080\n  fps: 60\n  bpm: 120\n  look: clean"


def _blocks(tmp: Path, rows: list[tuple[float, float, str, str]], extra: str = "") -> Path:
    """Slam text blocks like the Auren launch short: (from s, to s, text, extra line fields), mid-frame."""
    text = "".join(f"  - from: {a}s\n    to: {b}s\n    lines: [{{text: \"{t}\", size: 80, pos: [540, 500], "
                   f"at: {a}s, in: {{type: slam, by: word, stagger: 0.06s}}{more}}}]\n"
                   for a, b, t, more in rows)
    return _write(tmp, "  - {photo: a, hold: 6s}\n", ["a"], video=LANDSCAPE, extra=f"{extra}text:\n{text}")


AUREN = [(0, 1, "YOUR PC", ""), (1, 2, "PLAY *MUSIC*", ""), (2, 3, "*SPOTIFY* ON", ""),
         (3, 3.5, "*PAUSED*", ""), (3.5, 4, "SWITCH *WORKSPACES*", ""), (4, 5, "SWITCH MY *THEME*", "")]


def _rate(p: Path) -> list[str]:
    return [g for g in K.lines(resolve(load_spec(p), p.parent)) if "caption-rate" in g]


def test_rate_names_the_densest_block_by_spec_path_time_and_text(tmp_path: Path) -> None:
    got = _rate(_blocks(tmp_path, AUREN))
    assert len(got) == 1, got
    assert got[0].startswith("WARN caption-rate text[4] 3.5-4.0s 'SWITCH WORKSPACES' "), got
    assert "cps (max 20: hold longer or cut words; designed text, not a caption: role: title)" in got[0]


def test_window_never_bridges_a_gap() -> None:
    spans = [(20, 15), (20, 15)]  # two 0.5 s chunks at 30 fps
    assert window_cps(spans, 0, 30)[:2] == (0, 1)  # contiguous: one window
    assert window_cps(spans, 0, 30, {1})[:2] == (0, 0)  # the screen clears before span 1
    assert window_cps(spans, 1, 30, {1})[:2] == (1, 1)


def test_two_fast_chunks_three_seconds_apart_are_two_findings(tmp_path: Path) -> None:
    rows = [(0, 0.5, "SWITCH WORKSPACES", ""), (3.5, 4, "CHANGE WALLPAPERS", "")]
    got = _rate(_blocks(tmp_path, rows))
    assert [g.split(" ")[2] for g in got] == ["text[0]", "text[1]"], got


def test_a_slow_block_after_a_gap_does_not_hide_a_fast_one(tmp_path: Path) -> None:
    rows = [(0, 0.5, "SWITCH WORKSPACES", ""), (2, 6, "OK", "")]
    assert [g.split(" ")[2] for g in _rate(_blocks(tmp_path, rows))] == ["text[0]"]


def test_role_title_on_a_block_opts_out_of_every_caption_lint(tmp_path: Path) -> None:
    low = ("text:\n  - from: 0s\n    to: 0.5s\n    role: title\n    lines:\n"
           "      - {text: SWITCH WORKSPACES, style: sans, weight: 400, size: 20, y: 1900, at: 0s, box: chip}\n")
    p = _write(tmp_path, "  - {photo: a, hold: 3s}\n", ["a"], extra=low)
    spec = load_spec(p)
    assert spec.check is None
    assert K.lines(resolve(spec, tmp_path)) == []
    assert [w for w in text_warnings(spec) if "caption" in w] == []
    lint = load_spec(_write(tmp_path, "  - {photo: a, hold: 3s}\n", ["a"], extra=low.replace("    role: title\n", "")))
    assert K.lines(resolve(lint, tmp_path)) and [w for w in text_warnings(lint) if "caption" in w]


def test_role_title_in_a_style_opts_out(tmp_path: Path) -> None:
    rows = [(a, b, t, ", style: big") for a, b, t, _ in AUREN]
    p = _blocks(tmp_path, rows, extra="styles:\n  big: {role: title}\n")
    assert K.lines(resolve(load_spec(p), tmp_path)) == []


def test_role_caption_opts_a_high_block_in(tmp_path: Path) -> None:
    rows = [(0, 0.5, "SWITCH WORKSPACES", "")]
    p = _blocks(tmp_path, rows)
    high = p.read_text().replace("pos: [540, 500]", "pos: [540, 100]")
    p.write_text(high)
    assert _rate(p) == []
    p.write_text(high.replace("    lines:", "    role: caption\n    lines:"))
    got = _rate(p)
    assert got and "role: title" not in got[0]  # an explicit caption is not offered the title hint


def test_role_rejects_other_values(tmp_path: Path) -> None:
    p = _blocks(tmp_path, [(0, 1, "HI", "")])
    p.write_text(p.read_text().replace("    lines:", "    role: heading\n    lines:"))
    with pytest.raises(SpecError):
        load_spec(p)


def test_unset_role_leaves_the_block_dump_unchanged(tmp_path: Path) -> None:
    block = load_spec(_blocks(tmp_path, [(0, 1, "HI", "")])).text[0]
    assert "role" not in block.model_dump() and "role" not in block.lines[0].model_dump()


def test_check_json_has_structured_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    p = _blocks(tmp_path, [(0, 0.5, "SWITCH WORKSPACES", ""), (3.5, 4, "CHANGE WALLPAPERS", "")])
    (tmp_path / "out").mkdir()
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=60:d=6",
                    "-pix_fmt", "yuv420p", str(tmp_path / "out" / "final.mp4")], check=True)
    monkeypatch.setattr(K, "_face_boxes", lambda: (lambda image: []))
    cli.main(["--json", "-C", str(p.parent), "check"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["lines"] and "findings" in out
    rate = [f for f in out["findings"] if f["kind"] == "caption-rate"]
    assert [(f["path"], f["start"], f["end"], f["text"], f["level"], f["limit"]) for f in rate] == [
        ("text[0]", 0.0, 0.5, "SWITCH WORKSPACES", "WARN", 20),
        ("text[1]", 3.5, 4.0, "CHANGE WALLPAPERS", "WARN", 20)]
    assert all(f["value"] > 20 and f["message"] in out["lines"] for f in rate)
    assert all(set(f) >= {"kind", "level", "path", "start", "end", "text", "value", "limit"} for f in out["findings"])


def test_findings_keep_non_caption_warnings_and_unmet_expects() -> None:
    lines = ["OK video", "WARN black 1.00-1.20s", "INFO expect frozen 5–6s not found (no frozen frames anywhere)",
             "INFO frozen 1 span(s) ignored as intended", "WARN check: 1 warnings"]
    got = cli._findings(lines, [])
    assert [(f["kind"], f["level"], f["message"]) for f in got] == [("black", "WARN", lines[1]),
                                                                     ("expect", "INFO", lines[2])]


@pytest.mark.parametrize("kind", ["caption-rate", "caption-safe", "caption-box", "black", "frozen", "expect"])
def test_spec_search_explains_each_message_kind(kind: str) -> None:
    hits = [ln for ln in specref.topics()["messages"] if ln.startswith(f"- `{kind}`")]
    assert hits and "Fix" in hits[0]


def test_spec_search_caption_rate_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["spec", "--search", "caption-rate"]) == 0
    out = capsys.readouterr().out
    assert "- `caption-rate` — check WARN" in out and "role: title" in out and "check.expect" in out


def test_unmet_expect_names_the_short_black_it_missed(tmp_path: Path) -> None:
    video = tmp_path / "fade.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=white:s=320x240:r=30:d=2",
                    "-vf", "fade=t=out:st=1.5:d=0.43", "-pix_fmt", "yuv420p", str(video)], check=True)
    lines: list[str] = []
    detected = qa.black_spans(video)
    assert detected == []
    qa._note_expects(lines, "black", detected, [qa.Expect("black", (1.0, 1.5), None)],
                     lambda: qa.black_spans(video, min_s=0.0))
    assert len(lines) == 1 and lines[0].startswith("INFO expect black 1–1.5s not found (black 1.9"), lines
    assert "under the 0.10s minimum" in lines[0]


# --- review round: time order, `captions[N]`, chunker gaps, hash, expect nearest -------------------------


def _mixed(tmp: Path, caps: str, rows, authored: str) -> Path:
    from tests.test_captions import _words
    _words(tmp, rows)
    return _write(tmp, "  - {photo: a, hold: 7s}\n", ["a"], extra=f"captions: {caps}\n{authored}")


FAST_LOW = ("text:\n  - from: 4s\n    to: 4.5s\n    lines: [{text: SWITCH WORKSPACES, style: sans, weight: 700, "
            "size: 70, y: 1500, at: 4s}]\n")
SLOW = [("hello", 0.0, 0.5), ("there", 0.6, 1.2), ("friend", 1.3, 2.0)]


def test_authored_and_made_chunks_are_read_and_printed_in_time_order(tmp_path: Path) -> None:
    """The authored block sits first in `text:` but at 4 s, after the made captions: its window must not
    run on into them, and findings print in time order."""
    p = _mixed(tmp_path, "{source: talk.words.json, y: 1800}", SLOW, FAST_LOW)
    got = K.lines(resolve(load_spec(p), tmp_path))
    assert any(g.startswith("WARN caption-rate text[0] 4.0-4.5s 'SWITCH WORKSPACES'") for g in got), got
    starts = [float(g.split(" ")[3].split("-")[0]) for g in got if g.startswith("WARN")]
    assert starts == sorted(starts) and starts[0] == 0.0, got


@pytest.mark.parametrize("preset", ["punch", "karaoke"])
def test_captions_n_matches_print_numbering(tmp_path: Path, preset: str) -> None:
    """`yeah` at 1 s and again at 5 s are two chunks, and every later N still matches `--print`."""
    from montaj import captions as C
    rows = [("so", 0.0, 0.3), ("yeah", 1.0, 1.3), ("yeah", 5.0, 5.3), ("done", 6.0, 6.4)]
    p = _mixed(tmp_path, f"{{source: talk.words.json, preset: {preset}}}", rows, "")
    printed, fps = C.plan(p)
    tl = resolve(load_spec(p), tmp_path)
    made = [c for c in K.chunks(tl) if c.path.startswith("captions[")]
    assert len(made) == len(printed)
    for c in made:
        n = int(c.path[len("captions["):-1])
        assert (c.start, c.text.lower()) == (printed[n].start, " ".join(w.w for w in printed[n].words).lower())
        assert c.end - c.start < 2 * fps  # no chunk spans the 1 s -> 5 s gap


def test_chunker_split_changes_across_a_clearing_pause(tmp_path: Path) -> None:
    """Behaviour change (#57): the chunker's reading windows stop at a pause that clears the screen. Before,
    `go` (just ahead of a 2 s pause) was averaged with the fast words after it and joined `now`; read on its
    own side of the pause, `now now` / `go` is the cheaper split."""
    from montaj import captions as C
    from tests.test_captions import _words
    rows = [("now", 0.0, 0.2), ("now", 0.22, 0.52), ("go", 0.54, 0.69), ("okay", 2.71, 2.86),
            ("wonderful", 2.88, 3.0), ("tremendously", 3.02, 3.22), ("yes", 3.24, 3.44)]
    _words(tmp_path, rows)
    p = _write(tmp_path, "  - {photo: a, hold: 9s}\n", ["a"], extra="captions: {source: talk.words.json}\n")
    got = [" ".join(w.w for w in c.words) for c in C.plan(p)[0]]
    assert got == ["now now", "go", "okay", "wonderful", "tremendously", "yes"]  # gapless: "now", "now go", ...


def test_role_does_not_change_the_segment_hash(tmp_path: Path) -> None:
    from montaj.project import segment_hash
    base = _blocks(tmp_path, [(0, 1, "HI", "")]).read_text()

    def h(text: str) -> str:
        (tmp_path / "montaj.yaml").write_text(text)
        tl = resolve(load_spec(tmp_path / "montaj.yaml"), tmp_path)
        return segment_hash(tl, tl.segments[0], {"a": "x"}, "e1")

    plain = h(base)
    assert h(base.replace("    lines:", "    role: title\n    lines:")) == plain
    assert h(base.replace("size: 80,", "size: 80, role: caption,")) == plain
    assert h(base.replace("HI", "HO")) != plain  # the hash still sees the text


def test_unmet_caption_expect_names_the_nearest_finding(tmp_path: Path) -> None:
    p = _blocks(tmp_path, [(3.5, 4, "SWITCH WORKSPACES", "")], extra="check:\n  expect:\n    - {caption: [0s, 1s]}\n")
    got = K.lines(resolve(load_spec(p), tmp_path))
    assert got[-1] == ("INFO expect caption 0-1s not found (nearest finding caption-rate text[0] 3.5-4.0s "
                       "'SWITCH WORKSPACES')"), got
