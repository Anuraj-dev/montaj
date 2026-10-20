"""Event logs: normalisation, selectors, at_event -> clip_in, event: markers, ingest-rec. CPU only."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from montaj.cli import main
from montaj.events import (
    Hit, Miss, compile_at_event, compile_event_marker, match_event, normalise_events,
)
from montaj.project import segment_hash
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

LOG = [
    {"t": 1.0, "kind": "state", "state": "idle"},
    {"t": 3.5, "kind": "state", "state": "speaking", "text": "hello theme"},
    {"t": 6.0, "kind": "state", "state": "speaking", "text": "bye"},
    {"t": 7.0, "kind": "key", "key": "k"},
]


def _proj(tmp: Path, shots: str, events=LOG, extra: str = "") -> Path:
    assets = tmp / "assets"
    assets.mkdir(exist_ok=True)
    Image.new("RGB", (8, 8)).save(assets / "a.jpg")
    (assets / "take.mp4").write_bytes(b"x")
    if events is not None:
        (assets / "take.events.json").write_text(json.dumps(events))
    p = tmp / "montaj.yaml"
    p.write_text(
        "video:\n  size: 1080x1920\n  fps: 60\n  look: warm-film\nassets: assets\n"
        f"{extra}shots:\n{shots}"
    )
    return p


def _tl(tmp: Path, shots: str, **kw):
    p = _proj(tmp, shots, **kw)
    return resolve(load_spec(p), p.parent)


def _errs(tmp: Path, shots: str, **kw) -> str:
    p = _proj(tmp, shots, **kw)
    with pytest.raises(SpecError) as e:
        resolve(load_spec(p), p.parent)
    return str(e.value)


def test_normalise_subtracts_first_video_pts() -> None:
    out = normalise_events([{"t": 2.0, "kind": "k"}], 0.032)
    assert out == [{"t": pytest.approx(1.968), "kind": "k"}]
    assert LOG[0]["t"] == 1.0


def test_selector_nth_contains_and_miss() -> None:
    errs: list[str] = []
    sel = compile_at_event({"kind": "state", "state": "speaking", "nth": 2}, "x", errs)
    assert not errs and isinstance(m := match_event(LOG, sel), Hit) and m.event["t"] == 6.0
    sel = compile_at_event({"kind": "state", "contains": {"text": "theme"}}, "x", errs)
    assert match_event(LOG, sel).event["t"] == 3.5
    miss = match_event(LOG, compile_at_event({"kind": "state", "state": "done"}, "x", errs))
    assert isinstance(miss, Miss) and miss.hint == "have: idle, speaking"
    assert isinstance(match_event(LOG, compile_at_event({"state": "speaking", "nth": 3}, "x", errs)), Miss)
    assert compile_at_event({"nth": 0}, "x", errs) is None and errs


def test_marker_parse() -> None:
    errs: list[str] = []
    stem, sel = compile_event_marker("event:take/state=speaking#2", "m", errs)
    assert stem == "take" and sel.nth == 2 and sel.equals == (("state", "speaking"),)
    assert compile_event_marker("event:take", "m", errs) is None and errs


def test_at_event_sets_clip_in_with_pre(tmp_path: Path) -> None:
    tl = _tl(tmp_path, "  - {clip: take, hold: 30f, at_event: {state: speaking}, pre: 0.5s}\n")
    sh = tl.shots[0].spec
    assert sh.clip_in == "3s" and sh.at_event is None and sh.pre is None


def test_shift_moves_pick_by_exact_frames_and_nth_hashes(tmp_path: Path) -> None:
    def hashes(root: Path, log, nth: int):
        root.mkdir()
        tl = _tl(root, f"  - {{clip: take, hold: 30f, at_event: {{state: speaking, nth: {nth}}}}}\n  - {{photo: a, hold: 30f}}\n", events=log)
        shas = {"take": "s", "a": "s2"}
        return tl, [segment_hash(tl, s, shas, "e") for s in tl.segments]

    tl1, h1 = hashes(tmp_path / "one", LOG, 1)
    shifted = [{**e, "t": e["t"] + 1.0} for e in LOG]
    tl2, h2 = hashes(tmp_path / "two", shifted, 1)
    f = lambda tl: round(float(tl.shots[0].spec.clip_in[:-1]) * 60)
    assert f(tl2) - f(tl1) == 60
    assert h1[0] != h2[0] and h1[1] == h2[1]
    _, h3 = hashes(tmp_path / "three", LOG, 2)
    assert h3[0] != h1[0] and h3[1] == h1[1]


def test_miss_and_misuse_errors(tmp_path: Path) -> None:
    msg = _errs(tmp_path / "a", "  - {clip: take, hold: 30f, at_event: {state: done}}\n") if (tmp_path / "a").mkdir() is None else ""
    assert "shots[0]" in msg and "state=done" in msg and "have: idle, speaking" in msg
    (tmp_path / "b").mkdir()
    assert "mutually exclusive" in _errs(tmp_path / "b", "  - {clip: take, hold: 30f, clip_in: 1s, at_event: {kind: key}}\n")
    (tmp_path / "c").mkdir()
    assert "pre needs at_event" in _errs(tmp_path / "c", "  - {clip: take, hold: 30f, pre: 1s}\n")
    (tmp_path / "d").mkdir()
    assert "starts before the clip" in _errs(tmp_path / "d", "  - {clip: take, hold: 30f, at_event: {kind: key}, pre: 9s}\n")
    (tmp_path / "e").mkdir()
    assert "no events for 'take'" in _errs(tmp_path / "e", "  - {clip: take, hold: 30f, at_event: {kind: key}}\n", events=None)


def test_event_marker_places_a_cut(tmp_path: Path) -> None:
    tl = _tl(
        tmp_path,
        "  - {photo: a, until: hero}\n  - {photo: a, hold: 10f}\n",
        extra='markers:\n  hero: "event:take/state=speaking#2"\n',
    )
    assert tl.shots[0].end == 360


def test_ingest_rec_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    (tmp_path / "montaj.yaml").write_text("video:\n  size: 1080x1920\n  fps: 30\nshots: []\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30:duration=1",
         "-output_ts_offset", "0.5", "-c:v", "libx264", str(raw / "take.mkv")], check=True)
    (raw / "take.events.json").write_text(json.dumps(LOG))
    code = main(["-C", str(tmp_path), "ingest-rec", "raw/take.mkv", "--events", "raw/take.events.json"])
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("OK take ") and "4 events" in out
    rows = json.loads((tmp_path / "assets" / "take.events.json").read_text())
    assert (tmp_path / "assets" / "take.mkv").is_file() and len(rows) == 4
    assert rows[0]["t"] == pytest.approx(1.0 - 0.5, abs=0.05)
