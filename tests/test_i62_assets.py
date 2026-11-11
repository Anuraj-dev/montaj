"""#62 item 3: `montaj assets [STEM]` summarises each clip from what ingest wrote."""
from __future__ import annotations

import json
from pathlib import Path

from montaj.analysis import ANALYSIS_VERSION
from montaj.cli import main
from tests.test_linked_audio import make_clip

EVENTS = [
    {"t": 0.0, "kind": "state", "state": "idle", "detail": "", "tool": None},
    {"t": 0.5, "kind": "state", "state": "idle", "detail": "", "tool": None},  # a repeat: folded
    {"t": 1.25, "kind": "ask", "text": "Open Spotify and play something upbeat for the whole team today."},
    {"t": 2.0, "kind": "state", "state": "speaking", "detail": "On it.", "tool": None},
    {"t": 3.5, "kind": "key", "key": "super+4"},
]
WORDS = {"words": [{"i": 1, "w": "okay", "s": 1.0, "e": 1.2}, {"i": 2, "w": "fresh", "s": 1.3, "e": 1.5},
                   {"i": 3, "w": "start", "s": 1.6, "e": 1.9}, {"i": 4, "w": "now", "s": 5.1, "e": 5.3}]}


def _row(path: Path, **kw) -> dict:
    st = path.stat()
    return {"v": ANALYSIS_VERSION, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "kind": "clip",
            "duration": 8.0, "fps": 30.0, "w": 64, "h": 64, **kw}


def _project(tmp: Path) -> Path:
    assets = tmp / "assets"
    talk, mute = make_clip(assets / "talk.mov"), make_clip(assets / "mute.mov", audio=False)
    (assets / "talk.events.json").write_text(json.dumps(EVENTS))
    (assets / "talk.words.json").write_text(json.dumps(WORDS))
    rows = {"talk.mov": _row(talk, cuts=[4.0], speech=[{"s": 1.0, "e": 2.0}, {"s": 5.0, "e": 6.0}]),
            "mute.mov": _row(mute, cuts=[])}
    (assets / "analysis.json").write_text(json.dumps(rows))
    (tmp / "montaj.yaml").write_text("video: {size: 64x64, fps: 30}\nassets: assets\nshots: []\n")
    return tmp


def test_assets_summary_lines(tmp_path: Path, capsys) -> None:
    root = _project(tmp_path)
    assert main(["-C", str(root), "assets"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "OK assets: 2 clips"
    mute, talk, speech, events = out[:4]
    assert mute.startswith("mute 8s 64x64 30fps cuts none | no audio")
    assert talk.startswith("talk 8s 64x64 30fps cuts 4.0 | loud ") and "dBFS" in talk
    assert speech == '  speech 1.0-2.0 "okay fresh start…" | 5.0-6.0 "now…"'
    parts = events.removeprefix("  events ").split(" | ")
    assert parts[0] == "0.00 state state=idle"  # empty and null fields dropped
    assert parts[1].startswith('1.25 ask text="Open Spotify') and parts[1].endswith('…"')
    assert parts[2:] == ['2.00 state state=speaking detail="On it."', "3.50 key key=super+4"]  # repeat folded


def test_assets_stem_and_errors(tmp_path: Path, capsys) -> None:
    root = _project(tmp_path)
    assert main(["-C", str(root), "assets", "mute"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2 and out[-1] == "OK assets: 1 clip"
    assert main(["-C", str(root), "assets", "nope"]) == 1
    assert capsys.readouterr().out.strip().startswith("ERR assets: no clip 'nope'")


def test_caps_without_a_stem(tmp_path: Path, capsys) -> None:
    root = _project(tmp_path)
    many = [{"t": float(i), "kind": "key", "key": f"k{i}"} for i in range(20)]
    (root / "assets" / "talk.events.json").write_text(json.dumps(many))
    main(["-C", str(root), "assets"])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("  events")][0]
    assert line.endswith("+8 more (montaj assets talk)")
    main(["-C", str(root), "assets", "talk"])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("  events")][0]
    assert line.count(" | ") == 19
