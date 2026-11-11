"""#62 review fixes: `sfx: [text, cuts]` with emphasis zoom, which cuts are hard, and the placed-sound window."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from montaj.audio.mix import SR
from montaj.emphasis import CLOSE_S, sfx_peaks
from tests.test_audio_mix import write_wav
from tests.test_emphasis import _plan, _proj
from tests.test_linked_audio import film

SLAM = [{"from": "1s", "to": "3.5s", "lines": [
    {"text": "GO", "at": "1s", "size": 30, "pos": [540, 540], "in": "slam"},
    {"text": "NOW", "at": "2.5s", "size": 30, "pos": [540, 700], "in": "slam"}]}]


def _with(p: Path, **top) -> Path:
    data = yaml.safe_load(p.read_text())
    for k, v in top.items():
        if k == "audio":
            data["audio"].update(v)
        else:
            data[k] = v
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def _names(p: Path) -> list[tuple[float, str]]:
    return [(round(t, 2), n) for t, n in sfx_peaks(_plan(p)[1])]


def test_emphasis_zoom_with_text_and_cuts(tmp_path: Path) -> None:
    # Word 2 lands at 1.2 s, the first slam's impact (1 s + 0.2 s): its pop already covers it, so no hit there.
    shots = [{"clip": "talk", "range": ["0s", "4s"]}, {"clip": "talk", "range": ["1s", "3s"]}]
    p = _with(_proj(tmp_path, [{"word": 2, "zoom": 1.3, "sfx": "pop"}], shots=shots),
              text=SLAM, audio={"sfx": ["text", "cuts"]})
    tl, plan = _plan(p)
    assert any(k.zoom == 1.3 for k in tl.spec.shots[0].camera)
    got = [n for _, n in sfx_peaks(plan)]
    assert got == ["pop", "hit", "swish"]  # the cut back to 1 s is a hard cut at 4 s
    t = dict((n, s) for s, n in sfx_peaks(plan))
    assert abs(t["hit"] - 2.7) <= CLOSE_S and abs(t["swish"] - 4.0) <= CLOSE_S


def test_cuts_skip_jump_cuts_and_cuts_to_black(tmp_path: Path) -> None:
    shots = [{"clip": "talk", "range": ["0s", "2s"]},
             {"clip": "talk", "range": ["2.5s", "4s"]},  # the take carries on: a jump cut, not a hard cut
             {"clip": "talk", "range": ["0s", "1s"]},  # back in the take: a hard cut at 3.5 s
             {"card": {"color": "#000"}, "hold": "1s"},  # textless card: the end on black
             {"blank": True, "hold": "1s"}]
    p = _with(film(tmp_path, shots), audio={"sfx": ["cuts"]})
    assert _names(p) == [(3.5, "swish")]


def test_a_placed_sound_counts_only_its_loudest_moment_inside_in_out(tmp_path: Path) -> None:
    x = np.zeros(4 * SR)
    x[SR] = 0.9  # loud click at 1 s: outside the played window
    x[3 * SR] = 0.3  # quieter click at 3 s: the loudest moment that plays
    write_wav(tmp_path / "sfx2" / "two.wav", x)
    shots = [{"clip": "talk", "range": ["0s", "4s"]}, {"clip": "talk", "range": ["0s", "4s"]}]
    # Plays source 2-4 s from film 2 s: its heard peak is film 3 s, its start 2 s, the unplayed click film 1 s.
    clip = {"src": "sfx2/two.wav", "in": "2s", "out": "4s", "at": "2s", "gain": "-20dB"}
    edge = {"src": "sfx2/two.wav", "in": "2s", "out": "4s", "at": "4s", "gain": "-20dB"}  # starts on the cut
    p = _with(film(tmp_path, shots), audio={"sfx": ["cuts"], "clips": [clip, edge]})
    assert _names(p) == [(4.0, "swish")]  # a clip starting on the cut, peaking 1 s later, does not cover it
    clip = {**clip, "at": "3s"}  # now its heard peak lands on the 4 s cut
    p = _with(p, audio={"clips": [clip]})
    assert _names(p) == []
