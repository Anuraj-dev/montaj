"""#62 item 2: `audio.sfx: [auto, text, cuts]` — a hit on each slam impact and a swish on each hard cut."""
from __future__ import annotations

from pathlib import Path

import yaml

from montaj.audio import mix
from montaj.emphasis import CLOSE_S, sfx_peaks
from montaj.spec import _parse_beat0, load_spec
from montaj.timeline import resolve
from tests.test_spec import _write

FPS = 30
SHOTS = """  - {photo: a, hold: 2s}
  - {photo: a, hold: 2s, in: {type: whip, dur: 8f}}
  - {photo: a, hold: 2s}
  - {photo: a, hold: 2s}
"""
TEXT = """markers: {drop: 5s}
text:
  - {from: 1s, to: 1.9s, lines: [{text: TICK, at: 1s, y: 900, reveal: type}]}
  - {from: 2.5s, to: 3.5s, lines: [{text: SLAM, at: 2.5s, pos: [540, 900], style: caps, in: slam}]}
  - {from: 6.5s, to: 7.5s, lines: [{text: LATE, at: 6.5s, pos: [540, 900], style: caps, in: {type: slam, dur: 6f}}]}
"""
# `audio.plan_hash` of the fixture with `sfx: auto` on base 8cb50d9 (computed there with this exact spec).
AUTO_HASH = "d4cafbd6410af843"


def _spec(tmp: Path, audio: str) -> Path:
    return _write(tmp, SHOTS, ["a"], extra=TEXT + f"audio:\n{audio}")


def _plan(p: Path):
    tl = resolve(load_spec(p), p.parent)
    return mix.plan_audio(tl.spec, p.parent, tl.markers, FPS, tl.spec.video.bpm,
                          _parse_beat0(tl.spec.video.beat0, FPS, []) or 0.0, [], shots=tl.shots)


def test_auto_keeps_its_mix_hash(tmp_path: Path) -> None:
    plan = _plan(_spec(tmp_path, "  sfx: auto\n"))
    assert [n for _, n in sfx_peaks(plan)] == ["tick", "whoosh", "riser"]
    assert mix.plan_hash(plan) == AUTO_HASH


def test_list_auto_equals_auto(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = load_spec(_spec(tmp_path / "a", "  sfx: auto\n"))
    b = load_spec(_spec(tmp_path / "b", "  sfx: [auto]\n"))
    assert a.model_dump_json() == b.model_dump_json()


def test_text_hits_land_on_slam_impact_and_cuts_on_hard_cuts(tmp_path: Path) -> None:
    peaks = sfx_peaks(_plan(_spec(tmp_path, "  sfx: [text, cuts]\n")))
    want = [(2.5 + 0.2, "hit"), (4.0, "swish"), (6.0, "swish"), (6.5 + 6 / FPS, "hit")]
    assert [n for _, n in peaks] == [n for _, n in want]  # no swish at the whip (shot 1 has an `in`)
    for (t, _), (w, _) in zip(peaks, want):
        assert abs(t - w) <= 1 / FPS


def test_never_doubles_a_placed_sound(tmp_path: Path) -> None:
    # The author's own boom peaks on the 4 s cut; an emphasis-free `auto` riser sits on the 5 s drop.
    audio = "  sfx: [auto, text, cuts]\n  clips:\n    - {src: synth:boom, at: 4s}\n"
    p = _spec(tmp_path, audio)
    lead = mix.peak_lead("synth:boom")
    data = yaml.safe_load(p.read_text())
    data["audio"]["clips"][0]["offset"] = f"-{round(lead * 1000)}ms"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    plan = _plan(p)
    peaks = sfx_peaks(plan)
    names = [n for _, n in peaks]
    assert names.count("swish") == 1 and names.count("hit") == 2  # the 4 s cut already has the boom
    assert all(abs(t - 4.0) >= CLOSE_S for t, n in peaks if n == "swish")
    assert len(plan.clips) == len(peaks) + 1  # the boom stays as written
