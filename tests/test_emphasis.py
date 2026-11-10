"""`emphasis:` and `audio.sfx: auto` (issue #41): word-anchored edits expand into existing mechanisms."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from montaj.audio import mix
from montaj.cli import main
from montaj.emphasis import sfx_peaks
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve
from tests.test_audio_mix import write_wav
from tests.test_linked_audio import FPS, film

WORDS = [(f"w{k}" if k != 4 else "never", 0.5 + 0.7 * k) for k in range(10)]
LINES = [{"word": k + 1, "zoom": 1.3, "caption": "accent", "sfx": "pop", **({"flash": 0.4} if k % 2 else {})}
         for k in range(10)]


def _proj(tmp: Path, emphasis=None, audio=None, shots=None, caps: bool = True) -> Path:
    words = [{"i": k + 1, "w": w, "s": s, "e": s + 0.3} for k, (w, s) in enumerate(WORDS)]
    (tmp / "talk.words.json").write_text(json.dumps({"words": words}))
    p = film(tmp, shots or [{"clip": "talk", "range": ["0s", "8s"]}], audio)
    data = yaml.safe_load(p.read_text())
    if caps:
        data["captions"] = {"source": "talk.words.json"}
    if emphasis is not None:
        data["emphasis"] = emphasis
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def _plan(p: Path):
    tl = resolve(load_spec(p), p.parent)
    return tl, mix.plan_audio(tl.spec, p.parent, tl.markers, FPS, None, 0.0, [], shots=tl.shots)


def test_ten_emphasis_lines_land_within_a_frame_of_their_words(tmp_path: Path) -> None:
    p = _proj(tmp_path, LINES)
    spec = load_spec(p)
    assert spec.emphasis is None
    keys = spec.shots[0].camera
    peaks = [k for k in keys if k.zoom == 1.3]
    assert len(peaks) == 10
    for (_, s), k in zip(WORDS, peaks):
        assert abs(int(k.t[:-1]) - s * FPS) <= 1  # zoom peak
    from montaj.emphasis import expand_emphasis
    from montaj.spec import Spec

    pre = expand_emphasis(Spec.model_validate(yaml.safe_load(p.read_text())), p.parent, [])
    assert [o.word for o in pre.captions.overrides if o.emphasis] == list(range(1, 11))
    tl, plan = _plan(p)
    for (_, s), (t, name) in zip(WORDS, sfx_peaks(plan)):
        assert name == "pop" and abs(t - s) <= 1 / FPS  # sound peak
    flashes = [f for f in tl.spec.fx if f.flash]
    assert len(flashes) == 5
    assert {f.at for f in flashes} == {f"em{k}" for k in (2, 4, 6, 8, 10)}  # anchored to the word markers


def test_match_every_and_shake(tmp_path: Path) -> None:
    p = _proj(tmp_path, [{"match": "never", "shake": 30}, {"on": "every 2s", "flash": 0.3}])
    spec = load_spec(p)
    sh = spec.shots[0].effects[0]
    assert sh.type == "shake" and sh.at == "em5"
    assert len([f for f in spec.fx if f.flash]) == 4  # 0, 2, 4, 6 s of 8 s


def test_errors_are_actionable(tmp_path: Path) -> None:
    for bad, frag in (({"word": 99, "zoom": 1.2}, "not in the source"), ({"match": "nope", "flash": 1}, "montaj words"),
                      ({"word": 1}, "no action"), ({"word": 1, "match": "w1", "flash": 1}, "exactly one")):
        with pytest.raises(SpecError) as ei:
            load_spec(_proj(tmp_path, [bad]))
        assert any(frag in e for e in ei.value.errors), ei.value.errors


def test_emphasis_is_hashed_through_its_expansion(tmp_path: Path) -> None:
    a = load_spec(_proj(tmp_path, [{"word": 2, "shake": 30}]))
    b = load_spec(_proj(tmp_path, [{"word": 2, "shake": 40}]))
    assert a.model_dump_json() != b.model_dump_json()
    assert a.emphasis is None and a.shots[0].effects[0].amount == 30


def test_auto_sfx_whoosh_on_whip_with_peak_on_the_cut(tmp_path: Path) -> None:
    p = _proj(tmp_path, audio={"sfx": "auto"}, caps=False, shots=[
        {"clip": "talk", "range": ["0s", "2s"]},
        {"clip": "talk", "range": ["2s", "4s"], "in": {"type": "whip", "dur": "8f"}},
        {"clip": "talk", "range": ["4s", "6s"]}])
    tl, plan = _plan(p)
    peaks = sfx_peaks(plan)
    assert [n for _, n in peaks] == ["whoosh"]
    assert abs(peaks[0][0] - 2.0) <= 1 / FPS


def test_project_sfx_folder_overrides_builtin_and_peak_follows(tmp_path: Path) -> None:
    write_wav(tmp_path / "sfx" / "pop.wav", np.concatenate([np.zeros(mix.SR // 2), np.ones(200) * 0.9, np.zeros(mix.SR)]))
    p = _proj(tmp_path, [{"word": 3, "sfx": "pop"}])
    _, plan = _plan(p)
    (t, _), = sfx_peaks(plan)
    assert abs(t - WORDS[2][1]) <= 1 / FPS and plan.clips[0].src.endswith("pop.wav")


def test_new_synths_are_deterministic_and_have_a_peak() -> None:
    for name in ("pop", "click", "swish", "hit", "bass-drop", "shutter", "ding"):
        a, b = mix.synth(name), mix.synth(name)
        assert len(a) and np.array_equal(a, b) and 0 <= mix.peak_lead(f"synth:{name}") < len(a) / mix.SR


def test_words_command_and_check_warning(tmp_path: Path, capsys) -> None:
    p = _proj(tmp_path, [{"word": 2, "sfx": "pop"}, {"word": 2, "sfx": "click"}])
    assert main(["-C", str(tmp_path), "words", "--grep", "never"]) == 0
    out = capsys.readouterr().out
    assert "never" in out and "OK words: 1" in out
    assert main(["-C", str(tmp_path), "words", "--grep", "zzz"]) == 1
    _, plan = _plan(p)
    from montaj.emphasis import sfx_lines
    lines = sfx_lines(plan)
    assert lines[0].startswith("INFO sfx=") and any(x.startswith("WARN sfx") for x in lines)
