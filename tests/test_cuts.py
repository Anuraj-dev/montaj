"""Transcript cuts (#39): span planning, `cuts: auto` expansion, click-free joins, `montaj cut`."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from montaj import cuts
from montaj.audio import mix
from montaj.audio.mix import SR
from montaj.cli import main
from montaj.qa import join_clicks
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve
from tests.test_audio_mix import tone, write_wav
from tests.test_linked_audio import FPS, make_clip

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def w(text: str, s: float, e: float) -> dict:
    return {"w": text, "s": s, "e": e}


WORDS = [
    w("so", 0.5, 0.7), w("today", 0.75, 1.2), w("um", 1.3, 1.6), w("we", 1.65, 1.8), w("build", 1.85, 2.3),
    w("something", 3.5, 4.0), w("good", 4.05, 4.4),
    w("we", 5.0, 5.2), w("build", 5.25, 5.6), w("something", 5.65, 6.0), w("great", 6.05, 6.4),
    w("we", 7.0, 7.2), w("build", 7.25, 7.45), w("something", 7.5, 7.7), w("great", 7.75, 7.95),
]


def test_spans_cut_silence_and_fillers() -> None:
    spans, fillers = cuts.plan_spans(WORDS, "en")
    assert fillers == 1
    assert [(sp["s"], sp["e"]) for sp in spans] == [
        (0.5, 1.2), (1.65, 2.3), (3.5, 4.4), (5.0, 6.4), (7.0, 7.95)]
    assert [sp["i"] for sp in spans] == [1, 2, 3, 4, 5]
    assert spans[0]["text"] == "so today"


def test_a_short_gap_does_not_split() -> None:
    spans, _ = cuts.plan_spans([w("a", 0, 0.3), w("b", 0.5, 0.8)], "en")
    assert len(spans) == 1
    assert len(cuts.plan_spans([w("a", 0, 0.3), w("b", 0.5, 0.8)], "en", silence=0.1)[0]) == 2


def test_repeated_opening_is_flagged_as_a_retake() -> None:
    spans, _ = cuts.plan_spans(WORDS, "en")
    assert [sp.get("retake", False) for sp in spans] == [False, False, False, True, False]
    assert "retake?" in cuts.table({"spans": spans})[4]


def test_edges_snap_to_the_quietest_nearby_frame() -> None:
    db = np.full(1000, -20.0)
    db[48:50] = -80.0   # quiet just before the word at 0.5 s
    db[123:125] = -80.0  # quiet just after the word ends at 1.2 s
    spans, _ = cuts.plan_spans([w("hello", 0.5, 1.2)], "en", db=db)
    assert spans[0]["s"] == pytest.approx(0.485, abs=0.011)
    assert spans[0]["e"] == pytest.approx(1.235, abs=0.011)


def test_hindi_list_keeps_english_fillers_too() -> None:
    assert cuts.is_filler("हम्म", "hi") and cuts.is_filler("Um,", "hi")
    assert not cuts.is_filler("हम्म", "en")


# --- expansion -------------------------------------------------------------------------------------------


def project(tmp: Path, shot: dict, spans: list[dict] | None = None, extra: dict | None = None) -> Path:
    clip = tmp / "assets" / "talk.mov"
    if not clip.is_file():
        make_clip(clip, sound=tone(8.0, 200.0, 0.3))
    spans = spans or [{"i": 1, "s": 0.5, "e": 1.5, "text": "a"}, {"i": 2, "s": 2.0, "e": 3.0, "text": "b"},
                      {"i": 3, "s": 4.0, "e": 5.0, "text": "c"}]
    (tmp / "assets" / "talk.cuts.json").write_text(json.dumps({"v": 1, "duration": 8.0, "spans": spans}))
    spec = {"video": {"size": "64x64", "fps": FPS, "look": "warm-film"}, "assets": "assets",
            "shots": [{"clip": "talk", "cuts": "auto", **shot}], **(extra or {})}
    p = tmp / "montaj.yaml"
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    return p


@needs_ffmpeg
def test_cuts_expand_to_range_shots_with_padding(tmp_path: Path) -> None:
    spec = load_spec(project(tmp_path, {}))
    assert [sh.range for sh in spec.shots] == [("13f", "49f"), ("58f", "94f"), ("118f", "154f")]
    assert all(sh.cuts is None and sh.clip == "talk" for sh in spec.shots)
    tl = resolve(spec, tmp_path)
    assert [r.end - r.start for r in tl.shots] == [36, 36, 36]


@needs_ffmpeg
def test_drop_pad_and_punch(tmp_path: Path) -> None:
    p = project(tmp_path, {"drop": [2], "cuts_pad": ["0ms", "0ms"], "cuts_punch": "alternate"})
    spec = load_spec(p)
    assert [sh.range for sh in spec.shots] == [("15f", "45f"), ("120f", "150f")]
    assert [sh.zoom for sh in spec.shots] == [None, 1.12]


@needs_ffmpeg
def test_padding_never_reaches_into_the_cut_gap(tmp_path: Path) -> None:
    spans = [{"i": 1, "s": 1.0, "e": 2.0, "text": "a"}, {"i": 2, "s": 2.1, "e": 3.0, "text": "b"}]
    spec = load_spec(project(tmp_path, {"cuts_pad": ["500ms", "500ms"]}, spans))
    first, second = (sh.range for sh in spec.shots)
    assert first[1] == second[0] == "61f"  # halfway through the 0.1 s gap (2.05 s)


@needs_ffmpeg
def test_errors_name_the_shot(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match=r"shots\[0\]\.drop: no span 9"):
        load_spec(project(tmp_path, {"drop": [9]}))
    p = project(tmp_path, {})
    (tmp_path / "assets" / "talk.cuts.json").unlink()
    with pytest.raises(SpecError, match="run montaj cut talk"):
        load_spec(p)
    with pytest.raises(SpecError, match=r"shots\[0\]\.hold"):
        load_spec(project(tmp_path, {"hold": "2s"}))


@needs_ffmpeg
def test_linked_ids_and_duck_follow_the_spans(tmp_path: Path) -> None:
    p = project(tmp_path, {"audio": True, "id": "voice"},
                extra={"audio": {"duck": {"under": ["voice"]}}})
    spec = load_spec(p)
    assert [sh.id for sh in spec.shots] == ["voice", "voice_2", "voice_3"]
    assert spec.audio.duck.under == ["voice", "voice_2", "voice_3"]


# --- audio -----------------------------------------------------------------------------------------------


def mixed(p: Path) -> tuple[np.ndarray, mix.MixPlan]:
    tl = resolve(load_spec(p), p.parent)
    plan = mix.plan_audio(tl.spec, p.parent, tl.markers, FPS, None, 0.0, [], shots=tl.shots)
    assert plan is not None and plan.active
    out, _ = mix.mix_audio(plan)
    return out[:, 0], plan


@needs_ffmpeg
def test_joins_crossfade_without_a_track_and_without_clicks(tmp_path: Path) -> None:
    p = project(tmp_path, {"audio": True, "cuts_pad": ["0ms", "0ms"]}, extra={"audio": {}})
    out, plan = mixed(p)
    assert plan.track is None
    joins = [link.a / SR for link in plan.links if link.xin]
    assert len(joins) == 2 and all(link.xout for link in plan.links[:2])
    assert not join_clicks(out, joins)
    # the same joins cut hard (jump in a tone's phase) would step by far more than the tone's slope
    hard = np.concatenate([tone(0.5, 200.0), tone(0.5, 200.0)[37:]])
    assert join_clicks(hard, [0.5])


def test_join_clicks_flags_a_step_and_passes_a_smooth_cut() -> None:
    smooth = tone(1.0, 200.0, 0.3)
    assert not join_clicks(smooth, [0.5])
    step = smooth.copy()
    step[SR // 2 :] += 0.2
    got = join_clicks(step, [0.5])
    assert len(got) == 1 and got[0][1] > 0.19


@needs_ffmpeg
def test_hash_ignores_cuts_authoring_fields(tmp_path: Path) -> None:
    a = resolve(load_spec(project(tmp_path, {})), tmp_path)
    b = resolve(load_spec(project(tmp_path, {"audio": True})), tmp_path)
    from montaj.project import segment_hash
    shas = {"talk": "x"}
    assert [segment_hash(a, s, shas, "v") for s in a.segments] == [segment_hash(b, s, shas, "v") for s in b.segments]


# --- command ---------------------------------------------------------------------------------------------


@needs_ffmpeg
def test_cut_command_writes_the_plan_and_prints_the_table(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    project(tmp_path, {})
    (tmp_path / "assets" / "talk.cuts.json").unlink()
    (tmp_path / "assets" / "talk.words.json").write_text(json.dumps({"words": [
        {"i": i, "w": x["w"], "s": x["s"], "e": x["e"]} for i, x in enumerate(WORDS)]}))
    assert main(["-C", str(tmp_path), "cut", "talk", "--lang", "en"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("OK assets/talk.cuts.json 8.0s -> ") and "5 spans" in line and "fillers=1" in line
    plan = json.loads((tmp_path / "assets" / "talk.cuts.json").read_text())
    assert len(plan["spans"]) == 5 and plan["spans"][3]["retake"] is True
    assert main(["-C", str(tmp_path), "cut", "talk", "--print"]) == 0
    rows = capsys.readouterr().out.splitlines()
    assert rows[0].split() == ["span", "start", "end", "text"] and len(rows) == 7 and rows[4].endswith("retake?")
    assert rows[-1] == "OK 5 spans"


def _bursts(starts: list[float], dur: float, total: float) -> np.ndarray:
    n = int(total * SR)
    a = np.zeros((n, 2))
    for s in starts:
        t = np.arange(int(dur * SR)) / SR
        a[int(s * SR): int(s * SR) + len(t)] = (0.3 * np.sin(2 * np.pi * 220 * t))[:, None]
    return a


def test_pauses_inside_contiguous_words_split_the_spans() -> None:
    audio = _bursts([0.5, 2.1, 3.7], 1.0, 5.0)  # 0.6 s silences between bursts
    words = [w("a", 0.5, 1.0), w("b", 1.0, 1.9), w("c", 1.9, 2.6), w("d", 2.6, 3.65), w("e", 3.65, 4.7)]
    spans, _ = cuts.plan_spans(words, "en", db=cuts.energy_db(audio), duration=5.0)
    assert len(spans) == 3
    for sp, (lo, hi) in zip(spans, [(0.5, 1.5), (2.1, 3.1), (3.7, 4.7)]):
        assert lo - 0.11 <= sp["s"] <= lo + 0.04 and hi - 0.04 <= sp["e"] <= hi + 0.11
    assert spans[0]["e"] < 1.6 and spans[1]["s"] > 1.9 and spans[1]["e"] < 3.7 and spans[2]["s"] > 3.1


def test_real_word_gaps_plan_the_same_with_energy() -> None:
    audio = np.zeros((int(8.5 * SR), 2))
    for s, e in [(0.5, 1.2), (1.65, 2.3), (3.5, 4.4), (5.0, 6.4), (7.0, 7.95)]:
        audio[int(s * SR): int(e * SR)] = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int((e - s) * SR)) / SR)[:, None]
    db = cuts.energy_db(audio)
    with_db, _ = cuts.plan_spans(WORDS, "en", db=db, duration=8.5)
    plain, _ = cuts.plan_spans(WORDS, "en")
    assert [sp["text"] for sp in with_db] == [sp["text"] for sp in plain]
    assert all(abs(a["s"] - b["s"]) <= 0.11 and abs(a["e"] - b["e"]) <= 0.11 for a, b in zip(with_db, plain))


# --- word times through the cut layout (#53) -------------------------------------------------------------

TALK_WORDS = [  # source seconds: one word in each span, one in the gap between spans 1 and 2
    {"i": 1, "w": "alpha", "s": 0.6, "e": 0.9}, {"i": 2, "w": "gap", "s": 1.7, "e": 1.9},
    {"i": 3, "w": "bravo", "s": 2.2, "e": 2.5}, {"i": 4, "w": "charlie", "s": 4.5, "e": 4.8},
]


def cut_film(tmp: Path, shots: list[dict], extra: dict | None = None) -> Path:
    p = project(tmp, {}, extra=extra)
    (tmp / "assets" / "talk.words.json").write_text(json.dumps({"words": TALK_WORDS}))
    spec = yaml.safe_load(p.read_text())
    spec["shots"] = shots
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    return p


def film_words(p: Path) -> list[tuple[int, float, float]]:
    from montaj.emphasis import load_words

    errors: list[str] = []
    got = load_words(load_spec(p, expand=False), p.parent, errors)
    assert not errors
    return [(w.i, round(w.s * FPS, 3), round(w.e * FPS, 3)) for w in got]


@needs_ffmpeg
def test_captions_words_follow_the_cut_layout(tmp_path: Path) -> None:
    # spans play at source frames 13-49, 58-94, 118-154 -> film 0-36, 36-72, 72-108
    p = cut_film(tmp_path, [{"clip": "talk", "cuts": "auto"}], extra={"captions": {"source": "talk"}})
    assert film_words(p) == [(1, 5.0, 14.0), (3, 44.0, 53.0), (4, 89.0, 98.0)]  # the gap word is cut out
    from montaj.captions import plan_chunks

    errors: list[str] = []
    chunks = plan_chunks(load_spec(p, expand=False), tmp_path, errors)
    assert not errors and [c.starts[0] for c in chunks] == [5, 44, 89]


@needs_ffmpeg
def test_words_land_where_each_shot_of_the_clip_plays_them(tmp_path: Path) -> None:
    p = cut_film(tmp_path, [
        {"clip": "talk", "cuts": "auto", "drop": [1, 2]},  # span 3 first: film 0-36
        {"hold": "20f", "layers": [{"clip": "talk", "clip_in": "1.9s", "audio": True}]},  # source 57-77: film 36-56
        {"clip": "talk", "cuts": "auto", "drop": [2, 3]},  # span 1 last: film 56-92
    ], extra={"captions": {"source": "talk"}, "audio": {}})
    assert film_words(p) == [(4, 17.0, 26.0), (3, 45.0, 54.0), (1, 61.0, 70.0)]


@needs_ffmpeg
def test_emphasis_on_a_cut_clip_anchors_to_film_time(tmp_path: Path) -> None:
    p = cut_film(tmp_path, [{"clip": "talk", "cuts": "auto"}],
                 extra={"captions": {"source": "talk"}, "emphasis": [{"match": "charlie", "flash": 0.5}]})
    spec = load_spec(p)
    assert [spec.markers[fx.at] for fx in spec.fx] == [f"{89 / FPS:.4f}s"]


@needs_ffmpeg
def test_words_without_cuts_stay_in_source_time(tmp_path: Path) -> None:
    p = cut_film(tmp_path, [{"clip": "talk", "range": ["13f", "154f"], "audio": True}],
                 extra={"captions": {"source": "talk"}, "audio": {}})
    assert film_words(p)[0] == (1, 18.0, 27.0)
