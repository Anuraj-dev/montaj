"""`montaj spec`: every reference bullet lands in exactly one topic; unknown keys name the nearest real one."""

from __future__ import annotations

import yaml
from montaj import specref
from montaj.cli import main
from montaj.spec import SpecError, load_spec


def _bullets() -> list[str]:
    return [ln.rstrip() for ln in specref.doc_path().read_text(encoding="utf-8").splitlines() if ln.startswith("- ")]


def test_every_bullet_lands_in_exactly_one_topic():
    parsed = specref.topics()
    placed = [ln for lines in parsed.values() for ln in lines if ln.startswith("- ")]
    bullets = _bullets()
    assert sorted(placed) == sorted(bullets)
    assert len(set(placed)) == len(placed)  # no bullet in two topics
    assert parsed["other"] == [], f"no topic rule claims: {parsed['other']}; add one to specref._RULES"


def test_every_topic_is_described_and_exemplified_validly():
    parsed = specref.topics()
    for name, lines in parsed.items():
        if lines:
            assert name in specref.TOPICS, f"describe topic {name!r} in specref.TOPICS"
    for name, ex in specref.EXAMPLES.items():
        assert name in specref.TOPICS
        yaml.safe_load(ex)


def test_topic_output_is_only_that_topic(capsys):
    assert main(["spec", "captions"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert any(ln.startswith("- `captions.preset`") for ln in out)
    assert not any(ln.startswith("- `audio.") for ln in out)
    assert out[-1].startswith("OK spec captions")


def test_list_and_search_and_unknown(capsys):
    assert main(["spec"]) == 0
    listing = capsys.readouterr().out
    assert "captions" in listing and listing.rstrip().splitlines()[-1].startswith("OK spec")
    assert main(["spec", "--search", "shadow"]) == 0
    assert "shadow" in capsys.readouterr().out
    assert main(["spec", "captoins"]) == 1
    assert "did you mean 'captions'" in capsys.readouterr().out


def test_unknown_key_suggests_the_real_one(tmp_path):
    (tmp_path / "montaj.yaml").write_text(
        "video: {size: 1080x1920, fps: 30, look: clean}\n"
        "shots: [{blank: true, hold: 2s}]\n"
        "text:\n  - {from: 0s, to: 2s, lines: [{text: Hi, at: 0s, strok: {color: '#fff'}}]}\n"
    )
    try:
        load_spec(tmp_path / "montaj.yaml")
    except SpecError as exc:
        assert any("text[0].lines[0].strok: unknown key 'strok' — did you mean 'stroke'?" in e for e in exc.errors)
    else:
        raise AssertionError("expected SpecError")


def test_unknown_key_without_a_near_match_lists_the_valid_ones(tmp_path):
    (tmp_path / "montaj.yaml").write_text("video: {size: 1080x1920, fps: 30, look: clean, zzzzqq: 1}\nshots: [{blank: true, hold: 2s}]\n")
    try:
        load_spec(tmp_path / "montaj.yaml")
    except SpecError as exc:
        assert any("unknown key 'zzzzqq' (valid here:" in e and "bpm" in e for e in exc.errors)
    else:
        raise AssertionError("expected SpecError")
