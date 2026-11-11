"""#62 item 4: `montaj spec <topic>` prints the short form; `--full` the whole reference; `--search` is unchanged."""
from __future__ import annotations

import re

from montaj import specref
from montaj.cli import main


def _out(capsys, *argv: str) -> list[str]:
    assert main(["spec", *argv]) == 0
    return capsys.readouterr().out.splitlines()


def _keys(lines: list[str]) -> list[str]:
    return [m.group(1) for ln in lines if (m := re.match(r"- `([^`]+)`", ln))]


def test_short_form_keeps_every_field_and_is_smaller(capsys) -> None:
    for topic in ("text", "shots", "layers", "audio"):
        short, full = _out(capsys, topic), _out(capsys, topic, "--full")
        assert _keys(short) == _keys(full)
        assert len("\n".join(short)) < 0.75 * len("\n".join(full))
        assert all(len(ln) <= specref.SHORT_CAP for ln in short if ln.startswith("- "))
        assert "Example:" in short and short[-1].endswith("--full for every detail")
        assert full[:-1] == specref.render_topic(topic, specref.topics())  # --full is today's text


def test_text_topic_shows_the_one_line_title_and_the_at_default_near_the_top(capsys) -> None:
    short = _out(capsys, "text")
    top = "\n".join(short[:3])
    assert "`text[].text`" in top and "text: \"GO *NOW*\"}" in top and "default the line's `at`" in top
    at = next(ln for ln in short if ln.startswith("- `text[].lines[].at`"))
    assert "default the block's `from`" in at
    assert '{at: 4b, to: 6b, text: "SWITCH *WORKSPACES*"}' in "\n".join(short)


def test_search_prints_whole_lines(capsys) -> None:
    out = _out(capsys, "--search", "slam")
    full = [ln for ln in specref.doc_path().read_text().splitlines() if ln.startswith("- ")]
    assert out[:-1] and all(ln in full for ln in out[:-1])


def test_prose_only_topic_prints_whole(capsys) -> None:
    out = _out(capsys, "ingest")
    assert any("montaj assets [STEM]" in ln and "prints all" in ln for ln in out)


def test_audio_short_form_shows_the_sfx_list() -> None:
    line = next(ln for ln in specref.render_topic("audio", specref.topics(), full=False)
                if ln.startswith("- `audio.sfx`"))
    assert "[auto, text, cuts]" in line


def test_first_sentence_does_not_end_at_eg_or_ie() -> None:
    assert specref._first_sentence("Pick one, e.g. `a`, or i.e. B. Then more.") == "Pick one, e.g. `a`, or i.e. B."
