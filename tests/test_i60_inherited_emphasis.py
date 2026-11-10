"""#60: inherited emphasis is inert on unmarked lines; own emphasis there is an error; duplicate ERR lines print once."""
from pathlib import Path

import pytest

from montaj import cli
from montaj.render import glyphs
from montaj.spec import SpecError, load_spec
from tests.test_spec import _write

STYLE = 'styles:\n  slam: {engine: gpu, size: 80, y: 400, emphasis: {color: "#ff0000"}}\n'


def _load(tmp: Path, lines: list[str], extra_style: str = STYLE):
    body = "".join(f"      - {{{ln}}}\n" for ln in lines)
    return load_spec(_write(tmp, "  - {photo: a, hold: 4s}\n", ["a"],
                            extra=f"{extra_style}text:\n  - from: 0s\n    to: 3s\n    lines:\n{body}"))


def test_style_emphasis_validates_on_mixed_lines(tmp_path):
    spec = _load(tmp_path, ["text: a *big* day, style: slam, at: 0s", "text: plain day, style: slam, at: 0s"])
    marked, plain = spec.text[0].lines
    assert any(glyphs.parse_emphasis(marked.text)[1]) and marked.emphasis is not None
    assert plain.emphasis is None  # inert: drawn plain


def test_defaults_emphasis_is_inert_too(tmp_path):
    style = 'defaults:\n  text.lines: {emphasis: {color: "#ff0000"}}\n'
    _load(tmp_path, ["text: plain, engine: gpu, size: 80, y: 400, at: 0s"], style)


def test_own_emphasis_without_marks_fails_naming_the_line(tmp_path):
    with pytest.raises(SpecError) as ei:
        _load(tmp_path, ["text: a *big* day, style: slam, at: 0s",
                         "text: plain day, style: slam, at: 0s, emphasis: {color: '#00ff00'}"])
    assert any(e.startswith("text[0].lines[1].emphasis") and "no *marked* words" in e for e in ei.value.errors)


def test_emit_prints_duplicate_lines_once(capsys):
    cli._emit(["ERR x: bad (fix)", "ERR x: bad (fix)", "WARN y", "ERR x: bad (fix)", "OK done"], False)
    assert capsys.readouterr().out.splitlines() == ["ERR x: bad (fix)", "WARN y", "OK done"]


def test_inherited_emphasis_is_dropped_at_expansion_and_resolves(tmp_path):
    from montaj.timeline import resolve

    style = 'styles:\n  slam: {engine: gpu, size: 80, y: 400, emphasis: {color: "#ff0000", scale: 1.3, shake: 6}}\n'
    p = _write(tmp_path, "  - {photo: a, hold: 4s}\n", ["a"], extra=style + "text:\n  - from: 0s\n    to: 3s\n    lines:\n"
               "      - {text: a *big* day, style: slam, at: 0s}\n      - {text: plain day, style: slam, at: 0s}\n")
    spec = load_spec(p)
    marked, plain = spec.text[0].lines
    assert marked.emphasis is not None and plain.emphasis is None
    resolve(spec, tmp_path)  # validate_spec runs again here; must agree with load_spec


def test_emit_keeps_repeated_data_lines(capsys):
    cli._emit(["    engine: gpu", "    engine: gpu", "OK done"], False)
    assert capsys.readouterr().out.splitlines() == ["    engine: gpu", "    engine: gpu", "OK done"]
