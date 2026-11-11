"""#62 item 1: a one-line text block (`text:` a string) and a line's `at` defaulting to its block's `from`."""
from __future__ import annotations

from pathlib import Path

import pytest

from montaj.expand import TEXT_BLOCK_KEYS
from montaj.project import segment_hash
from montaj.spec import SpecError, TextBlock, load_spec
from montaj.timeline import resolve
from tests.test_spec import _write

SHOTS = "  - {photo: a, hold: 4s}\n  - {photo: a, hold: 4s}\n"
STYLE = """styles:
  slamx: {style: caps, size: 40, in: {type: slam}}
defaults:
  text.lines: {style: slamx, pos: [540, 500]}
"""
LONG = STYLE + """text:
  - from: 1b
    to: 4b
    lines:
      - {text: "SWITCH *WORKSPACES*", at: 1b}
  - from: 5b
    to: 8b
    fade_out: 0.3s
    lines:
      - {text: GO, at: 5b, size: 60}
      - {text: NOW, at: 6b}
"""
SHORT = STYLE + """text:
  - {at: 1b, to: 4b, text: "SWITCH *WORKSPACES*"}
  - {from: 5b, to: 8b, fade_out: 0.3s, lines: [{text: GO, size: 60}, {text: NOW, at: 6b}]}
"""


class _Shas(dict):
    def __missing__(self, key):
        return "sha-" + key


def _hashes(path: Path) -> list[str]:
    tl = resolve(load_spec(path), path.parent)
    return [segment_hash(tl, seg, _Shas(), "e1") for seg in tl.segments]


def _spec(tmp: Path, name: str, extra: str) -> Path:
    d = tmp / name
    d.mkdir()
    return _write(d, SHOTS, ["a"], extra=extra)


def test_short_form_equals_long_form(tmp_path: Path) -> None:
    long, short = _spec(tmp_path, "long", LONG), _spec(tmp_path, "short", SHORT)
    assert load_spec(short).model_dump(by_alias=True) == load_spec(long).model_dump(by_alias=True)
    assert _hashes(short) == _hashes(long)


def test_compact_block_keeps_block_fields_and_from_defaults_to_at(tmp_path: Path) -> None:
    spec = load_spec(_spec(tmp_path, "s", "text:\n  - {at: 2b, to: 4b, fade_in: 0.2s, role: title, "
                          "text: HI, y: 900, size: 80}\n"))
    b = spec.text[0]
    assert (b.from_, b.to, b.fade_in, b.role) == ("2b", "4b", "0.2s", "title")
    assert [(ln.text, ln.at, ln.y, ln.size) for ln in b.lines] == [("HI", "2b", 900, 80)]


def test_line_at_defaults_without_reuse_keys(tmp_path: Path) -> None:
    spec = load_spec(_spec(tmp_path, "s", "text:\n  - {from: 3b, to: 6b, lines: [{text: A, y: 900}]}\n"))
    assert spec.text[0].lines[0].at == "3b"


def test_text_with_lines_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SpecError) as ei:
        load_spec(_spec(tmp_path, "s", "text:\n  - {at: 0s, to: 2s, text: A, lines: [{text: B, at: 0s, y: 9}]}\n"))
    assert any("text[0]" in e and "one-line block" in e for e in ei.value.errors)


def test_block_keys_match_text_block_fields() -> None:
    fields = {f.alias or name for name, f in TextBlock.model_fields.items()} - {"lines"}
    assert set(TEXT_BLOCK_KEYS) == fields
