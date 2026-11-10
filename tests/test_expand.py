"""Spec reuse (#42): styles, defaults, components, include, --expand, source-mapped errors."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from PIL import Image

from montaj.cli import main
from montaj.expand import ExpandError, expand_spec
from montaj.project import segment_hash
from montaj.spec import TEXT_STYLES, SpecError, load_spec
from montaj.timeline import resolve
from tests.test_spec import _write

ROOT = Path(__file__).resolve().parents[1]
SHOT = "  - {photo: a, hold: 4s}\n"


def _load(tmp: Path, extra: str, shots: str = SHOT):
    return load_spec(_write(tmp, shots, ["a"], extra=extra))


def _line(tmp: Path, extra: str, shots: str = SHOT):
    return _load(tmp, f"text:\n  - from: 0s\n    to: 3s\n    lines:\n{extra}")


def _err(tmp: Path, extra: str, shots: str = SHOT) -> list[str]:
    with pytest.raises(SpecError) as ei:
        _load(tmp, extra, shots)
    return ei.value.errors


def test_style_fields_and_explicit_wins(tmp_path: Path) -> None:
    spec = _load(tmp_path, """styles:
  hook: {font: anton, size: 120, case: upper, stroke: {width: 8}}
text:
  - from: 0s
    to: 3s
    lines:
      - {text: A, style: hook, y: 900, at: 0s}
      - {text: B, style: hook, y: 900, at: 0s, size: 60, stroke: {color: "#000000"}}
""")
    a, b = spec.text[0].lines
    assert (a.font, a.size, a.case, a.stroke.width, a.style) == ("anton", 120, "upper", 8, "serif")
    assert b.size == 60 and b.stroke.width == 8 and b.stroke.color == "#000000"  # dicts merge


def test_extends_chain_and_builtin_style_inside(tmp_path: Path) -> None:
    spec = _load(tmp_path, """styles:
  base: {style: caps, size: 50, color: "#ffffff"}
  mid: {extends: base, size: 70}
  top: {extends: mid, color: "#ffe600"}
text:
  - {from: 0s, to: 3s, lines: [{text: A, style: top, y: 900, at: 0s}]}
""")
    ln = spec.text[0].lines[0]
    assert (ln.style, ln.size, ln.color) == ("caps", 70, "#ffe600")


def test_extends_cycle_and_unknown(tmp_path: Path) -> None:
    errs = _err(tmp_path, "styles:\n  a: {extends: b}\n  b: {extends: a}\n")
    assert any("extends cycle" in e and "a -> b -> a" in e for e in errs)
    errs = _err(tmp_path, "styles:\n  a: {extends: nope}\n")
    assert any("styles.a.extends" in e and "unknown style 'nope'" in e for e in errs)


def test_builtin_name_is_reserved(tmp_path: Path) -> None:
    errs = _err(tmp_path, "styles:\n  caps: {size: 10}\n")
    assert len(errs) == 1 and "'caps' is a built-in text style" in errs[0] and "styles.caps (line" in errs[0]


def test_style_may_not_set_a_non_builtin_style(tmp_path: Path) -> None:
    errs = _err(tmp_path, "styles:\n  a: {size: 1}\n  b: {style: a}\n")
    assert any("styles.b.style" in e for e in errs)


def test_defaults_precedence(tmp_path: Path) -> None:
    spec = _load(tmp_path, """styles:
  big: {size: 100}
defaults:
  shots: {focus: [4, 4]}
  text.lines: {style: big, size: 10, color: "#111111"}
text:
  - from: 0s
    to: 3s
    lines:
      - {text: A, y: 900, at: 0s}
      - {text: B, y: 900, at: 0s, size: 33}
      - {text: C, y: 900, at: 0s, style: caps}
""", shots="  - {photo: a, hold: 4s}\n  - {photo: a, hold: 1s, focus: [1, 2]}\n")
    a, b, c = spec.text[0].lines
    assert (a.size, a.color) == (100, "#111111")  # style beats defaults; defaults fill the rest
    assert b.size == 33
    assert (c.style, c.size, c.color) == ("caps", 10, "#111111")  # built-in explicit; defaults still fill
    assert spec.shots[0].focus == (4, 4) and spec.shots[1].focus == (1, 2)


def test_default_null_drops_a_default(tmp_path: Path) -> None:
    spec = _load(tmp_path, "defaults:\n  shots: {in: {type: fade, dur: 1s}}\n", "  - {photo: a, hold: 2s}\n  - {photo: a, hold: 2s, in: null}\n")
    assert spec.shots[0].in_.type == "fade" and spec.shots[1].in_ is None


COMP = """components:
  callout:
    params: [word, at]
    text:
      - {from: $at, to: $at+1.2s, lines: [{text: "$word!", y: 900, at: $at}]}
"""


def test_component_arithmetic_and_markers(tmp_path: Path) -> None:
    spec = _load(tmp_path, COMP + """markers: {m: 1s}
use:
  - {component: callout, word: FREE, at: 2s}
  - {component: callout, word: HI, at: m}
  - {component: callout, word: UP, at: m+0.3s}
""")
    got = [(b.from_, b.to, b.lines[0].text) for b in spec.text]
    assert got == [("2s", "3.2s", "FREE!"), ("m", "m+1.2s", "HI!"), ("m+0.3s", "m+1.5s", "UP!")]


def test_component_splices_in_place_and_param_defaults(tmp_path: Path) -> None:
    spec = _load(tmp_path, """components:
  hold1: {params: {sec: 1s}, shots: [{photo: a, hold: $sec}]}
""", "  - {photo: a, hold: 4s}\n  - {use: hold1}\n  - {use: hold1, sec: 2s}\n  - {photo: a, hold: 3s}\n")
    assert [s.hold for s in spec.shots] == ["4s", "1s", "2s", "3s"]


def test_component_errors(tmp_path: Path) -> None:
    assert any("needs param 'at'" in e and "use[0] (line" in e for e in _err(tmp_path, COMP + "use:\n  - {component: callout, word: X}\n"))
    assert any("no param 'zz'" in e for e in _err(tmp_path, COMP + "use:\n  - {component: callout, word: X, at: 1s, zz: 1}\n"))
    assert any("unknown component 'nope'" in e for e in _err(tmp_path, "use:\n  - {component: nope}\n"))
    assert any("unknown param $q" in e for e in _err(tmp_path, "components:\n  c: {params: [a], text: [{from: $q}]}\nuse:\n  - {component: c, a: 1s}\n"))
    assert any("mixed units" in e for e in _err(tmp_path, COMP + "use:\n  - {component: callout, word: X, at: 8f}\n"))


def test_errors_point_at_the_authors_key_and_line(tmp_path: Path) -> None:
    errs = _err(tmp_path, """styles:
  hook: {size: abc}
text:
  - from: 0s
    to: 3s
    lines:
      - {text: A, style: hook, y: 900, at: 0s}
""")
    assert len(errs) == 1
    assert re.match(r"styles\.hook\.size \(line 11\):", errs[0]) and "text[0]" not in errs[0]
    assert "components.callout.text[0].to (line" in _err(tmp_path, COMP + "use:\n  - {component: callout, word: X, at: 5}\n")[0]
    errs = _err(tmp_path, COMP + "use:\n  - {component: callout, word: X, at: nomark}\n")
    assert errs and all(e.startswith("use[0]") and "unknown marker" in e for e in errs)
    errs = _err(tmp_path, "defaults:\n  shots: {hold: 7}\n", "  - {photo: a}\n")
    assert any(e.startswith("defaults.shots.hold (line") for e in errs)


def test_include_shares_styles_and_cycles(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    (tmp_path / "brand.yaml").write_text("styles:\n  hook: {size: 99, font: anton}\n  other: {size: 1}\ndefaults:\n  shots: {focus: [1, 1]}\n")
    spec = _load(proj, """include: ../brand.yaml
styles:
  other: {size: 5}
text:
  - {from: 0s, to: 3s, lines: [{text: A, style: hook, y: 900, at: 0s}, {text: B, style: other, y: 900, at: 0s}]}
""")
    a, b = spec.text[0].lines
    assert (a.size, a.font, b.size) == (99, "anton", 5) and spec.shots[0].focus == (1, 1)
    (tmp_path / "brand.yaml").write_text("styles:\n  hook: {size: abc}\n")
    errs = _err(proj, "include: ../brand.yaml\ntext:\n  - {from: 0s, to: 3s, lines: [{text: A, style: hook, y: 900, at: 0s}]}\n")
    assert any(e.startswith("../brand.yaml:styles.hook.size (line 2)") for e in errs)
    (tmp_path / "brand.yaml").write_text("include: proj/montaj.yaml\n")
    assert any("include cycle" in e for e in _err(proj, "include: ../brand.yaml\n"))
    (tmp_path / "brand.yaml").write_text("shots: []\n")
    assert any("may only hold styles" in e for e in _err(proj, "include: ../brand.yaml\n"))
    assert any("cannot read" in e for e in _err(proj, "include: ../missing.yaml\n"))


def test_no_reuse_keys_leaves_the_spec_untouched() -> None:
    data = {"video": {}, "shots": []}
    out, smap = expand_spec(data, "", Path("x.yaml"), TEXT_STYLES)
    assert out is data and smap is None


def test_expand_error_type() -> None:
    with pytest.raises(ExpandError):
        expand_spec({"styles": {"caps": {}}}, "", Path("x.yaml"), TEXT_STYLES)


def test_validate_expand_prints_the_plain_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path, SHOT, ["a"], extra="""styles:
  hook: {size: 120}
text:
  - {from: 0s, to: 3s, lines: [{text: A, style: hook, y: 900, at: 0s}]}
""")
    assert main(["-C", str(tmp_path), "validate", "--expand"]) == 0
    out = capsys.readouterr().out
    data = yaml.safe_load(out.rsplit("\nOK ", 1)[0])
    assert data["text"][0]["lines"][0] == {"text": "A", "y": 900, "at": "0s", "size": 120}
    assert "styles" not in data


# --- the recipe: same frames, fewer lines ------------------------------------------------------


class _Shas(dict):
    def __missing__(self, key):
        return "sha-" + key


def _hashes(spec: Path) -> list[str]:
    tl = resolve(load_spec(spec), spec.parent)
    return [segment_hash(tl, seg, _Shas(), "e1") for seg in tl.segments]


def test_birthday_song_film_reuse_keeps_every_segment_hash(tmp_path: Path) -> None:
    new = ROOT / "recipes" / "birthday-song-film.yaml"
    old = ROOT / "tests" / "fixtures" / "birthday-song-film.flat.yaml"
    for stem in set(re.findall(r"photo: ([\w-]+)", old.read_text())):
        (tmp_path / "assets").mkdir(exist_ok=True)
        Image.new("RGB", (8, 8), (10, 20, 30)).save(tmp_path / "assets" / f"{stem}.jpg", "JPEG")
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "song.wav").write_bytes(b"RIFF")
    (tmp_path / "old.yaml").write_text(old.read_text())
    (tmp_path / "new.yaml").write_text(new.read_text())
    a, b = _hashes(tmp_path / "old.yaml"), _hashes(tmp_path / "new.yaml")
    assert len(a) == 56 and a == b
    assert len(new.read_text().splitlines()) <= 0.7 * len(old.read_text().splitlines())
