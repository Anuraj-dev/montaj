"""taste.md defaults and append, with MONTAJ_TASTE pointed at tmp."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from montaj import taste


def test_defaults_then_add(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cfg" / "taste.md"
    monkeypatch.setenv("MONTAJ_TASTE", str(path))
    text = taste.read()
    assert path.is_file()
    assert text == path.read_text()
    assert "photos never warp/drift unless the recipe asks" in text
    assert "motion lives in transitions" in text
    assert "cuts land on the beat grid" in text
    assert "text is short and centred" in text
    assert "end on black" in text
    n_default = sum(1 for ln in text.splitlines() if ln.strip())
    n = taste.add("no ken burns on faces")
    written = path.read_text()
    assert n == n_default + 1
    assert written.endswith(f"- no ken burns on faces ({date.today().isoformat()})\n")
    assert taste.read() == written
