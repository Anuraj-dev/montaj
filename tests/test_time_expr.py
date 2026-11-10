"""Time expressions: a marker or duration base plus any number of signed offsets in mixed units. CPU only."""
from __future__ import annotations

import pytest

from montaj.spec import SpecError, _expr_uses_beats, parse_time

M = {"replay": 600.0, "drop": 300.0}  # frames at 60 fps; 120 BPM -> 1 beat = 30f


@pytest.mark.parametrize(
    ("expr", "frames"),
    [
        ("replay", 600.0),
        ("replay+6b", 780.0),
        ("replay+6b+0.5s", 810.0),
        ("drop-2b+3f", 243.0),
        ("drop+1s-1s+0f", 300.0),
        ("replay-0.5s-2f-1b", 538.0),
    ],
)
def test_marker_with_chained_offsets(expr: str, frames: float) -> None:
    assert parse_time(expr, M, 60, 120.0, beat0=5.0) == pytest.approx(frames)


def test_duration_base_with_offsets_keeps_grid_rules() -> None:
    # A beat base is a grid time (adds beat0); offsets are spans and never do.
    assert parse_time("8b+2f", {}, 60, 120.0, beat0=5.0) == pytest.approx(5 + 240 + 2)
    assert parse_time("2s-1b", {}, 60, 120.0, beat0=5.0) == pytest.approx(90)
    assert parse_time("8b", {}, 60, 120.0, beat0=5.0) == pytest.approx(245)


@pytest.mark.parametrize("bad", ["replay+", "replay++1s", "replay+1", "replay 1s", "+replay", "1s+drop", "replay+1s+"])
def test_malformed_chain_is_an_error(bad: str) -> None:
    with pytest.raises(SpecError, match="invalid time|bare number"):
        parse_time(bad, M, 60, 120.0)


def test_unknown_marker_in_chain() -> None:
    with pytest.raises(SpecError, match="unknown marker 'nope'"):
        parse_time("nope+1s+2f", M, 60, 120.0)


def test_uses_beats_sees_any_offset() -> None:
    assert _expr_uses_beats("replay+0.5s+6b")
    assert _expr_uses_beats("2s-1b")
    assert not _expr_uses_beats("replay+6f+0.5s")
    assert not _expr_uses_beats("ch1b+2s")
