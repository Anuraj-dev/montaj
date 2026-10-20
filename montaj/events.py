"""Screen-recording event logs and selectors.

A log is a JSON list of objects. `t` is seconds from the file's first video frame
(after `ingest-rec` subtracts that frame's PTS). `at_event` and `event:` markers
both pick a row with `{kind, field: value, contains, nth}`.

No module state: every load takes a path and returns the rows.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED = frozenset({"kind", "nth", "contains"})


def _line(path: str, problem: str, hint: str) -> str:
    return f"{path}: {problem} ({hint})"


def _show(val: object) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float) and math.isfinite(val) and val == int(val):
        return str(int(val))
    text = val if isinstance(val, str) else str(val)
    if len(text) > 40:
        return text[:37] + "..."
    return text


def _values_equal(have: object, want: object) -> bool:
    """Exact match. Numbers compare numerically; bools never match 0 or 1."""
    if isinstance(have, bool) or isinstance(want, bool):
        return isinstance(have, bool) and isinstance(want, bool) and have == want
    if isinstance(have, (int, float)) and isinstance(want, (int, float)):
        return math.isfinite(float(have)) and math.isfinite(float(want)) and float(have) == float(want)
    if isinstance(have, (int, float)) and isinstance(want, str):
        if not math.isfinite(float(have)):
            return False
        try:
            return float(want) == float(have)
        except ValueError:
            return False
    return have == want


def format_seconds(seconds: float) -> str:
    """Duration string that `_dur_frames` reads back as the same float seconds."""
    seconds = float(seconds)
    if seconds < 0 or not math.isfinite(seconds):
        raise ValueError(f"bad seconds {seconds}")
    if seconds == 0:
        return "0s"
    body = repr(seconds)
    if any(ch in body for ch in "eEn"):
        body = format(seconds, ".12f")
    if "." in body:
        body = body.rstrip("0").rstrip(".")
    return f"{body or '0'}s"


def event_time(value: object) -> float | None:
    """Finite seconds, or None. Booleans are not numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    n = float(value)
    return n if math.isfinite(n) else None


@dataclass(frozen=True)
class Selector:
    """One pick. `nth` is 1-based among rows that match, in log order."""

    kind: str | None = None
    equals: tuple[tuple[str, object], ...] = ()
    contains: tuple[tuple[str, str], ...] = ()
    nth: int = 1


@dataclass(frozen=True)
class Hit:
    event: dict


@dataclass(frozen=True)
class Miss:
    """`label` is the filter that failed, or the selector plus `#n` when nth runs past the end."""

    label: str
    have: tuple[str, ...]

    @property
    def hint(self) -> str:
        return "have: " + (", ".join(self.have) if self.have else "none")


def _parse_nth(value: object, path: str, errors: list[str]) -> int | None:
    if isinstance(value, bool):
        errors.append(_line(path, "need a whole number >= 1, got a boolean", "use nth: 1"))
        return None
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, float) and math.isfinite(value) and value == int(value) and int(value) >= 1:
        return int(value)
    errors.append(_line(path, f"need a whole number >= 1, got {value!r}", "use nth: 1"))
    return None


def compile_at_event(raw: object, path: str, errors: list[str]) -> Selector | None:
    """`{kind, <field>: value, contains: {field: substring}, nth}` → a selector.

    Returns None after recording an error. An empty mapping matches every row.
    """
    if not isinstance(raw, dict):
        errors.append(_line(path, "need a mapping", "use {kind: state, state: speaking, nth: 1}"))
        return None
    bad = False
    kind: str | None = None
    if "kind" in raw:
        got = raw["kind"]
        if not isinstance(got, str) or not got:
            errors.append(_line(f"{path}.kind", f"need a string, got {got!r}", "use kind: state"))
            bad = True
        else:
            kind = got
    nth = _parse_nth(raw.get("nth", 1), f"{path}.nth", errors)
    if nth is None:
        bad = True
        nth = 1
    contains: list[tuple[str, str]] = []
    contains_raw = raw.get("contains", {})
    if not isinstance(contains_raw, dict):
        errors.append(_line(f"{path}.contains", "need a mapping", "use contains: {message: theme}"))
        bad = True
    else:
        for key, val in contains_raw.items():
            if not isinstance(key, str) or not _FIELD.match(key):
                errors.append(_line(f"{path}.contains", f"bad field {key!r}", "use a name like message"))
                bad = True
                continue
            if not isinstance(val, str):
                errors.append(_line(f"{path}.contains.{key}", "need a string", "substring to find"))
                bad = True
                continue
            contains.append((key, val))
    equals: list[tuple[str, object]] = []
    for key, val in raw.items():
        if key in _RESERVED:
            continue
        if not isinstance(key, str) or not _FIELD.match(key):
            errors.append(_line(path, f"bad field {key!r}", "use a name like state"))
            bad = True
            continue
        if isinstance(val, (dict, list)):
            errors.append(
                _line(f"{path}.{key}", "need a value", "exact match, or contains: for a substring")
            )
            bad = True
            continue
        equals.append((key, val))
    if bad:
        return None
    return Selector(kind=kind, equals=tuple(equals), contains=tuple(contains), nth=nth)


def is_event_marker(text: str) -> bool:
    return text.startswith("event:")


def compile_event_marker(text: str, path: str, errors: list[str]) -> tuple[str, Selector] | None:
    """`event:<clip>/<field>=<value>[,…][#n]`. `contains:<field>=<text>` or `field~=text` is a substring.

    Values are strings (numbers still match a numeric event field). `#n` defaults to 1.
    """
    raw = text.strip()
    hint = "use event:take/state=speaking#1"
    if not raw.startswith("event:"):
        errors.append(_line(path, f"invalid event marker {raw!r}", hint))
        return None
    body = raw[len("event:") :]
    if "/" not in body:
        errors.append(_line(path, f"invalid event marker {raw!r}", hint))
        return None
    stem, rest = body.split("/", 1)
    if not stem or any(ch.isspace() for ch in stem) or "/" in stem or stem in (".", ".."):
        errors.append(_line(path, f"invalid event marker {raw!r}", hint))
        return None
    nth = 1
    if "#" in rest:
        rest, nth_s = rest.rsplit("#", 1)
        if not nth_s.isdigit() or int(nth_s) < 1:
            errors.append(_line(path, f"invalid event marker {raw!r}", "nth starts at 1, e.g. #1"))
            return None
        nth = int(nth_s)
    if not rest.strip():
        errors.append(_line(path, f"invalid event marker {raw!r}", hint))
        return None
    kind: str | None = None
    equals: list[tuple[str, object]] = []
    contains: list[tuple[str, str]] = []
    for part in rest.split(","):
        part = part.strip()
        if not part:
            errors.append(_line(path, f"invalid event marker {raw!r}", hint))
            return None
        substring = False
        if part.startswith("contains:"):
            part = part[len("contains:") :]
            substring = True
        elif "~=" in part:
            substring = True
            key, val = part.split("~=", 1)
            part = f"{key}={val}"
        if "=" not in part:
            errors.append(_line(path, f"invalid event marker {raw!r}", hint))
            return None
        key, val = part.split("=", 1)
        key = key.strip()
        if not _FIELD.match(key) or val == "":
            errors.append(_line(path, f"invalid event marker {raw!r}", hint))
            return None
        if key == "nth":
            errors.append(_line(path, "nth is the #N suffix", "use event:take/state=speaking#2"))
            return None
        if substring:
            contains.append((key, val))
        elif key == "kind":
            kind = val
        else:
            equals.append((key, val))
    return stem, Selector(kind=kind, equals=tuple(equals), contains=tuple(contains), nth=nth)


def _filter_label(key: str, want: object, mode: str) -> str:
    if mode == "contains":
        return f"contains:{key}={want}"
    return f"{key}={_show(want)}"


def _selector_label(sel: Selector) -> str:
    parts = [f"{key}={_show(val)}" for key, val in sel.equals]
    parts.extend(f"contains:{key}={val}" for key, val in sel.contains)
    if sel.kind is not None:
        parts.insert(0, f"kind={sel.kind}")
    return ",".join(parts) if parts else "any"


def _distinct(events: list[dict], key: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for ev in events:
        if key not in ev or ev[key] is None:
            text = "none"
        else:
            text = _show(ev[key])
        if text not in seen:
            seen.add(text)
            out.append(text)
    if len(out) > 12:
        return tuple(out[:12] + ["…"])
    return tuple(out)


def _field_ok(ev: dict, key: str, want: object, mode: str) -> bool:
    have = ev.get(key)
    if mode == "contains":
        return isinstance(have, str) and isinstance(want, str) and want in have
    return _values_equal(have, want)


def match_event(events: list[dict], sel: Selector) -> Hit | Miss:
    """First failing filter wins, so the hint lists the values that filter saw."""
    pool = list(events)
    filters: list[tuple[str, object, str]] = []
    if sel.kind is not None:
        filters.append(("kind", sel.kind, "eq"))
    filters.extend((key, val, "eq") for key, val in sel.equals)
    filters.extend((key, val, "contains") for key, val in sel.contains)
    for key, want, mode in filters:
        nxt = [ev for ev in pool if _field_ok(ev, key, want, mode)]
        if not nxt:
            return Miss(_filter_label(key, want, mode), _distinct(pool, key))
        pool = nxt
    if sel.nth > len(pool):
        return Miss(f"{_selector_label(sel)}#{sel.nth}", (str(len(pool)),))
    return Hit(pool[sel.nth - 1])


def parse_event_log(data: object, path: str, errors: list[str]) -> list[dict] | None:
    """Validate a decoded log. None means the file cannot be selected against."""
    if not isinstance(data, list):
        errors.append(_line(path, "need a list of events", 'use [{"t": 1.5, "kind": "state"}]'))
        return None
    rows: list[dict] = []
    bad = False
    for i, row in enumerate(data):
        where = f"{path}[{i}]"
        if not isinstance(row, dict):
            errors.append(_line(where, "need an object", 'use {"t": 1.5, "kind": "state"}'))
            bad = True
            continue
        if "t" not in row or event_time(row["t"]) is None:
            got = row.get("t", "missing")
            errors.append(_line(f"{where}.t", f"need seconds, got {got!r}", "use a finite number"))
            bad = True
            continue
        rows.append(dict(row))
    if bad:
        return None
    return rows


def load_events_file(path: Path, err_path: str, errors: list[str]) -> list[dict] | None:
    """Read one log. `err_path` is the path shown to the agent (no colon)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(_line(err_path, "invalid JSON", str(exc)))
        return None
    return parse_event_log(data, err_path, errors)


def normalise_events(events: list[dict], video_start: float) -> list[dict]:
    """Subtract the first video PTS so `t` is seconds from that frame.

    Director logs are seconds from record start. Audio and video start a few
    milliseconds apart; `-ss` seeks from the first video frame.
    """
    start = float(video_start)
    out: list[dict] = []
    for event in events:
        row = dict(event)
        row["t"] = float(event["t"]) - start
        out.append(row)
    return out


def dump_events(events: list[dict]) -> str:
    return json.dumps(events, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
