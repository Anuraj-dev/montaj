"""`montaj assets [STEM]`: what is in each clip, in a few lines, from what `montaj ingest` wrote (#62).

The agent learns its footage here instead of reading `analysis.json`, the event logs or probing with ffmpeg:
duration and size, scene cuts, speech spans with their first words (`<stem>.words.json`), the events an
`at_event` can pick (`<stem>.events.json`, repeats folded), and loudness (decoded once, not stored by ingest).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from montaj.analysis import fresh, load_analysis, media

SPEECH_CAP, EVENT_CAP = 6, 12  # per clip without a stem; a stem shows everything
TEXT_CAP = 40  # characters of an event's text field
FIRST_WORDS = 4


def _short(v: object) -> str:
    s = str(v)
    s = s if len(s) <= TEXT_CAP else s[: TEXT_CAP - 1] + "…"
    return json.dumps(s, ensure_ascii=False) if (" " in s or not s) else s


def _events(path: Path) -> list[str]:
    """`t kind field=value …` for each event that differs from the last of its kind."""
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    last: dict[str, tuple] = {}
    out = []
    for r in rows if isinstance(rows, list) else ():
        if not isinstance(r, dict) or "t" not in r:
            continue
        kind = str(r.get("kind", "?"))
        fields = tuple((k, v) for k, v in r.items() if k not in ("t", "kind") and v not in (None, ""))
        if last.get(kind) == fields:
            continue
        last[kind] = fields
        body = " ".join(f"{k}={_short(v)}" for k, v in fields)
        out.append(f"{float(r['t']):.2f} {kind}" + (f" {body}" if body else ""))
    return out


def _words(path: Path) -> list[tuple[float, str]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows = data.get("words", []) if isinstance(data, dict) else data
    return [(float(w["s"]), str(w["w"])) for w in rows if isinstance(w, dict) and "s" in w and "w" in w]


def _speech(row: dict, words: list[tuple[float, str]]) -> list[str]:
    out = []
    for sp in row.get("speech") or ():
        s, e = float(sp["s"]), float(sp["e"])
        said = [w for t, w in words if s - 0.05 <= t < e][:FIRST_WORDS]
        out.append(f"{s:.1f}-{e:.1f}" + (f' "{" ".join(said)}…"' if said else ""))
    return out


def _loudness(path: Path) -> str:
    """`loud -24dBFS peak -2dBFS@12.4s` over the first audio stream; `no audio` when there is none."""
    from montaj.audio.mix import SR, decode
    from montaj.encode import EncodeError

    try:
        pcm = decode(str(path)).mean(axis=1)
    except (EncodeError, OSError):
        return "no audio"
    if not len(pcm):
        return "no audio"
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    i = int(np.argmax(np.abs(pcm)))
    peak = float(abs(pcm[i]))
    db = lambda x: f"{20 * np.log10(x):.0f}dBFS" if x > 1e-9 else "silent"  # noqa: E731
    return f"loud {db(rms)} peak {db(peak)}@{i / SR:.1f}s"


def _more(items: list[str], cap: int | None, stem: str) -> list[str]:
    if cap is None or len(items) <= cap:
        return items
    return [*items[:cap], f"+{len(items) - cap} more (montaj assets {stem})"]


def summary(assets_dir: Path, stem: str | None = None) -> list[str]:
    """Result lines for each analysed clip (or the one named `stem`), then `OK …` / `ERR …`."""
    rows = load_analysis(assets_dir)
    clips = [p for p, kind in media(assets_dir) if kind == "clip" and (stem is None or p.stem == stem)]
    if not clips:
        what = f"no clip {stem!r}" if stem else "no clips"
        return [f"ERR assets: {what} in {assets_dir} (montaj ingest --list shows every asset)"]
    cap = (SPEECH_CAP, EVENT_CAP) if stem is None else (None, None)
    out: list[str] = []
    for p in clips:
        row = rows.get(p.name)
        if not fresh(row, p):
            out.append(f"{p.stem} not analysed (run montaj ingest)")
            continue
        cuts = " ".join(f"{c:.1f}" for c in row.get("cuts") or ()) or "none"
        out.append(f"{p.stem} {row.get('duration', 0):g}s {row.get('w', 0)}x{row.get('h', 0)} "
                   f"{row.get('fps', 0):g}fps cuts {cuts} | {_loudness(p)}")
        speech = _speech(row, _words(p.with_name(f"{p.stem}.words.json")))
        if speech:
            out.append("  speech " + " | ".join(_more(speech, cap[0], p.stem)))
        events = _events(p.with_name(f"{p.stem}.events.json"))
        if events:
            out.append("  events " + " | ".join(_more(events, cap[1], p.stem)))
    out.append(f"OK assets: {len(clips)} clip{'s' if len(clips) != 1 else ''}")
    return out
