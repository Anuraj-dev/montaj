"""`montaj spec`: topic-scoped views of docs/SPEC-REFERENCE.md, generated from it so the two cannot drift.

Every bullet of the reference lands in exactly one topic: by the section it sits in (transitions, looks,
capture, ...) or, in the main field list, by the first rule in `_RULES` that matches its key. A bullet no
rule claims goes to `other`; tests/test_specref.py fails while `other` is non-empty.
"""

from __future__ import annotations

import re
from pathlib import Path

_DOC = "SPEC-REFERENCE.md"

# topic -> one-line summary, in listing order.
TOPICS: dict[str, str] = {
    "units": "durations, beats, markers, time expressions, photo vs design pixels",
    "video": "size, fps, bpm, beat grid, background, intro/outro, motion blur",
    "shots": "what a shot is: source, range, speed, hold, tone, frame, morph, card, sequence",
    "camera": "per-shot zoom/focus keys and drift",
    "cuts": "cut a clip on its transcript: cuts, pad, punch, drop",
    "transitions": "shots[].in: fade, swirl, whip, glitch, push, dip, ...",
    "effects": "per-shot effects, fx hits/leaks/bursts, flash",
    "layers": "clips and images stacked on a shot: position, scale, blend, mask, parenting",
    "keys": "keyframes on layer properties (t, ease) and easing curves",
    "wall": "photo wall: grid, prints, light, camera",
    "text": "text blocks and lines: style, reveal, in/out, backdrop, lower thirds, subs",
    "captions": "word-timed captions: source, preset, accent, emphasis, overrides",
    "emphasis": "word-anchored zoom, shake, flash and sfx",
    "audio": "track, markers, edit, clips, sfx, ducking, loudness",
    "grade": "looks, LUTs, colour pipeline, per-shot grade, tone, curves, scopes, shot matching",
    "overlays": "shapes, bars, sweeps and the tracks that drive them",
    "reuse": "styles, defaults, components, use, include",
    "check": "montaj check expectations: black/frozen, safe, loudness, pacing, content guards",
    "custom": "your own draw(ctx) shots, effects and transitions",
    "capture": "montaj capture scripts: record a web page to a clip",
    "inspect": "montaj inspect --track",
    "ingest": "montaj ingest, mattes, event logs of screen recordings",
    "gpu": "strict dGPU mode (machine config)",
    "export": "delivery codecs and export targets",
    "messages": "what each check/validate WARN means and how to fix or accept it",
    "other": "",
}

# Labelled sub-lists and `##` headings that open their own topic (matched on the start of the label).
_SECTIONS: dict[str, str] = {
    "transitions": "transitions",
    "custom code": "custom",
    "looks": "grade",
    "colour scopes": "grade",
    "easing": "keys",
    "capture": "capture",
    "content guards": "check",
    "reuse": "reuse",
    "inspecting a frame": "inspect",
    "asset analysis": "ingest",
    "event logs": "ingest",
    "strict dgpu": "gpu",
    "delivery codecs": "export",
    "check and validate messages": "messages",
}

# First match wins. Matched against the first backticked token of a main-list bullet.
_RULES: list[tuple[str, str]] = [
    (r"video\.(look|color)", "grade"),
    (r"video\b", "video"),
    (r"(assets|markers)$", "video"),
    (r"shots\[\]\.(grade|tone)\b|grades$", "grade"),
    (r"shots\[\]\.layers\[\]\.(keys|<prop>)", "keys"),
    (r"shots\[\]\.layers", "layers"),
    (r"shots\[\]\.(camera|drift)\b", "camera"),
    (r"shots\[\]\.(cuts|cuts_pad|cuts_punch|drop)$", "cuts"),
    (r"shots\[\]\.(audio|gain|audio_lead|audio_tail)$", "audio"),
    (r"shots\[\]\.(effects|flash)\b|fx\b", "effects"),
    (r"shots\[\]\.(custom|params)$", "custom"),
    (r"shots\[\]\.in\b", "transitions"),
    (r"shots\[\]\.wall$|wall\.", "wall"),
    (r"shots(\[\]\.|$)", "shots"),
    (r"audio\b", "audio"),
    (r"(text|subs)\b|validate$", "text"),
    (r"captions\b", "captions"),
    (r"emphasis\b", "emphasis"),
    (r"(tracks|overlays)\b", "overlays"),
    (r"(styles|defaults|components|use|include)\b", "reuse"),
    (r"check\b", "check"),
]


def doc_path() -> Path:
    """The reference: bundled into the wheel beside this module, else the checkout's docs/."""
    here = Path(__file__).resolve().parent
    for p in (here / _DOC, here.parent / "docs" / _DOC):
        if p.is_file():
            return p
    raise FileNotFoundError(f"{_DOC} not found beside the montaj package or in docs/")


def _section_topic(label: str) -> str | None:
    low = label.lower()
    for head, topic in _SECTIONS.items():
        if low.startswith(head):
            return topic
    return None


def _rule_topic(key: str) -> str:
    for pat, topic in _RULES:
        if re.match(pat, key):
            return topic
    return "other"


def parse(text: str) -> dict[str, list[str]]:
    """Topic -> its lines (bullets and the prose that introduces them), in document order.

    Prose in the main list attaches to the next bullet's topic; the opening paragraph is `units`.
    Inside a labelled section or `##` heading everything belongs to that section's topic.
    """
    out: dict[str, list[str]] = {t: [] for t in TOPICS}
    section: str | None = None
    pending: list[str] = []
    first = True
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.startswith("# "):
            continue
        label = None
        if line.startswith("## "):
            label = line[3:]
        elif re.match(r"^[A-Z][^`\n]*(\(.*\))?:$", line) or re.match(r"^Reuse \(", line):
            label = line
        if label is not None:
            topic = _section_topic(label)
            if topic is not None or not line.startswith("## "):
                section = topic or re.sub(r"\W+", "-", label.lower()).strip("-")
                out.setdefault(section, [])
                out[section].append(line)
            continue  # a `##` heading in the main list (Emphasis) only introduces the prose after it
        if line.startswith("- "):
            m = re.match(r"- `([^`]+)`", line)
            topic = section or (_rule_topic(m.group(1)) if m else "other")
            out.setdefault(topic, []).extend(pending)
            out[topic].append(line)
            pending = []
        elif section:
            out[section].append(line)
        elif first:
            out["units"].append(line)
        else:
            pending.append(line)
        first = False
    out["other"].extend(pending)
    return out


def topics() -> dict[str, list[str]]:
    return parse(doc_path().read_text(encoding="utf-8"))


def _fields(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.startswith("- ")]


EXAMPLES: dict[str, str] = {
    "video": "video: {size: 1080x1920, fps: 30, look: clean}",
    "shots": "shots:\n  - {photo: a.jpg, hold: 2s, in: {type: fade, dur: 8f}}",
    "camera": "shots:\n  - clip: take\n    camera: [{t: 0s, zoom: 1.0}, {t: 2s, zoom: 1.2, ease: outc}]",
    "cuts": "shots:\n  - {clip: voice, cuts: auto}",
    "transitions": "shots:\n  - {photo: b.jpg, hold: 2s, in: {type: whip, dur: 10f, axis: x}}",
    "text": ("text:   # one line per title; styles/pos from defaults: {text.lines: ...}\n"
             "  - {at: 4b, to: 6b, text: \"SWITCH *WORKSPACES*\"}\n"
             "  - {from: 8b, to: 12b, lines: [{text: Hello, in: {type: rise, dur: 12f}}, {text: there, at: 9b}]}"),
    "captions": "captions: {source: voice, preset: punch, accent: '#ffe600'}",
    "emphasis": "emphasis:\n  - {match: launch, zoom: 1.15, shake: 20}",
    "audio": "audio: {track: song.mp3, fade_out: 2s}",
    "grade": "shots:\n  - {photo: a.jpg, hold: 2s, grade: {exposure: 0.3, temp: 200}}",
    "layers": "shots:\n  - blank: true\n    hold: 3s\n    layers: [{image: logo.png, pos: [540, 960], scale: 0.5}]",
    "keys": "layers: [{image: logo.png, opacity: [{t: 0s, v: 0}, {t: 12f, v: 1, ease: outc}]}]",
    "effects": "shots:\n  - {photo: a.jpg, hold: 2s, effects: [{type: shake, at: 1s, dur: 6f}]}",
    "wall": "shots:\n  - {wall: {grid: 300x300, prints: [{photo: a.jpg, cell: [0, 0]}]}, hold: 3s}",
    "overlays": "overlays:\n  - {shape: ring, from: 0s, to: 2s, center: [540, 960], radius: 200}",
    "reuse": "styles: {big: {size: 96, weight: 700}}   # then style: big on a line",
    "check": "check: {pacing: {max_shot: 4s}, loudness: {i: -14}}",
}


SHORT_CAP = 120  # characters of a line in the short form


_SENTENCE_END = re.compile(r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<=\.)\s(?=[A-Z`(])")  # "e.g." / "i.e." end none


def _first_sentence(text: str) -> str:
    m = _SENTENCE_END.search(text)
    return text[: m.start()] if m else text


def short_line(line: str) -> str:
    """A bullet as key, type/default and the first sentence of its meaning; prose as its first sentence."""
    if line.startswith("- "):
        parts = line.split(" — ", 2)
        line = " — ".join([*parts[:2], *(_first_sentence(parts[2]) for _ in parts[2:3])])
    else:
        line = _first_sentence(line)
    return line if len(line) <= SHORT_CAP else line[: SHORT_CAP - 1] + "…"


def render_topic(name: str, parsed: dict[str, list[str]], full: bool = True) -> list[str]:
    lines = parsed[name]
    if not full and _fields(lines):  # a topic of prose alone (ingest, gpu) is already short: print it whole
        lines = [short_line(ln) for ln in lines]
    ex = EXAMPLES.get(name)
    return lines + (["", "Example:", *("  " + ln for ln in ex.splitlines())] if ex else [])


def search(word: str, parsed: dict[str, list[str]]) -> list[str]:
    w = word.lower()
    return [ln for ls in parsed.values() for ln in _fields(ls) if w in ln.lower()]


def listing(parsed: dict[str, list[str]]) -> list[str]:
    w = max(len(t) for t in parsed)
    return [f"{t.ljust(w)}  {TOPICS.get(t, '')}{f' ({n})' if (n := len(_fields(ls))) else ''}" for t, ls in parsed.items() if ls]


def run(topic: str | None, find: str | None, full: bool = False) -> list[str]:
    """Lines for `montaj spec [TOPIC [--full] | --search WORD]`, ending in an ERR line when nothing matches.

    A topic prints short by default (each field's key, type, default and first sentence: #62); `--full` and
    `--search` print whole reference lines.
    """
    parsed = topics()
    if find is not None:
        hits = search(find, parsed)
        if not hits:
            return [f"ERR spec: no field matches {find!r} (try a shorter word, or montaj spec lists the topics)"]
        return hits + [f"OK spec --search {find}: {len(hits)} fields"]
    if topic is None:
        return listing(parsed) + [f"OK spec: {len([t for t, ls in parsed.items() if ls])} topics; montaj spec <topic> prints one, --search <word> finds a field"]
    key = topic.lower()
    if key not in parsed or not parsed[key]:
        import difflib

        near = difflib.get_close_matches(key, [t for t, ls in parsed.items() if ls], n=1)
        hint = f"did you mean {near[0]!r}?" if near else "montaj spec lists the topics"
        return [f"ERR spec: unknown topic {topic!r} ({hint})"]
    n = len(_fields(parsed[key]))
    tail = "" if full or not n else "; short form, --full for every detail"
    return render_topic(key, parsed, full) + [f"OK spec {key}: {n} fields{tail}"]
