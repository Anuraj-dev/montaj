"""Talking-head cuts (#39): `montaj cut` plans spans from a transcript, `cuts: auto` plays them as jump cuts.

`plan_spans` turns word timings into the spans worth keeping: silence longer than `silence` and
filler words are cut out, and each edge is nudged to the quietest nearby 10 ms of the clip's audio so
a join lands where a cut cannot click. The plan is `assets/<clip>.cuts.json` (span numbers are 1-based,
the numbers `drop` takes). `expand_cuts` runs when the spec loads and replaces a `cuts: auto` shot with
ordinary `range` shots, so layout, the renderer and the segment cache need nothing new.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from montaj.spec import Audio, Shot, Spec, _assets_dir, _line

CUTS_VERSION = 2
SILENCE_S = 0.35  # a gap between words longer than this is cut
FILLERS_EN = frozenset("um umm ummm uh uhh uhm er erm err ah ahh hmm hm mm mmm".split())
FILLERS_HI = frozenset("अम्म अं आं उम हम्म हम्".split())
FILLERS = {"en": FILLERS_EN, "hi": FILLERS_EN | FILLERS_HI}
DEFAULT_PAD_S = (0.06, 0.12)  # kept before / after each span
PUNCH_ZOOM = 1.12
_SNAP_BACK_S = 0.10  # an edge may move this far out, onto a quieter frame
_SNAP_IN_S = 0.03  # ... or this far in
_ENERGY_HOP = 480  # 10 ms at 48 kHz
_PAUSE_REL_DB = 20.0  # a pause sits this far under the clip's speech level (p95) ...
_PAUSE_FLOOR_DB = -45.0  # ... but never needs to be quieter than this
_PAUSE_HYST_DB = 3.0  # a pause starts below the threshold and ends above threshold + this
_TIGHT_S = 0.02  # a retake needs this much speech to compare


def _norm(word: str) -> str:
    """Lowercase, edge punctuation off. Combining marks stay: a Devanagari word keeps its vowel signs."""
    word = word.lower()
    junk = {c for c in word if unicodedata.category(c)[0] in "PSZ" and c != "'"}
    return word.strip("".join(junk))


def is_filler(word: str, lang: str) -> bool:
    return _norm(word) in FILLERS.get(lang, FILLERS_EN)


def energy_db(audio: np.ndarray) -> np.ndarray:
    """RMS in dB of each 10 ms hop of a decoded `(n, 2)` clip."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    m = len(mono) // _ENERGY_HOP
    if m == 0:
        return np.zeros(0)
    power = (mono[: m * _ENERGY_HOP].astype(np.float64) ** 2).reshape(m, _ENERGY_HOP).mean(axis=1)
    return 10 * np.log10(np.maximum(power, 1e-12))


def _quietest(db: np.ndarray, lo: float, hi: float, prefer: float) -> float:
    """The quietest hop centre in `[lo, hi]` seconds; ties go to the one nearest `prefer`."""
    a = max(0, int(lo * 100))
    b = min(len(db), int(hi * 100) + 1)
    if b <= a:
        return prefer
    k = min(range(a, b), key=lambda j: (round(float(db[j]), 1), abs((j + 0.5) / 100 - prefer)))
    return (k + 0.5) / 100


def find_pauses(db: np.ndarray, silence: float) -> list[tuple[float, float]]:
    """`(start, end)` seconds of each run of at least `silence` below the clip-relative threshold.

    Whisper stretches words end to start, so a real pause hides inside a word; the audio shows it.
    Threshold: speech p95 minus 20 dB, floored at -45 dB; 3 dB hysteresis keeps a breathy tail in the pause.
    """
    if len(db) == 0:
        return []
    thr = max(float(np.percentile(db, 95)) - _PAUSE_REL_DB, _PAUSE_FLOOR_DB)
    out: list[tuple[float, float]] = []
    start: int | None = None
    for j, v in enumerate(db):
        if start is None:
            if v < thr:
                start = j
        elif v > thr + _PAUSE_HYST_DB:
            if (j - start) / 100 >= silence:
                out.append((start / 100, j / 100))
            start = None
    if start is not None and (len(db) - start) / 100 >= silence:
        out.append((start / 100, len(db) / 100))
    return out


def plan_spans(words: list[dict], lang: str = "hi", silence: float = SILENCE_S, db: np.ndarray | None = None,
               duration: float | None = None) -> tuple[list[dict], int]:
    """`(spans, fillers removed)`. A span is `{i, s, e, text}` (+ `retake: true`), seconds in the source."""
    groups: list[list[dict]] = []
    fillers = 0
    prev_end: float | None = None
    prev_mid = 0.0
    pauses = find_pauses(db, silence) if db is not None else []
    cut_here = True  # the next kept word opens a span
    for w in words:
        if is_filler(w["w"], lang):
            fillers += 1
            cut_here = True
            continue
        mid = (w["s"] + w["e"]) / 2  # a word straddling a pause belongs to the span holding its midpoint
        for ps, pe in pauses:
            if ps <= mid < pe:  # midpoint inside the pause: the word ends the span before it
                mid = ps
        paused = any(ps < mid and pe > prev_mid for ps, pe in pauses)
        if cut_here or prev_end is None or w["s"] - prev_end > silence or paused:
            groups.append([])
        groups[-1].append(w)
        prev_end = w["e"]
        prev_mid = mid
        cut_here = False
    spans = [{"s": g[0]["s"], "e": g[-1]["e"], "text": " ".join(x["w"] for x in g)} for g in groups]
    for sp in spans:  # an edge inside a pause moves to the pause edge
        for ps, pe in pauses:
            if ps <= sp["s"] < pe:
                sp["s"] = pe
            if ps < sp["e"] <= pe:
                sp["e"] = ps
    if db is not None and len(db):
        _snap(spans, db)
    if duration is not None:
        for sp in spans:
            sp["s"] = max(0.0, sp["s"])
            sp["e"] = min(duration, sp["e"])
    spans = [sp for sp in spans if sp["e"] - sp["s"] > _TIGHT_S]
    for i, sp in enumerate(spans, 1):
        sp["i"] = i
        sp["s"], sp["e"] = round(sp["s"], 3), round(sp["e"], 3)
    for a, b in zip(spans, spans[1:]):
        if _is_retake(a["text"], b["text"]):
            a["retake"] = True
    return [{"i": sp["i"], "s": sp["s"], "e": sp["e"], "text": sp["text"], **({"retake": True} if sp.get("retake") else {})}
            for sp in spans], fillers


def _snap(spans: list[dict], db: np.ndarray) -> None:
    """Move each edge to the quietest hop near it, never into a neighbouring span."""
    for k, sp in enumerate(spans):
        lo_bound = spans[k - 1]["e"] if k else 0.0
        hi_bound = spans[k + 1]["s"] if k + 1 < len(spans) else len(db) / 100
        s = _quietest(db, max(lo_bound, sp["s"] - _SNAP_BACK_S), sp["s"] + _SNAP_IN_S, sp["s"])
        e = _quietest(db, sp["e"] - _SNAP_IN_S, min(hi_bound, sp["e"] + _SNAP_BACK_S), sp["e"])
        if e - s > _TIGHT_S:
            sp["s"], sp["e"] = s, e


def _is_retake(a: str, b: str) -> bool:
    """`b` says `a` again: the same opening three words, or a close word-for-word match."""
    wa = [_norm(x) for x in a.split() if _norm(x)]
    wb = [_norm(x) for x in b.split() if _norm(x)]
    if min(len(wa), len(wb)) < 3:
        return False
    if wa[:3] == wb[:3]:
        return True
    return difflib.SequenceMatcher(None, wa, wb).ratio() >= 0.7


def cuts_path(clip: Path) -> Path:
    return clip.with_name(f"{clip.stem}.cuts.json")


def build(clip: Path, words_file: Path, *, lang: str = "hi", silence: float = SILENCE_S, log: Path | None = None) -> dict:
    """The plan for `clip` from its words file and its own audio energy."""
    from montaj.audio.mix import SR, decode

    data = json.loads(words_file.read_text(encoding="utf-8"))
    words = [{"w": str(w["w"]), "s": float(w["s"]), "e": float(w["e"])} for w in data["words"]]
    audio = decode(str(clip), 0, log=log)
    duration = len(audio) / SR
    spans, fillers = plan_spans(words, lang, silence, energy_db(audio), duration)
    kept = sum(sp["e"] - sp["s"] for sp in spans)
    return {"v": CUTS_VERSION, "clip": clip.name, "lang": lang, "silence": silence, "duration": round(duration, 3),
            "kept": round(kept, 3), "fillers": fillers, "spans": spans}


def read_plan(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("spans"), list):
        raise ValueError(f"{path.name} has no spans list")
    return data


def table(plan: dict) -> list[str]:
    """`span start end text` rows, one per span; a repeated opening is marked `retake?`."""
    out = ["span  start    end  text"]
    for sp in plan["spans"]:
        mark = "  retake?" if sp.get("retake") else ""
        out.append(f"{sp['i']:>4}  {sp['s']:6.2f}  {sp['e']:6.2f}  {sp['text']}{mark}")
    return [*out, f"OK {len(plan['spans'])} spans"]


# --- spec expansion ----------------------------------------------------------------------------------------


def expand_cuts(spec: Spec, spec_dir: Path | None, errors: list[str]) -> Spec:
    """Replace each `cuts: auto` shot by its kept spans as `range` shots. Errors leave the spec as it was."""
    if not any(sh.cuts for sh in spec.shots):
        return spec
    shots: list[Shot] = []
    renamed: dict[str, list[str]] = {}
    for i, sh in enumerate(spec.shots):
        if not sh.cuts:
            shots.append(sh)
            continue
        got = _expand_shot(spec, sh, i, spec_dir, errors)
        if got is None:
            shots.append(sh)
            continue
        shots.extend(got)
        if sh.id:
            renamed[sh.id] = [g.id for g in got if g.id]
    update: dict = {"shots": shots}
    if spec.audio is None and any(sh.audio for sh in shots):
        update["audio"] = Audio()  # linked clip audio mixes over silence when there is no music
    duck = spec.audio.duck if spec.audio is not None else None
    if duck is not None and any(u in renamed for u in duck.under):
        under = [x for u in duck.under for x in renamed.get(u, [u])]
        update["audio"] = spec.audio.model_copy(update={"duck": duck.model_copy(update={"under": under})})
    return spec.model_copy(update=update)


def _expand_shot(spec: Spec, sh: Shot, i: int, spec_dir: Path | None, errors: list[str]) -> list[Shot] | None:
    from montaj.project import clip_is_file, find_clip

    at = f"shots[{i}]"
    fps = spec.video.fps
    if sh.clip is None or clip_is_file(sh.clip):
        errors.append(_line(f"{at}.cuts", "cuts needs a clip in assets/", "use clip: <stem> and run montaj cut <stem>"))
        return None
    for name in ("range", "hold", "until", "clip_in", "speed", "freeze", "at_event"):
        if getattr(sh, name) is not None:
            errors.append(_line(f"{at}.{name}", f"{name} cannot combine with cuts", "the spans set the range"))
            return None
    assets = _assets_dir(spec, spec_dir)
    if assets is None or fps <= 0:
        errors.append(_line(f"{at}.cuts", "cuts needs the spec loaded from disk", "load the spec file"))
        return None
    try:
        file = find_clip(assets, sh.clip)
    except FileNotFoundError:
        return None  # the missing clip is its own error
    plan_file = cuts_path(file)
    if not plan_file.is_file():
        errors.append(_line(f"{at}.cuts", f"no {plan_file.name}", f"run montaj cut {sh.clip}"))
        return None
    try:
        plan = read_plan(plan_file)
    except (OSError, ValueError) as exc:
        errors.append(_line(f"{at}.cuts", f"cannot read {plan_file.name} ({exc})", f"run montaj cut {sh.clip}"))
        return None
    spans = plan["spans"]
    drop = set(sh.drop or ())
    bad = sorted(d for d in drop if not 1 <= d <= len(spans))
    if bad:
        errors.append(_line(f"{at}.drop", f"no span {bad[0]}", f"{sh.clip} has spans 1..{len(spans)}"))
        return None
    pad = _pad(sh, fps, spec.video.bpm, at, errors)
    if pad is None:
        return None
    total = float(plan.get("duration") or 0) or None
    keep = [k for k, sp in enumerate(spans) if sp["i"] not in drop]
    if not keep:
        errors.append(_line(f"{at}.drop", "drops every span", "keep at least one"))
        return None
    ranges: list[tuple[int, int]] = []
    for k in keep:
        s, e = _padded(spans, k, pad, total)
        s_f, e_f = round(s * fps), round(e * fps)
        if ranges and keep[len(ranges) - 1] + 1 == k:  # source-adjacent: never replay a frame
            s_f = max(s_f, ranges[-1][1])
        ranges.append((s_f, max(e_f, s_f + 1)))
    base = {k: v for k, v in sh.model_dump(by_alias=True, exclude_none=True).items()
            if k not in ("cuts", "cuts_pad", "cuts_punch", "drop")}
    out: list[Shot] = []
    for n, (s_f, e_f) in enumerate(ranges):
        d = dict(base)
        d.update(clip=sh.clip, range=(f"{s_f}f", f"{e_f}f"))
        d.setdefault("audio", True)  # the point of a talking cut is the voice; `audio: false` opts out
        if n:
            for name in ("in", "flash", "audio_lead"):
                d.pop(name, None)
        if n < len(ranges) - 1:
            d.pop("audio_tail", None)
        if sh.id:
            d["id"] = sh.id if n == 0 else f"{sh.id}_{n + 1}"
        if sh.cuts_punch == "alternate" and n % 2:
            d["zoom"] = round(float(sh.zoom or 1.0) * PUNCH_ZOOM, 4)
        out.append(Shot.model_validate(d))
    return out


def _pad(sh: Shot, fps: int, bpm, at: str, errors: list[str]) -> tuple[float, float] | None:
    if sh.cuts_pad is None:
        return DEFAULT_PAD_S
    from montaj.audio.mix import _span_s

    got = [_span_s(raw, fps, bpm, f"{at}.cuts_pad[{j}]", errors) for j, raw in enumerate(sh.cuts_pad)]
    if None in got:
        return None
    return got[0], got[1]


def _padded(spans: list[dict], k: int, pad: tuple[float, float], total: float | None) -> tuple[float, float]:
    """Span `k` widened by `pad`, but never past halfway to its source neighbours (a cut stays cut)."""
    s, e = spans[k]["s"], spans[k]["e"]
    lo = (spans[k - 1]["e"] + s) / 2 if k else 0.0
    hi = (e + spans[k + 1]["s"]) / 2 if k + 1 < len(spans) else (total if total is not None else e + pad[1])
    return max(lo, s - pad[0]), min(hi, e + pad[1])
