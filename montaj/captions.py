"""`captions:` block -> ordinary gpu `text` blocks (issue #23, design contract in docs/CAPTIONS.md).

`expand_captions` runs at load time (`spec.load_spec`, and `timeline.resolve` for a spec built in code):
word timings are chunked deterministically and every chunk becomes one or more `text[]` blocks appended
after the author's own. The renderer and the segment hash see only that expansion (ROADMAP "cache
contract holds"); nothing here is read at render time. Pure functions of the spec and the words file.

Chunking: a pause >= 0.3 s, sentence punctuation and a `chunk_break_before` always break. Between those,
the cut points are chosen together (a small dynamic programme) so that every chunk keeps the max word count,
fits the line budget (the HarfBuzz width at the preset's size, via `glyphs.measure`, inside the caption's
safe band) and never runs past a comma once it holds `words_per_chunk[0]` words; among those partitions the
fewest chunks over RATE_MAX_CPS win, then the fewest chunks, then the most even on-screen times.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from montaj.spec import Captions, Spec, SpecError, TextBlock, _line

PAUSE_S = 0.3  # a gap this long between words always starts a new chunk
CLEAR_S = 0.6  # a gap longer than this clears the caption instead of holding the last chunk
HOLD_S = 0.3  # how long the last chunk of a phrase stays after its last word
HARD_END = ".!?…।"
SOFT_END = ",;:"
DROP_END = ".,;:।"  # trailing marks removed from the caption text; `?`, `!` and `…` stay
PRESETS = ("punch", "karaoke", "highlight", "clean", "comic", "minimal")
DEVANAGARI = re.compile("[ऀ-ॿ]")
NEGATIONS = frozenset(
    "no not never none nothing nobody nowhere neither nor cannot can't won't don't doesn't didn't isn't "
    "aren't wasn't weren't haven't hasn't hadn't shouldn't couldn't wouldn't nahi nahin na mat".split()
)
STOPWORDS = frozenset(
    "a an the and or but if so of to in on at by for with from as is are was were be been it its this that "
    "these those i you he she we they me my your our their his her them us do does did have has had will "
    "would can could should just very really more most than then there here what when who how why".split()
)
SQUARE_ASPECT = 1.15  # width/height under this counts as portrait-ish for the default y (and the 9:16 zones)
RATE_MAX_CPS = 20.0  # characters per second a reader can follow (docs/CAPTIONS.md); `check` lints the same
# Platform UI on 9:16 (top bar, caption/title block, the right-hand button column) and on 16:9 (the seek bar).
ZONES_PORTRAIT = {"top": 0.12, "bottom": 0.22, "right": 0.14, "left": 0.0}
ZONES_LANDSCAPE = {"top": 0.0, "bottom": 0.08, "right": 0.0, "left": 0.0}
EDGE_MARGIN = 0.05  # every side keeps at least this fraction of the frame clear (timeline.SLAM_MARGIN)


@dataclass(frozen=True)
class Preset:
    """The look of one preset. Sizes are design px of a 1080-wide portrait frame."""

    size: float
    font: str | None = None
    style: str = "serif"
    weight: float | None = None
    case: str = "upper"
    stroke: float = 0.0  # fraction of the size
    shadow: tuple[float, float, float] = (4.0, 12.0, 0.6)  # y, blur, opacity
    mode: str = "chunk"  # chunk: one block per chunk | word: one block per spoken word
    enter: str | None = "pop"  # `in` type of the first block of a chunk (None: appears whole)
    emph_scale: float = 1.0
    emph_in: str | None = None
    emph_shake: float | None = None
    tilt: float = 0.0
    bar: bool = False
    serif_emphasis: bool = False


PRESET_DEFS = {
    "punch": Preset(size=100, font="montserrat-black", stroke=0.08, shadow=(6.0, 12.0, 0.6), emph_scale=1.15,
                    emph_in="slam", emph_shake=6.0),
    "karaoke": Preset(size=92, font="montserrat-black", stroke=0.08, shadow=(6.0, 12.0, 0.6), mode="word"),
    "highlight": Preset(size=78, font="poppins-extrabold", case="as-is", shadow=(4.0, 10.0, 0.5), mode="word",
                        enter=None, bar=True),  # no entry: the bar and the text land on the same frame
    "clean": Preset(size=80, style="sans", weight=600, case="lower", shadow=(4.0, 14.0, 0.6), enter="scale-in"),
    "comic": Preset(size=112, font="bangers", stroke=0.11, shadow=(6.0, 10.0, 0.6), enter="drop", emph_scale=1.12,
                    tilt=3.0),
    "minimal": Preset(size=68, style="sans", weight=400, case="lower", shadow=(3.0, 12.0, 0.55), enter="scale-in",
                      serif_emphasis=True),
}


@dataclass(frozen=True)
class Word:
    i: int
    w: str
    s: float
    e: float
    emph: bool = False


@dataclass(frozen=True)
class Chunk:
    """A run of words shown together. Times are frames; `emph` is the accented word's index in `words`."""

    index: int
    words: tuple[Word, ...]
    start: int
    end: int
    starts: tuple[int, ...]  # frame each word is spoken (strictly increasing, within [start, end))
    emph: int | None

    @property
    def text(self) -> str:
        return " ".join(w.w for w in self.words)


# --- words ------------------------------------------------------------------------------------------------


def read_words(path: Path, errors: list[str], where: str = "captions.source") -> list[Word]:
    """`[{i, w, s, e}]` from a transcript or markers JSON (`{words: [...]}` or a bare list)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(_line(where, f"cannot read {path.name}", str(exc)))
        return []
    rows = data.get("words") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        errors.append(_line(where, f"{path.name} has no words list", "use {words: [{i, w, s, e}, ...]}"))
        return []
    out: list[Word] = []
    for n, row in enumerate(rows):
        try:
            s, e = float(row["s"]), float(row["e"])
            w = str(row["w"]).strip()
            i = int(row.get("i", n + 1))
        except (KeyError, TypeError, ValueError, AttributeError):
            errors.append(_line(where, f"words[{n}] needs w, s and e", "each word is {i, w, s, e}"))
            return []
        if not (math.isfinite(s) and math.isfinite(e)) or e < s:
            errors.append(_line(where, f"words[{n}] has bad times", "s and e are finite seconds, e >= s"))
            return []
        if w:
            out.append(Word(i=i, w=w, s=s, e=e))
    return out


def _source_path(spec: Spec, spec_dir: Path, errors: list[str]) -> Path | None:
    cap = spec.captions
    assert cap is not None
    if cap.source == "audio.markers":
        if spec.audio is None or not spec.audio.markers:
            errors.append(_line("captions.source", "audio.markers is not set", "set audio.markers, or name a words file"))
            return None
        return spec_dir / spec.audio.markers
    for cand in (spec_dir / cap.source, spec_dir / spec.assets / f"{cap.source}.words.json",
                 spec_dir / "assets" / f"{cap.source}.words.json"):
        if cand.is_file():
            return cand
    errors.append(_line("captions.source", f"{cap.source!r} not found",
                        "audio.markers, a words JSON beside the spec, or a clip id transcribed to assets/<id>.words.json"))
    return None


def drop_end(text: str) -> str:
    """`text` without trailing `.`, `,`, `;`, `:`, `।`. `?`, `!`, `…` and a `..` ellipsis stay."""
    w = text.strip()
    while len(w) > 1 and w[-1] in DROP_END and not w.endswith(".."):
        w = w[:-1].rstrip()
    return w


def normalise(raw: str, keep: bool = False) -> str:
    """Trailing `.`, `,` (and `;:।`) dropped unless `keep`; `*` is escaped for the text engine."""
    w = raw.strip() if keep else drop_end(raw)
    return w.replace("*", "\\*")


def normalise_subs(spec: Spec) -> Spec:
    """`spec` with the trailing `.`/`,` dropped from every sub marked `punctuation: drop`."""
    subs = [s if s.punctuation != "drop" or drop_end(s.text) == s.text else s.model_copy(update={"text": drop_end(s.text)})
            for s in spec.subs]
    return spec if all(a is b for a, b in zip(subs, spec.subs)) else spec.model_copy(update={"subs": subs})


def _punct(raw: str) -> str:
    return "hard" if raw and raw[-1] in HARD_END else "soft" if raw and raw[-1] in SOFT_END else ""


# --- layout numbers ----------------------------------------------------------------------------------------


def _frame_size(spec: Spec) -> tuple[int, int]:
    w, h = (int(x) for x in spec.video.size.lower().split("x"))
    return w, h


def default_y(spec: Spec) -> float:
    """Design-px centre of the caption: the 9:16 safe zone (60-72 % of the height), lower third otherwise."""
    from montaj.render.glyphs import DESIGN_W

    w, h = _frame_size(spec)
    design_h = DESIGN_W * h / w
    return round(design_h * (0.646 if w / h < SQUARE_ASPECT else 0.80))


def safe_zones(spec: Spec) -> dict[str, float]:
    """Fractions of the frame a caption keeps clear of, per side: the aspect's platform zones with any
    `check.safe` side replacing its default."""
    w, h = _frame_size(spec)
    zones = dict(ZONES_PORTRAIT if w / h < SQUARE_ASPECT else ZONES_LANDSCAPE)
    safe = spec.check.safe if spec.check is not None else None
    if safe is not None:
        for side in zones:
            v = getattr(safe, side)
            if v is not None:
                zones[side] = float(v)
    return zones


def safe_band(spec: Spec) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) design px every caption's ink stays inside, stroke and entry motion included: the
    safe zones, and at least EDGE_MARGIN from each frame edge."""
    from montaj.render.glyphs import DESIGN_W

    w, h = _frame_size(spec)
    W, H = float(DESIGN_W), DESIGN_W * h / w
    z = safe_zones(spec)
    return (max(z["left"], EDGE_MARGIN) * W, max(z["top"], EDGE_MARGIN) * H,
            W - max(z["right"], EDGE_MARGIN) * W, H - max(z["bottom"], EDGE_MARGIN) * H)


def _budget(spec: Spec, preset: Preset) -> float:
    """Widest painted chunk (design px, stroke included) at rest: the band, less the room its entry
    overshoot grows into (a pop scales each word up to 1.08 about its own centre)."""
    from montaj.render import motion as M

    x0, _y0, x1, _y1 = safe_band(spec)
    grow = max(1.0, M.IN_OVERSHOOT.get(preset.enter, 1.0)) if preset.enter else 1.0
    return (x1 - x0) / grow


def _painted_width(book, text: str, preset: Preset, size: float, font, case) -> float:
    """Twice the farthest the painted ink reaches from the line's centre: glyph outlines (which can overhang
    their advance box), the stroke around them, turned by the preset's tilt. `text` may carry `*emphasis*`."""
    from montaj.render import glyphs

    run = glyphs.layout_line(book, text, style=preset.style, y=0.0, size=size, weight=preset.weight, font=font,
                             case=case, emphasis_scale=preset.emph_scale)
    x, _y, w, h = run.box
    cx = x + w / 2
    stroke = size * preset.stroke
    reach = max([w / 2] + [max(cx - g.ink[0], g.ink[2] - cx) for g in run.glyphs if g.ink[2] > g.ink[0]]) + stroke
    r = math.radians(preset.tilt)
    return 2 * (reach * abs(math.cos(r)) + (h / 2 + stroke) * abs(math.sin(r)))


def _fit_width(book, texts: list[str], preset: Preset, size: float, font, case, budget: float) -> float:
    """The `max_width` for the line's `fit` (which measures the advance box plus stroke) that keeps every one
    of `texts` painting within `budget`. Ink overhang scales with the size, so one ratio holds at any size."""
    from montaj.render import glyphs

    out = budget
    for t in texts:
        boxed = glyphs.measure(book, t, style=preset.style, size=size, weight=preset.weight, font=font, case=case,
                               emphasis_scale=preset.emph_scale, stroke=size * preset.stroke, rotate=preset.tilt)[1]
        out = min(out, budget * boxed / max(_painted_width(book, t, preset, size, font, case), boxed))
    return math.floor(out * 10) / 10


def _centre_x(spec: Spec) -> float:
    x0, _y0, x1, _y1 = safe_band(spec)
    return round((x0 + x1) / 2, 1)


def _size_for(spec: Spec, preset: Preset) -> float:
    w, h = _frame_size(spec)
    return round(preset.size * (1.0 if h >= w else 0.72))


def _font_for(preset: Preset, words: list[Word]) -> tuple[str | None, str]:
    """Devanagari captions use the Mukta face and keep their case."""
    if any(DEVANAGARI.search(w.w) for w in words):
        return "mukta-extrabold", "as-is"
    return preset.font, preset.case


# --- chunking ---------------------------------------------------------------------------------------------


def _segments(words: list[Word], raw: dict[int, str], breaks: set[int]) -> list[list[Word]]:
    """Runs of words between the cuts nothing may cross: a pause, sentence punctuation, an override."""
    segs: list[list[Word]] = []
    for w in words:
        if segs and not (w.i in breaks or w.s - segs[-1][-1].e >= PAUSE_S or _punct(raw[segs[-1][-1].i]) == "hard"):
            segs[-1].append(w)
        else:
            segs.append([w])
    return segs


def _tail_frame(seg: list[Word], nxt: list[Word] | None, fps: int) -> int:
    """Frame the segment's last chunk leaves on: the next segment's first word, or a hold after a long pause."""
    last = seg[-1]
    if nxt is None or nxt[0].s - last.e > CLEAR_S:
        end = round((last.e + HOLD_S) * fps)
        return min(end, round(nxt[0].s * fps)) if nxt is not None else end
    return round(nxt[0].s * fps)


def _chunk_words(words: list[Word], raw: dict[int, str], lo: int, hi: int, breaks: set[int], width,
                 budget: float, fps: int) -> list[list[Word]]:
    """Split into chunks (module docstring). `width(words)` is the painted width of a chunk in design px.

    A chunk is on screen from its first word's frame to the next chunk's first frame, so where a segment is
    cut decides each chunk's reading rate: a fast word joins a slower neighbour instead of flashing past.
    """
    segs = _segments(words, raw, breaks)
    out: list[list[Word]] = []
    for n_seg, seg in enumerate(segs):
        n = len(seg)
        frames = [round(w.s * fps) for w in seg] + [_tail_frame(seg, segs[n_seg + 1] if n_seg + 1 < len(segs) else None, fps)]
        # best[b]: (cost, start of the last chunk) of the best split of seg[:b]
        best: list[tuple[tuple, int] | None] = [None] * (n + 1)
        best[0] = ((0, 0.0, 0, 0.0), 0)
        for b in range(1, n + 1):
            for a in range(max(0, b - hi), b):
                if best[a] is None:
                    continue
                group = seg[a:b]
                # A comma inside the chunk is allowed only before it holds `lo` words.
                if any(_punct(raw[w.i]) == "soft" and k + 1 >= lo for k, w in enumerate(group[:-1])):
                    continue
                if b - a > 1 and width(group) > budget:
                    continue  # a single word too wide for the band is shrunk by `fit` instead
                secs = max(frames[b] - frames[a], 1) / fps
                cps = sum(len(w.w) for w in group) / secs
                over = cps > RATE_MAX_CPS + 1e-9
                c0 = best[a][0]
                cost = (c0[0] + over, c0[1] + max(0.0, cps - RATE_MAX_CPS), c0[2] + 1, c0[3] + secs * secs)
                if best[b] is None or cost < best[b][0]:
                    best[b] = (cost, a)
        cuts, b = [], n
        while b > 0:
            a = best[b][1]
            cuts.append(seg[a:b])
            b = a
        out.extend(reversed(cuts))
    return out


def _auto_emphasis(chunks: list[list[Word]]) -> list[int | None]:
    """Per chunk, the word to accent: a number or a negation, else a long content word (never in two chunks
    running, and not in a one-word chunk: an accent on every chunk is no accent)."""
    out: list[int | None] = []
    prev_plain = False
    for chunk in chunks:
        pick: int | None = None
        for k, w in enumerate(chunk):
            if any(c.isdigit() for c in w.w):
                pick = k
                break
        if pick is None:
            for k, w in enumerate(chunk):
                if w.w.lower().strip("?!…") in NEGATIONS:
                    pick = k
                    break
        strong = pick is not None
        if pick is None and len(chunk) >= 2 and not prev_plain:
            content = [(len(w.w), -k) for k, w in enumerate(chunk) if w.w.lower() not in STOPWORDS and len(w.w) >= 6]
            if content:
                pick = -max(content)[1]
        prev_plain = pick is not None and not strong
        out.append(pick)
    return out


def plan_chunks(spec: Spec, spec_dir: Path, errors: list[str]) -> list[Chunk]:
    """The resolved chunks of `spec.captions` in frames, or [] with `errors` filled."""
    cap = spec.captions
    assert cap is not None
    before = len(errors)
    if cap.preset not in PRESET_DEFS:
        errors.append(_line("captions.preset", f"unknown {cap.preset!r}", f"use {', '.join(PRESETS)}"))
    lo, hi = cap.words_per_chunk
    if not 1 <= lo <= hi <= 6:
        errors.append(_line("captions.words_per_chunk", f"{[lo, hi]} is not [min, max] within 1 to 6", "use [1, 3]"))
    if cap.emphasis not in ("auto", "none") and not isinstance(cap.emphasis, list):
        errors.append(_line("captions.emphasis", f"unknown {cap.emphasis!r}", "use auto, none or a list of word indexes"))
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", cap.accent):
        errors.append(_line("captions.accent", f"{cap.accent!r} is not #rrggbb", "use #ffe600"))
    for n, ov in enumerate(cap.overrides):
        if (ov.word is None) == (ov.chunk_break_before is None):
            errors.append(_line(f"captions.overrides[{n}]", "needs exactly one of word, chunk_break_before", "see docs/SPEC-REFERENCE.md"))
        if ov.word is None and (ov.text is not None or ov.emphasis is not None):
            errors.append(_line(f"captions.overrides[{n}]", "text and emphasis need a word", "add word: <index>"))
    path = _source_path(spec, spec_dir, errors)
    if len(errors) > before or path is None:
        return []
    words = read_words(path, errors)
    if len(errors) > before:
        return []
    if not words:
        errors.append(_line("captions.source", f"{path.name} has no words", "run montaj transcribe, or music analyze without --no-words"))
        return []
    fps = spec.video.fps
    offset = 0.0
    if cap.offset:
        from montaj.spec import _dur_frames

        frames = _dur_frames(cap.offset, fps, None, "captions.offset", errors)
        if frames is None:
            return []
        offset = frames / fps
    have = {w.i for w in words}
    for n, ov in enumerate(cap.overrides):
        for name, v in (("word", ov.word), ("chunk_break_before", ov.chunk_break_before)):
            if v is not None and v not in have:
                errors.append(_line(f"captions.overrides[{n}].{name}", f"word {v} not in the source", f"have {min(have)} to {max(have)}"))
    if isinstance(cap.emphasis, list):
        for v in cap.emphasis:
            if v not in have:
                errors.append(_line("captions.emphasis", f"word {v} not in the source", f"have {min(have)} to {max(have)}"))
    if len(errors) > before:
        return []

    raw = {w.i: w.w for w in words}
    text_of = {ov.word: ov.text for ov in cap.overrides if ov.word is not None and ov.text is not None}
    words = [Word(i=w.i, w=text_of[w.i].replace("*", "\\*") if w.i in text_of else normalise(w.w, cap.punctuation == "keep"),
                  s=w.s + offset, e=w.e + offset) for w in words]
    words = [w for w in words if w.w]
    preset = PRESET_DEFS[cap.preset]
    font, case = _font_for(preset, words)
    size = _size_for(spec, preset)
    book = _book()

    def width(group: list[Word]) -> float:
        try:
            return _painted_width(book, " ".join(w.w for w in group), preset, size, font, case)
        except ValueError:
            return 0.0  # a glyph the fonts lack: validation of the expanded line names it

    breaks = {ov.chunk_break_before for ov in cap.overrides if ov.chunk_break_before is not None}
    groups = _chunk_words(words, raw, lo, hi, breaks, width, _budget(spec, preset), fps)
    if cap.emphasis == "none":
        picks: list[int | None] = [None] * len(groups)
    elif isinstance(cap.emphasis, list):
        want = set(cap.emphasis)
        picks = [next((k for k, w in enumerate(g) if w.i in want), None) for g in groups]
    else:
        picks = _auto_emphasis(groups)
    for ov in cap.overrides:  # a per-word override wins over the list and over auto
        if ov.word is None or ov.emphasis is None:
            continue
        for gi, g in enumerate(groups):
            k = next((k for k, w in enumerate(g) if w.i == ov.word), None)
            if k is not None:
                picks[gi] = k if ov.emphasis else (None if picks[gi] == k else picks[gi])

    # Frames: a chunk starts on its first word's frame, holds to the next chunk (or clears after a long pause).
    n_chunks = len(groups)
    starts = [[round(w.s * fps) for w in g] for g in groups]
    out: list[Chunk] = []
    last_start = -1
    for gi, g in enumerate(groups):
        st = starts[gi]
        st[0] = max(st[0], last_start + 1)
        for k in range(1, len(st)):
            st[k] = max(st[k], st[k - 1] + 1)
        nxt = starts[gi + 1][0] if gi + 1 < n_chunks else None
        gap = (groups[gi + 1][0].s - g[-1].e) if gi + 1 < n_chunks else math.inf
        end = round((g[-1].e + HOLD_S) * fps) if gap > CLEAR_S else (nxt if nxt is not None else st[-1] + 1)
        if nxt is not None:
            end = min(end, max(nxt, st[0] + 1))
        end = max(end, st[-1] + 1)
        last_start = st[0]
        out.append(Chunk(index=gi, words=tuple(g), start=st[0], end=end, starts=tuple(st), emph=picks[gi]))
    # Overlap guard: a chunk never runs into the next one.
    fixed: list[Chunk] = []
    for gi, c in enumerate(out):
        if gi + 1 < len(out) and c.end > out[gi + 1].start:
            c = Chunk(c.index, c.words, c.start, max(c.start + 1, out[gi + 1].start), c.starts, c.emph)
        fixed.append(c)
    return fixed


def _book():
    from montaj.render.glyphs import FontBook

    return FontBook()


# --- expansion --------------------------------------------------------------------------------------------


def _mark(words: list[Word], k: int | None) -> str:
    return " ".join(f"*{w.w}*" if n == k else w.w for n, w in enumerate(words))


def _line_dict(spec: Spec, cap: Captions, preset: Preset, chunk: Chunk, text: str, at: int, *,
               first: bool, accent_word: int | None, bar: dict | None, y: float, size: float, font, case,
               max_width: float) -> dict:
    """One gpu `text` line for a chunk (or for one spoken word of it)."""
    line: dict = {
        "text": text, "pos": [_centre_x(spec), y], "at": f"{at}f", "style": preset.style, "color": "#ffffff",
        "size": size, "case": case, "fit": True, "max_width": max_width, "by": "word", "stagger": "0f", "engine": "gpu",
        "shadow": {"x": 0, "y": preset.shadow[0], "blur": preset.shadow[1], "color": "#000000", "opacity": preset.shadow[2]},
    }
    if font:
        line["font"] = font
    elif preset.weight is not None:
        line["weight"] = preset.weight
    if preset.stroke:
        line["stroke"] = {"width": round(size * preset.stroke, 1), "color": "#000000"}
    if preset.tilt:
        line["rotate"] = preset.tilt if chunk.index % 2 == 0 else -preset.tilt
    marked = accent_word is not None
    if marked:
        if preset.bar:
            line["emphasis"] = {"color": "#111111", "scale": 1.0}
        elif preset.mode == "word":
            line["emphasis"] = {"color": cap.accent, "scale": 1.0}
        else:
            emph: dict = {"color": cap.accent, "scale": preset.emph_scale}
            if preset.emph_in and first:
                emph["in"] = preset.emph_in
            if preset.emph_shake and first:
                emph["shake"] = preset.emph_shake
            line["emphasis"] = emph
    if first and preset.enter:
        line["in"] = {"type": preset.enter, "stagger": "0f"}
        if preset.enter == "drop":
            line["in"]["distance"] = 70
    elif not first or not preset.enter:
        line["dur"] = "0f"
        line["rise"] = 0
    if bar is not None:
        line["bar"] = bar
    return line


def _block(frm: int, to: int, lines: list[dict]) -> dict:
    return {"from": f"{frm}f", "to": f"{to}f", "fade_in": "0f", "fade_out": "0f", "lines": lines}


def _minimal_lines(spec: Spec, cap: Captions, chunk: Chunk, y: float, size: float, at: int, first: bool) -> list[dict]:
    """Sans words around an italic serif accent word, set side by side on one baseline (one line each)."""
    from montaj.render import glyphs

    book = _book()
    k = chunk.emph
    texts = [w.w for w in chunk.words]
    if k is None:
        parts = [(" ".join(texts), False)]
    else:
        parts = [(p, e) for p, e in ((" ".join(texts[:k]), False), (texts[k], True), (" ".join(texts[k + 1:]), False)) if p]

    def spec_of(text: str, emph: bool, sz: float) -> dict:
        return dict(style="serif" if emph else "sans", size=sz * (1.18 if emph else 1.0), italic=emph,
                    weight=None if emph else 400, case="lower")

    def width(text: str, emph: bool, sz: float) -> float:
        kw = spec_of(text, emph, sz)
        return glyphs.measure(book, text, style=kw["style"], size=kw["size"], weight=kw["weight"], italic=kw["italic"],
                              case="lower")[1]

    space = glyphs.measure(book, "a a", style="sans", size=size, weight=400, case="lower")[1] - \
        2 * glyphs.measure(book, "a", style="sans", size=size, weight=400, case="lower")[1]
    sz = size
    limit = _budget(spec, PRESET_DEFS["minimal"])
    for _ in range(3):  # shrink until the row fits the safe band, as `fit` would
        widths = [width(t, e, sz) for t, e in parts]
        total = sum(widths) + space * (len(parts) - 1)
        if total <= limit:
            break
        sz = sz * limit / total
    lines = []
    x = _centre_x(spec) - total / 2
    for (text, emph), w in zip(parts, widths):
        kw = spec_of(text, emph, sz)
        line: dict = {
            "text": text, "pos": [round(x + w / 2, 2), y], "at": f"{at}f", "style": kw["style"], "size": round(kw["size"], 1),
            "case": "lower", "color": cap.accent if emph else "#ffffff", "by": "word", "stagger": "0f", "engine": "gpu",
            "shadow": {"x": 0, "y": 3, "blur": 12, "color": "#000000", "opacity": 0.55},
        }
        if emph:
            line["italic"] = True
        else:
            line["weight"] = 400
        if first:
            line["in"] = {"type": "scale-in", "stagger": "0f"}
        else:
            line["dur"] = "0f"
            line["rise"] = 0
        lines.append(line)
        x += w + space
    return lines


def expand_captions(spec: Spec, spec_dir: Path | None, errors: list[str] | None = None) -> Spec:
    """`spec` with `captions` replaced by the `text` blocks it stands for. With `errors`, problems are
    appended and the spec returned unchanged; without, they raise SpecError."""
    if spec.captions is None:
        return spec
    own = errors is None
    errors = [] if errors is None else errors
    before = len(errors)
    if spec_dir is None:
        errors.append(_line("captions", "needs a spec file", "load the spec from disk"))
    chunks = plan_chunks(spec, spec_dir, errors) if len(errors) == before and spec_dir is not None else []
    if len(errors) > before:
        if own:
            raise SpecError(errors[before:])
        return spec
    cap = spec.captions
    preset = PRESET_DEFS[cap.preset]
    words_all = [w for c in chunks for w in c.words]
    font, case = _font_for(preset, words_all)
    size = _size_for(spec, preset)
    y = float(cap.y) if cap.y is not None else default_y(spec)
    blocks: list[dict] = []
    book, budget = _book(), _budget(spec, preset)
    for c in chunks:
        ws = list(c.words)
        if preset.serif_emphasis:
            blocks.append(_block(c.start, c.end, _minimal_lines(spec, cap, c, y, size, c.start, True)))
            continue
        # One width per chunk: `fit` shrinks a chunk whose accent (or a lone long word) would paint past the band.
        marks = [_mark(ws, c.emph)] if preset.mode == "chunk" else [_mark(ws, k) for k in range(len(ws))]
        try:
            fit_w = _fit_width(book, marks, preset, size, font, case, budget)
        except ValueError:
            fit_w = math.floor(budget * 10) / 10  # a glyph the fonts lack: validation of the expanded line names it
        if preset.mode == "chunk":
            line = _line_dict(spec, cap, preset, c, _mark(ws, c.emph), c.start, first=True, accent_word=c.emph,
                              bar=None, y=y, size=size, font=font, case=case, max_width=fit_w)
            blocks.append(_block(c.start, c.end, [line]))
        else:  # word: the chunk stays up, the spoken word is marked (and barred) from its own frame
            for k, st in enumerate(c.starts):
                nxt = c.starts[k + 1] if k + 1 < len(c.starts) else c.end
                if nxt <= st:
                    continue
                bar = None
                if preset.bar:
                    bar = {"word": k, "color": cap.accent, "pad": round(size * 0.2, 1)}
                    if k > 0:
                        bar["from"] = k - 1
                line = _line_dict(spec, cap, preset, c, _mark(ws, k), st, first=k == 0, accent_word=k, bar=bar,
                                  y=y, size=size, font=font, case=case, max_width=fit_w)
                blocks.append(_block(st, nxt, [line]))
    parsed = []
    for n, b in enumerate(blocks):
        try:
            parsed.append(TextBlock.model_validate(b))
        except Exception as exc:  # pydantic ValidationError: a bug in the preset, reported with the chunk
            errors.append(_line(f"captions block {n}", "expansion failed", str(exc).splitlines()[0]))
    if len(errors) > before:
        if own:
            raise SpecError(errors[before:])
        return spec
    return spec.model_copy(update={"text": [*spec.text, *parsed], "captions": None, "caption_from": len(spec.text)})


# --- review as text ----------------------------------------------------------------------------------------


def format_chunks(chunks: list[Chunk], fps: int) -> str:
    """`chunk  start  end  text`; the accented word is starred. Seconds, two decimals."""
    rows = ["chunk  start   end     text"]
    for c in chunks:
        text = " ".join(f"*{w.w}*" if k == c.emph else w.w for k, w in enumerate(c.words))
        rows.append(f"{c.index:<6} {c.start / fps:>6.2f}  {c.end / fps:>6.2f}  {text}")
    return "\n".join(rows)


# --- review and transcription entry points ----------------------------------------------------------------


def plan(spec_file: Path) -> tuple[list[Chunk], int]:
    """(chunks, fps) of the spec's `captions` block, for `montaj captions --print`."""
    from montaj.spec import load_spec

    spec = load_spec(spec_file, expand=False)
    if spec.captions is None:
        raise SpecError([_line("captions", "the spec has no captions block", "add captions: {source: ...}")])
    errors: list[str] = []
    chunks = plan_chunks(spec, spec_file.parent, errors)
    if errors:
        raise SpecError(errors)
    return chunks, spec.video.fps


def transcribe(clip: Path | str, out: Path | str | None = None, *, lang: str = "hi", prompt: str | None = None,
               config=None, log: Path | str | None = None) -> Path:
    """Word timings of `clip` (video or audio) as `{words: [{i, w, s, e}]}`, written to `out`.

    Runs the whisper worker from `~/.config/montaj/config.toml` on the GPU under the worker lock
    (`audio.analyze.analyze` holds it). `out` defaults to `<clip stem>.words.json` beside the clip.
    """
    from montaj.audio.analyze import analyze

    clip = Path(clip).expanduser()
    target = Path(out).expanduser() if out is not None else clip.with_name(f"{clip.stem}.words.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.markers")
    try:
        analyze(clip, tmp, lang=lang, prompt=prompt, config=config, log=log)
        markers = json.loads(tmp.read_text(encoding="utf-8"))
    finally:
        tmp.unlink(missing_ok=True)
    words = [{"i": int(w["i"]), "w": str(w["w"]), "s": float(w["s"]), "e": float(w["e"])}
             for w in markers.get("words") or ()]
    target.write_text(json.dumps({"words": words}, ensure_ascii=False, indent=1), encoding="utf-8")
    return target
