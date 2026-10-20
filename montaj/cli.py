"""`montaj <cmd>`. Progress stays in build/*.log; stdout is the result line.

`-C DIR` and `--json` are accepted before or after the command. `--json` prints
one object, `{"status": "OK"|"WARN"|"ERR", "lines": [...]}`, instead of the text.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import yaml

from montaj.audio import analyze as analyze_audio
from montaj.audio import compose as compose_audio
from montaj.audio import music as music_gen
from montaj.audio.mix import check_lines
from montaj.audio.music import parse_duration
from montaj.doctor import doctor
from montaj.encode import MediaInfo, export, probe
from montaj.ingest import IngestError, ingest, ingest_rec
from montaj import taste
from montaj.guards import content_guards
from montaj.qa import BLACK_PIX_TH, FREEZE_MIN_S, Expect, check, clip_overruns, subtract, union
from montaj.review.server import serve as serve_review
from montaj.review.summary import summary as review_summary
from montaj.sheet import MAX_TILES, every_times, pick_times, sheet, sheets
from montaj.spec import INCOMING_TRANS, SpecError, _dur_frames, load_spec
from montaj.timeline import Blank, Card, Sequence, Still, Wall, resolve, sequence_index, wall_static_spans
from montaj.watch import watch

RECIPES = Path(__file__).resolve().parents[1] / "recipes"


class _Usage(Exception):
    def __init__(self, message: str, as_json: bool = False):
        super().__init__(message)
        self.as_json = as_json


_COMMANDS = (
    "doctor", "new", "ingest", "ingest-rec", "validate", "render", "sheet", "check", "export", "music",
    "review", "watch", "taste",
)


class _Parser(argparse.ArgumentParser):
    """Argument errors raise. argparse's own `error` writes usage to stderr and exits 2."""

    def error(self, message: str) -> None:
        raise _Usage(message)


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="montaj")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor")

    new = sub.add_parser("new")
    new.add_argument("dir")
    new.add_argument("--recipe", required=True)
    new.add_argument("--photos", required=True)

    ing = sub.add_parser("ingest")
    ing.add_argument("--photos", required=True)

    rec = sub.add_parser("ingest-rec")
    rec.add_argument("video")
    rec.add_argument("--events", default=None)

    sub.add_parser("validate")

    ren = sub.add_parser("render")
    group = ren.add_mutually_exclusive_group()
    group.add_argument("--preview", action="store_true")
    group.add_argument("--final", action="store_true")

    sh = sub.add_parser("sheet")
    when = sh.add_mutually_exclusive_group()
    when.add_argument("--at", default=None)
    when.add_argument("--every", default=None)
    sh.add_argument("--video", default=None)

    ck = sub.add_parser("check")
    ck.add_argument("--video", default=None)

    ex = sub.add_parser("export")
    ex.add_argument("--target", required=True, choices=("master", "whatsapp"))

    music_p = sub.add_parser("music")
    music_sub = music_p.add_subparsers(dest="music_cmd", required=True, metavar="gen|analyze|compose")
    gen = music_sub.add_parser("gen")
    gen.add_argument("--caption", required=True)
    gen.add_argument("--lyrics", required=True)
    gen.add_argument("--bpm", type=float, default=music_gen.DEFAULT_BPM)
    gen.add_argument("--key", default=music_gen.DEFAULT_KEY)
    gen.add_argument("--lang", default=music_gen.DEFAULT_LANG)
    gen.add_argument("--duration", default=f"{music_gen.DEFAULT_DURATION:g}s")
    gen.add_argument("--n", type=int, default=music_gen.DEFAULT_N)
    ana = music_sub.add_parser("analyze")
    ana.add_argument("wav")
    ana.add_argument("--lang", default=analyze_audio.DEFAULT_LANG)
    ana.add_argument("--prompt", default=None)
    ana.add_argument("--bpm", type=float, default=None)
    ana.add_argument("--out", default=analyze_audio.DEFAULT_OUT)
    ana.add_argument("--no-words", action="store_true")
    comp = music_sub.add_parser("compose")
    comp.add_argument("script")
    comp.add_argument("--out", default="music/song.wav")
    comp.add_argument("--timeout", type=float, default=600.0)

    rv = sub.add_parser("review")
    rv.add_argument("--port", type=int, default=8765)
    rv.add_argument("--video", default=None)
    rv.add_argument("--summary", action="store_true")

    wa = sub.add_parser("watch")
    wa.add_argument("--final", action="store_true")

    ta = sub.add_parser("taste")
    ta.add_argument("action", nargs="?", choices=("add",))
    ta.add_argument("line", nargs="?")
    return p


def _split(argv: list[str]) -> tuple[list[str], Path | None, bool]:
    rest: list[str] = []
    chdir: Path | None = None
    as_json = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--json":
            as_json = True
        elif arg == "-C":
            if i + 1 >= len(argv):
                raise _Usage("cli: -C needs a directory", as_json)
            chdir = Path(argv[i + 1])
            i += 1
        else:
            rest.append(arg)
        i += 1
    return rest, chdir, as_json


def _root(chdir: Path | None) -> Path:
    base = Path.cwd() if chdir is None else chdir
    return base.expanduser().resolve()


def _at(root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _status(lines: list[str]) -> str:
    last = lines[-1] if lines else "ERR"
    for prefix in ("ERR", "WARN", "OK"):
        if last.startswith(prefix):
            return prefix
    return "ERR"


def _emit(lines: list[str], as_json: bool) -> int:
    status = _status(lines)
    if as_json:
        print(json.dumps({"status": status, "lines": lines}, ensure_ascii=False))
    else:
        for line in lines:
            print(line)
    return 0 if status != "ERR" else 1


def _one_line(text: str) -> str:
    """ffmpeg tails and YAML errors are multi-line. Stdout gets one result line."""
    return " ".join(text.split())


def _argv_cmd(rest: list[str]) -> str:
    return rest[0] if rest and rest[0] in _COMMANDS else "cli"


def _note_failure(root: Path, cmd: str, exc: BaseException) -> None:
    """Keep the original error text, newlines included, in `build/<cmd>.log`."""
    path = root / "build" / f"{cmd}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = str(exc)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")


def _assets_rel(spec_file: Path) -> str:
    if not spec_file.is_file():
        return "assets"
    data = yaml.safe_load(spec_file.read_text()) or {}
    if isinstance(data, dict) and isinstance(data.get("assets"), str):
        return data["assets"]
    return "assets"


def _assets_dir(base: Path, rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else base / path


def _recipe(name: str) -> Path:
    stem = name[:-5] if name.endswith(".yaml") else name
    src = RECIPES / f"{stem}.yaml"
    if not src.is_file():
        have = ", ".join(sorted(p.stem for p in RECIPES.glob("*.yaml"))) or "none"
        raise FileNotFoundError(f"recipe: {stem!r} not in recipes/ (have: {have})")
    return src


def _video(root: Path, given: str | None) -> Path:
    """Explicit path, else the newer of out/final.mp4 and out/preview.mp4."""
    if given:
        return _at(root, given)
    cands = [p for p in (root / "out" / "final.mp4", root / "out" / "preview.mp4") if p.is_file()]
    if not cands:
        raise FileNotFoundError("video: no out/final.mp4 or out/preview.mp4 (render first, or pass --video)")
    return max(cands, key=lambda p: p.stat().st_mtime)


def _export_source(root: Path) -> Path:
    """Prefer the final render; a preview is enough when that is all that exists."""
    final = root / "out" / "final.mp4"
    preview = root / "out" / "preview.mp4"
    if final.is_file():
        return final
    if preview.is_file():
        return preview
    raise FileNotFoundError("video: no out/final.mp4 or out/preview.mp4 (render first)")


def _cmd_doctor() -> list[str]:
    lines = doctor()
    failed = sum(line.startswith("FAIL") for line in lines)
    lines.append(f"ERR doctor: {failed} failed" if failed else "OK doctor")
    return lines


def _cmd_new(ns: argparse.Namespace, root: Path) -> list[str]:
    dest = _at(root, ns.dir)
    spec_out = dest / "montaj.yaml"
    if spec_out.exists():
        raise FileExistsError(f"{spec_out.as_posix()} exists: refusing to overwrite")
    photos = _at(root, ns.photos)
    src = _recipe(ns.recipe)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest / "montaj.yaml")
    assets = _assets_dir(dest, _assets_rel(src))
    ingest(photos, assets, sheet_path=dest / "build" / "ingest-sheet.jpg")
    shown = dest.as_posix()
    return [f"OK {shown}/montaj.yaml sheet={shown}/build/ingest-sheet.jpg"]


def _cmd_ingest_rec(ns: argparse.Namespace, root: Path) -> list[str]:
    assets = _assets_dir(root, _assets_rel(root / "montaj.yaml"))
    events = _at(root, ns.events) if ns.events else None
    try:
        rec = ingest_rec(_at(root, ns.video), assets, events=events, log=root / "build" / "ingest-rec.log")
    except IngestError as exc:
        return [f"ERR {exc}"]
    n_ev = f" {rec.n_events} events" if rec.events_path else ""
    audio = f" {rec.audio_streams} audio stream{'s' if rec.audio_streams != 1 else ''}"
    return [f"OK {rec.stem} {rec.duration:.1f}s {rec.fps:g}fps{audio}{n_ev} -> {rec.path.name}"]


def _cmd_ingest(ns: argparse.Namespace, root: Path) -> list[str]:
    photos = _at(root, ns.photos)
    assets = _assets_dir(root, _assets_rel(root / "montaj.yaml"))
    infos = ingest(photos, assets, sheet_path=root / "build" / "ingest-sheet.jpg")
    return [f"OK {len(infos)} photos sheet=build/ingest-sheet.jpg"]


def _cmd_validate(root: Path) -> list[str]:
    spec = load_spec(root / "montaj.yaml")
    tl = resolve(spec, root)
    seconds = tl.n_frames / tl.fps
    n_shots, n_segs = len(tl.shots), len(tl.segments)
    return [f"OK montaj.yaml {tl.n_frames}f {seconds:.2f}s {n_shots} shots {n_segs} segments"]


def _cmd_render(ns: argparse.Namespace, root: Path) -> list[str]:
    from montaj.pipeline import render

    mode = "final" if ns.final else "preview"
    t0 = time.perf_counter()
    result = render(root / "montaj.yaml", mode)
    dt = time.perf_counter() - t0
    mb = result.path.stat().st_size / (1024 * 1024)
    rel = result.path.resolve().relative_to(root.resolve()).as_posix()
    n = result.rendered + result.cached
    w, h = result.size
    audio = " audio" if result.audio else ""
    return [
        f"OK {rel} {result.seconds:.2f}s {w}x{h} {mb:.1f}MB "
        f"rendered {result.rendered}/{n} cached {result.cached}/{n} in {dt:.1f}s{audio}"
    ]


def _seconds(raw: str, flag: str = "--at") -> float:
    """A time in seconds: `4.5` or `4.5s`. `--every` must be a positive interval."""
    text = raw.strip()
    try:
        value = float(text[:-1] if text.endswith("s") else text)
    except ValueError:
        hint = "0.5 or 0.5s" if flag == "--every" else "4.5 or 4.5s"
        raise ValueError(f"{flag} {text!r}: use seconds like {hint}") from None
    if flag == "--every" and (not math.isfinite(value) or value <= 0):
        raise ValueError(f"{flag} {text!r}: use a positive interval like 0.5s")
    return value


def _shown_frames(times: list[float], info: MediaInfo) -> int:
    """How many distinct frames `extract` will keep. The OK line counts those."""
    n_frames = int(round(info.duration * info.fps)) or 1
    return len({min(max(0, int(round(t * info.fps))), n_frames - 1) for t in times})


def _cmd_sheet(ns: argparse.Namespace, root: Path) -> list[str]:
    video = _video(root, ns.video)
    log = root / "build" / "sheet.log"
    info = probe(video, log=log)
    out = root / "build" / "sheet.jpg"
    if ns.every is not None:
        interval = _seconds(str(ns.every), flag="--every")
        # Fixed stride from t=0. Pages past 12 frames so a 60s film at 0.5s fits.
        times = every_times(info.duration, interval, fps=info.fps)
        paths = sheets(video, out, times, log=log)
    else:
        at = None
        if ns.at:
            at = [_seconds(part) for part in str(ns.at).split(",") if part.strip()]
        # Pass the times in, so sheet does not probe into the process cwd's build/.
        # sheet() keeps at most MAX_TILES. Count the same list it will draw.
        times = pick_times(video, at, n=MAX_TILES, info=info)[:MAX_TILES]
        paths = [sheet(video, out, at=times, log=log)]
    shown = _shown_frames(times, info)
    rels = " ".join(path.resolve().relative_to(root.resolve()).as_posix() for path in paths)
    return [f"OK {rels} {shown} frames"]


def _static_still(spec) -> bool:
    """A still is animated only by a moving drift, a pulse, or a changing card tilt.
    Zoom, tone and an untilted frame card hold still, so their freeze is intended.
    A morph moves only inside its reveal window (see `_morph_moving`)."""
    d, fr = spec.drift, spec.frame
    drifts = d is not None and (d.zoom[0] != d.zoom[1] or any(d.pan))
    tilts = fr is not None and fr.tilt is not None and fr.tilt[0] != fr.tilt[1]
    return not drifts and not tilts and spec.pulse is None


def _morph_moving(sh, f: int, fps: float, bpm: float | None) -> bool:
    """True in the middle half of a morph's reveal. Its eased ends barely change the picture, and
    before and after it the shot is a still: a freeze there is intended (benchmark run 8)."""
    m = sh.spec.morph
    if m is None:
        return False
    at = _dur_frames(m.at, int(fps), bpm, "morph.at", []) or 0.0
    dur = _dur_frames(m.dur, int(fps), bpm, "morph.dur", []) or 0.0
    lo = sh.start + at + dur / 4
    return lo <= f < lo + dur / 2


def _intended_freeze(root: Path) -> list[tuple[float, float]]:
    """Time spans whose picture is designed not to move: static stills, blank holds,
    sequence steps, and wall frames where both camera tracks hold. A freeze of a drifting,
    pulsing or tilting shot, of a morph mid-reveal, or inside a transition is a defect."""
    spec_file = root / "montaj.yaml"
    if not spec_file.is_file():
        return []
    tl = resolve(load_spec(spec_file), root)
    fps = float(tl.fps)
    holds: list[tuple[int, int]] = []
    for sh in tl.shots:
        holds.extend(wall_static_spans(sh))

    def wall_hold(f: int) -> bool:
        return any(lo <= f < hi for lo, hi in holds)

    spans: list[tuple[float, float]] = []
    start: int | None = None
    for f in range(tl.n_frames):
        scene = tl.plan(f).scene
        # A blank hold is the background colour on purpose (the ink tail). A fade into it is not.
        if isinstance(scene, Still):
            sh = tl.shots[scene.shot]
            intended = _static_still(sh.spec) and not _morph_moving(sh, f, fps, tl.spec.video.bpm)
        elif isinstance(scene, Blank):
            intended = True
        elif isinstance(scene, Sequence):
            intended = True
        elif isinstance(scene, Card):
            intended = _card_settled(tl.shots[scene.shot], f, fps, tl.spec.video.bpm, scene)
        else:
            intended = isinstance(scene, Wall) and wall_hold(f)
        if intended:
            if start is None:
                start = f
        elif start is not None:
            spans.append((start / fps, f / fps))
            start = None
    if start is not None:
        spans.append((start / fps, tl.n_frames / fps))
    return _split_sequence_steps(tl, spans, fps)


def _split_sequence_steps(tl, spans: list[tuple[float, float]], fps: float) -> list[tuple[float, float]]:
    """Cut a gap around each sequence boundary where the item changes. A step is a designed
    hold, but a freeze running across such a boundary means the output is stuck on one item;
    the gap is longer than the freeze detector's minimum, so only that case leaves a report."""
    gap = FREEZE_MIN_S * 1.5
    cuts: list[tuple[float, float]] = []
    for sh in tl.shots:
        seq = sh.spec.sequence
        if seq is None or isinstance(seq.every, bool) or not isinstance(seq.every, int):
            continue
        if seq.every < 1 or len(seq.items) < 2:
            continue
        for lo in range(sh.start + seq.every, sh.end, seq.every):
            before = sequence_index(sh.start, seq.every, len(seq.items), lo - 1)
            after = sequence_index(sh.start, seq.every, len(seq.items), lo)
            if seq.items[before] != seq.items[after]:
                cuts.append((lo / fps - gap / 2, lo / fps + gap / 2))
    for cut in cuts:
        spans = [piece for span in spans for piece in subtract(span, [cut])]
    return spans


def _frames(spec, raw: str | None) -> float:
    if not raw:
        return 0.0
    got = _dur_frames(raw, spec.video.fps, spec.video.bpm, "duration", [])
    return 0.0 if got is None else float(got)


def _is_dark(color: str) -> bool:
    """True when a solid frame of `color` would trip blackdetect at `pix_th`."""
    text = str(color).strip()
    if len(text) != 7 or text[0] != "#":
        return False
    try:
        n = int(text[1:], 16)
    except ValueError:
        return False
    r = ((n >> 16) & 255) / 255.0
    g = ((n >> 8) & 255) / 255.0
    b = (n & 255) / 255.0
    return (0.299 * r + 0.587 * g + 0.114 * b) < BLACK_PIX_TH


def _sequence_dark_spans(tl, fps: float) -> list[tuple[float, float]]:
    """Each dark colour step of a sequence, on the shot clock. Photos are not included."""
    spans: list[tuple[float, float]] = []
    for sh in tl.shots:
        seq = sh.spec.sequence
        if seq is None or isinstance(seq.every, bool) or not isinstance(seq.every, int):
            continue
        if seq.every < 1 or not seq.items:
            continue
        step = 0
        while True:
            lo = sh.start + step * seq.every
            if lo >= sh.end:
                break
            hi = min(sh.end, lo + seq.every)
            item = seq.items[sequence_index(sh.start, seq.every, len(seq.items), lo)]
            if item.color is not None and _is_dark(item.color):
                spans.append((lo / fps, hi / fps))
            step += 1
    return spans


def _dark_plate(sh, background: str) -> bool:
    """A blank shows `video.background`. A card shows its own colour."""
    if sh.spec.card is not None:
        return _is_dark(sh.spec.card.color)
    return bool(sh.spec.blank) and _is_dark(background)


def _card_settled(sh, f: int, fps: float, bpm: float | None, scene: Card) -> bool:
    """True once every line has finished revealing and a gold sweep has ended.

    A settled card is a designed hold. While glyphs are still arriving, a freeze is a defect.
    """
    from montaj.render.text import _split

    card = sh.spec.card
    if card is None:
        return False
    fps_i = int(fps)
    for line in card.text:
        at = sh.start + (_dur_frames(line.at, fps_i, bpm, "card.at", []) or 0.0)
        if line.reveal == "pop":
            span = float(scene.span[1] - scene.span[0])
            if f < at + (span / 5.0 if span > 0 else 0.0):
                return False
        else:
            n = len(_split(line.text, line.by))
            if line.stagger:
                stagger = _dur_frames(line.stagger, fps_i, bpm, "card.stagger", []) or 0.0
            else:
                stagger = (0.07 if line.by == "char" else 0.16) * fps
            if line.dur:
                dur = _dur_frames(line.dur, fps_i, bpm, "card.dur", []) or 0.0
            else:
                dur = 0.9 * fps
            if f < at + max(0, n - 1) * stagger + dur:
                return False
        if line.gold:
            if line.sweep:
                a = sh.start + (_dur_frames(line.sweep[0], fps_i, bpm, "card.sweep", []) or 0.0)
                b = sh.start + (_dur_frames(line.sweep[1], fps_i, bpm, "card.sweep", []) or 0.0)
            else:
                a, b = float(sh.start), float(sh.end)
            if a <= f < b and b > a:
                return False
    return True


def _intended_black(root: Path) -> list[tuple[float, float]]:
    """Dark windows blackdetect should not call a defect.

    Shot 0's fade from a dark `video.background`, a blank on that background
    (hold plus its in/out transitions) or a dark card (hold plus its in/out
    transitions), a sequence step on a dark colour, and the outro fade to black.
    """
    spec_file = root / "montaj.yaml"
    if not spec_file.is_file():
        return []
    tl = resolve(load_spec(spec_file), root)
    spec = tl.spec
    fps = float(tl.fps)
    spans: list[tuple[float, float]] = []
    dark = _is_dark(spec.video.background)
    shots = tl.shots
    if shots:
        first = shots[0]
        tr = first.spec.in_
        if dark and not first.spec.blank and tr is not None and tr.type == "fade" and tr.dur:
            dur = _frames(spec, tr.dur)
            spans.append((first.start / fps, (first.start + dur) / fps))
        for i, sh in enumerate(shots):
            if not _dark_plate(sh, spec.video.background):
                continue
            before = 0.0
            incoming = sh.spec.in_
            if incoming is not None and incoming.type == "whip" and incoming.dur:
                before = _frames(spec, incoming.dur) / 2.0
            after = 0.0
            if i + 1 < len(shots):
                outgoing = shots[i + 1].spec.in_
                if outgoing is not None and outgoing.dur and outgoing.type != "cut":
                    nd = _frames(spec, outgoing.dur)
                    if outgoing.type == "whip":
                        after = nd / 2.0
                    elif outgoing.type in INCOMING_TRANS:
                        after = nd
            spans.append((max(0.0, (sh.start - before) / fps), (sh.end + after) / fps))
    spans.extend(_sequence_dark_spans(tl, fps))
    if tl.n_frames:
        fade = tl.plan(tl.n_frames - 1).fade_out
        if fade is not None:
            _dur, start = fade
            spans.append((start / fps, tl.n_frames / fps))
    return spans


def _check_overrides(root: Path) -> tuple[list[Expect], float | None, float | None]:
    """Resolved `check.expect` spans, plus an optional loudness target and tolerance."""
    spec_file = root / "montaj.yaml"
    if not spec_file.is_file():
        return [], None, None
    tl = resolve(load_spec(spec_file), root)
    expects = [Expect(item.kind, (item.start, item.end), item.why) for item in tl.check_expects]
    loud = None if tl.spec.check is None else tl.spec.check.loudness
    if loud is None:
        return expects, None, None
    return expects, float(loud.i), float(loud.tol)


def _av_tol(root: Path) -> float | None:
    """One frame at the film's fps when a soundtrack is set. Silent films keep qa's default."""
    spec_file = root / "montaj.yaml"
    if not spec_file.is_file():
        return None
    spec = load_spec(spec_file)
    if spec.audio is None or spec.video.fps <= 0:
        return None
    return 1.0 / float(spec.video.fps)


def _audio_mix_lines(root: Path) -> list[str]:
    """Mix stats from the last render, when the spec mixes clips, edits or ducking."""
    stats_file = root / "build" / "audio-mix.json"
    spec_file = root / "montaj.yaml"
    if not stats_file.is_file() or not spec_file.is_file():
        return []
    audio = load_spec(spec_file).audio
    if audio is None or not (audio.clips or audio.edit or audio.duck):
        return []
    return check_lines(json.loads(stats_file.read_text()))


def _clip_overruns(root: Path) -> list[str]:
    """Clip shots whose timeline read runs past the source. No spec means no new lines."""
    spec_file = root / "montaj.yaml"
    if not spec_file.is_file():
        return []
    tl = resolve(load_spec(spec_file), root)
    return clip_overruns(tl, _assets_dir(root, tl.spec.assets), spec_dir=root, log=root / "build" / "qa.log")


def _cmd_check(ns: argparse.Namespace, root: Path) -> list[str]:
    expects, loud_i, loud_tol = _check_overrides(root)
    black = union([*_intended_black(root), *(item.span for item in expects if item.kind == "black")])
    freeze = union([*_intended_freeze(root), *(item.span for item in expects if item.kind == "frozen")])
    lines = check(
        _video(root, ns.video), freeze,
        log=root / "build" / "qa.log", av_tol=_av_tol(root),
        intended_black=black, expect=expects,
        loudness_target=loud_i, loudness_tol=loud_tol,
    )
    spec_file = root / "montaj.yaml"
    spec = load_spec(spec_file) if spec_file.is_file() else None
    if spec is not None and spec.check is not None:
        tl = resolve(spec, root)
        lines += content_guards(tl, _video(root, ns.video), root / "build" / "guards",
                                log=root / "build" / "qa.log")
    lines += _audio_mix_lines(root)
    lines.extend(_clip_overruns(root))
    errs = sum(line.startswith("ERR") for line in lines)
    warns = sum(line.startswith("WARN") for line in lines)
    if errs:
        lines.append(f"ERR check: {errs} guard(s) unavailable")
    else:
        lines.append(f"WARN check: {warns} warnings" if warns else "OK check")
    return lines


def _cmd_export(ns: argparse.Namespace, root: Path) -> list[str]:
    target = root / "out" / f"{ns.target}.mp4"
    export(_export_source(root), target, ns.target, log=root / "build" / "export.log")
    mb = target.stat().st_size / (1024 * 1024)
    return [f"OK out/{ns.target}.mp4 {mb:.1f}MB"]


def _rel(root: Path, path: Path) -> str:
    """Path shown to the agent: relative to the project when it is inside it, else absolute."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _cmd_music_gen(ns: argparse.Namespace, root: Path) -> list[str]:
    duration = parse_duration(ns.duration)
    log = root / "build" / "music.log"
    try:
        made = music_gen.gen(
            root / "music",
            caption=ns.caption,
            lyrics=_at(root, ns.lyrics),
            bpm=ns.bpm,
            key=ns.key,
            lang=ns.lang,
            duration=parse_duration(ns.duration),
            n=ns.n,
            log=log,
        )
    except music_gen.PartialGenError as exc:
        lines = [
            f"{_rel(root, wav)} seed={seed} {_candidate_duration(wav, log):.1f}s"
            for wav, seed in zip(exc.files, exc.seeds, strict=True)
        ]
        lines.append(f"ERR music gen: {len(exc.files)}/{exc.n} candidates ({exc.reason})")
        return lines
    lines = [
        f"{_rel(root, wav)} seed={seed} {_candidate_duration(wav, log):.1f}s"
        for wav, seed in zip(made.files, made.seeds, strict=True)
    ]
    lines.append(
        f"OK music gen {len(made)} candidates {duration:g}s lang={ns.lang} "
        f"bpm={ns.bpm:g} key={ns.key}"
    )
    return lines


def _candidate_duration(path: Path, log: Path) -> float:
    info = probe(path, log=log)
    return info.audio_duration if info.audio_duration is not None else info.duration


def _cmd_music_analyze(ns: argparse.Namespace, root: Path) -> list[str]:
    analysis = analyze_audio.analyze(
        _at(root, ns.wav),
        _at(root, ns.out),
        lang=ns.lang,
        prompt=ns.prompt,
        bpm=ns.bpm,
        no_words=ns.no_words,
        log=root / "build" / "music.log",
    )
    duration = f"{analysis.duration:.1f}s"
    beat0 = f"{analysis.first_beat:.2f}s" if analysis.first_beat is not None else "none"
    if analysis.no_words:
        result = f"OK {_rel(root, analysis.path)} {duration} bpm={analysis.bpm:.1f} beat0={beat0} no words"
    elif analysis.first_word is not None:
        result = (
            f"OK {_rel(root, analysis.path)} {duration} bpm={analysis.bpm:.1f} beat0={beat0} "
            f"{analysis.n_words} words first_word={analysis.first_word:.2f}s lang={analysis.lang}"
        )
    else:
        result = (
            f"OK {_rel(root, analysis.path)} {duration} bpm={analysis.bpm:.1f} beat0={beat0} "
            f"{analysis.n_words} words lang={analysis.lang}"
        )
    return [*analysis.lines, result]


def _cmd_music_compose(ns: argparse.Namespace, root: Path) -> list[str]:
    try:
        result = compose_audio.compose(
            _at(root, ns.script),
            _at(root, ns.out),
            root=root,
            timeout=ns.timeout,
            log=root / "build" / "music.log",
        )
    except compose_audio.ComposeError as exc:
        return [f"ERR music compose: {exc}"]
    lines: list[str] = []
    if result.peak_dbfs >= -0.1:
        lines.append(f"WARN music compose: clipping (peak {compose_audio.format_db(result.peak_dbfs)} dBFS)")
    lines.append(
        f"OK {_rel(root, result.path)} {result.duration:.2f}s {result.sample_rate}Hz {result.channels}ch "
        f"peak {compose_audio.format_db(result.peak_dbfs)}dBFS {result.lufs:.1f}LUFS"
    )
    return lines


def _cmd_taste(ns: argparse.Namespace) -> list[str]:
    if ns.action == "add":
        if not (ns.line or "").strip():
            return ['ERR taste: add needs a line, e.g. montaj taste add "never zoom faces"']
        return [f"OK taste {taste.add(ns.line.strip())} lines"]
    text = taste.read()
    lines = text.splitlines()
    return lines + [f"OK taste {sum(1 for l in lines if l.strip())} lines"]


def _streaming(ns: argparse.Namespace, root: Path) -> bool:
    """`review` (server) and `watch` run until Ctrl-C and print their own lines."""
    if ns.cmd == "review" and not ns.summary:
        video = _at(root, ns.video) if ns.video else None
        serve_review(root, ns.port, video=video, log=root / "build" / "review.log")
        return True
    if ns.cmd == "watch":
        # Flush per line: watch output is often piped or tailed from a file.
        watch(root, mode="final" if ns.final else "preview", out=lambda line: print(line, flush=True))
        return True
    return False


def _dispatch(ns: argparse.Namespace, root: Path) -> list[str]:
    if ns.cmd == "doctor":
        return _cmd_doctor()
    if ns.cmd == "new":
        return _cmd_new(ns, root)
    if ns.cmd == "ingest":
        return _cmd_ingest(ns, root)
    if ns.cmd == "ingest-rec":
        return _cmd_ingest_rec(ns, root)
    if ns.cmd == "validate":
        return _cmd_validate(root)
    if ns.cmd == "render":
        return _cmd_render(ns, root)
    if ns.cmd == "sheet":
        return _cmd_sheet(ns, root)
    if ns.cmd == "check":
        return _cmd_check(ns, root)
    if ns.cmd == "export":
        return _cmd_export(ns, root)
    if ns.cmd == "music":
        if ns.music_cmd == "gen":
            return _cmd_music_gen(ns, root)
        if ns.music_cmd == "analyze":
            return _cmd_music_analyze(ns, root)
        return _cmd_music_compose(ns, root)
    if ns.cmd == "review":
        return review_summary(root)[0]
    if ns.cmd == "taste":
        return _cmd_taste(ns)
    raise _Usage(f"cli: unknown command {ns.cmd!r}")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        rest, chdir, as_json = _split(argv)
    except _Usage as exc:
        return _emit([f"ERR {exc}"], exc.as_json)
    try:
        ns = _parser().parse_args(rest)
    except _Usage as exc:
        return _emit([f"ERR {_argv_cmd(rest)}: {_one_line(str(exc))}"], as_json)
    except SystemExit as exc:
        # `--help` prints itself and exits 0. Any other exit becomes one ERR line.
        if exc.code in (0, None):
            return 0
        return _emit([f"ERR {_argv_cmd(rest)}: exit {exc.code}"], as_json)
    root = _root(chdir)
    try:
        if _streaming(ns, root):
            return 0
        if as_json and ns.cmd == "review":
            lines, digest = review_summary(root)
            print(json.dumps({"status": _status(lines), "lines": lines, "review": digest}, ensure_ascii=False))
            return 0 if _status(lines) != "ERR" else 1
        lines = _dispatch(ns, root)
    except SpecError as exc:
        lines = [f"ERR {err}" for err in exc.errors]
    except (yaml.YAMLError, RuntimeError, OSError, ValueError) as exc:
        _note_failure(root, ns.cmd, exc)
        lines = [f"ERR {ns.cmd}: {_one_line(str(exc))}"]
    return _emit(lines, as_json)


if __name__ == "__main__":
    sys.exit(main())
