"""`montaj <cmd>`. Progress stays in build/*.log; stdout is the result line.

`-C DIR` and `--json` are accepted before or after the command. `--json` prints
one object, `{"status": "OK"|"WARN"|"ERR", "lines": [...]}`, instead of the text.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from montaj.audio import analyze as analyze_audio
from montaj.audio import compose as compose_audio
from montaj.audio import music as music_gen
from montaj.audio.mix import _find_src, check_lines, has_links
from montaj.audio.music import parse_duration
from montaj.capture import run as run_capture
from montaj.doctor import doctor, link_skill
from montaj.encode import CODECS, TARGETS, MediaInfo, VideoSettings, export, probe
from montaj.analysis import analyze_assets, describe_all
from montaj.gpu_policy import GpuStrictError, load_policy
from montaj.ingest import IngestError, ingest, ingest_rec
from montaj import caption_check, inspector, pacing, scopes, taste
from montaj.guards import content_guards
from montaj.qa import BLACK_PIX_TH, FREEZE_MIN_S, Expect, check, clip_overruns, subtract, union
from montaj.review.server import serve as serve_review
from montaj.review.summary import summary as review_summary
from montaj.sheet import (
    BASE_DIR, MAX_TILES, every_times, motion_sheet, pick_times, sheet, sheets, step_frames, strip_row,
    transition_rows,
)
from montaj.project import resolve_clip
from montaj.spec import INCOMING_TRANS, SpecError, _dur_frames, layer_warnings, load_spec, text_warnings
from montaj.timeline import Blank, Card, Layers, Sequence, Still, Wall, resolve, sequence_index, wall_static_spans
from montaj.watch import watch

RECIPES = Path(__file__).resolve().parents[1] / "recipes"


class _Usage(Exception):
    def __init__(self, message: str, as_json: bool = False):
        super().__init__(message)
        self.as_json = as_json


_COMMANDS = (
    "doctor", "new", "ingest", "ingest-rec", "validate", "render", "sheet", "check", "export", "music",
    "review", "watch", "taste", "capture", "inspect", "captions", "transcribe", "ref", "custom", "cut", "words", "spec",
    "skill",
)


class _Parser(argparse.ArgumentParser):
    """Argument errors raise. argparse's own `error` writes usage to stderr and exits 2."""

    def error(self, message: str) -> None:
        raise _Usage(message)


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="montaj")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor")

    skill = sub.add_parser("skill")
    skill_sub = skill.add_subparsers(dest="skill_cmd", required=True, metavar="link")
    skill_sub.add_parser("link")

    new = sub.add_parser("new")
    new.add_argument("dir", nargs="?")
    new.add_argument("--recipe", default=None)
    new.add_argument("--photos", default=None)
    new.add_argument("--list", action="store_true", help="one line per recipe: name, aspect, length, purpose")

    ing = sub.add_parser("ingest")
    ing.add_argument("--photos", default=None)
    ing.add_argument("--fast", action="store_true")  # normalise only; no analysis
    ing.add_argument("--list", action="store_true")  # one line per analysed asset; no work
    ing.add_argument("--matte", action="append", default=None, metavar="STEM")  # subject matte only (#36)

    ast = sub.add_parser("assets")  # a few lines per clip from what ingest wrote (#62)
    ast.add_argument("stem", nargs="?")

    rec = sub.add_parser("ingest-rec")
    rec.add_argument("video")
    rec.add_argument("--events", default=None)

    val = sub.add_parser("validate")
    val.add_argument("--expand", action="store_true", help="print the spec with styles, defaults and components expanded")

    ren = sub.add_parser("render")
    group = ren.add_mutually_exclusive_group()
    group.add_argument("--preview", action="store_true")
    group.add_argument("--final", action="store_true")
    ren.add_argument("--codec", choices=CODECS, default=None)

    sh = sub.add_parser("sheet")
    when = sh.add_mutually_exclusive_group()
    when.add_argument("--at", default=None)
    when.add_argument("--every", default=None)
    when.add_argument("--strip", default=None)
    when.add_argument("--transitions", action="store_true")
    sh.add_argument("--diff", action="store_true")
    sh.add_argument("--frames", type=int, default=8)
    sh.add_argument("--step", default="1f")
    sh.add_argument("--video", default=None)

    ck = sub.add_parser("check")
    ck.add_argument("--video", default=None)

    ex = sub.add_parser("export")
    ex.add_argument("--target", required=True, choices=tuple(TARGETS))

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

    cu = sub.add_parser("custom")
    cu_sub = cu.add_subparsers(dest="custom_cmd", required=True, metavar="test")
    cut = cu_sub.add_parser("test")
    cut.add_argument("file")
    cut.add_argument("--kind", choices=("shot", "effect", "transition"), default="shot")
    cut.add_argument("--params", default="{}")
    cut.add_argument("--size", default="270x480")
    cut.add_argument("--frames", type=int, default=6)

    cp = sub.add_parser("captions")
    mode = cp.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print", dest="print_", action="store_true")
    mode.add_argument("--gallery", action="store_true")
    mode.add_argument("--strip", type=int, metavar="CHUNK", default=None)
    rf = sub.add_parser("ref")
    rf.add_argument("video")
    tr = sub.add_parser("transcribe")
    tr.add_argument("clip")
    tr.add_argument("--lang", default=analyze_audio.DEFAULT_LANG)
    tr.add_argument("--prompt", default=None)
    tr.add_argument("--out", default=None)

    cu = sub.add_parser("cut")
    cu.add_argument("clip")
    cu.add_argument("--lang", default=analyze_audio.DEFAULT_LANG)
    cu.add_argument("--silence", type=float, default=None, help="cut gaps longer than this many seconds (0.35)")
    cu.add_argument("--print", dest="print_", action="store_true", help="list span, start, end, text")

    rv = sub.add_parser("review")
    rv.add_argument("--port", type=int, default=8765)
    rv.add_argument("--video", default=None)
    rv.add_argument("--summary", action="store_true")

    wa = sub.add_parser("watch")
    wa.add_argument("--final", action="store_true")

    ta = sub.add_parser("taste")
    ta.add_argument("action", nargs="?", choices=("add",))
    ta.add_argument("line", nargs="?")

    cap = sub.add_parser("capture")
    cap.add_argument("script")

    ins = sub.add_parser("inspect")
    ins.add_argument("--at", default=None)
    ins.add_argument("--shot", type=int, default=None)
    ins.add_argument("--range", nargs=2, metavar=("FROM", "TO"), default=None)
    ins.add_argument("--every", default=None)
    ins.add_argument("--fast", action="store_true")
    ins.add_argument("--problems", action="store_true")
    ins.add_argument("--track", default=None, metavar="SHOT.LAYER.PROP")
    ins.add_argument("--samples", type=int, default=inspector.TRACK_SAMPLES)

    sp = sub.add_parser("spec")
    sp.add_argument("topic", nargs="?")
    sp.add_argument("--search", default=None, metavar="WORD")

    wd = sub.add_parser("words")
    wd.add_argument("--grep", default=None, metavar="TEXT")

    sc = sub.add_parser("scopes")
    sc.add_argument("--image", action="store_true")
    mt = sub.add_parser("match")
    mt.add_argument("--to", type=int, required=True, metavar="SHOT")
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
    seen: set[str] = set()  # one mistake hit by many lines prints once; data lines (YAML, JSON) are kept as is
    lines = [ln for ln in lines if not (ln.startswith(("ERR", "WARN")) and (ln in seen or seen.add(ln)))]
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
    """Explicit path, else the newer of out/final.mp4 and out/preview.mp4; with neither, the newest
    `export` output out/<target>.mp4 (a project that only ran `montaj export` still has a film)."""
    if given:
        return _at(root, given)
    out = root / "out"
    cands = [p for p in (out / "final.mp4", out / "preview.mp4") if p.is_file()]
    if not cands:
        cands = [p for p in (out / f"{name}.mp4" for name in TARGETS) if p.is_file()]
    if not cands:
        raise FileNotFoundError(
            "video: no out/final.mp4, out/preview.mp4 or out/<target>.mp4 (render first, or pass --video)")
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
    lines.append(f"ERR doctor: {failed} failed (fix the FAIL lines above, then rerun montaj doctor)" if failed else "OK doctor")
    return lines


def _recipe_list() -> list[str]:
    """`name  aspect  length  purpose`, read from each recipe itself: `video.size` and its first comment line."""
    import re

    rows = []
    for src in sorted(RECIPES.glob("*.yaml")):
        text = src.read_text(encoding="utf-8")
        first = next((ln[1:].strip() for ln in text.splitlines() if ln.startswith("#")), "")
        purpose = re.split(r"\.(?:\s|$)", first, maxsplit=1)[0]  # its first sentence
        size = str(((yaml.safe_load(text) or {}).get("video") or {}).get("size", ""))
        w, _, h = size.partition("x")
        aspect = f"{int(w) // math.gcd(int(w), int(h))}:{int(h) // math.gcd(int(w), int(h))}" if (
            w.isdigit() and h.isdigit() and int(h)) else "?"
        m = re.search(r"~?\d+(?:\.\d+)?(?:[-–]\d+)?\s?s\b", purpose)
        rows.append(f"{src.stem:<20} {aspect:<5} {m.group(0) if m else '-':<7} {purpose}")
    return [*rows, f"OK {len(rows)} recipes"]


def _cmd_new(ns: argparse.Namespace, root: Path) -> list[str]:
    if ns.list:
        return _recipe_list()
    if ns.dir is None or ns.recipe is None:
        return ["ERR new: need DIR and --recipe NAME (or --list)"]
    dest = _at(root, ns.dir)
    spec_out = dest / "montaj.yaml"
    if spec_out.exists():
        raise FileExistsError(f"{spec_out.as_posix()} exists: refusing to overwrite")
    src = _recipe(ns.recipe)
    demo = src.with_name(f"{src.stem}-demo.sh")
    if ns.photos is None and not demo.is_file():
        return [f"ERR new: recipe {src.stem!r} ships no demo assets: pass --photos <folder>"]
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest / "montaj.yaml")
    local = src.with_suffix("")  # recipes/<name>/: the recipe's own draw files, beside the spec
    if local.is_dir():
        shutil.copytree(local, dest / local.name, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    if ns.photos is None:
        return _recipe_demo(demo, dest)
    assets = _assets_dir(dest, _assets_rel(src))
    ingest(_at(root, ns.photos), assets, sheet_path=dest / "build" / "ingest-sheet.jpg")
    shown = dest.as_posix()
    return [f"OK {shown}/montaj.yaml sheet={shown}/build/ingest-sheet.jpg"]


def _recipe_demo(demo: Path, dest: Path) -> list[str]:
    """Run ``recipes/<recipe>-demo.sh <dest>``: a recipe's generated stand-in assets, so it renders as-is."""
    log = dest / "build" / "new-demo.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    exe = Path(sys.executable).with_name("montaj")
    if exe.is_file():
        env.setdefault("MONTAJ", str(exe))
    with log.open("w") as fh:
        rc = subprocess.run(["bash", str(demo), str(dest)], stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    shown = dest.as_posix()
    if rc:
        return [f"ERR new: {demo.name} exited {rc}: see {log.as_posix()}"]
    return [f"OK {shown}/montaj.yaml demo={demo.name} log={log.as_posix()}"]


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


def _make_analyzer():
    """The GPU analyser; torch loads only here, and only when a row needs work."""
    from montaj.vision import GpuAnalyzer

    return GpuAnalyzer()


def _cmd_ingest(ns: argparse.Namespace, root: Path) -> list[str]:
    assets = _assets_dir(root, _assets_rel(root / "montaj.yaml"))
    if ns.list:
        lines = describe_all(assets)
        if not lines:
            return [f"ERR ingest: no photos or clips in {assets} (run montaj ingest --photos DIR)"]
        return [*lines, f"OK {len(lines)} assets"]
    if ns.photos is None and ns.fast:
        return ["ERR ingest: --fast needs --photos DIR"]
    if ns.matte:
        if ns.photos is not None or ns.fast:
            return ["ERR ingest: --matte runs alone (ingest --photos first)"]
        return _cmd_matte(ns.matte, root, assets)
    out = []
    if ns.photos is not None:
        infos = ingest(_at(root, ns.photos), assets, sheet_path=root / "build" / "ingest-sheet.jpg")
        out.append(f"{len(infos)} photos sheet=build/ingest-sheet.jpg")
    if not ns.fast:
        done = analyze_assets(assets, _make_analyzer, log=root / "build" / "ingest.log")
        out.append(f"analyzed={done.analyzed} cached={done.cached} -> {assets.name}/analysis.json")
    return ["OK " + " ".join(out)]


def _cmd_matte(stems: list[str], root: Path, assets: Path) -> list[str]:
    """`ingest --matte STEM`: compute (or reuse) each asset's subject matte on the GPU."""
    import torch

    from montaj.gpu_policy import resolve_device
    from montaj.matte import JUMP_WARN, make_mattes
    from montaj.project import find_clip, find_photo

    sources = []
    for stem in dict.fromkeys(stems):
        for find in (find_clip, find_photo):
            try:
                sources.append((stem, find(assets, stem)))
                break
            except FileNotFoundError:
                continue
        else:
            return [f"ERR ingest: no clip or photo {stem!r} in {assets.name}/ (montaj ingest --list)"]
    policy = load_policy()
    device = resolve_device(policy.device) if torch.cuda.is_available() else "cpu"
    log = root / "build" / "ingest.log"
    results = make_mattes(assets, sources, device=device, policy=policy, log=log)
    made = [r for r in results if r.made]
    frames = sum(r.frames for r in made)
    out = [f"WARN matte {r.stem}: mean frame-to-frame change {r.jump:.3f}; check the cutout with montaj sheet"
           for r in made if r.jump > JUMP_WARN]
    out.append(f"OK mattes made={len(made)} cached={len(results) - len(made)} frames={frames} -> {assets.name}/mattes/")
    return out


def _cmd_inspect(ns: argparse.Namespace, root: Path, as_json: bool) -> list[str]:
    if ns.track is not None:
        try:
            payload = inspector.track(root, ns.track, ns.samples)
        except ValueError as exc:
            return [f"ERR inspect: {exc}"]
        return [json.dumps(payload, ensure_ascii=False)] if as_json else inspector.lines(payload)
    modes = [ns.at is not None, ns.shot is not None, ns.range is not None, ns.problems]
    if sum(modes) != 1:
        return ["ERR inspect: give exactly one of --at T, --shot N, --range A B, --problems, --track S.L.PROP"]
    if ns.every is not None and ns.range is None:
        return ["ERR inspect: --every goes with --range"]
    payload = inspector.inspect(
        root, at=None if ns.at is None else _seconds(ns.at), shot=ns.shot,
        rng=ns.range, every=ns.every, fast=ns.fast, problems=ns.problems,
    )
    if as_json:
        # The payload is the product here; one frame prints bare, like the issue's example.
        frames = payload.get("frames")
        shown = frames[0] if frames is not None and ns.range is None else (frames if frames is not None else payload)
        return [json.dumps(shown, ensure_ascii=False)]
    return inspector.lines(payload)


def _cmd_validate(root: Path, expand: bool = False) -> list[str]:
    if expand:
        from montaj.expand import ExpandError, expand_file
        from montaj.spec import TEXT_STYLES

        try:
            data = expand_file(root / "montaj.yaml", TEXT_STYLES)
        except ExpandError as e:
            raise SpecError(e.lines((root / "montaj.yaml").read_text())) from e
        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip("\n").split("\n")
        return [*body, "OK montaj.yaml expanded (not validated; run validate)"]
    spec = load_spec(root / "montaj.yaml")
    tl = resolve(spec, root)
    seconds = tl.n_frames / tl.fps
    n_shots, n_segs = len(tl.shots), len(tl.segments)
    return [*text_warnings(spec), *layer_warnings(spec, root), f"OK montaj.yaml {tl.n_frames}f {seconds:.2f}s {n_shots} shots {n_segs} segments"]


def _cmd_render(ns: argparse.Namespace, root: Path) -> list[str]:
    from montaj.pipeline import render

    mode = "final" if ns.final else "preview"
    t0 = time.perf_counter()
    result = render(root / "montaj.yaml", mode, **({"video": VideoSettings(ns.codec)} if ns.codec else {}))
    dt = time.perf_counter() - t0
    mb = result.path.stat().st_size / (1024 * 1024)
    rel = result.path.resolve().relative_to(root.resolve()).as_posix()
    n = result.rendered + result.cached
    w, h = result.size
    audio = " audio" if result.audio else ""
    codec = f" {result.video}" if ns.codec else ""  # omitted when unset: the default line is unchanged
    return [
        f"OK {rel} {result.seconds:.2f}s {w}x{h} {mb:.1f}MB{codec} "
        f"rendered {result.rendered}/{n} cached {result.cached}/{n} in {dt:.1f}s gpu={result.gpu}{audio}"
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


def _cmd_motion_sheet(ns: argparse.Namespace, root: Path, video: Path, info: MediaInfo, log: Path) -> list[str]:
    """`sheet --strip / --transitions / --diff`: one image of consecutive frames or transition rows."""
    if ns.at or ns.every:
        raise ValueError("sheet --diff: combine it with --strip or --transitions, not --at/--every")
    spec_file = root / "montaj.yaml"
    tl = resolve(load_spec(spec_file), root) if spec_file.is_file() else None
    n_frames = int(round(info.duration * info.fps)) or 1
    note = ""
    if ns.strip is not None:
        center = int(round(_seconds(str(ns.strip), flag="--strip") * info.fps))
        rows = strip_row(tl, min(max(0, center), n_frames - 1), ns.frames, step_frames(ns.step, info.fps), n_frames, info.fps)
        name = "sheet-strip.jpg"
    else:
        if tl is None:
            raise FileNotFoundError("sheet --transitions: no montaj.yaml in the project")
        rows, total = transition_rows(tl, info.fps)
        if not rows:
            raise ValueError("sheet --transitions: the spec has no transitions")
        note = f" of {total}" if total > len(rows) else ""
        name = "sheet-transitions.jpg"
    if ns.diff:
        name = "sheet-diff.jpg"
    out = root / "build" / name
    motion_sheet(video, out, rows, base_dir=root / "build" / BASE_DIR, diff=ns.diff, log=log)
    shown = sum(len(r.tiles) for r in rows)
    n_rows = sum(1 for r in rows if r.label)
    return [f"OK build/{name} {n_rows} rows{note} {shown} frames"]


def _cmd_sheet(ns: argparse.Namespace, root: Path) -> list[str]:
    video = _video(root, ns.video)
    log = root / "build" / "sheet.log"
    info = probe(video, log=log)
    if ns.strip is not None or ns.transitions or ns.diff:
        return _cmd_motion_sheet(ns, root, video, info, log)
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
        elif isinstance(scene, Layers):
            # Only stills on screen: a designed hold, like a still shot.
            intended = all(layer.clip is None for layer in tl.shots[scene.shot].spec.layers)
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
    spans = _split_sequence_steps(tl, spans, fps)
    # `video.outro.hold` (#58): the frames after the fade are black by design, so a freeze there is too.
    fade = tl.plan(tl.n_frames - 1).fade_out if tl.n_frames else None
    if fade is not None and fade[0] + fade[1] < tl.n_frames:
        spans.append(((fade[0] + fade[1]) / fps, tl.n_frames / fps))
    return spans


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
    spec = load_spec(spec_file)
    audio = spec.audio
    if audio is None or not (audio.clips or audio.edit or audio.duck or has_links(spec)):
        return []
    return check_lines(json.loads(stats_file.read_text()), loudnorm=audio.loudnorm)


def _join_lines(tl, root: Path, video: Path) -> list[str]:
    """Click check at every cut between two linked clip shots (the joins `cuts:` makes)."""
    from montaj.audio.mix import SR, plan_audio
    from montaj.qa import join_lines
    from montaj.spec import _parse_beat0

    if not has_links(tl.spec) or tl.spec.audio is None:
        return []
    plan = plan_audio(tl.spec, root, tl.markers, tl.fps, tl.spec.video.bpm,
                      _parse_beat0(tl.spec.video.beat0, tl.fps, []) or 0.0, [], shots=tl.shots)
    joins = [link.a / SR for link in (plan.links if plan else ()) if link.xin]
    return join_lines(video, joins, log=root / "build" / "qa.log")


def _sfx_lines(tl, root: Path) -> list[str]:
    """`INFO sfx=…` and a WARN when two emphasis / auto sounds peak within 80 ms."""
    from montaj.audio.mix import plan_audio
    from montaj.emphasis import sfx_lines
    from montaj.spec import _parse_beat0

    if tl.spec.audio is None or not tl.spec.audio.clips:
        return []
    plan = plan_audio(tl.spec, root, tl.markers, tl.fps, tl.spec.video.bpm,
                      _parse_beat0(tl.spec.video.beat0, tl.fps, []) or 0.0, [], shots=tl.shots)
    return sfx_lines(plan) if plan else []


def _clip_overruns(root: Path) -> list[str]:
    """Clip shots whose timeline read runs past the source. No spec means no new lines."""
    spec_file = root / "montaj.yaml"
    if not spec_file.is_file():
        return []
    tl = resolve(load_spec(spec_file), root)
    return clip_overruns(tl, _assets_dir(root, tl.spec.assets), spec_dir=root, log=root / "build" / "qa.log")


def _cmd_words(ns: argparse.Namespace, root: Path) -> list[str]:
    """`i  start  word` for the spec's words (audio.markers, else captions.source): the numbers `emphasis:` takes."""
    from montaj.emphasis import load_words
    from montaj.spec import Spec

    data = yaml.safe_load((root / "montaj.yaml").read_text(encoding="utf-8")) or {}
    data.pop("emphasis", None)  # the words are listed before emphasis is written
    errors: list[str] = []
    spec = Spec.model_validate(data)
    words = load_words(spec, root, errors)
    if errors:
        return [f"ERR {e}" for e in errors]
    if ns.grep is not None:
        n = len(ns.grep.split())
        keep = {words[j].i for k in range(len(words)) if _grep_hit(words, k, ns.grep) for j in range(k, min(k + n, len(words)))}
        if not keep:
            return [f"ERR words: no word matches {ns.grep!r}: try a shorter text"]
        words = [w for w in words if w.i in keep]
    return [f"{w.i:>4}  {w.s:6.2f}  {w.w}" for w in words] + [f"OK words: {len(words)}"]


def _grep_hit(words, k: int, text: str) -> bool:
    """Whether the words run starting at position `k` equals `text` (every occurrence, not just the first)."""
    from montaj.emphasis import _norm

    want = [t for t in (_norm(x) for x in text.split()) if t]
    return [_norm(w.w) for w in words[k:k + len(want)]] == want


def _findings(lines: list[str], captions: list) -> list[dict]:
    """`check --json` findings: every WARN and ERR line and each unmet `expect`, in output order. Caption
    findings carry their spec path, time span, text, value and limit; the rest only kind, level and message."""
    by_line = {f.line: f.as_dict() for f in captions}
    out = []
    for line in lines[:-1]:  # the last line is the summary
        level, _, rest = line.partition(" ")
        kind = rest.split(" ", 1)[0].rstrip(":")
        if line in by_line:
            out.append(by_line[line])
        elif level in ("WARN", "ERR") or (level == "INFO" and kind == "expect"):
            out.append({"kind": kind, "level": level, "path": None, "start": None, "end": None, "text": None,
                        "value": None, "limit": None, "message": line})
    return out


def _cmd_check(ns: argparse.Namespace, root: Path, numbers: dict | None = None,
               findings: list | None = None) -> list[str]:
    expects, loud_i, loud_tol = _check_overrides(root)
    black = union([*_intended_black(root), *(item.span for item in expects if item.kind == "black")])
    freeze = union([*_intended_freeze(root), *(item.span for item in expects if item.kind == "frozen")])
    video = _video(root, ns.video)
    # out/<target>.mp4 is checked against its export preset: codec, bit depth and LUFS target.
    preset = TARGETS.get(video.stem) if video.parent.resolve() == (root / "out").resolve() else None
    if preset is not None and loud_i is None:
        loud_i = preset.lufs
    lines = check(
        video, freeze,
        log=root / "build" / "qa.log", av_tol=_av_tol(root),
        intended_black=black, expect=expects,
        loudness_target=loud_i, loudness_tol=loud_tol,
        want=preset.video if preset is not None else None,
    )
    spec_file = root / "montaj.yaml"
    spec = load_spec(spec_file) if spec_file.is_file() else None
    if spec is not None and spec.check is not None:
        tl = resolve(spec, root)
        lines += content_guards(tl, _video(root, ns.video), root / "build" / "guards",
                                log=root / "build" / "qa.log")
    lines += _audio_mix_lines(root)
    lines.extend(_clip_overruns(root))
    caps: list = []
    if spec is not None:
        tl = resolve(spec, root)
        lines += _join_lines(tl, root, _video(root, ns.video))
        lines += _sfx_lines(tl, root)
        nums = pacing.metrics(tl, root, _video(root, ns.video), log=root / "build" / "qa.log")
        if numbers is not None:
            numbers.update(nums)
        lines += pacing.lines(nums, spec.check.pacing if spec.check else None,
                              fps=tl.fps, bpm=tl.spec.video.bpm)
        lines += caption_check.lines(tl, _video(root, ns.video), root / "build" / "captions-check",
                                     log=root / "build" / "qa.log", found=caps)
        lines += scopes.check_lines(root, spec)
    errs = sum(line.startswith("ERR") for line in lines)
    warns = sum(line.startswith("WARN") for line in lines)
    if errs:
        lines.append(f"ERR check: {errs} guard(s) unavailable (see the ERR lines above; uv sync --extra guards adds the optional ones)")
    else:
        lines.append(f"WARN check: {warns} warnings" if warns else "OK check")
    if findings is not None:
        findings += _findings(lines, caps)
    return lines


def _cmd_scopes(ns: argparse.Namespace, root: Path, payload: dict | None = None) -> list[str]:
    stats, sheet = scopes.scopes(root, image=ns.image)
    if payload is not None:
        payload["shots"] = [scopes.as_dict(s) for s in stats]
    lines = [scopes.line(s) for s in stats]
    tail = f" sheet {sheet}" if sheet is not None else ""
    lines.append(f"OK scopes {len(stats)} shots{tail}")
    return lines


def _cmd_match(ns: argparse.Namespace, root: Path) -> list[str]:
    rows, target = scopes.match(root, ns.to)
    lines = [scopes.match_line(row, target) for row in rows]
    lines.append(f"OK montaj.yaml: matched {len(rows)} shots to shot {ns.to}")
    return lines


def _cmd_export(ns: argparse.Namespace, root: Path) -> list[str]:
    """With a spec, the target is rendered from the float compositor (10-bit or dithered 8-bit,
    all on the GPU); without one, the newest out/ file is copied (master) or transcoded."""
    preset = TARGETS[ns.target]
    spec_file = root / "montaj.yaml"
    if spec_file.is_file():
        from montaj.pipeline import render

        t0 = time.perf_counter()
        result = render(spec_file, "final", target=preset)
        dt = time.perf_counter() - t0
        mb = result.path.stat().st_size / (1024 * 1024)
        n = result.rendered + result.cached
        w, h = result.size
        return [
            f"OK out/{ns.target}.mp4 {result.seconds:.2f}s {w}x{h} {mb:.1f}MB {preset.describe()} "
            f"rendered {result.rendered}/{n} cached {result.cached}/{n} in {dt:.1f}s gpu={result.gpu}"
        ]
    target = root / "out" / f"{ns.target}.mp4"
    export(_export_source(root), target, ns.target, log=root / "build" / "export.log")
    mb = target.stat().st_size / (1024 * 1024)
    # Loudness was fixed when the file was muxed, so a file export does not claim a LUFS target.
    width = f" w{preset.width}" if preset.width else ""
    how = "copy" if ns.target == "master" else f"{preset.video.describe()}{width} aac{preset.audio_kbps}k"
    return [f"OK out/{ns.target}.mp4 {mb:.1f}MB target={ns.target} {how} {preset.tags} from file"]


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
        lines.append(f"ERR music gen: {len(exc.files)}/{exc.n} candidates ({exc.reason}; rerun to get the rest)")
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


def _cmd_captions(ns: argparse.Namespace, root: Path) -> list[str]:
    if ns.gallery:
        from montaj.capgallery import gallery

        path, n = gallery(root)
        return [f"OK {_rel(root, path)} {n} presets x 2 frames"]
    if ns.strip is not None:
        from montaj.capgallery import strip

        path, n = strip(root, ns.strip)
        return [f"OK {_rel(root, path)} chunk {ns.strip} {n} frames"]
    from montaj.captions import format_chunks, plan

    chunks, fps = plan(root / "montaj.yaml")
    return [*format_chunks(chunks, fps).splitlines(), f"OK {len(chunks)} chunks"]


def _cmd_ref(ns: argparse.Namespace, root: Path) -> list[str]:
    from montaj.ref import study

    video = _at(root, ns.video)
    if not video.is_file():
        return [f"ERR ref: {ns.video} not found: pass a video file"]
    path, stats, shown = study(video, root / "build" / f"ref-{video.stem}.jpg", log=root / "build" / "ref.log")
    return [f"OK {_rel(root, path)} {stats.duration:.1f}s {stats.shots} shots {len(stats.cuts)} cuts "
            f"cut_rate={stats.per_10s:.1f}/10s median_shot={stats.median_shot:.2f}s {shown} frames"]


def _cmd_transcribe(ns: argparse.Namespace, root: Path) -> list[str]:
    from montaj.captions import transcribe

    clip = _at(root, ns.clip)
    if not clip.is_file():  # an assets id
        clip = resolve_clip(root, _assets_dir(root, _assets_rel(root / "montaj.yaml")), ns.clip)
    out = _at(root, ns.out) if ns.out else _assets_dir(root, _assets_rel(root / "montaj.yaml")) / f"{clip.stem}.words.json"
    path = transcribe(clip, out, lang=ns.lang, prompt=ns.prompt, log=root / "build" / "transcribe.log")
    n = len(json.loads(path.read_text(encoding="utf-8"))["words"])
    return [f"OK {_rel(root, path)} {n} words"]


def _cmd_custom_test(ns: argparse.Namespace, root: Path) -> list[str]:
    """Run one custom `draw(ctx)` at a small size: shape, range, determinism, ms/frame, peak VRAM, a strip."""
    import torch
    from PIL import Image

    from montaj.gpu_policy import load_policy, resolve_device
    from montaj.render.custom import probe

    path = _at(root, ns.file)
    if not path.is_file():
        return [f"ERR custom: {ns.file} not found"]
    try:
        params = json.loads(ns.params)
        W, H = (int(v) for v in ns.size.lower().split("x"))
    except (ValueError, TypeError):
        return ["ERR custom: --params must be a JSON object and --size WxH"]
    if not isinstance(params, dict):
        return ["ERR custom: --params must be a JSON object"]
    device = resolve_device(load_policy().device) if torch.cuda.is_available() else "cpu"
    rel = _rel(root, path)
    base, rel = (path.parent, path.name) if Path(rel).is_absolute() else (root, rel)
    frames, ms, peak = probe(base, rel, params, kind=ns.kind, W=W, H=H, device=device, frames=ns.frames)
    strip = torch.cat([f.clamp(0, 1) for f in frames], dim=2)
    pixels = (strip.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
    out = root / "build" / f"custom-{path.stem}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(out, quality=90)
    return [f"OK {rel} {ms:.1f}ms/frame peak {peak / 2**20:.0f}MB {device} strip {_rel(root, out)}"]


def _cmd_cut(ns: argparse.Namespace, root: Path) -> list[str]:
    from montaj import cuts
    from montaj.captions import transcribe

    assets = _assets_dir(root, _assets_rel(root / "montaj.yaml"))
    clip = _at(root, ns.clip)
    if not clip.is_file():  # an assets id
        clip = resolve_clip(root, assets, ns.clip)
    target = assets / f"{clip.stem}.cuts.json"
    if ns.print_ and ns.silence is None and target.is_file():
        return cuts.table(cuts.read_plan(target))
    words = assets / f"{clip.stem}.words.json"
    log = root / "build" / "cut.log"
    if not words.is_file():
        transcribe(clip, words, lang=ns.lang, log=log)
    silence = cuts.SILENCE_S if ns.silence is None else ns.silence
    if not silence > 0:
        raise ValueError("--silence must be above 0 seconds")
    plan = cuts.build(clip, words, lang=ns.lang, silence=silence, log=log)
    target.write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if ns.print_:
        return cuts.table(plan)
    return [f"OK {_rel(root, target)} {plan['duration']:.1f}s -> {plan['kept']:.1f}s  {len(plan['spans'])} spans  "
            f"silence>{silence:g}s removed  fillers={plan['fillers']}"]


BED_HEADROOM_DB = -6.0  # a composed bed's peak: room for voice and SFX on top (#64)


def _is_bed(root: Path, out: Path) -> bool:
    """`out` is (or will be) the spec's music bed: no spec yet, no `audio.track`, or it is the track.

    It is not when the spec's track is another file, or an `audio.clips` src names it, found the way
    `mix._clip_src` finds it: `synth:<name>` in `sfx/`, else the spec dir, then the assets dir, with
    the extension optional. Raw YAML, not `load_spec`: the spec often names a track that does not
    exist yet.
    """
    try:
        raw = yaml.safe_load((root / "montaj.yaml").read_text())
    except (OSError, yaml.YAMLError):
        return True
    raw = raw if isinstance(raw, dict) else {}
    audio = raw.get("audio") if isinstance(raw.get("audio"), dict) else {}
    track = audio.get("track")
    if isinstance(track, str) and track:
        return (root / track).resolve() == out
    assets = raw.get("assets") if isinstance(raw.get("assets"), str) else "assets"
    clips = audio.get("clips") if isinstance(audio.get("clips"), list) else []
    for c in clips:
        src = c.get("src") if isinstance(c, dict) else None
        if not isinstance(src, str):
            continue
        dirs = [root / "sfx"] if src.startswith("synth:") else [root, root / assets]
        name = src.removeprefix("synth:")
        found = next((f for d in dirs if (f := _find_src(d, name)) is not None), None)
        # An existing file is the clip `_clip_src` would pick; before compose writes it, any candidate.
        bases = [found.resolve()] if found is not None else [(d / name).resolve() for d in dirs]
        if {out, out.with_suffix("")} & set(bases):  # `src` may leave off the extension
            return False
    return True


def _cmd_music_compose(ns: argparse.Namespace, root: Path) -> list[str]:
    out = _at(root, ns.out)
    try:
        result = compose_audio.compose(
            _at(root, ns.script),
            out,
            root=root,
            timeout=ns.timeout,
            log=root / "build" / "music.log",
            headroom_db=BED_HEADROOM_DB if _is_bed(root, out) else None,
        )
    except compose_audio.ComposeError as exc:
        return [f"ERR music compose: {exc}"]
    lines: list[str] = []
    if result.scaled_db < 0:
        lines.append(f"INFO music compose: bed turned down {-result.scaled_db:.1f} dB to "
                     f"{BED_HEADROOM_DB:.0f} dBFS peak, room for voice/SFX (loudnorm at render restores the level)")
    if result.peak_dbfs >= -0.1:
        lines.append(f"WARN music compose: clipping (peak {compose_audio.format_db(result.peak_dbfs)} dBFS)")
    lines.append(
        f"OK {_rel(root, result.path)} {result.duration:.2f}s {result.sample_rate}Hz {result.channels}ch "
        f"peak {compose_audio.format_db(result.peak_dbfs)}dBFS {result.lufs:.1f}LUFS"
    )
    return lines


def _cmd_capture(ns: argparse.Namespace, root: Path) -> list[str]:
    result = run_capture(_at(root, ns.script), root, log=root / "build" / "capture.log", policy=load_policy())
    mb = result.path.stat().st_size / (1024 * 1024)
    w, h = result.size
    return [f"OK {_rel(root, result.path)} {result.seconds:.2f}s {w}x{h} {mb:.1f}MB"]


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
    if ns.cmd == "skill":
        return link_skill()
    if ns.cmd == "new":
        return _cmd_new(ns, root)
    if ns.cmd == "ingest":
        return _cmd_ingest(ns, root)
    if ns.cmd == "ingest-rec":
        return _cmd_ingest_rec(ns, root)
    if ns.cmd == "validate":
        return _cmd_validate(root, ns.expand)
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
    if ns.cmd == "custom":
        return _cmd_custom_test(ns, root)
    if ns.cmd == "captions":
        return _cmd_captions(ns, root)
    if ns.cmd == "ref":
        return _cmd_ref(ns, root)
    if ns.cmd == "transcribe":
        return _cmd_transcribe(ns, root)
    if ns.cmd == "cut":
        return _cmd_cut(ns, root)
    if ns.cmd == "words":
        return _cmd_words(ns, root)
    if ns.cmd == "assets":
        from montaj.footage import summary

        return summary(_assets_dir(root, _assets_rel(root / "montaj.yaml")), ns.stem)
    if ns.cmd == "spec":
        from montaj.specref import run as spec_ref

        return spec_ref(ns.topic, ns.search)
    if ns.cmd == "review":
        return review_summary(root)[0]
    if ns.cmd == "taste":
        return _cmd_taste(ns)
    if ns.cmd == "capture":
        return _cmd_capture(ns, root)
    if ns.cmd == "scopes":
        return _cmd_scopes(ns, root)
    if ns.cmd == "match":
        return _cmd_match(ns, root)
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
        cmd = _argv_cmd(rest)
        return _emit([f"ERR {cmd}: {_one_line(str(exc))} (montaj {'' if cmd == 'cli' else cmd + ' '}--help)"], as_json)
    except SystemExit as exc:
        # `--help` prints itself and exits 0. Any other exit becomes one ERR line.
        if exc.code in (0, None):
            return 0
        return _emit([f"ERR {_argv_cmd(rest)}: exit {exc.code} (montaj --help lists commands)"], as_json)
    root = _root(chdir)
    try:
        if _streaming(ns, root):
            return 0
        if as_json and ns.cmd == "review":
            lines, digest = review_summary(root)
            print(json.dumps({"status": _status(lines), "lines": lines, "review": digest}, ensure_ascii=False))
            return 0 if _status(lines) != "ERR" else 1
        if as_json and ns.cmd == "check":
            numbers: dict = {}
            found: list = []
            lines = _cmd_check(ns, root, numbers, found)
            print(json.dumps({"status": _status(lines), "lines": lines, "findings": found, "pacing": numbers},
                             ensure_ascii=False))
            return 0 if _status(lines) != "ERR" else 1
        if as_json and ns.cmd == "scopes":
            shots: dict = {}
            lines = _cmd_scopes(ns, root, shots)
            print(json.dumps({"status": _status(lines), "lines": lines, **shots}, ensure_ascii=False))
            return 0 if _status(lines) != "ERR" else 1
        if ns.cmd == "inspect":
            lines = _cmd_inspect(ns, root, as_json)
            if as_json and not lines[0].startswith("ERR"):
                print(lines[0])
                return 0
        else:
            lines = _dispatch(ns, root)
    except SpecError as exc:
        lines = [f"ERR {err}" for err in exc.errors]
    except GpuStrictError as exc:
        _note_failure(root, ns.cmd, exc)
        lines = [f"ERR {_one_line(str(exc))}"]  # `ERR gpu-strict: <what> would run on the CPU: <hint>`
    except (yaml.YAMLError, RuntimeError, OSError, ValueError) as exc:
        _note_failure(root, ns.cmd, exc)
        lines = [f"ERR {ns.cmd}: {_one_line(str(exc))}"]
    return _emit(lines, as_json)


if __name__ == "__main__":
    sys.exit(main())
