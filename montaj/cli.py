"""`montaj <cmd>`. Progress stays in build/*.log; stdout is the result line.

`-C DIR` and `--json` are accepted before or after the command. `--json` prints
one object, `{"status": "OK"|"WARN"|"ERR", "lines": [...]}`, instead of the text.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

from montaj.doctor import doctor
from montaj.encode import export, probe
from montaj.ingest import ingest
from montaj.qa import check
from montaj.sheet import MAX_TILES, pick_times, sheet
from montaj.spec import SpecError, load_spec
from montaj.timeline import Still, Wall, resolve, wall_static_spans

RECIPES = Path(__file__).resolve().parents[1] / "recipes"


class _Usage(Exception):
    def __init__(self, message: str, as_json: bool = False):
        super().__init__(message)
        self.as_json = as_json


_COMMANDS = ("doctor", "new", "ingest", "validate", "render", "sheet", "check", "export")


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

    sub.add_parser("validate")

    ren = sub.add_parser("render")
    group = ren.add_mutually_exclusive_group()
    group.add_argument("--preview", action="store_true")
    group.add_argument("--final", action="store_true")

    sh = sub.add_parser("sheet")
    sh.add_argument("--at", default=None)
    sh.add_argument("--video", default=None)

    ck = sub.add_parser("check")
    ck.add_argument("--video", default=None)

    ex = sub.add_parser("export")
    ex.add_argument("--target", required=True, choices=("master", "whatsapp"))
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
    return [
        f"OK {rel} {result.seconds:.2f}s {w}x{h} {mb:.1f}MB "
        f"rendered {result.rendered}/{n} cached {result.cached}/{n} in {dt:.1f}s"
    ]


def _cmd_sheet(ns: argparse.Namespace, root: Path) -> list[str]:
    video = _video(root, ns.video)
    log = root / "build" / "sheet.log"
    info = probe(video, log=log)
    at = None
    if ns.at:
        at = [float(part) for part in str(ns.at).split(",") if part.strip()]
    # Pass the times in, so sheet does not probe into the process cwd's build/.
    # sheet() keeps at most MAX_TILES. Count the same list it will draw.
    times = pick_times(video, at, n=MAX_TILES, info=info)[:MAX_TILES]
    sheet(video, root / "build" / "sheet.jpg", at=times, log=log)
    n_frames = int(round(info.duration * info.fps)) or 1
    shown = len({min(max(0, int(round(t * info.fps))), n_frames - 1) for t in times})
    return [f"OK build/sheet.jpg {shown} frames"]


def _intended_freeze(root: Path) -> list[tuple[float, float]]:
    """Time spans whose picture is designed not to move: Still scenes, and wall
    frames where both camera tracks hold (before the first key, after the last,
    or between keys that resolve to the same spec identity). A freeze inside a
    transition or a moving wall is still a defect."""
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
        intended = isinstance(scene, Still) or (isinstance(scene, Wall) and wall_hold(f))
        if intended:
            if start is None:
                start = f
        elif start is not None:
            spans.append((start / fps, f / fps))
            start = None
    if start is not None:
        spans.append((start / fps, tl.n_frames / fps))
    return spans


def _cmd_check(ns: argparse.Namespace, root: Path) -> list[str]:
    lines = check(_video(root, ns.video), _intended_freeze(root), log=root / "build" / "qa.log")
    warns = sum(line.startswith("WARN") for line in lines)
    lines.append(f"WARN check: {warns} warnings" if warns else "OK check")
    return lines


def _cmd_export(ns: argparse.Namespace, root: Path) -> list[str]:
    target = root / "out" / f"{ns.target}.mp4"
    export(_export_source(root), target, ns.target, log=root / "build" / "export.log")
    mb = target.stat().st_size / (1024 * 1024)
    return [f"OK out/{ns.target}.mp4 {mb:.1f}MB"]


def _dispatch(ns: argparse.Namespace, root: Path) -> list[str]:
    if ns.cmd == "doctor":
        return _cmd_doctor()
    if ns.cmd == "new":
        return _cmd_new(ns, root)
    if ns.cmd == "ingest":
        return _cmd_ingest(ns, root)
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
        lines = _dispatch(ns, root)
    except SpecError as exc:
        lines = [f"ERR {err}" for err in exc.errors]
    except (yaml.YAMLError, RuntimeError, OSError, ValueError) as exc:
        _note_failure(root, ns.cmd, exc)
        lines = [f"ERR {ns.cmd}: {_one_line(str(exc))}"]
    return _emit(lines, as_json)


if __name__ == "__main__":
    sys.exit(main())
