"""Re-render on spec/asset save. Human loop; stdout is one result (or ERR) per cycle."""
from __future__ import annotations

import stat
import time
from pathlib import Path
from threading import Event

import yaml

from montaj import pipeline
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

POLL_S = 0.5
DEBOUNCE_S = 0.3


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def _assets_dir(root: Path) -> Path:
    spec = root / "montaj.yaml"
    if spec.is_file():
        try:
            data = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        if isinstance(data, dict) and isinstance(data.get("assets"), str):
            p = Path(data["assets"])
            return p if p.is_absolute() else root / p
    return root / "assets"


def _sig(path: Path, key: str) -> tuple[str, int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    return (key, st.st_mtime_ns, st.st_size)


def _fingerprint(root: Path) -> tuple:
    items: list[tuple[str, int, int]] = []
    got = _sig(root / "montaj.yaml", "montaj.yaml")
    if got:
        items.append(got)
    assets = _assets_dir(root)
    try:
        kids = sorted(assets.iterdir(), key=lambda p: p.name) if assets.is_dir() else []
    except FileNotFoundError:
        kids = []
    for p in kids:
        got = _sig(p, p.name)
        if got:
            items.append(got)
    return tuple(items)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _ok_line(root: Path, result, dt: float) -> str:
    path = Path(result.path)
    mb = path.stat().st_size / (1024 * 1024) if path.is_file() else 0.0
    n = result.rendered + result.cached
    w, h = result.size
    return (
        f"OK {_rel(root, path)} {result.seconds:.2f}s {w}x{h} {mb:.1f}MB "
        f"rendered {result.rendered}/{n} cached {result.cached}/{n} in {dt:.1f}s"
    )


def _cycle(root: Path, mode: str, out, log: Path) -> None:
    spec_path = root / "montaj.yaml"
    try:
        spec = load_spec(spec_path)
        resolve(spec, root)
        _log(log, f"render {mode}")
        t0 = time.perf_counter()
        result = pipeline.render(spec_path, mode)
        dt = time.perf_counter() - t0
        line = _ok_line(root, result, dt)
        out(line)
        _log(log, line)
    except SpecError as exc:
        for err in exc.errors:
            line = f"ERR {err}"
            out(line)
            _log(log, line)
    except Exception as exc:
        line = f"ERR watch: {' '.join(str(exc).split())}"
        out(line)
        _log(log, line)


def _report(out, log: Path, exc: BaseException) -> None:
    line = f"ERR watch: {' '.join(str(exc).split())}"
    out(line)
    _log(log, line)


def _wait(stop: Event | None, seconds: float) -> bool:
    """Sleep `seconds`. True when `stop` is set."""
    if stop is None:
        time.sleep(seconds)
        return False
    return stop.wait(seconds)


def watch(project_dir: Path | str, mode: str = "preview", out=print, stop: Event | None = None) -> None:
    root = Path(project_dir)
    log = root / "build" / "watch.log"
    _log(log, f"watch {mode}")
    try:
        prev = _fingerprint(root)
    except Exception as exc:
        prev = ()
        _report(out, log, exc)
    try:
        while True:
            if _wait(stop, POLL_S):
                break
            try:
                cur = _fingerprint(root)
            except Exception as exc:
                _report(out, log, exc)
                continue
            if cur == prev:
                continue
            halted = False
            while True:
                if _wait(stop, DEBOUNCE_S):
                    halted = True
                    break
                try:
                    newer = _fingerprint(root)
                except Exception as exc:
                    _report(out, log, exc)
                    continue
                if newer != cur:
                    cur = newer
                    continue
                prev = cur
                _log(log, "change")
                _cycle(root, mode, out, log)
                break
            if halted:
                break
    except KeyboardInterrupt:
        pass
    out("OK watch stopped")
    _log(log, "stopped")
