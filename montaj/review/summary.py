"""`review.json` store and the agent's compact summary."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from montaj.spec import load_spec
from montaj.timeline import ResolvedShot, Timeline, resolve

REVIEW_NAME = "review.json"


class ReviewLoadError(Exception):
    """`review.json` exists but cannot be read or parsed. Mutations must not replace it."""


def _sec(frames: float, fps: float) -> float:
    return round(float(frames) / float(fps), 2)


def shot_label(sh: ResolvedShot) -> str:
    spec = sh.spec
    if spec.wall is not None:
        return "wall"
    if spec.clip is not None:
        return str(spec.clip)
    if spec.photo is not None:
        return str(spec.photo)
    return "blank"


def first_text(tl: Timeline, sh: ResolvedShot) -> str:
    """First line whose appearance (`at`) falls in the shot. Block overlap is half-open."""
    found: tuple[float, str] | None = None
    for block in tl.texts:
        if not (block.from_ < sh.end and block.to > sh.start):
            continue
        for line in block.lines:
            at = float(line.at)
            if at < sh.start or at >= sh.end:
                continue
            if found is None or at < found[0]:
                found = (at, line.text)
    return found[1] if found else ""


def timeline_shots(tl: Timeline) -> list[dict]:
    """Strip rows: exact frames for seek/assign, 2 dp seconds for display / review.json."""
    fps = float(tl.fps)
    out = []
    for sh in tl.shots:
        out.append({
            "index": sh.index,
            "start_f": sh.start,
            "end_f": sh.end,
            "start": _sec(sh.start, fps),
            "end": _sec(sh.end, fps),
            "label": shot_label(sh),
            "text": first_text(tl, sh),
        })
    return out


def empty_review(spec_sha: str, video: str, shots: list[dict]) -> dict:
    return {
        "version": 1,
        "spec_sha": spec_sha,
        "video": video,
        "updated": _now(),
        "shots": [_blank_shot(s) for s in shots],
        "general": [],
    }


def _blank_shot(s: dict) -> dict:
    return {
        "index": s["index"],
        "start": s["start"],
        "end": s["end"],
        "label": s["label"],
        "status": None,
        "comments": [],
    }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spec_sha(project_dir: Path) -> str:
    path = Path(project_dir) / "montaj.yaml"
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(project_dir: Path) -> dict | None:
    path = Path(project_dir) / REVIEW_NAME
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReviewLoadError(f"review.json: cannot read ({exc})") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReviewLoadError(f"review.json: invalid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise ReviewLoadError("review.json: expected an object")
    return data


def merge_review(raw: dict | None, shots: list[dict], sha: str, video: str) -> dict:
    """Keep notes by shot index. A label change sets stale and leaves the notes."""
    if raw is None:
        rev = empty_review(sha, video, shots)
        rev["spec_sha"] = sha
        return rev
    old_by_i = {}
    for sh in raw.get("shots") or []:
        if isinstance(sh, dict) and isinstance(sh.get("index"), int):
            old_by_i[sh["index"]] = sh
    merged = []
    for s in shots:
        old = old_by_i.get(s["index"])
        if old is None:
            merged.append(_blank_shot(s))
            continue
        label = s["label"]
        stale = bool(old.get("stale")) or (old.get("label") != label)
        item = {
            "index": s["index"],
            "start": s["start"],
            "end": s["end"],
            "label": label,
            "status": old.get("status"),
            "comments": list(old.get("comments") or []),
        }
        if stale:
            item["stale"] = True
        merged.append(item)
    return {
        "version": 1,
        "spec_sha": raw.get("spec_sha") or sha,
        "video": video or raw.get("video") or "",
        "updated": raw.get("updated") or _now(),
        "shots": merged,
        "general": list(raw.get("general") or []),
    }


def atomic_write(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def save_review(project_dir: Path, data: dict) -> None:
    data = dict(data)
    data["updated"] = _now()
    atomic_write(Path(project_dir) / REVIEW_NAME, data)


def load_timeline(project_dir: Path) -> Timeline:
    root = Path(project_dir)
    return resolve(load_spec(root / "montaj.yaml"), root)


def load_merged(project_dir: Path, video: str = "") -> tuple[list[dict], dict]:
    root = Path(project_dir)
    tl = load_timeline(root)
    shots = timeline_shots(tl)
    sha = spec_sha(root)
    rev = merge_review(load_raw(root), shots, sha, video)
    return shots, rev


def _status_word(status: object) -> str:
    if status == "rejected":
        return "REJECT"
    if status == "approved":
        return "APPROVE"
    return ""


def _quoted(parts: list[str]) -> str:
    return "; ".join(json.dumps(p, ensure_ascii=False) for p in parts)


def _comment_texts(items: list) -> list[str]:
    out = []
    for c in items:
        if isinstance(c, dict) and c.get("text"):
            out.append(str(c["text"]))
    return out


def _count(rev: dict) -> tuple[int, int, int]:
    a = r = c = 0
    for sh in rev.get("shots") or []:
        st = sh.get("status")
        if st == "approved":
            a += 1
        elif st == "rejected":
            r += 1
        c += len(sh.get("comments") or [])
    c += len(rev.get("general") or [])
    return a, r, c


def format_lines(rev: dict) -> list[str]:
    lines: list[str] = []
    for sh in rev.get("shots") or []:
        comments = _comment_texts(sh.get("comments") or [])
        word = _status_word(sh.get("status"))
        if not word and not comments:
            continue
        start = f"{float(sh.get('start', 0)):.2f}"
        end = f"{float(sh.get('end', 0)):.2f}"
        bits = [f"#{sh.get('index', 0)} {start}-{end} {sh.get('label', '')}"]
        if word:
            bits.append(word)
        if comments:
            bits.append(_quoted(comments))
        lines.append(" ".join(bits))
    general = _comment_texts(rev.get("general") or [])
    if general:
        lines.append("general " + _quoted(general))
    a, r, c = _count(rev)
    lines.append(f"OK review {a} approved {r} rejected {c} comments")
    return lines


def digest(rev: dict) -> dict:
    a, r, c = _count(rev)
    shots = []
    for sh in rev.get("shots") or []:
        comments = sh.get("comments") or []
        if sh.get("status") or comments or sh.get("stale"):
            shots.append(sh)
    return {
        "approved": a,
        "rejected": r,
        "comments": c,
        "shots": shots,
        "general": list(rev.get("general") or []),
    }


def summary(project_dir: Path | str) -> tuple[list[str], dict]:
    """Agent view of `review.json`. Empty file → only the OK counts line."""
    root = Path(project_dir)
    try:
        raw = load_raw(root)
    except ReviewLoadError as exc:
        return [f"ERR {exc}"], {"error": str(exc)}
    try:
        _shots, rev = load_merged(root, "")
    except ReviewLoadError as exc:
        return [f"ERR {exc}"], {"error": str(exc)}
    except Exception:
        rev = raw if raw is not None else empty_review("", "", [])
    return format_lines(rev), digest(rev)
