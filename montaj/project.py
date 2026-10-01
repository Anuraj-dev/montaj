"""Asset lookup and segment-hash cache keys."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from montaj.timeline import Segment, Timeline

_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_photo(assets_dir: Path, photo_id: str) -> Path:
    assets_dir = Path(assets_dir)
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"'{photo_id}' not in {assets_dir} (have: none)")
    matches = [
        p
        for p in assets_dir.iterdir()
        if p.is_file() and p.stem == photo_id and p.suffix.lower() in _EXTS
    ]
    if not matches:
        stems = sorted(
            {
                p.stem
                for p in assets_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _EXTS
            }
        )
        have = ", ".join(stems) if stems else "none"
        raise FileNotFoundError(f"'{photo_id}' not in {assets_dir} (have: {have})")
    order = {ext: i for i, ext in enumerate(_EXTS)}
    matches.sort(key=lambda p: (order.get(p.suffix.lower(), 99), p.name))
    return matches[0]


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scene_shots(scene: object) -> set[int]:
    from montaj.timeline import Still, Trans, Wall

    if isinstance(scene, Still):
        return {scene.shot}
    if isinstance(scene, Wall):
        return {scene.shot}
    if isinstance(scene, Trans):
        return _scene_shots(scene.a) | _scene_shots(scene.b)
    raise TypeError(f"unknown scene {type(scene)!r}")


def _shot_photos(shot) -> list[str]:
    if shot.photo is not None:
        return [shot.photo]
    if shot.wall is not None:
        return [p.photo for p in shot.wall.prints]
    return []


def segment_hash(
    tl: Timeline,
    seg: Segment,
    asset_sha: dict[str, str],
    engine_version: str,
) -> str:
    referenced: set[int] = set()
    plans = []
    for f in range(seg.start, seg.end):
        plan = tl.plan(f)
        plans.append(asdict(plan))
        referenced |= _scene_shots(plan.scene)
    shots = {
        str(i): tl.shots[i].spec.model_dump(mode="json", by_alias=True)
        for i in sorted(referenced)
    }
    photos: dict[str, str] = {}
    for i in sorted(referenced):
        for pid in _shot_photos(tl.shots[i].spec):
            photos[pid] = asset_sha[pid]
    payload = {
        "engine_version": engine_version,
        "video": tl.spec.video.model_dump(mode="json"),
        "plans": plans,
        "shots": shots,
        "photos": photos,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
