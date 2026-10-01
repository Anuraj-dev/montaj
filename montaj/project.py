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
_CLIP_EXTS = (".mp4", ".mov", ".webm", ".mkv")


def _find_stem(assets_dir: Path, stem: str, exts: tuple[str, ...]) -> Path:
    assets_dir = Path(assets_dir)
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"'{stem}' not in {assets_dir} (have: none)")
    matches = [
        p
        for p in assets_dir.iterdir()
        if p.is_file() and p.stem == stem and p.suffix.lower() in exts
    ]
    if not matches:
        stems = sorted(
            {
                p.stem
                for p in assets_dir.iterdir()
                if p.is_file() and p.suffix.lower() in exts
            }
        )
        have = ", ".join(stems) if stems else "none"
        raise FileNotFoundError(f"'{stem}' not in {assets_dir} (have: {have})")
    order = {ext: i for i, ext in enumerate(exts)}
    matches.sort(key=lambda p: (order.get(p.suffix.lower(), 99), p.name))
    return matches[0]


def find_photo(assets_dir: Path, photo_id: str) -> Path:
    return _find_stem(assets_dir, photo_id, _EXTS)


def find_clip(assets_dir: Path, clip_id: str) -> Path:
    return _find_stem(assets_dir, clip_id, _CLIP_EXTS)


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scene_shots(scene: object) -> set[int]:
    from montaj.timeline import Blank, Clip, Still, Trans, Wall

    if isinstance(scene, Blank):
        return set()
    if isinstance(scene, (Still, Wall, Clip)):
        return {scene.shot}
    if isinstance(scene, Trans):
        return _scene_shots(scene.a) | _scene_shots(scene.b)
    raise TypeError(f"unknown scene {type(scene)!r}")


def _shot_photos(shot) -> list[str]:
    ids: list[str] = []
    if shot.photo is not None:
        ids.append(shot.photo)
    if shot.wall is not None:
        ids.extend(p.photo for p in shot.wall.prints)
    if shot.morph is not None:
        ids.append(shot.morph.photo)
    return ids


def _lookup_sha(asset_sha: dict[str, str], stem: str, kind: str) -> str:
    """`photo:` and `clip:` keys stay distinct when a still and a clip share a stem.

    Callers that still pass a bare stem (one asset kind) keep resolving.
    """
    namespaced = f"{kind}:{stem}"
    if namespaced in asset_sha:
        return asset_sha[namespaced]
    return asset_sha[stem]


def _text_asset_shas() -> dict[str, str]:
    """sha of text.css (if present) and every file in montaj/fonts/."""
    pkg = Path(__file__).parent
    out: dict[str, str] = {}
    css = pkg / "render" / "text.css"
    if css.is_file():
        out["text.css"] = file_sha(css)
    fonts = pkg / "fonts"
    if fonts.is_dir():
        for p in sorted(fonts.iterdir()):
            if p.is_file():
                out[p.name] = file_sha(p)
    return out


def segment_hash(
    tl: Timeline,
    seg: Segment,
    asset_sha: dict[str, str],
    engine_version: str,
) -> str:
    referenced: set[int] = set()
    text_ids: set[int] = set()
    sub_ids: set[int] = set()
    plans = []
    for f in range(seg.start, seg.end):
        plan = tl.plan(f)
        plans.append(asdict(plan))
        referenced |= _scene_shots(plan.scene)
        text_ids.update(plan.texts)
        sub_ids.update(plan.subs)
    shots = {
        str(i): tl.shots[i].spec.model_dump(mode="json", by_alias=True)
        for i in sorted(referenced)
    }
    photos: dict[str, str] = {}
    clips: dict[str, str] = {}
    for i in sorted(referenced):
        shot = tl.shots[i].spec
        for pid in _shot_photos(shot):
            photos[pid] = _lookup_sha(asset_sha, pid, "photo")
        if shot.clip is not None:
            clips[shot.clip] = _lookup_sha(asset_sha, shot.clip, "clip")
    payload = {
        "engine_version": engine_version,
        "video": tl.spec.video.model_dump(mode="json"),
        "plans": plans,
        "shots": shots,
        "photos": photos,
    }
    if text_ids:
        payload["texts"] = {
            str(i): tl.spec.text[i].model_dump(mode="json", by_alias=True)
            for i in sorted(text_ids)
        }
    if sub_ids:
        payload["subs"] = {
            str(i): tl.spec.subs[i].model_dump(mode="json", by_alias=True)
            for i in sorted(sub_ids)
        }
    captions = any(
        tl.shots[i].spec.frame is not None and tl.shots[i].spec.frame.caption
        for i in referenced
    )
    if text_ids or sub_ids or captions:
        payload["text_assets"] = _text_asset_shas()
    if tl.markers:
        payload["markers"] = {k: tl.markers[k] for k in sorted(tl.markers)}
    if clips:
        payload["clips"] = clips
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
