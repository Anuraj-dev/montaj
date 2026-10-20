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


class AssetMissing(FileNotFoundError):
    """A clip path failed lookup. `hint` completes the spec error line."""

    def __init__(self, problem: str, hint: str) -> None:
        super().__init__(problem)
        self.hint = hint


def clip_is_file(clip_id: str) -> bool:
    """True when `clip` is a video path, not an assets stem.

    `intro` is a stem. `final.mp4`, `films/launch.mp4` and `/abs/final.mp4` are files.
    """
    path = Path(clip_id)
    if path.is_absolute() or len(path.parts) != 1:
        return True
    return path.suffix.lower() in _CLIP_EXTS


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


def resolve_clip(spec_dir: Path, assets_dir: Path, clip_id: str) -> Path:
    """Assets stem, or a video path relative to the spec file (absolute paths allowed)."""
    if not clip_is_file(clip_id):
        return find_clip(assets_dir, clip_id)
    path = Path(clip_id)
    if not path.is_absolute():
        path = Path(spec_dir) / path
    if path.suffix.lower() not in _CLIP_EXTS:
        raise AssetMissing(f"{clip_id!r} is not a video", "use mp4, mov, webm or mkv")
    if not path.is_file():
        raise AssetMissing(f"{clip_id!r} not found", "path relative to the spec file, or absolute")
    return path.resolve()


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scene_shots(scene: object) -> set[int]:
    from montaj.timeline import Blank, Card, Clip, Sequence, Still, Trans, Wall

    if isinstance(scene, Blank):
        return set()
    if isinstance(scene, (Still, Wall, Clip, Sequence, Card)):
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
    if shot.sequence is not None:
        ids.extend(item.photo for item in shot.sequence.items if item.photo)
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


# Plan fields added after the first release. A None value is left out of the hash input so
# films that never use the field keep their cached segments. Append here when adding one.
_HASH_OMIT_NONE = ("zoom",)


def _hash_form(v):
    if isinstance(v, dict):
        return {k: _hash_form(x) for k, x in v.items() if not (k in _HASH_OMIT_NONE and x is None)}
    if isinstance(v, (list, tuple)):
        return [_hash_form(x) for x in v]
    return v


def _shot_payload(shot) -> dict:
    """Shot JSON for the segment hash.

    An unset `range` is omitted so a film that does not reuse a clip keeps the
    hash it had before the field existed. A set range is part of the payload.
    """
    raw = shot.model_dump(mode="json", by_alias=True)
    if raw.get("range") is None:
        raw.pop("range", None)
    return raw


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
    overlay_ids: set[int] = set()
    for f in range(seg.start, seg.end):
        plan = tl.plan(f)
        plan_d = asdict(plan)
        # Films without overlays keep their old hashes: the empty field is not hashed.
        if not plan_d["overlays"]:
            del plan_d["overlays"]
        plans.append(_hash_form(plan_d))
        overlay_ids.update(plan.overlays)
        referenced |= _scene_shots(plan.scene)
        text_ids.update(plan.texts)
        sub_ids.update(plan.subs)
    shots = {str(i): _shot_payload(tl.shots[i].spec) for i in sorted(referenced)}
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
    look = tl.spec.video.look
    if not isinstance(look, str):  # the .cube bytes, not just its path, decide the pixels
        payload["lut"] = asset_sha[f"lut:{look.lut}"]
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
    if overlay_ids:
        # Resolved values (frames, defaults), not the spec text: a marker edit that moves
        # the window, or a default changing, must reach the hash.
        payload["overlays"] = {str(i): asdict(tl.overlays[i]) for i in sorted(overlay_ids)}
    captions = any(
        tl.shots[i].spec.frame is not None and tl.shots[i].spec.frame.caption
        for i in referenced
    )
    cards = any(tl.shots[i].spec.card is not None for i in referenced)
    if text_ids or sub_ids or captions or cards:
        payload["text_assets"] = _text_asset_shas()
    if tl.markers:
        payload["markers"] = {k: tl.markers[k] for k in sorted(tl.markers)}
    if clips:
        payload["clips"] = clips
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
