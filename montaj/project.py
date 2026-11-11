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
    from montaj.timeline import Blank, Card, Clip, Custom, Layers, Sequence, Still, Trans, Wall

    if isinstance(scene, Blank):
        return set()
    if isinstance(scene, (Still, Wall, Clip, Sequence, Card, Custom, Layers)):
        return {scene.shot}
    if isinstance(scene, Trans):
        return _scene_shots(scene.a) | _scene_shots(scene.b)
    raise TypeError(f"unknown scene {type(scene)!r}")


def scene_custom_files(scene: object, shots) -> set[str]:
    """Custom files a scene runs: a custom shot, its shots' effects, a custom transition.

    `shots` maps a shot index to its Shot. Walks the scene, not the referenced shots, so a
    transition's file reaches only the segments that draw that transition.
    """
    from montaj.timeline import Blank, Custom, Trans

    if isinstance(scene, Blank):
        return set()
    if isinstance(scene, Trans):
        out = scene_custom_files(scene.a, shots) | scene_custom_files(scene.b, shots)
        if scene.src is not None:
            out.add(scene.src)
        return out
    sh = shots[scene.shot]
    out = {fx.custom for fx in sh.effects or () if fx.custom is not None}
    if isinstance(scene, Custom) and sh.custom is not None:
        out.add(sh.custom)
    return out


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
    ids.extend(layer.image for layer in shot.layers or () if layer.image is not None)
    return ids


def shot_clips(shot) -> list[str]:
    """Clip stems (or paths) a shot reads: its own clip, then its layers' clips."""
    ids = [shot.clip] if shot.clip is not None else []
    ids.extend(layer.clip for layer in shot.layers or () if layer.clip is not None)
    return ids


def shot_mattes(shot) -> list[tuple[str, str]]:
    """(kind, stem) of every subject matte a shot reads (#36): a `matte:` clip shot, then `mask:` layers."""
    out: list[tuple[str, str]] = []
    if getattr(shot, "matte", None) is not None and shot.clip is not None:
        out.append(("clip", shot.clip))
    for layer in shot.layers or ():
        if getattr(layer, "mask", None) is not None:
            out.append(("clip", layer.clip) if layer.clip is not None else ("photo", layer.image))
    return out


def matte_source(spec_dir: Path, assets_dir: Path, kind: str, stem: str) -> Path:
    """The asset file a matte is computed from."""
    return resolve_clip(spec_dir, assets_dir, stem) if kind == "clip" else find_photo(assets_dir, stem)


def _lookup_sha(asset_sha: dict[str, str], stem: str, kind: str) -> str:
    """`photo:` and `clip:` keys stay distinct when a still and a clip share a stem.

    Callers that still pass a bare stem (one asset kind) keep resolving.
    """
    namespaced = f"{kind}:{stem}"
    if namespaced in asset_sha:
        return asset_sha[namespaced]
    return asset_sha[stem]


# Faces and licence files added after the first release, by the style that needs them. They are
# left out of the base digest so films that use none of them keep their cached segments.
# Display faces for `font:` map to "": only the gpu engine draws them, and their bytes reach the hash
# through the line's glyph run (FontRef.sha), never through the Chromium digest.
_EXT_FONT_STYLE = {
    "inter-latin-600": "sans", "inter-latin-700": "sans", "jetbrains-mono-": "mono",
    **{p: "" for p in ("montserrat-", "anton-", "bebas-neue-", "poppins-", "bangers-", "archivo-black-", "mukta-")},
}


def _ext_font_style(name: str) -> str | None:
    """Style owning an added font file; "" for licence files (never rendered); None for a base file."""
    if name.startswith("OFL-"):
        return ""
    for prefix, style in _EXT_FONT_STYLE.items():
        if name.startswith(prefix):
            return style
    return None


def _text_asset_shas(ext_styles: frozenset[str] | None = None) -> dict[str, str]:
    """sha of text.css and the base fonts, plus text-ext.css and the faces of `ext_styles` when given.

    `ext_styles` is None for a segment that uses nothing added after the first release.
    """
    pkg = Path(__file__).parent
    out: dict[str, str] = {}
    css = pkg / "render" / "text.css"
    if css.is_file():
        out["text.css"] = file_sha(css)
    fonts = pkg / "fonts"
    if fonts.is_dir():
        for p in sorted(fonts.iterdir()):
            if not p.is_file():
                continue
            style = _ext_font_style(p.name)
            if style is None or (ext_styles is not None and style in ext_styles):
                out[p.name] = file_sha(p)
    if ext_styles is not None:
        out["text-ext.css"] = file_sha(pkg / "render" / "text-ext.css")
    return out


def _uses_ext_text(lines) -> frozenset[str] | None:
    """Styles of text lines (a block or a card) that use anything added after the first release, else None."""
    used = False
    styles: set[str] = set()
    for ln in lines:
        styles.add(ln.style)
        used |= (ln.style in ("sans", "mono") or ln.reveal == "type" or ln.cursor
                 or ln.align != "center" or ln.box != "none")
    return frozenset(styles) if used else None


# Plan fields added after the first release. A None value is left out of the hash input so
# films that never use the field keep their cached segments. Append here when adding one.
# Names must be unique across plan dataclasses: another None-valued field with the same name would vanish too.
_HASH_OMIT_NONE = ("zoom", "strength", "seed", "dip_color", "src", "params")
# Same for list fields that are empty unless the film uses the feature.
_HASH_OMIT_EMPTY = ("splits",)


def _hash_form(v):
    if isinstance(v, dict):
        return {k: _hash_form(x) for k, x in v.items() if not (k in _HASH_OMIT_NONE and x is None) and not (k in _HASH_OMIT_EMPTY and not x)}
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
    # Linked audio (#40) changes the mix, not the pixels: retiming a J/L cut re-renders nothing.
    for name in ("audio", "gain", "audio_lead", "audio_tail"):
        raw.pop(name, None)
    for layer in raw.get("layers") or ():
        layer.pop("audio", None)
    raw.pop("match", None)  # `match: false` (#63) only tells check a luma jump is designed
    for line in (raw.get("card") or {}).get("text") or ():
        line.pop("role", None)  # a style's `role` reaches card lines too; it is never drawn
    return raw


def _text_payload(block) -> dict:
    """Text block JSON for the segment hash. `role` (#57) only tells `check` what the text is; it draws
    nothing, so setting it re-renders nothing."""
    raw = block.model_dump(mode="json", by_alias=True)
    raw.pop("role", None)
    for line in raw.get("lines") or ():
        line.pop("role", None)
    return raw


def _overlay_payload(ov) -> dict:
    """Overlay JSON for the segment hash.

    `levels` is None unless the shape is a waveform. Dropping that key keeps every
    film that never draws one on the hash it had before the field existed. A waveform's
    levels are the audio the renderer paints, so a source change shows up here.
    """
    raw = asdict(ov)
    if raw.get("levels") is None:
        raw.pop("levels", None)
    return raw


def _run_payload(run) -> dict:
    """A GlyphRun for the hash. Fields added after #20 are dropped at their defaults so a run that
    uses none of them hashes as it did before they existed."""
    raw = asdict(run)
    if raw.get("rotate") == 0.0:
        raw.pop("rotate")
    for g in raw["glyphs"]:
        if g.get("emph") == 0:
            g.pop("emph")
        if g.get("scale") == 1.0:
            g.pop("scale")
    return raw


def segment_hash(
    tl: Timeline,
    seg: Segment,
    asset_sha: dict[str, str],
    engine_version: str,
    encode: dict | None = None,
) -> str:
    referenced: set[int] = set()
    text_ids: set[int] = set()
    sub_ids: set[int] = set()
    plans = []
    overlay_ids: set[int] = set()
    custom_files: set[str] = set()
    by_index = {sh.index: sh.spec for sh in tl.shots}
    for f in range(seg.start, seg.end):
        plan = tl.plan(f)
        plan_d = asdict(plan)
        # Films without overlays keep their old hashes: the empty field is not hashed.
        if not plan_d["overlays"]:
            del plan_d["overlays"]
        plans.append(_hash_form(plan_d))
        overlay_ids.update(plan.overlays)
        referenced |= _scene_shots(plan.scene)
        custom_files |= scene_custom_files(plan.scene, by_index)
        text_ids.update(plan.texts)
        sub_ids.update(plan.subs)
    shots = {str(i): _shot_payload(tl.shots[i].spec) for i in sorted(referenced)}
    photos: dict[str, str] = {}
    clips: dict[str, str] = {}
    mattes: dict[str, str] = {}
    for i in sorted(referenced):
        shot = tl.shots[i].spec
        for pid in _shot_photos(shot):
            photos[pid] = _lookup_sha(asset_sha, pid, "photo")
        for cid in shot_clips(shot):
            clips[cid] = _lookup_sha(asset_sha, cid, "clip")
        for kind, stem in shot_mattes(shot):
            mattes[f"{kind}:{stem}"] = asset_sha[f"matte:{kind}:{stem}"]
    payload = {
        "engine_version": engine_version,
        "video": tl.spec.video.model_dump(mode="json"),
        "plans": plans,
        "shots": shots,
        "photos": photos,
    }
    if mattes:  # the matte file's bytes decide the cutout; absent for every film without one
        payload["mattes"] = mattes
    look = tl.spec.video.look
    if not isinstance(look, str):  # the .cube bytes, not just its path, decide the pixels
        payload["lut"] = asset_sha[f"lut:{look.lut}"]
    if text_ids:
        payload["texts"] = {
            str(i): _text_payload(tl.spec.text[i])
            for i in sorted(text_ids)
        }
        # gpu-engine lines: the materialised layout (glyph ids, positions, font shas, cell params).
        # Only present when a line uses it, so every older film keeps its hash.
        runs = {
            f"{i}.{j}": _run_payload(line.glyphs)
            for i in sorted(text_ids)
            for j, line in enumerate(tl.texts[i].lines)
            if line.glyphs is not None
        }
        if runs:
            payload["glyph_runs"] = runs
        # Text motion (#22): the resolved frame values (defaults, exit clocks, seeds). Absent when unused.
        motions = {
            f"{i}.{j}": {k: v for k, v in asdict(line.motion).items() if not (k == "slam_from" and v is None)}
            for i in sorted(text_ids)
            for j, line in enumerate(tl.texts[i].lines)
            if line.motion is not None
        }
        if motions:
            payload["text_motions"] = motions
    if sub_ids:
        payload["subs"] = {
            str(i): tl.spec.subs[i].model_dump(mode="json", by_alias=True)
            for i in sorted(sub_ids)
        }
    if overlay_ids:
        # Resolved values (frames, defaults), not the spec text: a marker edit that moves
        # the window, or a default changing, must reach the hash. `levels` is None on every
        # shape except waveform; leaving that key out keeps a ring film's hash identical.
        payload["overlays"] = {str(i): _overlay_payload(tl.overlays[i]) for i in sorted(overlay_ids)}
    captions = any(
        tl.shots[i].spec.frame is not None and tl.shots[i].spec.frame.caption
        for i in referenced
    )
    ext: set[str] = set()
    used_ext = False
    line_sets = [tl.spec.text[i].lines for i in text_ids]
    line_sets += [tl.shots[i].spec.card.text for i in referenced if tl.shots[i].spec.card is not None]
    for lines in line_sets:
        got = _uses_ext_text(lines)
        if got is not None:
            used_ext = True
            ext |= got
    cards = any(tl.shots[i].spec.card is not None for i in referenced)
    if text_ids or sub_ids or captions or cards:
        payload["text_assets"] = _text_asset_shas(frozenset(ext)) if used_ext else _text_asset_shas()
    if tl.markers:
        payload["markers"] = {k: tl.markers[k] for k in sorted(tl.markers)}
    if clips:
        payload["clips"] = clips
    if custom_files:
        # The file bytes and the helper API decide the pixels; params ride in the shot/plan payloads.
        from montaj.render.custom import HELPER_VERSION

        payload["custom"] = {
            "api": HELPER_VERSION,
            "files": {rel: asset_sha[f"custom:{rel}"] for rel in sorted(custom_files)},
        }
    if encode:  # `VideoSettings.hash_key()`: absent for the default H.264, so old films keep hashes
        payload["encode"] = encode
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
