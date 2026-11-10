"""`check` pacing numbers (#33): timeline-derived values against a hand count of the recipe's cut list."""
from __future__ import annotations

import json
import re
from pathlib import Path
from shutil import copyfile

import pytest
import yaml
from PIL import Image

from montaj import pacing
from montaj.cli import main
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

ROOT = Path(__file__).resolve().parents[1]
STEMS = ["09", "10", "12", "16", "19", "20", "21", "22", "28", "29", "32", "33", "34", "35", "37", "38"]


def _project(tmp: Path, extra: str = "") -> Path:
    for stem in STEMS:
        p = tmp / "assets" / f"{stem}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(p, "JPEG")
    copyfile(ROOT / "recipes" / "birthday-short.yaml", tmp / "montaj.yaml")
    if extra:
        with (tmp / "montaj.yaml").open("a") as fh:
            fh.write(extra)
    return tmp


def _hand_counts() -> dict:
    """Recompute everything from the recipe's `hold:` strings alone (120 bpm, 30 fps)."""
    raw = yaml.safe_load((ROOT / "recipes" / "birthday-short.yaml").read_text())
    spf = 1 / raw["video"]["fps"]
    spb = 60 / raw["video"]["bpm"]
    secs = []
    for sh in raw["shots"]:
        n, unit = re.fullmatch(r"([\d.]+)([bf])", sh["hold"]).groups()
        secs.append(float(n) * (spb if unit == "b" else spf))
    starts = [sum(secs[:i]) for i in range(len(secs))]
    off = [i for i in range(1, len(secs)) if abs(starts[i] / spb - round(starts[i] / spb)) * spb > spf / 2 + 1e-9]
    kinds = [(sh.get("in") or {"type": "cut"})["type"] for sh in raw["shots"]][1:]
    run = best = 1
    at = best_at = 1
    for j in range(1, len(kinds)):
        run = run + 1 if kinds[j] == kinds[j - 1] else 1
        at = at if run > 1 else j + 1
        if run > best:
            best, best_at = run, at
    return {"secs": secs, "total": sum(secs), "off": off, "kinds": kinds, "run": (best, best_at)}


def test_birthday_short_numbers_match_the_cut_list(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    tl = resolve(load_spec(proj / "montaj.yaml"), proj)
    m = pacing.metrics(tl, proj)
    h = _hand_counts()
    secs = sorted(h["secs"])
    p = m["pacing"]
    assert p["cuts"] == len(h["secs"]) - 1 == 12
    assert p["per10s"] == pytest.approx(12 * 10 / h["total"], abs=0.01)
    assert p["median_shot"] == pytest.approx(secs[len(secs) // 2], abs=0.001)
    assert p["longest"] == pytest.approx(13.0, abs=0.001)
    assert p["longest_shot"] == h["secs"].index(max(h["secs"]))
    assert m["hook"] == {"first_text": None, "first_cut": pytest.approx(h["secs"][0], abs=0.001)}
    assert m["beats"]["cuts"] == 12
    assert [o["shot"] for o in m["beats"]["off"]] == h["off"] and h["off"] == [11]
    v = m["variety"]
    assert v["transition_kinds"] == len(set(h["kinds"])) == 4
    assert (v["repeated_in_a_row"], v["repeated_from"]) == h["run"]
    assert "captions" not in m and "audio" not in m


def test_lines_format_and_no_warn_without_a_target(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    tl = resolve(load_spec(proj / "montaj.yaml"), proj)
    out = pacing.lines(pacing.metrics(tl, proj), None, fps=tl.fps, bpm=120)
    assert [ln.split()[1] for ln in out] == ["pacing", "hook", "beats", "variety"]
    assert all(ln.startswith("INFO") for ln in out)
    assert out[0].startswith("INFO pacing cuts=12 per10s=")


def test_targets_warn_only_when_missed(tmp_path: Path) -> None:
    proj = _project(tmp_path, "check:\n  pacing: {max_shot: 4s, min_per10s: 9, hook_text_by: 1s}\n")
    tl = resolve(load_spec(proj / "montaj.yaml"), proj)
    out = pacing.lines(pacing.metrics(tl, proj), tl.spec.check.pacing, fps=tl.fps, bpm=120)
    warns = [ln for ln in out if ln.startswith("WARN")]
    assert [w.split()[1] for w in warns] == ["pacing", "pacing", "hook"]
    ok = _project(tmp_path / "ok", "check:\n  pacing: {max_shot: 13s, min_per10s: 1}\n")
    tl = resolve(load_spec(ok / "montaj.yaml"), ok)
    assert not [ln for ln in pacing.lines(pacing.metrics(tl, ok), tl.spec.check.pacing, fps=30, bpm=120)
                if ln.startswith("WARN")]


def test_bad_target_is_a_spec_error(tmp_path: Path) -> None:
    proj = _project(tmp_path, "check:\n  pacing: {max_shot: 4}\n")
    with pytest.raises(SpecError):
        resolve(load_spec(proj / "montaj.yaml"), proj)


def test_captions_coverage_from_text_and_word_spans(tmp_path: Path) -> None:
    proj = tmp_path
    for stem in ("a",):
        (proj / "assets").mkdir(exist_ok=True)
        Image.new("RGB", (8, 8)).save(proj / "assets" / f"{stem}.jpg", "JPEG")
    words = [{"i": i, "w": "x", "s": s, "e": s + 0.3} for i, s in enumerate([0.0, 0.4, 0.8, 1.2, 4.0, 4.4])]
    (proj / "words.json").write_text(json.dumps({"words": words}))
    (proj / "t.wav").write_bytes(b"RIFF")
    (proj / "montaj.yaml").write_text(
        "video: {size: 1080x1920, fps: 30, look: warm-film}\n"
        "audio: {track: t.wav, markers: words.json}\n"
        "shots: [{photo: a, hold: 6s}]\n"
        "text:\n  - from: 0s\n    to: 1.6s\n    lines: [{text: hi, at: 0s, y: 800}]\n"
    )
    tl = resolve(load_spec(proj / "montaj.yaml"), proj)
    c = pacing.metrics(tl, proj)["captions"]
    # speech stretches: 0.0-1.5 (joined, gaps 0.1) and 4.0-4.7; text covers 0-1.6 only.
    assert c["coverage"] == pytest.approx(1.5 / 2.2, abs=0.001)
    assert c["max_gap"] == pytest.approx(0.7, abs=0.001)


def test_json_carries_the_numbers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    proj = _project(tmp_path)
    (proj / "out").mkdir()
    import subprocess

    from montaj.encode import FFMPEG
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=30:d=2",
                    "-pix_fmt", "yuv420p", str(proj / "out" / "x.mp4")], check=True)
    code = main(["-C", str(proj), "--json", "check", "--video", "out/x.mp4"])
    payload = json.loads(capsys.readouterr().out)
    assert code in (0, 1)
    assert payload["pacing"]["pacing"]["cuts"] == 12
    assert payload["pacing"]["hook"]["first_motion"] is None  # a flat clip never moves
    assert any(ln.startswith("INFO pacing") for ln in payload["lines"])
