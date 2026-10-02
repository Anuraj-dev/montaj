"""Audio: the machine config, the external-worker seam, beat tracking, markers.json, the GPU lock.

No ACE-Step and no whisper run here. Both are stubbed the way montaj actually calls them — a
`config.toml` (via `MONTAJ_CONFIG`) naming a fake interpreter, which runs a stub worker instead —
so the parent side is tested without the GPU. The beat tracker is the one piece of real maths here:
it is pure numpy inside the worker and runs in this process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from montaj import config as config_mod
from montaj.audio import _whisper_worker as worker
from montaj.audio.music import gen, parse_duration
from montaj.cli import main
from montaj.gpulock import hold, lock_path

SR = 16000

# --- stubs ----------------------------------------------------------------------------------

WHISPER_STUB = '''\
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--wav", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lang", default="hi")
ap.add_argument("--prompt")
ap.add_argument("--bpm", type=float)
ns = ap.parse_args()

MARKERS = {
    "duration": 165.0,
    "bpm": ns.bpm or 76.0,
    "beats": [0.5, 1.289, 2.079],
    "words": [{"i": 1, "w": "Arshiya", "s": 5.2, "e": 6.28},
              {"i": 2, "w": "Jaipur", "s": 12.0, "e": 13.08}],
    "segments": [{"s": 5.2, "e": 6.28, "text": "Arshiya", "first_word": 1, "last_word": 1},
                 {"s": 12.0, "e": 13.74, "text": "Jaipur ki dhup mein", "first_word": 2, "last_word": 2}],
    "rms": [0.01, 0.5, 0.2],
}
with open(ns.out, "w") as fh:
    json.dump(MARKERS, fh)
print("chatter montaj must not read", file=sys.stderr)
for seg in MARKERS["segments"]:
    print(f"{seg['s']:.2f}-{seg['e']:.2f} w{seg['first_word']}-{seg['last_word']} {seg['text']}")
if os.environ.get("STUB_FAIL"):
    print("model exploded", file=sys.stderr)
    sys.exit(3)
'''

ACE_STUB = '''\
import argparse, json, os, struct, sys, wave
from pathlib import Path

RESULT = os.dup(1)  # the real worker points fd 1 at stderr before loading ACE-Step
os.dup2(2, 1)

ap = argparse.ArgumentParser()
ap.add_argument("--ace-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lyrics", required=True)
ap.add_argument("--caption", required=True)
ap.add_argument("--bpm", type=float, required=True)
ap.add_argument("--key")
ap.add_argument("--lang")
ap.add_argument("--duration", type=float, required=True)
ap.add_argument("--n", type=int, required=True)
ns = ap.parse_args()

out = Path(ns.out)
out.mkdir(parents=True, exist_ok=True)
scratch = out / ".ace-scratch"
scratch.mkdir(parents=True, exist_ok=True)
print("loading acestep", flush=True)  # library chatter: must not reach montaj's protocol
lyrics_path = Path(ns.lyrics)
if not lyrics_path.is_file():
    print("lyrics missing: " + ns.lyrics, file=sys.stderr)
    sys.exit(3)
if os.environ.get("STUB_FAIL"):
    print("cuda out of memory", file=sys.stderr)
    sys.exit(3)
fail_after = os.environ.get("STUB_FAIL_AFTER")
fail_after = int(fail_after) if fail_after is not None else ns.n
first_error = None
for i in range(ns.n):
    if i >= fail_after:
        print("FAILED cuda hiccup", flush=True)
        if first_error is None:
            first_error = "cuda hiccup"
        continue
    seed = f"{i:04d}beef"
    raw = scratch / f"{seed}.wav"
    with wave.open(str(raw), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(struct.pack("<" + "h" * 1600, *([800] * 1600)))
    wav = out / f"cand-{seed}.wav"
    os.replace(raw, wav)
    (out / f"cand-{seed}.json").write_text(json.dumps(
        {"seed": seed, "caption": ns.caption, "lyrics": ns.lyrics, "bpm": ns.bpm,
         "key": ns.key, "lang": ns.lang, "duration": ns.duration,
         "params": {"seed": seed, "bpm": ns.bpm}}))
    os.write(RESULT, ('CAND ' + json.dumps({"path": str(wav), "seed": seed}) + chr(10)).encode())
if first_error is not None:
    os.write(RESULT, ("FAIL " + first_error + chr(10)).encode())
    sys.exit(1)
'''


def stub_python(tmp_path: Path, worker_src: str, *, name: str = "stub-python") -> Path:
    """A fake `ace_python`/`whisper_python`: it drops montaj's worker path and runs `worker_src`."""
    stub = tmp_path / f"{name}-worker.py"
    stub.write_text(worker_src)
    exe = tmp_path / name
    exe.write_text(f'#!/bin/sh\nshift\nexec "{sys.executable}" "{stub}" "$@"\n')
    exe.chmod(0o755)
    return exe


def config_file(tmp_path: Path, *, whisper_src: str = WHISPER_STUB, missing: str = "",
                extra: str = "", python: Path | None = None) -> Path:
    """A config.toml naming stub interpreters. `missing` drops a key, `extra` adds an unknown one."""
    (tmp_path / "ace").mkdir(exist_ok=True)
    (tmp_path / "site-packages").mkdir(exist_ok=True)
    exe = python or stub_python(tmp_path, whisper_src)
    keys = {
        "ace_step_dir": tmp_path / "ace",
        "ace_python": exe,
        "ace_site_packages": tmp_path / "site-packages",
        "whisper_python": exe,
    }
    keys = {k: v for k, v in keys.items() if k != missing}
    body = "".join(f'{k} = "{v}"\n' for k, v in keys.items())
    path = tmp_path / "config.toml"
    path.write_text(f"{body}{extra}")
    return path


def project(tmp_path: Path) -> Path:
    """A project dir with a fake song and lyrics; `MONTAJ_CONFIG` is the caller's business."""
    proj = tmp_path / "proj"
    (proj / "build").mkdir(parents=True, exist_ok=True)
    (proj / "lyrics.txt").write_text("tum saare umar ke ghar ho\n")
    (proj / "song.wav").write_bytes(b"RIFF" + b"\0" * 64)
    return proj


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, list[str], str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


# --- config ---------------------------------------------------------------------------------

def test_config_reads_every_key_and_builds_the_worker_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "ace").mkdir()
    (tmp_path / "site").mkdir()
    path = tmp_path / "config.toml"
    path.write_text(
        f'ace_step_dir = "{tmp_path / "ace"}"\nace_python = "/py/ace"\n'
        f'ace_site_packages = "{tmp_path / "site"}"\nwhisper_python = "/py/whisper"\n'
    )
    monkeypatch.setenv("MONTAJ_CONFIG", str(path))
    cfg = config_mod.load()
    assert (cfg.ace_step_dir, cfg.ace_python, cfg.whisper_python) == (
        str(tmp_path / "ace"), "/py/ace", "/py/whisper",
    )
    assert cfg.ace_argv("/w.py", "--out", 3) == ["/py/ace", "/w.py", "--out", "3"]
    assert cfg.whisper_argv("/w.py", "--bpm") == ["/py/whisper", "/w.py", "--bpm"]
    monkeypatch.setenv("PYTHONPATH", "/keep/me")
    assert cfg.ace_env()["PYTHONPATH"] == f"{tmp_path / 'site'}{os.pathsep}/keep/me"


def test_config_missing_key_names_it_and_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = config_file(tmp_path, missing="whisper_python")
    monkeypatch.setenv("MONTAJ_CONFIG", str(path))
    with pytest.raises(config_mod.ConfigError) as exc:
        config_mod.load()
    assert str(exc.value) == f"config: whisper_python missing (set it in {path})"

    monkeypatch.setenv("MONTAJ_CONFIG", str(tmp_path / "absent.toml"))
    with pytest.raises(config_mod.ConfigError) as exc:
        config_mod.load()
    assert "absent.toml missing" in str(exc.value)

    monkeypatch.setenv("MONTAJ_CONFIG", str(path))
    path.write_text(path.read_text() + 'ace_pythonn = "typo"\n')
    with pytest.raises(config_mod.ConfigError) as exc:
        config_mod.load()
    assert "unknown key(s) ace_pythonn" in str(exc.value)


def test_config_default_path_is_the_documented_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    assert config_mod.config_path() == Path.home() / ".config" / "montaj" / "config.toml"


def test_parse_duration_reads_the_spec_spellings() -> None:
    assert parse_duration("165s") == 165.0
    assert parse_duration("165") == 165.0
    assert parse_duration("2m45s") == 165.0
    assert parse_duration(90) == 90.0
    for bad in ("", "later", "3x", "-4s", "0s"):
        with pytest.raises(ValueError):
            parse_duration(bad)


# --- music gen + analyze through the CLI ------------------------------------------------------

def test_music_analyze_lines_json_and_markers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config_file(tmp_path)
    monkeypatch.setenv("MONTAJ_CONFIG", str(cfg))
    proj = project(tmp_path)
    monkeypatch.setenv("MONTAJ_GPU_LOCK", str(tmp_path / "gpu.lock"))

    code, lines, err = _run(["-C", str(proj), "music", "analyze", "song.wav", "--lang", "hi"], capsys)
    assert code == 0 and err == "", lines
    assert lines == [
        "5.20-6.28 w1-1 Arshiya",
        "12.00-13.74 w2-2 Jaipur ki dhup mein",
        "OK music/markers.json 165.0s bpm=76.0 beat0=0.50s 2 words first_word=5.20s lang=hi",
    ]
    markers = json.loads((proj / "music" / "markers.json").read_text())
    assert set(markers) == {"duration", "bpm", "beats", "words", "segments", "rms"}
    assert [w["i"] for w in markers["words"]] == [1, 2] and markers["words"][0]["w"] == "Arshiya"
    assert markers["segments"][1]["first_word"] == 2 and markers["segments"][1]["last_word"] == 2
    # the stub's chatter went to the log, not to stdout
    assert "chatter montaj must not read" in (proj / "build" / "music.log").read_text()

    code, lines, err = _run(["-C", str(proj), "--json", "music", "analyze", "song.wav", "--bpm", "76"], capsys)
    assert code == 0 and err == ""
    payload = json.loads("\n".join(lines))
    assert payload["status"] == "OK"
    assert payload["lines"][-1] == "OK music/markers.json 165.0s bpm=76.0 beat0=0.50s 2 words first_word=5.20s lang=hi"
    assert json.loads((proj / "music" / "markers.json").read_text())["bpm"] == 76.0

    code, lines, err = _run(["-C", str(proj), "music", "analyze", "song.wav", "--out", "build/m.json"], capsys)
    assert code == 0 and lines[-1] == "OK build/m.json 165.0s bpm=76.0 beat0=0.50s 2 words first_word=5.20s lang=hi", lines
    assert (proj / "build" / "m.json").is_file()


def test_music_analyze_missing_wav_and_config_are_err_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONTAJ_GPU_LOCK", str(tmp_path / "gpu.lock"))
    cfg = config_file(tmp_path)
    proj = project(tmp_path)
    monkeypatch.setenv("MONTAJ_CONFIG", str(tmp_path / "absent.toml"))
    code, lines, err = _run(["-C", str(proj), "music", "analyze", "song.wav"], capsys)
    assert code == 1 and err == "" and len(lines) == 1
    assert lines[0].startswith("ERR music: config: ") and "absent.toml missing" in lines[0]

    monkeypatch.setenv("MONTAJ_CONFIG", str(cfg))
    code, lines, err = _run(["-C", str(proj), "music", "analyze", "nope.wav"], capsys)
    assert code == 1 and len(lines) == 1, lines
    assert lines[0].startswith("ERR music: ") and "nope.wav" in lines[0] and "does not exist" in lines[0]

    code, lines, err = _run(["-C", str(proj), "music", "analyze", "song.wav", "--bad"], capsys)
    assert code == 1 and lines[0].startswith("ERR music: ") and "--bad" in lines[0], lines
    code, lines, err = _run(["-C", str(proj), "music"], capsys)
    assert code == 1 and lines[0].startswith("ERR music: ") and "gen|analyze" in lines[0], lines


def test_music_gen_candidates_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config_file(tmp_path, whisper_src=ACE_STUB)
    monkeypatch.setenv("MONTAJ_CONFIG", str(cfg))
    proj = project(tmp_path)
    monkeypatch.setenv("MONTAJ_GPU_LOCK", str(tmp_path / "gpu.lock"))

    code, lines, err = _run(
        ["-C", str(proj), "music", "gen", "--caption", "romantic ballad", "--lyrics", "lyrics.txt",
         "--n", "2", "--duration", "30s"], capsys,
    )
    assert code == 0 and err == "", lines
    assert lines[-1] == "OK music gen 2 candidates 30s lang=hi bpm=76 key=D major"
    assert lines[:-1] == ["music/cand-0000beef.wav seed=0000beef 0.1s", "music/cand-0001beef.wav seed=0001beef 0.1s"]
    params = json.loads((proj / "music" / "cand-0001beef.json").read_text())
    assert params["caption"] == "romantic ballad" and params["bpm"] == 76.0 and params["duration"] == 30.0
    assert (proj / "music" / "cand-0001beef.wav").is_file()
    log = (proj / "build" / "music.log").read_text()
    assert "loading acestep" in log  # the worker's own prints are logged, never printed

    monkeypatch.setenv("STUB_FAIL", "1")
    code, lines, err = _run(
        ["-C", str(proj), "music", "gen", "--caption", "x", "--lyrics", "lyrics.txt", "--n", "1"], capsys,
    )
    assert code == 1 and err == "" and len(lines) == 1
    assert lines[0].startswith("ERR music: ace-step failed (3)") and "cuda out of memory" in lines[0]


def test_music_gen_partial_failure_keeps_files_and_errs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--n 3` with two successes must keep those wavs, exit 1, and name the first reason."""
    cfg = config_file(tmp_path, whisper_src=ACE_STUB)
    monkeypatch.setenv("MONTAJ_CONFIG", str(cfg))
    proj = project(tmp_path)
    monkeypatch.setenv("MONTAJ_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("STUB_FAIL_AFTER", "2")

    code, lines, err = _run(
        ["-C", str(proj), "music", "gen", "--caption", "x", "--lyrics", "lyrics.txt", "--n", "3"], capsys,
    )
    assert code == 1 and err == "", lines
    assert lines[-1] == "ERR music gen: 2/3 candidates (cuda hiccup)"
    assert lines[:-1] == [
        "music/cand-0000beef.wav seed=0000beef 0.1s",
        "music/cand-0001beef.wav seed=0001beef 0.1s",
    ]
    assert (proj / "music" / "cand-0000beef.wav").is_file()
    assert (proj / "music" / "cand-0001beef.wav").is_file()
    assert not (proj / "music" / "cand-0002beef.wav").is_file()


def test_music_gen_resolves_relative_lyrics_and_out_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ACE worker chdirs into the checkout; relative lyrics/out must still hit the caller's cwd."""
    cfg = config_file(tmp_path, whisper_src=ACE_STUB)
    monkeypatch.setenv("MONTAJ_CONFIG", str(cfg))
    monkeypatch.setenv("MONTAJ_GPU_LOCK", str(tmp_path / "gpu.lock"))
    proj = project(tmp_path)
    monkeypatch.chdir(proj)

    made = gen("music", caption="x", lyrics="lyrics.txt", n=1, log="build/music.log")
    assert len(made) == 1
    wav = proj / "music" / "cand-0000beef.wav"
    assert wav.is_file() and made.files[0] == wav
    params = json.loads((proj / "music" / "cand-0000beef.json").read_text())
    assert Path(params["lyrics"]) == proj / "lyrics.txt"


def test_music_gen_rejects_a_candidate_the_worker_never_wrote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exit 0 is not proof: montaj checks the wav is really there before reporting a candidate."""
    lying = ACE_STUB.replace('{"path": str(wav), "seed": seed}', '{"path": str(out / "gone.wav"), "seed": seed}')
    cfg = config_file(tmp_path, whisper_src=lying)
    monkeypatch.setenv("MONTAJ_CONFIG", str(cfg))
    proj = project(tmp_path)
    monkeypatch.setenv("MONTAJ_GPU_LOCK", str(tmp_path / "gpu.lock"))

    code, lines, err = _run(
        ["-C", str(proj), "music", "gen", "--caption", "x", "--lyrics", "lyrics.txt", "--n", "1"], capsys,
    )
    assert code == 1 and err == "" and len(lines) == 1, lines
    assert lines[0].startswith("ERR music: ace-step: candidate") and "gone.wav" in lines[0]


# --- beat tracking ---------------------------------------------------------------------------

def click_track(bpm: float, seconds: float = 24.0, accents: tuple[float, ...] = (1.0, 0.95, 0.6, 0.6)):
    """Clicks on a beat grid, accented per bar — the accent is what tells a beat from its half."""
    x = np.zeros(int(seconds * SR), dtype=np.float32)
    period = 60.0 / bpm
    click = np.zeros(96, dtype=np.float32)
    click[:32] = np.hanning(32)
    times: list[float] = []
    t = 0.05
    while t + period * 0.5 <= seconds:
        start = int(t * SR)
        x[start : start + 96] += accents[len(times) % len(accents)] * click
        times.append(t)
        t += period
    return x, np.array(times)


@pytest.mark.parametrize("bpm", (76.0, 128.0))
def test_beat_tracker_finds_tempo_and_phase(bpm: float) -> None:
    x, true = click_track(bpm)
    env, rate = worker.onset_envelope(x)
    got, beats = worker.beat_track(env, rate, duration=len(x) / SR)

    assert abs(got - bpm) <= 1.0, (got, bpm)
    period = 60.0 / bpm
    inside = [b for b in beats if true[0] - period / 2 <= b <= true[-1] + period / 2]
    assert len(inside) == len(true), (len(inside), len(true))
    worst = max(min(abs(b - t) for t in true) for b in inside)
    assert worst <= 0.02, f"worst beat error {worst * 1000:.1f} ms"


@pytest.mark.parametrize("bpm", (76.0, 128.0))
def test_beat_tracker_accepts_a_given_tempo(bpm: float) -> None:
    """`--bpm` skips the search but not the phase: the grid must still land on the beats."""
    x, true = click_track(bpm)
    env, rate = worker.onset_envelope(x)
    got, beats = worker.beat_track(env, rate, bpm, duration=len(x) / SR)

    assert got == bpm
    period = 60.0 / bpm
    inside = [b for b in beats if true[0] - period / 2 <= b <= true[-1] + period / 2]
    worst = max(min(abs(b - t) for t in true) for b in inside)
    assert worst <= 0.02, f"worst beat error {worst * 1000:.1f} ms"


def test_rms_envelope_hops_half_a_second() -> None:
    x = np.ones(SR, dtype=np.float32) * 0.5
    env = worker.rms_envelope(x)
    assert len(env) == 1 and env[0] == pytest.approx(0.5, abs=1e-6)


def test_tempo_refinement_stays_inside_bpm_range() -> None:
    env = np.ones(4000, dtype=np.float64)
    rate, duration = 200.0, 20.0
    for bpm in (worker.BPM_MIN, worker.BPM_MAX, 50.0, 200.0):
        got = worker._refine_tempo(env, rate, bpm, duration)
        assert worker.BPM_MIN <= got <= worker.BPM_MAX, (bpm, got)


# --- markers.json schema ---------------------------------------------------------------------

def test_build_markers_numbers_words_and_reports_word_indices() -> None:
    markers = worker.build_markers(
        [
            {"start": 5.201, "end": 6.284, "text": "  Arshiya  ", "words": [{"w": " Arshiya", "s": 5.201, "e": 6.284}]},
            {"start": 12.0, "end": 13.74, "text": "Jaipur ki dhup mein, tera woh sharmaana",
             "words": [{"w": " Jaipur", "s": 12.0, "e": 13.08}, {"w": " ", "s": 13.08, "e": 13.08},
                       {"w": " ki", "s": 13.08, "e": 13.46}]},
            {"start": 30.0, "end": 30.5, "text": "", "words": []},
        ],
        bpm=76.0,
        beats=[0.5, 1.289],
        rms=[0.1, 0.2],
        duration=165.0,
    )
    assert markers["duration"] == 165.0 and markers["bpm"] == 76.0
    assert [w["i"] for w in markers["words"]] == [1, 2, 3]  # 1-based and global: spec markers are `word:1`
    assert markers["words"][0] == {"i": 1, "w": "Arshiya", "s": 5.2, "e": 6.28}
    assert markers["words"][1] == {"i": 2, "w": "Jaipur", "s": 12.0, "e": 13.08}
    assert [s["first_word"] for s in markers["segments"]] == [1, 2, None]
    assert [s["last_word"] for s in markers["segments"]] == [1, 3, None]
    assert markers["segments"][1]["text"] == "Jaipur ki dhup mein, tera woh sharmaana"
    assert json.loads(json.dumps(markers)) == markers  # the file must survive a round trip


def test_segment_line_format_and_ellipsis() -> None:
    assert worker.segment_line({"s": 12.0, "e": 23.74, "first_word": 1, "last_word": 15,
                                "text": "Jaipur ki dhup mein, tera woh sharmaana"}) == (
        "12.00-23.74 w1-15 Jaipur ki dhup mein, tera woh sharmaana"
    )
    long_line = worker.segment_line({"s": 1.0, "e": 2.0, "first_word": 1, "last_word": 9, "text": "x" * 90})
    assert long_line.endswith("…") and len(long_line.split("w1-9 ")[1]) == 61
    assert worker.segment_line({"s": 1.0, "e": 2.0, "first_word": None, "last_word": None, "text": ""}) == (
        "1.00-2.00 w0-0"
    )


# --- the GPU lock -----------------------------------------------------------------------------

HOLD_SNIPPET = """
import sys, time
from pathlib import Path
sys.path.insert(0, {repo!r})
from montaj.gpulock import hold
ready, release = Path(sys.argv[1]), Path(sys.argv[2])
with hold({log!r}):
    ready.write_text("held")
    while not release.exists():
        time.sleep(0.05)
"""


def _wait_path(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def _wait_text(path: Path, needle: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    body = ""
    while time.monotonic() < deadline:
        if path.is_file():
            body = path.read_text()
            if needle in body:
                return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {needle!r} in {path}: {body!r}")


def test_gpu_lock_excludes_two_processes(tmp_path: Path) -> None:
    """Two processes, one GPU: the second waits for the first and says so exactly once."""
    lock, log = tmp_path / "gpu.lock", tmp_path / "gpu.log"
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "MONTAJ_GPU_LOCK": str(lock)}
    snippet = HOLD_SNIPPET.format(repo=repo.as_posix(), log=log.as_posix())
    holder_ready, holder_release = tmp_path / "holder.ready", tmp_path / "holder.release"
    waiter_ready, waiter_release = tmp_path / "waiter.ready", tmp_path / "waiter.release"

    first = subprocess.Popen(
        [sys.executable, "-c", snippet, str(holder_ready), str(holder_release)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
    )
    second = None
    try:
        _wait_path(holder_ready)
        second = subprocess.Popen(
            [sys.executable, "-c", snippet, str(waiter_ready), str(waiter_release)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
        )
        _wait_text(log, "waiting for GPU")
        assert second.poll() is None, "the second holder entered while the lock was held"
        assert not waiter_ready.exists()
        holder_release.write_text("go")
        _wait_path(waiter_ready)
        waiter_release.write_text("go")
        assert first.wait(timeout=60) == 0, first.stderr.read() if first.stderr else ""
        assert second.wait(timeout=60) == 0, second.stderr.read() if second.stderr else ""
    finally:
        for proc in (first, second):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
    assert log.read_text().count("waiting for GPU") == 1
    assert lock.is_file()


def test_hold_releases_on_error_and_reuses_one_lock_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONTAJ_GPU_LOCK", str(tmp_path / "gpu.lock"))
    log = tmp_path / "nested.log"
    with pytest.raises(RuntimeError), hold(log):
        raise RuntimeError("boom")
    with hold(log) as path:
        assert path.is_file()
    assert lock_path() == tmp_path / "gpu.lock"
    assert "waiting for GPU" not in log.read_text()  # nothing ever waited


def test_music_commands_hold_the_lock(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("MONTAJ_GPU_LOCK", str(lock))
    monkeypatch.setenv("MONTAJ_CONFIG", str(config_file(tmp_path)))
    proj = project(tmp_path)
    code, lines, err = _run(["-C", str(proj), "music", "analyze", "song.wav"], capsys)
    assert code == 0, lines
    assert "gpu lock" in (proj / "build" / "music.log").read_text()
    assert lock.is_file()

def test_default_prompt_comes_from_lyrics_beside_the_wav(tmp_path: Path) -> None:
    from montaj.audio.analyze import default_prompt

    wav = tmp_path / "cand-1.wav"
    assert default_prompt(wav) == ""  # no lyrics: no hint, never another film's names
    (tmp_path / "lyrics.txt").write_text("[verse]\nhappy birthday Arshiya\n\n[chorus]\njanamdin mubarak\n")
    assert default_prompt(wav) == "happy birthday Arshiya, janamdin mubarak"
