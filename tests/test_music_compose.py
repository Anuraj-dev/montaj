"""W8b music compose and beat-only analysis contracts."""
from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from montaj import config as config_mod
from montaj import cli as cli_mod
from montaj.audio.analyze import Analysis
from montaj.audio.music import Candidates
from montaj.cli import main
from montaj.doctor import config_lines


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, list[str], str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


def _write_pcm_wav(path: Path, *, seconds: float = 1.0, sample: int = 16000) -> None:
    samples = np.full(round(seconds * 48000), sample, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(samples.tobytes())


def _float_wav_script(path: Path, peak_db: float) -> None:
    peak = 10 ** (peak_db / 20)
    path.write_text(
        "import math, struct, sys\n"
        f"peak = {peak!r}\n"
        "samples = [peak * math.sin(2 * math.pi * 440 * i / 48000) for i in range(48000)]\n"
        "samples[1] = peak\n"
        "raw = struct.pack('<' + 'f' * len(samples), *samples)\n"
        "fmt = struct.pack('<4sIHHIIHH', b'fmt ', 16, 3, 1, 48000, 192000, 4, 32)\n"
        "data = b'data' + struct.pack('<I', len(raw)) + raw\n"
        "riff = b'RIFF' + struct.pack('<I', 4 + len(fmt) + len(data)) + b'WAVE' + fmt + data\n"
        "open(sys.argv[1], 'wb').write(riff)\n"
    )


def test_music_compose_example_writes_valid_30_second_stereo_wav(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("MONTAJ_GPU_LOCK", str(lock))
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    script = Path(__file__).parents[1] / "recipes" / "music" / "compose-example.py"

    code, lines, err = _run(
        ["-C", str(project), "music", "compose", str(script), "--out", "music/song.wav"], capsys,
    )

    assert code == 0 and err == "", lines
    assert lines[-1].startswith("OK music/song.wav 30.00s 48000Hz 2ch peak ")
    assert lines[-1].endswith("LUFS")
    assert not any(line.startswith("WARN") for line in lines)
    assert not lock.exists()
    with wave.open(str(project / "music/song.wav"), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (48000, 2, 2)
        assert wav.getnframes() == 30 * 48000


def test_music_compose_passes_absolute_output_and_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    script = project / "check.py"
    script.write_text(
        "import os, pathlib, sys, wave\n"
        "assert pathlib.Path.cwd() == pathlib.Path(sys.argv[1]).parents[1]\n"
        "assert pathlib.Path(sys.argv[1]).is_absolute()\n"
        "assert os.environ['MONTAJ_OUT'] == sys.argv[1]\n"
        "assert os.environ['MONTAJ_SR'] == '48000'\n"
        "with wave.open(sys.argv[1], 'wb') as f:\n"
        "    f.setnchannels(1); f.setsampwidth(2); f.setframerate(48000)\n"
        "    f.writeframes((b'\\x00\\x40' * 48000))\n"
    )

    code, lines, err = _run(["-C", str(project), "music", "compose", "check.py"], capsys)

    assert code == 0 and err == "", lines
    assert lines[-1].startswith("OK music/song.wav 1.00s 48000Hz 1ch peak ")


def test_music_compose_does_not_accept_a_stale_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    output = project / "music/song.wav"
    output.parent.mkdir()
    _write_pcm_wav(output)
    old = output.read_bytes()
    script = project / "noop.py"
    script.write_text("# deliberately leaves MONTAJ_OUT untouched\n")

    code, lines, err = _run(["-C", str(project), "music", "compose", "noop.py"], capsys)

    assert code == 1 and err == ""
    assert lines == ["ERR music compose: output WAV missing: " + str(output) + " (see build/music.log)"]
    assert output.read_bytes() == old


@pytest.mark.parametrize("bpm", ["0", "-1", "nan", "inf", "-inf"])
def test_music_analyze_rejects_nonpositive_or_nonfinite_bpm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, bpm: str,
) -> None:
    monkeypatch.setenv("MONTAJ_CONFIG", str(tmp_path / "missing.toml"))
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    _write_pcm_wav(project / "song.wav", seconds=2.0)

    code, lines, err = _run(
        ["-C", str(project), "music", "analyze", "song.wav", "--no-words", f"--bpm={bpm}"], capsys,
    )

    assert code == 1 and err == ""
    assert lines == ["ERR music: bpm must be finite and positive"]


@pytest.mark.parametrize(
    ("peak_db", "code", "marker"),
    [(-60.04, 1, "silent"), (-0.14, 0, "-0.1dBFS")],
)
def test_music_compose_peak_thresholds_use_float_precision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
    peak_db: float, code: int, marker: str,
) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    script = project / "float.py"
    _float_wav_script(script, peak_db)

    actual, lines, err = _run(["-C", str(project), "music", "compose", "float.py"], capsys)

    assert actual == code and err == "", lines
    assert marker in "\n".join(lines)
    if code == 0:
        assert not any(line.startswith("WARN") for line in lines)


def test_music_compose_logs_non_utf8_script_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    script = project / "bad-bytes.py"
    script.write_bytes(b"import sys\nsys.stderr.buffer.write(b'\\xff\\n')\nsys.exit(1)\n")

    code, lines, err = _run(["-C", str(project), "music", "compose", "bad-bytes.py"], capsys)

    assert code == 1 and err == ""
    assert lines == ["ERR music compose: � (see build/music.log)"]
    assert "�" in (project / "build/music.log").read_text()


def test_music_analyze_no_words_needs_no_machine_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONTAJ_CONFIG", str(tmp_path / "missing.toml"))
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("MONTAJ_GPU_LOCK", str(lock))
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    samples = np.zeros(3 * 16000, dtype="<i2")
    samples[0:80] = 24000
    with wave.open(str(project / "song.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())

    code, lines, err = _run(
        ["-C", str(project), "music", "analyze", "song.wav", "--no-words", "--bpm", "96"], capsys,
    )

    assert code == 0 and err == "", lines
    assert lines[-1] == "OK music/markers.json 3.0s bpm=96.0 beat0=0.00s no words"
    markers = json.loads((project / "music/markers.json").read_text())
    assert markers["words"] == [] and markers["segments"] == []
    assert markers["bpm"] == 96.0
    assert not lock.exists()


def test_config_validates_only_the_keys_a_command_needs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'whisper_python = "{sys.executable}"\n')
    monkeypatch.setenv("MONTAJ_CONFIG", str(path))
    cfg = config_mod.load(required=config_mod.WHISPER_KEYS)
    assert cfg.whisper_python == sys.executable
    with pytest.raises(config_mod.ConfigError, match="ace_step_dir missing"):
        config_mod.load(required=config_mod.ACE_KEYS)


def test_analyze_missing_whisper_suggests_no_words(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'ace_step_dir = "{tmp_path}"\nace_python = "{sys.executable}"\n'
                   f'ace_site_packages = "{tmp_path}"\n')
    monkeypatch.setenv("MONTAJ_CONFIG", str(cfg))
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    (project / "song.wav").write_bytes(b"not a wav")

    code, lines, err = _run(["-C", str(project), "music", "analyze", "song.wav"], capsys)

    assert code == 1 and err == ""
    assert lines == ["ERR music: config: whisper_python missing (set it in " + str(cfg)
                    + "); use --no-words for beat-only analysis"]


def test_music_gen_result_lines_echo_duration_and_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    output = project / "music" / "cand-123.wav"

    def fake_gen(out_dir: Path, **_kwargs: object) -> Candidates:
        output.parent.mkdir(parents=True)
        _write_pcm_wav(output, seconds=0.75)
        return Candidates((output,), ("123",))

    monkeypatch.setattr(cli_mod.music_gen, "gen", fake_gen)
    code, lines, err = _run(
        ["-C", str(project), "music", "gen", "--caption", "x", "--lyrics", "l.txt",
         "--duration", "1.5s", "--lang", "en", "--bpm", "96", "--key", "D major", "--n", "1"], capsys,
    )

    assert code == 0 and err == ""
    assert lines == [
        "music/cand-123.wav seed=123 0.8s",
        "OK music gen 1 candidates 1.5s lang=en bpm=96 key=D major",
    ]


def test_music_analyze_result_line_answers_timing_questions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    markers = project / "music" / "markers.json"
    monkeypatch.setattr(
        cli_mod.analyze_audio,
        "analyze",
        lambda *_args, **_kwargs: Analysis(
            path=markers, bpm=80.0, n_words=50, n_beats=50, lines=("6.00-8.00 w1-4 hello",),
            duration=30.0, first_beat=0.41, first_word=6.62, lang="en", no_words=False,
        ),
    )

    code, lines, err = _run(["-C", str(project), "music", "analyze", "song.wav", "--lang", "en"], capsys)

    assert code == 0 and err == ""
    assert lines == [
        "6.00-8.00 w1-4 hello",
        "OK music/markers.json 30.0s bpm=80.0 beat0=0.41s 50 words first_word=6.62s lang=en",
    ]


def test_music_compose_reports_last_stderr_line_and_keeps_it_in_the_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    script = project / "fail.py"
    script.write_text("import sys\nprint('first', file=sys.stderr)\nprint('last', file=sys.stderr)\nsys.exit(4)\n")

    code, lines, err = _run(["-C", str(project), "music", "compose", "fail.py"], capsys)

    assert code == 1 and err == ""
    assert lines == ["ERR music compose: last (see build/music.log)"]
    assert "last" in (project / "build/music.log").read_text()


def test_music_compose_rejects_silence_and_timeout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONTAJ_CONFIG", raising=False)
    project = tmp_path / "project"
    (project / "build").mkdir(parents=True)
    silent = project / "silent.py"
    silent.write_text(
        "import sys, wave\n"
        "with wave.open(sys.argv[1], 'wb') as f:\n"
        "    f.setnchannels(1); f.setsampwidth(2); f.setframerate(48000); f.writeframes(b'\\0\\0' * 48000)\n"
    )
    code, lines, err = _run(["-C", str(project), "music", "compose", "silent.py"], capsys)
    assert code == 1 and err == "" and "silent" in lines[0]

    slow = project / "slow.py"
    slow.write_text("import time\ntime.sleep(2)\n")
    code, lines, err = _run(["-C", str(project), "music", "compose", "slow.py", "--timeout", "0.05"], capsys)
    assert code == 1 and err == "" and lines == ["ERR music compose: timed out after 0.05s (see build/music.log)"]


def test_doctor_reports_missing_config_keys_individually(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'whisper_python = "{sys.executable}"\n')
    monkeypatch.setenv("MONTAJ_CONFIG", str(path))
    lines = config_lines()
    assert any(line == "FAIL ace_step_dir: missing" for line in lines)
    assert any(line == "FAIL ace_python: missing" for line in lines)
    assert any(line == "FAIL ace_site_packages: missing" for line in lines)
    assert any(line.startswith("OK whisper_python:") for line in lines)
