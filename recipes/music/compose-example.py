"""A small numpy-only celebratory backing track for `montaj music compose`."""
from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import numpy as np

BPM = 96.0
SR = int(os.environ.get("MONTAJ_SR", "48000"))
DURATION = 30.0


def midi(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def add_note(track: np.ndarray, start: float, length: float, frequency: float, level: float, decay: float) -> None:
    first = max(0, int(start * SR))
    count = min(len(track) - first, max(0, int(length * SR)))
    if count <= 0:
        return
    t = np.arange(count, dtype=np.float32) / SR
    envelope = np.minimum(1.0, t * 45.0) * np.exp(-decay * t)
    tone = np.sin(2.0 * np.pi * frequency * t) + 0.16 * np.sin(4.0 * np.pi * frequency * t)
    track[first:first + count] += level * envelope * tone


def add_kick(track: np.ndarray, start: float) -> None:
    first = int(start * SR)
    count = min(len(track) - first, int(0.22 * SR))
    if count <= 0:
        return
    t = np.arange(count, dtype=np.float32) / SR
    frequency = 110.0 - 65.0 * np.minimum(1.0, t * 12.0)
    track[first:first + count] += 0.72 * np.sin(2.0 * np.pi * frequency * t) * np.exp(-18.0 * t)


def add_snare(track: np.ndarray, start: float) -> None:
    first = int(start * SR)
    count = min(len(track) - first, int(0.16 * SR))
    if count <= 0:
        return
    t = np.arange(count, dtype=np.float32) / SR
    noise = np.random.default_rng(int(start * 1000) + 7).standard_normal(count).astype(np.float32)
    track[first:first + count] += 0.18 * noise * np.exp(-24.0 * t)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: compose-example.py OUTPUT.wav")
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    beat = 60.0 / BPM
    track = np.zeros(int(DURATION * SR), dtype=np.float32)

    # Four warm chords, one per bar. The first kick is exactly on t=0.
    progression = ((60, 64, 67), (57, 60, 64), (53, 57, 60), (55, 59, 62))
    bar = 4.0 * beat
    for index, chord in enumerate(progression * 3):
        start = index * bar
        for note in chord:
            add_note(track, start, bar * 1.08, midi(note + 12), 0.12, 0.8)
        add_note(track, start, bar * 0.9, midi(chord[0] - 12), 0.2, 2.0)

    # A simple, singable melody with a held note at the end of each phrase.
    melody = (72, 74, 76, 79, 76, 74, 72, 67, 69, 72, 74, 76, 74, 72, 69, 67)
    for index, note in enumerate(melody * 3):
        add_note(track, index * 2.0 * beat, beat * 1.65, midi(note), 0.11, 1.7)

    for index in range(int(DURATION / beat)):
        start = index * beat
        if index % 4 == 0:
            add_kick(track, start)
        if index % 4 in (1, 3):
            add_snare(track, start)
        if index % 2 == 1:
            add_note(track, start, 0.07, 5200.0, 0.045, 42.0)

    peak = float(np.max(np.abs(track))) or 1.0
    track = track * (0.78 / peak)
    fade_in = min(int(0.04 * SR), len(track))
    fade_out = min(int(0.6 * SR), len(track))
    track[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    track[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    stereo = np.column_stack((track * 0.98, track)).astype(np.float32)
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")

    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
