"""A calm, curious 96 BPM explainer bed for `montaj music compose` (numpy only), 45 s = 72 beats.

Sections follow recipes/explainer.yaml: hook 0-8b (pad + pluck), parts 8b-60b (adds soft kick and hats,
a lift at 20b and 40b), takeaway 60b-72b (drums drop, pad resolves). The first beat is exactly t=0.
"""
from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import numpy as np

BPM = 96.0
SR = int(os.environ.get("MONTAJ_SR", "48000"))
BEAT = 60.0 / BPM
BEATS = 72
DURATION = BEATS * BEAT


def midi(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def add_tone(track: np.ndarray, start: float, length: float, freq: float, level: float,
             attack: float, decay: float, bright: float = 0.16) -> None:
    first = max(0, int(start * SR))
    count = min(len(track) - first, max(0, int(length * SR)))
    if count <= 0:
        return
    t = np.arange(count, dtype=np.float32) / SR
    env = np.minimum(1.0, t / attack) * np.exp(-decay * t)
    env *= np.minimum(1.0, (count - np.arange(count)) / (0.05 * SR))  # no click at the note end
    tone = np.sin(2 * np.pi * freq * t) + bright * np.sin(4 * np.pi * freq * t) + 0.05 * np.sin(6 * np.pi * freq * t)
    track[first:first + count] += level * env * tone


def add_kick(track: np.ndarray, start: float, level: float) -> None:
    first = int(start * SR)
    count = min(len(track) - first, int(0.25 * SR))
    if count <= 0:
        return
    t = np.arange(count, dtype=np.float32) / SR
    freq = 95.0 - 50.0 * np.minimum(1.0, t * 10.0)
    track[first:first + count] += level * np.sin(2 * np.pi * freq * t) * np.exp(-14.0 * t)


def add_hat(track: np.ndarray, start: float, level: float, seed: int) -> None:
    first = int(start * SR)
    count = min(len(track) - first, int(0.06 * SR))
    if count <= 0:
        return
    t = np.arange(count, dtype=np.float32) / SR
    noise = np.random.default_rng(seed).standard_normal(count).astype(np.float32)
    noise = np.diff(noise, prepend=0.0)  # crude high-pass: a tick, not a hiss
    track[first:first + count] += level * noise * np.exp(-70.0 * t)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: explainer-compose.py OUTPUT.wav")
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    track = np.zeros(int(DURATION * SR), dtype=np.float32)
    bar = 4 * BEAT

    # Pad: Dmaj7 - Bm7 - Gmaj7 - A, one chord per bar, slow attack.
    chords = ((62, 66, 69, 73), (59, 62, 66, 69), (55, 59, 62, 66), (57, 61, 64, 69))
    for b in range(BEATS // 4):
        chord = chords[b % 4] if b < 16 else chords[0]  # the takeaway resolves on the tonic
        for note in chord:
            add_tone(track, b * bar, bar * 1.1, midi(note), 0.05, 0.35, 0.4, bright=0.05)
        add_tone(track, b * bar, bar * 0.95, midi(chord[0] - 24), 0.13, 0.02, 1.2)

    # Pluck arpeggio in eighths: the "thinking" motif.
    for e in range(BEATS * 2):
        chord = chords[(e // 8) % 4] if e < 128 else chords[0]
        note = chord[(0, 2, 1, 3, 2, 1, 3, 2)[e % 8]] + 12
        level = 0.05 if e < 16 or e >= 120 else 0.065
        add_tone(track, e * BEAT / 2, BEAT * 0.9, midi(note), level, 0.004, 7.0, bright=0.3)

    # Drums only under the three parts (8b-60b); a soft lift with a high bell at 20b and 40b.
    for i in range(8, 60):
        if i % 2 == 0:
            add_kick(track, i * BEAT, 0.42)
        add_hat(track, i * BEAT + BEAT / 2, 0.05, seed=i)
    for lift in (20, 40):
        add_tone(track, lift * BEAT, 2 * BEAT, midi(86), 0.06, 0.005, 2.5, bright=0.4)

    peak = float(np.max(np.abs(track))) or 1.0
    track = track * (0.75 / peak)
    fade_in, fade_out = int(0.03 * SR), int(1.0 * SR)
    track[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    track[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    stereo = np.column_stack((track * 0.97, track)).astype(np.float32)
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
