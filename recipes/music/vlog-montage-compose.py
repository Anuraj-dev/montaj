"""60 s, 100 BPM warm indie bed for recipes/vlog-montage.yaml (1 beat = 0.6 s, 100 beats). Output: argv[1].

Sections follow the recipe's chapters: 0-8b pad alone (the opening shot's own sound carries the scene), 8-32b
drums + bass + pluck, 32-64b full groove, 64-72b breakdown (pad + pluck), 72-96b full groove, 96-100b ring-out.
"""
from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import numpy as np

BPM = 100.0
SR = int(os.environ.get("MONTAJ_SR", "48000"))
BEAT = 60.0 / BPM
DURATION = 100 * BEAT + 1.0


def midi(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def put(track: np.ndarray, start: float, sig: np.ndarray) -> None:
    first = int(round(start * SR))
    n = min(len(track) - first, len(sig))
    if n > 0:
        track[first:first + n] += sig[:n]


def tone(length: float, freq: float, level: float, decay: float, harm: float = 0.3, attack: float = 60.0) -> np.ndarray:
    t = np.arange(int(length * SR), dtype=np.float32) / SR
    env = np.minimum(1.0, t * attack) * np.exp(-decay * t)
    wave_ = np.sin(2 * np.pi * freq * t) + harm * np.sin(4 * np.pi * freq * t) + 0.4 * harm * np.sin(6 * np.pi * freq * t)
    return (level * env * wave_).astype(np.float32)


def kick() -> np.ndarray:
    t = np.arange(int(0.3 * SR), dtype=np.float32) / SR
    f = 110.0 - 60.0 * np.minimum(1.0, t * 12.0)
    return (0.75 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-11.0 * t)).astype(np.float32)


def noise(length: float, seed: int, decay: float, level: float, hp: bool) -> np.ndarray:
    n = int(length * SR)
    t = np.arange(n, dtype=np.float32) / SR
    x = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    if hp:
        x = np.diff(x, prepend=0.0)
    return level * x * np.exp(-decay * t)


def snare(seed: int) -> np.ndarray:
    return noise(0.25, seed, 18.0, 0.16, False) + tone(0.25, 190.0, 0.18, 20.0, 0.0)


def main() -> None:
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    track = np.zeros(int(DURATION * SR), dtype=np.float32)
    # I - V - vi - IV in D major, one bar (4 beats) each
    chords = ((62, 66, 69), (57, 61, 64), (59, 62, 66), (55, 59, 62))
    for bar in range(25):
        b0 = bar * 4
        start = b0 * BEAT
        chord = chords[bar % 4]
        full = 8 <= b0 < 64 or 72 <= b0 < 96
        groove = 32 <= b0 < 64 or 72 <= b0 < 96
        for n in chord:  # soft pad, every section
            put(track, start, tone(4 * BEAT + 0.3, midi(n), 0.05, 0.35, 0.2, attack=6.0))
        if b0 >= 8 and b0 < 96:  # plucked arpeggio in eighths
            for k in range(8):
                note = chord[k % 3] + (12 if k >= 4 else 0)
                put(track, start + k * BEAT / 2, tone(0.5, midi(note), 0.06 if groove else 0.045, 7.0, 0.25))
        if full:
            for k in range(4):  # bass on the beats
                put(track, start + k * BEAT, tone(BEAT, midi(chord[0] - 24), 0.2, 3.5, 0.2))
            for k in range(4):
                beat = b0 + k
                if k in (0, 2) or (groove and k == 3 and bar % 2):
                    put(track, beat * BEAT, kick())
                if k in (1, 3):
                    put(track, beat * BEAT, snare(beat))
                put(track, beat * BEAT + BEAT / 2, noise(0.05, 1000 + beat, 60.0, 0.06, True))
                if groove:
                    put(track, beat * BEAT, noise(0.04, 2000 + beat, 70.0, 0.04, True))
    # final chord rings out from 96b
    for n in chords[0]:
        put(track, 96 * BEAT, tone(5.0, midi(n), 0.07, 0.7, 0.25, attack=20.0))
    put(track, 96 * BEAT, kick())
    track *= 0.8 / (float(np.max(np.abs(track))) or 1.0)
    fo = int(0.8 * SR)
    track[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    pcm = np.clip(np.column_stack((track, track)) * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
