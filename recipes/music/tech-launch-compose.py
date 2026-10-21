"""12 s, 120 BPM electronic bed for recipes/tech-launch.yaml: pad + bass + four-on-the-floor + hats. Output: argv[1]."""
from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import numpy as np

BPM = 120.0
SR = int(os.environ.get("MONTAJ_SR", "48000"))
DURATION = 12.0


def midi(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def put(track: np.ndarray, start: float, sig: np.ndarray) -> None:
    first = int(start * SR)
    n = min(len(track) - first, len(sig))
    if n > 0:
        track[first:first + n] += sig[:n]


def tone(length: float, freq: float, level: float, decay: float, harm: float = 0.3) -> np.ndarray:
    t = np.arange(int(length * SR), dtype=np.float32) / SR
    env = np.minimum(1.0, t * 60.0) * np.exp(-decay * t)
    saw = np.sin(2 * np.pi * freq * t) + harm * np.sin(4 * np.pi * freq * t) + 0.5 * harm * np.sin(6 * np.pi * freq * t)
    return (level * env * saw).astype(np.float32)


def kick() -> np.ndarray:
    t = np.arange(int(0.25 * SR), dtype=np.float32) / SR
    f = 120.0 - 70.0 * np.minimum(1.0, t * 14.0)
    return (0.8 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-14.0 * t)).astype(np.float32)


def hat(seed: int) -> np.ndarray:
    n = int(0.05 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    noise = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    return 0.07 * np.diff(noise, prepend=0.0) * np.exp(-60.0 * t)


def main() -> None:
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    beat = 60.0 / BPM
    track = np.zeros(int(DURATION * SR), dtype=np.float32)
    chords = ((57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62))  # one bar (4 beats) each
    for bar in range(int(DURATION / (4 * beat))):
        chord = chords[bar % 4]
        for n in chord:
            put(track, bar * 4 * beat, tone(4 * beat, midi(n + 12), 0.07, 0.5))
        for k in range(8):  # eighth-note bass
            put(track, bar * 4 * beat + k * beat / 2, tone(beat / 2, midi(chord[0] - 12), 0.22, 6.0, 0.15))
    for i in range(int(DURATION / beat)):
        put(track, i * beat, kick())
        put(track, i * beat + beat / 2, hat(i))
    track *= 0.78 / (float(np.max(np.abs(track))) or 1.0)
    fo = int(0.4 * SR)
    track[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    pcm = np.clip(np.column_stack((track, track)) * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
