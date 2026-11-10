"""Helpers for recipes/talking-short.yaml. Run with the montaj venv's python (`uv run python recipes/talking-short.py`).

  demo  PROJECT  write the synthetic talking head: assets/voice.mov (picture + a voice-like signal),
                 assets/voice.words.json (its word timings), assets/broll1.mp4, assets/broll2.mp4.
  words PROJECT  map the voice's words through the film's cut layout into talk.words.json beside the spec
                 (film time: what `captions` and `emphasis` read) and print one row per cut span:
                 span, source frames, film frames, text. Run it after `montaj cut voice` and after every
                 edit that changes which spans play or where a b-roll shot sits.

Why `words` exists: `cuts: auto` removes pauses, so the film runs ahead of the source clip, but `captions`
and `emphasis` read word times as film times. The map is built from the shots in order: a `cuts: auto`
shot plays its kept spans (same padding and rounding as the engine), a `layers` shot holding the voice
clip plays `clip_in .. clip_in + hold`, anything else just takes its hold.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

FPS = 30
SR = 48000
VOICE = "voice"
PAD = (0.06, 0.12)  # montaj.cuts.DEFAULT_PAD_S: kept before / after each span
GAP_PAUSE = 0.7  # silence between sentences: longer than montaj cut's 0.35 s, so it is cut
WORD_GAP = 0.045

# One entry per kept span. "um" between two halves is a filler montaj cut removes.
SCRIPT = [
    "Stop editing talking videos by hand.",
    "This whole short came from one small text file.",
    ("Every pause you leave in,", "um", "gets cut for you."),
    "Captions land on every word, and the big ones punch.",
    "When I say zoom, the camera pushes in.",
    "When I say boom, you hear it.",
    "B-roll pops in over the top while I keep talking.",
    "Three quick tips.",
    "Hook them in the first second.",
    "Cut every breath and every pause.",
    "Keep it under forty seconds.",
    "Then render it again, for free.",
    "Follow for part two.",
]


# --- demo assets -------------------------------------------------------------------------------------------


def _layout() -> list[dict]:
    """Word timings in source seconds. Every span's padded edges sit on whole frames, so the cut plan
    (and the b-roll shots' frame numbers in the recipe) come out the same on any machine."""
    words: list[dict] = []
    frame = 9  # 0.3 s of room before the first word

    def phrase(text: str, start_frame: int) -> int:
        """Lay `text` out from `start_frame`; return the frame its padded end lands on."""
        t = start_frame / FPS + PAD[0]
        toks = text.split()
        for n, w in enumerate(toks):
            letters = len(re.sub(r"[^A-Za-z]", "", w))
            d = 0.11 + 0.055 * letters
            words.append({"w": w, "s": round(t, 4), "e": round(t + d, 4)})
            t += d + WORD_GAP + (0.12 if w.endswith(",") and n + 1 < len(toks) else 0)
        end = math.ceil((words[-1]["e"] + PAD[1]) * FPS)
        words[-1]["e"] = round(end / FPS - PAD[1], 4)  # stretch the last word onto the frame grid
        return end

    for item in SCRIPT:
        if isinstance(item, tuple):
            a, filler, b = item
            end = phrase(a, frame)
            s = (end + 3) / FPS  # 0.1 s after the padded end, 0.22 s after the word
            words.append({"w": filler, "s": round(s, 4), "e": round(s + 0.3, 4)})
            frame = end + 22  # the filler and silence around it: 0.73 s
            frame = phrase(b, frame)
        else:
            frame = phrase(item, frame)
        frame += round(GAP_PAUSE * FPS)
    for i, w in enumerate(words, 1):
        w["i"] = i
    return words


def _voice(words: list[dict], total: float) -> np.ndarray:
    """A voice-like signal: a glottal harmonic series through two vowel formants, one syllable bump per
    vowel group, a breathy consonant onset, falling pitch over each sentence. Silence between words."""
    out = np.zeros(int(total * SR), dtype=np.float64)
    rng = np.random.default_rng(48)
    vowels = [(730, 1090), (270, 2290), (530, 1840), (300, 870), (640, 1190), (490, 1350)]
    sent_pos = 0
    for w in words:
        a, b = int(w["s"] * SR), int(w["e"] * SR)
        n = b - a
        u = np.arange(n) / SR
        filler = w["w"].lower() == "um"
        f0 = (125.0 if filler else 165.0 - 4.0 * min(sent_pos, 8)) + 6.0 * np.sin(2 * np.pi * 3.1 * (u + w["s"]))
        phase = 2 * np.pi * np.cumsum(f0) / SR
        f1, f2 = (250, 600) if filler else vowels[sum(map(ord, w["w"])) % len(vowels)]
        f0m = float(f0.mean())
        sig = np.zeros(n)
        for k in range(1, 40):
            fk = k * f0m
            if fk > 5000:
                break
            amp = (1 / (1 + ((fk - f1) / 90) ** 2) + 0.6 / (1 + ((fk - f2) / 120) ** 2)) / k ** 0.6
            sig += amp * np.sin(k * phase)
        syl = max(1, len(re.findall(r"[aeiouy]+", w["w"].lower())))
        env = 0.45 + 0.55 * np.sin(np.pi * syl * u / max(u[-1], 1e-6)) ** 2
        ramp_in, ramp_out = int(0.012 * SR), int(0.02 * SR)
        env[:ramp_in] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp_in))
        env[-ramp_out:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, ramp_out))
        sig *= env
        if not filler:  # consonant: a short noise burst
            m = min(n, int(0.03 * SR))
            sig[:m] += 0.25 * np.diff(rng.standard_normal(m + 1)) * np.linspace(1, 0, m)
        out[a:b] += sig * (0.45 if filler else 1.0)
        sent_pos = 0 if w["w"][-1] in ".?!" else sent_pos + 1
    return out / np.abs(out).max() * 0.6


def _write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def demo(proj: Path) -> None:
    assets = proj / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    words = _layout()
    total = words[-1]["e"] + 1.0
    _write_wav(assets / "voice.wav.tmp", _voice(words, total))
    font = subprocess.run(["fc-match", "-f", "%{file}", "sans:bold"], capture_output=True, text=True).stdout.strip()
    # A stand-in presenter: head and shoulders that sway, and a source timecode so the jump cuts show on the
    # contact sheet. The fast nod terms keep the head moving where both slow sways peak together (freeze QA).
    talk = (
        f"[0]drawbox=0:1250:1080:670:0x23324a:fill,"
        f"drawtext=fontfile={font}:text='SOURCE %{{pts\\:hms}}':fontsize=34:fontcolor=0x9fb3c8:x=60:y=150[bg];"
        f"[1]format=rgba,geq=r=224:g=180:b=138:a='255*lte(hypot((X-170)/170,(Y-210)/210),1)',loop=loop=-1:size=1[head];"
        f"[2]format=rgba,geq=r=59:g=110:b=165:a='255*lte(abs(X-330),180+Y*0.4)',loop=loop=-1:size=1[body];"
        f"[bg][body]overlay=x='210+30*sin(t*1.7)':y='1080+8*sin(t*2.9)'[a];"
        f"[a][head]overlay=x='370+40*sin(t*1.7)+10*sin(t*5.3)':y='600+20*sin(t*2.9)+8*sin(t*6.7)',"
        f"drawtext=fontfile={font}:text='TALKING HEAD':fontsize=40:fontcolor=0x9fb3c8:x=(w-tw)/2:y=1780,"
        f"format=yuv420p[v]"
    )
    _ff("-f", "lavfi", "-i", f"color=c=0x1b2433:s=1080x1920:r={FPS}:d={total}",
        "-f", "lavfi", "-i", f"color=c=black:s=340x420:r={FPS}:d=0.1",
        "-f", "lavfi", "-i", f"color=c=black:s=660x840:r={FPS}:d=0.1",
        "-i", str(assets / "voice.wav.tmp"),
        "-filter_complex", talk, "-map", "[v]", "-map", "3:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-g", "30", "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le", "-ar", str(SR), "-ac", "2", "-t", f"{total:.3f}", str(assets / "voice.mov"))
    (assets / "voice.wav.tmp").unlink()
    (assets / "voice.words.json").write_text(json.dumps({"words": words}, indent=1) + "\n", encoding="utf-8")
    # B-roll: two short generated clips (a fractal zoom, a cellular field), 16:9 and 9:16.
    _ff("-f", "lavfi", "-i", f"mandelbrot=s=640x360:r={FPS}:maxiter=256:end_scale=0.05:outer=normalized_iteration_count",
        "-t", "8", "-vf", "scale=1280:720,hue=h=200,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        str(assets / "broll1.mp4"))
    _ff("-f", "lavfi", "-i", f"life=s=270x480:r={FPS}:mold=8:ratio=0.12:life_color=#ffd166:death_color=#14213d:seed=7",
        "-t", "8", "-vf", "scale=1080:1920:flags=neighbor,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "24", str(assets / "broll2.mp4"))


# --- film-time words ---------------------------------------------------------------------------------------


def _frames(raw, fps: int) -> float:
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(f|s|ms)\s*", str(raw))
    if not m:
        raise SystemExit(f"ERR words: cannot read duration {raw!r} (use f, s or ms)")
    v, unit = float(m.group(1)), m.group(2)
    return v if unit == "f" else v * fps if unit == "s" else v * fps / 1000


def words(proj: Path) -> list[str]:
    from montaj.cuts import DEFAULT_PAD_S, _padded, read_plan

    spec = yaml.safe_load((proj / "montaj.yaml").read_text(encoding="utf-8"))
    fps = int(spec["video"]["fps"])
    assets = proj / spec.get("assets", "assets")
    clip = next((sh["clip"] for sh in spec["shots"] if sh.get("cuts")), VOICE)
    plan = read_plan(assets / f"{clip}.cuts.json")
    spans = plan["spans"]
    source = json.loads((assets / f"{clip}.words.json").read_text(encoding="utf-8"))["words"]
    # Film segments: (film start frame, source start frame or None, frames)
    segs: list[tuple[int, int | None, int]] = []
    film = 0
    for sh in spec["shots"]:
        if sh.get("cuts"):
            pad = tuple(_frames(p, fps) / fps for p in sh["cuts_pad"]) if sh.get("cuts_pad") else DEFAULT_PAD_S
            drop = set(sh.get("drop") or ())
            keep = [k for k, sp in enumerate(spans) if sp["i"] not in drop]
            prev: tuple[int, int] | None = None
            for n, k in enumerate(keep):
                s, e = _padded(spans, k, pad, float(plan.get("duration") or 0) or None)
                s_f, e_f = round(s * fps), round(e * fps)
                if prev is not None and keep[n - 1] + 1 == k:
                    s_f = max(s_f, prev[1])
                e_f = max(e_f, s_f + 1)
                prev = (s_f, e_f)
                segs.append((film, s_f, e_f - s_f))
                film += e_f - s_f
            continue
        hold = round(_frames(sh["hold"], fps)) if "hold" in sh else None
        if hold is None:
            raise SystemExit("ERR words: a shot without hold next to cuts (use hold: <n>f)")
        layer = next((ly for ly in sh.get("layers") or () if ly.get("clip") == clip), None)
        src = round(_frames(layer.get("clip_in", "0f"), fps)) if layer is not None else None
        if sh.get("clip") == clip:
            src = round(_frames(sh.get("clip_in", "0f"), fps))
        segs.append((film, src, hold))
        film += hold
    out_words = []
    for w in source:
        sf = float(w["s"]) * fps
        for f0, s0, n in segs:
            if s0 is not None and s0 <= sf < s0 + n:
                off = (f0 - s0) / fps
                e = min(float(w["e"]) + off, (f0 + n) / fps)
                out_words.append({"i": w["i"], "w": w["w"], "s": round(w["s"] + off, 4), "e": round(e, 4)})
                break
    (proj / "talk.words.json").write_text(json.dumps({"words": out_words}, indent=1) + "\n", encoding="utf-8")
    rows = ["span  src_in  frames  film_at  text"]
    for k, sp in enumerate(spans):
        s, e = _padded(spans, k, DEFAULT_PAD_S, float(plan.get("duration") or 0) or None)
        s_f, e_f = round(s * fps), round(e * fps)
        at = next((f0 + s_f - s0 for f0, s0, n in segs if s0 is not None and s0 <= s_f < s0 + n), None)
        rows.append(f"{sp['i']:>4}  {s_f:>5}f  {e_f - s_f:>5}f  {'-' if at is None else f'{at}f':>7}  {sp['text']}")
    rows.append(f"OK talk.words.json {len(out_words)}/{len(source)} words  film {film}f {film / fps:.2f}s")
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("demo", "words"):
        print("usage: talking-short.py demo|words PROJECT", file=sys.stderr)
        return 2
    proj = Path(argv[1])
    if argv[0] == "demo":
        demo(proj)
        print(f"OK {proj.as_posix()}/assets voice.mov voice.words.json broll1.mp4 broll2.mp4")
    else:
        print("\n".join(words(proj)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
