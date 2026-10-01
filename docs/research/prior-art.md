# Prior art to port — pointers, not copies
> All paths under `~/Anuraj-dev/claude-test/`. Read the source only when porting that piece.

| Montaj module | Port from | What's there |
|---|---|---|
| render/core.py | `bday-video/v2/film.py` | `Img` (texture + crop + print/card border + shadow), `camera()` pinhole, `draw()`, `solo()`, `shift()`, easing (`smooth/cubic/outc/inc/expo/ramp`), `key()/keys_s()` keyframes, `fit_s()` |
| render/transitions.py | `bday-video/v2/film2.py` | `swirl`, `whip`, `fade`, match-cut framings, wall of prints + flat drone move (`build_wall`, `s_wall`) |
| render/finish.py | `bday-video/v2/film2.py` `finish()`, `render_frame()` | sharpen, S-curve grade, vignette, light leaks, flashes, bloom; SUB=6 sub-frame motion blur |
| render/encode.py | film2 `__main__`, `bday-video/render.sh` | rawvideo → `h264_nvenc -preset p7 -tune hq -rc vbr -cq 16`; loudnorm I=-14; WhatsApp 720p export |
| render/text.py | `design-test/showcase-edit/render.sh` + `cards.html` | headless Chromium `--screenshot` of `cards.html#<id>` with transparent background |
| text styles | `bday-video/film.html` CSS | Cormorant / Great Vibes / Tiro Devanagari, gold gradient text, subtitles; fonts in `bday-video/fonts/` |
| audio/music.py | `bday-video/music/generate.py`, `music/py.sh` | ACE-Step 1.5 turbo, 8 GB offload flags, caption/lyrics/bpm/key/duration params |
| audio/analyze.py | `bday-video/music/analyze.py` | faster-whisper large-v3-turbo word timestamps (venv: `design-test/voice-bot/.venv`), 0.5 s RMS envelope |
| ingest depth (optional) | `bday-video/v2/depth.py` | Depth-Anything-V2-Large depth maps |
| M5 motion recipes | `reel/reel.html`, `portfolio-video/reel.html`, `Sonnet 5.5/{common,reel,audio}.py` | pure-function-of-t motion graphics, beat timelines, synth scores |
| CDP exporter (fallback only) | `bday-video/export.mjs` | parallel headless Chromium frame capture, sub-frame shutter |

## Known traps
- Headless Chromium renders on CPU (SwiftShader) regardless of GPU flags — measured, no speedup. Don't retry.
- ACE-Step venv `bin/python` symlink is dangling → run uv CPython 3.12 with its site-packages (`music/py.sh`).
- 8 GB VRAM: ACE-Step needs `offload_to_cpu` + `offload_dit_to_cpu`, else the LM phase OOMs. Don't render at the same time.
- ACE-Step location: `design-test/voice-bot/tools/ACE-Step-1.5` (19 GB) — reference via config, don't move.
- Taste: no photo warp/drift/depth-parallax on stills; motion lives in transitions; cuts on a beat grid.
