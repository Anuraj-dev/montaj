# 002 — M2 contract: text, music, clips, golden-film
> Interfaces every M2 module codes against. Change this file first, then the code. 001 still holds unless
> this file says otherwise. Target: `recipes/birthday-song-film.yaml` renders the Arshiya film
> (`claude-test/bday-video/film.html`, song `music/song.wav`, 165 s, 1080x1920, 30 fps).

## Parity oracle
`film.html` is a browser film: every visual property is a pure function of `t`. Montaj ports its behaviour
to the GPU. Pixel parity is impossible (browser text raster, CSS blur), so the gate is visual: PSNR of
montaj frames against Chromium frames of `film.html` (`node export.mjs <dir> 4 1 <frames>`, one sub-frame,
no motion blur) at sampled frames. Every formula below quotes the `film.html` line it ports; follow the
source when this file is ambiguous and note the gap.

CSS semantics that matter (all in sRGB 0..1 values, the way Chromium composites):
- `blur(r)` = gaussian with σ = r design px. `text-shadow`/`box-shadow`/`drop-shadow` blur radius R → σ = R/2.
- `opacity a` composites a layer as `out = a·layer_premult + (1 − a·alpha)·below` (plain source-over).
- `mix-blend-mode: screen` with layer alpha α: `out = (1−α)·b + α·(b + s − b·s)`.
- `mix-blend-mode: overlay`: `B = b ≤ .5 ? 2·b·s : 1 − 2(1−b)(1−s)`, then mixed by layer alpha/opacity.
- `filter: brightness(x)` multiplies rgb; `contrast(c)`: `(v − .5)·c + .5`; `saturate(s)`: the W3C Filter
  Effects saturate matrix (luma .2126/.7152/.0722). Functions apply in the order written.
- Gradients interpolate premultiplied colours linearly between stops.

## Time
- Easing: film.html `es` is **`sine(x) = .5 − .5·cos(πx)`** (new, added to `core.EASE`), not M1's
  smoothstep `smooth`; `eo` = `outc`; `eio` = `cubic`. Use `sine` wherever film.html uses `es`.
- A **time expression** is a duration from film start (`12.0s`, `360f`, `16b`) or `<marker>[±<duration>]`
  (`ch1`, `ch1+2.4s`, `pre2-6f`). Names match `[a-z][a-z0-9_]*`.
- `parse_time(raw, markers, fps, bpm) -> float` returns **float frames, not rounded**. Text reveals need
  sub-frame precision. Shot cut points (`until`) round with `round_frame`.
- Unknown marker → `text[2].lines[0].at: unknown marker 'ch9' (have: ch1, ch1b, …)`.
- Text-internal timings (`stagger`, `dur`, `fade`, …) are durations converted to float frames, unrounded.

## Spec additions (montaj/spec.py; all models `extra="forbid"`)
```yaml
video:
  background: "#07060a"   # default "#000000"; what shows through a fade from/to nothing
  look: golden-film       # new look (see Finish); warm-film unchanged
  outro: {type: fade, dur: 1.2s}   # new outro type: everything fades to black over the last dur
markers:                  # name -> time expression from film start, or `word:<i>` (start of word i in audio.markers)
  hum: 5.2s
  v1: word:1
audio:                    # optional; no audio block = silent film (M1 behaviour)
  track: music/song.wav   # relative to the spec file; sha-hashed like a photo
  markers: music/markers.json   # optional; needed only for `word:` markers
  fade_out: 3s            # default 0s; afade at the film's end
  loudnorm: true          # default true (-14 LUFS, TP -1.5, LRA 11)
shots:
  - photo: hands-light
    until: v1+0.3s        # instead of hold: the shot ends at this time (exactly one of hold | until)
    focus: [512, 430]     # source px, as before
    drift: {zoom: [1.25, 1.04], pan: [0, 40]}  # Ken Burns (see Shots)
    tone: {brightness: .5, saturate: .9, contrast: 1.05}   # CSS filter, fixed order b→s→c; each default 1
    frame: {caption: "my shy girl", tilt: [2, -1.5]}       # polaroid card on a blurred copy (film.html mode 'frame')
    morph: {photo: masjid-painted, at: 0.9s, dur: 1.6s, center: [.55, .5]}  # at: from shot start; center: screen fraction
    pulse: heartbeat      # beat-locked scale pulse on video.bpm, phase from the shot start
    in: {type: fade, dur: 0.5s, blur: 14}   # new: blur (design px) on the fading layers; default 0
  - clip: intro-video     # video file stem in assets/ (mp4, mov, webm, mkv); photo | wall | clip
    clip_in: 1.5s         # source offset; default 0s
    hold: 3s
text:                     # overlay blocks; each line reveals unit by unit
  - from: 0s              # block visible window [from, to]
    to: 7.2s
    fade_in: 0s           # default 0 (film.html fi .01 = instant)
    fade_out: 0.9s        # default 0.8s; block blurs by (1-a)*blur while fading
    blur: 10              # default 10
    lines:
      - {text: "a little film for", style: caps, y: 700, at: 1.2s, stagger: 0.12s}
      - {text: Arshiya, style: script, size: 230, y: 760, at: hum-0.3s, by: char, stagger: 0.09s, dur: 1.4s, rise: 10}
      - {text: "4", style: serif, gold: true, size: 420, weight: 300, line_height: 1, y: 480, at: chaar+0.2s}
      - {text: "35,064", style: serif, gold: true, size: 250, y: 640, at: br+5f, reveal: pop}
subs:                     # lyric subtitles, one fixed style (film.html #sub)
  - {from: 12.0s, to: 17.7s, text: "Jaipur ki dhoop mein, tera woh sharmaana"}
fx:
  - {hit: 0.7, at: ch1}                    # golden radial flash + optional punch (see Finish)
  - {burst: 70, at: ch1, pos: [540, 1100]} # sparkle burst, design px; seed = 100 + index among bursts
tracks:                   # slow global levels; keys are piecewise curves like wall camera keys
  dust:  [{t: 0s, v: .35}, {t: ch1-1s, v: .35}, {t: ch1, v: .85, ease: linear}]
  glow:  [{t: 0s, v: .25}, …]      # strength of the two drifting light leaks
  bars:  [{t: 0s, v: 150}, {t: ch1+0.4s, v: 0}]   # letterbox height, design px; ease default sine
```
Rules and fields:
- `hold` xor `until`. `until` must lie after the shot's start (error otherwise, hint the start time).
- `photo` | `wall` | `clip`: exactly one. `drift`, `frame`, `morph`, `tone`, `pulse` need `photo`.
- `drift.zoom` default `[1, 1]`; `drift.pan` default `[0, 0]` design px. `frame` and `zoom` (the M1 field)
  are mutually exclusive with `drift`'s framing rules below; `zoom` + `drift` is an error.
- `in` on shot 0: only `fade` (from `video.background`). Other transitions on shot 0 stay an error.
- `pulse`: `heartbeat` only (M2). Requires `video.bpm`.
- `tracks.*[].t` is a time expression; `v` a number; `ease` as camera keys plus the new `sine` (default `sine` for bars,
  `linear` for dust and glow). Before the first key: first value; after the last: last value.
- Text line fields: `text` (required), `style` ∈ `serif | script | caps | deva` (default `serif`),
  `italic` (bool; default false, `serif` only), `gold` (bool), `color` (`cream | gold | ink | #rrggbb`,
  default the style's), `size` (design px; default the style's), `weight`, `tracking` (em),
  `line_height`, `y` (design px top of the line box, required), `at` (time expression, required),
  `by` (`word` | `char`, default `word`), `stagger` (default `0.16s` word / `0.07s` char), `dur` (default
  `0.9s`), `rise` (design px, default 26), `reveal` (`rise` | `pop`, default `rise`),
  `sweep` (`[from, to]` time expressions; gold gradient travels 100%→0% across it; default the block window),
  `shadow` (bool; default true — false = no text shadow, film.html's ink lines).
- Every new field gets one line in `docs/SPEC-REFERENCE.md`.

## Text styles (montaj/render/text.css; film.html lines 24–35)
| style | font | size | other |
|---|---|---|---|
| `serif` | Cormorant Garamond 300 | 84 | line-height 1.1, letter-spacing .01em, shadow `0 4px 30px rgba(0,0,0,.65)`; `italic` → 300 italic |
| `script` | Great Vibes 400 | 84 | colour gold `#f0c77e`, shadow `0 0 40px rgba(240,199,126,.45), 0 4px 24px rgba(0,0,0,.6)` |
| `caps` | Cormorant 600 | 26 | uppercase, letter-spacing .42em, gold, shadow `0 2px 16px rgba(0,0,0,.8)` |
| `deva` | Tiro Devanagari Hindi 400 | 62 | line-height 1.5, shadow `0 4px 24px rgba(0,0,0,.8)` |
| subs | Cormorant 500 italic 44 | — | `rgba(251,241,220,.92)`, shadows `0 2px 18px rgba(0,0,0,.95), 0 0 4px rgba(0,0,0,.8)`, box left/right 60, bottom 190, centred |
Default colour cream `#fbf1dc`; `ink` = `#4a2c1a`. Lines are centred in a 1080-wide box (`.txt`).
Fonts ship in `montaj/fonts/` (OFL; copied from `claude-test/bday-video/fonts/`).

**Gold** (film.html 31–33): glyphs filled with `linear-gradient(100deg, #b87a35 0%, #f7dca0 30%, #fff6de 45%,
#e9b566 60%, #a86d2c 100%)`, background-size 300% × 100% of **each unit's own box**, background-position
`bp% 0` (offset_x = (w − 3w)·bp/100). bp lerps 100→0 across `sweep`. The line gets
`drop-shadow(0 6px 30px rgba(0,0,0,.55))` instead of a text shadow. The sweep is evaluated on the GPU from a
glyph-coverage mask, so it is continuous in t.

## Text raster + compositor (montaj/render/text.py)
```python
raster_line(line, *, scale: float, cache_dir: Path) -> RasterLine
#   Headless Chromium (doctor.chromium()), --headless=new, deviceScaleFactor=scale (= Canvas.k), transparent
#   background. Lays the whole line out once (so shaping/kerning match film.html), splits it into units
#   (words incl. trailing space, or chars), and captures per unit: `fill` premultiplied RGBA (or, for gold,
#   a coverage mask) and `shadow` premultiplied RGBA, each with its design-px box. Cached on disk under
#   cache_dir/<sha of html+css+font bytes+scale>/ as PNG + JSON; a hit spawns no browser.
RasterLine.units: list[Unit]        # Unit.box (x, y, w, h) design px, .fill (4,h,w), .shadow (4,h',w'), .shadow_box
draw_text(cv, img, blocks, t: float, rasters) -> Tensor   # composites active blocks at sub-frame t (frames)
```
Raster all lines of a film in one browser session (batch API `raster_lines(lines, …) -> list[RasterLine]`).
Animation (film.html `Words`, 166–189), per unit i of a line, with `a_i = at + i·stagger`:
`k = outCubic(clamp((t − a_i)/dur))` (film.html `eo`); opacity k; blur σ = (1−k)·12; translateY (1−k)·rise.
`reveal: pop` (film.html 289–291) is per line, not per unit: `pop = outCubic(clamp((t − at)·5/span))` where
span = the block's window; scale lerp(1.25, 1, pop) about the line centre, opacity pop, blur (1−pop)·14.
Block opacity `a = vis(t)` (film.html `vis`, 408–411: `min(fi>.02 ? sine((t−from)/fi) : 1,
fo>.02 ? sine((to−t)/fo) : 1)`, 0 outside [from, to]); the composited block is blurred σ = (1−a)·blur
then drawn with opacity a. Subs (film.html 431–432): opacity `min(sine((t−(from−.2s))/.5s),
sine(((to+.3s)−t)/.5s))`, at most one sub visible (the first whose window contains t).

## Shots (montaj/render/shots.py; film.html `fitCover`/`Photo`, 113–163)
```python
kenburns(cv, photo, fx, fy, z, px, py) -> (3,H,W)     # fitCover: s = max(W/w, H/h)·z (no 1.005 overscan),
                                                       # o = clamp(f·size·s − screen/2, 0, size·s − screen), image at −o + p
tone(img, brightness, saturate, contrast) -> img
frame_card(cv, photo, caption, rasters) -> Card       # card #f6efe2, pad 22, caption area 120, max 900x1260,
                                                       # top (H−ch)/2 − 40, box-shadow 0 40px 120px rgba(0,0,0,.7)
                                                       # + 1px rgba(0,0,0,.2); caption Great Vibes 64 #5a3b24 bottom 18
card_frame(cv, card, bg, z, tilt_deg, px, py) -> (3,H,W)   # translate(p) scale(z·.97) rotate(tilt) about card centre
                                                       # bg = kenburns(.5,.5,1.25) → blur 38, brightness .42, saturate 1.2
morph_mask(cv, x, center) -> (1,H,W)                   # r = inOutCubic(x)·1900 design px; 1 inside r−380, 0 at r
pulse_heartbeat(phase) -> scale                        # 1 + .035·e^(−18b) + (b>.22 ? .02·e^(−18(b−.22)) : 0)
css_blur(img, sigma_px) -> img                         # separable gaussian; downsample when σ > 8 device px
```
Drift progress: `k = clamp((t − v0)/(v1 − v0))`, `z = lerp(z0, z1, sine(k))`, pan `(pan·(k − .5))`.
`fx, fy` = focus relative to the crop (`(focus − crop0)/crop_size`), default .5. `[v0, v1)` is the shot's
**visible span**: from its start (or its whip start) to the end of the next shot's transition.
Morph reuses the shot's framing for the second photo; `x = (t − (start + at))/dur`; morph centre is a screen
fraction. Pulse scales the finished shot about the screen centre. Tone applies to the shot's photo only
(not the frame-card background, which has its own fixed filter).
Fade with `blur`: each side is blurred σ = (1 − its opacity)·blur while it fades.

## Clips (montaj/render/clip.py)
`ClipSource(path, fps)` decodes with ffmpeg (`-ss clip_in -i path -vf fps=<fps>,scale=…:force_original_aspect_ratio=increase,crop=W:H -f rawvideo rgb24`) into
CPU frames lazily per shot; frame for timeline `t` = `floor(t − shot.start)` clamped to the clip length.
Framing: cover, centred (focus/crop/drift not supported on clips in M2).

## Timeline (montaj/timeline.py) — additions
```python
Still(shot: int, span: tuple[int, int] | None = None)   # span = (v0, v1) for drift/pulse/morph shots, else None
Clip(shot: int, t0: int)                                 # t0 = shot start
Trans(kind, a, b, t0, dur, axis="x", center=None, blur: float = 0.0)
FramePlan(..., texts: tuple[int, ...] = (),             # text blocks whose [from, to] contains f (±1 frame for sub-frames)
               subs: tuple[int, ...] = (),
               hits: tuple[tuple[float, float], ...] = (),    # (at, k) with −3 ≤ f − at < 36 (1.2 s at 30 fps)
               bursts: tuple[tuple[int, float, float, float, int], ...] = (),  # (index, at, x, y, n) with 0 ≤ f−at ≤ 3.4 s
               dust: float = 0.0, glow: float = 0.0, bars: float = 0.0,  # track values at f, rounded to 1e-6
               fade_out: tuple[int, int] | None = None)            # (dur, start) when f ≥ start
Timeline.markers: dict[str, float]   # resolved, float frames
Timeline.texts / .subs               # resolved blocks (time fields as float frames)
```
New fields default so M1 plans are unchanged (M1 parity and M1 hashes must not move except via RENDER_REV).
`parse_time` errors and `until` rounding go through `SpecError` lines as in 001.

## Segment hash — additions (montaj/project.py)
Payload adds, only when present: the `model_dump` of every text block / sub referenced by the segment's
plans, `markers` (resolved), the sha of `montaj/render/text.css` and every file in `montaj/fonts/`, clip
file shas, and morph photo shas. Audio is not in the segment hash (muxed after concat).

## Finish — golden-film (montaj/render/golden.py; film.html 364–436)
`finish(cv, img, f, look, plan, overlay)` — `overlay(img) -> img` composites text + subs at the look's text
slot. warm-film: overlay runs last (M1 output unchanged when there is no text). golden-film order:
1. scene (shots, transitions, pulse)
2. leaks (373–429): two 900/1000 px discs, `radial-gradient(circle, rgba(255,140,60,.55·lk) → transparent 65%)`
   and `rgba(255,70,110,.35·lk)`, blur 60, screen. Positions from film.html 426–429 with t in seconds.
   `lk = clamp(glow, 0, 1)·(1 − clamp((t − (DUR−2s))/2s))`.
3. dust (365–397): 70 motes, rng = mulberry32(22) drawn in film.html order; bursts rng = mulberry32(100 + i).
   Each mote/particle is a radial gradient disc drawn additively ("lighter") into a transparent layer, then
   that layer goes source-over. Mote level `k = clamp(dust, 0, 1.2)·clamp(t/2s)`.
4. text + subs (overlay)
5. vignette (37): `radial-gradient(ellipse 75% 60% at 50% 48%, transparent 45%, rgba(0,0,0,.55) 85%, rgba(0,0,0,.85) 100%)`
6. grain (398–422): four 256² tiles, tile i = mulberry32(7+i) gray noise; frame uses tile `f % 4` offset
   `((f·97) % 256, (f·57) % 256)` design px, overlay blend, opacity .16
7. hits (423–424): golden radial `#fff8e6 → #f3c47a 60% → #b0662a` at (50%, 45%), screen, opacity
   `d<0 ? (d+.08s)/.08s·k : k·e^(−3.2·d)` with d in seconds, max over hits
8. bars (406, 430): black rects of height `bars` design px top and bottom
9. fade_out (435): multiply by `1 − clamp((f − start)/dur)`.
`mulberry32` = film.html line 74 exactly (32-bit integer math), shared by dust, bursts and grain.

## Audio (montaj/audio/, montaj/config.py)
- `~/.config/montaj/config.toml` (env `MONTAJ_CONFIG` overrides the path): `ace_step_dir`, `ace_python` +
  `ace_site_packages` (ACE-Step runs as `PYTHONPATH=<ace_site_packages> <ace_python>`, the dangling-venv
  trick), `whisper_python` (a python with `faster_whisper`). `montaj.config.load() -> Config`; missing keys → `ERR config: <key> missing (set it in
  ~/.config/montaj/config.toml)`. `doctor` reports each.
- `montaj music gen --caption TEXT --lyrics FILE [--bpm 76] [--key "D major"] [--lang hi] [--duration 165s] [--n 3]`
  → `music/cand-<seed>.wav` + `.json` (params + seed). Runs `montaj/audio/_ace_worker.py` under `ace_python`
  with offload_to_cpu + offload_dit_to_cpu. Result: one line per candidate, then `OK music gen <n> candidates`.
- `montaj music analyze WAV [--lang hi] [--prompt WORDS] [--bpm N]` → `music/markers.json`:
  `{duration, bpm, beats: [s…], words: [{i, w, s, e}], segments: [{s, e, text, first_word, last_word}], rms: [0.5 s hop]}`.
  Runs `_whisper_worker.py` under `whisper_python` (large-v3-turbo, cuda float16, word_timestamps,
  beam 5). Beats: bpm (given, else autocorrelation of an onset envelope over 60–180) on a grid whose phase
  maximises onset energy. Stdout: one line per segment `12.00-23.74 w1-w15 Jaipur ki dhup mein…`, then
  `OK music/markers.json <n> words <m> beats`.
- GPU lock: `render` and `music gen|analyze` hold an exclusive `fcntl` lock on `~/.cache/montaj/gpu.lock`;
  a second one waits (logs "waiting for GPU") — never runs both on 8 GB.
- Pipeline: when `audio.track` is set, `render` muxes after concat: `mux(video, track, out, loudnorm,
  fade_out)` (afade out over the last `fade_out`). The film's length wins (001 mux rule). Result line
  unchanged plus `audio` suffix.
- Encoder outputs carry bt709 tags (`-colorspace bt709 -color_primaries bt709 -color_trc bt709`).
- `check` already reports loudness and A/V offset; with a track, the A/V offset must be within 1 frame.

## Recipe and gate
`recipes/birthday-song-film.yaml` is the Arshiya port: markers = film.html `C`, subs = `LYR`, shots and text
per film.html 193–362, tracks from `dustK`/`leakK`/`barH`, hits = `HITS`, bursts = `BURSTS`, outro fade 1.2 s,
`audio: {track: music/song.wav, fade_out: 3s}`. Gate: a real render with the claude-test photos and song,
PSNR vs the Chromium frames at sampled times, `montaj check` clean, then Raja watches it.
