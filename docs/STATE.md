# Montaj — State
> Agent-native video harness: YAML spec in, finished video out, zero-token re-render/review · Last checkpoint: 2026-10-02

## 🚧 In progress / next
- M4.5 hardening (from the 2026-10-02 study: Sol + grok-4.7 design studies, Luna transcript triage). In order:
  1. Agent-contract fixes: `validate` rejects unknown `look` / `intro` / `outro`; `fx.burst` int; hint fixes;
     `subs` size/colour; analyze/gen `OK` lines echo bpm, lang, duration, first sung word; skill `word:12` and
     analyze `--lang` lines.
  2. Composed music: an agent may write its own Python synth (`music/compose.py` → wav) and use it as the track.
  3. Beat lock: `b` units follow the analyzed beat grid, not a scalar bpm from frame 0.
  4. Clean render-time measurement (30 s preview/final, one-edit re-render) with an uncontended GPU.
  5. Analyze without ACE-Step config; bound renderer memory for 3–5 min films.
  Then M5 (narrowed): one `custom` shot, pure `fn(t)`, source bytes in the segment hash.
- Product goal (Raja, 2026-10-02): Claude Sonnet 5.5 / Opus 5.5 do the creative work; the CLI absorbs every tedious
  step so their token cost stays minimal. Other models finishing a video at all is fine. Keep the skill non-strict.

## Status
- M1–M4 merged to `main`, pushed to the public GitHub repo (origin/main). Model-output branches stay local only.
- M1 engine (Vidyut parity on raw frames). M2: text cards, subtitles, Ken Burns, frame cards, morph, clips,
  golden-film, markers, music gen/analyze, two-pass loudnorm. Arshiya port: PSNR mean 32.9 / min 22.5 dB.
- M3: `montaj review` page, `watch`, `taste` (checked in real Chrome).
- M4: `skills/make-video/SKILL.md` linked into `~/.claude/skills` and `~/.codex/skills`. Silent benchmark meets
  token/turn/context targets; music benchmark (grok-4.7, Sol, Luna) all `OK check` but 0.42–0.80M input-eq.
  Luna Hindi remake `OK check`; Raja found its song weak (lyrics/melody). Results: `docs/BENCHMARK.md`.
- Music-model research: nothing clearly better than ACE-Step 1.5 locally on 8 GB; 0.6B LM tried and deleted;
  Suno has no public API; Lyria is the paid hosted option; bring-your-own track already works.
- Tests: `uv run pytest -q -m "not gpu"` 222 passed (261 with GPU).

## Architecture map
- Contracts -> `docs/specs/001-m1-engine.md`, `002-m2-text-music.md`, `003-m3-review-loop.md`
- Spec fields -> `docs/SPEC-REFERENCE.md` · recipes -> `recipes/birthday-short.yaml`, `birthday-song-film.yaml`
- spec/timeline/hash -> `montaj/spec.py`, `timeline.py`, `project.py`
- GPU renderer -> `montaj/render/` (`frame.py` Renderer; `text.py` Chromium rasters; `shots.py`; `golden.py`; `clip.py`)
- pipeline/cache -> `montaj/pipeline.py` · audio -> `montaj/audio/`, `gpulock.py`, `config.py`
- review loop -> `montaj/review/`, `watch.py`, `taste.py` · CLI -> `montaj/cli.py` · skill -> `skills/make-video/`

## Stack & run
- Stack: Python 3.12 (uv) + torch 2.9.1 cu128 (RTX 5050, 8 GB) + ffmpeg h264_nvenc + headless Chromium (text)
- Run: `montaj <cmd>` (or `uv run montaj`) · Test: `uv run pytest -q` (CPU only: `-m "not gpu"`)
- Machine config: `~/.config/montaj/config.toml` (ACE-Step + whisper pythons); taste: `~/.config/montaj/taste.md`
- Run outputs live in `~/Anuraj-dev/montaj-projects/` (outside the repo; never commit photos, renders, wavs).

## Key decisions (top 5; full log in decisions.md)
- Agents write a YAML spec, never engine code; every capability is a spec field.
- One engine: Python + torch GPU compositor; film2.py / film.html are parity oracles.
- Segment-hash cache: renderer reads only FramePlan + shots + assets, so one edit re-renders its segments.
- Engine built in waves by external models, scored in `docs/BENCHMARK.md`; Sol reviews.
- Commit straight to `main` (Raja's call: no CI); one concern per commit.

## Gotchas
- Model routing: grok-4.6 tight-spec + fix rounds, grok-4.7 render/integration, gpt-5.6-luna xhigh fast-tier
  implementer on trial, Sol (gpt-6.1-sol) review + architecture. LongCat/Ling/Nemotron dropped.
- No model names in commit messages (only BENCHMARK.md).
- Anything the renderer reads must reach the segment hash; prefer folding it into FramePlan.
- `b` durations use scalar `video.bpm` from frame 0; analyzed beats are not used yet (beat-lock item above).
- Whole marker map is in every segment hash: any marker edit re-renders the film.
- `music analyze` and `music gen` default to Hindi (`--lang hi`) and gen to 165 s × 3 candidates.
- Music-benchmark render times were taken with 3 agents sharing the GPU lock: not a valid perf measurement.
- NVENC rejects frames under ~145 px a side; tests use ≥ 320 px.
- Encodes are bt709; parity = raw-frame tests (mp4-vs-mp4 vs film2 is 43.8 dB).
- 8 GB VRAM: render and music gen share `~/.cache/montaj/gpu.lock`; never run both.
- Raja's taste: photos never warp/drift unless the recipe asks; motion in transitions; cuts on the beat grid.
