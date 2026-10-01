# Montaj — State
> Agent-native video harness: YAML spec in, finished video out, zero-token re-render/review · Last checkpoint: 2026-10-02

## 🚧 In progress / next
- Stacked branches, none pushed, no PRs: `feat/m1-engine` → `feat/m2-text-music` → `feat/m3-review-loop` →
  `feat/m4-skill-benchmark` (checked out). Each later branch contains the earlier ones.
- Raja's gate: watch the Vidyut and Arshiya finals, try the review page, decide on push/PRs.
- M4: fresh-agent benchmark results → `docs/BENCHMARK.md` § Skill benchmark; iterate `skills/make-video/SKILL.md`
  until the PLAN.md targets hold.
- M5 (later): motion-graphics `custom:` layers. Idea from claude-test `Sonnet 5.5/`: one beat clock shared by
  picture and synthesized sound, every frame a pure function of t.

## Status
- M1 engine (Vidyut parity, PSNR 49 dB vs film2). M2: text cards, subtitles, Ken Burns, frame cards, morph, clips,
  golden-film look, markers, music gen/analyze (real gen: 30 s song in 105 s), two-pass loudnorm. Arshiya port: PSNR vs film.html Chromium frames
  mean 32.9 dB / min 22.5 (40 frames); real 165 s final in 646 s; `OK check` (−14.0 LUFS, A/V +0.00 s).
- M3: `montaj review` (page on 127.0.0.1, review.json, `--summary`), `montaj watch`, `montaj taste`. Checked in
  real Chrome: seek, approve/reject, stamped comments, reload on re-render keeps the playhead.
- M4: `skills/make-video/SKILL.md`, linked into `~/.claude/skills` and `~/.codex/skills`; `~/.local/bin/montaj`
  links to the repo venv.
- Tests: `uv run pytest -q` 261 passed (GPU included).

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

## Key decisions (top 5; full log in decisions.md)
- Agents write a YAML spec, never engine code; every capability is a spec field.
- One engine: Python + torch GPU compositor; film2.py / film.html are parity oracles.
- Segment-hash cache: renderer reads only FramePlan + shots + assets, so one edit re-renders its segments.
- Engine built in waves by external models, scored in `docs/BENCHMARK.md`; Sol reviews.
- CLI, not MCP.

## Gotchas
- Model routing: grok-4.6 tight-spec + fix rounds, grok-4.7 render/integration, gpt-5.6-luna xhigh fast-tier on
  trial, Sol (gpt-6.1-sol medium) review + architecture. Space Bunny stalled once; LongCat/Ling/Nemotron dropped.
- No model names in commit messages (only BENCHMARK.md).
- Anything the renderer reads must reach the segment hash; prefer folding it into FramePlan (the leak tail did).
- NVENC rejects frames under ~145 px a side; tests use ≥ 320 px.
- Encodes are bt709 since M2; film2's final is bt601. mp4-vs-mp4 PSNR vs film2 is now 43.8 dB (was 49.2) because
  the two encodes no longer share YUV input; each sits 43–53 dB from the raw frames. Parity = raw-frame tests.
- 8 GB VRAM: render and music gen share `~/.cache/montaj/gpu.lock`; never run both.
- Raja's taste: photos never warp/drift unless the recipe asks; motion in transitions; cuts on the beat grid.
