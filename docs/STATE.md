# Montaj — State
> Agent-native video harness: YAML spec in, finished video out, zero-token re-render/review · Last checkpoint: 2026-10-21

## 🚧 In progress / next
- GitHub issues #1–19 all merged into `main` and pushed (2026-10-21); issues closed with their merge sha.
- Open worktrees with uncommitted, unmerged work (keep): `montaj-wt/launch-film` (`feat/engine-launch-film`),
  `montaj-wt/export-wa-audio` (`fix/export-whatsapp-audio-copy`). `montaj-wt/w10-grok47` is merged and clean;
  remove it with `git worktree remove` + `git branch -d wip/w10-grok47`. 13 unmerged `wip/w*` branches stay local.
- Backlog: lower-thirds are a title chip + subtitle (no bar/slide yet); wall prints at full res (downsample at build);
  hold edit re-renders later segments; M5 `custom` shot; BENCHMARK.md entry for the issue batch.

## Status
- M1–M4.5 and the issue batch pushed to origin/main. `wip/*` branches stay local only.
- M1 engine (Vidyut parity on raw frames). M2: text cards, subtitles, Ken Burns, frame cards, morph, clips,
  golden-film, markers, music gen/analyze, two-pass loudnorm. M3: `review` page, `watch`, `taste`. M4: skill
  `skills/make-video/SKILL.md` linked into `~/.claude/skills` and `~/.codex/skills`.
- M4.5 (waves 8–9, `docs/BENCHMARK.md`):
  - Contract: `validate` rejects unknown `look`/`intro`/`outro`, non-integer `fx.burst`; `subs[].size`/`color`.
  - Music: `montaj music compose music/compose.py` runs an agent-written numpy synth → wav, checks length/peak/LUFS;
    example `recipes/music/compose-example.py`. `music analyze --no-words` (instrumentals), `--lang`; per-command
    config; `OK` lines echo bpm, beat0, lang, duration.
  - Beat lock: `b` units snap to the grid `video.beat0 + k·60/bpm`; with `audio.markers` and no `video.bpm` the grid
    comes from the analysed song. Verified e2e: 12 cuts within 1 ms of detected beats.
  - Perf: golden-film PRNG vectorised (bit-identical). 30 s golden preview 104 → 30 s, final 117 → 40 s; 165 s
    Arshiya final 646 → ~225 s. Bounded renderer memory (per-Renderer LRUs, `begin_segment` lifetime, 96-frame clip
    budget, ≤ 2 live readers per clip source): 165 s peak VRAM 4.17 → 3.61 GB, frames identical.
  - `check` no longer flags a morph's eased ends as frozen; `sheet --at 4s` accepts seconds.
- Composed-music benchmark: Sonnet 5.5 0.27M input-eq / 19 turns / $0.65; Opus 5.5 0.31M / 23 turns / $1.27; both
  `OK check` with music they wrote. Films in `~/Anuraj-dev/montaj-projects/arshiya/{sonnet,opus}-5.5-composed/`.
- Tests (main at `1deaf74`): CPU 641 passed, GPU 48 passed (run GPU serially, 8 GB).

## Architecture map
- Contracts -> `docs/specs/001-m1-engine.md`, `002-m2-text-music.md`, `003-m3-review-loop.md`
- Spec fields -> `docs/SPEC-REFERENCE.md` · recipes -> `recipes/birthday-short.yaml`, `birthday-song-film.yaml`,
  `recipes/music/compose-example.py`
- spec/timeline/hash -> `montaj/spec.py` (`layout_shots`, beat grid), `timeline.py` (waveform levels,
  `mux_fade`), `project.py`
- GPU renderer -> `montaj/render/` (`frame.py` Renderer + LRUs; `text.py` Chromium rasters; `shots.py`; `golden.py`;
  `clip.py` forward readers)
- pipeline/cache -> `montaj/pipeline.py` · audio -> `montaj/audio/` (`mix.py` multitrack + `source_window`,
  `compose.py`, `analyze.py`), `gpulock.py`,
  `config.py`
- review loop -> `montaj/review/`, `watch.py`, `taste.py` · CLI -> `montaj/cli.py` · skill -> `skills/make-video/`

## Stack & run
- Stack: Python 3.12 (uv) + torch 2.9.1 cu128 (RTX 5050, 8 GB) + ffmpeg h264_nvenc + headless Chromium (text)
- Run: `montaj <cmd>` (or `uv run montaj`) · Test: `uv run pytest -q` (CPU only: `-m "not gpu"`)
- Machine config: `~/.config/montaj/config.toml` (ACE-Step + whisper pythons); taste: `~/.config/montaj/taste.md`
- Run outputs live in `~/Anuraj-dev/montaj-projects/` (outside the repo; never commit photos, renders, wavs).

## Key decisions (top 5; full log in decisions.md)
- Agents write a YAML spec, never engine code; every capability is a spec field. Music may be agent-written Python.
- One engine: Python + torch GPU compositor; film2.py / film.html are parity oracles.
- Segment-hash cache: renderer reads only FramePlan + shots + assets, so one edit re-renders its segments.
- Engine built in waves by external models, scored in `docs/BENCHMARK.md`; Sol reviews.
- Issue work: one worktree `~/Anuraj-dev/montaj-wt/g47-i<N>` per issue, Sol review, Sonnet fixes, `--no-ff` local merge.

## Gotchas
- Leave alone: the `launch-film` and `export-wa-audio` worktrees (uncommitted work). Workers never `git stash`.
- Waveform overlay bars follow the heard mix: post-edit/duck stems, the mix peak ceiling, and mux's `afade`
  (`timeline.mux_fade`, sample-exact vs ffmpeg). Loudnorm (one constant gain) is deliberately not mirrored.
- Hash compatibility: new optional fields use `Field(default=None, exclude_if=lambda v: v is None)` or
  `project._HASH_OMIT_NONE/_EMPTY`; existing films' segment hashes must not change.
- Model routing: grok-4.6 tight-spec logic, grok-4.7 render work, gpt-5.6-luna xhigh fast-tier implementer
  (CLI/audio), Sol (gpt-6.1-sol) medium review. Fix rounds resume the worker (`grok -r <id>`); a resumed grok
  session keeps its old cwd, so name the worktree path in the prompt and check the diff landed there.
- No model names in commit messages (only BENCHMARK.md).
- Anything the renderer reads must reach the segment hash; derived values (beat grid) are materialised onto
  `Timeline.spec` so the hash and runtime see the same thing.
- Whole marker map is in every segment hash: any marker edit re-renders the film.
- `music analyze` and `music gen` default to Hindi (`--lang hi`); gen to 165 s × 3 candidates.
- Arshiya project: its `assets/` symlinks point into the deleted `claude-test/bday-video`, and the original
  `song.wav` is gone (only the muxed audio in `arshiya/out/final.mp4` remains).
- Measure render time only with the GPU idle (another render, music gen or GPU tests skews it 30–40%).
- NVENC rejects frames under ~145 px a side; tests use ≥ 320 px.
- Encodes are bt709; parity = raw-frame tests (mp4-vs-mp4 vs film2 is 43.8 dB).
- 8 GB VRAM: render and music gen share `~/.cache/montaj/gpu.lock`; never run both.
- Raja's taste: photos never warp/drift unless the recipe asks; motion in transitions; cuts on the beat grid.
