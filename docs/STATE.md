# Montaj — State
> Agent-native video harness: YAML spec in, finished video out, zero-token re-render/review · Last checkpoint: 2026-10-01

## 🚧 In progress / next
- **M1 engine is built on branch `feat/m1-engine`** (not pushed, no PR). Gate left: Raja watches the real
  render of the Vidyut film (`montaj new … --recipe birthday-short` + `montaj render --final`).
- Wave 4 done (a94de05): check ignores held wall cameras, preview sharpen scales with k, intro lift is a
  fraction of the intro. `montaj check` on the real final: `OK check`.
- Then M2 per `docs/PLAN.md` § Milestones.

## Status
- CLI works end to end: `doctor new ingest validate render sheet check export`, `-C DIR`, `--json`.
- Real gate (2026-10-01): final 1080×1920 840f in 38.1s (target ≤180s), preview 9.6s, cached re-render
  0.5s, one-shot edit re-renders 2/13 segments in 3.0s, PSNR vs film2 `final.mp4` mean 49.2 dB / min 44.6.
- Tests: `uv run pytest -q` 104 passed (GPU included). Driver-held hidden tests live only in the session
  scratchpad; they are not in the repo.

## Architecture map
- Contract -> `docs/specs/001-m1-engine.md` (schema, FramePlan, segment hash, renderer/pipeline/CLI APIs)
- Spec fields -> `docs/SPEC-REFERENCE.md` · reference spec -> `recipes/birthday-short.yaml`
- spec/timeline/hash -> `montaj/spec.py`, `timeline.py`, `project.py`
- GPU renderer -> `montaj/render/` (`frame.py` Renderer, `core.py` draw with screen-window culling)
- pipeline/cache -> `montaj/pipeline.py` (`build/segments/<hash>.mp4`, concat → `out/<mode>.mp4`)
- IO -> `montaj/encode.py`, `ingest.py`, `qa.py`, `sheet.py`, `doctor.py` · CLI -> `montaj/cli.py`

## Stack & run
- Stack: Python 3.12 (uv) + torch 2.9.1 cu128 (RTX 5050, sm_120, 8 GB) + ffmpeg h264_nvenc
- Run: `uv run montaj <cmd>` · Test: `uv run pytest -q` (CPU only: `-m "not gpu"`)

## Key decisions (top 5; full log in decisions.md)
- Agents write a YAML spec, never engine code; every capability is a spec field.
- One engine: Python + torch GPU compositor; film2.py is the parity oracle (≤2/255).
- Segment-hash cache: renderer reads only FramePlan + shots + assets, so one edit re-renders its segments.
- Engine is built in reviewed waves by external models, scored in `docs/BENCHMARK.md`.
- CLI, not MCP.

## Gotchas
- Model routing from wave 3: grok-4.6 tight-spec fixes, grok-4.7 render/integration, Space Bunny free
  second entry, Sol review + driver cross-check. LongCat/Ling/Nemotron dropped.
- No model names in commit messages (only BENCHMARK.md).
- NVENC rejects frames under ~145 px a side; tests use ≥ 320 px.
- Frame-exact comparisons (ffmpeg psnr) need uniform timestamps; concat now writes per-chunk durations.
- 8 GB VRAM: never run music generation and rendering at the same time.
- Raja's taste: photos never warp/drift; motion only in transitions; cuts on the beat grid.
