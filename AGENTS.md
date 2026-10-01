# Agent entry — Montaj

Montaj is a CLI + skill that lets an AI agent produce a finished video from a small YAML spec, so creative
intent costs few tokens and every re-render / review costs none.

## Start here
1. `docs/STATE.md` — what exists, what's next, gotchas.
2. `docs/INDEX.md` — map of every doc. Read only what your task needs; the docs exist so you never scan the repo.
3. Engine work: `docs/specs/001-m1-engine.md` is the contract (schema, timeline, renderer, IO). Code against it.

## Map
| Path | Holds |
|---|---|
| `montaj/spec.py` · `timeline.py` · `project.py` | YAML schema → frame plans → segments + cache hashes |
| `montaj/render/` | GPU compositor: canvas, camera/draw, transitions, wall, finish (ports of `film.py`/`film2.py`) |
| `montaj/encode.py` · `ingest.py` · `qa.py` · `sheet.py` · `doctor.py` | ffmpeg/NVENC, asset prep, QA, contact sheets, env check |
| `montaj/cli.py` | the only entry point: `montaj <cmd>` |
| `recipes/*.yaml` | reference specs; `birthday-short.yaml` is the M1 parity target |
| `tests/` | pytest; `@pytest.mark.gpu` for CUDA-only tests |

## Commands
- Install: `uv sync`
- Fast tests (CPU): `uv run pytest -q -m "not gpu"` · Everything: `uv run pytest -q`
- CLI: `uv run montaj <cmd>`

## Developing Montaj
- The spec is the product API: every new capability is a spec field documented in `docs/SPEC-REFERENCE.md`
  (one line each), not a new script.
- Commands are quiet: progress goes to `build/*.log`; stdout ends with one result line
  (`OK <path> …` / `ERR <what>: <hint>`); `--json` for machines.
- The renderer reads only a `FramePlan`, the shots it names, and asset files — the segment cache is correct
  only while that holds. Pass state as arguments; module-level mutable globals are a cache bug.
- Units: photo coordinates in source px; screen values in design px (1080 wide), scaled by `Canvas.k`.
- Parity oracle: `~/Anuraj-dev/claude-test/bday-video/v2/film2.py` (photos in `v2/src/`). Port behaviour
  exactly; improve only after a parity test pins the original.
- Known traps live in `montaj doctor` output and `docs/research/prior-art.md`; add new traps there.

## Dispatched workers
When your prompt names owned files, you are a worker:
- Edit only the files you own. Other work may be in progress in the tree; leave it untouched.
- The driver owns git and the shared docs (`docs/STATE.md`, `docs/sessions/`, `docs/BENCHMARK.md`,
  `docs/decisions.md`). Leave commits and those files to the driver.
- Done = your new tests pass and `uv run pytest -q -m "not gpu"` exits 0 (plus `-m gpu` if you touched
  `render/`). End with: files changed, each command run with its exit code, contract gaps you hit.
- A gap or contradiction in `docs/specs/` → follow the spec's intent, note it in your final message.

## Making a video with Montaj (token budget)
- Take a photo/asset **folder path**. Never ask the user to paste images into the chat.
- Look only at `montaj ingest` / `montaj sheet` images; at most 2 images per iteration.
- In a creative session never read engine source. On an engine bug: report the command + one-line error, stop.
- Run long commands (`render`, `music gen`) in the background and wait for the notification. No polling.
- Ask one brief round first (length, aspect, text, motion style, music). Read and update `~/.config/montaj/taste.md`.
- Start a fresh session between phases (engine dev vs. making a video); `/checkpoint` before `/clear`.

Conventions: `docs/conventions.md` · Decisions: `docs/decisions.md` · Model results: `docs/BENCHMARK.md`
