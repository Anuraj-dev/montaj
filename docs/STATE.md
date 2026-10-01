# Montaj — State
> Agent-native video harness: YAML spec in, finished video out, zero-token re-render/review · Last checkpoint: 2026-10-01

## 🚧 In progress / next
- M0 done (plan + context + token audit). Next: **M1 engine + montage** — see `docs/PLAN.md` § Milestones.
- First concrete step: `uv init` the package, port camera/draw/transitions/finish from
  `~/Anuraj-dev/claude-test/bday-video/v2/film.py` + `film2.py` into `montaj/render/`, then write
  `recipes/birthday-short.yaml` reproducing the Vidyut film.

## Status
- Planned only. No product code yet.
- `tools/token-report.py` works: reproduces the baseline audit (38.23M cache-read, 7.56M input-eq over 7 sessions).

## Architecture map (planned — see PLAN.md § Architecture)
- CLI -> `montaj/cli.py` · spec -> `montaj/spec.py` · renderer -> `montaj/render/`
- Audio -> `montaj/audio/` · review page -> `montaj/review/` · agent skill -> `skills/make-video/SKILL.md`
- Prior code to port -> `docs/research/prior-art.md`

## Stack & run
- Stack: Python (uv) + torch CUDA (RTX 5050, sm_120) + ffmpeg NVENC + headless Chromium (text raster only)
- Run: TBD in M1 · Test: TBD (pytest)

## Key decisions (top 5; full log in decisions.md)
- Agents write a YAML spec, never engine code.
- One engine: Python + torch GPU compositor.
- Text = HTML/CSS rasterised once per card by Chromium, animated on GPU.
- Segment-hash cache -> only changed segments re-render.
- CLI, not MCP.

## Gotchas
- Headless Chromium cannot GPU-raster; per-frame Chromium render is CPU-bound. Don't investigate again.
- ACE-Step venv python symlink is dangling; use `claude-test/bday-video/music/py.sh` trick.
- 8 GB VRAM: never run music generation and rendering at the same time.
- Raja's taste: photos never warp/drift; motion only in transitions; cuts on the beat grid.
