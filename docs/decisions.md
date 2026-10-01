# Decisions — Montaj
> Append-only log of load-bearing choices and WHY. Newest at the bottom.
> Format: `## YYYY-MM-DD — <decision>` then a short **Why:** line.

## 2026-10-01 — Name: Montaj, CLI `montaj`
**Why:** short to type and cheap in transcripts; means "montage" in Hindi/Urdu. Renaming later is costly.

## 2026-10-01 — Agents write a YAML spec, not engine code
**Why:** 50–150 spec lines replace 500–1,900 bespoke lines per video (5–10× fewer output tokens); validatable; humans edit it with zero tokens. Rejected: Python DSL (invites re-inventing).

## 2026-10-01 — One engine: Python + torch GPU compositor
**Why:** four runtimes across six videos caused the re-invention; torch path (`bday-video/v2`) is proven on the RTX 5050 + NVENC. Rejected: per-frame headless Chromium (CPU-only, slow, caused a ~7M-token GPU detour). Bespoke motion goes through a small `custom:` layer.

## 2026-10-01 — Text rasterised once per card via Chromium HTML/CSS
**Why:** keeps CSS typography quality (fonts, gradients, Devanagari shaping) without per-frame browser rendering; textures animate on GPU.

## 2026-10-01 — Segment-hash render cache
**Why:** a one-shot edit should re-render only that segment, making human iteration cheap in time and free in tokens.

## 2026-10-01 — Timing by beats and named markers
**Why:** cut-on-beat becomes the default; `markers.json` from whisper/beat analysis lets specs say `at: chorus1` instead of hand-computed seconds.

## 2026-10-01 — CLI, not MCP server
**Why:** works in Claude, Codex, opencode; no per-turn tool-schema tokens; composes in bash. Revisit only if a host cannot shell out.

## 2026-10-01 — Build the engine in waves by external models, scored in BENCHMARK.md
**Why:** Raja wants evidence on non-Claude models. Each wave runs head-to-head where cheap, is scored by driver-held hidden tests, and is reviewed by Sol. From wave 3, routing follows the benchmark (grok-4.6, grok-4.7, Space Bunny, Sol). Rejected: Opus-by-default implementation.

## 2026-10-01 — Cache granularity is one segment per shot
**Why:** segments split only at shot starts, so a shot edit re-renders that shot plus any transition segment that reads it (2/13 on the real film). Finer splits add concat joins without a measured need.

## 2026-10-01 — Concat writes per-chunk durations from known frame counts
**Why:** NVENC segments mux a short last packet; without explicit durations, every later frame drifts 0.3 ms and frame-exact tools misalign. Probing counts cost +4.9 s per render, so the pipeline passes them.
