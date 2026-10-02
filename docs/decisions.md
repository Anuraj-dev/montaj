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

## 2026-10-02 — M2 ports film.html as sequential shots
**Why:** film.html layers overlap freely; Montaj's timeline plays one shot at a time with transitions. The port keeps the schema simple and the cache per shot; parity still clears the gate (mean 32.9 dB). Rejected: a free layer stack (deferred to M5 `custom:` layers).

## 2026-10-02 — Duration-dependent effects live in FramePlan, not in the renderer
**Why:** the golden-film leak fades over the last 2 s. Hashing the film length re-rendered every segment on any hold edit; folding the ramp into `plan.glow` limits it to the tail segments.

## 2026-10-02 — Two-pass loudnorm on mux
**Why:** single-pass dynamic loudnorm left the Arshiya song at −12.6 LUFS (target −14). Two-pass linear lands −14.0 for one extra audio decode.

## 2026-10-02 — Stacked branches per milestone (m1 → m2 → m3 → m4)
**Why:** one milestone per PR keeps reviews readable; each builds on the previous engine. Raja decides when to push.

## 2026-10-02 — Harden M4 before M5
**Why:** two independent design studies found agent-contract leaks (silent Hindi default, late look errors, beat
holds ignoring analyzed beats) that every music session pays for, and the render-time target was never measured.
M5's `custom:` code would also widen the cache contract. Rejected: starting M5 now.

## 2026-10-02 — Claude models create, the CLI absorbs the tedium
**Why:** Raja wants Sonnet 5.5 / Opus 5.5 free to compose (including their own Python-synthesised music) at minimal
token cost; other models finishing a video is enough. Skill guidance stays advisory, not strict; new facts go into
`ERR` lines, `OK` lines and spec fields, not new bans.
