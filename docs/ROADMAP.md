# Roadmap — agent-first editor
> Written 2026-11-10. Tracker: GitHub #49. This plan replaces the single "M5 Motion-graphics" line in PLAN.md.

## Goal
An agent writes a small YAML spec and gets a film with After Effects-level motion, Premiere-level editing and
Resolve-level colour. Every pixel is produced on the discrete GPU. A film costs few tokens, and more of those
tokens go to creative choices than to boilerplate.

## Principles
- **The spec is the API.** A new capability is a spec field with one line in `SPEC-REFERENCE.md`.
- **Open vocabulary.** Where a field cannot exist yet, the agent writes a small GPU function (`custom:`, #37).
  The engine author's imagination must not be the ceiling.
- **Numbers before images.** A question the engine can answer (`inspect`, `check`, `scopes`) is answered as text.
  An image is for taste only: one gallery, one strip per new kind of motion.
- **Good defaults carry taste.** Presets and normalisation (captions without full stops, stroke instead of a box)
  make the default output good. Facts reach the agent through `OK` / `WARN` / `ERR` lines, not skill bans
  (decision 2026-10-02).
- **dGPU only for pixels.** Decode, composite, text and encode run on `cuda:0`. In strict mode a CPU pixel path
  is an `ERR`. Shaping, demux/mux, audio and Python orchestration stay on the CPU.
- **Cache contract holds.** Sugar (captions, cuts, emphasis, styles, `focus: face`) expands at layout time into
  values on the timeline; the renderer and the segment hash see only the expansion.
- **Existing films do not change.** New fields are omitted from the hash when unset; an existing style moves to a
  new engine only after a raw-frame parity test.

## Where Montaj stands (2026-11-10)
| Area | Today | Gap |
|---|---|---|
| Type | 6 film-title styles, cream/gold/ink, `rise`/`pop`/`type`, Chromium raster on CPU | No stroke, display fonts, per-word colour or motion; text is the largest CPU pixel path |
| Captions | `subs`: one style, hand-timed lines | Not built from word timings; no presets; nothing stops boxes or full stops |
| GPU | Compositor on CUDA, NVENC encode | Software clip decode; rgb24 pipe + CPU colour conversion; silent `libx264` fallback |
| Feedback | Still contact sheets; `check` for black, frozen, loudness | Agent cannot see motion or measure layout, pacing or colour |
| Compositing | One source per shot; keyframes on camera and 3 tracks | No layers, blend modes, masks, mattes, general keyframes or code-drawn shots |
| Editing | Beat grid, markers, ranges, speed, SFX clips, ducking | No transcript cuts, linked clip audio, J/L cuts or word-anchored emphasis |
| Colour | 4 looks or a film-wide LUT; CSS `tone` on photos | No per-shot grade, linear light, secondaries, scopes or 10-bit |
| Skill | Conservative; reads the whole spec reference | No topic-scoped reference; recipes cover birthday and launch films only |

## Milestones
Each ends with a recipe that uses it, a fresh-agent benchmark run in `BENCHMARK.md`, and `/checkpoint`.

### M5 Type and captions — the visible leap
| # | Issue | Depends on |
|---|---|---|
| 20 | GPU glyph rendering (HarfBuzz + distance-field atlas) | — |
| 21 | Stroke, shadow, glow, display fonts, x position, rotation, per-word colour | 20 |
| 22 | Per-unit pop, bounce, shake, slide and exits | 20 |
| 23 | `captions` block from word timings, with presets | 21, 22 |
| 24 | Caption hygiene: no full stops, no static boxes, contrast and safe zones | 23 |
| 25 | Text fit: auto-size, wrap, overflow as an `ERR` | 20 |
| 26 | Caption gallery and reference study | 23 |

Exit: a 30 s voice clip is captioned in under 10 lines of YAML with the `punch` preset, no Chromium process
runs, and `check` prints no caption `WARN`. Guide: `docs/CAPTIONS.md`.

### M6 All-dGPU pipeline
| # | Issue | Depends on |
|---|---|---|
| 28 | GPU colour conversion; NVENC without the rgb24 pipe (step 1: NV12, step 2: zero copy) | — |
| 27 | NVDEC clip decode into CUDA tensors | — |
| 29 | Strict dGPU mode and `doctor` stage report | 20, 27, 28 |

Exit: with `[gpu] strict = true` a clip + text + photo film prints `gpu=all`; breaking NVENC gives an `ERR`.
#28 step 1 is small and independent: do it first. M6 can run in parallel with M5.

### M7 Agent eyes
| # | Issue | Depends on |
|---|---|---|
| 30 | `montaj inspect`: on-screen elements as JSON | — |
| 31 | Motion strips for transitions and animation | — |
| 32 | Semantic ingest: faces, saliency, cuts, transcripts; `focus: face` | — |
| 33 | `check`: pacing and retention metrics | — |

Exit: a recipe frames every portrait with `focus: face`, and an agent fixes an off-frame caption from
`inspect --problems` without opening an image. #30 and #31 are cheap and help every later milestone; start them
alongside M5.

### M8 Compositing and motion — After Effects level
| # | Issue | Depends on |
|---|---|---|
| 34 | Layers: several sources per shot, transform, opacity, blend modes | — |
| 35 | Keyframes on any property; spring, overshoot, bezier, expressions | 34 |
| 37 | `custom:` code-drawn GPU shots, effects and transitions | — |
| 38 | Layer effects: shake, blur, bloom, chroma, trails | 34 |
| 36 | Subject mattes: cutouts, text behind the subject | 34, 32 |

Exit: benchmark run 9 (black-hole explainer) re-renders with `custom:` shots and no CPU pre-render (was
7 min 14 s), and a recipe shows a b-roll inset, a sticker and a word behind the speaker.

### M9 Editing — Premiere level
| # | Issue | Depends on |
|---|---|---|
| 40 | Linked clip audio with J and L cuts | — |
| 39 | Talking-head cuts from the transcript | 32, 40 |
| 41 | Word-anchored emphasis and automatic transition SFX | 35, 38 |
| 42 | Styles, defaults and components | — |

Exit: a 90 s raw talking clip becomes a tight, captioned, sound-designed short from a spec under 60 lines.
#42 is independent and saves tokens at once; it can land any time after M5.

### M10 Colour — Resolve level
| # | Issue | Depends on |
|---|---|---|
| 43 | Linear-light working space and per-shot primary grade | — |
| 45 | Scopes as numbers; shot matching | 43, 32 |
| 44 | Secondaries, power windows, film emulation | 43, 36 |
| 46 | 10-bit HEVC / AV1 through NVENC | 28 |

Exit: a mixed-camera montage is matched with `montaj match`, graded per shot with one tracked face window, and
exported 10-bit with no banding in a test gradient.

### M11 Skill, recipes, benchmark
| # | Issue | Depends on |
|---|---|---|
| 47 | Skill v4: creative latitude, `montaj spec <topic>`, teaching errors | grows with each milestone |
| 48 | Recipes: talking-short, vlog-montage, explainer, gaming-highlight | one per enabling milestone |

Not a final phase: #47 and #48 advance at the end of every milestone.

## Order of work
1. #28 step 1, #30, #31 — small, independent, immediate payoff.
2. M5 in order #20 → #21, #22, #25 → #23 → #24, #26.
3. Rest of M6 (#27, #28 step 2, #29).
4. #32, #33, #42.
5. M8: #34 → #35, #38; #37 in parallel; #36 last.
6. M9, then M10.

Dispatch: one worktree per issue, `--no-ff` local merge, one results file per dispatch kept outside `/tmp`.

## Routing (set 2026-11-10; Claude models only)
No per-issue review. Each worker verifies its own work: new tests for the issue's contract, `uv run pytest -q`
(GPU suite when it touches `render/` or encode), and a real render when the issue changes pixels. The driver
re-runs those checks, holds the real-GPU gate (strict mode, NVDEC/NVENC, VRAM) and visual approval, then merges.

| # | Issue | Implementer |
|---|---|---|
| 20 | GPU text engine | opus-high |
| 21 | Text styling | opus-medium |
| 22 | Text animation | opus-medium |
| 23 | `captions` block | sonnet-medium |
| 24 | Caption hygiene lints | sonnet-medium |
| 25 | Text fit | sonnet-medium |
| 26 | Caption gallery, `ref` | sonnet-medium |
| 28 | NVENC: step 1 NV12 / step 2 zero copy | opus-medium / opus-high |
| 27 | NVDEC decode | opus-medium |
| 29 | Strict dGPU mode | sonnet-medium |
| 30 | `inspect` | sonnet-medium |
| 31 | Motion strips | sonnet-medium |
| 32 | Semantic ingest | opus-medium |
| 33 | Pacing metrics | sonnet-medium |
| 34 | Layers | opus-high |
| 35 | Keyframes, easing, expressions | sonnet-medium |
| 36 | Subject mattes | opus-medium |
| 37 | `custom:` shots | opus-high |
| 38 | Layer effects | opus-medium |
| 39 | Transcript cuts | sonnet-medium |
| 40 | Linked clip audio, J/L cuts | opus-medium |
| 41 | Word-anchored emphasis, auto SFX | sonnet-medium |
| 42 | Styles, defaults, components | sonnet-medium |
| 43 | Linear light + primary grade | opus-high |
| 44 | Secondaries, film emulation | opus-medium |
| 45 | Scopes, shot matching | opus-medium |
| 46 | 10-bit HEVC / AV1 | opus-medium |
| 47 | Skill v4 + `montaj spec` | driver (skill text), sonnet-medium (CLI) |
| 48 | Recipes | opus-medium, one per recipe; driver watches the render |

Review (once, at the end):
- Fresh reviewers in parallel over the full M5–M11 diff, split by area: `fable-high` for the foundations
  (#20, #28, #34, #37, #43), `opus-high` for the rest. Diffs inlined.
- One fix round for all findings, then the fresh-agent benchmark as the product test.
- Escalation: a worker whose result fails the driver's checks goes up one step (sonnet-medium → opus-medium →
  opus-high → fable-high); never a third patch by the same route.

## Targets
| Metric | Today | Target after M11 |
|---|---|---|
| Captioned 30 s talking short, input-equivalent tokens | not possible from a spec | ≤ 0.3M |
| Spec lines for that short | — | ≤ 60 |
| Images viewed per film | ≤ 2 per iteration | ≤ 4 in total |
| Spec reference read per film | all 270 lines | topics used only |
| CPU pixel paths in a final render | 4 (text, decode, colour conversion, fallback encode) | 0 in strict mode |
| Final render, 30 s 1080×1920, RTX 5050 | ~40 s | ≤ 40 s with layers and captions |
| Human quality score (benchmark, 1–5) | not recorded | recorded per run; ≥ 4 on the talking-short recipe |

"Today" figures are from STATE.md and BENCHMARK.md; the targets are goals, not measurements.

## Risks
- **8 GB VRAM.** NVDEC sessions, matte and face models, the glyph atlas and 16-bit float frames all compete.
  Measure peak VRAM per issue; analysis models take the GPU lock like music gen.
- **Text parity.** Distance fields round sharp corners at large sizes. Existing styles keep Chromium until a
  parity test passes (#20).
- **`custom:` runs code.** A spec with `custom:` has the trust of its author, like `music compose`. It widens the
  cache contract to file bytes + params + helper version (#37).
- **Library fit.** GPU decode/encode libraries must match the pinned torch 2.9.1 cu128 / sm_120 build; verify
  before choosing (#27, #28).
- **Spec growth vs tokens.** Every milestone adds fields. #42 (reuse) and #47 (`montaj spec <topic>`) are what
  keep the per-film read cost flat; do not defer them.
- **Scope.** Parity with three professional tools is a direction. Each milestone must ship a recipe that a fresh
  agent can use, or it is not done.
