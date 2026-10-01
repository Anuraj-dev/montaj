# Model benchmark — Montaj build
> Real tasks from this repo, scored by the driver (Claude Opus 5.5) against tests and review findings.
> Append a row per dispatch. Scores are evidence from this repo only, not general rankings.

## Models in play
| Route | CLI | Cost to Raja |
|---|---|---|
| grok-4.7, grok-4.6, grok-4.5 | `grok -p -m <id>` | grok quota |
| gpt-6.1-sol (review), gpt-6-luna, gpt-5.6-luna (xhigh, fast tier) | `codex-run.sh` (`-c service_tier="fast"`) | codex quota |
| space-bunny-free | `opencode run -m opencode/space-bunny-free` | free (until 2026-10-05) |
| muse-spark-1.3-contributor-free | `opencode run -m opencode/muse-spark-1.3-contributor-free` | free |
| longcat-2.5-preview, ling-3.0-flash-fin, nemotron-3.5-lightning | `opencode run -m opencode/<id>-free` | free |
| Claude sonnet/opus medium/high | Agent tool (`sonnet-medium`, `sonnet-high`, `opus-medium`, `opus-high`) | Claude plan |

## Scoring
- **Hidden** = driver-held acceptance tests the worker never saw: passed / total.
- **Own tests** = worker's suite exit status. **Review** = findings from the reviewer (H high / M med / L low).
- **Driver fix** = lines the driver changed before merge (0 = merged as-is). **Wall** = wall-clock minutes.

## Implementation runs
| Wave | Task | Model | Wall | Hidden | Own tests | Review | Driver fix | Merged | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | W1a spec/timeline/hash | grok-4.6 | 11.5m ($0.31, 0.83M tok) | 20/20 | 25 pass | Sol 5M | 0 (fix round by same model: 4m, $0.19) | yes | Fixes clean, hidden stayed 20/20; terse, correct code |
| 1 | W1a spec/timeline/hash | space-bunny | 40.5m (free, 9.0M tok) | 19/20 | 76 pass | — | — | no | Wall t0 off by fade length (spec gap); added crop-bounds check vs real image size |
| 1 | W1b GPU renderer port | muse-spark-1.3 | 15.3m (free, 1.4M tok) | parity 8/8 | 8 pass | Sol 3M (tests only) | 0 | yes | Smallest diff (587 lines), scaled focal/bloom by k unprompted; fix-round resume hung 12m, fresh retry hit 503 overload |
| 1 | W1b GPU renderer port | grok-4.7 | 16.9m ($0.36, 0.96M tok) | parity 8/8 | 8 pass | Sol 3M + 1L (impl) | — | no | Focal length and sharpen not scaled for preview; 760 lines |
| 1 | W1b fix round (on muse code) | space-bunny | 8.8m (free) | parity 10/10 | 11 pass | — | 0 | yes | Wall end-to-end via recipe camera bit-exact; CPU-only import fixed |
| 1 | W1c encode/ingest/qa/sheet/doctor | space-bunny | 19.6m (free, 5.8M tok) | n/a | 21 pass | Sol 3H 4M 1L | 0 (fix round: 10.3m) | yes | Ran real NVENC smoke unprompted; fix round reverted each fix to prove its test fails |
| 2 | W2 frame renderer + pipeline + CLI | grok-4.7 | 26.2m ($0.96, 3.9M tok) | 4/4 | 85 pass | Sol 1H 3M 1L (+1H Sol missed, found by cross-check) | 0 (fix round by same model: 11.2m, $0.39, 90 pass) | yes | 1,034 lines; proved each fix's test failed first; real preview 26.5s |
| 2 | W2 frame renderer + pipeline + CLI | space-bunny | 19.9m (free, 7.4M tok) | 4/4 | 119 pass | Sol 1H 6M 5L | — | no | 1,330 lines; 8/9 parity frames bit-exact; flagged 6 real contract gaps; 3 tests order-dependent |
| 2 | W2 frame renderer + pipeline + CLI | longcat-2.5-preview | 15.6m + 21.8m resume (free, 3.7M tok) | 4/4 | 83 pass | Sol 2H 3M 2L | — | no | First run hit output-length limit after 11 read-only steps; after one resume, leanest diff (717 lines); preview swirl centre double-scaled by k |
| 2 | W2 frame renderer + pipeline + CLI | ling-3.0-flash-fin | 1.8m | — | — | — | — | no | Provider "Endpoint is unavailable" (400) on every call, incl. one-word smoke tests: no data |
| 2 | W2 frame renderer + pipeline + CLI | nemotron-3.5-lightning | ~50m, stopped | — | — | — | — | no | Still unfinished after the others were merged; stopped and unscored |
| 3 | W3a wall perf (parity-pinned) | grok-4.7 | 19.8m ($0.62, 2.3M tok) | 2/2 (wall 0.055s, 9 frames bit-exact) | 92 pass | Sol 0H 3M 0L (2 latent, 1 out of domain) | 2 lines (clone bg, 2-D light) | yes | 0.56s → 0.055s per wall frame; final render 236s → 37.6s; 122 lines |
| 3 | W3a wall perf (parity-pinned) | space-bunny | ~45m, killed by a driver crash | 2/2 (wall 0.056s) | 90 pass | — | — | no | Same speedup in 83 lines; died while verifying all 840 frames; required perf test not yet written |
| 3 | W3b concat slip, check freezes, int crops | grok-4.6 | 9.0m ($0.32, 1.2M tok) | 2/2 | 94 pass | driver: 1 perf regression | 0 (fix round by same model: 3.0m, $0.23) | yes | First fix decoded every segment to count frames (+4.9s per render); fix round passes known counts |
| 4 | W4 wall-hold freezes, k-scaled sharpen, intro lift | grok-4.6 | 14.3m ($0.54, 2.1M tok) | 7/7 (regression) + real `check` OK | 104 pass | driver inline (small diff) | 1 line (RENDER_REV bump the worker flagged but did not own) | yes | Flagged the stale-preview-cache risk itself; parity at k=1 unchanged |
| 5 | W5a M2 schema + timeline (markers, until, text, subs, fx, tracks) | grok-4.6 | 23.4m ($1.53, 7.4M tok) | 7/7 | 124 pass | Sol 4H 4M (7 real, 1 allowed by prompt) | 0 (fix round by same model: 5.6m, $0.33) | yes | Two of Sol's findings were integration gaps carried into W6 |
| 5 | W5b Chromium text cards + GPU compositor | grok-4.7 | 67.3m ($1.78) | fidelity vs Chromium < 2/255 | 10 tests | Sol 0H 6M | 0 (fix round: 12.6m, $0.66) | yes | Calibrated against Chromium before coding; captured the gold shadow from Chromium |
| 5 | W5c Ken Burns, frame cards, morph, clips, golden-film layers | grok-4.7 | 60.1m ($2.45) | per-layer PSNR vs Chromium 36–99 dB | pass | Sol 1H 5M | 0 (fix round: 18.6m, $0.47) | yes | HIGH: clips decoded the whole tail as float (~123 GB); fixed to a bounded uint8 window |
| 5 | W5d music gen/analyze, config, GPU lock | space-bunny | 26.9m (free) | real analyze: 164/164 words, 0 ms median | 128 pass | Sol 4H 4M 2L (3 false: driver's diff base) | 0 (fixes by grok-4.6: 8.7m, $0.31) | yes | Driver's review script diffed against a moved base; fixed to use merge-base |
| 5 | W6r Arshiya recipe (55 shots) | grok-4.6 | 13.3m ($0.58) | `validate` OK 4950f | 4 tests | driver | 0 | yes | Flagged two schema gaps (blank shot, indent) that W6 then added |
| 6 | W6 M2 integration: renderer, finish, pipeline, CLI, real render | grok-4.7 | 46.3m ($2.68, 90 turns) | PSNR vs film.html mean 32.9 dB, min 22.5 (gate 30/22) | 222 pass | Sol 2H 6M 1L | ~20 lines (leak tail moved into FramePlan so a hold edit no longer re-renders every segment) | yes | Exited mid-mux with no final report. Fix round (17m, $1.44): all fixed, plus 2-pass loudnorm and intended-black in `check`. Real final 165 s in 646 s, `OK check` |
| 7 | W7 M3 review page, watch, taste | space-bunny | 28m, killed | — | — | — | — | no | opencode stalled at bootstrap with zero model calls |
| 7 | W7 M3 review page, watch, taste | grok-4.6 | 13.3m ($0.46) | 7/7 | 181 pass | Sol 0H 9M 2L | 0 (fix round: 11.2m, $0.53, 14/14 fixed) | yes | Real Chrome: seek, reject, stamped comment, reload on re-render keeps the playhead |
| 7 | W7 M3 review page, watch, taste | gpt-5.6-luna xhigh, fast tier | 9.3m (3.06M in, 2.96M cached, 32k out) | 7/7 | 175 pass | Sol 0H 8M 1L | — | no | Leanest (882 vs 1,394 lines) but 4 tests vs 10; page swallowed a re-render reload. First Luna implementation run |

## Review runs
| Wave | Diff | Reviewer | Wall | Findings (H/M/L) | Confirmed real | False positives | Notes |
|---|---|---|---|---|---|---|---|
| 1 | W1a grok-4.6 | gpt-6.1-sol high | 4.2m | 0/5/0 | 5 | 0 | Overlapping transitions, unnormalised wall key times, NaN bpm |
| 1 | W1a grok-4.6 | muse-spark-1.3 | 5.1m | 0/0/0 | — | — | Missed all 5 Sol findings: weak reviewer |
| 1 | W1b muse-spark | gpt-6.1-sol high | 2.9m | 0/3/0 | 3 | 0 | Test quality only; missed global-RNG + sharpen-k issues it flagged on grok-4.7's diff |
| 1 | W1b grok-4.7 | gpt-6.1-sol high | 4.5m | 0/3/1 | 4 | 0 | Global RNG, focal and sharpen not scaled by k |
| 1 | W1c space-bunny | gpt-6.1-sol high | 3.6m | 3/4/1 | 8 | 0 | mux re-encode + truncation, QA swallowing ffmpeg failures |
| 2 | W2 grok-4.7 | gpt-6.1-sol high | 4.3m | 1/3/1 | 5 | 0 | Missed the `new` overwrite bug it flagged on both sibling diffs |
| 2 | W2 space-bunny | gpt-6.1-sol high | 5.7m | 1/6/5 | 12 | 0 | Missed argparse `--json` exit-2 it flagged on grok's diff |
| 2 | W2 longcat | gpt-6.1-sol high | ~5m | 2/3/2 | 7 | 0 | Caught a preview-only math bug full-res parity cannot see |
| 3 | W3a grok-4.7 | gpt-6.1-sol high | 4.4m | 0/3/0 | 2 (latent) | 1 (1e9 world coords) | Reproduced each on CPU; none affects current scenes |
| 5 | W5a grok-4.6 | gpt-6.1-sol high | ~5m | 4/4/0 | 7 | 1 (prompt allowed it) | Frame-card spans, caption font hashing, subtitle window, beats regex, NaN words |
| 5 | W5b grok-4.7 | gpt-6.1-sol high | 6.8m | 0/6/0 | 6 | 0 | Gold per-unit shadow, integer-px rise, swallowed font errors, self-referential fidelity test |
| 5 | W5c grok-4.7 | gpt-6.1-sol high | 6.1m | 1/5/0 | 6 | 0 | Clip tail decoded as float; straight-alpha bilinear; oracle about:blank race |
| 5 | W5d space-bunny | gpt-6.1-sol high | ~5m | 4/4/2 | 6 | 3 (driver's diff base, not the reviewer) | Partial-gen exit code, fade clamp, relative paths, racy lock test |
| 6 | W6 grok-4.7 | gpt-6.1-sol medium | ~6m | 2/6/1 | 9 | 0 | Leak duration and photo/clip stem collision outside the segment hash; warm-film ignored fade; zoom lost with tone |
| 7 | W7 grok-4.6 | gpt-6.1-sol medium | ~6m | 0/9/2 | 11 | 0 | Rounded seconds vs frame boundaries, NaN timestamps, `Content-Length: -1` hang, stat/open race |
| 7 | W7 gpt-5.6-luna | gpt-6.1-sol medium | ~5m | 0/8/1 | 9 | 0 | Same HTTP classes as grok's plus a page reload bug; 6 of 9 overlap the grok findings |

## Takeaways so far (wave 1)
- **Implementation:** grok-4.6 was fastest to a fully correct result on a tightly specified pure-logic task. Space Bunny is slow and token-heavy, but it was the most rigorous: it ran real-machine smokes and proved its tests fail without the fix. Muse Spark wrote the leanest, most faithful GPU port.
- **Review:** gpt-6.1-sol found real issues on every diff, with no false positives so far. One Sol pass still misses things it catches on a sibling diff, so cross-check shared patterns by hand. Muse Spark is not a usable reviewer.
- **Reliability:** the free Muse tier failed twice (a hung session resume, then a 503 overload). Space Bunny and grok had no infrastructure failures.

## Takeaways (wave 2)
- **Integration:** grok-4.7, Space Bunny and LongCat all passed every hidden test (parity, cache locality, CLI). Sol's findings decided it: grok-4.7 had the fewest real issues per line and fixed all six in one 11-minute round.
- **LongCat 2.5:** slow, read-heavy start that hit the output-token limit, but after one resume it produced the smallest correct diff.
- **Dropped from wave 3 on (Raja's call):** LongCat, Ling, Nemotron and similar free trial models. From wave 3, routing follows this table: grok-4.6 for tight-spec code, grok-4.7 and Space Bunny for render/integration, Sol for review.
- **Sol misses about one finding per diff** that it catches on a sibling diff. Head-to-head reviews plus a driver cross-check found 2 bugs that a single review would have shipped.
- **Full-resolution parity cannot see preview-scale bugs** (k = 1 hides any missing or doubled `Canvas.k`). Future render contracts need a preview-scale assertion.

## Takeaways (wave 3)
- **grok-4.6 stays the pick for tight-spec fixes:** 12 minutes and $0.55 for three bugs across encode, CLI and schema, including one driver-requested rework.
- **grok-4.7 vs Space Bunny on GPU perf:** both found the same 10× win (cull each print to its screen box). grok finished in under 20 minutes; Space Bunny was still verifying at about 45. On a tight task, grok's speed matters more than Space Bunny's extra rigor.
- **The driver's own review caught a regression that the hidden tests missed**: correct output, 5 s slower per render. Hidden tests need a timing budget wherever speed is part of the contract.

## Takeaways (waves 5–7: M2 and M3)
- **grok-4.7 owns hard render work:** text cards, golden-film layers and integration each landed with PSNR-pinned parity, then
  one fix round apiece. Cost $1.8–2.7 and ~1 h per task; it calibrates against the oracle before coding. On W6 it exited
  while its own final render was still muxing, so the driver re-ran the real gate.
- **grok-4.6 stays the tight-spec pick:** schema, recipe, M3 and every resumed fix round finished in 6–23 min under $1.60.
- **gpt-5.6-luna (xhigh, fast tier) is a viable implementer:** fastest M3 run (9.3 min), passed all hidden tests, same
  Sol finding classes as grok-4.6. It wrote fewer tests, which lost the head-to-head. Worth more head-to-heads.
- **Space Bunny reliability dropped:** one W7 run hung at opencode bootstrap with zero model calls. Watch a fresh run's
  log for model calls within the first minute.
- **Sol medium matched Sol high in signal:** 9–11 real findings per diff, no false positives. Both cache-correctness
  HIGHs in W6 came from Sol; the driver caught one overreach in the fix (hashing the film length re-rendered every
  segment) and one real-render bug no test saw (single-pass loudnorm missed −14 LUFS by 1.4 LU).

## Skill benchmark (M4)
Fresh headless `claude -p` session, only the `make-video` skill installed, same prompt each run: a 30 s 9:16 silent
birthday montage from the 29 Arshiya photos, brief answered up front, no human review. Measured with
`tools/token-report.py` (input-equivalent = input + 1.25·cache write + 0.1·cache read + 5·output).

| Run | Skill rev | Model | Wall | Turns | Start ctx | Images | Input-eq | Cost | Result |
|---|---|---|---|---|---|---|---|---|---|
| 1 | v1 | Sonnet 5.5 | 1.9m | 15 | 28k | 3 | 0.18M | $0.45 | `OK check`; 14 stills, 0.7 s dissolves, gold script title, fade to black |
| 2 | v1 | Sonnet 5.5 | 2.3m | 22 | 27k | 4 | 0.24M | $0.58 | 2 false `WARN frozen` on untilted frame cards (engine bug, fixed); 3 text-size renders |
| 3 | v1 | Opus 5.5 | 1.2m | 9 | 28k | 1 | 0.12M | $0.40* | *Stopped after backgrounding the preview: headless `-p` gets no notification (skill fixed: short renders in foreground) |
| 4 | v2 | Sonnet 5.5 | 2.4m | 16 | 27k | 3 | 0.19M | $0.47 | `OK check`; dimmed the closing photo for title contrast |
| 5 | v2 | Opus 5.5 | 2.7m | 11 | 25k | 2 | 0.16M | $0.71 | `OK check`; polaroid cards for landscape selfies, photo wall, two-line gold title on black. One false freeze WARN on a swirl tail (fixed) |
| 6 | v2 | Sonnet 5.5, review round | 1.1m | 7 | 25k | 1 | 0.09M | $0.25 | Fresh session, one user message ("apply my notes"): read `review --summary` (4 notes), applied all 3 edits, re-rendered, `OK check` |

Targets (PLAN.md): input-eq ≤ 0.3M ✅ (0.12–0.24M vs 1.0–2.7M before Montaj), turns ≤ 20 ✅ on v2 (16, 11),
starting context ≤ 55k ✅ (25–28k). Feedback round via review.json: one user message, 7 agent turns, 0.09M (PLAN target ≤ 3 turns counts user turns ✅).
Not yet measured: a run with music (`music analyze` / `gen`).
