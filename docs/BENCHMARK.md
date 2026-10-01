# Model benchmark — Montaj build
> Real tasks from this repo, scored by the driver (Claude Opus 5.5) against tests and review findings.
> Append a row per dispatch. Scores are evidence from this repo only, not general rankings.

## Models in play
| Route | CLI | Cost to Raja |
|---|---|---|
| grok-4.7, grok-4.6, grok-4.5 | `grok -p -m <id>` | grok quota |
| gpt-6.1-sol (review), gpt-6-luna | `codex-run.sh` | codex quota |
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
