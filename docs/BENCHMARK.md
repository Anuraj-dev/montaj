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

## Review runs
| Wave | Diff | Reviewer | Wall | Findings (H/M/L) | Confirmed real | False positives | Notes |
|---|---|---|---|---|---|---|---|
| 1 | W1a grok-4.6 | gpt-6.1-sol high | 4.2m | 0/5/0 | 5 | 0 | Overlapping transitions, unnormalised wall key times, NaN bpm |
| 1 | W1a grok-4.6 | muse-spark-1.3 | 5.1m | 0/0/0 | — | — | Missed all 5 Sol findings: weak reviewer |
| 1 | W1b muse-spark | gpt-6.1-sol high | 2.9m | 0/3/0 | 3 | 0 | Test quality only; missed global-RNG + sharpen-k issues it flagged on grok-4.7's diff |
| 1 | W1b grok-4.7 | gpt-6.1-sol high | 4.5m | 0/3/1 | 4 | 0 | Global RNG, focal and sharpen not scaled by k |
| 1 | W1c space-bunny | gpt-6.1-sol high | 3.6m | 3/4/1 | 8 | 0 | mux re-encode + truncation, QA swallowing ffmpeg failures |

## Takeaways so far (wave 1)
- **Implementation:** grok-4.6 was fastest to a fully correct result on a tightly specified pure-logic task. Space Bunny is slow and token-heavy, but it was the most rigorous: it ran real-machine smokes and proved its tests fail without the fix. Muse Spark wrote the leanest, most faithful GPU port.
- **Review:** gpt-6.1-sol found real issues on every diff, with no false positives so far. One Sol pass still misses things it catches on a sibling diff, so cross-check shared patterns by hand. Muse Spark is not a usable reviewer.
- **Reliability:** the free Muse tier failed twice (a hung session resume, then a 503 overload). Space Bunny and grok had no infrastructure failures.
