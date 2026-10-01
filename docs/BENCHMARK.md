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

## Review runs
| Wave | Diff | Reviewer | Wall | Findings (H/M/L) | Confirmed real | False positives | Notes |
|---|---|---|---|---|---|---|---|
