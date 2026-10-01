# Token audit — pre-Montaj video sessions (2026-10-01)
> Baseline for Montaj's targets. Reproduce:
> `python3 tools/token-report.py --only 4a14d5e4,2fbafb84,a4108ea3,dbd3021f,38d1158c,99281b98,c5772170`
> input-eq = input + 1.25·cache_write + 0.1·cache_read + 5·output (relative weights, not prices).

## Videos made (all in `~/Anuraj-dev/claude-test/`), and how
| Video | Folder | Method | Session | Model |
|---|---|---|---|---|
| Claude motion reel 15s | `reel/` | canvas HTML + CDP frame export + numpy synth score | a4108ea3 | Opus 5.5 |
| Portfolio showreel 15s | `portfolio-video/` | canvas HTML (rewritten) + export.mjs + score.py | dbd3021f | Opus 5.5 |
| Sonnet 5.5 reel 15s | `Sonnet 5.5/` | pycairo + numpy, numba synth audio | 38d1158c (+99281b98) | Sonnet 5.5 |
| Opus showcase (Twitter) | `design-test/showcase-edit/` | Chromium card screenshots + Python compose over screen recording | c5772170 | Opus 5.5 |
| Arshiya birthday 2:45 | `bday-video/film.html` | canvas/DOM HTML + forked export.mjs + ACE-Step song + whisper | 4a14d5e4 | Opus 5.5 |
| Vidyut birthday 28s | `bday-video/v2/` | torch GPU camera renderer + depth maps (film.py → film2.py) | 2fbafb84 | Opus 5.5 |

Every video used a new pipeline. Re-written each time: easing libs (3 JS, 2 Python), frame exporter (forked 3×), audio synth (3×), ffmpeg encode settings.

## Numbers
| session | turns | ctx at turn 1 | ctx max | images | cache_read | cache_write | output | input-eq |
|---|---|---|---|---|---|---|---|---|
| 2fbafb84 Vidyut | 24 | 111k | 224k | 49 | 3.87M | 0.20M | 69k | 0.98M |
| 4a14d5e4 Arshiya | 97 | 109k | 262k | 41 | 18.96M | 0.24M | 94k | 2.66M |
| 38d1158c Sonnet reel | 28 | 50k | 260k | 13 | 4.70M | 0.24M | 135k | 1.44M |
| 99281b98 Sonnet reel (retry) | 5 | 50k | 58k | 0 | 0.24M | 0.03M | 4k | 0.09M |
| a4108ea3 Claude reel | 23 | 49k | 116k | 5 | 1.97M | 0.09M | 51k | 0.57M |
| c5772170 Opus showcase | 33 | 28k | 132k | 10 | 2.94M | 0.11M | 43k | 0.65M |
| dbd3021f Portfolio reel | 45 | 50k | 185k | 10 | 5.56M | 0.16M | 82k | 1.17M |
| **total** | | | | | **38.23M** | **1.07M** | **478k** | **7.56M** |

Split of input-eq: ~50% context re-reads, ~32% output, ~18% cache writes.

## The six sinks (evidence)
1. **Pasted photos.** Birthday sessions started at ~110k vs ~50k baseline: ~60k of images re-read every turn (≈5.8M of Arshiya's 19M).
2. **Dead-end exploration.** ~30 turns at 230–255k chasing GPU rendering in headless Chromium (result: impossible) ≈ 7M. Also ACE-Step dangling venv, font hunting.
3. **Engine authored from scratch.** Single Writes of film.html / film.py / reel.py = 20–35k output tokens; Sonnet session 135k output total.
4. **Polling renders.** Repeated `tail` / `sleep` / `ls frames | wc -l` turns, each re-reading ~200k context.
5. **Taste via rejected drafts.** Vidyut: "photos must not warp" surfaced after 2 drafts → full rewrite film.py → film2.py.
6. **Review images persist.** Every contact sheet stays in context for the rest of the session (49 images in Vidyut).
