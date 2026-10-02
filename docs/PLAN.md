# Montaj — plan
> Source of truth for scope and milestones. Written 2026-10-01 (M0). Update when scope changes; log why in decisions.md.

## Context
Six videos were made in `~/Anuraj-dev/claude-test/` (claude reel, portfolio showreel, Sonnet 5.5 reel, Opus showcase edit, Arshiya birthday film, Vidyut birthday montage), and **each one used a new hand-built pipeline**. Token audit of the 7 video sessions: 38.2M cache-read, 1.08M cache-write, 478k output tokens, about 7.6M input-equivalent. The biggest sinks:
- photos pasted inline (+60k context re-read on every turn, about 30% of the Arshiya session);
- render/GPU dead ends (about 30 turns at 240k, roughly 7M);
- engines rewritten from scratch (20–35k output per Write);
- polling renders;
- learning taste through rejected drafts;
- review images that stay in context forever.

Goal: a CLI + skill + fixed project layout. An agent states creative intent in a small spec, the engine does everything mechanical, and the human re-renders and reviews **without spending tokens**. Raja asked me to self-question and to optimise for future agents (me). This plan is that output.

Name: **Montaj** (Hindi/Urdu "montage"). Repo `~/Anuraj-dev/montaj`, CLI `montaj`, short to type and short in transcripts. Renaming is cheap now and expensive later.

---

## Self-interrogation (questions I asked myself → answers that shape the design)
1. **What does a future agent actually need to decide?** Only creative intent: which photos, in what order, held how long, which transitions, what text, what music mood. Everything else (camera math, encode flags, motion blur, grading, A/V mux, QA) must be engine defaults. → **The spec is the only thing an agent writes.**
2. **Code or data for the spec?** Data. A YAML spec of about 50–150 lines replaces 500–1,900 lines of bespoke code (5–10× fewer output tokens), can be validated, diffs cleanly, and a human can edit it with zero tokens. → YAML + pydantic schema; `montaj validate` returns one-line errors with fix hints.
3. **One engine or several?** One. Four runtimes (canvas-HTML, pycairo, torch, compose.py) is the root of the re-invention. → **Python + torch GPU compositor** (proven in `v2/film.py`/`film2.py`, runs on the RTX 5050 with NVENC). Escape hatch for bespoke motion graphics: a `custom:` layer pointing at a small project-local Python function, so only the novel part is code.
4. **How does text look as good as CSS without per-frame Chromium (CPU-only, slow, the source of the GPU detour)?** Rasterise each text card/word **once** with headless Chromium (HTML/CSS: fonts, gold gradients, Devanagari shaping, the pattern already used in `showcase-edit/render.sh`) into RGBA + bbox metadata, cached by hash. Animate those textures on the GPU (opacity, blur, transform, mask sweep).
5. **How do renders cost zero tokens?** Deterministic + cached:
   - every segment (cut-to-cut or transition span) is hashed (spec slice + asset hashes + engine version) and cached as an encoded chunk;
   - a re-render recomputes only changed segments, then concats and muxes;
   - `montaj watch` re-renders the preview whenever the spec is saved, so the human iterates alone.
6. **How does review cost zero tokens?** `montaj review` serves a local page: player, shot timeline from the spec, a comment box at the current timestamp, approve/reject per shot. It writes `review.json`, so the agent reads about 300 tokens of structured feedback instead of the human typing prose or the agent re-watching frames.
7. **When the agent must look, what is the cheapest look?** `montaj sheet`: one contact-sheet image of ≤12 frames at fixed ≤1536px (~1.5k tokens), chosen from segment midpoints or `--at` times. Plus `montaj check`, a text-only QA (black/frozen frames, duration, loudness, A/V offset, size targets). Skill rule: ≤2 images per iteration.
8. **How do I stop learning taste through rejected drafts?**
   - one up-front brief (≤5 AskUserQuestion items: length, aspect, text/no text, motion style, music);
   - a persistent `~/.config/montaj/taste.md` (e.g. "photos never warp; motion only in transitions; cuts on the beat grid") that every agent reads (small) and updates when the user rejects something.
9. **How do I stop dead-end exploration?** `montaj doctor` checks GPU/CUDA, NVENC, Chromium, the ACE-Step path and dangling-venv fix, fonts and VRAM, and prints the known traps ("headless Chromium cannot GPU-raster; don't try"). The skill says: on an engine bug, stop and report; don't debug engine internals inside a creative session.
10. **How do I stop polling?** Every long command blocks quietly, logs progress to `build/*.log`, and ends with exactly one line (`OK out/final.mp4 28.0s 1080x1920 39MB`). Agents run it with `run_in_background` and wait for the notification.
11. **How does the next agent avoid re-reading the codebase?**
    - creative agents read only `skills/make-video/SKILL.md` (~150 lines) + `docs/SPEC-REFERENCE.md` (every field, transition and look preset, one line each) + a recipe;
    - engine developers read `docs/STATE.md`;
    - nobody needs to read engine source to make a video.
12. **Music?** Wrap what already works: ACE-Step 1.5 local (`bday-video/music/generate.py` + the `py.sh` dangling-venv trick) and faster-whisper word timestamps (`music/analyze.py`). Also accept a user-supplied track. In both cases `montaj music analyze` emits `markers.json` (beats, sections, lyric words, rms), and the spec times things by marker name (`at: chorus1`) or beats (`hold: 4b`). Cutting on the beat becomes the default instead of a skill.
13. **MCP server or CLI?** CLI. It works for Claude, Codex and opencode alike, costs no tool-schema tokens per turn, and composes in bash. Revisit only if a host can't shell out.
14. **How will I know it worked?** A benchmark: a fresh agent session with only the skill makes a 30s birthday montage from a photo folder; measure it with `tools/token-report.py`. Targets are below. Without this, "saves tokens" is a claim, not a fact.
15. **What is out of scope (v1)?** GUI editor, cloud rendering, stock footage, general NLE features, MCP, a synth-score library. Video-clip input comes in M2 (the Opus showcase needed it).

---

## Architecture
```
montaj/
  pyproject.toml          uv project; console script `montaj`; torch cu128 (sm_120 for RTX 5050)
  montaj/
    cli.py                subcommands; quiet; one-line result; --json
    spec.py               pydantic models, defaults, validation hints
    project.py            project layout, asset + segment hashing
    timeline.py           beats/markers/durations -> frame ranges -> segments
    ingest.py             EXIF rotate, resize (2048 px cap), assets.json (w, h), numbered sheet
    render/core.py        camera + layer sampling            <- v2/film.py (camera, draw, Img, solo, shift, key/keys_s)
    render/transitions.py cut, fade, whip, swirl, matchcut, flash, wall-drone  <- v2/film2.py
    render/finish.py      grade, vignette, leaks, bloom, grain, motion blur    <- film2.finish / render_frame
    render/text.py        HTML/CSS card -> RGBA via headless Chromium (once, cached) <- showcase-edit/render.sh
    render/encode.py      NVENC rawvideo pipe, segment cache, concat, mux       <- film2 __main__, bday render.sh
    audio/music.py        ACE-Step wrapper                    <- bday-video/music/generate.py, py.sh
    audio/analyze.py      whisper words + beats + rms -> markers.json  <- music/analyze.py
    qa.py                 check
    review/               static page + tiny server -> review.json
  templates/montage/      default spec + looks
  recipes/                birthday-short.yaml (Vidyut port), birthday-song-film.yaml (Arshiya port)
  skills/make-video/SKILL.md
  tools/token-report.py
  tests/
  docs/                   context system + PLAN, SPEC-REFERENCE, research/
```
User projects live outside the repo: `<anywhere>/<name>/{montaj.yaml, assets/, music/, build/, out/, review.json}`.

### CLI surface (the whole agent-facing API)
| Command | Does | Agent reads |
|---|---|---|
| `montaj doctor` | env check + known traps | 5–10 lines |
| `montaj new <dir> --recipe birthday-short --photos <folder>` | scaffold + ingest | 1 line + sheet path |
| `montaj ingest` | normalise photos, focus points, `assets.json`, one numbered sheet | 1 image |
| `montaj music gen\|analyze\|compose` | song candidates / markers.json / agent-composed wav | short table |
| `montaj validate` | schema + timeline errors with hints | errors only |
| `montaj render [--preview\|--final]` | cached segment render + mux | 1 line |
| `montaj sheet [--at …]` | ≤12-frame contact sheet | 1 image |
| `montaj check` | QA | ≤10 lines |
| `montaj review` / `watch` | human loop, zero tokens | `review.json` |
| `montaj export --target master\|whatsapp` | presets from render.sh | 1 line |

### Spec sketch (illustrative)
```yaml
video: {size: 1080x1920, fps: 30, look: warm-film, length: 28s}
audio: {track: music/song.wav}          # or bpm: 120 for silent
shots:
  - {photo: 16, hold: 4b, focus: [470, 745], zoom: 1.3}
  - {photo: 32, in: swirl, dur: 22f}
  - {photos: [28, 20, 21, 12, 19], hold: 1b}   # beat chain
  - {photo: 9, in: whip-x}
text:
  - {at: chorus1, text: "Arshiya", style: script-gold, y: .58}
end: {wall: [34, 9, 10, 16, 38, 32], hero: 10}
```

### Agent rules (go into AGENTS.md + the skill)
- Take a photo **folder path**; never ask the user to paste images.
- Look only at `ingest` and `sheet` outputs; ≤2 images per iteration.
- Never read engine source in a creative session; on an engine bug, report it and stop.
- Long commands run in the background; no polling.
- One brief round first; read and update `taste.md`.
- Use a fresh session between phases; run `/checkpoint` at the end.

---

## Milestones (each ends with `/checkpoint`)
- **M0 Scaffold** (this session): repo, plan, context, token audit.
- **M1 Engine + montage**:
  - spec/timeline/render/encode/cache, plus `new`, `ingest`, `validate`, `render`, `sheet`, `check`, `export`, `doctor`;
  - port Vidyut `film2.py` to `recipes/birthday-short.yaml`.
  - Exit: renders a visually equivalent film from YAML; a one-shot edit re-renders only that segment.
- **M2 Text + music + clips**:
  - Chromium text cards, ACE-Step/whisper markers, user tracks, video-clip layer;
  - port Arshiya to `recipes/birthday-song-film.yaml`.
- **M3 Zero-token loop**: `review` page, `watch`, `taste.md`.
- **M4 Skill + benchmark**:
  - `skills/make-video/SKILL.md`, `SPEC-REFERENCE.md`, symlinked into `~/.claude/skills` and Codex;
  - run the fresh-agent benchmark and iterate the docs until the targets are met.
- **M5 Motion-graphics** (later): `custom:` layer API; port `reel/reel.html` as a recipe.

## Targets (measured by `tools/token-report.py`)
| Metric | Today | Target |
|---|---|---|
| New 30s montage, input-equivalent tokens | 1.0M (Vidyut), 2.7M (Arshiya) | ≤ 0.3M |
| Agent turns per new video | 24–97 | ≤ 20 |
| Starting context | 110k with pasted photos | ≤ 55k |
| Human re-render / review round | 1 agent session | 0 tokens |
| Feedback round via review.json | multi-turn prose | ≤ 3 turns |
| Final render, 30s 1080×1920 on RTX 5050 | unmeasured | ≤ 3 min; preview ≤ 60s |

## Risks
- torch must support sm_120: pin to the ACE-Step venv's working version.
- 8GB VRAM: music generation and rendering can't run concurrently, so the CLI serialises them.
- ACE-Step is 19GB inside claude-test: reference it via `~/.config/montaj/config.toml`; don't move it.
- Text raster without per-frame CSS loses live gradient shimmer: emulate with a mask sweep.

## Verification per milestone
- M0: repo + context files exist; `python3 tools/token-report.py --only <ids>` reproduces the audit (see docs/research/token-audit-2026-10-01.md).
- M1+: pytest (spec validation, timeline resolution, segment-hash cache invalidation, golden low-res frame hashes), then a real render of the recipe watched by Raja, with the `montaj check` output pasted into the session log.
- M4: fresh-agent benchmark measured with tools/token-report.py against the Targets table.
