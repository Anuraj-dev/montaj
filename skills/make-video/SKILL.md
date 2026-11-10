---
name: make-video
description: Make or edit a video with Montaj from a folder of photos, clips or screen recordings — birthday films, montages, reels, shorts, vlogs, explainers, launches, highlights. Use when the user asks for a video, film, montage, reel or short, or brings review notes on a Montaj render.
---

# make-video

You direct; Montaj renders. Your whole output is one YAML spec, `<project>/montaj.yaml` (plus any music or
`custom:` Python you choose to write). Every render, check and review round after that is a `montaj` command
that costs no tokens. Run every command as `montaj -C <project> <cmd>`. Each prints one result line,
`OK …`, `WARN …` or `ERR <what>: <fix>`; progress goes to `<project>/build/*.log`.

## Budget
| Command | Cost | Worth it when |
|---|---|---|
| `montaj spec <topic>` | ~0.3–3k tokens | before using a feature you haven't used this session |
| `montaj spec --search <word>` | ~0.2k | you know the word, not the field |
| `validate`, `check`, `inspect --problems`, `scopes` | one line each | every iteration — numbers, not pictures |
| `sheet`, `ingest-sheet.jpg`, `--strip`, `--transitions` | ~1.5k per image | composition, framing, motion you can't judge from numbers |

- At most 2 images per iteration. The photos and clips stay on disk; never ask the user to paste them.
- Engine source is out of scope. On an engine bug, report the command and its `ERR` line, then stop.
- A 30 s film renders in under a minute: run `render` and `check` in the foreground. Long finals and
  `music gen` take minutes: run them in the background and wait for the notification.

## Steps

1. **Brief.** `montaj taste` prints the user's standing preferences; they bind. Ask one round of questions,
   all at once: length, aspect, who it's for, on-screen text, motion style, music (none / their track / a
   generated song / one you compose). Then write yourself a five-line treatment:
   - **Hook**: what holds the first second.
   - **Arc**: how energy moves (open, build, peak, release).
   - **Type**: one or two text styles and where text sits.
   - **Colour**: a look plus any per-shot grade intent.
   - **Sound**: the music, its tempo and where it hits.
   Done when each question has an answer or a default you stated.

2. **Project.** `montaj new --list` prints one line per recipe. Pick the one closest to the treatment:
   `montaj new <project> --recipe <recipe> [--photos <folder>]`. Without `--photos`, a recipe with a demo makes
   synthetic assets, so it renders as-is. Read `recipes/<recipe>.md` (what to ask, what to change). Look at
   `build/ingest-sheet.jpg` once; `assets/analysis.json` has faces, subjects, cuts, speech and motion as
   numbers. Done when you can name each asset's content in a few words.

3. **Music** (skip for a silent film). `montaj spec audio` has the fields.
   - Their track: copy it into `music/`, then `montaj music analyze music/<file>.wav --lang <code>` →
     `music/markers.json` (beats; words with times). Instrumentals take `--no-words`. Default lang is `hi`.
   - Generate (ACE-Step): write `music/lyrics.txt`, then `montaj music gen --caption "<style>" --lyrics
     music/lyrics.txt --bpm <n> --duration <len>s --lang <code> --n 2` (background). The user picks by ear.
   - Compose: write `music/compose.py` (numpy; writes a wav to `sys.argv[1]`), run `montaj music compose
     music/compose.py`, and read the `OK` line (length, peak, LUFS) since you can't listen.
     `recipes/music/compose-example.py` is a starting point. Start the first downbeat at 0 s and set the
     same `video.bpm`.
   - Gain staging: keep the bed under voice and SFX (`music compose` turns a bed down to -6 dBFS peak when
     the spec has clips). For an `INFO clip` peak, set `gain` on the clip `montaj check` names, not `compose.py`.
   Done when the song exists and you know its tempo and first beat.

4. **Spec.** Start from the recipe and depart from it freely: the treatment decides, not the recipe. Before
   using a feature, read its topic (`montaj spec` lists 24; `captions`, `layers`, `keys`, `grade`, `emphasis`,
   `cuts`, `custom`, …). Things that are easy to miss:
   - Cuts land on the beat grid (`4b`) or on markers (`until: chorus`; a word is `"word:12"`).
   - Talking footage: `cuts:` cuts on the transcript; `captions:` word-times it; `emphasis:` hits words.
   - Anything not in a field: `custom:` (shot), `effects[].custom` or `in.src` (transition) run your own
     `draw(ctx)` on the GPU. `montaj custom test <file.py>` checks it in seconds.
   - Repetition: `styles:`, `defaults:`, `components:` + `use:`; `montaj validate --expand` shows the result.
   `montaj validate` and fix every `ERR` (each names its fix). Done on `OK montaj.yaml …`.

5. **Preview.** `montaj render`, then numbers first: `montaj inspect --problems` (text off-frame, low
   contrast), `montaj scopes` (exposure, clipping, cast, skin); `montaj match --to N` evens out a scene's
   grade. Then `montaj sheet` and look at `build/sheet.jpg` for framing and the arc. For motion, one
   `sheet --strip <t>` per new kind of motion (not per shot) or `sheet --transitions`; after a re-render
   `--diff` shows only what changed. Only changed segments re-render. Iterate until the sheet shows the
   treatment; stop when another round would change nothing the viewer would notice.

6. **Review: zero tokens.** Hand the user two commands for their own terminal:
   `montaj -C <project> review` (play, approve/reject shots, comment at the playhead) and
   `montaj -C <project> watch` (re-renders on every save). When they come back, `montaj review --summary`
   prints one line per noted shot; apply every note. A note that states a lasting preference becomes
   `montaj taste add "<preference>"`.

7. **Final.** `montaj render --final`, then `montaj check`: done on `OK check`. Each `WARN` names a defect to
   fix in the spec; `INFO` is informational. For phone sharing, `montaj export --target whatsapp`. Report the
   output paths.
