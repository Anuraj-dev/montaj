---
name: make-video
description: Make or edit a video with Montaj from a folder of photos, clips or screen recordings — birthday films, montages, reels, shorts, vlogs, explainers, launches, highlights. Use when the user asks for a video, film, montage, reel or short, or brings review notes on a Montaj render.
---

# make-video

You direct; Montaj renders. Your output is one YAML spec, `<project>/montaj.yaml` (plus music or `custom:`
Python). Every render, check and review after that is a `montaj` command that costs no tokens. Run each as
`montaj -C <project> <cmd>`; it prints one result line, `OK …`, `WARN …` or `ERR <what>: <fix>`.

## Budget
| Command | Cost | Worth it when |
|---|---|---|
| `montaj spec <topic>` | ~1–2k tokens | before a feature you haven't used this session; one topic at a time |
| `montaj spec --search <word>` | ~0.2k | you know the word, not the field (`--full` only for a detail the short form lacks) |
| `montaj assets [stem]` | a few lines per clip | learning the footage: cuts, speech, events, loudness |
| `validate`, `check`, `inspect --problems`, `scopes` | one line each | every iteration — numbers, not pictures |
| `sheet`, `ingest-sheet.jpg`, `--strip`, `--transitions` | ~1.5k per image | framing or motion you can't judge from numbers |

- At most 2 images per iteration. The assets stay on disk; never ask the user to paste them.
- Learn the footage from `montaj assets`, never from `analysis.json`, `*.events.json` or ffmpeg.
- Write the spec once with Write and change it with Edit; never generate it from a script.
- A 20 s short fits in 60 lines. If yours does not, a shorthand is missing: say which in your report.
- Engine source is out of scope. On an engine bug, report the command and its `ERR` line, then stop.
- `render` and `check` of a short run in the foreground; long finals and `music gen` in the background.

## Steps

1. **Brief.** `montaj taste` prints standing preferences; they bind. Ask one round, all at once: length,
   aspect, audience, on-screen text, motion style, music (none / theirs / generated / composed). Write a
   five-line treatment: **Hook** (the first second), **Arc** (energy), **Type** (one or two styles),
   **Colour** (a look), **Sound** (tempo, where it hits).

2. **Project.** `montaj new --list`, then `montaj new <project> --recipe <recipe> [--photos <folder>]`
   (no `--photos`: a demo recipe renders as-is). Read `recipes/<recipe>.md`. Look at `build/ingest-sheet.jpg`
   once and run `montaj assets`. Done when you can name each asset's content in a few words.

3. **Music** (skip for silence). `montaj spec audio`.
   - Their track: copy into `music/`, `montaj music analyze music/<file>.wav --lang <code>` (`--no-words`
     for instrumentals) → `music/markers.json`.
   - Generate: write `music/lyrics.txt`, `montaj music gen --caption "<style>" --lyrics music/lyrics.txt
     --bpm <n> --duration <len>s --lang <code> --n 2` (background); the user picks by ear.
   - Compose: `montaj new` already created `music/compose.py` when the recipe has a song: Read it, then Edit
     it (else start from `recipes/music/compose-example.py`). `montaj music compose music/compose.py`; read
     the `OK` line (length, peak, LUFS). First downbeat at 0 s; the same `video.bpm`.
   - Gain: for an `INFO clip` peak, apply the cut `montaj check` names (a clip's `gain`, or `audio.sfx_gain`).

4. **Spec.** The treatment decides, not the recipe. Read one `montaj spec <topic>` at a time (short form;
   `montaj spec` lists them). Shorthands that keep a short short:
   - Titles: one line each, `- {at: 4b, to: 6b, text: "SWITCH *WORKSPACES*"}`; style and position come
     from `styles:`/`defaults:`. In a `lines:` block a line's `at` defaults to the block's `from`.
   - Sound: `audio.sfx: [text, cuts]` puts a hit on each slam and a swish on each hard cut (not a jump cut
     within a take, nor the cut to a blank or textless card), never doubling a sound you placed; `auto` adds
     transition whooshes, the `drop` riser and the `reveal: type` tick.
   - A PNG sequence is one layer: `{frames: "m*.png", every: 6f}`, not a layer per image.
   - Endings: `video.outro: {type: fade, dur: 30f, hold: 15f}` ends on true black; `card: {color: "#000"}`
     is a textless card.
   - Cuts on the beat grid (`4b`) or markers (`until: chorus`); talking footage: `cuts:`, `captions:`,
     `emphasis:`; anything else: `custom:` (`montaj custom test <file.py>`).
   - Repetition: `styles:`, `defaults:`, `components:` + `use:`; `montaj validate --expand` shows the result.
   `montaj validate` and fix every `ERR` (each names its fix). Done on `OK montaj.yaml …`.

5. **Preview.** `montaj render`, then numbers: `montaj inspect --problems` (text off-frame, overlap, low
   contrast), `montaj scopes` (`match --to N` evens a scene's grade). Then `montaj sheet` for framing and arc;
   one `sheet --strip <t>` per new kind of motion or `--transitions`; `--diff` after a re-render. Stop when
   another round would change nothing a viewer would notice.

6. **Review: zero tokens.** Hand the user `montaj -C <project> review` and `montaj -C <project> watch`.
   Then `montaj review --summary` lists the notes; apply every one. A lasting preference becomes
   `montaj taste add "<preference>"`.

7. **Final.** `montaj render --final`, then `montaj check`: done on `OK check`. Fix each `WARN` in the spec.
   For phones, `montaj export --target whatsapp`. Report the output paths.
