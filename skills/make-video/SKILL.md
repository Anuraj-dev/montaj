---
name: make-video
description: Make or edit a video with Montaj from a folder of photos or clips — birthday films, montages, reels, lyric videos. Use when the user asks for a video, film, montage or reel, or brings review notes on a Montaj render.
---

# make-video

You direct; Montaj renders. Your whole output is one small YAML spec, `<project>/montaj.yaml`. Every render,
check and review round after that is a `montaj` command that costs no tokens. Run every command as
`montaj -C <project> <cmd>`. Each prints one result line, `OK …` or `ERR <what>: <hint>`; progress goes to
`<project>/build/*.log`.

## Budget
- Look only at Montaj contact sheets (`build/*.jpg`), at most 2 images per iteration. The photos stay on disk.
- Read `~/Anuraj-dev/montaj/docs/SPEC-REFERENCE.md` (every spec field, one line each) and one recipe. Engine
  source is out of scope: on an engine bug, report the command and its `ERR` line to the user and stop.
- Films up to 60 s render in under a minute: run `render` and `check` in the foreground. Longer finals and
  `music gen` take minutes: run them in the background and wait for the notification.

## Steps

1. **Brief.** `montaj taste` prints the user's standing preferences; follow them. Then ask one round of
   questions, all at once: length, aspect, who it is for, on-screen text, motion style, music (none / their
   track / generate). Done when each question has an answer or a default you stated.

2. **Project.** `montaj new <project> --recipe <recipe> --photos <folder>`, then look at
   `<project>/build/ingest-sheet.jpg` once. Recipes (`~/Anuraj-dev/montaj/recipes/`):
   - `birthday-short` — 28 s photo montage: cuts, fades and swirls on a beat grid, a photo wall, warm-film.
   - `birthday-song-film` — 165 s lyric film: text cards, subtitles, song markers, Ken Burns, golden-film.
   `assets/assets.json` lists each stem's `w`, `h`: a landscape photo loses its sides in 9:16, so frame it with
   `frame:` (polaroid card) or aim `focus` at the face. Done when you can name each stem's content in a few words.

3. **Music** (skip for a silent film).
   - Their track: copy it into `<project>/music/`, then `montaj music analyze music/<file>.wav` →
     `music/markers.json` (beats; words with times).
   - Generate: write `music/lyrics.txt`, then `montaj music gen --caption "<style>" --lyrics music/lyrics.txt
     --bpm <n> --duration <len>s --lang <code> --n 2` (minutes, background). `--lang` defaults to `hi`; set it
     to the lyrics' language (`en`, `hi`, …). The user picks a candidate by ear; analyze it.
   Done when `markers.json` exists and you have read the analyze output.

4. **Spec.** Rewrite `montaj.yaml` from the recipe: keep its look, transitions and timing grammar; replace the
   shots, text and markers.
   - Cuts land on the beat grid (`4b`, `8b`) or on markers (`until: v3`, `word:12`).
   - Order shots as an arc: calm open, build, peak, quiet end.
   - Text is short and centred; one idea per card. Width budget on a 1080-wide frame: about 24 characters of
     `script` at `size: 100`; scale `size` down in proportion for longer lines.
   - Text reads best on a calm backdrop: a dark or plain moment, a dimmed photo or its own card, not a face. A
     closing title usually holds to the last frame.
   - Subtitles follow the song: start them at the first sung word in `markers.json`; an instrumental intro
     can carry a title or no text.
   - Framing: `focus: [x, y]` (source px) puts a face at the screen centre; `crop` trims.
   `montaj validate` and fix every `ERR` line. Done on `OK montaj.yaml …`.

5. **Preview.** `montaj render` (preview), then `montaj sheet` and look at `build/sheet.jpg`. Fix what the
   sheet shows (framing, order, holds) and re-render; only changed segments re-render. Done when the sheet
   shows the arc you planned, after at most 2 rounds.

6. **Review — zero tokens.** Hand the user two commands for their own terminal:
   `montaj -C <project> review` (a page: play, approve/reject shots, comment at the playhead) and
   `montaj -C <project> watch` (re-renders on every save). When they come back, `montaj review --summary`
   prints one line per noted shot; apply every note. A note that states a lasting preference becomes
   `montaj taste add "<preference>"`.

7. **Final.** `montaj render --final`, then `montaj check`: done on `OK check`. A `WARN` line names a defect to
   fix in the spec; `INFO` lines are informational. For phone sharing add `montaj export --target whatsapp`.
   Report the output paths.
