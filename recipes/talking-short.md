# talking-short: slots and retargeting

`recipes/talking-short.yaml`: ~36 s, 1080x1920 @ 30, `clean` look. One presenter clip (`assets/voice.*`) cut on its
transcript, `punch` captions, word-anchored zoom/shake/flash + SFX (`emphasis` + `audio.sfx: auto`), two b-roll inserts
built with `layers`, a hook title on screen from frame 0, and `check.pacing` targets (shot <= 4.5 s, >= 3 cuts per
10 s, text by 1 s). Demo: `montaj new demo --recipe talking-short` (no `--photos`) runs `recipes/talking-short-demo.sh`,
which writes a synthetic presenter + voice, its word timings and two generated b-roll clips, then runs
`montaj cut voice`; log: `demo/build/new-demo.log`. Captions and emphasis read `assets/voice.words.json`; the engine
moves each word to where the cut film plays it.

## Ask the user (one round)
- Footage folder: the talking-head clip (any name; the recipe calls it `voice`) and 0-3 b-roll clips.
- Length target (30-45 s for a short) and which lines to drop if it runs long.
- Hook: 2 short lines (<= 4 words each), shown 0-1.6 s. No trailing full stop.
- Emphasis words: which words get a zoom, shake, flash or SFX (3-6 total; more turns into noise).
- Where each b-roll goes (which sentence) and inset card or full frame.
- Caption style: `punch` (default), or another preset from `docs/SPEC-REFERENCE.md`.

## Slots to change
| Slot | Where | Notes |
|---|---|---|
| presenter clip | every `clip: voice` | rename the file to `assets/voice.<ext>` or change the stem everywhere |
| cut spans per shot | `drop:` on each `cuts: auto` shot | 1-based span numbers from `montaj cut voice --print` |
| b-roll inserts | the `hold:` + `layers:` shots | voice layer `clip_in` = span start - 0.06 s, `hold` = span end + 0.12 s - `clip_in` (seconds from `montaj cut voice --print`, in frames) |
| voice under b-roll | `audio: true` on the voice layer | its sound plays locked to that layer's picture |
| hook | `text[0].lines` | keep `from: 0s`; `*word*` takes the `emphasis` colour |
| emphasis | `emphasis[].match` | words as spoken; `sfx` = a `synth:` name from `audio.clips[].src` in `docs/SPEC-REFERENCE.md` |
| pacing | `check.pacing` | loosen `max_shot` for a slower talk; keep `hook_text_by: 1s` |

## Real footage
1. Copy the clip to `assets/voice.mp4` (and b-roll to `assets/broll1.mp4`, ...).
2. `montaj cut voice --lang en` (transcribes, writes `assets/voice.cuts.json`; `--print` lists the spans with text).
3. Choose the layout: which spans play as presenter jump cuts and which become b-roll shots; set each shot's `drop`.
4. For each b-roll span set the voice layer's `clip_in` and the shot's `hold` from its `--print` times (table above;
   the pad stops halfway to a neighbouring span), and keep `audio: true` on that layer.
5. `montaj render`, `montaj check`, `montaj sheet`.
