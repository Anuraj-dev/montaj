gaming-highlight: 16:9 30 s match highlight from an event log, freeze-and-zoom moment, impact hits, comic captions.

# gaming-highlight: slots and retargeting

`recipes/gaming-highlight.yaml`: 30 s, 1920x1080 @ 60, 120 BPM (1 beat = 0.5 s, 60 beats), `cool-tech` look.
Every gameplay shot takes its in-point from the recording's event log (`shots[].at_event` + `pre`), so swapping in
another match is an edit to the events, not to timecodes. Demo: `montaj new demo --recipe gaming-highlight` (no
`--photos`) runs `recipes/gaming-highlight-demo.sh demo`, which writes a synthetic 60 s match (`assets/match.mp4`)
with its event log through `montaj ingest-rec --events`, the caption words (`words.json`) and a drum bed
(`music/song.wav`); its log is `demo/build/new-demo.log`.

## Timeline
| Part | Beats | Time | What |
|---|---|---|---|
| Hook | 0-2 | 0-1 s | the key moment first, 0.5x with a chroma split; "WATCH THIS" |
| Run | 2-26 | 1-13 s | round start, then five kills, 4b each; each kill lands 2b in (`pre: 2b`) on markers `hit1`..`hit5` (`shown:match/kind=kill#N`) with a shake + chroma hit and `synth:hit` |
| Moment | 26-48 | 13-24 s | build-up cut on the kill (`pre` = `hold`); at `drop` a white dip, `freeze` 6f in, camera push to 1.85x, shake/chroma/bloom, boom + bass drop; 0.4x replay with trails at `replay`; a second look at `replay+6b` (its kill at `replay+6b+0.5s`) |
| Out | 48-60 | 24-30 s | round won (`win`), then GG over a blurred frame 2 s after the win (`pre: -2s`, `gg`), 1 s fade to black |

`audio.sfx: auto` adds whooshes on the whip/zoom transitions and a riser peaking at `drop`.

## Ask the user (one round)
1. The recording and its event log (game API / overlay / OBS plugin output). Convert it to the event JSON list
   (`[{"t": seconds-from-record-start, "kind": "...", ...fields}]`) and run
   `montaj ingest-rec raw/match.mp4 --events raw/match.events.json`. Then `montaj sheet` to see it.
2. Which event is the key moment (the freeze) and which kills make the run (default: the first five).
3. Captions: a commentary/voice track (`montaj transcribe <clip>`, then `captions.source: <clip>`) or reaction
   words they type (a words JSON like the demo's). Accent colour (default yellow).
4. Music: their track (`audio.track`; drop `video.bpm` and add `audio.markers` from `montaj music analyze`), or a
   `music gen` bed. Length (20-45 s).

## Slots to fill
- Every gameplay shot: `at_event` (`{kind, <field>: value, contains: {...}, nth}` against
  `assets/<clip>.events.json`; a miss lists the values present) and `pre` (how much lead-in before the event).
  The demo log has `kind: round` (`phase: start|end`, `result`) and `kind: kill` (`streak`, `weapon`, `label: ace`
  on the key one). Real logs differ: match on whatever fields they carry.
- `hitN` markers are `shown:match/kind=kill#N`: the kill on the film clock, through the shot whose `at_event` picked
  it. Change a `pre`, a hold or a `speed` and the marker (with its effects and `synth:hit`) moves with the shot;
  pick different kills by changing the shot's `at_event` and the marker's selector together.
- Freeze shot: `freeze` is when the picture stops (just after the kill flash), `camera` `focus` is what to punch
  in on, in source px (demo: `[800, 320]` holds the target and the ACE banner 6f after the kill). Find it with
  `montaj sheet`; the camera push keeps moving over the frozen frame, so `check` does not see a stall.
- Replay: `speed` x hold must stay inside the clip (`ERR shots[i]: clip ends at …` says by how much); the kill lands
  at `pre / speed` into the shot, so `replay+2s` follows `pre: 0.8s` at `0.4`.
- Last shot: the win event with a negative `pre` (`-2s`: start 2 s after it), any calm frame under the blurred GG
  title; it must still start inside the clip.
- Effect strength: shake px and chroma split ramp up toward the key moment; keep that order when retuning.

## Pacing targets (`check.pacing`)
`max_shot: 4s` (the freeze is the longest, 8b), `min_per10s: 3`, `hook_text_by: 0.5s` (the first caption is at
0.05 s). Keep them when retargeting.

## Retarget to 9:16 (1080x1920)
Set `video.size: 1080x1920`. Per gameplay shot add `crop: [x0, 0, x0 + 1080·h/1920, h]` around the crosshair
(source px; `h` = clip height), and recentre the freeze `focus` inside it. Captions move to the 9:16 safe band by
themselves; set the GG line `y: 900`. Timing, events, effects and audio are unchanged.
