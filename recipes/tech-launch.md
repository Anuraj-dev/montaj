# tech-launch: slots and retargeting

`recipes/tech-launch.yaml`: 14 s, 1920x1080 @ 60, 120 BPM (1 beat = 0.5 s, 28 beats), `cool-tech`. Demo:
`montaj new demo --recipe tech-launch` (no `--photos`) runs `recipes/tech-launch-demo.sh demo`, which writes synthetic
"UI" clips, event logs, placeholder voices and a composed bed; its log is `demo/build/new-demo.log`.

## Timeline
| Act | Beats | Time | What |
|---|---|---|---|
| Boot | 0-4 | 0-2 s | blank dark ground, ring + brackets + line HUD, mono typewriter line, voice 1 at 0.9 s |
| Build | 4-16 | 2-8 s | four 3b clip shots A-D, dip/push in, punch-in `camera`, chip caption, whoosh + tick per cut |
| Drop | 16-21 | 8-10.5 s | riser (12b) then boom at `drop`; 2b glitch, 2b zoom + `freeze`, 1b glitch |
| Logo | 21-28 | 10.5-14 s | boom at `logo`, `LUMEN` wordmark + tagline, ring, voice 2, fade to black |

## Slots to fill
Per build shot (A-D = `ui1`..`ui4`): `clip`, `at_event` (or `clip_in`), `camera[].focus` (source px of the UI region),
the chip `text` (and its `from`/`to`/`at`), the `brackets` box (design px) over that region. Drop shots reuse the same
clips (`clip_in` 5 s). Voice: `assets/voice1`, `assets/voice2`. Music: `music/song.wav`. Brand: `LUMEN` and the tagline.

Swap in real recordings by editing only those fields:
1. `montaj ingest-rec raw/voice.mp4 --events raw/voice.events.json` per recording (stem = file name).
2. Point each shot's `clip` at the stem; `at_event: {kind, <field>: value}` picks the moment (change `pre` for lead-in).
   No event log: replace `at_event`/`pre` with `clip_in`.
3. Set `camera` focus to the UI region in source pixels; keep zoom 1.3-1.5.
4. Update chip text and the brackets `box` (design px: x in 0-1080, y in 0-607 for 16:9).
5. Source clips must last hold + transition (3b = 1.5 s) after the event; the drop shots read from `clip_in: 5s`.
Timing, transitions, SFX and the `check.expect` spans stay as they are; if you move the freeze (`freeze: 0.5s` on the
first 2b drop shot, so 9.5-10 s) or the end fade, move the matching `check.expect` span.

## Retarget to 9:16 (1080x1920)
Set `video.size: 1080x1920`. Screen values are design px of a 1080-wide frame, so the frame is now 1080x1920:
- Per clip shot add `crop: [x0, 0, x0 + 608, 1080]` (a 9:16 window of a 1920x1080 recording, centred on the feature) and
  set `camera[].focus` inside it. Examples for the demo layout: A `crop [1000,0,1608,1080]` focus `[1330,330]`;
  B `[256,0,864,1080]` focus `[560,620]`; C `[656,0,1264,1080]` focus `[960,260]`; D `[1146,0,1754,1080]` focus `[1450,760]`.
- Overlays: ring centre `[540,700]`, brackets/line boxes move to the same region in the 1080x1920 frame.
- Text: chip `y: 1500`, wordmark `y: 760` size 130, tagline `y: 960`, boot line `y: 1000`; keep `size >= 40` chips.
The audio, beats and `check.expect` are unchanged.
