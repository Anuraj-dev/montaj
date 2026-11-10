vlog-montage: 16:9 60 s three-chapter vlog, beat cuts, speed ramps, L cuts into music, per-shot grades, film look.

# vlog-montage: slots and retargeting

`recipes/vlog-montage.yaml`: 60 s, 1920x1080 @ 30, 100 BPM (1 beat = 0.6 s, 100 beats), `warm-film` look,
`video.color.working: linear`. Demo: `montaj new demo --recipe vlog-montage` (no `--photos`) runs
`recipes/vlog-montage-demo.sh demo`, which writes eight synthetic 10 s clips with ambience (`assets/c1`..`c8`) and a
composed bed (`recipes/music/vlog-montage-compose.py` → `music/song.wav`); its log is `demo/build/new-demo.log`.
Film-emulation fields (`look.film`) are not on `main` yet; the film look is the `warm-film` preset plus `fx` leaks.

## Timeline
| Chapter | Beats | Time | What |
|---|---|---|---|
| Morning | 0-32 | 0-19.2 s | fade up on an 8b linked shot (its own sound, music ducked), L cut 2b into the music; 2b/4b cuts, a 1→3→1 ramp, a 0.5x shot |
| The Road | 32-64 | 19.2-38.4 s | dip in, 6b linked shot + L cut, mostly 2b cuts, a 1→3.5→0.5 ramp, a whip at 48b with a whoosh |
| Golden Hour | 64-100 | 38.4-60 s | 8b linked shot over the music breakdown + L cut, ramps, an 8b 0.5x close, sign-off line, 1 s fade |

Chapter titles: a `caps` kicker and an italic serif title for ~3.5 s at `ch1`/`ch2`/`ch3`; sign-off at `sign_off`.

## Ask the user (one round)
1. Clip folder (`montaj ingest` it, then `montaj sheet`); which clips are the chapter openers (they keep their sound).
2. Chapter names (default three; two or four also work) and the sign-off line.
3. Music: their track (set `audio.track`; drop `video.bpm` and add `audio.markers` from `montaj music analyze` so cuts
   follow the song), `music gen`, or a composed bed. Length (45-90 s).
4. Look: warm/golden/cool. Maps to the `grades:` table and `video.look`.

## Slots to fill
- Per shot: `clip` (stem in `assets/`), `clip_in` (where the action is), `hold` (keep whole beats: 2b/4b, 6b-8b only on
  the linked openers), `grade` (a name from `grades:`, or an inline mapping for a one-off fix).
- Chapter openers (`audio: true`, `id: scene`, `audio_tail: 2b`): pick clips with good natural sound. `gain` evens
  their level; `audio_tail` is the L cut (at most the next shot's hold). The music is ducked under `scene`.
- Speed ramps: `speed: [{t, v, ease}]` keys on the shot clock. The source must cover the integral of the speed
  (`ERR shots[i]: clip ends at …` says by how much); move `clip_in` earlier or lower the peak.
- `markers` `ch2`/`ch3`/`sign_off`: move them with the chapter boundaries; titles, leaks and the whoosh follow.
- `grades`: `dawn`, `day`, `road`, `golden`, `dusk`. Tune per real footage: `exposure` (stops) and `temp`/`tint` first,
  then `contrast`, `saturation`. Edit one grade → only the shots using it re-render.

Swapping footage: edit only `clip`, `clip_in`, `grade` and, if a clip is short, the ramp keys. Total holds must stay
100b for 60 s; change `markers.end`, the last shot's hold and the music length together for another length.

## Pacing targets (`check.pacing`)
`max_shot: 5s` (the 8b openers are 4.8 s), `min_per10s: 3`, `hook_text_by: 1s` (the chapter-one title appears at
0.4 s). Keep them when retargeting; a longer film keeps the same targets.

## Retarget to 9:16 (1080x1920)
Set `video.size: 1080x1920`. The frame is now 1080x1920 design px:
- Per shot add `crop: [x0, 0, x0 + 1080·h/1920, h]` around the subject (source px; `h` = clip height), or
  `focus: subject` after `montaj ingest`.
- Titles: kicker `y: 820`, title `y: 860`; sign-off `y: 900`, `size: 56`.
- Shorten to ~40 s: drop the middle chapter (its shots and title), delete `ch2` and its leak, and move `ch3`,
  `sign_off` and `end` 32b earlier (the composed bed is 60 s; `audio.edit` `cut: [32b, 64b]` trims it to match).
Audio, grades and ramps are unchanged.
