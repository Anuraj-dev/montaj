# Captions — what good looks like
> For spec authors (agents). Status 2026-11-10: the `captions` block, presets and lints below are planned
> (GitHub #20–#26); this page is their design contract. The numbers are starting defaults from common practice
> on short-form video, not measurements. Tune them against `montaj captions --gallery` and a reference (#26).

A caption on a YouTube video is not a subtitle. A subtitle transcribes; a caption holds attention. It is read in
one glance with the sound off, so it is a few heavy words that change in time with the voice.

## The rules that matter most
1. **One to three words at a time.** Never a sentence. One line; two only when a phrase must stay together.
2. **Heavy type.** Weight 800–900 sans. Thin, light or grey text is invisible on video.
3. **Stroke and shadow, never a box.** Legibility comes from a dark outline and a soft shadow on the glyphs.
   A rectangle behind the caption is the mark of a template.
4. **White plus one accent.** The accent colours the one word that carries the meaning.
5. **No full stop.** No trailing `.` or `,`. Keep `?` and `!`.
6. **It moves on the word.** A chunk appears when its first word is spoken, quickly, and is replaced by the next.
7. **Keep clear of faces and platform UI.**

## Anatomy
| Property | Default (9:16, 1080 wide design px) | Notes |
|---|---|---|
| Words per chunk | 1–3, at most 4 | Break on pauses and punctuation first |
| Characters per line | ≤ 16 | Longer: shrink (`fit`) or split the chunk |
| Font | `montserrat-black`, `anton`, `poppins-extrabold` | Devanagari: `mukta-extrabold` |
| Size | 84–110 (about 5–6% of frame height) | 16:9: 56–72 on a 1920-wide frame |
| Case | Upper for `punch` and `comic`; lower for `clean` and `minimal` | Devanagari has no case |
| Fill | `#ffffff` | Off-white on very bright footage |
| Stroke | 7–9% of the size, `#000000`, outside the fill | Round joins |
| Shadow | y 4–8, blur 8–16, black at 50–70% | Lifts text off mid-tones |
| Line height | 1.05 Latin, 1.4 Devanagari | |
| Tracking | 0 to −0.01 em | Never letter-space Devanagari |

## Position and safe zones
- **9:16:** centre x; baseline region between 60% and 72% of the height (y ≈ 1150–1380 of 1920).
  Keep out of the top 12%, the bottom 22% and the right 14%: platform buttons and titles sit there.
- **16:9:** lower third, at least 8% above the bottom edge; centre, or left-aligned beside a speaker.
- Never across a face. If the speaker fills the frame, move the caption down to chest height, not onto the chin.
- Hold the position for the whole clip. A caption that jumps around is noise; move it only for a designed beat.

## Colour
- Base is white. Pick **one** accent per film and keep it.
- Accent meanings an audience already reads:
  - yellow `#ffe600` — emphasis, the default
  - green `#39e75f` — gain, money, yes, correct
  - red `#ff3b30` — loss, danger, no, wrong
  - cyan `#3ad8ff` — tech, calm, facts
- A second colour is allowed only for a real opposition (green "before" vs red "after").
- One accented word per chunk. If every word is yellow, none is.
- Contrast: the fill against what is behind it should reach about 4.5:1; the stroke guarantees it on mixed
  footage. Do not match the accent to the footage's dominant colour (yellow text on a sunlit wall disappears).

## Motion
| Moment | Default | Avoid |
|---|---|---|
| Enter | `pop`: scale 0.8 → 1.08 → 1 over 4–6 frames, per chunk or per word | Fades longer than 3 frames; slow rises |
| Active word | Accent colour or scale 1.1 switches on the word's start | Smooth colour blends |
| Emphasis word | Scale 1.15–1.3, accent, optional ±3° tilt, a 6–8 frame shake on a hit | Emphasis on every chunk |
| Exit | Cut: the next chunk replaces it | Fade-outs; overlapping chunks |
| Pause > 0.6 s | Clear the caption | Leaving stale words on screen |

Timing: a chunk may lead its word by up to 2 frames, never lag it. Reading speed above about 20 characters per
second is too fast; make chunks smaller.

Vary with intent. Sixty seconds of identical pops goes flat: let the hook, the numbers and the punchline get the
bigger treatment, and keep the rest plain.

## Text
- Drop trailing `.` and `,` (and `।`). Keep `?`, `!`, `…`.
- Numbers as digits with their unit: `₹50,000`, `10x`, `3 days`.
- Do not split a phrase that reads as one unit ("New York", "no way", a number and its unit).
- Fix transcription errors with `captions.overrides`, not by rewriting the track.
- Emoji: at most one per several chunks, only where it replaces a word's meaning.
- Hindi / Hinglish: caption in the script the audience reads; do not mix scripts inside one chunk.

## Presets (#23)
| Preset | Look | Use for |
|---|---|---|
| `punch` | Heavy caps, white, black stroke, one accent word, pop in, cut out | Talking shorts, hooks, business and fitness content |
| `karaoke` | Whole chunk visible, spoken word takes the accent | Songs, lyric videos, slower speech |
| `highlight` | A tight rounded bar slides behind the spoken word | Podcasts, interviews |
| `clean` | Lower-case semibold sans, soft shadow, no stroke | Vlogs, documentaries, calm footage |
| `comic` | Thick stroke, alternating tilt, bounce | Gaming, comedy, reactions |
| `minimal` | Small sans with italic serif emphasis | Cinematic, lifestyle, essays |

`highlight` is the one case where a filled shape sits behind text: it hugs a single word, has rounded ends and
moves with the voice. A static rectangle behind the whole caption is never a preset.

## Hygiene lines (#24)
Facts, not bans. `captions` drops a trailing `.` `,` or `।` from every chunk (`?` `!` `…` stay);
`captions.punctuation: keep` opts out. `subs` drop them only with `punctuation: drop` on the sub (unset keeps the
text and its segment hash, and `validate` warns instead).
- `validate` (spec only), for caption blocks (made by `captions`, or authored: every line in the lower 45%,
  at most 8 words, no `dot`):
  `WARN text[i].lines[j] ends with a full stop`, `WARN caption-box text[i] ...` (chip or solid backdrop),
  `WARN caption-weight text[i] ...` (under 600), `WARN caption-size text[i] ...` (under 5% of the frame height).
- `check` (measured), per caption chunk (numbers match `montaj captions --print`):
  `WARN caption-contrast chunk=N ratio=X` (under 4.5; a stroke of at least 4% of the size counts as the backdrop),
  `WARN caption-safe chunk=N <side>` (zones above; set `check.safe: {top, bottom, left, right}` as fractions),
  `WARN caption-face chunk=N` (needs `uv sync --extra guards`), `WARN caption-rate chunk=N Ncps` (over 20 cps).
  Authored text blocks without per-glyph layout have no ink box, so only the rate is measured for them.
- A designed exception goes under `check.expect: [{caption: [a, b], why: ...}]`; findings inside the span become
  `INFO caption a-bs ignored as intended: why`.
- The `punch` slam never carries the accented word past the 5% edge margin or into its neighbour; the entry
  scale is capped per word (`Motion.slam_from`).

## Slop list
`validate` and `check` warn on most of these (#24).
- A rectangle, pill or card behind the caption.
- Full sentences, or two full lines of text.
- A trailing full stop.
- Light or regular weight; grey or semi-transparent fill.
- Three or more colours; every word accented.
- Slow fades or rises; captions that lag the voice.
- Captions in the bottom fifth of a vertical video, or across a face.
- The same animation on every chunk for the whole film.
- A film subtitle (`subs`) used where a caption was wanted.

## How to look before writing (#26)
1. `montaj captions --gallery` → `build/captions-gallery.jpg` — every preset over this project's own footage (brightest and darkest caption moment), one 1536-px image. Pick by eye, once.
2. `montaj ref <video>` → `build/ref-<stem>.jpg` — a sheet sampled at the cuts of a reference the user points to, and `cut_rate=<cuts per 10 s>` and `median_shot=<s>` on the `OK` line.
3. `montaj captions --print` — review the chunking as text, at no image cost.
4. `montaj captions --strip <chunk>` (chunk index from `--print`) → `build/captions-strip.jpg` — eight frames from the frame before the chunk starts, only when the motion is new.
