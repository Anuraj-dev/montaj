# Spec reference

Photo fields (`crop`, `focus`, swirl `center`, wall `at`) are source-image pixels. `zoom`, the wall grid and leak size are design pixels of a 1080-wide frame, so a half-size preview matches the final framing. Durations are `22f` (frames), `4b` (beats on the grid) or `1.5s`; a bare number is an error. The beat grid is `video.bpm` plus a phase `beat0` (see below). A plain beat time (`at: 8b`, `until: 16b`, a marker defined as `8b`) means `beat0 + 8` beats; durations (`hold`, `dur`, `stagger`, `in.dur`) and offsets (`ch1+2b`) are spans. A time expression is a duration from film start or a marker, then any number of `±duration` offsets in any units (`ch1`, `ch1+2.4s`, `pre2-6f`, `replay+6b+0.5s`, `8b+2f`). Marker names are `[a-z][a-z0-9_]*`. Marker values may be a duration, `event:<clip>/<field>=<value>[,<field>~=<substring>][#n]` (time of that event in `assets/<clip>.events.json`, same contract as `at_event`), `shown:<clip>/…` (same selector; that event on the film clock, through the first shot whose `at_event` picked it and shows it: shot start + (event − clip_in)/speed; an error if no shot shows it, if that shot has keyed `speed`, or if a shot's `until` uses it), `word:<i>` (start of word `i` in `audio.markers`), or another marker plus an optional duration offset; cycles are an error.

- `video.size` — string, required — output `WxH`, both numbers even. Preview halves each side, rounded down to even.
- `video.fps` — integer, required — frames per second.
- `video.bpm` — number, default none — beats per minute. If omitted, `audio.markers` `bpm` sets the grid (beats follow the analysed song). Needed, one way or the other, when any duration uses `b`.
- `video.beat0` — duration (`f` or `s`), default see text — time of the first beat. Default: `0s` when `video.bpm` is set; else `beats[0]` from `audio.markers`.
- `video.look` — `warm-film`, `golden-film`, `clean`, `cool-tech` or a mapping with `lut`, required — finish preset applied to every frame. See looks below.
- `video.look.lut` — path to a `.cube` file, required in the mapping form — relative to the spec file. 3D up to 65 points per side, or 1D. The file bytes are part of the cache key.
- `video.look.strength` — number 0..1, default `1` — mix from the ungraded picture (`0`) to the LUT result (`1`).
- `video.look.vignette` — number 0..1, default `0` — edge darkening, applied after the LUT. `0.1` is subtle.
- `video.look.grain` — number 0..1, default `0` — film grain mix; `1` is the golden-film grain.
- `video.background` — `#rrggbb`, default `#000000` — colour that shows through a fade from or to nothing.
- `video.color` — mapping, default none — colour pipeline. Absent is display mode, unchanged from before.
- `video.color.working` — `display` or `linear`, default `display` — `linear`: each shot is converted sRGB→linear on the GPU, then graded; effects, transitions (fades, dips, blurs, glows) and the motion-blur average run in linear light; the composite is converted back to sRGB before `look`, overlays and text, which stay display-referred. Fixes the dark mid-fade dip between bright shots. Order: input transform → `shots[].grade` → effects/transitions → output transform → `look` → text.
- `video.intro` — mapping, default none — opening treatment on the first frames.
- `video.intro.type` — `white-lift`, required with intro — a white bloom that settles onto the picture, plus a short lift from white.
- `video.intro.dur` — duration, required with intro — how long the opening bloom lasts.
- `video.outro` — mapping, default none — ending treatment. Starts at `n_frames - dur`.
- `video.outro.type` — `glow` or `fade`, required with outro — `glow`: a bloom and warm rim that grow and then stay. `fade`: everything fades to black over the last `dur`.
- `video.outro.dur` — duration, required with outro — how long the ending lasts.
- `video.motion_blur` — integer, default `1` — sub-frames averaged into each frame. `1` is off. Preview uses at most 2. Text and subtitles are composited once at the integer frame, after that average.
- `assets` — string, default `assets` — photo folder relative to the spec file. A photo is the file stem (`jpg`, `jpeg`, `png` or `webp`).
- `shots` — list, required — played in order. Each shot starts when the previous one ends.
- `shots[].photo` — string, one source: photo, wall, clip, blank, sequence or card — stem of the still. An integer in the YAML is read as a string. Mutually exclusive with the other sources.
- `shots[].clip` — string, one source: photo, wall, clip, blank, sequence or card — video source. A stem (`intro`) is a file in `assets/` (`mp4`, `mov`, `webm`, `mkv`). A directory or a video suffix (`../old/out/final.mp4`) is a path relative to the spec file, or absolute. Mutually exclusive with the other sources. `check` warns `shots[i] <stem> runs <t>s past its clip` when the hold plus the outgoing fade, swirl, or half a whip, after `clip_in`, reads past the file.
- `shots[].matte` — `subject`, default none — on a clip shot: its subject matte (from `montaj ingest --matte <stem>`) is the plane `text[].behind` draws under. A missing matte is a `validate` error naming the ingest command.
- `shots[].clip_in` — duration, default `0s` — source offset into the clip. Mutually exclusive with `range`.
- `shots[].at_event` — mapping, default none — pick `clip_in` from `assets/<clip>.events.json`: `{kind, <field>: value, contains: {<field>: substring}, nth}`; `nth` is 1-based among matches. A miss is an error listing the values present. Needs `clip`; exclusive with `clip_in` and `range`. The resolved time is written into the shot, so changing `nth` re-renders only that shot.
- `shots[].pre` — signed duration, default `0s` — start this long before the `at_event` time; negative (`-1s`) starts after it. An error if that is before the clip start or at/after its end. Needs `at_event`.
- `shots[].range` — `[start, end]` durations or `speech:N`, default none — source span of `clip` played as this shot. `speech:N` is the N-th (1-based) loud span in `assets/analysis.json` (assets clip stems only), resolved before hashing. Needs `clip`. Shot length is `end − start` and does not snap to the beat grid. Playback stops at `end`: an outgoing transition holds the last frame. The file needs a video stream and `start` must fall inside it. Mutually exclusive with `hold`, `until` and `clip_in`.
- `shots[].camera` — list of keys, default none — keyframed punch-in on a clip, on the shot clock (`t` from the shot's start, fade-in included). Needs `clip`. Mutually exclusive with `focus` and `zoom`.
- `shots[].camera[].t` — duration, required — when this key is reached. Keys run in increasing time; before the first key the camera holds it, after the last it holds that.
- `shots[].camera[].zoom` — number >= 1, optional — multiple of cover: `1` shows the whole crop, `1.35` is a 35% punch-in. Leave it off to hold the previous zoom (first key: `1`). Moves geometrically between keys.
- `shots[].camera[].focus` — `[x, y]` or `face`/`face:N`/`subject`, optional — source-video pixel placed at the screen centre, kept inside the crop's cover so no border shows. Leave it off to hold the previous focus (first key: the crop centre).
- `shots[].camera[].ease` — easing name, default `smooth` — how the camera travels into this key.
- `shots[].speed` — number > 0 or list of `{t, v, ease}` keys, default `1` — playback speed of a clip. `2` plays twice as fast. Keys ease the speed (`v`) like a camera track; the source position is its integral at frame resolution. Needs `clip`; mutually exclusive with `range`. The source must last the hold (`ERR shots[3]: clip ends at 4.1s of source, needs 5.2s`). No reverse.
- `shots[].freeze` — duration, default none — from this time in the shot to its end the clip shows the source frame it had here. Needs `clip`; must fall inside the hold.
- `shots[].audio` — boolean, default none — `true` mixes the clip's own sound in, locked to its picture: it follows `clip_in`/`range`, `speed` (varispeed, so pitch moves with it) and goes silent at `freeze`. Needs `clip` with an audio stream and an `audio` block. 15 ms fades at each audio cut. Audio fields never re-render pixels.
- `shots[].gain` — dB string, default `0dB` — level of the shot's own sound. Needs `audio: true`.
- `shots[].audio_lead` — duration (`ms`, `s`, `f`, `b`), default none — J cut: the sound starts this early, over the previous shot, reading the source before `clip_in`. Needs `audio: true`; not on the first shot; at most the previous shot's hold; not with an `audio_tail` on a linked previous shot.
- `shots[].audio_tail` — duration, default none — L cut: the sound runs this long past the cut, under the next shot (replacing a linked next shot's sound there). Needs `audio: true`; not on the last shot; at most the next shot's hold.
- `shots[].cuts` — `auto`, default none — talking-head cuts: expands at load into one `range` shot per kept span of `assets/<clip>.cuts.json` (`montaj cut`), linked audio joined by a 20 ms crossfade. Captions and emphasis reading that clip's `assets/<clip>.words.json` follow the cut layout: each word lands where the film plays it, words in removed spans drop.
- `shots[].cuts_pad` — `[before, after]` durations, default `[60ms, 120ms]` — kept before / after each span, never into the cut gap. Needs `cuts`.
- `shots[].cuts_punch` — `alternate`, default none — every second span zooms 1 / 1.12 so jump cuts read as intent. Needs `cuts`.
- `shots[].drop` — span indexes, default none — spans removed after reading `montaj cut --print`. Needs `cuts`.
- `shots[].id` — `[a-z][a-z0-9_]*`, default none — names the shot's own sound for `audio.duck.under` and waveform `overlays[].source`, like `audio.clips[].id`. Needs `audio: true`.
- `shots[].hold` — duration, hold, until or range — how long this shot is on screen. Exactly one of `hold`, `until` or `range`. A whole-beat hold (`4b`) ends on a beat of the grid (moving forward if the shot started off-beat), and beat holds never drift from rounding.
- `shots[].until` — time expression, hold or until — the shot ends at this time from film start. Must fall after the shot's start.
- `shots[].drift` — mapping, default none — Ken Burns zoom/pan on a photo. Needs `photo`. Mutually exclusive with `zoom`.
- `shots[].drift.zoom` — `[from, to]`, default `[1, 1]` — scale over the shot's visible span.
- `shots[].drift.pan` — `[x, y]`, default `[0, 0]` — pan in design px; the image sits at `pan·(k − .5)`.
- `shots[].tone` — mapping, default none — CSS filter on the photo, order brightness → saturate → contrast. Needs `photo`.
- `shots[].tone.brightness` — number, default `1` — CSS `brightness`.
- `shots[].tone.saturate` — number, default `1` — CSS `saturate`.
- `shots[].tone.contrast` — number, default `1` — CSS `contrast`.
- `shots[].grade` — mapping or a `grades:` name, default none — primary grade on this shot's picture (any kind but `blank`), applied in linear light before its effects and transitions, in this order: exposure → temp/tint → contrast → lift/gamma/gain → saturation → vibrance → curves. Works in both `video.color.working` modes. A named grade is part of each using shot's cache key, so editing it re-renders only those shots.
- `shots[].grade.exposure` — number −8..8, default `0` — stops; `1` doubles the light.
- `shots[].grade.temp` — number −4000..4000, default `0` — white balance in kelvin from 6500 K; `+` is warmer. Gains keep a grey's luminance.
- `shots[].grade.tint` — number −100..100, default `0` — `+` is magenta, `−` green.
- `shots[].grade.contrast` — number > 0, default `1` — power curve about `pivot` in linear light.
- `shots[].grade.pivot` — number 0..1, default `0.46` — display value contrast leaves fixed; `0.46` is 18% grey.
- `shots[].grade.lift` — number or `[r, g, b]`, −1..1, default `0` — raises shadows: `y = gain·(x + lift·(1 − x))` on display values.
- `shots[].grade.gamma` — number or `[r, g, b]`, > 0, default `1` — midtones: `y^(1/gamma)` after lift and gain; above 1 brightens.
- `shots[].grade.gain` — number or `[r, g, b]`, 0..8, default `1` — scales highlights.
- `shots[].grade.saturation` — number 0..4, default `1` — about Rec.709 luminance; `0` is monochrome.
- `shots[].grade.vibrance` — number −1..1, default `0` — saturation that favours muted pixels.
- `shots[].grade.curves.luma` — `[[x, y], …]`, default none — monotone cubic on display-encoded luminance, ≥ 2 points in 0..1 with x increasing; hue is kept. Beyond the end points it continues at slope 1.
- `shots[].grade.secondaries` — list, default none — after the primary, each entry blends `grade(picture, adjust)` in by its mask: key × window × matte (the ones given), then `invert`.
- `shots[].grade.secondaries[].key` — mapping, default none — HSL qualifier on display values: `hue: [lo, hi]` degrees 0..360 (`[340, 20]` wraps), `sat`/`luma: [lo, hi]` 0..1; at least one.
- `shots[].grade.secondaries[].key.soft` — number 0..1, default `0.05` — soft edge as a fraction of each scale (hue: ·360).
- `shots[].grade.secondaries[].window.ellipse` — mapping, default none — `center: [x, y]` design px, `subject` (matte centroid, per frame) or `face` (largest ingest face box at the shot start, mapped through the camera; no boxes → subject centroid with a `validate` warning); `radius: r | [rx, ry]` design px; `feather` design px (default `0`), centred on the radius; `offset: [dx, dy]` design px.
- `shots[].grade.secondaries[].window.gradient` — mapping, default none — `from: [x, y]`, `to: [x, y]` design px; 1 at `from`, 0 at `to`. A window has exactly one of ellipse or gradient.
- `shots[].grade.secondaries[].matte` — `subject`, default none — the shot's subject matte; needs `matte: subject` on the shot (or a masked layer). `center: subject|face` needs it too.
- `shots[].grade.secondaries[].invert` — bool, default `false` — selects everything the mask does not.
- `shots[].grade.secondaries[].adjust` — mapping, required — any primary grade fields (exposure … curves).
- `shots[].grade.protect` — `skin`, default none — keeps skin-tone chroma through the stock and split tone (luminance still changes). Does not undo secondaries.
- `shots[].grade.look.film.stock` — `warm-negative` | `cool-negative` | `print` | `bleach-bypass`, default none — a fixed primary (toe/shoulder curve, crossover) applied after the secondaries.
- `shots[].grade.look.film.halation` — number 0..1, default `0` — red-weighted glow around highlights above display 0.5.
- `shots[].grade.look.film.bloom` — number 0..1, default `0` — wide neutral glow from highlights.
- `shots[].grade.look.film.grain` — `{amount 0..1, size 0.5..4 design px (default 1)}`, default none — monochrome grain in linear light, strongest in midtones, none in black; seeded per output frame.
- `shots[].grade.look.film.gate_weave` — number 0..1, default `0` — smooth sub-pixel frame wander (up to 3 design px).
- `shots[].grade.look.split_tone` — `{shadows, highlights: "#rrggbb", balance −1..1 (default 0), amount 0..1 (default 0.3)}`, default none — tints shadows and highlights while keeping each pixel's luminance.
- `grades` — mapping of name → grade, default none — reusable grades; `shots[].grade: warm-day` uses one. Names are lowercase letters, digits, `-` and `_`.
- `shots[].frame` — mapping, default none — polaroid card on a blurred copy of the photo. Needs `photo`.
- `shots[].frame.caption` — string, default none — Great Vibes caption on the card.
- `shots[].frame.tilt` — `[from, to]` degrees, default none — card rotation over the visible span.
- `shots[].morph` — mapping, default none — a second photo blooms in as a circle. Needs `photo`.
- `shots[].morph.photo` — string, required with morph — stem of the arriving photo.
- `shots[].morph.at` — duration, required with morph — when the morph starts, from this shot's start.
- `shots[].morph.dur` — duration, required with morph — how long the circle takes to open.
- `shots[].morph.center` — `[x, y]`, default none — screen fraction where the circle is centred.
- `shots[].pulse` — `heartbeat`, default none — beat-locked scale pulse on the grid tempo, phased from the shot start. Needs `photo` and a grid (`video.bpm` or `audio.markers`).
- `shots[].blank` — boolean, default false — show only `video.background`. Mutually exclusive with the other sources. Holds and fades like any other shot.
- `shots[].sequence` — mapping, one source: photo, wall, clip, blank, sequence or card — cycle photos or colours across this shot. Mutually exclusive with the other sources.
- `shots[].sequence.every` — duration (`f` or `s`), required — how long each item stays. Rounds to a whole frame of at least 1. Beats and booleans are an error. `check` treats each step as a designed hold but still flags a freeze that runs across an item change.
- `shots[].sequence.items` — list, required — at least two. A photo stem, a quoted `#rrggbb`, `{photo: stem}` or `{color: "#rrggbb"}`. The list repeats until the shot ends.
- `shots[].card` — mapping, one source: photo, wall, clip, blank, sequence or card — a solid colour and text as this shot's picture, with no blank shot and no `text` block. Mutually exclusive with the other sources.
- `shots[].card.color` — `#rrggbb`, required — the card's background.
- `shots[].card.text` — list, required — lines with the same fields as `text[].lines[]`. `at`, `stagger`, `dur` and `sweep` are durations from this shot's start (`0s`, `12f`, `1b`), not film markers. The colour and the lines are one picture, so a fade or whip carries both.
- `shots[].custom` — path to a `.py` file, relative to the spec, one source: photo, wall, clip, blank, sequence, card or custom — the shot is drawn by that file's `draw(ctx)` on the GPU. Mutually exclusive with the other sources; `crop`, `focus` and `zoom` do not apply. **A spec with `custom:` executes that code in-process with its author's trust (like `music compose`); Montaj does not sandbox it.** See custom code.
- `shots[].params` — mapping, default none — values handed to the shot's `draw` as `ctx.params`. Needs `custom`. Part of the segment hash.
- `shots[].effects` — list, default none — effects applied in order to this shot's picture (any source except blank), before the look; each entry is a built-in `type` or a `custom` file. A transition carries each side's effects.
- `shots[].effects[].custom` — path to a `.py` file, one of `custom`/`type` — `draw(ctx)` gets the shot picture as `ctx.inputs[0]` and returns the new picture.
- `shots[].effects[].params` — mapping, default none — `ctx.params` for a `custom` effect.
- `shots[].effects[].type` — `shake` · `blur` · `bloom` · `chroma` · `trails`, one of `custom`/`type` — a built-in effect; amount 0 is off.
- `shots[].effects[].amount` — number or `[{t, v}, …]` keyframes (`t` from the shot start, linear, held at the ends), default per type — strength: shake peak px (12, max 200), blur sigma px (8, max 200), bloom gain (0.6, max 4), chroma split (0.3; 1 = 32 px at the corner, max 4), trails mix (1, max 1). Px are design px.
- `shots[].effects[].at` — time or marker (film clock), default none — a hit: full strength at `at`, falling linearly to off over `dur`; off outside.
- `shots[].effects[].dur` — duration, default `8f` — length of the `at` hit; needs `at`.
- `shots[].effects[].seed` — int, default `0` — shake only: which shake pattern (a pure function of seed and time).
- `shots[].effects[].threshold` — number 0 to below 1, default `0.75` — bloom only: luma above which pixels glow.
- `shots[].effects[].radius` — design px above 0 to 400, default `24` — bloom only: glow spread (sigma).
- `shots[].effects[].frames` — int 1 to 16, default `4` — trails only: how many earlier frames echo (re-rendered, so a segment's first frame matches the whole film).
- `shots[].effects[].decay` — number above 0 to 1, default `0.6` — trails only: each older echo's weight relative to the next newer one.
- `shots[].layers` — list, default none — several sources composited back to front over `video.background`; replaces the shot's single source and its `crop`/`focus`/`zoom`/`camera`/`speed`/`freeze`. A transition carries the whole stack. One full-frame clip layer renders bit-identical to a clip shot.
- `shots[].layers[].clip` — string, one of clip or image — a video stem. Without `from` it starts with the shot, so it plays through the incoming transition like a clip shot. The source must last the layer's span.
- `shots[].layers[].image` — string, one of clip or image — a PNG (alpha honoured, decoded on the GPU) or JPEG stem.
- `shots[].layers[].audio` — boolean, default none — mixes the layer clip's own sound locked to its picture, from `from` (or the shot's audio cut) to `to` (or the next one), as `shots[].audio`; crossfades with linked neighbours. Needs `clip` with an audio stream and an `audio` block.
- `shots[].layers[].clip_in` — duration, default `0s` — source time shown at the layer's first frame. Needs `clip`.
- `shots[].layers[].focus` — `[x, y]` source px, default the source centre — the point kept centred when `cover` crops the source to the box.
- `shots[].layers[].from` — duration, default the shot's start — when the layer appears, from this shot's start. Must fall inside the hold.
- `shots[].layers[].to` — duration, default past the shot's end — when the layer disappears. After `from`, at most the hold.
- `shots[].layers[].box` — `[x0, y0, x1, y1]` design px or list of keys `{t, v: [x0, y0, x1, y1], ease}`, default the full frame — where the layer sits. Fully off-frame is an error, partly off-frame a `validate` warning. Mutually exclusive with `pos`. A keyed box (keys or `expr`) re-crops the source to each rect's aspect (cover), so content is never stretched; `fit: contain` needs a fixed box.
- `shots[].layers[].fit` — `cover` or `contain`, default `cover` — how the source fills the box: crop to fill, or fit whole inside it. Not with `pos`.
- `shots[].layers[].pos` — `[x, y]` or list of keys `{t, v: [x, y], ease}`, default none — centre of the layer at its natural size (1 source px = 1 design px). Keys fly it from the first key's point (the placement) along the eased path. Mutually exclusive with `box`.
- `shots[].layers[].scale` — number > 0 or list of keys `{t, v, ease}`, default `1` — size multiple about the layer's centre. Keys hold the first value before and the last after.
- `shots[].layers[].rotate` — degrees or list of keys `{t, v, ease}`, default `0` — clockwise rotation about the layer's centre.
- `shots[].layers[].opacity` — number 0 to 1 or list of keys `{t, v, ease}`, default `1` — layer alpha, multiplied with a PNG's own alpha. An overshooting ease is clamped to 0..1.
- `shots[].layers[].<prop>.expr` — string, or a list (2 for `pos`, 4 for `box`), instead of a value or keys, on `pos scale rotate opacity box radius shadow` — computed per frame. Safe subset only: numbers, `+ - * / % **` (exponent at most 64), unary minus, comparisons, `a if c else b`, and `t` (shot seconds), `frame`, `fps`, `beat` (beats on the grid, needs `bpm`), `bpm`, `word(N)` (shot-clock seconds of audio word N's start, needs `audio.markers`), `sin cos abs min max clamp(x, lo, hi) lerp(a, b, x) noise(seed, x)` (smooth, deterministic, -1..1). Anything else is `ERR ...expr: 'foo' ...` naming the token. `box` with `expr` needs `fit: cover`.
- `shots[].layers[].keys[].on` — `beat` or `word:N`, optional on any layer key — anchors the key: `beat` snaps the shot start plus `t` to the nearest beat (needs `bpm`), `word:N` is word N's start plus `t` (needs `audio.markers`). `t` is then an offset, default `0f`.
- `shots[].layers[].id` — name, optional — lets other layers of the shot name this one as a `parent`. Unique in the shot.
- `shots[].layers[].parent` — layer `id`, optional — the layer follows its parent's scale, rotate, opacity and move, about the parent's centre (a pivot `anchor` counts). No loops.
- `shots[].layers[].anchor` — `[dx, dy]` design px from the layer's centre, optional — pivot for `scale` and `rotate`, default the centre.
- `shots[].layers[].<prop>[].t` — duration, required — when the key lands, from the shot's start (`0f`, `10f`, `0.5s`, `2b`). Strictly increasing, inside the hold; whole frames after resolve.
- `shots[].layers[].<prop>[].ease` — easing name, default `smooth` — how the value travels into this key. Core: `linear smooth cubic outc inc expo ramp sine`. Keyframe easings, valid on every `ease` (camera, speed excluded, wall, tracks, overlays, layers; closed form, may leave 0..1): `back-in`, `back-out`, `bounce`, `elastic`, `spring` (or `spring(stiffness, damping)`, default `spring(100, 10)`: 16% overshoot, settles inside the key), `cubic-bezier(x1, y1, x2, y2)` (CSS; x1, x2 in 0..1), `steps(n)`.
- `shots[].layers[].blend` — `normal`, `screen`, `add`, `multiply`, `overlay`, `soft-light` or `difference`, default `normal` — how the layer mixes with what is below (W3C formulas, in display sRGB).
- `shots[].layers[].radius` — design px >= 0 or list of keys `{t, v, ease}`, default `0` — antialiased rounded corners on the layer's rectangle.
- `shots[].layers[].shadow` — number 0 to 1 or list of keys `{t, v, ease}`, default `0` — soft drop shadow under the layer's rectangle, this strong.
- `shots[].layers[].in` — `{type, dur}`, default none — how the layer arrives at `from`: `fade` (opacity) or `pop` (overshoot scale and quick fade). `dur` is a duration > 0.
- `shots[].layers[].out` — `{type, dur}`, default none — how the layer leaves, ending at `to` (or the hold's end). Same types as `in`.
- `shots[].layers[].effects` — list of built-in effects (same fields as `shots[].effects[]`, no `custom`), default none — on this layer only: shake moves it, trails draw fading earlier copies, blur/bloom/chroma filter its pixels (spreading past its box).
- `shots[].layers[].mask` — `subject`, default none — cuts the layer to its source's subject matte (`montaj ingest --matte <stem>`), and makes it the plane `text[].behind` draws under. Not with `shadow`.
- `shots[].crop` — `[x0, y0, x1, y1]`, default the full image — the part of the photo that is shown. On a clip it is a box of the source video frame, in source px, inside the video.
- `shots[].focus` — `[x, y]`, `face`, `face:N` or `subject`, default the crop centre — the photo point placed at the screen centre. On a clip it is a source-video pixel inside the crop, a one-key `camera`. `face` is the centre of the largest face box, `face:N` the N-th largest, `subject` the saliency peak, all read from `assets/analysis.json` and written into the shot as `[x, y]` before hashing; on a clip they come from the scene holding the shot's start. A photo with no face is `ERR shots[i].focus: no face in <stem> (use subject or [x, y])`.
- `shots[].zoom` — number, default cover — design pixels per photo pixel. Cover fits the crop with a hair of overscan. On a clip it is a multiple of cover (>= 1), a one-key `camera`, as in `camera[].zoom`.
- `shots[].in` — mapping, default none — how this shot arrives from the previous one. Omit `in` for a cut. See transitions.
- `shots[].in.type` — `cut`, `fade`, `swirl`, `whip`, `glitch`, `zoom`, `dip`, `push`, `wall-pullback`, `burn-white` or `custom`, required inside `in` — which transition.
- `shots[].in.dur` — duration, required except for a cut — length of the transition. It has to fit in the shots it overlaps.
- `shots[].in.center` — `[x, y]`, default the outgoing shot's focus — where a swirl opens, in the outgoing photo.
- `shots[].in.axis` — `x` or `y`, default `x` — which way a whip or push travels.
- `shots[].in.strength` — number in `(0, 4]`, default `1` — glitch tear and channel split, or zoom-blur reach. Glitch and zoom only.
- `shots[].in.seed` — whole number `>= 0`, default `0` — glitch pattern seed; the tear is drawn from seed and frame, so a render repeats exactly. Glitch only.
- `shots[].in.color` — `#rrggbb`, default `#000000` — colour the picture dips through. Dip only.
- `shots[].in.blur` — number, default `0` — design-px blur on fading layers. Shot 0 may use `fade` (from `video.background`); other transitions on shot 0 stay an error.
- `shots[].in.zoom` — number, default `3.2` — wall-pullback only. Magnification of the incoming frame at the start; the camera eases to 1. Must be > 1.
- `shots[].in.src` — path to a `.py` file, required for `custom` — the transition's `draw(ctx)`; `ctx.inputs` is `[outgoing, incoming]` and `ctx.p` runs 0..1 across the window. Custom only.
- `shots[].in.params` — mapping, default none — `ctx.params` for the custom transition. Custom only.
- `shots[].flash` — number, default none — white flash on this shot's first frame, fading out over the next 6 frames.
- `shots[].wall` — mapping, one source: photo, wall, clip, blank, sequence or card — a flat field of prints. The camera only pans and zooms. Mutually exclusive with the other sources.
- `wall.grid` — `WxH` string, required — distance between print cells, in world design pixels.
- `wall.seed` — integer, required — seed for the jitter. Random draws happen in `prints` list order.
- `wall.jitter` — number, required — a non-straight print shifts by up to this many pixels in x, then in y.
- `wall.rot` — number, required — a non-straight print rotates by up to this many degrees.
- `wall.print_height` — number, required — print size. Its scale is this divided by the taller of the crop height and 1.15 times the crop width.
- `wall.prints` — list, required — the prints, back to front in list order.
- `wall.prints[].photo` — string, required — stem printed on that cell.
- `wall.prints[].cell` — `[column, row]`, required — which grid cell the print sits on. Column and row are integer steps of `grid`.
- `wall.prints[].straight` — boolean, default false — no shift, no rotation, and it does not take a random draw.
- `wall.prints[].scale` — number, default from `print_height` — replaces the computed print scale. `1` means one photo pixel per world pixel.
- `wall.prints[].crop` — `[x0, y0, x1, y1]`, default the full image — which part of the photo is on the print.
- `wall.light.center` — `[x, y]`, required — brightest point on the wall, world pixels.
- `wall.light.radius` — `[rx, ry]`, required — how wide that pool of light is. Brightness runs from half at the edge to full at the centre.
- `wall.camera.pos` — list of keys, required — where the camera looks. `t` is measured from the start of this shot, including its fade-in.
- `wall.camera.zoom` — list of keys, required — zoom track on the same clock as `pos`.
- `wall.camera.pos[].t` — duration, required — when this key is reached.
- `wall.camera.pos[].print` — string, optional — which print `at` is measured on.
- `wall.camera.pos[].at` — `[x, y]`, optional — a point on that print, in photo pixels. With `print`, this is the camera target. Leave both off to hold the previous target.
- `wall.camera.pos[].ease` — easing name, default `smooth` — how the camera travels into this key.
- `wall.camera.zoom[].t` — duration, required — when this key is reached.
- `wall.camera.zoom[].value` — number, optional — zoom at this key. Leave it off to hold the previous zoom.
- `wall.camera.zoom[].print` — string, optional — if set, `value` is divided by that print's scale, so the number is a zoom relative to the print.
- `wall.camera.zoom[].ease` — easing name, default `smooth` — how the zoom changes into this key.
- `fx` — list, default none — light leaks, flashes, hits and bursts at absolute times, not tied to a shot.
- `fx[].leak` — number, leak or flash or hit or burst or rgb_split — strength of a warm light leak around `at`.
- `fx[].flash` — number, leak or flash or hit or burst or rgb_split — strength of a white flash at `at`, fading over 6 frames.
- `fx[].hit` — number, leak or flash or hit or burst or rgb_split — golden radial flash (and optional punch) at `at`.
- `fx[].rgb_split` — number in `(0, 4]`, leak or flash or hit or burst or rgb_split — one-off red/blue channel split along x on the composite at `at`, decaying to nothing over `dur`. About `32 × value` design px at its peak.
- `fx[].dur` — duration, default `6f` — length of an rgb_split hit. rgb_split only.
- `fx[].burst` — whole number, leak or flash or hit or burst or rgb_split — sparkle-particle count. An integral float such as `3.0` is accepted. Needs `pos`. Seed is `100` plus the burst's index among bursts.
- `fx[].pos` — `[x, y]`, required with a burst — burst origin in design px.
- `fx[].at` — time expression, required — when it happens, from the start of the film.
- `fx[].dir` — `1` or `-1`, required with a leak — which side the leak comes in from. Not used on a flash, hit or burst.
- `markers` — mapping, default none — name to a time expression from film start, `event:`, `shown:` or `word:<i>`.
- `audio` — mapping, default none — soundtrack. No audio block is a silent film.
- `audio.track` — string, default none (a silent bed, so linked clip audio needs no music) — wav/path relative to the spec file: their track, a `music gen` candidate, or `music/song.wav` from `montaj music compose`.
- `audio.markers` — string, default none — JSON `{bpm, beats: [s…], words: [{i, w, s, e}, …]}` from `montaj music analyze`, relative to the spec file. Needed for `word:` markers; supplies the beat grid when `video.bpm` is omitted. `bpm` must be > 0 and `beats` ≥ 0.
- `audio.fade_out` — duration, default `0s` — afade at the end of the film.
- `audio.loudnorm` — boolean, default true — loudnorm `-14` LUFS, TP `-1.5`, LRA `11`.
- `audio.edit` — list, default none — beat-grid edits of the music, applied in order before everything else; each op sees the music as the earlier ops left it. Times are music times (`20b`, `12.5s`; beats sit on the film's grid), not markers. `word:` markers follow the edit; a word inside a cut is gone. Every op must end inside the music as it stands then (a `loop: [3s, 6s]` on a 4 s track is an error).
- `audio.edit[].cut` — `[start, end]`, one of cut or loop — removes that span. The join gets a 12 ms equal-power crossfade; the music is exactly the cut shorter.
- `audio.edit[].loop` — `[start, end]`, one of cut or loop — repeats that span.
- `audio.edit[].times` — whole number ≥ 2, default `2` — loop only: total plays of the span.
- `audio.clips` — list, default none — voice and SFX placed on the film's clock, mixed with the music into one stereo 48 kHz wav in `build/audio/` (cached by input bytes). Without `clips`, `edit` and `duck`, `audio.track` is muxed untouched. A mix over -0.2 dBFS is scaled down and `check` warns.
- `audio.clips[].src` — string, required — file relative to the spec, or the same name in the `assets` folder (extension optional; wav/flac/mp3/ogg/m4a, or a video file's audio). A spec-relative file wins when both exist. Or `synth:whoosh`, `synth:riser`, `synth:boom`, `synth:tick`, `synth:glitch`, `synth:pop`, `synth:click`, `synth:swish`, `synth:hit`, `synth:bass-drop`, `synth:shutter`, `synth:ding` (seeded, deterministic). A file `sfx/<name>.*` beside the spec replaces the built-in `synth:<name>`.
- `audio.clips[].at` — time expression or list of them, required — start time(s) from film start (`0.8s`, `2b`, `logo`). A list places the clip once per time.
- `audio.clips[].offset` — signed duration (`ms`, `s`, `f`, `b`), default `0s` — added to every `at`, e.g. `-0.16s` to land a whoosh before the cut. Anything before the film start is dropped.
- `audio.clips[].id` — string, default none — name `duck.under` refers to. Several clips may share an id.
- `audio.clips[].stream` — whole number, default `0` — which audio stream of a video file (`0:a:N`).
- `audio.clips[].in` — duration, default `0s` — start inside the source.
- `audio.clips[].out` — duration, default end of source — end inside the source; must be after `in`.
- `audio.clips[].gain` — dB string, default `0dB` — level, e.g. `-10dB`; within ±120 dB.
- `audio.clips[].pan` — number in `[-1, 1]`, default `0` — constant-power pan, `-1` left.
- `audio.sfx` — `auto`, default none — adds sound design from the layout: `whoosh` on `whip` / `push` / `zoom` transitions (peak on the cut for whip, mid-window for push/zoom), `riser` peaking at marker `drop` if defined, `tick` at each `reveal: type` line. Each sound starts early by its own peak lead so the peak lands on the event. Expands to ordinary `audio.clips` (id `sfx`).
- `audio.sfx_gain` — dB string, default `-8dB` — gain of every `emphasis` and `sfx: auto` sound.
- `audio.duck` — mapping, default none — dips the music (not other clips) while the named clips speak: RMS over 120 ms above -45 dB, eased by attack and release.
- `audio.duck.under` — list of clip ids (or linked shot ids), required — which clips trigger the duck.
- `audio.duck.depth` — dB string below 0, default `-6dB` — how far the music drops.
- `audio.duck.attack` — duration (`40ms`), default `40ms` — time to reach the depth.
- `audio.duck.release` — duration, default `200ms` — time to come back.
- `text` — list, default none — overlay blocks. Each line reveals unit by unit.
- `text[].from` — time expression, required — block visible window start.
- `text[].to` — time expression, required — block visible window end.
- `text[].fade_in` — duration, default `0s` — block fade in. `0` is instant.
- `text[].fade_out` — duration, default `0.8s` (`0s` when a line sets `out`) — block fade out; the block blurs by `(1 − a)·blur`.
- `text[].blur` — number, default `10` — design-px blur while the block fades.
- `text[].motion_blur` — boolean, default false — draws the block as the average of `max(video.motion_blur, 5)` sub-frame times (same shutter as `video.motion_blur`), so moving text smears.
- `text[].backdrop` — mapping, `true`, `gradient` or `solid`, default none — dark scrim behind the block, painted by the compositor. `true` is the default gradient. Fades with the block.
- `text[].behind` — `subject`, default none — draws the block between the picture and the on-screen subject (a `matte:` clip shot or `mask:` layers; through a fade both sides). Without a subject on screen it draws on top. Needs a matte somewhere in the film.
- `text[].role` — `title` or `caption`, default none — what the block is for `check` and `validate`: `title` (designed text, slam words) is never judged as a caption; `caption` always is. Unset, a block is a caption when every line sits below 55% of the frame height, it has at most 8 words and no `dot`. Wins over its lines' `role`.
- `text[].backdrop.type` — `gradient` or `solid`, default `gradient` — `gradient` fades out; `solid` is flat over the same area.
- `text[].backdrop.strength` — number, default `0.8` — opacity of the darkest part, from 0 to 1.
- `text[].backdrop.place` — `bottom`, `top` or `band`, default `band` — `band` covers the lines (full strength on the line box, cosine fade across a pad of 48 design px or 0.45× the line size); `bottom` and `top` keep the outer 28% of the frame fully dark, then fade across the next 22%.
- `text[].lines` — list, required — the lines in this block.
- `text[].lines[].text` — string, required — the line. `*words*` mark emphasis (gpu engine); `\*` is a literal asterisk; an unclosed `*` is an error.
- `text[].lines[].style` — `serif`, `script`, `caps`, `deva`, `sans` (Inter) or `mono` (JetBrains Mono), default `serif`. `sans` size `48`, weights 400/600/700; `mono` size `32`, weights 400/700; both cream, tracking 0.
- `text[].lines[].italic` — boolean, default false — serif only.
- `text[].lines[].gold` — boolean, default false — gold gradient fill with a travelling sweep.
- `text[].lines[].color` — `cream`, `gold`, `ink` or `#rrggbb`, default serif/deva cream `#fbf1dc`, script/caps gold `#f0c77e`.
- `text[].lines[].size` — number, default serif/script `84`, caps `26`, deva `62` — font size in design px.
- `text[].lines[].weight` — number, default serif `300`, script/deva `400`, caps `600`.
- `text[].lines[].tracking` — number, default serif `0.01`, caps `0.42`, script/deva `0` — letter-spacing in em.
- `text[].lines[].line_height` — number, default serif `1.1`, deva `1.5`; script and caps leave it unset.
- `text[].lines[].y` — number, required unless `pos` is set — design-px top of the line box.
- `text[].lines[].at` — time expression, required — when this line starts revealing.
- `text[].lines[].by` — `word` or `char`, default `word` — reveal unit.
- `text[].lines[].stagger` — duration, default `0.16s` word / `0.07s` char — delay between units.
- `text[].lines[].dur` — duration, default `0.9s` — reveal length of one unit.
- `text[].lines[].rise` — number, default `26` — design-px rise of each unit.
- `text[].lines[].reveal` — `rise`, `pop` or `type`, default `rise`. `type` is a typewriter: one char per `stagger` (default `0.05s`), no rise or blur, each char instant (`dur` is rejected); forces `by: char`.
- `text[].lines[].sweep` — `[from, to]` time expressions, default the block window — gold gradient travels 100%→0% across it.
- `text[].lines[].shadow` — boolean or `{x, y, blur, color, opacity}`, default true — false drops the text shadow (ink lines). The mapping (gpu engine; defaults `0`, `6`, `12` design px, `#000000`, `0.6`) replaces the style's shadow with one drop shadow.
- `text[].lines[].indent` — number, default `0` — extra `margin-left` in em, on top of the centred line.
- `text[].lines[].align` — `center`, `left` or `right`, default `center` — `left`/`right` inset the line 72 design px from that edge.
- `text[].lines[].cursor` — boolean, default false — `reveal: type` only; a bar after the last typed char, solid while typing, then blinking every `0.53s`.
- `text[].lines[].box` — `none` or `chip`, default `none` — `chip` draws a rounded translucent pill behind the line, revealed with its first unit.
- `text[].lines[].box_color` — `cream`, `gold`, `ink` or `#rrggbb`, default dark translucent — chip fill; needs `box: chip`.
- `text[].lines[].dot` — `cream`, `gold`, `ink` or `#rrggbb`, default none — status dot before the text inside the chip; needs `box: chip`.
- `text[].lines[].engine` — `chromium` or `gpu`, default `gpu` when the line sets any styling field below (`font`, `case`, `emphasis`, `stroke`, `glow`, `pos`, `rotate`, a `shadow` mapping, `in`, `out`, `idle`) or marks `*words*`, else the style's engine (`montaj doctor` lists it; all `chromium` today) — `gpu` shapes with HarfBuzz and draws distance-field glyphs on the GPU, no browser; rejects `gold`, `box`, `cursor`, card lines and characters the fonts lack. `engine: chromium` with a styling field is an error.
- `text[].lines[].font` — `montserrat-black`, `anton`, `bebas`, `poppins-extrabold`, `bangers`, `archivo-black` or `mukta-extrabold` (Latin + Devanagari), default the style's font — bundled display face (OFL); excludes `weight` and `italic`. An unknown name lists the fonts.
- `text[].lines[].case` — `upper`, `lower` or `as-is`, default the style's (`caps` upper, others as-is).
- `text[].lines[].emphasis` — `{color, scale, in, shake}`, default `{color: "#ffe600", scale: 1}` when the text marks `*words*` — fill and size multiplier (0–3) of the marked words; needs marked words.
- `text[].lines[].emphasis.in` — an `in.type` other than `type`, default the line's — entry of the units holding marked words, at that type's default `dur`, `overshoot` and `distance`.
- `text[].lines[].emphasis.shake` — number 0–60, default none — design-px seeded shake of the marked units for `0.25s` after they reach full size, decaying to 0 (turns up to 0.4° per px).
- `text[].lines[].stroke` — `{width, color}`, default none — outline `width` design px (≤ 37.5% of `size`; `validate` warns over 20%), `color` default `#000000`.
- `text[].lines[].glow` — number 0–1, default none — halo in the fill colour, blur 0.25× `size`, at this opacity.
- `text[].lines[].pos` — `[x, y]` design px, default none — centre of the line box; replaces `y` and `align`.
- `text[].lines[].rotate` — degrees, default `0` — turns the whole line about its box centre (positive is clockwise).
- `text[].lines[].fit` — boolean, default false — shrink `size` (whole design px) until the line fits `max_width`. Measured from HarfBuzz shaping of the bundled fonts, which equals Chromium's layout width, so it holds for every style and engine; a chip's padding is not counted. Not on card lines yet; needs glyphs the style's fonts have.
- `text[].lines[].wrap` — `balance`, default none — break at spaces into the fewest lines (at most `max_lines`) that fit `max_width`, of similar width, never leaving a short word alone on the last line. The lines stack at the line-box height; the reveal continues down the stack. With `fit`, the size shrinks only if `max_lines` is not enough. At most 24 words.
- `text[].lines[].max_width` — number, default `936` (frame width minus 2 × 72 design px) — the width `fit` and `wrap` aim for; set without `fit` or `wrap`, a wider line is an `ERR`.
- `text[].lines[].max_lines` — number 1 to 4, default `2` — most lines `wrap: balance` may use; needs `wrap`.
- `text[].lines[].in` — type name or mapping, default none (the `reveal`) — per-unit entry (gpu engine); replaces `reveal` (`reveal: pop`/`type` with `in` is an error). `in: pop` = `in: {type: pop}`. Closed form, so every frame is a function of time.
- `text[].lines[].in.type` — `pop` (0.8→overshoot→1, `0.2s`), `slam` (1.8→1 accelerating, then a 2-frame squash by the overshoot, `0.2s`), `slide` (`0.3s`), `drop` (falls `distance` and bounces, `0.5s`), `rise` (the `reveal: rise` motion, identical pixels), `type` (the typewriter, identical pixels), `flip` (vertical unfold, `0.3s`) or `scale-in` (0→1, `0.25s`), required.
- `text[].lines[].in.by` — `word` or `char`, default the line's `by` — entry unit.
- `text[].lines[].in.dur` — duration, default per type above — entry length of one unit; overrides `dur`. Rejected for `type`.
- `text[].lines[].in.stagger` — duration, default the line's `stagger` — delay between units; overrides `stagger`.
- `text[].lines[].in.overshoot` — number 1–2, default pop `1.08`, slam `1.06`, slide `1.04`, flip/scale-in `1` — peak past rest (scale, height or travel), reached at 60% of `dur`; `pop`, `slam`, `slide`, `flip`, `scale-in` only.
- `text[].lines[].in.from` — `left`, `right`, `top` or `bottom`, default `left` — side a `slide` enters from.
- `text[].lines[].in.distance` — design px 0–2000, default slide `160`, drop `220`, rise the line's `rise` — travel of `slide`, `drop`, `rise`.
- `text[].lines[].out` — type name or mapping, default none — per-unit exit (gpu engine); the last unit's exit ends at the block's `to`.
- `text[].lines[].out.type` — `cut` (instant), `fade` (`0.2s`), `pop-out` (swells to 1.08, shrinks to 0, `0.2s`) or `slide` (`0.25s`), required.
- `text[].lines[].out.dur` — duration, default per type above — exit length of one unit. Rejected for `cut`.
- `text[].lines[].out.stagger` — duration, default `0s` — delay between units leaving, continuous across `wrap` lines.
- `text[].lines[].out.to` — `left`, `right`, `top` or `bottom`, default `right` — side a `slide` exit leaves toward.
- `text[].lines[].out.distance` — design px 0–2000, default `160` — travel of a `slide` exit.
- `text[].lines[].idle` — `float` or `pulse` (or `{type, amount}`), default none — loop every `1.5s` once a unit lands: `float` bobs `amount` design px (default `4`), `pulse` scales by `amount` percent (default `4`); 0–100.
- `text[].lines[].bar` — `{word, from?, color?, pad?}`, default none — a rounded bar behind the 0-based word `word` of the line (gpu engine), hugging its ink; `from` (another word) makes it slide from there over 3 frames; `color` default `#ffe600`; `pad` design px, default `14`. Written by `captions` for `preset: highlight`.
- `text[].lines[].role` — `title` or `caption`, default none — as `text[].role`, for a style to carry (`styles: {slam: {role: title, ...}}`): a `title` line makes its block a title.
- `validate` — shapes every text line (card lines too) and reports `ERR text[1].lines[0]: 132px wider than the frame at size 100 (fits at size 88; or set fit: true)` when the line box leaves the 1080-wide frame. Lines with a character the bundled fonts lack are not measured.
- `text[].lower_third` — `{title, subtitle?, at?, side?, y?}`, replaces `lines` — sugar for a bold sans title chip with a gold dot and a sans subtitle 110 design px below. `at` default the block `from`, `side` `left` or `right` default `left`, `y` default `1500` on a 1920-tall (9:16) frame and the same fraction of the frame height on any other, kept 210 design px above the bottom edge (design px, title top). `fade_in`/`fade_out` default `0.4s`. No bar or slide yet.
- `captions` — mapping, default none — word timings expanded at load into gpu `text` blocks after your own (the renderer and the cache see only those); `montaj captions --print` lists `chunk start end text`. Rules and presets: `docs/CAPTIONS.md`.
- `captions.source` — string, required — `audio.markers`, a words JSON beside the spec (`{words: [{i, w, s, e}]}`), or a clip id whose `assets/<id>.words.json` `montaj transcribe <id>` wrote. With a clip id and no `audio` block, the shots showing that clip get `audio: true` (the voice is heard); any `audio` block, or `audio: false` on the shot, turns that off.
- `captions.preset` — `punch`, `karaoke`, `highlight`, `clean`, `comic` or `minimal`, default `punch` — the look; each sets font, stroke, shadow, entry and the accent treatment.
- `captions.words_per_chunk` — `[min, max]` within 1 to 6, default `[1, 3]` — a chunk breaks at `max` words, at a pause of 0.3 s, at `. ! ? …`, at a comma once it holds `min`, or when its width passes the frame budget.
- `captions.accent` — `#rrggbb`, default `#ffe600` — colour of the accented word (the `highlight` bar, the `minimal` serif word).
- `captions.emphasis` — `auto`, `none` or a list of word indexes, default `auto` — which word per chunk takes the accent; `auto` picks a number, a negation, or a long content word, never in two chunks running.
- `captions.y` — number, default 64.6 % of the frame height on portrait, 80 % otherwise — design-px centre of the caption.
- `captions.offset` — duration, default none — shifts every word time (a trimmed clip).
- `captions.punctuation` — `drop` or `keep`, default `drop` — `drop` removes a trailing `.` `,` `।` from every chunk (`?` `!` `…` stay).
- `captions.overrides` — list, default none — per-word fixes: `{word, text?, emphasis?}` rewrites a word or forces/clears its accent; `{chunk_break_before}` starts a chunk at that word. An index not in the source is an error. Trailing `.` and `,` are dropped from every word; `?` `!` stay.

## Emphasis (word-anchored edits)

`emphasis` expands at load into shot `effects`, clip `camera` keys, `fx`, `audio.clips` and `captions.overrides`; the expansion is what is hashed. Word numbers: `montaj words [--grep text]`.

- `emphasis` — list, default none — each entry names one moment (`word`, `match` or `on`) and any actions below. Every action lands on that word's start, so it survives `audio.edit`.
- `emphasis[].word` — integer, one of `word`/`match`/`on` — word index `i` in `audio.markers`, else in the `captions.source` words file.
- `emphasis[].match` — string, one of `word`/`match`/`on` — words to find (case and punctuation ignored); `text#2` picks the second occurrence. No match is an error.
- `emphasis[].on` — `every <N>b|s|f`, one of `word`/`match`/`on` — repeats the actions on a grid from film start (beats need `video.bpm`). No `caption`.
- `emphasis[].zoom` — number ≥ 1, default none — camera push peaking exactly on the word (ramps from 4f before, back after 12f). Clip shots without `camera` only; replaces the shot's `zoom`/`focus` with keys.
- `emphasis[].shake` — amount (max 200), default none — an 8f `shake` effect at the word.
- `emphasis[].flash` — number 0–1, default none — a top-level `fx` flash at the word.
- `emphasis[].sfx` — synth name or file, default none — a sound whose peak (not its first sample) lands on the word.
- `emphasis[].caption` — `accent`, default none — forces the word's caption emphasis. Needs `captions`.
- `subs` — list, default none — lyric subtitles, one fixed style.
- `subs[].from` — time expression, required — subtitle window start.
- `subs[].to` — time expression, required — subtitle window end.
- `subs[].text` — string, required — the lyric line.
- `subs[].punctuation` — `drop` or `keep`, default unset — `drop` removes a trailing `.` `,` `।` from the line; unset keeps it as written and `validate` warns `WARN subs[i] ends with a full stop or comma`.
- `subs[].size` — number, default `44` — font size in design px of a 1080-wide frame.
- `subs[].color` — `cream`, `gold`, `ink` or `#rrggbb`, default `rgba(251,241,220,.92)` — `.sub` fill. Omit it to keep that translucent cream.
- `tracks` — mapping, default none — slow global levels. Keys are piecewise curves like wall camera keys. Before the first key: first value; after the last: last value.
- `tracks.dust` — list of keys, default none — mote level.
- `tracks.glow` — list of keys, default none — strength of the two drifting light leaks.
- `tracks.bars` — list of keys, default none — letterbox height in design px.
- `tracks.*[].t` — time expression, required — when this key is reached.
- `tracks.*[].v` — number, required — value at this key.
- `tracks.*[].ease` — easing name, default `linear` for dust and glow, `sine` for bars — the curve into this key.
- `overlays` — list, default none — vector HUD shapes drawn over the picture, anti-aliased on the GPU. Graded with the picture, under text. Screen values are design px. A shape is drawn on frames `from` ≤ f < `to`.
- `overlays[].shape` — `ring`, `brackets`, `line` or `waveform`, required.
- `overlays[].from` — time expression, required — first frame the shape shows.
- `overlays[].to` — time expression, required — end of the window; must be after `from`.
- `overlays[].center` — `[x, y]`, required with ring — ring centre.
- `overlays[].radius` — number or `[from, to]`, required with ring — ring radius; a pair animates over the window with `ease`.
- `overlays[].sweep` — number or `[from, to]` degrees, 0–360, default `360` — arc extent, clockwise from 12 o'clock; a pair draws the arc on over the window. Ring only.
- `overlays[].box` — `[x0, y0, x1, y1]`, required with brackets and waveform — brackets: four corner marks, no edges. waveform: vertical bars fill the region, mirrored about its horizontal centre.
- `overlays[].source` — clip id or `music`, required with waveform — an `audio.clips[].id` or `shots[].id` (every clip or linked shot with that id, summed after gain and placement) or `music` (the track after `audio.edit` and the duck). Bars follow what is heard: the mix's peak ceiling and `audio.fade_out` apply; loudnorm, one constant gain, does not. Unknown source, or no `audio` block, is an error. `music` is the track even if a clip uses that id.
- `overlays[].bars` — integer 1–128, default `32`, waveform only — how many vertical bars. Each bar is the peak of its slice of that frame, quantised to 1/255.
- `overlays[].arm` — number, default `28` — bracket arm length. Brackets only; capped at half the box side.
- `overlays[].points` — list of `[x, y]`, at least 2, required with line — a polyline.
- `overlays[].draw` — duration, default none — brackets grow / the line draws on from its first point over this long from `from`, eased with `ease`. Omit it to show the shape fully the whole window. Brackets and line only.
- `overlays[].width` — number, default `3` — stroke width in design px. On a waveform, the bar thickness, capped at the slot so bars stay inside the box.
- `overlays[].color` — `#rrggbb`, default `#ffffff`.
- `overlays[].opacity` — number 0–1, default `1`.
- `overlays[].glow` — number ≥ 0, default `0` — soft halo around the stroke, screened under it. About `0.6` is a gentle neon.
- `overlays[].ease` — easing name, default `smooth` — curve for radius, sweep and draw.
- `overlays[].above_finish` — boolean, default false — skip the look, so the shape stays crisp and ungraded; still under text, and fades with the picture.
- `styles` — mapping, default none — named bundles of `text[].lines[]` fields, used as `style: <name>` on a line. A name must not be a built-in text style (`serif`, `script`, `caps`, `deva`, `sans`, `mono`): that is an error. A style may set the built-in `style` itself. Precedence: the line's own fields, then its style, then `defaults`; nested mappings merge field by field.
- `styles.<name>.extends` — string, default none — another `styles` entry to start from; chains are allowed, a cycle or unknown name is an error.
- `defaults` — mapping, default none — fields filled in where an item leaves them out, keyed by path: `shots`, `text`, `text.lines`, `fx`, `subs`, `overlays`. An explicit value always wins; `in: null` drops a default transition.
- `components` — mapping, default none — named groups of `shots`, `text`, `fx`, `subs` or `overlays` items. `params` is a list of required names or a `{name: default}` mapping. `$name` in a value is replaced by the argument (a whole-value `$name` keeps its type); `$at+1.2s` adds a duration to a duration or marker offset, one unit only.
- `use` — list, default none — `{component: name, <param>: value}` instances; their items are appended to the matching top-level lists. Inside `shots`, `text`, … an item `{use: name, <param>: value}` is replaced in place by the component's items.
- `include` — path or list, default none — files (relative to the spec) holding only `styles`, `defaults` and `components`; the including file wins on a clash.
- `check` — mapping, default none — designed QA exceptions. The renderer does not read this, so it does not change the segment hash.
- `check.expect` — list, default none — black or frozen spans `check` treats as designed. A span that matches no detected span prints `INFO expect <kind> <start>–<end>s not found (<nearest>)`, naming the nearest detected span or one under the detector's minimum (black 0.10 s, frozen 0.5 s).
- `check.expect[].black` — `[start, end]` time expressions, black or frozen — a designed black window. Set this or `frozen`, not both. `end` must be later than `start`.
- `check.expect[].frozen` — `[start, end]` time expressions, black or frozen — a designed hold. Set this or `black`, not both. `end` must be later than `start`.
- `check.expect[].why` — string, default none — one line, added when this span matches a detected one.
- `check.expect[].caption` — `[start, end]` time expressions — a designed caption exception: `check` silences `caption-*` findings inside it. Set exactly one of `black`, `frozen`, `caption`.
- `check.safe` — mapping, default 9:16 top 0.12, bottom 0.22, right 0.14 (16:9 bottom 0.08) — platform safe-zone fractions for `WARN caption-safe text[i] <a>-<b>s '<text>' <side> (...)`; keys `top` `bottom` `left` `right`.
- `check.loudness` — mapping, default none — replaces the `-14` LUFS target used by `check`.
- `check.loudness.i` — number, required with loudness — integrated-loudness target, in LUFS.
- `check.loudness.tol` — number, default `1` — how far the measurement may sit from `i`, in LU.
- `check.pacing` — mapping, default none — pacing targets. `check` always prints `INFO pacing|hook|beats|captions|audio|variety` numbers (also under `--json` as `pacing`); a `WARN` appears only for a target set here and missed. Like all of `check`, it stays out of the segment hash.
- `check.pacing.max_shot` — duration, default none — longest allowed shot. Misses print `WARN pacing longest=<t>@shot<i>`.
- `check.pacing.min_per10s` — number ≥ 0, default none — fewest shot changes per 10 s (`cuts` is shots − 1). Misses print `WARN pacing per10s=<n>`.
- `check.pacing.hook_text_by` — duration, default none — latest time the first on-screen text (text lines or subs) may appear. A film with no text also warns.

Transitions (`shots[].in`):

- `cut` — the new shot replaces the old one on its first frame. Takes no extra frames.
- `fade` — crossfade across the first `dur` frames of the incoming shot.
- `swirl` — a swirling wipe that opens around `center`, across the first `dur` frames of the incoming shot.
- `whip` — both shots slide along `axis`. The move is `dur` frames long and centred on the cut, half on each side.
- `glitch` — across the first `dur` frames of the incoming shot, random blocks swap to the new shot while horizontal bands tear sideways and red/blue split apart. Peaks mid-way, clean at both ends.
- `zoom` — a zoom-blur punch across the first `dur` frames: the old shot rushes in and smears, the new one arrives from the same rush and settles; they cross over in the middle.
- `dip` — across the first `dur` frames the old shot goes to `color`, then the new shot comes up from it.
- `push` — across the first `dur` frames the new shot slides in along `axis` and shoves the old one out. Starts on the cut, softer ease than whip.
- `wall-pullback` — camera eases with `outc` from `zoom` (default 3.2) down to the incoming frame. The outgoing frame shrinks as an opaque centre card, then dissolves, so the incoming shot opens as a wall. First `dur` frames of the incoming shot.
- `burn-white` — the outgoing shot burns to white over the first half (highlights clip first) and the incoming shot emerges from white over the second half. First `dur` frames of the incoming shot.
- `custom` — the `draw(ctx)` in `src` mixes `ctx.inputs = [outgoing, incoming]` as `ctx.p` goes 0..1. First `dur` frames of the incoming shot.

Custom code (`custom:`, `effects[].custom`, `in.src`):

- Contract — `def draw(ctx)` at the file's top level, pure (no IO, no globals, no randomness outside `ctx`), returns a float32 tensor on `ctx.device`: `(3, H, W)` RGB in 0..1, or `(4, H, W)` premultiplied RGBA composited over `inputs[0]` (or `video.background`). Output is clamped to 0..1 before the look.
- `ctx` — `uv (H, W, 2)` pixel centres, x right, y up, short side spans [-1, 1]; `t` seconds since the shot (or transition) start; `p` progress 0..1; `f` film frame; `fps`; `beat` (beats since the grid start, or None with no `video.bpm`); `params` (a copy); `inputs` (copies); `W`, `H`, `k` (design px scale); `device`.
- Helpers — `ctx.noise(seed)` (H, W) uniform, `hash(p, seed)`, `vnoise(p, seed)`, `fbm(p, octaves, seed)`, `sd_circle(p, r, center)`, `sd_box(p, half, center)`, `fill(d, soft)`, `polar(p)` -> (r, angle), `palette(t, a, b, c, d)`, `rgb('#hex')`, `sample(img, p)` (warp by uv coordinates), `blur(img, sigma)` in design px, `smoothstep`, `mix`. Seeded and deterministic.
- Cache — the segment hash covers each used file's bytes, its params and the helper API version, so editing a file re-renders only the segments that draw it.
- Errors — `validate` checks the file exists, parses and defines `draw`. At render, an exception, a wrong shape or dtype, NaN/inf or a CPU tensor is `ERR render: custom <file>:<line>: <problem> (<hint>)`; the traceback goes to `build/render.log`.
- `montaj custom test <file.py> [--kind shot|effect|transition] [--params JSON] [--size WxH]` — runs `draw` at a small size across a few frames, checks range and determinism, writes a strip and prints `OK <file> <ms>ms/frame peak <n>MB strip <path>`.

Looks (`video.look`):

- `warm-film` — sharpen, softer contrast, lifted blacks, a vignette, and amber leaks. Uses `shots[].flash`, `fx[].flash`, `fx[].leak`, `video.intro`, and a glow outro. Ignores `fx[].hit`, `fx[].burst`, and `tracks` (`dust`, `glow`, `bars`). `outro: {type: fade}` still fades the frame to black.
- `golden-film` — `tracks.glow` leaks, `tracks.dust`, `fx[].hit`, `fx[].burst`, `tracks.bars`, and grain. Ignores `shots[].flash`, `fx[].flash`, `fx[].leak`, `video.intro`, and a glow outro. Does not fade to black unless the spec sets `outro: {type: fade}`.

- `clean` — near-identity: light sharpen, no contrast curve, true blacks, a faint vignette. No cast, so screen recordings keep their colours. Uses `shots[].flash`, `fx[].flash`, and `outro: {type: fade}`. Ignores `fx[].leak`, `fx[].hit`, `fx[].burst`, `tracks`, `video.intro`, and a glow outro.
- `cool-tech` — `clean` plus teal-leaning shadows, cooler whites, a slight contrast curve and a light vignette. Same fx as `clean`.
- `{lut: …}` — the LUT applied to the untouched picture, then `vignette` and `grain`. Same fx as `clean`.

Easing (the curve into a camera key):

- `linear` — constant speed.
- `sine` — half a cosine, the film.html `es` curve. Default for `tracks.bars`.
- `smooth` — gentle slow start and slow end. This is the default.
- `cubic` — a stronger slow start and slow end.
- `outc` — leaves quickly and settles into the key.
- `inc` — creeps, then arrives quickly.
- `expo` — very slow, then very fast, then very slow.
- `ramp` — slow, fast through the middle, then slow again.

Capture (`montaj capture SCRIPT`):

The script compiles to one list of clicks and scrolls. The same file always schedules the same actions on the same frames. `montaj capture SCRIPT` plays that list and writes a clip. Progress is `build/capture.log`. Stdout is `OK <path> <dur> <WxH> <MB>`.

- `url` — string, required — page to record. `http`, `https` or `file`.
- `size` — `WxH`, default `3840x2160` — viewport in pixels, both even. This is the 4K default.
- `fps` — integer, default `60` — frames per second. Must be >= 1.
- `dur` — duration (`f` or `s`), required — recording length. Beats are an error (there is no grid). Times snap to frames.
- `css` — string, default none — stylesheet injected after load, before the first frame. The same text every run.
- `out` — path, default `<assets>/<script stem>.mp4` — clip path. Relative to the project unless absolute. `.mp4` or `.mov`. `<assets>` is `montaj.yaml`'s `assets` folder, else `assets`, so `shots[].clip` can name the stem.
- `actions` — list, default none — clicks and scrolls in list order.
- `actions[].at` — duration (`f` or `s`), required — when the action starts, from the capture start. Must land on a frame inside `dur`.
- `actions[].click` — string, click or scroll — CSS selector. The click runs at `at`, then that frame is shot.
- `actions[].scroll` — number or mapping, click or scroll — target in CSS pixels. A number sets `y` and leaves `x` at 0.
- `actions[].scroll.x` — number, default `0` — horizontal target.
- `actions[].scroll.y` — number, default `0` — vertical target.
- `actions[].scroll.dur` — duration (`f` or `s`), default `0s` — straight move from the position already scheduled before this action's first frame. `0` jumps. Each scroll keeps its own line; a later action does not rewrite it. The move has to end inside `dur`.

Content guards (under `check`, read by `montaj check` only; they never change the render):

- `check.digits` — `none` or `spec`, default off — OCR the sampled frames. `none`: any on-screen digit warns. `spec`: a digit run warns unless it appears in `text[].lines[].text`, `subs[].text` or `shots[].frame.caption`. Needs the `tesseract` binary.
- `check.faces` — `none`, default off — a face in a sampled frame warns. Needs `uv sync --extra guards` (OpenCV).
- `check.sample` — integer 1 to 30, default `3` — frames sampled per shot, evenly across the shot. Output: `WARN digits shot=<n> frame=<f> "<digits>"`, `WARN faces shot=<n> frame=<f> (<k>)`; a missing backend gives `ERR <guard>: <install hint>`.

Reuse (`styles`, `defaults`, `components`, `use`, `include`) is expanded before validation, so the expanded spec is what is hashed and rendered; errors name the author's key and line. `montaj validate --expand` prints the expanded spec.

## Inspecting a frame
`montaj inspect --track REF [--samples N]` prints one animated value's sampled curve (`OK track ...`, no render): REF is `SHOT.LAYER.PROP` (a keyed or `expr` layer property; LAYER is an index or `id`; shot-local frames), `SHOT.camera.zoom|focus` (a clip shot's camera keys; focus in source px) or `tracks.dust|glow|bars` (film frames). N is 2 to 500, default 12; between the first and last key (an `expr` is sampled over the whole hold). `montaj inspect --at 4.2s|--shot N|--range A B --every 10f|--problems [--fast] [--json]` prints what is on screen as numbers: per text element (cards, `text`, `subs`) the ink `box` in output px, `size`, `off_frame_px`, `safe` (inside a 5% margin) and, unless `--fast`, `contrast` (WCAG ratio vs. the frame around the box; `--problems` flags < 3); `over_face` is always null (no face boxes in the engine). Geometry needs no render; `--problems` ends `OK inspect 0 problems` or `WARN inspect N problems`. Unless `--fast`, each frame also carries `scopes` (the numbers below for the graded picture at that frame; null on a transition).

## Colour scopes and shot matching
- `montaj scopes [--json]` — one line per picture shot from ~4 sampled frames of the graded picture (after grade and effects, before overlays, text and the look; preview size, GPU): `luma p1/p50/p99` (Rec.709 Y'), `clip_hi`/`clip_lo` (% of pixels with Y' ≥ 0.985 / ≤ 0.015), `sat_mean` (HSV), `cast` (+ warm, R−B over near-neutral midtones), `tint` (+ green), and `skin_hue` (degrees; skin line 10–34°) when `analysis.json` has a face on a plain clip or static still. Ends `OK scopes N shots`; `--json` adds `shots: [...]`. Blank and card shots are skipped.
- `montaj scopes --image` — also writes `build/scopes.jpg`: one tile per shot, luma waveform | vectorscope with the skin line.
- `montaj match --to N` — writes `grade: {exposure, temp, tint, saturation}` under every other picture shot so its p50 luma, cast, tint and saturation meet shot N's; other grade fields are kept, a named grade becomes inline. Only those YAML spans change (comments and layout survive). Prints `shot i  exposure a→b  temp …  p50 x→y (target z)` per shot, then `OK montaj.yaml: matched K shots to shot N`. Refuses a file whose `shots` list expands (include, `use`, `cuts: auto`).
- `montaj check` adds `WARN scopes shot N: clip_hi X% > 2%`, `WARN scopes shot N: skin hue X° off the skin line` and `WARN scopes shots A→B: p50 luma jumps D within one scene` (adjacent shots of the same clip or photo, |D| > 0.06); a spec that no longer renders gives `INFO scopes: unavailable (…)`.

## Asset analysis
`montaj ingest [--photos DIR]` normalises photos, then analyses every photo and clip in `assets/` on the GPU into `assets/analysis.json` (keyed by file name, cached by file bytes: unchanged files run no model; the GPU lock is held only while something needs work). Photos: `faces: [{box: [x0, y0, x1, y1], eyes: [[x, y], [x, y]], score}]` largest first, `subject: [x, y]`, `sharpness` (Laplacian variance), `brightness` (0..1), `colors` (hex, most-covered first). Clips: `duration`, `fps`, `cuts: [t]`, `speech` / `silence: [{s, e}]` (audio energy, not a voice model), `motion` (mean frame difference per second, 0..1), `scenes: [{s, e, faces, subject}]` (one mid-scene frame each). `words` is reserved for transcripts. Pixels are source pixels. `--fast` skips analysis; `--list` prints one line per asset. Faces: RetinaFace MobileNet-0.25 (MIT), weights downloaded once to `~/.cache/montaj/models/` (`MONTAJ_MODELS` overrides).

`montaj ingest --matte STEM` (repeatable) computes a person matte for each clip or photo on the GPU into `assets/mattes/<sha16>.subject-v1.mp4|png` (grey; keyed by the source's bytes, so a changed file needs a new matte; unchanged ones are reused). Model: torchvision DeepLabV3-ResNet101 (COCO/VOC person class, BSD), weights downloaded once to `~/.cache/montaj/models/`; edges are refined against the frame with a guided filter and steadied in time. A CPU device is an error under `[gpu] strict`. Warns when the mean frame-to-frame matte change is large.

## Event logs (screen recordings)
`montaj ingest-rec raw/take.mp4 --events raw/take.events.json` copies the recording to `assets/` and writes `assets/<stem>.events.json`. Input and output are a JSON list `[{"t": 3.42, "kind": "state", ...fields}]`; `t` is finite seconds. Input `t` is seconds from record start; ingest subtracts the container `start_time` (earliest stream), so in `assets/` `t` is seconds on the clock `clip_in` seeks. Re-ingesting different video bytes under the same stem without `--events` deletes the old `<stem>.events.json`. Rows stay in log order. Field values are exact-matched (numbers numerically; a missing field never equals `null`); in `event:` markers `true`/`false`/`null` are JSON literals and `"true"` quoted is the string; `contains:` / `~=` match substrings of string fields.

## Strict dGPU mode (machine config, not the spec)
`~/.config/montaj/config.toml`:
- `[gpu] strict = true` — any CPU pixel path stops the command with `ERR gpu-strict: <what> would run on the CPU: <hint>` instead of falling back. Default false. `MONTAJ_GPU_STRICT=1|0` overrides it for one run.
- `[gpu] device = "cuda:0"` — the CUDA device every stage uses; also a GPU UUID or name substring (`"RTX 5050"`), so an integrated GPU is never picked. Default: the current CUDA device.
- Policed pixel stages: **decode** (clip not openable by NVDEC -> ffmpeg), **photo** (nvJPEG for JPEG under strict, PIL otherwise so non-strict stays film2-exact; PNG/WebP and prints' PIL shadow blur are CPU), **text** (lines on the Chromium engine, card lines, subs, frame captions are CPU; `engine: gpu` is not), **encode** (`libx264` when NVENC is unusable; NVENC through the ffmpeg pipe still counts as GPU), **custom** (`draw(ctx)` runs on the canvas device; a CPU canvas is a fallback, and a CPU tensor returned on CUDA is an error, never a copy). A cached segment runs nothing, so strict does not recheck it.
- `render` / `watch` `OK` lines end `gpu=all`, or `gpu=partial(text,decode)` naming stages that fell back (non-strict). `montaj doctor` prints one line per stage (`composite`, `decode`, `photo`, `text`, `encode`, `capture`); under strict a stage that would fall back is `FAIL`.
- Not pixel work, not policed: text shaping, demux/mux, audio mix, Python orchestration. Named exceptions: `capture` rasterises the page in headless Chromium (CPU); its encode uses NVENC (libx264 only when NVENC is unusable, an error under strict). `ingest` and `sheet` use PIL on the CPU.

## Delivery codecs and export targets (CLI, not the spec)
- `montaj render --final --codec h264|hevc10|av1` — default `h264` (8-bit, unchanged; segment hashes unchanged). `hevc10` is HEVC Main10 (`hvc1` tag), `av1` AV1 Main 10-bit; both are float compositor → P010 on the GPU → NVENC, with no CPU fallback: an NVENC that cannot encode them gives `ERR encode: <codec> unavailable: <why>`. Segments are cached per codec. The OK line names the codec only when `--codec` is passed.
- `montaj export --target youtube|shorts|reels|whatsapp|master` — with a `montaj.yaml`, renders `out/<target>.mp4` from the float compositor using the preset table `TARGETS` in `montaj/encode.py` (codec, cq, bitrate cap, width, LUFS, AAC kbps, tags); the OK line echoes it (`target=youtube hevc10 10-bit cq18 max40M -14LUFS aac256k bt709`). 8-bit targets (`reels`, `whatsapp`) get an ordered dither on the GPU before rounding. Without a spec it copies (`master`) or transcodes the newest `out/` file.
- Every output is tagged bt709 primaries/transfer/matrix, limited range. `montaj check` prints `OK video <codec> <bits>-bit <pix_fmt> bt709` from the file, `WARN` on missing tags; on `out/<target>.mp4` it also pins the preset's codec, bit depth and LUFS target.

## Check and validate messages
`montaj check` and `montaj validate` print `WARN <kind> ...`; `montaj spec --search <kind>` finds it here. Caption findings name `text[i]` (the block's index in `text:`; `montaj validate --expand` shows the list after reuse) or `captions[N]` (chunk N of `montaj captions --print`), the time span, the text, the value and the fix. `check --json` adds `findings: [{kind, level, path, start, end, text, value, limit, message}]`; non-caption findings fill only `kind`, `level` and `message`.
- `caption-rate` — check WARN — a reading window of contiguous caption chunks (at least 5/6 s on screen, read in time order; a gap where the screen clears ends it) runs over 20 characters per second. Named by the window's densest chunk, with that chunk's own rate. Fix, as printed: `... 32cps (max 20: hold longer or cut words)`, plus `; designed text, not a caption: role: title` on an authored block with no `role`; a `captions` chunk says `(max 20: fast speech here; cut words in the edit, or accept it with check.expect caption)`.
- `caption-safe` — check WARN — the chunk's ink reaches a platform UI zone (`check.safe`) on the named side. Fix, as printed: `... bottom (in the platform UI zone, bottom 22%: move it up, or set check.safe)`; with two sides the fix is `make it smaller or wrap it`; an authored block with no `role` adds `; designed text: role: title`.
- `caption-contrast` — check WARN — fill against what is behind it, on the rendered frame. Fix, as printed: `... ratio 1.2 (min 4.5: add a stroke or a backdrop, or change the colour)`. A stroke at least 4% of the size always passes.
- `caption-face` — check WARN — the chunk covers a detected face. Fix, as printed: `... over a face (move it off the face)`, with `y` / `pos`, or `captions.y`. Needs `uv sync --extra guards`.
- `caption-box` — validate WARN — a caption on a static box (`box: chip` or a solid `backdrop`). Fix: stroke + shadow; designed text: `role: title`.
- `caption-weight` — validate WARN — a caption lighter than weight 600. Fix: `weight: 700` or a display `font`; designed text: `role: title`.
- `caption-size` — validate WARN — a caption under 5% of the frame height. Fix: a larger `size`; designed text: `role: title`.
- `expect` — check INFO — a `check.expect` span matched nothing; the bracket says what was nearest: for black and frozen the nearest detected span or one under the minimum, for caption the nearest caption finding (`nearest finding caption-rate text[4] 3.5-4.0s '...'`), else the nearest chunk. Fix the span's times, or delete it when the film no longer has that span.
- `black` — check WARN — near-black frames for at least 0.10 s that no `check.expect` black span, fade or dark blank accounts for. Fix: the shot or transition there; designed: `check.expect: [{black: [a, b], why}]`.
- `frozen` — check WARN — an unchanging picture for at least 0.5 s that is not a `hold`, a photo or a `check.expect` frozen span. Fix: motion there (camera, `drift`); designed: `check.expect: [{frozen: [a, b], why}]`.
- `video` — check WARN — the file's codec, bit depth or colour tags differ from what the render asked for. Fix: re-render; report it as an engine bug when it repeats.
- `click` — check WARN — an audio step at a join, larger than the audio around it. Fix: a crossfade or a cut on a quieter beat there.
- `loudness` — check WARN — integrated loudness off the target (-14 LUFS ± 1 by default). Fix: `audio` gains; another target: `check.loudness`.
- `av-offset` — check WARN — audio and video streams start apart by more than the tolerance. Fix: re-render; check `audio.clips[].at` offsets.
- `pacing` / `hook` — check WARN — a `check.pacing` target missed. Fix: shorter shots, more cuts or earlier text; or relax the target.
