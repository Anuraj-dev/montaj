# Spec reference

Photo fields (`crop`, `focus`, swirl `center`, wall `at`) are source-image pixels. `zoom`, the wall grid and leak size are design pixels of a 1080-wide frame, so a half-size preview matches the final framing. Durations are `22f` (frames), `4b` (beats at `video.bpm`) or `1.5s`; a bare number is an error.

- `video.size` — string, required — output `WxH`, both numbers even. Preview halves each side, rounded down to even.
- `video.fps` — integer, required — frames per second.
- `video.bpm` — number, default none — beats per minute. Required when any duration uses `b`.
- `video.look` — string, required — finish preset applied to every frame. See looks below.
- `video.intro` — mapping, default none — opening treatment on the first frames.
- `video.intro.type` — string, required with intro — `white-lift`: a white bloom that settles onto the picture, plus a short lift from white.
- `video.intro.dur` — duration, required with intro — how long the opening bloom lasts.
- `video.outro` — mapping, default none — ending treatment. Starts at `n_frames - dur`.
- `video.outro.type` — string, required with outro — `glow`: a bloom and warm rim that grow and then stay.
- `video.outro.dur` — duration, required with outro — how long the ending lasts.
- `video.motion_blur` — integer, default `1` — sub-frames averaged into each frame. `1` is off. Preview uses at most 2.
- `assets` — string, default `assets` — photo folder relative to the spec file. A photo is the file stem (`jpg`, `jpeg`, `png` or `webp`).
- `shots` — list, required — played in order. Each shot starts when the previous one ends.
- `shots[].photo` — string, photo or wall — stem of the still. An integer in the YAML is read as a string. Mutually exclusive with `wall`.
- `shots[].hold` — duration, required — how long this shot is on screen.
- `shots[].crop` — `[x0, y0, x1, y1]`, default the full image — the part of the photo that is shown.
- `shots[].focus` — `[x, y]`, default the crop centre — the photo point placed at the screen centre.
- `shots[].zoom` — number, default cover — design pixels per photo pixel. Cover fits the crop with a hair of overscan.
- `shots[].in` — mapping, default a cut — how this shot arrives from the previous one. See transitions.
- `shots[].in.type` — `cut`, `fade`, `swirl` or `whip`, default `cut` — which transition.
- `shots[].in.dur` — duration, required except for a cut — length of the transition. It has to fit in the shots it overlaps.
- `shots[].in.center` — `[x, y]`, default the outgoing shot's focus — where a swirl opens, in the outgoing photo.
- `shots[].in.axis` — `x` or `y`, default `x` — which way a whip travels.
- `shots[].flash` — number, default none — white flash on this shot's first frame, fading out over the next 6 frames.
- `shots[].wall` — mapping, photo or wall — a flat field of prints. The camera only pans and zooms. Mutually exclusive with `photo`.
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
- `fx` — list, default none — light leaks and flashes at absolute frame times, not tied to a shot.
- `fx[].leak` — number, leak or flash — strength of a warm light leak around `at`.
- `fx[].flash` — number, leak or flash — strength of a white flash at `at`, fading over 6 frames.
- `fx[].at` — duration, required — when it happens, in frames from the start of the film.
- `fx[].dir` — `1` or `-1`, required with a leak — which side the leak comes in from. Not used on a flash.

Transitions (`shots[].in`):

- `cut` — the new shot replaces the old one on its first frame. Takes no extra frames.
- `fade` — crossfade across the first `dur` frames of the incoming shot.
- `swirl` — a swirling wipe that opens around `center`, across the first `dur` frames of the incoming shot.
- `whip` — both shots slide along `axis`. The move is `dur` frames long and centred on the cut, half on each side.

Looks (`video.look`):

- `warm-film` — a little sharpen, softer contrast, lifted blacks, a vignette, and amber light leaks.

Easing (the curve into a camera key):

- `linear` — constant speed.
- `smooth` — gentle slow start and slow end. This is the default.
- `cubic` — a stronger slow start and slow end.
- `outc` — leaves quickly and settles into the key.
- `inc` — creeps, then arrives quickly.
- `expo` — very slow, then very fast, then very slow.
- `ramp` — slow, fast through the middle, then slow again.
