# 001 — M1 engine contract
> The interfaces every M1 module codes against. Change this file first, then the code.
> Target: `recipes/birthday-short.yaml` renders the Vidyut film (`claude-test/bday-video/v2/film2.py`).

## Pipeline
```
montaj.yaml --load_spec--> Spec --resolve--> Timeline --plan(f)--> FramePlan --render_frame--> RGB tensor
                                                  \--segments--> Segment hash --cache hit? skip : encode chunk--> concat + mux
```
The renderer reads **only** a `FramePlan`, the resolved shot specs it names, and asset files. That makes the
segment hash complete by construction: hash those three things and a cache hit can never be stale.

## Units
- Photo coordinates (`crop`, `focus`, swirl `center`, wall `at`): source-image pixels.
- Screen/world values (`zoom`, whip distance, leak size, wall grid): **design px** of a 1080-wide canvas.
  The renderer multiplies by `Canvas.k = W / 1080`, so a preview at 540x960 looks identical, smaller.
- Time: integer frames after resolution. Spec durations are strings: `22f`, `4b` (beats at `video.bpm`),
  `1.5s`. Seconds and beats round to the nearest frame. A bare number is an error (hint: "use 22f, 4b or 1.5s").

## Spec schema (montaj/spec.py, pydantic v2, `extra="forbid"`)
```yaml
video:
  size: 1080x1920        # WxH, both even
  fps: 30
  bpm: 120               # required if any duration uses "b"
  look: warm-film        # finish preset name
  intro: {type: white-lift, dur: 30f}   # optional
  outro: {type: glow, dur: 110f}        # optional; starts at n_frames - dur
  motion_blur: 6         # sub-frames per frame; 1 = off
assets: assets           # photo dir, relative to the spec file
shots:                   # sequential; shot i starts where shot i-1 ends
  - photo: "16"          # file stem in assets/ (jpg/jpeg/png/webp, case-insensitive); ints coerced to str
    hold: 4b             # required
    crop: [x0, y0, x1, y1]   # default: full image
    focus: [x, y]            # photo point at screen centre; default: crop centre
    zoom: 1.3                # screen px per photo px (design); default: cover = fit_s(...) * 1.005
    in: {type: swirl, dur: 22f, center: [431, 578]}   # transition from the previous shot; default cut
    flash: 0.3               # white flash at this shot's start
  - wall: {...}          # see Wall
    hold: 390f
fx:                      # absolute-time effects
  - {leak: 0.35, at: 70f, dir: 1}     # dir: 1 | -1
  - {flash: 0.3, at: 120f}
```
`load_spec(path) -> Spec` raises `SpecError(errors: list[str])`; each error is one line
`<yaml path>: <problem> (<fix hint>)`, e.g. `shots[3].photo: '99' not in assets/ (have: 09, 10, 12, …)`.

### Transitions (`in:` on the incoming shot)
| type | frames occupied | progress |
|---|---|---|
| `cut` | none | — |
| `fade`, `swirl` | `[start, start+dur)` — the first `dur` frames of the incoming shot | `(t - start) / dur` |
| `whip` | `[start - dur/2, start + dur/2)` — centred on the cut | `(t - (start - dur/2)) / dur` |

- `whip.axis`: `x` (default) or `y`. `swirl.center`: a point in the **outgoing** shot's photo; default its focus.
- Validation: `dur` must fit inside the shots it overlaps (fade/swirl: ≤ incoming hold; whip: half ≤ each side).

### Wall (a flat field of printed photos; the camera only pans and zooms)
```yaml
wall:
  grid: 1500x1950        # cell spacing, world px
  seed: 5                # rng for jitter; draws happen in prints-list order
  jitter: 70             # px; non-straight prints get uniform(-j, j) x then y, then rot uniform(-rot, rot)
  rot: 4                 # degrees
  print_height: 1300     # print scale p = print_height / max(h, w * 1.15) of its crop
  prints:
    - {photo: "34", cell: [-2, 0], straight: true}           # straight: no jitter, rot 0, no rng draws
    - {photo: "10", cell: [0, 0], straight: true, scale: 1.0}  # scale overrides p
    - {photo: "22", cell: [-2, -1], crop: [60, 180, 865, 1330]}
  light: {center: [-1500, 0], radius: [4200, 3000]}   # brightness 0.5 + 0.5 * gaussian
  camera:                # two independent key tracks; t is relative to the shot start
    pos:  [{t: 0f, print: "34", at: [420, 640]}, {t: 110f}, {t: 220f, print: "10", at: [512, 740], ease: cubic}]
    zoom: [{t: 0f, value: 1.606, print: "34"}, {t: 45f}, {t: 110f, value: 0.42, ease: cubic}]
```
A key without a value holds the previous value. `zoom` with `print:` is relative to that print
(`value / p`). `ease` is the easing *into* that key: `linear | smooth | cubic | outc | inc | expo | ramp`
(default `smooth`). The background and noise match `film.py` `wall_bg`.

## Timeline (montaj/timeline.py)
```python
resolve(spec: Spec, spec_dir: Path) -> Timeline
Timeline.fps, .size (W, H), .n_frames
Timeline.shots: list[ResolvedShot]      # .index, .start, .end (exclusive), .kind ('still' | 'wall'), .spec (the Shot model)
Timeline.plan(f: int) -> FramePlan
Timeline.segments: list[Segment]        # .index, .start, .end; split at every shot start, contiguous, cover [0, n_frames)
```
Scene nodes (frozen dataclasses, JSON-serialisable via `dataclasses.asdict`):
```python
Still(shot: int)
Wall(shot: int, t0: int)                # t0 = the wall shot's start (fade-in included); local time = t - t0
Trans(kind: str, a: Scene, b: Scene, t0: float, dur: float, axis: str = "x", center: tuple[float, float] | None = None)
FramePlan(f: int, scene: Scene,
          flashes: tuple[tuple[int, float], ...],          # (at, k) with 0 <= f - at < 6
          leaks: tuple[tuple[int, float, int], ...],       # (at, k, dir) with |f - at| <= 24
          intro: tuple[str, int] | None,                   # (type, dur) when f < dur
          outro: tuple[str, int, int] | None)              # (type, dur, start) when f >= start
```
Scene selection uses the integer frame `f`; the renderer evaluates sub-frame times `t` (motion blur) against
`t0`/`dur`. Shot-level `flash:` becomes a flash at that shot's start.

## Segment hash (montaj/project.py)
```python
find_photo(assets_dir: Path, photo_id: str) -> Path   # FileNotFoundError listing available stems
file_sha(path: Path) -> str                            # sha256 hex of bytes
segment_hash(tl: Timeline, seg: Segment, asset_sha: dict[str, str], engine_version: str) -> str
```
sha256 of canonical JSON (`sort_keys=True`, compact) of: engine version, `video` block, every plan in the
segment, the `model_dump()` of every shot those plans reference, and the sha of every photo those shots use
(wall prints included). Editing one shot changes only the segments whose plans reference it.

## Renderer primitives (montaj/render/) — ports of film.py / film2.py, explicit arguments, no module globals
```python
Canvas(W, H, device="cuda", focal=1400.0)   # .k = W/1080, .XX, .YY, .lownoise, .wall_noise (seeded exactly like film.py)
Photo(cv, path, crop=None, card=False, pad=90)        # film.Img minus the depth channel
camera(cv, T, s, yaw=0, pitch=0); draw(cv, cam, layers, bg=None); fit_s(cv, photo, T, yaw)
still(cv, photo, focus=None, zoom=None) -> (3,H,W)    # film2.still; zoom in design px
easing: smooth, cubic, outc, inc, expo, ramp, linear; key(t, keys); keys_s(t, keys)
transitions: swirl(cv, a, b, x, center_screen), whip(cv, a, b, x, axis, d=1), fade(a, b, x), shift(cv, img, dx, dy)
wall: build_wall(cv, wall_spec, photo_loader) -> WallScene; wall_frame(cv, wall, T, s) ; wpt(print, src)
finish(cv, img, f, look, flashes, leaks, intro, outro) -> (3,H,W) clamp 0..1   # film2.finish, parametrised
LOOKS["warm-film"]   # film2 constants: sharpen .3, s-curve .3, lift .012/.98, vignette .28, leak σ 8f, ...
```
At `W=1080, H=1920` every primitive must match film2.py within mean |Δ| < 1e-3 and max |Δ| < 2/255
(gpu-marked parity tests import film2 from `~/Anuraj-dev/claude-test/bday-video/v2`).

## IO modules (montaj/encode.py, ingest.py, qa.py, sheet.py, doctor.py)
- `encode.Encoder(path, W, H, fps)` context manager; `.write(frame_uint8_HWC)`; NVENC flags from film2
  (`h264_nvenc -preset p7 -tune hq -rc vbr -cq 16 -b:v 0 -profile:v high -pix_fmt yuv420p`), libx264 fallback
  when NVENC is missing. `concat(chunks, out)` (concat demuxer, `-c copy`), `mux(video, audio, out, loudnorm=True)`
  (`loudnorm=I=-14:TP=-1.5:LRA=11`, aac 256k 48k), `export(src, target)` presets: `master` (copy),
  `whatsapp` (720 wide, cq 27, aac 160k).
- `ingest(src_dir, assets_dir) -> list[AssetInfo]`: EXIF-rotate, longest side ≤ 2048, write
  `<assets>/<stem>.jpg` (q 92) + `assets.json` (`stem, w, h, sha`), and one numbered contact sheet
  `build/ingest-sheet.jpg` ≤ 1536 px wide (stem printed on each tile).
- `sheet(video, out, at=None, n=12)`: ≤ 12 frames, one image ≤ 1536 px, timestamp on each tile.
- `check(video) -> list[str]`: duration, size, fps, black frames (`blackdetect`), frozen spans
  (`freezedetect`, ignoring spans the caller marks as intended), integrated loudness, A/V duration offset.
- `doctor() -> list[str]`: CUDA + capability, NVENC, ffmpeg/ffprobe, chromium, free VRAM, plus the known traps.
- All of them log to `build/*.log` and never print progress; the CLI prints the one result line.

## Integration (wave 2): frame renderer, pipeline, CLI
```python
# montaj/render/frame.py
Renderer(tl: Timeline, spec_dir: Path, cv: Canvas)   # loads Photo/WallScene lazily, cached per (stem, crop, card)
Renderer.frame(f: int) -> Tensor (3,H,W) in 0..1     # motion blur: t = f + (k/(n-1) - .5) * .5, k < n; then finish
```
Scene evaluation at sub-frame `t`: `Still` -> `still(cv, photo, focus, zoom)`; `Wall` -> `wall_frame(cv, wall,
*wall_camera(wall, camera, t - t0))`; `Trans` -> progress `x = (t - t0) / dur` into `fade|swirl|whip`; swirl's
`center` (outgoing photo px, default its focus) maps to screen via the outgoing shot's focus and zoom.
`motion_blur: 1` renders one sample at `t = f`.

```python
# montaj/pipeline.py
render(spec_path: Path, mode: "preview" | "final") -> RenderResult(path, n_frames, seconds, size,
                                                                   rendered: int, cached: int)
```
- preview = `video.size` halved (rounded down to even) and `motion_blur: min(2, n)`, applied to the spec
  before `resolve`, so preview and final hash differently by construction.
- Per segment: `segment_hash` -> `build/segments/<hash>.mp4`; hit = skip; miss = render its frames through
  `Encoder` to a temp file, then atomic rename. Then `concat` -> `out/<mode>.mp4`. Progress -> `build/render.log`.
- Engine version string = `montaj.__version__` + a `RENDER_REV` constant in pipeline.py (bump it whenever
  rendering math changes).

### CLI (montaj/cli.py; every command takes `-C DIR` (default cwd) and `--json`)
| Command | Result line |
|---|---|
| `doctor` | the doctor lines, then `OK doctor` / `ERR doctor: <n> failed` |
| `new DIR --recipe NAME --photos FOLDER` | copies `recipes/NAME.yaml` to `DIR/montaj.yaml`, ingests -> `OK DIR/montaj.yaml sheet=DIR/build/ingest-sheet.jpg` |
| `ingest --photos FOLDER` | `OK <n> photos sheet=build/ingest-sheet.jpg` |
| `validate` | `OK montaj.yaml 840f 28.00s 13 shots 13 segments` or one `ERR` line per spec error |
| `render [--preview\|--final]` (default preview) | `OK out/preview.mp4 28.00s 540x960 4.1MB rendered 2/13 cached 11/13 in 31.2s` |
| `sheet [--at 1.5,3,9] [--video PATH]` | `OK build/sheet.jpg 12 frames` (default video: newest of out/final.mp4, out/preview.mp4) |
| `check [--video PATH]` | the QA lines, then `OK check` / `WARN check: <n> warnings` |
| `export --target master\|whatsapp` | `OK out/<target>.mp4 <size>MB` |
Exit code 0 on OK/WARN, 1 on ERR. `--json` prints one JSON object instead of the text lines.
Every documented spec field gets one line in `docs/SPEC-REFERENCE.md`.
