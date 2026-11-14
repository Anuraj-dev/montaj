<h1 align="center">Montaj</h1>

<p align="center">
  Write a YAML spec, get a finished video.<br />
  Montaj renders, checks and encodes the film on your GPU, and a re-render redoes only what changed.
</p>

<p align="center">
  <a href="#installation">Install</a>
  ·
  <a href="#quick-start">Quick start</a>
  ·
  <a href="#commands">Commands</a>
  ·
  <a href="#documentation">Docs</a>
</p>

Montaj builds a video from a spec, not from a timeline you drag around. The spec is one file, `montaj.yaml`. It lists the photos and clips, how long each shot holds, the transitions between them, and the text and music on top. `montaj render` turns it into an MP4. `montaj check` then reports black frames, frozen frames and clips that run past their shot.

Montaj is a Python 3.12 command-line tool. It needs an NVIDIA GPU with CUDA and NVENC, ffmpeg, and Chromium for text. It calls no online service and needs no API key. Music generation and transcription run local models that you install yourself and point to in `~/.config/montaj/config.toml`.

## What you get

- **One spec per film.** `montaj.yaml` holds every shot, transition, caption and sound cue. Holds can be counted in beats, so `hold: 4b` lasts four beats of the tempo grid. `montaj validate` prints an `ERR` line for each problem, and each line names the fix.
- **Cached segments.** Each segment is hashed from its frame plan, the shots it names and the asset files it reads. A re-render encodes only the segments whose hash changed. A hold edit still re-renders the segments after it.
- **GPU compositing, NVENC encoding.** Torch composes the frames on the GPU. ffmpeg encodes with `h264_nvenc`, 10-bit `hevc_nvenc` or `av1_nvenc`. Set `strict = true` under `[gpu]` in the config, and any CPU fallback becomes an error instead of a slower path.
- **Numbers before pictures.** `montaj check` looks for black frames, frozen frames and overrunning clips in the render. `montaj inspect --problems` reports text off the frame, overlapping text and low contrast. `montaj scopes` reports the colour of each shot.
- **Cuts on the beat.** `montaj music analyze` finds the beats and words in your track and writes `music/markers.json`. Beat units snap to the grid `video.beat0 + k·60/bpm`, or to the analysed song when the spec sets no `video.bpm`.
- **Captions and cuts from speech.** `montaj transcribe` runs Whisper locally on a clip. `montaj cut` removes pauses longer than 0.35 seconds by default. Captions come in six presets: `punch`, `karaoke`, `highlight`, `clean`, `comic` and `minimal`.
- **Export for each platform.** `montaj export --target` re-encodes the film for `youtube`, `shorts`, `reels`, `whatsapp` or `master`. Each target sets its own codec, bitrate and width.
- **Standing taste.** `montaj taste` prints your saved preferences from `~/.config/montaj/taste.md`, and `montaj taste add` appends one. The `make-video` skill reads them before it writes a spec.

## Installation

Montaj runs from a checkout.

### Requirements

- Python 3.12. `pyproject.toml` requires `>=3.12,<3.13`.
- uv, which creates the environment. The project pins PyTorch 2.9.1 with CUDA 12.8 wheels.
- An NVIDIA GPU with a CUDA driver. `montaj doctor` reports `FAIL cuda` without one.
- `ffmpeg` and `ffprobe` on `PATH`. Set `MONTAJ_FFMPEG` and `MONTAJ_FFPROBE` to use other binaries.
- Google Chrome or Chromium. Text is rasterised in headless Chromium.

### Set up

```sh
git clone https://github.com/Anuraj-dev/montaj.git
cd montaj
uv sync
uv run montaj skill link
uv run montaj doctor
```

`uv sync` installs PyTorch and its CUDA libraries from the PyTorch index. `skill link` points the `make-video` skill at each existing agent skill directory under your home directory. It replaces a symlink and never a real file or directory. The `skill` line of `doctor` must read `OK`. Run `skill link` again after you move the checkout.

## Wait, does it cost anything?

Montaj itself costs nothing to run and needs no key. Everything renders on your GPU. Music generation uses ACE-Step and transcription uses Whisper, both installed on your machine. The real cost is hardware and disk. ACE-Step alone takes about 18 GB. Face detection and the subject matte download their model weights from GitHub and PyTorch's servers on first use, then check them by SHA-256.

## Quick start

```sh
uv run montaj new --list                          # the ten recipes, with aspect and purpose
uv run montaj new tech-film --recipe tech-launch  # no --photos, so the recipe's demo assets come along
uv run montaj -C tech-film render --preview       # writes tech-film/out/preview.mp4
uv run montaj -C tech-film check                  # prints OK check when the film is clean
```

### Your own photos

Pass a folder with `--photos`. Montaj normalises the images into `assets/` and writes a contact sheet to `build/ingest-sheet.jpg`.

```sh
uv run montaj new trip --recipe vlog-montage --photos ~/Pictures/trip-2026
uv run montaj -C trip ingest --list    # one line per analysed photo
```

A photo's stem is its file name without the extension. Recipes refer to photos by stem, so change the shot list in `montaj.yaml` to match your files.

### Review and finish

```sh
uv run montaj -C trip review             # review page on port 8765; notes go to review.json
uv run montaj -C trip review --summary   # the notes as one text list
uv run montaj -C trip render --final
uv run montaj -C trip check
uv run montaj -C trip export --target shorts   # writes out/shorts.mp4
```

## Commands

| Command | What it does |
|---|---|
| `montaj new <dir> --recipe <name> [--photos <folder>]` | Start a project from a recipe. `--list` shows all ten. |
| `montaj validate` | Check `montaj.yaml`. `--expand` prints it with styles and components filled in. |
| `montaj render --preview` or `--final` | Render `out/preview.mp4` or `out/final.mp4`. Segments already encoded are skipped. |
| `montaj watch` | Re-render each time the spec or an asset is saved. |
| `montaj check` | Look for black frames, frozen frames and overrunning clips in the render. |
| `montaj inspect` | Sample frames and tracks. `--problems` lists text off the frame, overlaps and low contrast. |
| `montaj sheet` | Contact sheet of frames, from `--at`, `--every`, `--strip` or `--transitions`. |
| `montaj review` | Review page with notes. `--summary` prints them. |
| `montaj music gen`, `analyze`, `compose` | Generate a song with ACE-Step, analyse a track for beats and words, or run a Python composer. |
| `montaj export --target <name>` | Re-encode the film for `youtube`, `shorts`, `reels`, `whatsapp` or `master`. |
| `montaj spec <topic>` | Print the spec reference one topic at a time. |
| `montaj doctor` | Check CUDA, NVENC, ffmpeg, Chromium, the config and the skill link. |

Each command prints one result line, `OK`, `WARN` or `ERR <what>: <fix>`. Progress goes to `build/<command>.log`. Add `--json` for an object with `status` and `lines`. `-C <dir>` runs a command inside a project.

## Configuration

Montaj reads one machine file, `~/.config/montaj/config.toml`, and never writes it. Only music generation and transcription need it. ACE-Step and Whisper each run in their own Python, because their dependencies are not in Montaj's environment, so the file names those interpreters.

```toml
ace_step_dir = "/path/to/ACE-Step-1.5"
ace_python = "/path/to/cpython-3.12/bin/python"
ace_site_packages = "/path/to/ACE-Step-1.5/.venv/lib/python3.12/site-packages"
whisper_python = "/path/to/whisper-venv/bin/python"

[gpu]
strict = true
```

Without `strict`, a step that falls back to the CPU still runs, and the `render` result line says which steps ran on the CPU.

| Variable | Default | Use |
|---|---|---|
| `MONTAJ_CONFIG` | `~/.config/montaj/config.toml` | Path to the machine config. |
| `MONTAJ_TASTE` | `~/.config/montaj/taste.md` | Path to the standing preferences. |
| `MONTAJ_MODELS` | `~/.cache/montaj/models` | Where downloaded model weights are kept. |
| `MONTAJ_FFMPEG`, `MONTAJ_FFPROBE` | `ffmpeg`, `ffprobe` | The ffmpeg binaries to run. |

## How it works

A spec becomes a timeline of frame plans. The renderer turns each plan into frames on the GPU, and it reads only the plan, the shots it names and their asset files. That rule is what makes the segment cache correct. Text is shaped with HarfBuzz and rasterised by headless Chromium.

## Documentation

- [Spec reference](docs/SPEC-REFERENCE.md): one line per field, transition, look and easing.
- [make-video skill](skills/make-video/SKILL.md): the steps from brief to final, one command at a time.
- [Captions guide](docs/CAPTIONS.md): what a good caption looks like, and the presets.
- [Recipes](recipes/): ten starting specs. Some have a `.md` note beside them.
- [Docs index](docs/INDEX.md) and [roadmap](docs/ROADMAP.md).

## Contributing

There is no CONTRIBUTING file yet. Start with [AGENTS.md](AGENTS.md) for the developer map and [docs/STATE.md](docs/STATE.md) for what exists now. Two rules shape most changes:

- A new capability is a spec field, documented in `docs/SPEC-REFERENCE.md`, not a new script.
- Renderer code reads only the frame plan, the shots it names and asset files. Module-level mutable state breaks the segment cache.

Before a PR, run `uv run pytest -q -m "not gpu"`. On a CUDA machine, also run `uv run pytest -q`.

## License

MIT. See [LICENSE](LICENSE).
