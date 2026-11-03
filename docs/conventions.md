# Conventions — Montaj
- Stack: Python ≥3.12 via uv; torch CUDA; ffmpeg (h264_nvenc); headless Chromium for text raster.
- Run the CLI: `uv run montaj <cmd>` · Tests: `uv run pytest -q` (CPU only: `-m "not gpu"`; run GPU tests serially)
- CLI output: one result line on stdout (`OK <path> <dur> <size> <MB>` / `ERR <what>: <hint>`); progress -> `build/*.log`; `--json` flag everywhere.
- Spec: YAML file `montaj.yaml` in each user project; durations as `4b` (beats), `22f` (frames), `1.5s`; named markers (`at: chorus1`) from `audio.markers`.
- User projects live outside this repo: `<dir>/{montaj.yaml, assets/, music/, build/, out/, review.json}`.
- Git: branch `<type>/<scope>-<desc>`; Conventional Commits; never commit to main directly after the first commit.
