# Conventions — Montaj
- Stack: Python ≥3.12 via uv; torch CUDA; ffmpeg (h264_nvenc); headless Chromium for text raster.
- Run the CLI: `montaj <cmd>` (TBD in M1) · Run tests: `uv run pytest` (TBD in M1)
- CLI output: one result line on stdout (`OK <path> <dur> <size> <MB>` / `ERR <what>: <hint>`); progress -> `build/*.log`; `--json` flag everywhere.
- Spec: YAML file `montaj.yaml` in each user project; durations as `4b` (beats), `22f` (frames), `1.5s`. Named markers (`at: chorus1`) arrive with M2 audio; M1 specs use durations only.
- User projects live outside this repo: `<dir>/{montaj.yaml, assets/, music/, build/, out/, review.json}`.
- Git: branch `<type>/<scope>-<desc>`; Conventional Commits; never commit to main directly after the first commit.
