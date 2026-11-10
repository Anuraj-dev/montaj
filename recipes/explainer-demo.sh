#!/usr/bin/env bash
# Set up recipes/explainer.yaml's demo in a project made from it.
# `montaj new demo --recipe explainer` (no --photos) runs it as `explainer-demo.sh demo`, with MONTAJ set to
# the venv's montaj; by hand: `recipes/explainer-demo.sh demo` on a project holding montaj.yaml.
# Every picture is code-drawn, so the only "assets" are the draw files (`montaj new` copies recipes/explainer/ to explainer/ beside the spec)
# and the composed score (recipes/music/explainer-compose.py -> music/song.wav). Deterministic; nothing committed.
set -euo pipefail
proj="${1:?usage: explainer-demo.sh PROJECT_DIR}"
here="$(cd "$(dirname "$0")" && pwd)"
mt=(${MONTAJ:-uv run --project "$here/.." montaj})
mkdir -p "$proj/assets" "$proj/music"

cp "$here/music/explainer-compose.py" "$proj/music/compose.py"
"${mt[@]}" -C "$proj" music compose music/compose.py
