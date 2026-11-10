#!/usr/bin/env bash
# Generate the synthetic demo assets for recipes/talking-short.yaml into a project made from it.
# `montaj new demo --recipe talking-short` (no --photos) runs it as `talking-short-demo.sh demo`, with MONTAJ set
# to the venv's montaj; by hand: `recipes/talking-short-demo.sh demo` on a project holding montaj.yaml.
# Writes assets/voice.mov (+ voice.words.json, voice.cuts.json), assets/broll1-2.mp4 and talk.words.json.
# Nothing here is committed binary.
set -euo pipefail
proj="${1:?usage: talking-short-demo.sh PROJECT_DIR}"
here="$(cd "$(dirname "$0")" && pwd)"
mt=(${MONTAJ:-uv run --project "$here/.." montaj})
py=(uv run --project "$here/.." python)
if [[ -n "${MONTAJ:-}" && -x "$(dirname "$MONTAJ")/python" ]]; then py=("$(dirname "$MONTAJ")/python"); fi

"${py[@]}" "$here/talking-short.py" demo "$proj"
# The words file is already there, so this plans the cuts from it and the clip's audio (no transcription).
"${mt[@]}" -C "$proj" cut voice --lang en
"${py[@]}" "$here/talking-short.py" words "$proj"
