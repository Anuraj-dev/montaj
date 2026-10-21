#!/usr/bin/env bash
# Generate the synthetic demo assets for recipes/tech-launch.yaml into a project made from it.
# `montaj new demo --recipe tech-launch` (no --photos) runs it as `tech-launch-demo.sh demo`, with MONTAJ set to
# the venv's montaj; by hand: `recipes/tech-launch-demo.sh demo` on a project holding montaj.yaml.
# Writes assets/ui1-4.mp4 (+ events), assets/voice1-2.wav, music/song.wav. Nothing here is committed binary.
set -euo pipefail
proj="${1:?usage: tech-launch-demo.sh PROJECT_DIR}"
here="$(cd "$(dirname "$0")" && pwd)"
mt=(${MONTAJ:-uv run --project "$here/.." montaj})
font="$(fc-match -f '%{file}' sans)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
mkdir -p "$proj/assets" "$proj/music"

# name:label:hue:event-time:target  (a fake app: sidebar, header, cards, a sweeping highlight)
for spec in ui1:VOICE:0:1.2:voice ui2:THEMES:90:1.0:themes ui3:SEARCH:180:1.5:search ui4:SYNC:270:1.1:sync; do
  IFS=: read -r name label hue et target <<<"$spec"
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i "color=c=0x101a2e:s=1920x1080:r=60:d=8" \
    -f lavfi -i "color=c=0x5de0ff:s=420x70:r=60:d=8" \
    -filter_complex "[0]drawbox=0:0:300:1080:0x16233d:fill,drawbox=300:0:1620:110:0x1b2c4d:fill,\
drawbox=360:170:560:360:0x1f3358:fill,drawbox=960:170:560:360:0x1f3358:fill,drawbox=1560:170:300:360:0x1f3358:fill,\
drawbox=360:560:1500:440:0x182a49:fill,drawtext=fontfile=$font:text='$label':fontsize=64:fontcolor=white:x=360:y=30,\
drawtext=fontfile=$font:text='lumen':fontsize=44:fontcolor=0x5de0ff:x=40:y=36[a];\
[1]format=rgba[bar];[a][bar]overlay=x='360+1100*abs(sin(t*1.3))':y=600[b];\
[b]hue=h=$hue,format=yuv420p[v]" \
    -map "[v]" -c:v libx264 -preset veryfast -crf 24 -g 30 -pix_fmt yuv420p "$tmp/$name.mp4"
  printf '[{"t": %s, "kind": "click", "target": "%s"}]\n' "$et" "$target" >"$tmp/$name.events.json"
  "${mt[@]}" -C "$proj" ingest-rec "$tmp/$name.mp4" --events "$tmp/$name.events.json" >/dev/null
done

# placeholder voice lines: a short fluttering tone burst each
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "sine=f=190:d=1.4:r=48000" -af "tremolo=f=5:d=0.8,afade=t=out:st=1.1:d=0.3,volume=0.7" "$proj/assets/voice1.wav"
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "sine=f=170:d=1.6:r=48000" -af "tremolo=f=4:d=0.8,afade=t=out:st=1.3:d=0.3,volume=0.7" "$proj/assets/voice2.wav"

cp "$here/music/tech-launch-compose.py" "$proj/music/compose.py"
"${mt[@]}" -C "$proj" music compose music/compose.py
