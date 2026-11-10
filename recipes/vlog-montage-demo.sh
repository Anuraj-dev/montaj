#!/usr/bin/env bash
# Generate the synthetic demo assets for recipes/vlog-montage.yaml into a project made from it.
# `montaj new demo --recipe vlog-montage` (no --photos) runs it as `vlog-montage-demo.sh demo`, with MONTAJ set to
# the venv's montaj; by hand: `recipes/vlog-montage-demo.sh demo` on a project holding montaj.yaml.
# Writes assets/c1-c8.mp4 (10 s 1280x720 @ 30, scaled up by cover, each with its own ambience track) and music/song.wav.
# Every source is an ffmpeg lavfi generator with fixed parameters, so the output is deterministic. Nothing is committed.
set -euo pipefail
proj="${1:?usage: vlog-montage-demo.sh PROJECT_DIR}"
here="$(cd "$(dirname "$0")" && pwd)"
mt=(${MONTAJ:-uv run --project "$here/.." montaj})
mkdir -p "$proj/assets" "$proj/music"

# name : sky top : sky bottom : ground : mover colour : mover px/s (1080p) : hue drift deg/s : noise colour : tone Hz
# A moving colour sky, a ground band, a "vehicle" box crossing the frame and film-ish noise; ambience is coloured
# noise plus a soft tone, so the linked-audio and L-cut moments are audible in the demo.
for spec in \
  c1:0x2b3a67:0xffb37a:0x3d2c2e:0xfff1d0:260:4:pink:220 \
  c2:0x6fa8c0:0xd8cfa6:0x5b7f3a:0xd94f30:520:0:brown:180 \
  c3:0x1e6091:0x99d98c:0x34623f:0xffffff:700:-6:pink:260 \
  c4:0xf9c74f:0xf3722c:0x6d4c41:0x277da1:420:8:brown:150 \
  c5:0x4d908e:0xe9ff70:0x2d3047:0xff006e:900:0:white:300 \
  c6:0xff7b00:0x8b2fc9:0x231942:0xffd166:350:-10:pink:200 \
  c7:0x14213d:0xfca311:0x000000:0xe5e5e5:600:5:brown:120 \
  c8:0x0b132b:0x5bc0be:0x1c2541:0xffffff:180:-3:pink:240; do
  IFS=: read -r name top bot ground mover speed hue noise tone <<<"$spec"
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i "gradients=s=1280x720:r=30:d=10:c0=$top:c1=$bot:x0=640:y0=0:x1=640:y1=506:speed=0.015:seed=7" \
    -f lavfi -i "color=c=$mover:s=160x60:r=30:d=10" \
    -f lavfi -i "anoisesrc=color=$noise:amplitude=0.18:seed=11:r=48000:d=10" \
    -f lavfi -i "sine=f=$tone:r=48000:d=10" \
    -filter_complex "[0]drawbox=0:506:1280:214:$ground:fill,drawbox=0:503:1280:4:0xffffff@0.25:fill[sky];\
[1]format=rgba[car];[sky][car]overlay=x='mod(t*$speed*2/3,1440)-160':y='460+5*sin(t*9)'[a];\
[a]hue=h='t*$hue',noise=alls=10:allf=t+u,format=yuv420p[v];\
[3]tremolo=f=0.4:d=0.6,volume=0.25[tn];[2][tn]amix=inputs=2:normalize=0,aformat=channel_layouts=stereo,\
afade=t=in:d=0.2[au]" \
    -map "[v]" -map "[au]" -c:v libx264 -preset veryfast -crf 26 -g 30 -pix_fmt yuv420p \
    -c:a aac -b:a 128k -shortest "$proj/assets/$name.mp4"
done

cp "$here/music/vlog-montage-compose.py" "$proj/music/compose.py"
"${mt[@]}" -C "$proj" music compose music/compose.py
