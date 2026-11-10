#!/usr/bin/env bash
# Generate the synthetic demo assets for recipes/gaming-highlight.yaml into a project made from it.
# `montaj new demo --recipe gaming-highlight` (no --photos) runs it as `gaming-highlight-demo.sh demo`, with MONTAJ
# set to the venv's montaj; by hand: `recipes/gaming-highlight-demo.sh demo` on a project holding montaj.yaml.
# Writes assets/match.mp4 + assets/match.events.json (through `montaj ingest-rec --events`), words.json (the comic
# captions' word timings, on the film clock) and music/song.wav (a 120 BPM drum bed).
# Every source is an ffmpeg lavfi generator or fixed text, so the output is deterministic. Nothing is committed.
set -euo pipefail
proj="${1:?usage: gaming-highlight-demo.sh PROJECT_DIR}"
here="$(cd "$(dirname "$0")" && pwd)"
mt=(${MONTAJ:-uv run --project "$here/.." montaj})
font="$(fc-match -f '%{file}' sans:bold)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
mkdir -p "$proj/assets" "$proj/music"

# The "match": 60 s 1280x720 @ 60. A dark arena, a crosshair at the centre and a red target that swings across it,
# crossing the crosshair on every whole second, so each kill time below has the target under the crosshair.
# A white hit flash and a banner mark each kill; a running HUD clock keeps every frame distinct.
kills=(6 12 18 25 31)   # ordinary kills, streak 1..5
ace=44                  # the key moment
win=50                  # round end
flash="" banner=""
for k in "${kills[@]}"; do
  flash+="between(t,$k,$k+0.08)+"; banner+="between(t,$k,$k+1.2)+"
done
flash+="between(t,$ace,$ace+0.08)"; banner="${banner%+}"
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "gradients=s=1280x720:r=60:d=60:c0=0x0b1026:c1=0x2a1650:x0=0:y0=0:x1=1280:y1=720:speed=0.01:seed=3" \
  -f lavfi -i "color=c=0xff2d55:s=120x200:r=60:d=60" \
  -f lavfi -i "anoisesrc=color=brown:amplitude=0.2:seed=5:r=48000:d=60" \
  -filter_complex "[0]drawbox=0:520:1280:200:0x120b24:fill,drawbox=0:516:1280:4:0x00e5ff@0.5:fill,\
drawbox=180:300:140:220:0x1b2448:fill,drawbox=960:260:160:260:0x1b2448:fill[bg];\
[1]format=rgba[tg];[bg][tg]overlay=x='580+420*sin(PI*t)':y=300[a];\
[a]drawbox=x=630:y=358:w=20:h=4:color=white:t=fill,drawbox=x=638:y=350:w=4:h=20:color=white:t=fill,\
drawbox=0:0:1280:720:white@0.75:fill:enable='$flash',\
drawtext=fontfile=$font:text='ELIMINATED':fontsize=56:fontcolor=0xff2d55:borderw=4:x=(w-tw)/2:y=150:enable='$banner',\
drawtext=fontfile=$font:text='ACE':fontsize=120:fontcolor=0xffe600:borderw=6:x=(w-tw)/2:y=120:enable='between(t,$ace,$ace+3)',\
drawtext=fontfile=$font:text='ROUND WON':fontsize=72:fontcolor=0x39e75f:borderw=5:x=(w-tw)/2:y=150:enable='gte(t,$win)',\
drawtext=fontfile=$font:text='%{pts\\:hms}':fontsize=30:fontcolor=white:x=40:y=30,\
drawtext=fontfile=$font:text='HP 100':fontsize=30:fontcolor=0x39e75f:x=40:y=670,\
noise=alls=6:allf=t+u,format=yuv420p[v];[2]aformat=channel_layouts=stereo,volume=0.5[au]" \
  -map "[v]" -map "[au]" -c:v libx264 -preset veryfast -crf 24 -g 60 -pix_fmt yuv420p \
  -c:a aac -b:a 128k -shortest "$tmp/match.mp4"

# The game's event log: seconds from record start. `ingest-rec` rebases it onto the clip clock.
{
  printf '[{"t": 1.0, "kind": "round", "phase": "start", "n": 7}'
  for i in "${!kills[@]}"; do
    printf ',\n {"t": %s.0, "kind": "kill", "streak": %d, "weapon": "rifle"}' "${kills[$i]}" "$((i + 1))"
  done
  printf ',\n {"t": %s.0, "kind": "kill", "streak": 6, "weapon": "pistol", "label": "ace"}' "$ace"
  printf ',\n {"t": %s.0, "kind": "round", "phase": "end", "result": "win", "n": 7}]\n' "$win"
} >"$tmp/match.events.json"
"${mt[@]}" -C "$proj" ingest-rec "$tmp/match.mp4" --events "$tmp/match.events.json" >/dev/null

# Comic-caption word timings on the film clock (stand-in for `montaj transcribe` on a commentary track).
# i:word:start:end. The outro "GG" is a text title, not a caption.
words=(
  "0:WATCH:0.05:0.45" "1:THIS:0.45:0.95"
  "2:one:3.95:4.6" "3:two:5.95:6.6" "4:three:7.95:8.6" "5:four:9.95:10.6" "6:five!:11.95:12.7"
  "7:wait:13.4:13.9" "8:for:13.9:14.2" "9:it:14.2:14.8"
  "10:NO:15.05:15.5" "11:WAY!:15.5:16.2" "12:ACE!:17.0:18.4"
  "13:one:19.2:19.6" "14:more:19.6:20.0" "15:time:20.0:20.8"
  "16:clean:22.2:22.7" "17:pistol:22.7:23.4" "18:ace:23.4:24.0"
)
{
  printf '{"words": ['
  sep=""
  for w in "${words[@]}"; do
    IFS=: read -r i word s e <<<"$w"
    printf '%s\n {"i": %d, "w": "%s", "s": %s, "e": %s}' "$sep" "$i" "$word" "$s" "$e"; sep=","
  done
  printf ']}\n'
} >"$proj/words.json"

# Music: 30 s, 120 BPM. Kick on every beat, a noise hat on the off-beats, a sub bass line.
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "aevalsrc=exprs='0.7*sin(2*PI*(50+90*exp(-30*mod(t,0.5)))*mod(t,0.5))*exp(-9*mod(t,0.5))\
+0.18*sin(2*PI*(55+27.5*lt(mod(t,4),2))*t)*(0.6+0.4*cos(2*PI*t*2))':s=48000:d=30" \
  -f lavfi -i "anoisesrc=color=white:amplitude=0.25:seed=9:r=48000:d=30" \
  -filter_complex "[1]highpass=f=6000,volume='0.6*exp(-40*mod(t+0.25,0.5))':eval=frame[hat];\
[0][hat]amix=inputs=2:normalize=0,aformat=channel_layouts=stereo,volume=0.6,afade=t=in:d=0.05[m]" \
  -map "[m]" -c:a pcm_s16le "$proj/music/song.wav"
