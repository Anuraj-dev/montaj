explainer: 16:9 45 s explainer, code-drawn diagrams that animate in, kinetic type, lower thirds, composed score.

# explainer: slots and retargeting

`recipes/explainer.yaml`: 45 s, 1920x1080 @ 30, 96 BPM (1 beat = 0.625 s, 72 beats), `clean` look on a navy
`#0b1020` ground. Every picture is drawn by `recipes/explainer/*.py` (`custom:` shots and a custom wipe), so there is
no footage. Demo: `montaj new demo --recipe explainer` (no `--photos`) runs `recipes/explainer-demo.sh demo`, which
copies the draw files to `demo/explainer/` and composes the score (`recipes/music/explainer-compose.py` →
`music/song.wav`); its log is `demo/build/new-demo.log`. Test a draw file alone: `montaj custom test explainer/bars.py`.

## Timeline
| Section | Beats | Time | Picture (`custom:` file) | Text |
|---|---|---|---|---|
| Hook | 0-8b | 0-5 s | `grid.py` drifting dot grid | the question: `slam` words, then an Anton line with a shaking `*emphasis*` |
| Part 1 `load` | 8b-20b | 5-12.5 s | `bars.py` bars grow, dashed capacity line | headline + lower third |
| Part 2 `wait` | 20b-40b | 12.5-25 s | `curve.py` line draws, then a probe slides; `zone` shades past 80 % | typed formula, lower third, `drop` headline at `zone` |
| Part 3 `fix` | 40b-60b | 25-37.5 s | `flow.py` queue backs up with 1 server, drains with 2 at `two` | headline, lower third, `pop` headline |
| Takeaway | 60b-72b | 37.5-45 s | `grid.py` (cyan glow) | caps kicker + rule of thumb with `idle: float`, 1 s fade |

## Ask the user (one round)
1. Topic and the one-sentence takeaway (the last text block). Three parts by default; two or four also work.
2. For each part: what the diagram shows (a ranking → `bars.py`, a relationship → `curve.py`, a process → `flow.py`,
   or a new draw file) and its headline (≤ 6 words, one `*emphasis*` word).
3. Lower-third titles/subtitles (section names, or a speaker's name and role).
4. Voice-over? If yes, add `audio.clips` with `id: voice` and `audio.duck: {under: [voice]}`, and pace holds to it.
5. Accent colours (defaults amber `#ffb84d`, cyan `#4dd6ff`) and music: the composed bed, their track or `music gen`.

## Slots to fill
- Diagram data is `params`, not code: `bars.py` `values` (0..1 of chart height), `highlight` (index), `cap`/`cap_at`;
  `curve.py` `start`/`draw` (s), `drawn: 1` to start complete, `zone`/`zone_at`; `flow.py` `servers`,
  `backlog: [from, to]`, `span`, `rate`. Changing params re-renders only that shot.
- A new diagram: copy the closest file in `explainer/`, keep `ground(ctx)` (its drifting grid stops `check` calling
  a finished diagram frozen) and keep something moving after the build. Draw inside uv x -0.55..1.55, y -0.22..0.62
  so the top headline (`y: 36`) and the lower third (`y: 410`) stay clear. Run `montaj custom test` before render.
- Headlines: one block per beat of the argument, `y: 36`, `size` 40-52; markers `load wait zone fix two takeaway`
  move the shots' text with them. Keep each line under ~34 characters at size 46 (`validate` names overflows).
- Lower thirds: `lower_third: {title, subtitle, y: 410}`; the default `y` (1500) is for 9:16 and is off a 16:9 frame.
- Total holds must stay 72b for 45 s; for another length change `markers.end`, the last hold and the score's `BEATS`.

## Pacing targets (`check.pacing`)
`max_shot: 8s` (diagram shots are 7.5 s; split a longer one into two shots with different `params`),
`min_per10s: 1` (explainers breathe: 1.3 here), `hook_text_by: 0.5s` (the question starts at 0.25 s).

## Retarget to 9:16 (1080x1920)
Set `video.size: 1080x1920`; the frame is 1080x1920 design px and uv x now spans -1..1, y -1.78..1.78.
- In each draw file move the chart box (`X0 X1 BASE HEIGHT` / `X0 X1 Y0 Y1` / `SRC QUEUE SRV_X`) to x -0.85..0.85,
  y -0.4..0.6, or pass them as params.
- Headlines `y: 260`, size 56 with `wrap: balance`; hook lines `y: 700`/`800`; lower thirds drop `y` (default 1500).
Audio, markers and pacing targets are unchanged.
