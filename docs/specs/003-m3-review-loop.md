# 003 — M3 contract: zero-token review loop
> The human re-renders and reviews without an agent; the agent reads ≤ 300 tokens of structured feedback.
> 001/002 still hold. PLAN.md questions 5, 6 and 8.

## `montaj review [--port 8765] [--video PATH]` (montaj/review/server.py + montaj/review/page.html)
- Python stdlib `http.server` (ThreadingHTTPServer) bound to **127.0.0.1 only**. No new dependencies.
- Serves one self-contained page (inline CSS/JS, no CDN) and the video with HTTP Range support (seeking).
  Default video: newest of `out/preview.mp4`, `out/final.mp4`.
- Page: a vertical-friendly player; under it a shot strip built from the resolved timeline (index, start–end
  seconds, photo/clip stem or `wall`/`blank`, first text line shown in that span); the strip highlights
  the shot under the playhead and seeks on click. Per shot: Approve / Reject toggles and a comment list.
  A comment box stamps the current time and attaches the comment to the shot under the playhead
  (`general` when none). Keyboard: space play/pause, ←/→ one frame, `a` approve, `r` reject, `c` focus comment.
  When the video file's mtime changes (a re-render by `watch`), the page reloads it and keeps the time.
- API (JSON): `GET /api/state` → `{video, video_mtime, fps, n_frames, shots: [...], review}`;
  `POST /api/status {shot, status: "approved"|"rejected"|null}`; `POST /api/comment {t, text}` (server assigns
  the shot from t); `DELETE /api/comment/<id>`. Every mutation rewrites `review.json` atomically
  (temp + `os.replace`).
- `review.json` (project root):
  `{version: 1, spec_sha, video, updated, shots: [{index, start, end, label, status, comments: [{id, t, text}]}],
  general: [{id, t, text}]}`. Times in seconds (2 dp). `spec_sha` = sha256 of montaj.yaml when the review
  started; entries survive a re-render (matched by shot index; a shot whose label changed keeps its notes
  and gets `"stale": true`).
- Result line on start: `OK review http://127.0.0.1:<port> video=<path>`; it runs until Ctrl-C, logs requests
  to `build/review.log`, never prints per request.
- `montaj review --summary`: the agent's view, no server. One line per shot with a status or comments
  (`#12 4.20-6.10 venice REJECT "too dark"; "hold longer"`), then general notes, then
  `OK review <a> approved <r> rejected <c> comments`. Shots with nothing are omitted. Target ≤ 300 tokens
  for a 55-shot film with 10 notes.

## `montaj watch [--final]` (montaj/watch.py)
- Polls `montaj.yaml` and the assets dir (mtime + size, 0.5 s) — no new dependency. On change, debounce
  0.3 s, then validate → render (preview by default) through `pipeline.render`. A spec error prints the
  `ERR` lines and keeps watching; it never exits on a bad save.
- One stdout line per cycle: the render result line, or the ERR lines. Progress goes to `build/watch.log`.
- Holds the GPU lock only while rendering (the render does that itself).
- Ctrl-C exits 0 with `OK watch stopped`.

## `taste.md` (montaj/taste.py)
- Path `~/.config/montaj/taste.md` (env `MONTAJ_TASTE` overrides). First use writes the defaults:
  photos never warp/drift unless the recipe asks; motion lives in transitions; cuts land on the beat grid;
  text is short and centred; end on black.
- `montaj taste` prints it. `montaj taste add "<line>"` appends `- <line> (YYYY-MM-DD)` and prints
  `OK taste <n> lines`. Agents read it at the start of a creative session and add a line when the user
  rejects something for a reason that will recur.

## CLI
New commands `review`, `watch` and `taste`, with the same `-C DIR` and `--json` as 001. In `--json` mode
`review --summary` prints the review.json digest as JSON.
