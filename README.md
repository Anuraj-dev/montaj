# Montaj

A CLI and agent skill that turns a small YAML spec into a finished video. See `AGENTS.md` for the
developer map and `docs/SPEC-REFERENCE.md` for the spec.

## Install

```sh
uv sync
uv run montaj skill link   # points ~/.claude/skills and ~/.codex/skills at this checkout's make-video skill
uv run montaj doctor       # the skill line must read OK; re-run skill link after moving the checkout
```

`skill link` only touches skill dirs that already exist and never replaces a real directory.
