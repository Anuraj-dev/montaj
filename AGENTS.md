# Agent entry — Montaj

Montaj is a CLI + skill that lets an AI agent produce a finished video from a small YAML spec, so creative
intent costs few tokens and every re-render / review costs none.

BEFORE doing anything in this repo:
1. Read `docs/STATE.md` — current state (what's done, in progress, gotchas).
2. Skim `docs/INDEX.md` — the map of every doc.

Then read ONLY the further docs your task needs. Do not scan the repo blindly — the docs exist so you don't
have to. `docs/PLAN.md` holds scope and milestones; `docs/decisions.md` holds the why.

At the END of a work session, run `/checkpoint` — it rewrites `docs/STATE.md` and logs the session.

## Rules for agents making a video with Montaj (token budget)
- Take a photo/asset **folder path**. Never ask the user to paste images into the chat.
- Look only at `montaj ingest` / `montaj sheet` images; at most 2 images per iteration.
- In a creative session never read engine source. On an engine bug: report the command + one-line error, stop.
- Run long commands (`render`, `music gen`) in the background and wait for the notification. No polling.
- Ask one brief round first (length, aspect, text, motion style, music). Read and update `~/.config/montaj/taste.md`.
- Start a fresh session between phases (engine dev vs. making a video); `/checkpoint` before `/clear`.

## Rules for agents developing Montaj
- The spec is the product API: every new capability is a spec field documented in `docs/SPEC-REFERENCE.md`
  (one line each), not a new script.
- Commands are quiet: progress goes to `build/*.log`; stdout ends with one result line; `--json` for machines.
- Known traps live in `montaj doctor` output and `docs/research/prior-art.md`; add new traps there.

- Conventions: `docs/conventions.md` · Decisions: `docs/decisions.md` · Complex features: `docs/specs/`
