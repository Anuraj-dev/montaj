"""#62 item 6: the make-video skill states the token rules and the shorthands, and stays short."""
from __future__ import annotations

from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / "skills" / "make-video" / "SKILL.md"
MAIN_LINES = 83  # SKILL.md before #62

MUST = [
    "Write the spec once with Write and change it with Edit; never generate it from a script",
    "`montaj new` already created `music/compose.py`",
    "Read it, then Edit",
    "Read one `montaj spec <topic>` at a time",
    "Learn the footage from `montaj assets`, never from `analysis.json`, `*.events.json` or ffmpeg",
    "A 20 s short fits in 60 lines. If yours does not, a shorthand is missing",
    "`{frames: \"m*.png\", every: 6f}`",
    "`- {at: 4b, to: 6b, text: \"SWITCH *WORKSPACES*\"}`",
    "`audio.sfx: [text, cuts]`",
    "`video.outro: {type: fade, dur: 30f, hold: 15f}`",
    "`card: {color: \"#000\"}`",
]


def test_skill_says_the_rules_and_is_shorter() -> None:
    text = SKILL.read_text()
    flat = " ".join(text.split())
    missing = [m for m in MUST if " ".join(m.split()) not in flat]
    assert not missing, missing
    lines = text.splitlines()
    assert len(lines) < MAIN_LINES
    assert [ln for ln in lines if ln.startswith("## ")] == ["## Budget", "## Steps"]  # structure kept
    assert [ln.split(".")[0] for ln in lines if ln[:2] in {f"{i}." for i in range(1, 8)}] == list("1234567")
