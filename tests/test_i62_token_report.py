"""#62 item 5: `tools/token-report.py` ctx0 is the turn-0 context (the first API call's whole input)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _asst(mid: str, inp: int, wr: int, rd: int, out: int) -> dict:
    usage = {"input_tokens": inp, "cache_creation_input_tokens": wr, "cache_read_input_tokens": rd, "output_tokens": out}
    return {"type": "assistant", "message": {"id": mid, "usage": usage}}


def test_ctx0_is_the_first_calls_input(tmp_path: Path) -> None:
    rows = [{"type": "user", "message": {"content": "make a launch short"}},
            _asst("a", 3, 20_000, 7_000, 10), _asst("a", 3, 20_000, 7_000, 10),  # a repeated message id counts once
            _asst("b", 5, 4_000, 27_000, 900)]
    log = tmp_path / "s1.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows))
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "token-report.py"), str(log)],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    cols = out[1].split()
    assert cols[1:4] == ["2", "27k", "31k"]  # turns, ctx0 = 3 + 20000 + 7000, ctxmax = 5 + 4000 + 27000
