"""ACE-Step candidate generation. Runs under `ace_python` with `ace_site_packages` on PYTHONPATH
(the dangling-venv trick of `claude-test/bday-video/music/py.sh` lines 3-4); imports nothing from montaj.

Port of `claude-test/bday-video/music/generate.py`:
- line 26: `offload_to_cpu=True, offload_dit_to_cpu=True`, `config_path='acestep-v15-turbo'` — on an
  8 GB card the LM phase OOMs without them, which is why `montaj music gen` holds the GPU lock.
- lines 27-28: `LLMHandler.initialize(..., backend='pt', device='cuda', offload_to_cpu=True)`.
- lines 30-38: `GenerationParams(caption, lyrics, vocal_language, bpm, keyscale, duration, shift=3.0)`.
- lines 41-45: one `GenerationConfig(batch_size=1, audio_format='wav', use_random_seed=True)` per
  candidate, seeded by ACE-Step, reported back as `{path, seed}`.

What changed: each candidate is moved to `<out>/cand-<seed>.wav` with `<out>/cand-<seed>.json`
beside it (params + seed), so a spec can name a candidate and a re-run can reproduce it. Results go
out on a private fd — ACE-Step prints progress on stdout, which must not corrupt the JSON lines.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

DEFAULT_LM = "acestep-5Hz-lm-1.7B"  # generate.py line 22
CONFIG_PATH = "acestep-v15-turbo"  # generate.py line 26
SHIFT = 3.0  # generate.py line 37
SCRATCH = ".ace-scratch"
RESULT_FD = 1


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ACE-Step music candidates -> cand-<seed>.wav/.json")
    ap.add_argument("--ace-dir", required=True, help="ACE-Step checkout; imported and used as cwd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--caption", required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--key", default="D major")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--lm", default=DEFAULT_LM)
    return ap


def _claim_stdout() -> None:
    """Send every later print to stderr and keep the real fd 1 for the result lines."""
    global RESULT_FD
    RESULT_FD = os.dup(1)
    os.dup2(2, 1)


def _emit(obj: dict[str, object]) -> None:
    os.write(RESULT_FD, ("CAND " + json.dumps(obj, ensure_ascii=False) + "\n").encode())


def main(argv: list[str] | None = None) -> int:
    ns = _parser().parse_args(argv)
    _claim_stdout()
    ace = os.path.abspath(ns.ace_dir)
    sys.path.insert(0, ace)  # generate.py lines 7-8
    os.chdir(ace)
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    lyrics = Path(ns.lyrics).read_text(encoding="utf-8")
    out = Path(ns.out).resolve()
    scratch = out / SCRATCH
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    dit, llm = AceStepHandler(), LLMHandler()
    print(dit.initialize_service(project_root=ace, config_path=CONFIG_PATH, device="cuda",
                                 offload_to_cpu=True, offload_dit_to_cpu=True), flush=True)
    print(llm.initialize(checkpoint_dir=os.path.join(ace, "checkpoints"), lm_model_path=ns.lm,
                         backend="pt", device="cuda", offload_to_cpu=True), flush=True)

    params = GenerationParams(
        caption=ns.caption,
        lyrics=lyrics,
        vocal_language=ns.lang,
        bpm=ns.bpm,
        keyscale=ns.key,
        duration=ns.duration,
        shift=SHIFT,
    )
    made = 0
    first_error = None
    for _ in range(ns.n):
        cfg = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
        res = generate_music(dit, llm, params, cfg, save_dir=str(scratch))
        if not res.success:
            err = str(res.error or "generate_music failed")
            print("FAILED", err, flush=True)  # generate.py line 44
            if first_error is None:
                first_error = err
            continue
        for audio in res.audios:
            made += 1
            _save(audio, out, ns)
    print(f"generated {made}/{ns.n} candidates into {out}", flush=True)
    if made < ns.n:
        reason = first_error or "incomplete"
        os.write(RESULT_FD, ("FAIL " + reason + "\n").encode())
        return 1
    return 0


def _save(audio: dict[str, object], out: Path, ns: argparse.Namespace) -> None:
    """Move one generated file to `cand-<seed>.wav` and write its params beside it."""
    params = dict(audio.get("params") or {})
    seed = str(params.get("seed") or uuid4().hex)  # a random id keeps the name unique
    wav = out / f"cand-{seed}.wav"
    os.replace(str(audio["path"]), wav)  # same filesystem: scratch is inside out
    (out / f"cand-{seed}.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "caption": ns.caption,
                "lyrics": Path(ns.lyrics).as_posix(),
                "bpm": ns.bpm,
                "key": ns.key,
                "lang": ns.lang,
                "duration": ns.duration,
                "lm": ns.lm,
                "params": params,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    _emit({"path": wav.as_posix(), "seed": seed})


if __name__ == "__main__":
    sys.exit(main())