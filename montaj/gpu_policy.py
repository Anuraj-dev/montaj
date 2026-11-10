"""GPU policy: where pixels may be made, passed as an argument, never read from a global.

`[gpu] strict = true` in the machine config makes every CPU pixel path an error instead of a
fallback; `device` pins the CUDA device so an integrated GPU is never picked. Each pixel seam
(decode, text, photo, encode) asks the policy once per decision: `gpu(stage)` when it takes the GPU
path, `cpu(stage, what, hint)` when it would take a CPU one. `cpu` raises `GpuStrictError` under
strict and otherwise records the fallback, so the `render` result line can say `gpu=all` or
`gpu=partial(text,decode)` and a test can assert which seams ran (`policy.seams`).

Not pixel work, so not policed: text shaping, demux/mux, audio mix, Python orchestration. Named
exception: `capture` rasterises the page in headless Chromium (CPU, SwiftShader); only its encode
is policed.
"""
from __future__ import annotations

import os
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

STRICT_ENV = "MONTAJ_GPU_STRICT"
STAGES = ("decode", "photo", "text", "encode")  # report order


class GpuStrictError(RuntimeError):
    """A CPU pixel path under `strict`. `str(exc)` is the whole `ERR` payload: `gpu-strict: ...`."""

    def __init__(self, what: str, hint: str) -> None:
        super().__init__(f"gpu-strict: {what} would run on the CPU: {hint}")
        self.what, self.hint = what, hint


@dataclass
class Seams:
    """Counts of seam decisions, `(stage, "gpu"|"cpu") -> n`, plus the first cpu detail per stage."""

    counts: Counter = field(default_factory=Counter)
    notes: dict[str, str] = field(default_factory=dict)

    def n(self, stage: str, where: str) -> int:
        return self.counts[(stage, where)]


@dataclass(frozen=True)
class GpuPolicy:
    """Frozen settings plus one mutable `Seams` tally owned by this instance (not module state)."""

    strict: bool = False
    device: str = "cuda"
    seams: Seams = field(default_factory=Seams, compare=False, repr=False)

    def gpu(self, stage: str) -> None:
        self.seams.counts[(stage, "gpu")] += 1

    def cpu(self, stage: str, what: str, hint: str) -> None:
        """A seam is about to use the CPU: raise under strict, else count it."""
        if self.strict:
            raise GpuStrictError(what, hint)
        self.seams.counts[(stage, "cpu")] += 1
        self.seams.notes.setdefault(stage, what)

    def cpu_stages(self) -> list[str]:
        used = {stage for (stage, where), n in self.seams.counts.items() if where == "cpu" and n}
        return [s for s in STAGES if s in used] + sorted(used - set(STAGES))

    def summary(self) -> str:
        """`all`, or `partial(text,decode)` naming the stages that fell back to the CPU."""
        stages = self.cpu_stages()
        return f"partial({','.join(stages)})" if stages else "all"


def resolve_device(spec: str) -> str:
    """`cuda`, `cuda:N`, a GPU UUID or a name substring -> `cuda:N`. ValueError when nothing matches.

    Pinning by UUID/name is what keeps an integrated GPU (or a second card) from being picked.
    """
    import torch

    if not torch.cuda.is_available():
        raise ValueError(f"device {spec!r}: no CUDA device")
    text = spec.strip()
    n = torch.cuda.device_count()
    if text in ("cuda", ""):
        return f"cuda:{torch.cuda.current_device()}"
    if text.startswith("cuda:") and text[5:].isdigit():
        if int(text[5:]) >= n:
            raise ValueError(f"device {spec!r}: only {n} CUDA device(s)")
        return text
    want = text.lower().removeprefix("gpu-")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        if want == str(getattr(props, "uuid", "")).lower() or want in props.name.lower():
            return f"cuda:{i}"
    raise ValueError(f"device {spec!r}: no CUDA device with that name or UUID")


def load_policy(path: Path | None = None) -> GpuPolicy:
    """Policy from `[gpu]` in the machine config (absent file or table = defaults, not strict).

    `MONTAJ_GPU_STRICT=1|0` overrides `strict` (CI, one-off runs). The device is resolved lazily by
    the caller via `resolve_device`; a bad `device` surfaces there as one ERR line.
    """
    from montaj.config import config_path

    cfg = path or config_path()
    table: dict = {}
    if cfg.is_file():
        try:
            table = tomllib.loads(cfg.read_text(encoding="utf-8")).get("gpu", {})
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise RuntimeError(f"config: {cfg} is not readable TOML: {' '.join(str(exc).split())}") from None
    if not isinstance(table, dict) or set(table) - {"strict", "device"}:
        raise RuntimeError(f"config: [gpu] in {cfg} takes only strict (bool) and device (string)")
    strict = table.get("strict", False)
    device = table.get("device", "cuda")
    if not isinstance(strict, bool) or not isinstance(device, str):
        raise RuntimeError(f"config: [gpu] in {cfg}: strict must be true/false, device a string")
    env = os.environ.get(STRICT_ENV, "").strip()
    if env in ("0", "1"):
        strict = env == "1"
    return GpuPolicy(strict=strict, device=device)
