"""Adobe/Resolve `.cube` LUT parser. numpy only, so spec validation stays torch-free."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_3D = 65  # 65^3 x 3 floats = 3.3 MB; larger cubes only slow the render
MAX_1D = 65536


@dataclass(frozen=True)
class Cube:
    """`table` is (N, 3) for a 1D LUT, or (N, N, N, 3) indexed [b][g][r] for a 3D LUT.

    The file lists red fastest, so a plain C-order reshape gives that layout.
    """

    dims: int
    size: int
    table: np.ndarray
    dmin: tuple[float, float, float]
    dmax: tuple[float, float, float]


def _triple(words: list[str], key: str) -> tuple[float, float, float]:
    if len(words) != 3:
        raise ValueError(f"{key} needs three numbers")
    vals = tuple(float(w) for w in words)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{key} must be finite")
    return vals  # type: ignore[return-value]


def parse_cube(text: str) -> Cube:
    """Raises ValueError with a one-line reason on any malformed or unsupported file."""
    size1 = size3 = None
    dmin = (0.0, 0.0, 0.0)
    dmax = (1.0, 1.0, 1.0)
    rows: list[list[float]] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        words = line.split()
        head = words[0]
        try:
            if head[0].isalpha():
                key = head.upper()
                if key == "TITLE":
                    continue
                if key == "LUT_3D_SIZE" and len(words) == 2:
                    size3 = int(words[1])
                elif key == "LUT_1D_SIZE" and len(words) == 2:
                    size1 = int(words[1])
                elif key == "DOMAIN_MIN":
                    dmin = _triple(words[1:], key)
                elif key == "DOMAIN_MAX":
                    dmax = _triple(words[1:], key)
                else:
                    raise ValueError(f"unsupported keyword {head!r}")
            else:
                vals = [float(w) for w in words]
                if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
                    raise ValueError("a data row is three finite numbers")
                rows.append(vals)
        except ValueError as e:
            raise ValueError(f"line {n}: {e}") from None
    if (size1 is None) == (size3 is None):
        raise ValueError("need exactly one of LUT_3D_SIZE or LUT_1D_SIZE")
    dims, size = (3, size3) if size3 is not None else (1, size1)
    limit = MAX_3D if dims == 3 else MAX_1D
    if size < 2 or size > limit:
        raise ValueError(f"LUT_{dims}D_SIZE {size} outside 2..{limit}")
    want = size**3 if dims == 3 else size
    if len(rows) != want:
        raise ValueError(f"LUT_{dims}D_SIZE {size} needs {want} rows, found {len(rows)}")
    if any(hi <= lo for lo, hi in zip(dmin, dmax)):
        raise ValueError("DOMAIN_MAX must exceed DOMAIN_MIN")
    arr = np.asarray(rows, dtype=np.float32)
    if dims == 3:
        arr = arr.reshape(size, size, size, 3)
    return Cube(dims, size, arr, dmin, dmax)


def load_cube(path: Path | str) -> Cube:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ValueError("not a text .cube file") from None
    return parse_cube(text)
