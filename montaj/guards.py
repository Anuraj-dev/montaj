"""Content guards for `montaj check`: on-screen digits (OCR) and faces, sampled per shot.

Declared under `check:` in the spec; absent means nothing here runs. Backends load lazily so the
core install never needs them: digits shell out to the `tesseract` binary, faces use OpenCV's
bundled Haar cascade (`uv sync --extra guards`). A missing backend is one `ERR <guard>: <hint>`
line, never a crash. Sampled frames are decoded from the rendered video into `build/guards/` (cleared each run).
"""
from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from montaj.encode import FFMPEG, run
from montaj.timeline import Timeline

OCR_MIN_CONF = 60.0  # tesseract noise on photos is low-confidence; real text is not
DIGITS_RE = re.compile(r"\d+")

# (frame image) -> digit runs found / number of faces found. Seams for tests.
Ocr = Callable[[Path], list[str]]
Faces = Callable[[Path], int]


def sample_frames(start: int, end: int, n: int) -> list[int]:
    """`n` distinct frames evenly spread across [start, end), at slot midpoints."""
    length = max(end - start, 1)
    frames = {min(start + int((k + 0.5) * length / n), max(end - 1, start)) for k in range(n)}
    return sorted(frames)


def spec_digits(tl: Timeline) -> set[str]:
    """Digit runs the spec itself puts on screen (text lines, subtitles, frame captions)."""
    spec = tl.spec
    texts = [line.text for block in spec.text for line in block.lines]
    texts += [sub.text for sub in spec.subs]
    texts += [sh.frame.caption for sh in spec.shots if sh.frame and sh.frame.caption]
    return {run_ for text in texts for run_ in DIGITS_RE.findall(text)}


def tesseract_ocr(image: Path) -> list[str]:
    """Digit runs in confidently read words; raises RuntimeError when tesseract is absent."""
    exe = shutil.which("tesseract")
    if exe is None:
        raise RuntimeError("install tesseract (pacman -S tesseract tesseract-data-eng)")
    proc = subprocess.run([exe, str(image), "stdout", "--psm", "11", "tsv"],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed ({proc.returncode})")
    found: list[str] = []
    rows = csv.reader(io.StringIO(proc.stdout), delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in rows:
        if len(row) < 12 or row[0] == "level":
            continue
        try:
            conf = float(row[10])
        except ValueError:
            continue
        if conf >= OCR_MIN_CONF:
            found += DIGITS_RE.findall(row[11])
    return found


def opencv_faces() -> Faces:
    """A face counter on OpenCV's frontal Haar cascade; raises RuntimeError when cv2 is absent."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("install OpenCV: uv sync --extra guards") from exc
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))

    def count(image: Path) -> int:
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"cannot decode {image.name}")
        return len(cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(48, 48)))

    return count


def _extract(video: Path, frame: int, fps: int, out: Path, log: Path | str | None) -> bool:
    """Decode one frame to `out`; False when the video ends first (ffmpeg still exits 0 then)."""
    run([FFMPEG, "-hide_banner", "-nostats", "-y", "-ss", f"{frame / fps:.4f}", "-i", str(video),
         "-frames:v", "1", str(out)], log=log, desc="qa")
    return out.is_file() and out.stat().st_size > 0


def content_guards(tl: Timeline, video: Path | str, work: Path, *, log: Path | str | None = None,
                   ocr: Ocr | None = None, faces: Faces | None = None) -> list[str]:
    """`WARN`/`ERR` lines for the declared guards; empty when none is declared or all pass.

    `ocr` / `faces` replace the real backends (tests).
    """
    guards = tl.spec.check
    if guards is None or (guards.digits is None and guards.faces is None):
        return []
    lines: list[str] = []
    count_faces: Faces | None = None
    read_digits: Ocr | None = None
    if guards.digits is not None:
        read_digits = ocr or tesseract_ocr
        if ocr is None and shutil.which("tesseract") is None:
            lines.append("ERR digits: install tesseract (pacman -S tesseract tesseract-data-eng)")
            read_digits = None
    if guards.faces is not None:
        try:
            count_faces = faces or opencv_faces()
        except RuntimeError as exc:
            lines.append(f"ERR faces: {exc}")
    if read_digits is None and count_faces is None:
        return lines
    allowed = spec_digits(tl) if guards.digits == "spec" else set()
    # Fresh dir each run: a leftover image must never stand in for a frame ffmpeg failed to write.
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    for shot in tl.shots:  # blank cards too: their text and subs are rendered overlays
        for f in sample_frames(shot.start, shot.end, guards.sample):
            image = work / f"shot{shot.index}-f{f}.png"
            if not _extract(Path(video), f, tl.fps, image, log):
                lines.append(f"ERR frames: shot={shot.index} frame={f} is past the end of the video; rerender")
                return lines
            try:
                if read_digits is not None:
                    bad = [d for d in read_digits(image) if d not in allowed]
                    if bad:
                        lines.append(f'WARN digits shot={shot.index} frame={f} "{" ".join(bad)}"')
                if count_faces is not None and (n := count_faces(image)):
                    lines.append(f"WARN faces shot={shot.index} frame={f} ({n})")
            except RuntimeError as exc:
                lines.append(f"ERR guards: shot={shot.index} frame={f}: {exc}")
                return lines
    return lines


__all__ = ["content_guards", "sample_frames", "spec_digits"]
