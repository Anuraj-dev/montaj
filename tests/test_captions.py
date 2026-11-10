"""`captions:` block (issue #23): word timings chunked into ordinary gpu text lines."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from montaj import captions as C
from montaj.cli import main
from montaj.project import segment_hash
from montaj.spec import SpecError, load_spec
from montaj.timeline import resolve

from tests.test_spec import _write

ROOT = Path(__file__).resolve().parents[1]
FPS = 30

# (word, start s, end s): a pause, trailing punctuation, a number, a negation and a long content word.
TALK = [
    ("Okay,", 0.00, 0.40), ("so", 0.43, 0.60), ("this", 0.62, 0.90), ("is", 0.92, 1.00),
    ("a", 1.02, 1.10), ("demonstration.", 1.12, 1.90), ("I", 2.60, 2.70), ("have", 2.72, 2.95),
    ("never", 2.97, 3.30), ("seen", 3.32, 3.60), ("3", 3.62, 3.90), ("people", 3.92, 4.30),
    ("react", 4.32, 4.60), ("like", 4.62, 4.80), ("that?", 4.82, 5.30), ("Wow!", 5.35, 5.90),
]


def _words(tmp: Path, rows=TALK, name: str = "talk.words.json") -> None:
    words = [{"i": n + 1, "w": w, "s": s, "e": e} for n, (w, s, e) in enumerate(rows)]
    (tmp / name).write_text(json.dumps({"words": words}))


def _proj(tmp: Path, caps: str = "{source: talk.words.json}", extra: str = "") -> Path:
    _words(tmp)
    return _write(tmp, "  - {photo: a, hold: 7s}\n", ["a"], extra=f"captions: {caps}\n{extra}")


def _plan(p: Path):
    chunks, fps = C.plan(p)
    return chunks, fps


def test_chunk_starts_within_one_frame_of_word_starts(tmp_path: Path) -> None:
    chunks, fps = _plan(_proj(tmp_path))
    assert chunks and fps == FPS
    for c in chunks:
        assert abs(c.start / fps - (c.words[0].s)) <= 1 / fps
        assert all(a < b for a, b in zip(c.starts, c.starts[1:]))
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:]))


def test_chunking_breaks_on_pause_punctuation_and_max_words(tmp_path: Path) -> None:
    chunks, _ = _plan(_proj(tmp_path, "{source: talk.words.json, words_per_chunk: [1, 3]}"))
    texts = [c.text for c in chunks]
    assert all(len(c.words) <= 3 for c in chunks)
    assert not any("demonstration" in t and "I" in t.split() for t in texts)  # the 0.7 s pause splits them
    assert any(t.endswith("demonstration") for t in texts)  # full stop dropped, word kept
    assert "that?" in " ".join(texts) and "Wow!" in " ".join(texts)  # ? and ! stay
    assert not any(w.endswith((".", ",")) for c in chunks for w in c.text.split())


def test_hard_punctuation_always_breaks_and_comma_waits_for_the_minimum(tmp_path: Path) -> None:
    rows = [("a,", 0.0, 0.2), ("b", 0.22, 0.4), ("c.", 0.42, 0.6), ("d", 0.62, 0.8)]
    _words(tmp_path, rows)
    p = _write(tmp_path, "  - {photo: a, hold: 2s}\n", ["a"],
               extra="captions: {source: talk.words.json, words_per_chunk: [2, 3]}\n")
    chunks, _ = _plan(p)
    assert [c.text for c in chunks] == ["a b c", "d"]  # the comma after one word waits; the full stop breaks


def test_auto_emphasis_picks_numbers_negations_and_long_words(tmp_path: Path) -> None:
    chunks, _ = _plan(_proj(tmp_path))
    marked = {c.words[c.emph].w for c in chunks if c.emph is not None}
    assert "3" in marked and "never" in marked
    none, _ = _plan(_proj(tmp_path, "{source: talk.words.json, emphasis: none}"))
    assert all(c.emph is None for c in none)
    listed, _ = _plan(_proj(tmp_path, "{source: talk.words.json, emphasis: [8]}"))
    assert [c.words[c.emph].w for c in listed if c.emph is not None] == ["have"]


def test_overrides_text_break_and_emphasis(tmp_path: Path) -> None:
    caps = ("{source: talk.words.json, emphasis: none, overrides: ["
            "{word: 6, text: Demo}, {chunk_break_before: 3}, {word: 2, emphasis: true}]}")
    chunks, _ = _plan(_proj(tmp_path, caps))
    assert any(c.words[0].w == "so" for c in chunks)  # word 3 is "this": break before it
    assert any("Demo" in c.text for c in chunks)
    assert any(c.emph is not None and c.words[c.emph].w == "so" for c in chunks)


def test_unmatched_override_is_a_spec_error(tmp_path: Path) -> None:
    p = _proj(tmp_path, "{source: talk.words.json, overrides: [{word: 99, text: x}]}")
    with pytest.raises(SpecError) as ei:
        load_spec(p)
    assert any("captions.overrides[0].word" in e for e in ei.value.errors)


@pytest.mark.parametrize("preset", C.PRESETS)
def test_every_preset_expands_into_gpu_text_blocks(tmp_path: Path, preset: str) -> None:
    p = _proj(tmp_path, f"{{source: talk.words.json, preset: {preset}}}")
    spec = load_spec(p)
    assert spec.captions is None and spec.text
    assert all(ln.engine == "gpu" for b in spec.text for ln in b.lines)
    tl = resolve(spec, p.parent)
    assert tl.texts and all(r.glyphs is not None for t in tl.texts for r in t.lines)
    if preset == "highlight":
        assert any(r.bar is not None for t in tl.texts for r in t.lines)
    else:
        assert all(r.bar is None for t in tl.texts for r in t.lines)


def test_highlight_bar_hugs_the_spoken_word_and_slides(tmp_path: Path) -> None:
    spec = load_spec(_proj(tmp_path, "{source: talk.words.json, preset: highlight}"))
    bars = [ln.bar for b in spec.text for ln in b.lines if ln.bar is not None]
    assert bars and bars[0].from_ is None
    assert any(b.from_ is not None and b.from_ == b.word - 1 for b in bars)


def test_expansion_is_deterministic_and_resolves_from_an_unexpanded_spec(tmp_path: Path) -> None:
    p = _proj(tmp_path, "{source: talk.words.json, preset: karaoke}")
    a, b = load_spec(p), load_spec(p)
    assert a.model_dump_json() == b.model_dump_json()
    raw = load_spec(p, expand=False)
    assert raw.captions is not None
    tl = resolve(raw, tmp_path)  # resolve expands a spec built in code
    assert tl.spec.captions is None and len(tl.texts) == len(a.text)


def test_subs_and_other_segments_hashes_are_unchanged(tmp_path: Path) -> None:
    subs = "subs:\n  - {from: 0s, to: 2s, text: hello}\n"
    p = _proj(tmp_path, extra=subs)
    with_caps = resolve(load_spec(p), tmp_path)
    q = tmp_path / "plain"
    q.mkdir()
    plain = resolve(load_spec(_write(q, "  - {photo: a, hold: 7s}\n", ["a"], extra=subs)), q)
    assert with_caps.subs == plain.subs


def test_no_captions_keeps_the_dump_and_hash_free_of_new_fields(tmp_path: Path) -> None:
    p = _write(tmp_path, "  - {photo: a, hold: 3s}\n", ["a"],
               extra="text:\n  - from: 0s\n    to: 3s\n    lines:\n      - {text: hi, y: 900, at: 0s, engine: gpu}\n")
    dump = load_spec(p).model_dump_json(by_alias=True)
    assert "captions" not in dump and "bar" not in dump


def test_captions_change_the_segment_hash(tmp_path: Path) -> None:
    class Shas(dict):
        def __missing__(self, key):
            return "sha-" + key

    p = _proj(tmp_path)
    tl = resolve(load_spec(p), tmp_path)
    q = tmp_path / "plain"
    q.mkdir()
    plain = resolve(load_spec(_write(q, "  - {photo: a, hold: 7s}\n", ["a"])), q)
    with_h = [segment_hash(tl, s, Shas(), "e1") for s in tl.segments]
    plain_h = [segment_hash(plain, s, Shas(), "e1") for s in plain.segments]
    assert with_h != plain_h


def test_errors_name_the_field(tmp_path: Path) -> None:
    for caps, needle in (
        ("{source: talk.words.json, preset: loud}", "captions.preset"),
        ("{source: nope}", "captions.source"),
        ("{source: talk.words.json, words_per_chunk: [3, 1]}", "captions.words_per_chunk"),
        ("{source: talk.words.json, accent: red}", "captions.accent"),
    ):
        with pytest.raises(SpecError) as ei:
            load_spec(_proj(tmp_path, caps))
        assert any(needle in e for e in ei.value.errors), caps


def test_cli_print_lists_chunk_start_end_text(tmp_path: Path, capsys) -> None:
    _proj(tmp_path)
    assert main(["-C", str(tmp_path), "captions", "--print"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ["chunk", "start", "end", "text"]
    assert out[1].split()[:3] == ["0", "0.00", out[1].split()[2]]
    assert out[-1] == f"OK {len(out) - 2} chunks"


def test_recipe_is_under_ten_lines_and_is_a_valid_block() -> None:
    src = (ROOT / "recipes" / "captions-short.yaml").read_text()
    assert len(src.strip().splitlines()) < 10
    data = yaml.safe_load(src)
    assert data["captions"]["source"] == "voice"


def _voice_film(tmp: Path, shot: str = "{clip: voice, range: [0s, 6s]}", extra: str = "", audio: bool = True) -> Path:
    from tests.test_linked_audio import make_clip

    make_clip(tmp / "assets" / "voice.mov", audio=audio)
    _words(tmp / "assets", name="voice.words.json")
    p = tmp / "montaj.yaml"
    p.write_text(f"video: {{size: 64x64, fps: 30, look: clean}}\nshots:\n  - {shot}\n"
                 f"captions: {{source: voice}}\n{extra}")
    return p


def test_captions_of_a_clip_link_its_voice_when_the_film_has_no_audio(tmp_path: Path) -> None:
    """recipes/captions-short.yaml rendered silent: captions named the clip, nothing linked its sound."""
    spec = load_spec(_voice_film(tmp_path))
    assert spec.shots[0].audio is True and spec.audio is not None and spec.audio.track is None
    from montaj.audio import mix

    tl = resolve(spec, tmp_path)
    plan = mix.plan_audio(tl.spec, tmp_path, tl.markers, FPS, None, 0.0, [], shots=tl.shots)
    assert plan is not None and plan.active and len(plan.links) == 1
    # Pixels do not move: the same film with the shot muted hashes alike.
    muted = tmp_path / "muted"
    muted.mkdir()
    off = resolve(load_spec(_voice_film(muted, "{clip: voice, range: [0s, 6s], audio: false}")), muted)
    sha = {"clip:voice": "x"}
    assert [segment_hash(tl, s, sha, "e1") for s in tl.segments] == [
        segment_hash(off, s, sha, "e1") for s in off.segments]


@pytest.mark.parametrize("shot, extra, audio", [
    ("{clip: voice, range: [0s, 6s], audio: false}", "", True),  # the author muted the shot
    ("{clip: voice, range: [0s, 6s]}", "audio: {clips: [{src: voice.mov, at: 0s}]}\n", True),  # voice placed by hand
    ("{clip: voice, range: [0s, 6s]}", "", False),  # a clip without sound stays a silent film
])
def test_captions_leave_audio_alone_when_the_author_decided(tmp_path: Path, shot: str, extra: str, audio: bool) -> None:
    spec = load_spec(_voice_film(tmp_path, shot, extra, audio))
    assert not spec.shots[0].audio
    assert (spec.audio is None) == (not extra)


def test_transcribe_is_a_callable_that_writes_words(tmp_path: Path, monkeypatch) -> None:
    import montaj.audio.analyze as A

    def fake(wav, out, **kw):
        Path(out).write_text(json.dumps({"words": [{"i": 1, "w": "hi", "s": 0.0, "e": 0.4}], "beats": [1]}))

    monkeypatch.setattr(A, "analyze", fake)
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    out = C.transcribe(clip)
    assert out == tmp_path / "v.words.json"
    assert json.loads(out.read_text()) == {"words": [{"i": 1, "w": "hi", "s": 0.0, "e": 0.4}]}


@pytest.mark.gpu
def test_highlight_bar_is_drawn_behind_the_spoken_word(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from montaj.render.canvas import Canvas
    from montaj.render.frame import Renderer

    p = _proj(tmp_path, "{source: talk.words.json, preset: highlight}")
    tl = resolve(load_spec(p), tmp_path)
    cv = Canvas(*tl.size, device="cuda")
    r = Renderer(tl, tmp_path, cv)
    chunks, fps = _plan(p)
    first = chunks[0].starts[0]

    def yellow(img) -> torch.Tensor:  # #ffe600
        return (img[0] > 0.85) & (img[1] > 0.75) & (img[2] < 0.25)

    with torch.no_grad():
        before = r.frame(round(2.4 * fps))  # the pause before "I": no caption is up
        during = r.frame(first + 8)
    assert int(yellow(before).sum()) == 0
    mask = yellow(during)
    assert int(mask.sum()) > 400
    cols = torch.nonzero(mask.any(dim=0)).flatten()
    assert 0 < float(cols.min()) and float(cols.max()) < during.shape[-1]  # a bar, inside the frame
