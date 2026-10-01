"""The model side of Montaj's audio: ACE-Step candidates and whisper word/beat timings.

Both models run as subprocesses under the interpreters named in `~/.config/montaj/config.toml`
(002 §Audio). This package is the parent side only — it holds no torch, no ctranslate2, no whisper,
so `montaj music …` stays as cheap to start as any other command.

Submodules are the API (`montaj.audio.music`, `montaj.audio.analyze`); they are not re-exported
here, because a re-export named `analyze` or `music` would shadow the module of the same name.
"""
