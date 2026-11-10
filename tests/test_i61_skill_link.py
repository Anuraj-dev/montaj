"""`montaj doctor` skill-link check and `montaj skill link` repair (#61). A temp HOME; never the real one."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from montaj import doctor
from montaj.cli import main


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    (h / ".claude" / "skills").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    return h


def _codex(home: Path) -> Path:
    (home / ".codex" / "skills").mkdir(parents=True, exist_ok=True)
    return home / ".codex" / "skills"


def _other_checkout(tmp: Path) -> Path:
    """A second checkout that still exists: a link into it is valid but points at another checkout."""
    skill = tmp / "other-checkout" / "skills" / "make-video"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# other\n")
    return skill


def _warns(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.startswith("WARN skill make-video")]


def test_dangling_link_warns_then_link_repairs_it(home: Path, tmp_path: Path) -> None:
    link = home / ".claude" / "skills" / "make-video"
    link.symlink_to(tmp_path / "gone" / "skills" / "make-video")

    assert doctor.skill_lines() == [f"WARN skill make-video: {link} is dangling (montaj skill link)"]

    assert doctor.link_skill()[-1] == "OK skill make-video linked in 1 dir: " + str(home / ".claude" / "skills")
    assert os.readlink(link) == str(doctor.skill_source())
    assert doctor.skill_lines() == [f"OK skill make-video: {link}"]


def test_wrong_target_warns_and_is_repaired(home: Path, tmp_path: Path) -> None:
    other = _other_checkout(tmp_path)
    link = home / ".claude" / "skills" / "make-video"
    link.symlink_to(other)

    assert _warns(doctor.skill_lines()) == [
        f"WARN skill make-video: {link} points at another checkout {other.resolve()} (montaj skill link)"
    ]

    doctor.link_skill()
    assert os.readlink(link) == str(doctor.skill_source())
    assert _warns(doctor.skill_lines()) == []


def test_correct_link_is_left_alone(home: Path) -> None:
    link = home / ".claude" / "skills" / "make-video"
    link.symlink_to(doctor.skill_source())

    assert doctor.skill_lines() == [f"OK skill make-video: {link}"]
    first = doctor.link_skill()
    second = doctor.link_skill()
    assert first == second == [f"OK skill make-video linked in 1 dir: {home / '.claude' / 'skills'}"]
    assert os.readlink(link) == str(doctor.skill_source())


def test_codex_chain_through_claude_is_relinked_directly_and_missing_is_created(
    home: Path, tmp_path: Path,
) -> None:
    _codex(home)
    claude = home / ".claude" / "skills" / "make-video"
    claude.symlink_to(tmp_path / "gone" / "skills" / "make-video")
    codex = home / ".codex" / "skills" / "make-video"
    codex.symlink_to(claude)  # dangling through the claude link, as in the issue

    assert len(_warns(doctor.skill_lines())) == 2

    lines = doctor.link_skill()
    assert lines[-1].startswith("OK skill make-video linked in 2 dirs: ")
    assert os.readlink(codex) == str(doctor.skill_source())
    assert doctor.skill_lines() == [f"OK skill make-video: {claude}", f"OK skill make-video: {codex}"]

    # A dir with no link yet gets one, and the chain check reports it as missing first.
    (home / ".codex" / "skills" / "make-video").unlink()
    assert doctor.skill_lines()[1] == (
        f"WARN skill make-video: no link in {home / '.codex' / 'skills'} (montaj skill link)"
    )
    assert doctor.link_skill()[-1].startswith("OK skill make-video linked in 2 dirs: ")
    assert os.readlink(codex) == str(doctor.skill_source())


def test_real_directory_is_never_replaced(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    real = home / ".claude" / "skills" / "make-video"
    real.mkdir()
    (real / "SKILL.md").write_text("hand-copied\n")

    assert _warns(doctor.skill_lines()) == [
        f"WARN skill make-video: {real} is not a symlink (remove it, then montaj skill link)"
    ]

    lines = doctor.link_skill()
    assert lines[0] == f"ERR skill make-video: {real} is not a symlink: remove it first"
    assert lines[-1] == "ERR skill link: 1 of 1 dirs not linked (see above)"
    assert not real.is_symlink()
    assert (real / "SKILL.md").read_text() == "hand-copied\n"

    code = main(["skill", "link"])
    out = capsys.readouterr().out.splitlines()
    assert code == 1
    assert out[-1] == "ERR skill link: 1 of 1 dirs not linked (see above)"


def test_missing_skill_dirs_are_skipped_and_fail_the_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = tmp_path / "bare-home"
    empty.mkdir()
    monkeypatch.setenv("HOME", str(empty))

    assert doctor.skill_lines() == []
    assert doctor.link_skill() == [
        "ERR skill link: no ~/.claude/skills or ~/.codex/skills to link into (create one first)"
    ]


def test_cli_skill_link_prints_one_result_line_and_exits_zero(
    home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _codex(home)
    assert main(["skill", "link"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-1].startswith("OK skill make-video linked in 2 dirs: ")
    assert (home / ".claude" / "skills" / "make-video").is_symlink()
    assert (home / ".codex" / "skills" / "make-video").is_symlink()


def test_doctor_report_carries_the_skill_check(home: Path, tmp_path: Path) -> None:
    _codex(home)
    (home / ".claude" / "skills" / "make-video").symlink_to(tmp_path / "gone")

    lines = doctor.doctor()
    dangling = home / ".claude" / "skills" / "make-video"
    assert f"WARN skill make-video: {dangling} is dangling (montaj skill link)" in lines


def test_link_is_left_alone_when_it_resolves_to_this_checkout(home: Path, tmp_path: Path) -> None:
    """A link stored through a symlinked parent still counts as correct: not deleted and recreated."""
    alias = tmp_path / "alias"
    alias.symlink_to(doctor.skill_source().parent)
    link = home / ".claude" / "skills" / "make-video"
    link.symlink_to(alias / "make-video")
    assert doctor.link_skill()[-1].startswith("OK skill make-video linked in 1 dir")
    assert os.readlink(link) == str(alias / "make-video")


def test_unwritable_dir_reports_per_dir_error_and_links_the_rest(home: Path, tmp_path: Path) -> None:
    claude = home / ".claude" / "skills"
    (claude / "make-video").symlink_to(tmp_path / "gone")
    codex = _codex(home)
    claude.chmod(0o555)
    try:
        lines = doctor.link_skill()
    finally:
        claude.chmod(0o755)
    assert any(ln.startswith(f"ERR skill make-video: {claude / 'make-video'}:") for ln in lines)
    assert lines[-1].startswith("ERR skill link: 1 of 2 dirs not linked")
    assert os.readlink(claude / "make-video") == str(tmp_path / "gone")  # old link kept
    assert (codex / "make-video").resolve() == doctor.skill_source().resolve()
