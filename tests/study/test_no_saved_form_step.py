"""The sweep takes no saved-form step: a study's points and a run's overrides live in run folders and results, never in
the saved form, so ``CURRENT`` is the saved form of the release branch's tip this branch merges into, or one form on
where the branch takes a step of its own and records it as every step is recorded: its version's paragraph in
``_versions.py`` (``**Version N**``) and its form test (``tests/saved/test_formN.py``). The tip is read from
``origin/release/v0.3.0`` where the checkout holds it, or from the first parent of a pull request's merge commit,
which is that tip, as CI checks a pull request out."""

import re
import subprocess
from pathlib import Path

import pytest

from mime.saved._versions import CURRENT

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = "src/mime/saved/_versions.py"


def _git(*args: str) -> str | None:
    done = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)
    return done.stdout if done.returncode == 0 else None


def current_of(source: str) -> int | None:
    found = re.search(r"^CURRENT = (\d+)$", source, re.MULTILINE)
    return int(found[1]) if found else None


def recorded_step(source: str, version: int) -> bool:
    """Whether ``source`` (``_versions.py``) records ``version`` as a saved-form step: its paragraph, beside its form
    test."""
    return f"**Version {version}**" in source and (ROOT / "tests" / "saved" / f"test_form{version}.py").is_file()


def tip_current() -> tuple[str, int | None] | None:
    """The tip's ``CURRENT`` and where it was read, or ``None`` where the checkout holds neither."""
    for ref in ("origin/release/v0.3.0",):
        held = _git("show", f"{ref}:{VERSIONS}")
        if held is not None:
            return ref, current_of(held)
    parents = (_git("rev-list", "--parents", "-n", "1", "HEAD") or "").split()
    if len(parents) == 3:
        held = _git("show", f"{parents[1]}:{VERSIONS}")
        if held is not None:
            return f"{parents[1]} (HEAD's first parent)", current_of(held)
    return None


def test_the_saved_form_is_the_tips():
    found = tip_current()
    if found is None:
        pytest.skip("the checkout holds neither origin/release/v0.3.0 nor a merge commit whose first parent is its tip")
    where, held = found
    assert held is not None, f"no CURRENT in {where}'s {VERSIONS}"
    source = (ROOT / VERSIONS).read_text(encoding="utf-8")
    assert CURRENT == held or (CURRENT == held + 1 and recorded_step(source, CURRENT)), (
        f"CURRENT is {CURRENT} here and {held} on {where}, and no step to it is recorded (its paragraph in {VERSIONS} "
        f"and tests/saved/test_form{CURRENT}.py): the sweep takes no saved-form step")


def test_the_control_a_form_the_tip_does_not_hold_is_seen():
    """The reading the comparison rests on: ``CURRENT`` read from the module's own source, and a source one form on."""
    source = (ROOT / VERSIONS).read_text(encoding="utf-8")
    assert current_of(source) == CURRENT
    assert current_of(source.replace(f"CURRENT = {CURRENT}\n", f"CURRENT = {CURRENT + 1}\n")) == CURRENT + 1


def test_the_control_a_step_the_branch_does_not_record_is_not_taken_for_one():
    """A form one on from the tip passes only as a recorded step: without its paragraph, or without its form test, it
    is the sweep's unrecorded step and fails."""
    source = (ROOT / VERSIONS).read_text(encoding="utf-8")
    assert recorded_step(source, CURRENT)
    assert not recorded_step(source.replace(f"**Version {CURRENT}**", "**Version ?**"), CURRENT)
    assert not recorded_step(source + f"\n**Version {CURRENT + 1}**\n", CURRENT + 1), "no form test for it"
    assert not recorded_step(source, CURRENT + 1)
