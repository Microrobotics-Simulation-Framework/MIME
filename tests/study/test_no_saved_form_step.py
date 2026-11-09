"""The sweep takes no saved-form step: a study's points and a run's overrides live in run folders and results, never in
the saved form, so ``CURRENT`` is the saved form of the release branch's tip this branch merges into, but for the one
step a design row allows by name: form 11, a medium's fields written by role (the solved-field contract's SF-1; the
FVM's declaration, F0's D2 F1a), taken from form 10. A later step is not the sweep's to take, and fails here until a
design row allows it by name and this guard says so. The tip is read from ``origin/release/v0.3.0`` where the checkout holds it, or from the first
parent of a pull request's merge commit, which is that tip, as CI checks a pull request out."""

import re
import subprocess
from pathlib import Path

import pytest

from mime.saved._versions import CURRENT

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = "src/mime/saved/_versions.py"
#: The one saved-form step a branch may take beside the sweep: from form 10 to form 11 (a medium's fields by role).
ALLOWED_STEP = (10, 11)


def _git(*args: str) -> str | None:
    done = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)
    return done.stdout if done.returncode == 0 else None


def current_of(source: str) -> int | None:
    found = re.search(r"^CURRENT = (\d+)$", source, re.MULTILINE)
    return int(found[1]) if found else None


def allowed(current: int, held: int) -> bool:
    """Whether a branch whose saved form is ``current`` may merge into a tip whose form is ``held``: the tip's form, or
    the one step allowed by name. Nothing the branch writes beside it (a version's paragraph, a form test) counts."""
    return current == held or (held, current) == ALLOWED_STEP


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


def test_the_saved_form_is_the_tips_or_the_one_step_allowed_by_name():
    found = tip_current()
    if found is None:
        pytest.skip("the checkout holds neither origin/release/v0.3.0 nor a merge commit whose first parent is its tip")
    where, held = found
    assert held is not None, f"no CURRENT in {where}'s {VERSIONS}"
    assert allowed(CURRENT, held), (
        f"CURRENT is {CURRENT} here and {held} on {where}: the sweep takes no saved-form step, and the one step allowed "
        f"by name is {ALLOWED_STEP[0]} to {ALLOWED_STEP[1]} (a medium's fields by role)")


def test_the_control_a_form_the_tip_does_not_hold_is_seen():
    """The reading the comparison rests on: ``CURRENT`` read from the module's own source, and a source one form on."""
    source = (ROOT / VERSIONS).read_text(encoding="utf-8")
    assert current_of(source) == CURRENT
    assert current_of(source.replace(f"CURRENT = {CURRENT}\n", f"CURRENT = {CURRENT + 1}\n")) == CURRENT + 1


@pytest.mark.parametrize(("current", "held", "passes"), [
    (11, 10, True), (11, 11, True), (10, 10, True),
    (12, 11, False), (12, 10, False), (11, 9, False), (10, 9, False), (10, 11, False)])
def test_the_control_only_the_step_named_passes(current: int, held: int, passes: bool):
    """A step to 12 fails, whatever the branch records beside it: the form 12 paragraph and an empty
    ``tests/saved/test_form12.py`` are not read. A branch behind the tip fails too."""
    assert allowed(current, held) is passes
