"""The sweep takes no saved-form step: a study's points and a run's overrides live in run folders and results, never in
the saved form, so ``CURRENT`` is the saved form of the release branch's tip this branch merges into, but for the
steps design rows allow by name, each from the form before it: form 11, a medium's fields written by role (the
solved-field contract's SF-1; the FVM's declaration, F0's D2 F1a), form 12, the experiment's declared ranges
(``exp.declare_range``; the maintainer's decision on design row #258's note, 2026-11-08: a declared range lives on the
experiment, saved with it), and form 13, the FVM's setting ``report_rows`` named ``solve_report_rows`` (design row
#303). A branch takes at most one of them past the tip; a later step is not the sweep's to take,
and fails here until a design row allows it by name and this guard says so. The tip is read from
``origin/release/v0.3.0`` where the checkout holds it, or from the first
parent of a pull request's merge commit, which is that tip, as CI checks a pull request out."""

import re
import subprocess
from pathlib import Path

import pytest

from mime.saved._versions import CURRENT

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = "src/mime/saved/_versions.py"
#: The saved-form steps a branch may take beside the sweep, each from the form before it: form 10 to 11 (a medium's
#: fields by role), 11 to 12 (the declared ranges, row #258's note) and 12 to 13 (the FVM's setting renamed, row #303).
ALLOWED_STEPS = ((10, 11), (11, 12), (12, 13))


def _git(*args: str) -> str | None:
    done = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)
    return done.stdout if done.returncode == 0 else None


def current_of(source: str) -> int | None:
    found = re.search(r"^CURRENT = (\d+)$", source, re.MULTILINE)
    return int(found[1]) if found else None


def allowed(current: int, held: int) -> bool:
    """Whether a branch whose saved form is ``current`` may merge into a tip whose form is ``held``: the tip's form, or
    one step allowed by name. Nothing the branch writes beside it (a version's paragraph, a form test) counts."""
    return current == held or (held, current) in ALLOWED_STEPS


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


def test_the_saved_form_is_the_tips_or_one_step_allowed_by_name():
    found = tip_current()
    if found is None:
        pytest.skip("the checkout holds neither origin/release/v0.3.0 nor a merge commit whose first parent is its tip")
    where, held = found
    assert held is not None, f"no CURRENT in {where}'s {VERSIONS}"
    assert allowed(CURRENT, held), (
        f"CURRENT is {CURRENT} here and {held} on {where}: the sweep takes no saved-form step, and the steps allowed "
        f"by name are {', '.join(f'{a} to {b}' for a, b in ALLOWED_STEPS)} (a medium's fields by role; the declared "
        f"ranges, row #258's note; the FVM's setting renamed solve_report_rows, row #303), one at a time")


def test_the_control_a_form_the_tip_does_not_hold_is_seen():
    """The reading the comparison rests on: ``CURRENT`` read from the module's own source, and a source one form on."""
    source = (ROOT / VERSIONS).read_text(encoding="utf-8")
    assert current_of(source) == CURRENT
    assert current_of(source.replace(f"CURRENT = {CURRENT}\n", f"CURRENT = {CURRENT + 1}\n")) == CURRENT + 1


@pytest.mark.parametrize(("current", "held", "passes"), [
    (13, 12, True), (12, 11, True), (11, 10, True), (13, 13, True), (12, 12, True), (11, 11, True),
    (14, 13, False), (13, 11, False), (12, 10, False), (14, 12, False), (11, 9, False), (12, 13, False),
    (11, 12, False), (10, 11, False)])
def test_the_control_only_the_step_named_passes(current: int, held: int, passes: bool):
    """A step to 14 fails, whatever the branch records beside it: a form 14 paragraph and an empty
    ``tests/saved/test_form14.py`` are not read. Two steps at once fail, a branch behind the tip fails too, and so does
    the step before."""
    assert allowed(current, held) is passes


def steps_of(versions: dict[str, object]) -> dict[int, list[str]]:
    """The saved-form versions the module's ``*_ADDED`` and ``*_RENAMED`` constants name, each with its constants."""
    named: dict[int, list[str]] = {}
    for name, value in versions.items():
        if name.isupper() and name.endswith(("_ADDED", "_RENAMED")) and type(value) is int:
            named.setdefault(value, []).append(name)
    return named


def test_each_saved_form_step_is_one_change_and_the_last_is_current():
    """Two branches that each take the same form's step (both name form 12, one merged after the other without its
    renumbering) pass the step guard above, which reads ``CURRENT`` alone; here each version is named by one constant,
    and the highest by one is ``CURRENT``."""
    import mime.saved._versions as versions

    named = steps_of(vars(versions))
    assert {version: names for version, names in named.items() if len(names) > 1} == {}
    assert max(named) == CURRENT


def test_the_control_two_changes_naming_one_form_are_seen():
    assert steps_of({"ROWS_RENAMED": 12, "OTHER_ADDED": 12, "TERMS_ADDED": 4, "FIELDS_KEY": "fields"}) == {
        12: ["ROWS_RENAMED", "OTHER_ADDED"], 4: ["TERMS_ADDED"]}
