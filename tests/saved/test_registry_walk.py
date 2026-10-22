"""The registry walk: every registered entry, MIME's toys and the lab fixture's, and every kind of thing has a
round-trip case, and every case round-trips: built, saved, loaded in a fresh process and rebuilt, it gives the same
derived hash and the same first steps, bit for bit. An entry registered without a case fails here.

The controls: removing a case that is the only one to use an entry is caught, naming that entry; so is an entry
that no case uses. Each case's parameter leaves also reach its compiled step without a recompilation.
"""

import importlib.util
import os

import pytest
from cases import CASES, Case

from mime.testing import (
    covered,
    installed_entries,
    param_reach_failures,
    refused_round_trip_failures,
    round_trip_failures,
)
from mime.things import ThingKind

LAB_FIXTURE = "mime_lab_fixture"


def _needs_lab_fixture(case: Case) -> None:
    if case.name == "nudged_toy" and importlib.util.find_spec(LAB_FIXTURE) is None:
        if os.environ.get("CI"):
            pytest.fail("the lab fixture package is not installed; every CI lane installs it from ./lab_fixture")
        pytest.skip("the lab fixture package is not installed: uv pip install --no-deps ./lab_fixture")


def uncovered(cases: tuple[Case, ...], registered: list[tuple[str, str, str]]) -> list[str]:
    entries: set[tuple[str, str, str]] = set()
    kinds: set[str] = set()
    for case in cases:
        if case.name == "nudged_toy" and importlib.util.find_spec(LAB_FIXTURE) is None:
            continue
        found_entries, found_kinds = covered(case.make().document())
        entries |= found_entries
        kinds |= found_kinds
    missing = [f"{kind} {name} {version}" for kind, name, version in registered if (kind, name, version) not in entries]
    missing += [f"a thing of kind {kind.value}" for kind in ThingKind if kind.value not in kinds]
    return missing


@pytest.fixture(scope="module")
def registered():
    found = installed_entries()
    assert ("interaction", "toy.spring", "0.1.0") in found, "the walk would pass over an empty registry"
    return found


def test_every_registered_entry_and_every_kind_of_thing_has_a_round_trip_case(registered):
    missing = uncovered(CASES, registered)
    assert not missing, "no round-trip case uses: " + "; ".join(missing) + ". Add one to tests/saved/cases.py."


def test_removing_the_only_case_for_an_entry_fails_the_walk(registered):
    """The control that the walk can fail: each case that alone uses some entry or kind of thing is removed in turn,
    and the walk names what it leaves without a case."""
    load_bearing = 0
    for case in CASES:
        rest = tuple(c for c in CASES if c is not case)
        missing = uncovered(rest, registered)
        if missing:
            load_bearing += 1
            print(f"without {case.name}: no round-trip case uses {'; '.join(missing)}")
    assert load_bearing >= 3


def test_an_entry_registered_without_a_case_fails_the_walk(registered):
    assert uncovered(CASES, [*registered, ("interaction", "toy.uncased", "0.1.0")]) == [
        "interaction toy.uncased 0.1.0"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_every_case_round_trips_through_a_fresh_process(case, tmp_path):
    _needs_lab_fixture(case)
    if case.refused:
        assert refused_round_trip_failures(case.make(), tmp_path, case.refused) == []
    else:
        assert round_trip_failures(case.make(), tmp_path) == []


@pytest.mark.parametrize("case", [c for c in CASES if not c.refused], ids=lambda case: case.name)
def test_every_cases_parameter_leaves_reach_its_compiled_step(case):
    _needs_lab_fixture(case)
    assert param_reach_failures(case.make().build()) == []
