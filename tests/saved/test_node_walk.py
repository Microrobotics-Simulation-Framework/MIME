"""The node walk: every registered entry, MIME's toys and the lab fixture's, has a node case in ``node_cases.py``,
and every node it builds, in the first round-trip case that uses it, passes MADDENING's ``verify_node`` battery (and
``verify_node_order`` on each order a node declares). An entry registered without a node case fails here, as one
without a round-trip case fails the registry walk.

The controls: each node case removed in turn is named as missing; an entry registered without one is caught; a case
that says an entry builds no node fails when it builds one, and an entry that builds none without saying why fails.
"""

import importlib.util
import os

import pytest
from cases import CASES
from node_cases import NODE_CASES

from mime.testing import NodeCase, covered, installed_entries, node_failures
from mime.testing.toys import coupled_toy

LAB_FIXTURE = "mime_lab_fixture"


def uncased(node_cases, registered) -> list[str]:
    return [" ".join(key) for key in map(tuple, registered) if key not in node_cases]


@pytest.fixture(scope="module")
def registered():
    found = installed_entries()
    assert ("interaction", "toy.spring", "0.1.0") in found, "the walk would pass over an empty registry"
    return found


def test_every_registered_entry_has_a_node_case(registered):
    missing = uncased(NODE_CASES, registered)
    assert not missing, "no node case for: " + "; ".join(missing) + ". Add one to tests/saved/node_cases.py."


def test_removing_any_node_case_fails_the_walk_naming_its_entry(registered):
    for key in NODE_CASES:
        assert uncased({k: c for k, c in NODE_CASES.items() if k != key}, registered) == [" ".join(key)]


def test_an_entry_registered_without_a_node_case_fails_the_walk(registered):
    assert uncased(NODE_CASES, [*registered, ("interaction", "toy.uncased", "0.1.0")]) == [
        "interaction toy.uncased 0.1.0"]


def first_case_using(key):
    for case in CASES:
        if case.name == "nudged_toy" and importlib.util.find_spec(LAB_FIXTURE) is None:
            if os.environ.get("CI"):
                pytest.fail("the lab fixture package is not installed; every CI lane installs it from ./lab_fixture")
            pytest.skip("the lab fixture package is not installed: uv pip install --no-deps ./lab_fixture")
        if key in covered(case.make().document())[0]:
            return case
    pytest.fail(f"no round-trip case uses {' '.join(key)}; the registry walk requires one")


@pytest.mark.parametrize("key", sorted(NODE_CASES), ids=" ".join)
def test_every_node_an_entry_builds_passes_maddenings_verification(key):
    case, node_case = first_case_using(key), NODE_CASES[key]
    if case.refused:
        assert node_case.no_node, f"{' '.join(key)} is built only by {case.name}, which is refused: say why it " \
                                  f"builds no node"
        return
    failures = node_failures(case.make().build(), key, node_case)
    assert not failures, "\n".join(failures)


def test_a_case_that_says_no_node_fails_when_its_entry_builds_one_and_an_unexplained_absence_fails():
    built = coupled_toy().build()
    (built_one,) = node_failures(built, ("interaction", "toy.spring", "0.1.0"), NodeCase(no_node="none, it says"))
    assert "its case says it builds no node (none, it says), and it built ['toy.spring:probe']" in built_one
    (unexplained,) = node_failures(built, ("interaction", "toy.couple", "0.1.0"), NodeCase())
    assert "built no node in this experiment" in unexplained
