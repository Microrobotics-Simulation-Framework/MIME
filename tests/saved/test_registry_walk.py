"""The registry walk: every registered entry, MIME's toys and the lab fixture's, and every kind of thing has a
round-trip case, and every case round-trips: built, saved, loaded in a fresh process and rebuilt, it gives the same
derived hash and the same first steps, bit for bit. An entry registered without a case fails here. Every geometry kind
that samples its surface passes the kind-conformance harness (``mime.testing.kind_problems``) at each declaration a
case makes of it, and a kind no case declares fails.

The controls: removing a case that is the only one to use an entry is caught, naming that entry; so is an entry
that no case uses. Each case's parameter leaves also reach its compiled step without a recompilation.
"""

import importlib.util
import os

import pytest
from cases import CASES, Case

from mime.core import jnp
from mime.docs import load_namespaces
from mime.registry import EntryKind, Samples, implements
from mime.registry import registered as registered_entries
from mime.scene import DeJonghHelix
from mime.testing import (
    covered,
    installed_entries,
    kind_problems,
    param_reach_failures,
    refused_round_trip_failures,
    round_trip_failures,
    sampling_kinds,
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


# -- the kind-conformance harness ----------------------------------------------------------------------------------

def kinds_in_cases() -> dict[type, list[tuple[object, dict]]]:
    """Every geometry kind the round-trip cases declare, a structure's or a body's part's, with its settings in SI as
    the plan resolved them."""
    found: dict[type, list[tuple[object, dict]]] = {}
    for case in CASES:
        if case.refused or (case.name == "nudged_toy" and importlib.util.find_spec(LAB_FIXTURE) is None):
            continue
        plans = case.make().plan()
        held = [(g.geometry, dict(g.settings)) for g in plans.geometries.values()]
        held += [(line.solid, dict(line.settings)) for line in plans.materials.parts]
        for kind, settings in held:
            found.setdefault(type(kind), []).append((kind, settings))
    return found


def spacing_of(kind, settings) -> float:
    """A fortieth of the kind's reach from its axis: about 250 samples round it, where the helix's lateral area, a
    periodic integral the trapezoid rule converges on exponentially, is exact to rounding (at a twelfth it is 8e-12
    off)."""
    return kind.radial_reach(settings) / 40.0 if implements(kind, "radial_reach") else 1e-4


@pytest.fixture(scope="module")
def declared_kinds():
    return kinds_in_cases()


def test_every_kind_that_samples_its_surface_passes_the_kind_conformance_harness(declared_kinds):
    load_namespaces()
    sampling = sampling_kinds([record.cls for record in registered_entries(EntryKind.GEOMETRY)])
    assert sampling, "the walk would pass over no kind"
    unchecked = [cls.label() for cls in sampling if cls not in declared_kinds]
    assert unchecked == [], f"no round-trip case declares {unchecked}, so the harness checks nothing of them"
    for cls in sampling:
        for kind, settings in declared_kinds[cls]:
            assert kind_problems(kind, settings, spacing_of(kind, settings)) == []
    print(f"\nchecked: {', '.join(sorted(cls.label() for cls in sampling))}")


@pytest.mark.parametrize("planted", ["volume", "inertia", "area", "normals", "points"])
def test_control_a_kind_whose_closed_form_or_samples_are_planted_wrong_fails_the_harness(declared_kinds, planted):
    """Each check can fail: the helix's volume, inertia or area off by one part in 10^9, its samples' normals turned or
    its points moved out by that much, each fails, on every run (nothing here is drawn)."""
    (kind, settings), *_ = declared_kinds[DeJonghHelix]
    wrong = 1.0 + 1e-9

    class Planted(DeJonghHelix):
        def volume(self, s):
            return super().volume(s) * (wrong if planted == "volume" else 1.0)

        def inertia_per_mass(self, s):
            held = super().inertia_per_mass(s)
            return tuple(tuple(c * (wrong if planted == "inertia" and i == j == 0 else 1.0) for j, c in enumerate(row))
                         for i, row in enumerate(held))

        def area(self, s, patch):
            return super().area(s, patch) * (wrong if planted == "area" else 1.0)

        def samples(self, parameters, patch, counts):
            held = super().samples(parameters, patch, counts)
            if planted == "normals":
                turned = held.normals.at[:, 0].add(1e-9)
                return Samples(held.points, turned / jnp.linalg.norm(turned, axis=1, keepdims=True), held.weights)
            if planted == "points":
                return Samples(held.points * wrong, held.normals, held.weights)
            return held

    planted_kind = Planted(**{name: getattr(kind, name) for name in ("radius", "length", "wavenumber", "modulation",
                                                                      "starts")})
    problems = kind_problems(planted_kind, settings, spacing_of(kind, settings))
    print(f"\n{planted}: {problems}")
    assert problems != []
