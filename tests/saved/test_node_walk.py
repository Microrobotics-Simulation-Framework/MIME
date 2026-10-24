"""The node walk: every registered entry, MIME's toys and the lab fixture's, has a node case in ``node_cases.py``,
and every node it builds, in the first round-trip case that uses it, passes MADDENING's ``verify_node`` battery (and
``verify_node_order`` on each order a node declares). An entry registered without a node case fails here, as one
without a round-trip case fails the registry walk.

The same case holds the entry to its ``builds``: for each rung the case declares it with, the node classes it builds
are those its ``builds`` names, and a node that carries a symbol table answers its rows' names and units. A rung that
case does not build is declared, unchecked: it is listed in ``docs/ratchets/builds_unchecked.txt``, whose length the
walk prints.

The controls: each node case removed in turn is named as missing; an entry registered without one is caught; a case
that says an entry builds no node fails when it builds one, and an entry that builds none without saying why fails;
an entry declaring a node it does not build, and one building a node it does not declare, each fail by name; a rung
no case builds fails until it is listed, as a listed rung a case now builds fails until its line is deleted; and the
walk's own check of an entry, run on a planted entry whose built node's symbol row names an input the node does not
declare, fails naming it, while the same node with the row named right passes.
"""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import pytest
from cases import CASES, Case
from maddening.core.compliance.metadata import NodeMeta
from node_cases import NODE_CASES

from mime.core import Symbol, SymbolKind
from mime.docs import RatchetList, instance_problems, list_problems, read_list, rows_of
from mime.docs._check import LISTS, list_path
from mime.registry import BuildContext, Dynamics, EntryKind, lookup, register_interaction
from mime.testing import NodeCase, builds_failures, covered, installed_entries, node_failures, nodes_by_entry
from mime.testing.toys import ToyBody, ToyDrag, ToySpring, coupled_toy
from mime.testing.toys._entries import _respond
from mime.testing.toys._nodes import ToyDragNode, ToySpringNode
from mime.things import Body, State
from mime.units import m, m_per_s
from mime.vocabulary import lookup as interface_named
from mime.vocabulary import register_namespace

LAB_FIXTURE = "mime_lab_fixture"
ROOT = Path(__file__).resolve().parents[2]


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


# -- builds ---------------------------------------------------------------------------------------------------------

def rungs_declared(case: Case, key) -> set[str]:
    """The rungs ``case``'s document declares the entry ``key`` with, ``""`` for an entry without rungs."""
    if key[0] != "interaction":
        return {""}
    return {str(doc.get("settings", {}).get("rung") or "") for doc in case.make().document()["interactions"]
            if (doc["name"], doc["version"]) == key[1:]}


def published_units(built, key, node_name: str) -> dict[str, str]:
    """The units of each state field of ``node_name`` that the dynamics ``key`` publishes, from the vocabulary."""
    record = lookup(EntryKind(key[0]), key[1], key[2])
    if record is None or not issubclass(record.cls, Dynamics):
        return {}
    owner = next(plan.owner for plan in built.plans.entries if node_name in plan.node_names())
    local = node_name.removeprefix(owner).removeprefix(".")
    found = {}
    for interface, port in record.cls.publishes.items():
        held = interface_named(interface)
        if port.node == local and held is not None:
            found[port.field] = held.units
    return found


@pytest.mark.parametrize("key", sorted(NODE_CASES), ids=" ".join)
def test_every_entry_builds_the_node_classes_its_builds_declares(key):
    case = first_case_using(key)
    if case.refused:
        assert " ".join(key) in unchecked_rungs([key], CASES), f"{' '.join(key)} is built only by {case.name}, " \
                                                                f"which is refused, so its builds is unchecked"
        return
    failures = walk_failures(case.make().build(), key, rungs_declared(case, key),
                             read_list(list_path(ROOT, "symbol_units"), LISTS["symbol_units"]))
    assert not failures, "\n".join(failures)


def walk_failures(built, key, rungs: set[str], units: RatchetList) -> list[str]:
    """The walk's check of the entry ``key`` in ``built``: the node classes it builds for each of ``rungs`` against its
    ``builds``, and each built node that carries a symbol table against its rows' names and units."""
    failures = builds_failures(built, key, rungs)
    record = lookup(EntryKind(key[0]), key[1], key[2])
    assert record is not None
    for name, node in nodes_by_entry(built).get(key, {}).items():
        if rows_of(type(node)):
            failures += instance_problems(node, settings=record.settings,
                                          published=published_units(built, key, name), units=units)
    return failures


def unchecked_rungs(keys, cases) -> set[str]:
    """Each declared rung of each entry that the first case using it does not build, as the list names it."""
    found: set[str] = set()
    for key in keys:
        record = lookup(EntryKind(key[0]), key[1], key[2])
        assert record is not None, key
        case = next(case for case in cases if key in covered(case.make().document())[0])
        built = set() if case.refused else rungs_declared(case, key)
        kind = key[0].replace(" ", "_")
        found |= {" ".join(part for part in (kind, *key[1:], rung) if part) for rung in set(record.cls.builds) - built}
    return found


def test_every_declared_rung_no_case_builds_is_listed():
    if importlib.util.find_spec(LAB_FIXTURE) is None:
        pytest.skip("the lab fixture package is not installed: uv pip install --no-deps ./lab_fixture")
    held = read_list(list_path(ROOT, "builds_unchecked"), LISTS["builds_unchecked"])
    hits = unchecked_rungs(sorted(NODE_CASES), CASES)
    print(f"\nbuilds declared, unchecked: {len(hits)}" + "".join(f"\n  {hit}" for hit in sorted(hits)))
    problems = list_problems(held, hits, "a declared rung no round-trip case builds (declared, unchecked)")
    assert not problems, "\n".join(problems)


register_namespace("nodewalk", "0.0.1")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class OverDeclaredSpring(ToySpring):
    """A spring whose builds names a node it never builds."""

    name = "nodewalk.over_declared_spring"
    builds = {"": (ToySpringNode, ToyDragNode)}


@register_interaction
@dataclass(frozen=True, kw_only=True)
class UnderDeclaredSpring(ToySpring):
    """A spring whose builds names nothing, though it builds its node."""

    name = "nodewalk.under_declared_spring"
    builds = {"": ()}


@register_interaction
@dataclass(frozen=True, kw_only=True)
class TwoRungDrag(ToyDrag):
    """The toy drag under a name of its own, so that its rungs are counted apart from the toy's."""

    name = "nodewalk.two_rung_drag"


def built_with(entry):
    experiment = coupled_toy()
    experiment.attach(entry, body="probe")
    return experiment.build()


def test_a_node_declared_and_not_built_fails_naming_it():
    key = ("interaction", "nodewalk.over_declared_spring", "0.1.0")
    assert builds_failures(built_with(OverDeclaredSpring()), key, {""}) == [
        "interaction nodewalk.over_declared_spring 0.1.0, its only rung: builds declares ToyDragNode, and the "
        "experiment builds no such node"]


def test_a_node_built_and_not_declared_fails_naming_it():
    key = ("interaction", "nodewalk.under_declared_spring", "0.1.0")
    assert builds_failures(built_with(UnderDeclaredSpring()), key, {""}) == [
        "interaction nodewalk.under_declared_spring 0.1.0, its only rung: the experiment builds ToySpringNode, which "
        "builds does not declare"]


def test_an_entry_built_as_it_declares_passes():
    key = ("interaction", "toy.spring", "0.1.0")
    assert builds_failures(coupled_toy().build(), key, {""}) == []


def test_a_rung_no_case_builds_is_unchecked_and_fails_until_listed(tmp_path):
    key = ("interaction", "nodewalk.two_rung_drag", "0.2.0")
    case = Case("two_rung", lambda: _with(TwoRungDrag.Quadratic()))
    hits = unchecked_rungs([key], [case])
    assert hits == {"interaction nodewalk.two_rung_drag 0.2.0 linear"}
    empty = tmp_path / "builds_unchecked.txt"
    empty.write_text("")
    (problem,) = list_problems(read_list(empty, LISTS["builds_unchecked"]), hits, "declared, unchecked")
    assert problem.startswith("interaction nodewalk.two_rung_drag 0.2.0 linear: declared, unchecked, and it is not on")
    listed = tmp_path / "listed.txt"
    listed.write_text("interaction nodewalk.two_rung_drag 0.2.0 linear  # no case builds it\n")
    assert list_problems(read_list(listed, LISTS["builds_unchecked"]), hits, "declared, unchecked") == []
    both = Case("two_rungs", lambda: _with(TwoRungDrag.Quadratic(), TwoRungDrag.Linear()))
    assert unchecked_rungs([key], [both]) == set()
    (stale,) = list_problems(read_list(listed, LISTS["builds_unchecked"]), set(), "declared, unchecked")
    assert stale.endswith("'interaction nodewalk.two_rung_drag 0.2.0 linear' now passes; delete this line")


class TabledSpringNode(ToySpringNode):
    """The toy spring's node with a symbol table of one row, its input."""

    meta = NodeMeta(algorithm_id="MIME-NODE-902",
                    governing_equations=r"\mathbf{F} = -k \, (\mathbf{x} - \mathbf{x}_{0})")
    symbols = (Symbol(r"\mathbf{x}", SymbolKind.INPUT, "x", "m", "mime.testing.toys._nodes.ToySpringNode.law"),)


class MisnamedSpringNode(TabledSpringNode):
    """``TabledSpringNode`` with its row naming an input it does not have."""

    symbols = (Symbol(r"\mathbf{x}", SymbolKind.INPUT, "position", "m", "mime.testing.toys._nodes.ToySpringNode.law"),)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class TabledSpring(ToySpring):
    """The toy spring building ``TabledSpringNode``."""

    name = "nodewalk.tabled_spring"
    builds = {"": (TabledSpringNode,)}
    node: ClassVar[type[ToySpringNode]] = TabledSpringNode

    @override
    def build(self, ctx: BuildContext, b) -> None:
        _respond(b, self.node(ctx.node_name(), ctx.timestep, k=ctx.scalar("k"), x0=ctx.vector("x0")), "x", "load")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class MisnamedSpring(TabledSpring):
    """The toy spring building ``MisnamedSpringNode``."""

    name = "nodewalk.misnamed_spring"
    builds = {"": (MisnamedSpringNode,)}
    node: ClassVar[type[ToySpringNode]] = MisnamedSpringNode


def test_the_walk_checks_the_symbol_rows_of_a_node_an_entry_builds(tmp_path):
    units = tmp_path / "symbol_units.txt"
    units.write_text("")
    held = read_list(units, LISTS["symbol_units"])
    key = ("interaction", "nodewalk.misnamed_spring", "0.1.0")
    assert walk_failures(built_with(MisnamedSpring()), key, {""}, held) == [
        f"MisnamedSpringNode: the row {MisnamedSpringNode.symbols[0].latex!r}: input 'position' is not declared; the "
        f"node's are ['x']"]
    assert walk_failures(built_with(TabledSpring()), ("interaction", "nodewalk.tabled_spring", "0.1.0"), {""},
                         held) == []


def _with(*entries):
    """The coupled toy with each of ``entries`` on a body of its own: the probe, then a second toy body."""
    experiment = coupled_toy()
    for index, entry in enumerate(entries):
        body = "probe" if index == 0 else f"second_{index}"
        if index:
            experiment.add_body(Body(body, dynamics=ToyBody(),
                                     initial=State(position=m((0.0, 0.0, 0.0)), velocity=m_per_s((0.0, 0.0, 0.0)))))
        experiment.attach(entry, body=body)
    return experiment
