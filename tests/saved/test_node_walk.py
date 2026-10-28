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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, override

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cases import CASES, Case
from maddening.core.compliance.metadata import NodeMeta
from node_cases import NODE_CASES, shape_evaluations

from mime.core import Symbol, SymbolKind, shape_named
from mime.core.switch import Declaration
from mime.docs import RatchetList, instance_problems, list_problems, read_list, rows_of
from mime.docs._check import LISTS, list_path
from mime.registry import BuildContext, Dynamics, EntryKind, lookup, register_interaction
from mime.testing import NodeCase, builds_failures, covered, installed_entries, node_failures, node_results, nodes_by_entry
from mime.testing._nodes import jit_consistent_off_switches
from mime.testing.toys import ToyBody, ToyDrag, ToySpring, coupled_toy
from mime.testing.toys._entries import _respond
from mime.testing.toys._nodes import ToyDragNode, ToyRegionProbeNode, ToySpringNode
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
    """The rungs ``case``'s document declares the entry ``key`` with, ``""`` for an entry without rungs: an
    interaction's, or a chain segment's."""
    document = case.make().document()
    if key[0] == "dynamics":
        segments = [segment for thing in document["things"].values() for segment in thing.get("chain", [])
                    if (segment["name"], segment["version"]) == key[1:]]
        return {str(segment.get("settings", {}).get("rung") or "") for segment in segments} or {""}
    if key[0] != "interaction":
        return {""}
    return {str(doc.get("settings", {}).get("rung") or "") for doc in document["interactions"]
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


# -- jit consistency on a switch --------------------------------------------------------------------------------------

SCREW = {"shape": "dejongh_screw", "centre": (0.0, 0.0, 0.0), "orientation": (1.0, 0.0, 0.0, 0.0), "radius": 1.56e-3,
         "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33, "starts": 2}
SPHERE = {"shape": "sphere", "centre": (0.0, 0.0, 0.0), "radius": 1.56e-3}
PROBE_CASE = NODE_CASES[("observation", "toy.region_probe", "0.1.0")]
assert PROBE_CASE.jumps is not None
JUMPS = PROBE_CASE.jumps


def probe_that_differs_when_compiled_inside(moved, region=SPHERE):
    """The probe of ``region`` whose state, traced inside another compilation, is ``moved(state, position)``: an error
    the walk's jit consistency must find off the switches, and, on one, only where the state it moves to is not a valid
    branch."""

    class Planted(ToyRegionProbeNode):
        def update(self, state, boundary_inputs, dt, *, params=None):
            out = super().update(state, boundary_inputs, dt, params=params)
            if isinstance(boundary_inputs["position"], jax.core.Tracer):
                out = moved(out, boundary_inputs["position"])
            return out

    return Planted("planted", 0.01, **region)


ROTATED_HALF_A_TURN = jnp.array([1.0, -1.0, -1.0])  # the screw is the same turned half a turn about its axis (two starts)
MIRRORED = lambda out: {**out, "nearest": out["nearest"] * ROTATED_HALF_A_TURN, "normal": out["normal"] * ROTATED_HALF_A_TURN}


def at_the_sphere_centre(out, position):
    """The sphere probe's state as a second compilation may take it within the band about the sphere's centre (its
    medial set), where the probe reports the point itself and a zero normal: the nearest point any boundary point,
    here the one towards the probed point, for a positive x; for a negative one, 0.1 % off the boundary."""
    toward = position / jnp.linalg.norm(position)
    radius = jnp.where(position[0] > 0, 1.0, 1.001) * SPHERE["radius"]
    return {**out, "nearest": radius * toward, "normal": toward}


#: The sphere probe's state differing between the compilations: at the sphere's centre as above, elsewhere mirrored.
MOVED = lambda out, position: jax.tree.map(lambda a, b: jnp.where(jnp.linalg.norm(position) < 1e-15, a, b),
                                           at_the_sphere_centre(out, position), MIRRORED(out))
#: Within the band about the sphere's centre (0.07 of its resolution) at a positive and a negative x; 70 resolutions
#: from it; far from it.
UNIT = np.ones(3) / np.sqrt(3.0)
IN_THE_BAND, ALSO_IN_THE_BAND, JUST_OUTSIDE_THE_BAND, FAR = (1e-16 * UNIT, -1e-16 * UNIT, 1e-13 * UNIT,
                                                            (1e-3, 2e-3, -1e-3))
#: The screw's axis, and a nearest point on it.
ON_THE_AXIS = (3.4928192e-3, 0.0, 1.77e-122)


def jit_consistency(check, node, position):
    """The walk's jit consistency ``check`` of ``node`` at ``position``: raises the assertion it would fail with."""
    state, given = node.initial_state(), {"position": jnp.array(position)}
    check(state, node.update(state, given, 0.01), given, 0.01)


def test_a_field_is_exempt_only_where_a_switch_in_band_declares_it_not_smooth():
    node = ToyRegionProbeNode("probe", 0.01, **SCREW)
    seam, focus = "dejongh_screw.seam", "dejongh_screw.focus"
    assert JUMPS.exempt(node, {seam}) == {"nearest", "normal"}  # the normal jumps, the nearest point with it
    assert JUMPS.exempt(node, {focus}) == {"nearest", "normal"}  # declared 0: continuous, with no derivative
    assert JUMPS.exempt(node, set()) == frozenset()
    assert JUMPS.exempt(node, {"capsule.end.low"}) == frozenset()  # the capsule's switch is not the screw's

    def declared(orders):
        return replace(JUMPS, declared=lambda _: [Declaration(owner="toy.switch", indicator="a made-up indicator, m",
                                                              orders=orders, route="none needed: made up")])
    assert declared({"distance": 1, "normal": 1}).exempt(node, {"toy.switch"}) == frozenset()
    assert declared({"distance": 1, "normal": 0}).exempt(node, {"toy.switch"}) == {"nearest", "normal"}
    assert declared({"distance": 0, "normal": 1}).exempt(node, {"toy.switch"}) == {"nearest"}


def test_a_switch_jump_naming_a_field_the_node_does_not_have_is_refused():
    node = ToyRegionProbeNode("probe", 0.01, **SPHERE)
    with pytest.raises(ValueError, match="nearset"):
        jit_consistent_off_switches(node, replace(JUMPS, fields={"nearset": ("normal",)}))


def screws_true_state():
    node = ToyRegionProbeNode("probe", 0.01, **SCREW)
    given = {"position": jnp.array(ON_THE_AXIS)}
    return node, given, {key: np.asarray(value) for key, value in node.update(node.initial_state(), given, 0.01).items()}


def boundary_point_at(node, state, angle):
    """The screw's boundary at ``angle`` about its axis, at the x of the state's nearest point: a boundary point that
    is not the nearest one."""
    signed_distance, _ = shape_evaluations("dejongh_screw", {key: node.constants()[f"region_{key}"]
                                                              for key in shape_named("dejongh_screw").parameter_names})
    low, high = 0.0, 4e-3
    for _ in range(80):
        middle = (low + high) / 2
        inside = float(signed_distance(jnp.array([state["nearest"][0], middle * np.cos(angle), middle * np.sin(angle)]))) < 0
        low, high = (middle, high) if inside else (low, middle)
    return np.array([state["nearest"][0], low * np.cos(angle), low * np.sin(angle)])


def test_on_the_screws_axis_a_state_is_a_valid_branch_only_if_its_nearest_point_and_normal_are_the_shapes():
    node, given, state = screws_true_state()
    holds = lambda **changed: JUMPS.branch_holds(node, given, {**state, **{k: np.asarray(v) for k, v in changed.items()}},
                                                 state)
    nearest, normal = state["nearest"], state["normal"]
    assert holds() is None
    assert holds(nearest=nearest * [1.0, -1.0, -1.0], normal=normal * [1.0, -1.0, -1.0]) is None  # the other side
    assert "off the boundary" in holds(nearest=nearest + [0.0, 1e-4, 0.0])
    for angle in np.linspace(0.0, 2.0 * np.pi, 13)[:-1]:  # a boundary point, but not at |distance| from the point
        assert "not |distance|" in holds(nearest=boundary_point_at(node, state, angle))
    assert "outward normal" in holds(normal=-normal)
    assert "outward normal" in holds(normal=np.array([normal[0], -normal[2], normal[1]]))
    assert "outward normal" in holds(normal=np.array([1.0, 0.0, 0.0]))
    assert "outward normal" in holds(normal=normal * [1.0, -1.0, -1.0])  # the other side's normal, the first side's nearest
    assert "outward normal" in holds(normal=normal * (1.0 + 1e-6))


def test_on_a_switch_either_nearest_point_is_a_valid_branch_and_one_off_the_surface_is_not():
    node = probe_that_differs_when_compiled_inside(MOVED)
    check = jit_consistent_off_switches(node, JUMPS)
    jit_consistency(check, node, IN_THE_BAND)
    with pytest.raises(AssertionError, match="not both valid.*off the boundary"):
        jit_consistency(check, node, ALSO_IN_THE_BAND)


def test_outside_the_band_a_nearest_point_that_differs_between_jit_and_eager_fails_even_if_it_is_a_valid_branch():
    node = probe_that_differs_when_compiled_inside(MOVED)
    check = jit_consistent_off_switches(node, JUMPS)
    for position in (JUST_OUTSIDE_THE_BAND, FAR):
        with pytest.raises(AssertionError, match="jit/eager mismatch in 'nearest'"):
            jit_consistency(check, node, position)


def test_the_batterys_jit_consistency_of_the_planted_probe_fails_in_every_run():
    planted = probe_that_differs_when_compiled_inside(lambda out, position: MIRRORED(out))
    # A control, not the node's property: it fails on the first drawn position, the sphere's only switch its centre,
    # and Hypothesis then spends its time shrinking the counterexample, so it draws few.
    only_jit = {**PROBE_CASE.options, "checks": ["jit_consistent"], "max_examples": 10}
    results = node_results(planted, only_jit, jumps=JUMPS)
    assert results["jit_consistent"].status == "FAIL" and "'nearest'" in results["jit_consistent"].detail
