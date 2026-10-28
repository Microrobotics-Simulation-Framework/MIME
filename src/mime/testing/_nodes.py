"""MADDENING's node verification for every node an entry builds: the ``verify_node`` battery, and
``verify_node_order`` on each axis a node declares an order of accuracy on; and the comparison of what an entry builds
with what its ``builds`` declares."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import numpy as np
from maddening.core.node import SimulationNode
from maddening.testing import verification
from maddening.testing.mms import RefinementAxis, declared_order, verify_node_order
from maddening.testing.verification import VerificationResult

from mime.core import switch
from mime.core._node import MimeNode, jit, jnp
from mime.experiment import BuiltExperiment
from mime.registry import Entry, EntryKind, lookup, registration_of

#: ``maddening.testing.verification.verify_node``, typed: its ``node`` parameter is unannotated.
run_verify_node: Callable[..., dict[str, VerificationResult]] = cast(Any, verification).verify_node

type EntryKey = tuple[str, str, str]


@dataclass(frozen=True)
class OrderLadder:
    """A refinement ladder on one axis, as ``verify_node_order`` takes it: ``error_at(level)`` is the error at each
    of ``levels``, coarsest first."""

    axis: RefinementAxis
    error_at: Callable[[Any], float]
    levels: Sequence[Any]
    h_of: Callable[[Any], float] | None = None


@dataclass(frozen=True)
class SwitchJumps:
    """Outputs of a node that are not smooth at a switch it evaluates, and so may take either side there in two
    compilations of the same step. ``fields`` maps each such state field to the capabilities of a switch's declared
    ``orders`` it is made of (a nearest point, of the distance and the normal); ``declared`` gives the switches the
    node's step evaluates. A field is exempt from the battery's jit consistency where a switch within its resolution
    declares an order below 1 (a jump or a kink) for one of its capabilities, and never where every order it declares
    is 1 or more. An exempt field is not compared; ``branch_holds`` takes the node, the boundary inputs, the compiled
    step's state and the nested compilation's, and answers what a branch still owes (the other outputs are compared as
    everywhere): ``None`` when both are valid branches, else why not."""

    fields: Mapping[str, Collection[str]]
    declared: Callable[[SimulationNode], Collection[switch.Declaration]]
    branch_holds: Callable[[SimulationNode, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str | None]

    def exempt(self, node: SimulationNode, in_band: Collection[str]) -> frozenset[str]:
        """The fields not compared where the switches in ``in_band`` (owners) are within their resolution."""
        found: set[str] = set()
        for declaration in self.declared(node):
            if declaration.owner not in in_band:
                continue
            for name, capabilities in self.fields.items():
                orders = [declaration.orders[capability] for capability in capabilities
                          if capability in declaration.orders]
                if orders and min(orders) < 1:
                    found.add(name)
        return frozenset(found)


@dataclass(frozen=True)
class NodeCase:
    """How an entry's nodes are verified. ``options`` go to ``verify_node`` for each node (sampling bounds,
    ``constrain_state``, strategies: what makes a drawn example a valid state of that node); ``ladders`` give, by a
    node's local name (``""`` for an entry's only node), the refinement ladder of each axis it declares an order on;
    ``jumps`` the outputs that may differ between two compilations on a switch. An entry that builds no node says why
    in ``no_node``, and is then held to building none."""

    options: Mapping[str, Any] = field(default_factory=dict[str, Any])
    ladders: Mapping[str, Sequence[OrderLadder]] = field(default_factory=dict[str, Sequence[OrderLadder]])
    no_node: str = ""
    jumps: SwitchJumps | None = None


def computing_dtype(node: SimulationNode) -> np.dtype[Any]:
    """The floating dtype ``node``'s initial state is in: what it computes in, in this process's precision."""
    floating = [np.dtype(value.dtype) for value in cast(dict[str, Any], cast(Any, node).initial_state()).values()
                if np.issubdtype(np.dtype(value.dtype), np.floating)]
    return max(floating, key=lambda dtype: dtype.itemsize) if floating else np.dtype(np.float32)


def jit_consistent_off_switches(node: SimulationNode, jumps: SwitchJumps,
                                rtol: float = 1e-5, atol: float = 1e-6) -> Callable[..., bool]:
    """The battery's jit consistency (the step against ``jax.jit`` of it, every field within ``rtol``, ``atol``), with
    the fields ``jumps`` exempts compared only where no switch declaring them not smooth is within its resolution:
    there each state must satisfy ``jumps.branch_holds`` instead. Which switches are reached is read from a trace of
    the step made afresh (a switch context is read while a step is traced), compiled once."""
    held = cast(MimeNode, node)
    unknown = set(jumps.fields) - set(cast(dict[str, Any], cast(Any, node).initial_state()))
    if unknown:
        raise ValueError(f"SwitchJumps names {sorted(unknown)}, which are not state fields of {node.name!r}")
    nested = jit(held.update)

    @jit
    def reached(state: dict[str, Any], boundary: dict[str, Any], dt: Any) -> dict[str, Any]:
        within: dict[str, Any] = {}
        with switch.collecting() as rows:
            held.fresh_update(state, boundary, dt)
        for row in rows:
            here = jnp.any(jnp.abs(row.indicator) <= row.resolution)
            within[row.owner] = jnp.logical_or(within[row.owner], here) if row.owner in within else here
        return within

    def predicate(state: dict[str, Any], out: dict[str, Any], boundary: Mapping[str, Any], dt: float) -> bool:
        other = nested(state, boundary, dt)
        in_band = {owner for owner, here in reached(state, dict(boundary), dt).items() if bool(here)}
        exempt = jumps.exempt(node, in_band)
        for name in out:
            if name in exempt:
                continue
            x, y = np.asarray(jax.device_get(out[name])), np.asarray(jax.device_get(other[name]))
            assert np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True), (
                f"jit/eager mismatch in '{name}': max abs diff {np.max(np.abs(x - y))}")
        if exempt:
            problem = jumps.branch_holds(node, boundary, out, other)
            assert problem is None, f"on a switch, the two compilations' branches are not both valid: {problem}"
        return True

    return predicate


def node_results(node: SimulationNode, options: Mapping[str, Any] | None = None,
                 ladders: Sequence[OrderLadder] = (), jumps: SwitchJumps | None = None) -> dict[str, VerificationResult]:
    """MADDENING's battery on ``node``, and its order on each axis it declares one on or a ladder is given for. A
    declared order with no ladder to measure it fails: a claim nothing measures is not verified. The battery draws
    in the dtype the node computes in (:func:`computing_dtype`) unless ``options`` name one: a float32 draw of a node
    that computes in float64 measures the promotion, not the node. ``jumps`` replaces the battery's jit consistency
    with :func:`jit_consistent_off_switches`, under the same name."""
    given: dict[str, Any] = {"dtype": computing_dtype(node), **dict(options or {})}
    if jumps is not None:
        checks = [name for name in given.pop("checks", verification.DEFAULT_CHECKS) if name != "jit_consistent"]
        invariants: dict[str, Any] = {**given.get("invariants", {}),
                                      "jit_consistent": jit_consistent_off_switches(node, jumps)}
        given = {**given, "checks": checks, "invariants": invariants}
    results = dict(run_verify_node(node, **given))
    declared = declared_order(node)
    for axis in RefinementAxis:
        name = f"{axis.order_attribute}_order"
        claimed = getattr(declared, axis.order_attribute, None) if declared is not None else None
        ladder = next((given for given in ladders if given.axis is axis), None)
        if ladder is not None:
            results[name] = verify_node_order(node, axis=axis, error_at=ladder.error_at, levels=ladder.levels,
                                              h_of=ladder.h_of)
        elif claimed is not None:
            results[name] = VerificationResult(name, "FAIL", detail=(
                f"{type(node).__name__} declares a {axis.order_attribute} order of {claimed}, and its case gives no "
                f"refinement ladder to measure it: add an OrderLadder for {axis} to its NodeCase"))
    return results


def entry_key(entry: Entry | type[Entry]) -> EntryKey:
    """``(kind, name, version)`` of a registered entry, as the registry walk names it."""
    record = registration_of(entry)
    if record is None:
        raise ValueError(f"{entry!r} is not a registered entry")
    return (record.kind.value, record.name, record.version)


def nodes_by_entry(built: BuiltExperiment) -> dict[EntryKey, dict[str, SimulationNode]]:
    """Every node of ``built`` by the entry that built it, ``(kind, name, version)``, then by node name."""
    found: dict[EntryKey, dict[str, SimulationNode]] = {}
    for plan in built.plans.entries:
        record = plan.registered
        nodes = found.setdefault((record.kind.value, record.name, record.version), {})
        for name in plan.node_names():
            nodes[name] = built.graph.get_node(name)
    return found


def nodes_by_rung(built: BuiltExperiment, key: EntryKey) -> dict[str, set[type[SimulationNode]]]:
    """The classes of the nodes ``built`` holds for the entry ``key``, by the rung each declaration of it was written
    with (``""`` for an entry without rungs)."""
    found: dict[str, set[type[SimulationNode]]] = {}
    for plan in built.plans.entries:
        record = plan.registered
        if (record.kind.value, record.name, record.version) == key:
            rung = cast(str | None, getattr(plan.entry, "rung", None)) or ""
            found.setdefault(rung, set()).update(type(built.graph.get_node(name)) for name in plan.node_names())
    return found


def builds_failures(built: BuiltExperiment, key: EntryKey, rungs: Collection[str]) -> list[str]:
    """Every way the nodes ``built`` holds for the entry ``key`` differ from its ``builds`` declaration, for each rung
    in ``rungs`` (those the experiment declares it with): a node class declared and not built, and one built and not
    declared. An entry the experiment uses that plans no node, as a geometry kind, builds none."""
    record = lookup(EntryKind(key[0]), key[1], key[2])
    if record is None:
        return [f"{' '.join(key)} is not registered"]
    declared, built_nodes, label = dict(record.cls.builds), nodes_by_rung(built, key), " ".join(key)
    failures: list[str] = []
    for rung in sorted(rungs):
        shown = f"rung {rung!r}" if rung else "its only rung"
        if rung not in declared:
            failures.append(f"{label}: the experiment builds {shown}, and builds declares only {sorted(declared)}")
            continue
        names, nodes = set(declared[rung]), built_nodes.get(rung, set())
        failures += [f"{label}, {shown}: builds declares {node.__qualname__}, and the experiment builds no such node"
                     for node in sorted(names - nodes, key=lambda cls: cls.__qualname__)]
        failures += [f"{label}, {shown}: the experiment builds {node.__qualname__}, which builds does not declare"
                     for node in sorted(nodes - names, key=lambda cls: cls.__qualname__)]
    return failures


def node_failures(built: BuiltExperiment, entry: Entry | type[Entry] | EntryKey,
                  case: NodeCase | None = None) -> list[str]:
    """Every way the nodes ``built`` holds for ``entry`` fail MADDENING's verification, one line per check, each
    with its counterexample: empty when every node passes. An entry that builds no node here fails unless its case
    says why; one whose case says it builds none fails when it does."""
    case = case or NodeCase()
    key = entry if isinstance(entry, tuple) else entry_key(entry)
    nodes = nodes_by_entry(built).get(key, {})
    label = " ".join(key)
    if case.no_node:
        return [f"{label}: its case says it builds no node ({case.no_node}), and it built {sorted(nodes)}"] if nodes \
            else []
    if not nodes:
        return [f"{label} built no node in this experiment: give it an experiment that builds it, or say why it builds "
                f"none with NodeCase(no_node=...)"]
    owners = {name: plan for plan in built.plans.entries for name in plan.node_names()}
    failures: list[str] = []
    for name, node in nodes.items():
        plan = owners[name]
        local = name.removeprefix(plan.owner).removeprefix(".")
        for result in node_results(node, case.options, case.ladders.get(local, ()), case.jumps).values():
            if not result.passed:
                failures.append(f"{label}, node {name} ({type(node).__name__}): {result}")
    return failures


def assert_nodes_verified(built: BuiltExperiment, entry: Entry | type[Entry] | EntryKey,
                          case: NodeCase | None = None) -> None:
    """:func:`node_failures` as an assertion, for a lab's own test of its entry's nodes."""
    failures = node_failures(built, entry, case)
    assert not failures, f"{len(failures)} check(s) failed:\n" + "\n".join(failures)
