"""MADDENING's node verification for every node an entry builds: the ``verify_node`` battery, and
``verify_node_order`` on each axis a node declares an order of accuracy on."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from maddening.core.node import SimulationNode
from maddening.testing import verification
from maddening.testing.mms import RefinementAxis, declared_order, verify_node_order
from maddening.testing.verification import VerificationResult

from mime.experiment import BuiltExperiment
from mime.registry import Entry, registration_of

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
class NodeCase:
    """How an entry's nodes are verified. ``options`` go to ``verify_node`` for each node (sampling bounds,
    ``constrain_state``, strategies: what makes a drawn example a valid state of that node); ``ladders`` give, by a
    node's local name (``""`` for an entry's only node), the refinement ladder of each axis it declares an order on.
    An entry that builds no node says why in ``no_node``, and is then held to building none."""

    options: Mapping[str, Any] = field(default_factory=dict[str, Any])
    ladders: Mapping[str, Sequence[OrderLadder]] = field(default_factory=dict[str, Sequence[OrderLadder]])
    no_node: str = ""


def computing_dtype(node: SimulationNode) -> np.dtype[Any]:
    """The floating dtype ``node``'s initial state is in: what it computes in, in this process's precision."""
    floating = [np.dtype(value.dtype) for value in cast(dict[str, Any], cast(Any, node).initial_state()).values()
                if np.issubdtype(np.dtype(value.dtype), np.floating)]
    return max(floating, key=lambda dtype: dtype.itemsize) if floating else np.dtype(np.float32)


def node_results(node: SimulationNode, options: Mapping[str, Any] | None = None,
                 ladders: Sequence[OrderLadder] = ()) -> dict[str, VerificationResult]:
    """MADDENING's battery on ``node``, and its order on each axis it declares one on or a ladder is given for. A
    declared order with no ladder to measure it fails: a claim nothing measures is not verified. The battery draws
    in the dtype the node computes in (:func:`computing_dtype`) unless ``options`` name one: a float32 draw of a node
    that computes in float64 measures the promotion, not the node."""
    results = dict(run_verify_node(node, **{"dtype": computing_dtype(node), **dict(options or {})}))
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
        for result in node_results(node, case.options, case.ladders.get(local, ())).values():
            if not result.passed:
                failures.append(f"{label}, node {name} ({type(node).__name__}): {result}")
    return failures


def assert_nodes_verified(built: BuiltExperiment, entry: Entry | type[Entry] | EntryKey,
                          case: NodeCase | None = None) -> None:
    """:func:`node_failures` as an assertion, for a lab's own test of its entry's nodes."""
    failures = node_failures(built, entry, case)
    assert not failures, f"{len(failures)} check(s) failed:\n" + "\n".join(failures)
