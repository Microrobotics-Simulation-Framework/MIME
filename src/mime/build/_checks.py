"""The post-build checks: properties of the constructed graph that MADDENING does not enforce, asked of the graph
itself rather than of what the builders recorded, so a defect planted after construction is caught too."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable
from typing import Any, cast

from maddening import GraphManager
from maddening.core.edge import EdgeSpec
from maddening.core.node import BoundaryInputSpec

from mime.build._control import Mode, ReadClass
from mime.build._maddening import edge_transform, external_inputs, inputs, node_dict, node_fields_of
from mime.build._plan import PlanSet
from mime.things import did_you_mean
from mime.build._refusals import MIME_DEFECT, Refusal, RefusalClass
from mime.build._report import graph_config


def _refuse(code: str, message: str) -> Refusal:
    return Refusal(code, RefusalClass.INTERNAL, message, MIME_DEFECT)


def _inputs(graph: GraphManager, node: str) -> dict[str, BoundaryInputSpec]:
    return inputs(graph.get_node(node))


def _units_refusals(graph: GraphManager, built_by: Callable[[str], str]) -> list[Refusal]:
    """Every node input declares its units, and every edge is labelled at both ends with its target input's units:
    MIME converts no units on an edge, so a label that differs anywhere is a value read in units it was not
    computed in."""
    refusals: list[Refusal] = []
    for name in graph.node_names:
        for field, spec in _inputs(graph, name).items():
            if not spec.expected_units:
                refusals.append(_refuse("post-build.input-units",
                                        f"input {field!r} of node {built_by(name)} declares no expected_units: every "
                                        f"node input declares its units, spelt as the vocabulary spells them"))
    for edge in graph.edges:
        what = f"the edge {built_by(edge.source_node)}.{edge.source_field} -> {edge.target_node}.{edge.target_field}"
        spec = _inputs(graph, edge.target_node).get(edge.target_field)
        if spec is None:
            refusals.append(_refuse("post-build.input-units",
                                    f"{what} delivers into an input its node does not declare, so nothing states "
                                    f"the units it takes"))
            continue
        if edge.source_units is None or edge.target_units is None:
            refusals.append(_refuse("post-build.units",
                                    f"{what} carries no units label at its "
                                    f"{'source' if edge.source_units is None else 'target'}: every edge is labelled "
                                    f"from the vocabulary"))
        elif not (edge.source_units == edge.target_units == spec.expected_units):
            refusals.append(_refuse("post-build.units",
                                    f"{what} is labelled {edge.source_units!r} -> {edge.target_units!r}, and its "
                                    f"target takes {spec.expected_units!r}: MIME converts no units on an edge"))
    return refusals


def _writer_refusals(graph: GraphManager, plans: PlanSet) -> list[Refusal]:
    """Every command has one writer, its drive's edge or the external input the controller's write derives, and the
    graph declares no external input that a write does not derive: MADDENING fills an input nobody feeds with
    zeros, so an underived one would run on zeros."""
    refusals: list[Refusal] = []
    edges = Counter((edge.target_node, edge.target_field) for edge in graph.edges)
    declared = set(external_inputs(graph))
    for plan in plans.entries:
        command = plan.command
        if command is not None and (edges[(command.node, command.field)] != 1 or
                                    (command.node, command.field) in declared):
            refusals.append(_refuse("post-build.command-writers",
                                    f"the command {command.command!r} ({command.node}.{command.field}) is driven by "
                                    f"{plan.owner}, and it takes {edges[(command.node, command.field)]} edges and "
                                    f"{'an' if (command.node, command.field) in declared else 'no'} external input: "
                                    f"a command has one writer"))
    controller = plans.controller
    writes = {(w.node, w.field): w for w in controller.writes} if controller is not None else {}
    in_graph = controller is not None and controller.mode is Mode.GRAPH
    for key, write in writes.items():
        if (key in declared, edges[key]) != ((False, 1) if in_graph else (True, 0)):
            fed = ("the edge from the controller's node, and nothing else" if in_graph else
                   "the external input its write derives, and nothing else")
            refusals.append(_refuse("post-build.command-writers",
                                    f"the controller's write {write.label!r} ({write.path}) feeds {write.node}."
                                    f"{write.field}, which takes {edges[key]} edges and "
                                    f"{'an' if key in declared else 'no'} external input: a command the controller "
                                    f"writes takes {fed}"))
    derived: set[tuple[str, str]] = set() if in_graph else set(writes)
    for node, field in sorted(declared - derived):
        refusals.append(_refuse("post-build.underived-input",
                                f"the graph declares an external input {node}.{field} that no write of the "
                                f"controller derives: nothing would feed it, and MADDENING would fill it with zeros"))
    return refusals


def _read_refusals(graph: GraphManager, plans: PlanSet) -> list[Refusal]:
    """Every node field the controller reads is a field of that node's state."""
    refusals: list[Refusal] = []
    for read in plans.controller.reads if plans.controller is not None else ():
        held = node_fields_of(graph, read.node)
        if read.field in held:
            continue
        if read.read_class is ReadClass.OBSERVATION:
            refusals.append(Refusal("controller.read-unknown", RefusalClass.USER,
                                    f"the controller's read {read.label!r} names {read.path!r}, and the observation's "
                                    f"node {read.node} holds no {read.field!r}",
                                    f"read one of the observation's outputs, reads={{label: \"<observation "
                                    f"id>.<output>\"}}. {did_you_mean(read.field, held)}"))
        else:
            refusals.append(_refuse("post-build.read-field",
                                    f"the controller's read {read.label!r} ({read.path}) is published by node "
                                    f"{read.node} as {read.field!r}, which its state does not hold"))
    return refusals


def post_build_checks(graph: GraphManager, plans: PlanSet) -> list[Refusal]:
    refusals: list[Refusal] = []

    def built_by(node: str) -> str:
        return node if plans.owner_of(node) is not None else f"{node} (a node no entry planned)"

    into_target: dict[tuple[str, str], list[EdgeSpec]] = {}
    for edge in graph.edges:
        target = plans.targets.get((edge.target_node, edge.target_field))
        if target is None:
            continue
        into_target.setdefault((edge.target_node, edge.target_field), []).append(edge)
        what = (f"the edge {built_by(edge.source_node)}.{edge.source_field} -> {edge.target_node}.{edge.target_field}, "
                f"into {target.interface} of {target.thing!r} (a {target.kind.value} target)")
        if not target.kind.delivered_by_edge:
            refusals.append(_refuse("post-build.not-by-edge", f"{what}: this kind of target is taken at construction, "
                                                              f"never by an edge"))
        elif target.kind.additive and not edge.additive:
            refusals.append(_refuse("post-build.additive",
                                    f"{what} is not additive: on MADDENING 0.4 a non-additive edge replaces what was "
                                    f"summed into its input before it, so every edge into a summed target must add"))
        if edge_transform(edge) is not None:
            refusals.append(_refuse("post-build.transform",
                                    f"{what} carries a transform: a target receives the delivered amount as the "
                                    f"delivering law computed it, so no edge into a target transforms it"))
    for (node, field), edges in into_target.items():
        target = plans.targets[(node, field)]
        if target.kind.exclusive and len(edges) > 1:
            sources = ", ".join(f"{built_by(e.source_node)}.{e.source_field}" for e in edges)
            refusals.append(_refuse("post-build.exclusive",
                                    f"{target.interface} of {target.thing!r} is an {target.kind.value} and takes one "
                                    f"edge; {len(edges)} deliver into {node}.{field}: {sources}"))
    for spec in cast(list[dict[str, Any]], graph_config(graph)["external_inputs"]):
        target = plans.targets.get((spec["target_node"], spec["target_field"]))
        if target is not None:
            refusals.append(_refuse("post-build.external-input",
                                    f"{target.interface} of {target.thing!r} ({spec['target_node']}."
                                    f"{spec['target_field']}) is also an external input: an external input replaces "
                                    f"everything delivered into its field, so a target is never one"))
    refusals += _writer_refusals(graph, plans) + _read_refusals(graph, plans)
    refusals += _units_refusals(graph, built_by)
    for name in graph.node_names:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            node_dict(graph.get_node(name))
        for warning in caught:
            refusals.append(_refuse("post-build.unsaved-arguments",
                                    f"node {built_by(name)} would not be saved whole: {warning.message}"))
    return refusals
