"""The post-build checks: properties of the constructed graph that MADDENING does not enforce, asked of the graph
itself rather than of what the builders recorded, so a defect planted after construction is caught too."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, cast

from maddening import GraphManager
from maddening.core.edge import EdgeSpec
from maddening.core.node import BoundaryInputSpec

from mime.build._maddening import edge_transform, graph_dict, inputs, node_dict
from mime.build._plan import PlanSet
from mime.build._refusals import MIME_DEFECT, Refusal, RefusalClass


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
    for spec in cast(list[dict[str, Any]], graph_dict(graph, strict_mappings=False)["external_inputs"]):
        target = plans.targets.get((spec["target_node"], spec["target_field"]))
        if target is not None:
            refusals.append(_refuse("post-build.external-input",
                                    f"{target.interface} of {target.thing!r} ({spec['target_node']}."
                                    f"{spec['target_field']}) is also an external input: an external input replaces "
                                    f"everything delivered into its field, so a target is never one"))
    refusals += _units_refusals(graph, built_by)
    for name in graph.node_names:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            node_dict(graph.get_node(name))
        for warning in caught:
            refusals.append(_refuse("post-build.unsaved-arguments",
                                    f"node {built_by(name)} would not be saved whole: {warning.message}"))
    return refusals
