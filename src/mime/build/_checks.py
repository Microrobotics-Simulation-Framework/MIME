"""The post-build checks: properties of the constructed graph that MADDENING does not enforce, asked of the graph
itself rather than of what the builders recorded, so a defect planted after construction is caught too."""

from __future__ import annotations

import warnings
from typing import Any

from mime.build._plan import PlanSet
from mime.build._refusals import MIME_DEFECT, Refusal, RefusalClass
from mime.build._report import graph_config


def _refuse(code: str, message: str) -> Refusal:
    return Refusal(code, RefusalClass.INTERNAL, message, MIME_DEFECT)


def post_build_checks(graph: Any, plans: PlanSet) -> list[Refusal]:
    refusals: list[Refusal] = []

    def built_by(node: str) -> str:
        return node if plans.owner_of(node) is not None else f"{node} (a node no entry planned)"

    into_target: dict[tuple[str, str], list[Any]] = {}
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
        if edge.transform is not None:
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
    for spec in graph_config(graph)["external_inputs"]:
        target = plans.targets.get((spec["target_node"], spec["target_field"]))
        if target is not None:
            refusals.append(_refuse("post-build.external-input",
                                    f"{target.interface} of {target.thing!r} ({spec['target_node']}."
                                    f"{spec['target_field']}) is also an external input: an external input replaces "
                                    f"everything delivered into its field, so a target is never one"))
    for name in graph.node_names:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph.get_node(name).to_dict()
        for warning in caught:
            refusals.append(_refuse("post-build.unsaved-arguments",
                                    f"node {built_by(name)} would not be saved whole: {warning.message}"))
    return refusals
