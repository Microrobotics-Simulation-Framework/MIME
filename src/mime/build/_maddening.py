"""MADDENING 0.4.x and JAX at MIME's boundary. Some of their type hints leave a container or a callable bare
(``dict``, ``Callable``), which a strict checker reads as unknown; each call MIME makes through one of those is made
here, once, and given the type MIME relies on. Nothing here changes what is called."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import jax
import numpy as np
from jax import Array
from jax.extend.core import Jaxpr
from maddening import GraphManager
from maddening.core.edge import EdgeSpec
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec

type NodeFields = dict[str, dict[str, Array]]


def _held(value: object) -> Any:
    return value


def add_edge(graph: GraphManager, source: tuple[str, str], target: tuple[str, str], *, additive: bool,
             units: str) -> None:
    _held(graph).add_edge(source[0], target[0], source[1], target[1], additive=additive, source_units=units,
                          target_units=units)


def add_coupling_group(graph: GraphManager, nodes: Sequence[str], *, max_iterations: int, tolerance: float) -> None:
    """A coupling group with every setting that decides what it returns written out: ``"ift"`` is exact only at a
    converged fixed point, and ``atol`` above zero can accept a group that has not converged."""
    _held(graph).add_coupling_group(list(nodes), max_iterations=max_iterations, tolerance=tolerance, solver="ift",
                                    atol=0.0)


def graph_schedule(graph: GraphManager) -> list[str]:
    """The order a compiled graph steps its nodes in."""
    return cast(list[str], _held(graph).schedule)


def graph_edges(graph: GraphManager) -> list[tuple[str, str, str, str]]:
    """Every edge as ``(source node, source field, target node, target field)``, in the order added."""
    return [(e.source_node, e.source_field, e.target_node, e.target_field) for e in graph.edges]


def edge_transform(edge: EdgeSpec) -> object | None:
    return cast(object | None, _held(edge).transform)


def inputs(node: SimulationNode) -> dict[str, BoundaryInputSpec]:
    return dict(node.boundary_input_spec() or {})


def zero_inputs(node: SimulationNode) -> dict[str, np.ndarray]:
    """A zero for every input ``node`` declares with a concrete shape, at that shape and dtype; an input whose shape
    leaves a dimension symbolic (negative) takes its shape from what the graph delivers."""
    zeros: dict[str, np.ndarray] = {}
    for name, spec in inputs(node).items():
        shape = cast(tuple[int, ...], _held(spec).shape)
        if all(size >= 0 for size in shape):
            zeros[name] = np.zeros(shape, dtype=np.dtype(spec.dtype or np.float32))
    return zeros


def resolved_inputs(graph: GraphManager, name: str) -> dict[str, Any]:
    """The inputs node ``name`` receives from the graph's current state: every incoming edge applied, mappings
    included, so each has the shape and dtype the compiled step gives it."""
    return cast(dict[str, Any], _held(graph).resolve_boundary_inputs(name))


def node_update(node: SimulationNode) -> Callable[..., object]:
    return cast(Callable[..., object], _held(node).update)


def node_fluxes(node: SimulationNode) -> Callable[..., object]:
    return cast(Callable[..., object], _held(node).compute_boundary_fluxes)


def node_dict(node: SimulationNode) -> dict[str, Any]:
    return cast(dict[str, Any], _held(node).to_dict())


def graph_dict(graph: GraphManager, *, strict_mappings: bool) -> dict[str, Any]:
    return cast(dict[str, Any], _held(graph).to_dict(strict_mappings=strict_mappings))


def static_data(node: SimulationNode) -> dict[str, Any]:
    return cast(dict[str, Any], _held(node).static_data)


def node_constructor_arguments(node: SimulationNode) -> dict[str, Any]:
    return cast(dict[str, Any], _held(node).params)


def node_state(graph: GraphManager, name: str) -> dict[str, Array]:
    return cast(dict[str, Array], _held(graph).get_node_state(name))


def parameter_leaves(graph: GraphManager) -> dict[str, dict[str, Any]]:
    """``params["nodes"]``: the leaves the compiled step reads, by node."""
    return cast(dict[str, dict[str, Any]], _held(graph).params.get("nodes", {}))


def parameter_specs(graph: GraphManager) -> dict[str, dict[str, ParamSpec]]:
    return cast(dict[str, dict[str, ParamSpec]], _held(graph).param_specs().get("nodes", {}))


def run_scan(graph: GraphManager, steps: int, params: Mapping[str, Any] | None = None) -> NodeFields:
    return cast(NodeFields, _held(graph).run_scan(steps, params=params))


def run_scan_with_history(graph: GraphManager, steps: int) -> tuple[NodeFields, NodeFields]:
    return cast(tuple[NodeFields, NodeFields], _held(graph).run_scan_with_history(steps))


def device_array(value: np.ndarray) -> Array:
    return cast(Array, _held(jax).device_put(value))


def make_jaxpr(fn: Callable[..., object], *args: object) -> Jaxpr:
    return cast(Jaxpr, _held(jax).make_jaxpr(fn)(*args))

