"""The structure key: which compiled program an assembled graph needs. Two graphs with the same key run one compiled
step, each with its own parameter leaves and initial state; any other difference is another program. Found from what
was assembled, never declared: MADDENING's description of the graph (every node, edge, coupling group and external
input, with the contents of every static array) with each parameter leaf's value replaced by its dtype and shape, but
for a leaf a static is derived from (``static_data_deps``), which keeps its value, since the step reads the static and
not the leaf; and, for a controller inside the compiled step, the program its step traces to, since its law is no
parameter and the description does not hold it."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np
from maddening import GraphManager

from mime.build._derived import array_record, canonical_json, derive
from mime.build._in_graph import program_of
from mime.build._maddening import parameter_tree, static_dependencies
from mime.nodes.control import ControllerNode

#: JAX's tracer, whose hints name types from jaxlib, which ships no stubs.
_jax: Any = cast(Any, jax)


@dataclass(frozen=True)
class StructureKey:
    """A graph's structure key (``key``), the derived hash of its physics with its values (``physics_hash``, the build
    report's), and its parameter tree on the host, by node and leaf (``tree``), read once for both."""

    key: str
    physics_hash: str
    tree: Mapping[str, Mapping[str, np.ndarray]]
    #: What the key is the hash of: the description with the leaves' values removed, the controllers' programs and the
    #: precision; what :func:`split_reasons` compares.
    record: Mapping[str, Any]


def _layout(value: np.ndarray) -> dict[str, Any]:
    return {"leaf": str(value.dtype), "shape": list(value.shape)}


def _controller_programs(graph: GraphManager) -> dict[str, Any]:
    """Each controller node's step, traced from its own start and zero reads: the program text with its callables'
    addresses removed, and its constants' contents."""
    found: dict[str, Any] = {}
    for name in graph.node_names:
        node: object = graph.get_node(name)
        if not isinstance(node, ControllerNode):
            continue
        constants = node.constants()
        start = cast(Mapping[str, Any], constants["initial_state"])
        reads = cast(Mapping[str, tuple[str, tuple[int, ...]]], constants["reads"])
        zeros = {label: np.zeros(shape, dtype=_jax.numpy.zeros(()).dtype) for label, (_, shape) in reads.items()}
        text, consts = program_of(_jax.make_jaxpr(node.law_step)(dict(start), zeros))
        found[name] = {"program": hashlib.sha256(text.encode()).hexdigest(),
                       "constants": [array_record(held) for held in consts]}
    return found


def structure_key(graph: GraphManager, observation_nodes: Iterable[str]) -> StructureKey:
    """``graph``'s structure key, assembled or built (:class:`StructureKey`)."""
    derived = derive(graph, observation_nodes)
    tree = {name: {key: np.asarray(value) for key, value in leaves.items()}
            for name, leaves in parameter_tree(graph).items()}
    kept = {name: {param for params in static_dependencies(graph.get_node(name)).values() for param in params}
            for name in graph.node_names}

    def stripped(section: Mapping[str, Any]) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        for node in cast(list[dict[str, Any]], section["nodes"]):
            name = str(node["name"])
            leaves = tree.get(name, {})
            params = {key: (_layout(leaves[key]) if key in leaves and key not in kept.get(name, set()) else value)
                      for key, value in dict(node.get("params") or {}).items()}
            nodes.append({**node, "params": params})
        return {**section, "nodes": nodes}

    record = {"physics": stripped(derived.physics), "observations": stripped(derived.observations),
              "controllers": _controller_programs(graph), "x64": bool(_jax.config.jax_enable_x64)}
    key = "structure:" + hashlib.sha256(canonical_json(record).encode()).hexdigest()
    return StructureKey(key, derived.physics_hash, tree, record)


def split_reasons(first: Mapping[str, Any], other: Mapping[str, Any]) -> list[str]:
    """Why two structure keys differ, from their records, in a few words each: the precision, the controller's step,
    the node set, a node's rate, a node's static or structural setting, a leaf a static is derived from, the edges, the
    coupling groups or the external inputs."""
    found: list[str] = []
    if first.get("x64") != other.get("x64"):
        found.append("the precision")
    if first.get("controllers") != other.get("controllers"):
        found.append("the controller's step, which its law closes over")
    for part in ("physics", "observations"):
        a = cast(Mapping[str, Any], first.get(part, {}))
        b = cast(Mapping[str, Any], other.get(part, {}))
        nodes_a = {str(n["name"]): n for n in cast(list[Mapping[str, Any]], a.get("nodes", []))}
        nodes_b = {str(n["name"]): n for n in cast(list[Mapping[str, Any]], b.get("nodes", []))}
        if set(nodes_a) != set(nodes_b):
            changed = sorted(set(nodes_a) ^ set(nodes_b))
            found.append(f"the node set ({', '.join(changed[:4])}{', ...' if len(changed) > 4 else ''})")
        for name in sorted(set(nodes_a) & set(nodes_b)):
            na, nb = nodes_a[name], nodes_b[name]
            if na.get("type") != nb.get("type"):
                found.append(f"node {name}'s kind")
            if na.get("timestep") != nb.get("timestep"):
                found.append(f"node {name}'s rate (dt)")
            pa = cast(Mapping[str, Any], na.get("params") or {})
            pb = cast(Mapping[str, Any], nb.get("params") or {})
            for key in sorted(set(pa) | set(pb)):
                if pa.get(key) != pb.get(key):
                    held = pa.get(key)
                    layout = isinstance(held, Mapping) and "leaf" in cast(Mapping[str, Any], held)
                    kind = "its leaf's layout" if layout else \
                        "a value its static is derived from" if isinstance(held, (float, list)) else "a setting"
                    found.append(f"node {name}'s {key} ({kind})")
            sa = cast(Mapping[str, Any], cast(Mapping[str, Any], a.get("static") or {}).get(name) or {})
            sb = cast(Mapping[str, Any], cast(Mapping[str, Any], b.get("static") or {}).get(name) or {})
            if sa != sb:
                keys = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
                found.append(f"node {name}'s static data ({', '.join(keys)})")
        for section in ("edges", "external_inputs", "coupling_groups", "param_specs"):
            if a.get(section) != b.get(section):
                found.append(f"the {section.replace('_', ' ')}")
    return found or ["the graph's description"]



__all__ = ["StructureKey", "split_reasons", "structure_key"]
