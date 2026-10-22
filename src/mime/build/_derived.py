"""The derived hash: what a build constructed, as canonical JSON of MADDENING's own description of the graph plus the
contents of every static array, which MADDENING's description gives by shape and dtype only. Observations are hashed
apart from the physics, so adding one leaves the physics' hash alone."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from maddening import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray

from mime.build._maddening import graph_dict, static_data

HASH_RECIPE = "mime.hash/1"

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]


def _float(value: float) -> Json:
    if math.isfinite(value):
        return value
    return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")


def array_record(array: Any) -> dict[str, Json]:
    """An array as its dtype, shape and the SHA-256 of its little-endian C-order bytes: what it holds, not only its
    layout."""
    held = np.asarray(array)
    little = np.ascontiguousarray(held, dtype=held.dtype.newbyteorder("<"))
    return {"dtype": str(held.dtype), "shape": list(held.shape),
            "sha256": hashlib.sha256(little.tobytes()).hexdigest()}


def plain(value: object) -> Json:
    """``value`` as JSON data: tuples as lists, arrays as :func:`array_record`, non-finite floats as MADDENING spells
    them. Anything else is a ``TypeError``: a hash over a ``repr`` would hash an address."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in cast(Mapping[object, object], value).items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in cast(Iterable[object], value)]
    if isinstance(value, np.generic):
        return plain(cast(object, value.item()))
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return array_record(value)
    raise TypeError(f"{type(value).__name__} has no canonical JSON form")


def canonical_json(value: object) -> str:
    """One text per value: sorted keys, no spaces, floats as Python's shortest round-trip form."""
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_of(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def static_arrays(node: SimulationNode) -> dict[str, np.ndarray]:
    """Every array in ``node``'s static data, by key."""
    found: dict[str, np.ndarray] = {}
    for key, value in static_data(node).items():
        held = value.value if isinstance(value, StaticArray) else value
        if hasattr(held, "shape") and hasattr(held, "dtype"):
            found[key] = np.asarray(held)
    return found


def _static_record(node: SimulationNode) -> dict[str, Json]:
    record: dict[str, Json] = {}
    for key, value in sorted(static_data(node).items()):
        if isinstance(value, StaticArray):
            record[key] = {**array_record(value.value), "replication": value.replication,
                           "shard_axis": value.shard_axis}
        else:
            record[key] = plain(value)
    return record


@dataclass(frozen=True)
class DerivedGraph:
    """The derived graph, split into the physics and the observations, each with its hash. ``static`` holds every
    static array's contents by ``<node>/<key>``, for the load check that compares them within a tolerance."""

    physics: dict[str, Json]
    observations: dict[str, Json]
    physics_hash: str
    observations_hash: str
    static: dict[str, np.ndarray]


def derive(graph: GraphManager, observation_nodes: Iterable[str]) -> DerivedGraph:
    observing = set(observation_nodes)
    config = cast(dict[str, Any], plain(graph_dict(graph, strict_mappings=False)))
    nodes = cast(list[dict[str, Any]], config["nodes"])
    edges = cast(list[dict[str, Any]], config["edges"])
    inputs = cast(list[dict[str, Any]], config["external_inputs"])
    specs = cast(dict[str, Json], config.get("param_specs", {}))
    statics = {name: _static_record(graph.get_node(name)) for name in graph.node_names}

    def part(observed: bool) -> dict[str, Json]:
        def keep(name: object) -> bool:
            return (name in observing) == observed

        section: dict[str, Json] = {
            "recipe": HASH_RECIPE,
            "nodes": [n for n in nodes if keep(n["name"])],
            "edges": [e for e in edges if keep(e["target_node"])],
            "external_inputs": [i for i in inputs if keep(i["target_node"])],
            "param_specs": {owner: spec for owner, spec in specs.items() if keep(owner)},
            "static": {name: record for name, record in statics.items() if keep(name)},
        }
        if not observed:
            section["coupling_groups"] = cast(list[Json], config.get("coupling_groups", []))
        return section

    physics, observations = part(False), part(True)
    arrays = {f"{name}/{key}": array for name in graph.node_names
              for key, array in static_arrays(graph.get_node(name)).items()}
    return DerivedGraph(physics, observations, sha256_of(physics), sha256_of(observations), arrays)
