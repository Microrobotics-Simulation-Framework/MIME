"""How a confining wall reaches the boundary-element method: as the BEM's correction for a wall kind, a registered
function the BEM's node evaluates, keyed by the wall kind's name and version (``mime.registry.WallKind`` is the
declaration every solver shares: the geometry it matches, the parameters it reads, its validity and gradient status).

A correction takes the body's samples in the world frame (``N x 3``), its parameters by name (each a leaf of the
node's ``wall_<name>``) and the viscosity, and returns the ``3N x 3N`` block matrix added to the free-space matrix, the
velocity at each sample per unit force at each that the wall's image system adds, in m/(N s), the matrix's own units;
and whether every read it made of its data was accepted (a table read inside its readable range), which the node
publishes. ``prepare`` turns the parameters the wall kind reads from the geometry kind, in SI, into those node
constants at build: a frame across the tube's axis, a table loaded and compiled for the run's dtype. The node knows
Stokes kernels and quadrature, and nothing about tubes or planes.

Free space is the trivial wall, with no correction. The straight circular tube's (``_tube``: its image system read
from a table) is in place for the tube's wall kind, which is reserved until its tables are built again."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from jax import Array

from mime.core import jnp

#: The free-space wall kind's name: a medium with no wall, unbounded.
FREE_SPACE = "free_space"


class Corrected(NamedTuple):
    """A wall's correction at a body's samples: the ``3N x 3N`` ``matrix``, and whether every read it made was
    ``accepted``."""

    matrix: Array
    accepted: Array


def _nothing_to_prepare(parameters: Mapping[str, Any], dtype: Any) -> dict[str, object]:
    return {}


def _any_reach(parameters: Mapping[str, Any], reach: float) -> list[tuple[str, str]]:
    return []


@dataclass(frozen=True)
class WallCorrection:
    """The BEM's correction for the wall kind ``name`` ``version``: the ``parameters`` it reads (the node's
    ``wall_<name>`` constants, in SI), each one's ``units``, those ``held`` as built (not a study's leaf: a frame, a
    table), the function, and ``prepare``, which makes the constants from what the wall kind reads of the geometry,
    for a run in a dtype; and ``beyond`` says, when the experiment is planned, why a body that reaches ``reach`` (m)
    from its centre would be read outside the correction's data wherever it is (a table's axial reach), from what the
    wall kind reads, each as a message and an action."""

    name: str
    version: str
    parameters: tuple[str, ...]
    correction: Callable[[Array, Mapping[str, Any], Any], Corrected]
    prepare: Callable[[Mapping[str, Any], Any], dict[str, object]] = _nothing_to_prepare
    units: Mapping[str, str] = field(default_factory=dict[str, str])
    held: frozenset[str] = frozenset()
    beyond: Callable[[Mapping[str, Any], float], list[tuple[str, str]]] = _any_reach


_corrections: dict[tuple[str, str], WallCorrection] = {}


def register_wall_correction(wall: WallCorrection) -> WallCorrection:
    held = _corrections.get((wall.name, wall.version))
    if held is not None and held is not wall:
        raise ValueError(f"a wall correction {wall.name} {wall.version} is already registered")
    _corrections[(wall.name, wall.version)] = wall
    return wall


def wall_correction_named(name: str, version: str) -> WallCorrection:
    if (name, version) not in _corrections:
        raise KeyError(f"no wall correction {name} {version} is registered for the boundary-element method; those "
                       f"registered are {wall_corrections()}")
    return _corrections[(name, version)]


def wall_corrections() -> list[str]:
    """The registered corrections, as ``name version``."""
    return sorted(f"{name} {version}" for name, version in _corrections)


def _unbounded(points: Array, parameters: Mapping[str, Any], viscosity: Any) -> Corrected:
    """No wall: nothing to add, and no read to refuse."""
    count = 3 * points.shape[0]
    return Corrected(jnp.zeros((count, count), dtype=points.dtype), jnp.asarray(True))


#: Free space's correction, which adds nothing.
UNBOUNDED = register_wall_correction(WallCorrection(FREE_SPACE, "0.1.0", (), _unbounded))


__all__ = ["FREE_SPACE", "UNBOUNDED", "Corrected", "WallCorrection", "register_wall_correction",
           "wall_correction_named", "wall_corrections"]
