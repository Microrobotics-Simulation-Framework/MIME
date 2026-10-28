"""How a confining wall reaches the boundary-element method: as a wall correction, a registered function the BEM's node
evaluates by name. It takes the body's samples in the world frame (``N x 3``), its parameters by name (each a leaf of
the node's ``wall_<name>``) and the viscosity, and returns the ``3N x 3N`` block matrix added to the free-space matrix:
the velocity at each sample per unit force at each that the wall's image system adds, in m/(N s), the matrix's own
units. The node knows Stokes kernels and quadrature, and nothing about tubes or planes.

Free space is the trivial wall, with no correction. The straight circular tube's correction is the next (its table of
the Liron & Shahar image system), registered beside its wall kind."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from jax import Array

from mime.core import jnp

#: The free-space wall's name: a medium with no wall, unbounded.
FREE_SPACE = "free_space"


@dataclass(frozen=True)
class WallCorrection:
    """A wall's correction to the free-space matrix: its ``name``, the names of the ``parameters`` it reads (the
    node's ``wall_<name>`` leaves, in SI), and the function."""

    name: str
    parameters: tuple[str, ...]
    correction: Callable[[Array, Mapping[str, Any], Any], Array]


_corrections: dict[str, WallCorrection] = {}


def register_wall_correction(wall: WallCorrection) -> WallCorrection:
    held = _corrections.get(wall.name)
    if held is not None and held is not wall:
        raise ValueError(f"a wall correction {wall.name!r} is already registered")
    _corrections[wall.name] = wall
    return wall


def wall_correction_named(name: str) -> WallCorrection:
    if name not in _corrections:
        raise KeyError(f"no wall correction {name!r} is registered; those registered are {sorted(_corrections)}")
    return _corrections[name]


def wall_corrections() -> list[str]:
    """The registered walls' names."""
    return sorted(_corrections)


def _unbounded(points: Array, parameters: Mapping[str, Any], viscosity: Any) -> Array:
    """No wall: nothing to add."""
    count = 3 * points.shape[0]
    return jnp.zeros((count, count), dtype=points.dtype)


register_wall_correction(WallCorrection(FREE_SPACE, (), _unbounded))


__all__ = ["FREE_SPACE", "WallCorrection", "register_wall_correction", "wall_correction_named", "wall_corrections"]
