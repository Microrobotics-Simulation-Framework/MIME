"""The refinement patch's two transfers (the patch phase; F0's C4, item 59's Q2): a field on a coarser MAC lattice read
at the patch's given values, and the patch's field read at a body's points; each a trilinear read with its exact
transpose, a scatter by the same weights. They are pure functions of the field (linear) and of the places (the patch's
origin, the points): the registered kinds in ``mime.mappings`` wrap them for the lagged edge route, and the one-pass
route calls them directly.

**Lattices.** A MAC component sits on its own lattice: along its own axis at the faces (``i h``), along the other two at
the centres (``(j + 1/2) h``). A coarse field's lattice starts at its ``corner`` (m, in the flow grid's frame: the flow's
grid at zero, a free box at its shift); a periodic axis wraps, any other axis holds its end value beyond its last entry
(an open plane's tangential velocity has zero normal gradient) and counts the points it held. The patch's padded field
(:class:`NestedMACGrid`) is read the same way, its origin in whole fine cells on the fine lattice from the flow grid's
corner, and counts the points outside the box.

**The cell read** (decided 4 (3), the window spike's untested option): each force point's reading is its cell's
average over the body's quadrature (the points nearest it, their weights), as the MRS's image is integrated over the
source's cell (``mime.nodes.hydro.mrs._quadrature.cell_integrated``); by reciprocity of a wall's image the two are
transposes, and the target's average costs only a read at the quadrature's points."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import numpy as np
from jax import Array

from mime.core import jnp
from mime.nodes.flow.fvm._grid import MACGrid, Velocity
from mime.nodes.flow.fvm._nested import NestedMACGrid

_lax: Any = jax.lax
_ops: Any = jax.ops

#: The corners of a cell, as offsets along each axis.
_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int32)

#: The given values, each component's flat in ``NestedMACGrid.given_mask``'s order.
type Given = tuple[Array, Array, Array]


class Stencil(NamedTuple):
    """A trilinear read of ``m`` points on one lattice: each point's eight corners' indices ``(m, 8, 3)`` and weights
    ``(m, 8)``, and how many points fell outside the lattice's extent on an axis that does not wrap."""

    lattice: Array
    weight: Array
    outside: Array


def stencil(coordinates: Array, shape: tuple[int, ...], periodic: tuple[bool, bool, bool]) -> Stencil:
    """The trilinear stencil at ``coordinates`` ``(m, 3)``, in lattice units from its first entry, on a lattice of
    ``shape``: a periodic axis wraps; any other holds its end (the coordinate clamped), each point beyond counted."""
    held: list[Array] = []
    beyond = jnp.zeros(coordinates.shape[0], dtype=bool)
    for axis in range(3):
        q = coordinates[:, axis]
        if not periodic[axis]:
            top = float(shape[axis] - 1)
            beyond = beyond | (q < 0.0) | (q > top)
            q = jnp.clip(q, 0.0, top)
        held.append(q)
    q = jnp.stack(held, axis=1)
    base = jnp.floor(_lax.stop_gradient(q)).astype(jnp.int32)
    fraction = q - base.astype(q.dtype)
    corners = base[:, None, :] + jnp.asarray(_CORNERS)[None, :, :]
    weights = jnp.prod(jnp.where(jnp.asarray(_CORNERS)[None, :, :] == 1, fraction[:, None, :],
                                 1.0 - fraction[:, None, :]), axis=2)
    for axis in range(3):
        if periodic[axis]:
            corners = corners.at[:, :, axis].set(jnp.mod(corners[:, :, axis], shape[axis]))
        else:
            corners = corners.at[:, :, axis].set(jnp.clip(corners[:, :, axis], 0, shape[axis] - 1))
    return Stencil(corners, weights, jnp.sum(beyond.astype(jnp.int32)))


def sampled(field: Array, at: Stencil) -> Array:
    """The trilinear read ``(m,)`` of ``field`` by ``at``."""
    values = field[at.lattice[:, :, 0], at.lattice[:, :, 1], at.lattice[:, :, 2]]
    return jnp.sum(at.weight * values, axis=1)


def spread(values: Array, at: Stencil, shape: tuple[int, ...]) -> Array:
    """The read's transpose: ``values`` ``(m,)`` spread onto a lattice of ``shape`` by the same weights."""
    out = jnp.zeros(shape, dtype=values.dtype)
    return out.at[at.lattice[:, :, 0], at.lattice[:, :, 1], at.lattice[:, :, 2]].add(at.weight * values[:, None])


def _lattice_coordinates(points: Array, spacing: Any, component: int, corner: Array) -> Array:
    """Points in a MAC component's lattice units: faces along its own axis, centres along the others."""
    shift = jnp.asarray([0.0 if axis == component else 0.5 for axis in range(3)], dtype=points.dtype)
    return (points - corner) / spacing - shift


def patch_points(grid: NestedMACGrid, origin: Array, component: int, dtype: Any) -> Array:
    """The ``component``-th given values' points in the flow grid's frame (m): the patch's corner at ``origin`` whole
    fine cells."""
    corner = origin.astype(dtype) * jnp.asarray(grid.spacing, dtype=dtype)
    return grid.given_points(component, dtype) + corner


def boundary_stencils(grid: NestedMACGrid, origin: Array, coarse: MACGrid,
                      shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
                      corner: Array, dtype: Any) -> tuple[Stencil, Stencil, Stencil]:
    """The trilinear stencils from a coarse field's component lattices (``shapes``, its corner at ``corner``) to the
    patch's given values."""
    held: list[Stencil] = []
    for component in range(3):
        points = patch_points(grid, origin, component, dtype)
        coordinates = _lattice_coordinates(points, jnp.asarray(coarse.spacing, dtype=dtype), component,
                                           jnp.asarray(corner, dtype=dtype))
        held.append(stencil(coordinates, shapes[component], coarse.periodic))
    return (held[0], held[1], held[2])


def patch_boundary(grid: NestedMACGrid, origin: Array, coarse: MACGrid, velocity: Velocity, corner: Array
                   ) -> tuple[Given, Array]:
    """A coarse MAC field (on ``coarse``, its corner at ``corner``, m) read trilinearly at the patch's given values;
    and how many of them fell beyond a coarse axis that does not wrap (held at its end)."""
    dtype = velocity[0].dtype
    shapes = (tuple(velocity[0].shape), tuple(velocity[1].shape), tuple(velocity[2].shape))
    stencils = boundary_stencils(grid, origin, coarse, shapes, corner, dtype)
    values = (sampled(velocity[0], stencils[0]), sampled(velocity[1], stencils[1]), sampled(velocity[2], stencils[2]))
    return values, stencils[0].outside + stencils[1].outside + stencils[2].outside


def patch_boundary_transpose(grid: NestedMACGrid, origin: Array, coarse: MACGrid,
                             shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], corner: Array,
                             cotangent: Given) -> Velocity:
    """:func:`patch_boundary`'s transpose: the given values' cotangent spread onto the coarse field's lattices."""
    dtype = cotangent[0].dtype
    stencils = boundary_stencils(grid, origin, coarse, shapes, corner, dtype)
    return (spread(cotangent[0], stencils[0], shapes[0]), spread(cotangent[1], stencils[1], shapes[1]),
            spread(cotangent[2], stencils[2], shapes[2]))


def read_stencils(grid: NestedMACGrid, origin: Array, points: Array) -> tuple[Stencil, Stencil, Stencil]:
    """The trilinear stencils from the patch's padded components to ``points`` (``(n, 3)``, flow frame, m)."""
    dtype = points.dtype
    h = jnp.asarray(grid.spacing, dtype=dtype)
    local = points - origin.astype(dtype) * h
    held: list[Stencil] = []
    for component in range(3):
        shift = jnp.asarray([0.0 if axis == component else -0.5 for axis in range(3)], dtype=dtype)
        held.append(stencil(local / h - shift, grid.padded_shape(component), (False, False, False)))
    return (held[0], held[1], held[2])


def patch_read(grid: NestedMACGrid, origin: Array, field: Velocity, points: Array) -> tuple[Array, Array]:
    """The patch's padded field read trilinearly at ``points``, ``(n, 3)``; and how many points' reads fell outside
    the box on any component (held at its end)."""
    stencils = read_stencils(grid, origin, points.astype(field[0].dtype))
    values = jnp.stack([sampled(field[c], stencils[c]) for c in range(3)], axis=1)
    return values, stencils[0].outside + stencils[1].outside + stencils[2].outside


def patch_read_transpose(grid: NestedMACGrid, origin: Array, points: Array, cotangent: Array) -> Velocity:
    """:func:`patch_read`'s transpose: ``(n, 3)`` cotangents spread onto the padded components."""
    stencils = read_stencils(grid, origin, points.astype(cotangent.dtype))
    held = [spread(cotangent[:, c], stencils[c], grid.padded_shape(c)) for c in range(3)]
    return (held[0], held[1], held[2])


class Quadrature(NamedTuple):
    """A body's quadrature as the cell read takes it: its points ``(q, 3)`` (flow frame, m), their weights ``(q,)``
    (m^2), the force point each belongs to ``(q,)`` (int32), and each force point's cell area ``(n,)``."""

    points: Array
    weights: Array
    owners: Array
    areas: Array


def cell_read(grid: NestedMACGrid, origin: Array, field: Velocity, quadrature: Quadrature) -> tuple[Array, Array]:
    """Each force point's cell average of the patch's padded field over the body's quadrature, ``(n, 3)``; and how
    many quadrature reads fell outside the box."""
    at, outside = patch_read(grid, origin, field, quadrature.points)
    weighted = at * quadrature.weights[:, None].astype(at.dtype)
    sums = _ops.segment_sum(weighted, quadrature.owners, num_segments=quadrature.areas.shape[0])
    return sums / quadrature.areas[:, None].astype(at.dtype), outside


def cell_read_transpose(grid: NestedMACGrid, origin: Array, quadrature: Quadrature, cotangent: Array) -> Velocity:
    """:func:`cell_read`'s transpose."""
    dtype = cotangent.dtype
    per_cell = cotangent / quadrature.areas[:, None].astype(dtype)
    at_points = per_cell[quadrature.owners] * quadrature.weights[:, None].astype(dtype)
    return patch_read_transpose(grid, origin, quadrature.points, at_points)


__all__ = ["Given", "Quadrature", "Stencil", "boundary_stencils", "cell_read", "cell_read_transpose",
           "patch_boundary", "patch_boundary_transpose", "patch_points", "patch_read", "patch_read_transpose",
           "read_stencils", "sampled", "spread", "stencil"]
