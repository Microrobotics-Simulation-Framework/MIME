"""Separated quadrature for the method of regularised Stokeslets: the nearest-neighbour discretisation (Smith 2018;
Gallagher & Smith 2021), in which the unknowns sit at the body's force points and the integral is taken on a finer
quadrature.

**The discretisation.** The traction is constant on each force point's cell, the quadrature points nearest it (the map
``mime.scene.quadrature_owners`` makes once, at build, and the node holds), and the velocity it makes is collocated at
the force points:

    A[m, n] = sum over q in cell n of S_eps(x_m, y_q) w_q,   R = W^T A^-1 M,

with ``M`` the force points' rigid motions (``rigid_motions``), and ``W`` each cell's wrench per unit traction: its
area, and its area times its centroid's arm for the torque, which a traction constant on the cell makes exact. The
unknown is the traction, so ``A`` is not symmetric (its columns carry the cells' areas) and neither, at the
discretisation's error, is ``R``: the node holds its symmetric part and the build checks the part it drops. With the
quadrature the force points themselves, one to a cell, ``A`` is the point-force matrix with each column times its
sample's weight, and ``R`` is the point-force method's (``mime.nodes.hydro.mrs.resistance``) to rounding.

**Streamed over the quadrature.** The quadrature's points are taken in the order of their force points (a
:class:`Chunks`, made once at build), in chunks of a fixed count: each chunk's pairs with every force point are made,
summed into the window of cells the chunk covers (``segment_sum``, an addition in the points' order) and added to the
matrix, and dropped before the next chunk. No array of every force point against every quadrature point is ever held,
only one chunk's, of at most :data:`CHUNK_BYTES`; the cells' sizes may differ widely (a tensor grid crowds its poles)
at no cost in padding. Each chunk is checkpointed, so the reverse mode keeps a chunk's inputs and makes its pairs
again, never the pairs. No sum here is a matrix product, so each one's order is the compiled program's on any core
(MIME-ANO-125's split was the matrix product's through Eigen).

**At unit viscosity, in the body's frame.** ``A`` is made at viscosity 1 from the samples in the body's frame: the
regularised Stokeslet scales as ``1 / mu`` and turns with the body (``S(Q x, Q y) = Q S(x, y) Q^T``), so the matrix
of a rigid body is the same at every pose, and the node makes it once per build and turns it (:func:`turned`)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np
from jax import Array

from mime.core import jnp
from mime.nodes.hydro.mrs._solve import PRECISION, force_sums, point_forces, rigid_motions

#: The bytes one chunk's temporaries may hold: the pairs of every force point with a chunk's quadrature points, each
#: :data:`PAIR_VALUES` floating values.
CHUNK_BYTES = 1 << 26
#: The temporaries one pair holds while its chunk is made, in floating values: its offset (3), its squared length and
#: scale (2), and its block (9), with room for XLA's fusion of them.
PAIR_VALUES = 16

_lax: Any = cast(Any, jax).lax
_ops: Any = cast(Any, jax).ops
_custom_jvp: Any = cast(Any, jax).custom_jvp
_jvp: Any = cast(Any, jax).jvp
checkpoint: Any = cast(Any, jax).checkpoint


@dataclass(frozen=True, eq=False)
class Chunks:
    """The quadrature in chunks of ``size`` points taken in the order of their force points: ``points`` (``k x
    size`` quadrature indices, the last chunk padded with the quadrature's count, which reads weight zero), each
    point's cell in its chunk's window (``slots``, ``k x size``), each window's first cell (``starts``, ``k``) and the
    windows' width (``width``), so that every chunk's cells lie in ``starts[k] .. starts[k] + width``. Made once, at
    build, from the owner map and the force points' count; numpy arrays, constants of the compiled step."""

    points: np.ndarray
    slots: np.ndarray
    starts: np.ndarray
    width: int
    cells: int

    @classmethod
    def of(cls, owners: np.ndarray, cells: int, itemsize: int = 8) -> Chunks:
        """The chunks of a quadrature whose point ``q`` is in cell ``owners[q]`` of ``cells``: as many points to a
        chunk as :data:`CHUNK_BYTES` holds against every force point."""
        count = len(owners)
        size = max(1, min(count, CHUNK_BYTES // (cells * PAIR_VALUES * itemsize)))
        order = np.argsort(owners, kind="stable")
        chunks = -(-count // size)
        points = np.full(chunks * size, count, dtype=np.int32)
        points[:count] = order
        owner = np.concatenate([owners[order], np.full(chunks * size - count, owners[order][-1])]).reshape(chunks,
                                                                                                         size)
        width = int(np.max(owner[:, -1] - owner[:, 0])) + 1
        starts = np.minimum(owner[:, 0], cells - width).astype(np.int32)
        slots = (owner - starts[:, None]).astype(np.int32)
        return cls(points.reshape(chunks, size), slots, starts, width, cells)


def empty_cells(owners: np.ndarray, cells: int) -> list[int]:
    """The force points no quadrature point is nearest: each one's cell is empty, and the matrix singular."""
    return np.flatnonzero(np.bincount(owners, minlength=cells) == 0).tolist()


def padded(points: Array, weights: Array) -> tuple[Array, Array]:
    """The quadrature's points and weights with one more of each, at the origin with weight zero: what a chunk's
    padding reads."""
    return (jnp.concatenate([points, jnp.zeros((1, 3), dtype=points.dtype)]),
            jnp.concatenate([weights, jnp.zeros((1,), dtype=weights.dtype)]))


def cell_moments(points: Array, weights: Array, owners: np.ndarray, cells: int) -> tuple[Array, Array]:
    """Each cell's area, the sum of its quadrature weights, and its centroid, ``(N,)`` and ``(N, 3)``, each a sum in
    the quadrature's order."""
    held = jnp.asarray(owners)
    area = _ops.segment_sum(weights, held, num_segments=cells)
    first = _ops.segment_sum(weights[:, None] * points, held, num_segments=cells)
    return area, first / area[:, None]


def separated_matrix(force: Array, points: Array, weights: Array, chunks: Chunks, eps_squared: Any) -> Array:
    """``A``, ``3N x 3N`` at unit viscosity: the velocity at each of the ``N`` force points per unit traction on each
    cell, ``sum over q in cell n of S_eps(x_m, y_q) w_q``, from the quadrature's ``points`` and ``weights`` taken in
    ``chunks`` (the module's docstring)."""
    count = force.shape[0]
    held, mass = padded(points, weights)
    eye = jnp.eye(3, dtype=force.dtype)
    width = chunks.width

    def window(index: Array, slot: Array) -> Array:
        y, w = held[index], mass[index]
        r = force[None, :, :] - y[:, None, :]
        r2 = jnp.sum(r * r, axis=-1)
        s = w[:, None] / (8.0 * math.pi * (r2 + eps_squared) ** 1.5)
        blocks = s[..., None, None] * (r[..., :, None] * r[..., None, :] + (r2 + 2.0 * eps_squared)[..., None, None]
                                       * eye)
        return _ops.segment_sum(blocks, slot, num_segments=width, indices_are_sorted=True)

    made = checkpoint(window)

    def add(matrix: Array, chunk: tuple[Array, Array, Array]) -> tuple[Array, None]:
        index, slot, start = chunk
        zero = jnp.zeros((), dtype=start.dtype)
        held_window = _lax.dynamic_slice(matrix, (start, zero, zero, zero), (width, count, 3, 3))
        return _lax.dynamic_update_slice(matrix, held_window + made(index, slot), (start, zero, zero, zero)), None

    start = jnp.zeros((count, count, 3, 3), dtype=force.dtype)
    by_column, _ = _lax.scan(add, start, (jnp.asarray(chunks.points), jnp.asarray(chunks.slots),
                                          jnp.asarray(chunks.starts)))
    return by_column.transpose(1, 2, 0, 3).reshape(3 * count, 3 * count)


def turned(matrix: Array, rotation: Array) -> Array:
    """A ``3N x 3N`` matrix of ``3 x 3`` blocks in the body's frame, in the world's: each block ``Q B Q^T``."""
    count = matrix.shape[0] // 3
    blocks = matrix.reshape(count, 3, count, 3)
    world = jnp.einsum("ia,manb,jb->minj", rotation, blocks, rotation, precision=PRECISION)
    return world.reshape(3 * count, 3 * count)


def turned_resistance(resistance: Array, rotation: Array) -> Array:
    """A 6 x 6 resistance in the body's frame about its origin, in the world's: ``T R T^T``, ``T`` the rotation on
    the force's rows and on the torque's, and its symmetric part, which the turn keeps but for its rounding."""
    zero = jnp.zeros((3, 3), dtype=rotation.dtype)
    both = jnp.block([[rotation, zero], [zero, rotation]])
    found = jnp.matmul(jnp.matmul(both, resistance, precision=PRECISION), both.T, precision=PRECISION)
    return (found + found.T) / 2.0


def separated_resistance(matrix: Array, force: Array, area: Array, centroid: Array, centre: Array) -> Array:
    """``R = W^T A^-1 M`` about ``centre``: the force points' rigid motions solved for each cell's traction, and the
    wrench of those tractions on the cells (their areas and centroids). Not symmetrised."""
    motions = rigid_motions(force, centre)
    wrench = rigid_motions(centroid, centre) * jnp.repeat(area, 3)[:, None]
    return force_sums(wrench, point_forces(matrix, motions))


def unchanged(theta: Mapping[str, Any], built: Mapping[str, Any]) -> Array:
    """Whether every leaf of ``theta`` equals its value in ``built``, read in the leaf's own dtype: a leaf the graph
    holds as built compares equal, and one a study has moved does not."""
    same = [jnp.all(jnp.asarray(theta[key]) == jnp.asarray(built[key], dtype=jnp.asarray(theta[key]).dtype))
            for key in sorted(theta)]
    return jnp.all(jnp.stack(same))


def held_or_made(make: Callable[[Mapping[str, Any]], Array], built: Mapping[str, Any]) -> Callable[
        [Mapping[str, Any], Array], Array]:
    """``make`` held: a function of ``(theta, held)`` whose value is ``held`` (``make(built)``, made once at build)
    where every leaf of ``theta`` is still ``built``'s, and ``make(theta)`` where one has moved (a study's step, a
    difference by a perturbed leaf), so that a held value is never stale; and whose derivative in ``theta`` is
    ``make``'s at ``theta`` in either case (forward mode through ``make``, checkpointed, so that the reverse mode makes
    the value again from ``theta`` rather than keeping what ``make`` holds inside). ``held`` itself has no derivative:
    it is ``make``'s value, not a quantity of its own. Under ``vmap`` over a moved leaf both branches are evaluated,
    since a batched condition selects."""

    def kept(theta: Mapping[str, Any], held: Array) -> Array:
        return held

    def made(theta: Mapping[str, Any], held: Array) -> Array:
        return make(theta).astype(held.dtype)

    @_custom_jvp
    def value(theta: Mapping[str, Any], held: Array) -> Array:
        return _lax.cond(unchanged(theta, built), kept, made, theta, held)

    def tangent(primals: tuple[Mapping[str, Any], Array], tangents: tuple[Mapping[str, Any], Array]) -> tuple[
            Array, Array]:
        theta, held = primals
        found = value(theta, held)
        _, moved = _jvp(checkpoint(make), (theta,), (tangents[0],))
        return found, moved.astype(found.dtype)

    value.defjvp(tangent)
    return cast(Callable[[Mapping[str, Any], Array], Array], value)


def asymmetry(found: Array) -> Array:
    """``|R - R^T| / |R|`` (Frobenius) of a resistance, the part its symmetric part drops, with each row and column
    scaled by the square root of its diagonal entry, so that a force and a torque are compared on one scale."""
    scale = jnp.sqrt(jnp.abs(jnp.diag(found)))
    unit = found / (scale[:, None] * scale[None, :])
    return jnp.linalg.norm(unit - unit.T) / jnp.linalg.norm(unit)


__all__ = ["CHUNK_BYTES", "Chunks", "asymmetry", "cell_moments", "empty_cells", "held_or_made", "padded",
           "separated_matrix", "separated_resistance", "turned", "turned_resistance", "unchanged"]
