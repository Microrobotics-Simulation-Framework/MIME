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
    def of(cls, owners: np.ndarray, cells: int, itemsize: int = 8, pair_values: int = PAIR_VALUES) -> Chunks:
        """The chunks of a quadrature whose point ``q`` is in cell ``owners[q]`` of ``cells``: as many points to a
        chunk as :data:`CHUNK_BYTES` holds against every force point, each pair ``pair_values`` floating values."""
        count = len(owners)
        size = max(1, min(count, CHUNK_BYTES // (cells * pair_values * itemsize)))
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


#: A kernel between two point sets: the ``3 x 3`` block at each target per unit force at each source, ``(T, S, 3, 3)``
#: from ``targets`` ``(T, 3)`` and ``sources`` ``(S, 3)``, and whether every evaluation was accepted (a wall's table read
#: inside its range; always for a closed form).
type Kernel = Callable[[Array, Array], tuple[Array, Array]]


def cell_integrated(kernel: Kernel, targets: Array, points: Array, weights: Array, chunks: Chunks) -> tuple[
        Array, Array]:
    """A kernel integrated over source cells: ``3T x 3C``, the velocity at each of the ``T`` targets per unit traction on
    each of the ``C`` cells, ``sum over q in cell c of K(x_t, y_q) w_q``, the quadrature's ``points`` and ``weights``
    taken in ``chunks`` (each chunk's pairs with every target made, summed into the window of cells it covers and
    dropped: never an array of every target against every source); and whether every evaluation was accepted. The seam
    between any targets and any body's cells: the free part's matrix is the regularised Stokeslet's
    (:func:`separated_matrix`, targets the body's own force points), a wall's image the wall's kernel
    (``mime.nodes.hydro.mrs.cell_image``), and a body's cells against another body's force points the same call. The
    chunks are cut for the kernel's temporaries (``Chunks.of``'s ``pair_values``: a tabulated kernel holds more)."""
    count = targets.shape[0]
    held, mass = padded(points, weights)
    width = chunks.width

    def window(index: Array, slot: Array) -> tuple[Array, Array]:
        blocks, accepted = kernel(targets, held[index])
        weighted = (blocks * mass[index][None, :, None, None]).transpose(1, 0, 2, 3)
        return _ops.segment_sum(weighted, slot, num_segments=width, indices_are_sorted=True), accepted

    made = checkpoint(window)

    def add(carry: tuple[Array, Array], chunk: tuple[Array, Array, Array]) -> tuple[tuple[Array, Array], None]:
        matrix, accepted = carry
        index, slot, start = chunk
        zero = jnp.zeros((), dtype=start.dtype)
        held_window = _lax.dynamic_slice(matrix, (start, zero, zero, zero), (width, count, 3, 3))
        found, ok = made(index, slot)
        return (_lax.dynamic_update_slice(matrix, held_window + found, (start, zero, zero, zero)),
                accepted & ok), None

    start = (jnp.zeros((chunks.cells, count, 3, 3), dtype=targets.dtype), jnp.asarray(True))
    (by_column, accepted), _ = _lax.scan(add, start, (jnp.asarray(chunks.points), jnp.asarray(chunks.slots),
                                                      jnp.asarray(chunks.starts)))
    return by_column.transpose(1, 2, 0, 3).reshape(3 * count, 3 * chunks.cells), accepted


def regularised_stokeslet(eps_squared: Any) -> Kernel:
    """The regularised Stokeslet at unit viscosity as a :data:`Kernel`, ``((r^2 + 2 eps^2) I + r r^T) / (8 pi (r^2 +
    eps^2)^(3/2))``, ``r`` the target less the source."""

    def kernel(targets: Array, sources: Array) -> tuple[Array, Array]:
        r = targets[:, None, :] - sources[None, :, :]
        r2 = jnp.sum(r * r, axis=-1)
        s = 1.0 / (8.0 * math.pi * (r2 + eps_squared) ** 1.5)
        eye = jnp.eye(3, dtype=targets.dtype)
        blocks = s[..., None, None] * (r[..., :, None] * r[..., None, :] + (r2 + 2.0 * eps_squared)[..., None, None]
                                       * eye)
        return blocks, jnp.asarray(True)

    return kernel


def separated_matrix(force: Array, points: Array, weights: Array, chunks: Chunks, eps_squared: Any) -> Array:
    """``A``, ``3N x 3N`` at unit viscosity: the velocity at each of the ``N`` force points per unit traction on each
    cell, ``sum over q in cell n of S_eps(x_m, y_q) w_q``, from the quadrature's ``points`` and ``weights`` taken in
    ``chunks`` (the module's docstring): the regularised Stokeslet integrated over the body's own cells
    (:func:`cell_integrated`)."""
    return cell_integrated(regularised_stokeslet(eps_squared), force, points, weights, chunks)[0]


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


def scaled_least_eigenvalue(found: Array) -> Array:
    """The least eigenvalue of a resistance's symmetric part with each row and column scaled by the square root of its
    diagonal entry's magnitude: a force and a torque on one scale, and above zero exactly where the resistance is
    positive definite (the scaling is a congruence, which keeps the signs of the eigenvalues, a negative diagonal entry
    included). Not a number where a diagonal entry is zero."""
    symmetric = (found + found.T) / 2.0
    scale = jnp.sqrt(jnp.abs(jnp.diag(symmetric)))
    return jnp.linalg.eigvalsh(symmetric / (scale[:, None] * scale[None, :]))[0]


def asymmetry(found: Array) -> Array:
    """``|R - R^T| / |R|`` (Frobenius) of a resistance, the part its symmetric part drops, with each row and column
    scaled by the square root of its diagonal entry, so that a force and a torque are compared on one scale."""
    scale = jnp.sqrt(jnp.abs(jnp.diag(found)))
    unit = found / (scale[:, None] * scale[None, :])
    return jnp.linalg.norm(unit - unit.T) / jnp.linalg.norm(unit)


__all__ = ["CHUNK_BYTES", "Chunks", "Kernel", "asymmetry", "cell_integrated", "cell_moments", "empty_cells",
           "held_or_made", "padded", "regularised_stokeslet", "scaled_least_eigenvalue", "separated_matrix", "separated_resistance", "turned", "turned_resistance",
           "unchanged"]
