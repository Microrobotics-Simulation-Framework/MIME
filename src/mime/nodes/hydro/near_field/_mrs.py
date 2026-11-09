"""The MRS as the near field's first body model: dev1's separated quadrature, composed from the body's own
``MRSDragNode`` (built as the per-body rung builds it, held by the joint node, never added to the graph), so that no
piece of the MRS is written twice.

**Its self-block** is dev1's: the free part's matrix made once per build in the body's frame at unit viscosity
(``held``), turned to the pose and divided by the viscosity, with the wall's self-image added exactly as dev1's step
adds it (at the force points, and integrated over the cells nearest the wall on a wall quadrature).

**Its cross-kernel** is its regularised Stokeslet between its cells and another body's force points, switched by the
pair's distance as dev1's wall image is by the wall's: a pair farther apart than the reach takes each cell's kernel
at its force point (times its area); a pair within half the reach takes every cell's kernel integrated over the cell's
own quadrature; between, the two blocks blended smoothly by the pair's least distance (:func:`cross_weight`). The
distance is each target's from each cell's force point in that cell's own size, ``sqrt(area)``, not in a global
spacing, since feature-graded sampling varies the spacing over a body. The switch is the pair's, not each cell's:
near contact the joint system amplifies a small error in any entry of a cross-block, so a cross-block within the reach
is integrated whole (``CROSS_SHARE``'s measurement). **The budget** is a fixed count of the source's quadrature
points (a share of them, fixed at build, so the step's shapes are static; by default all of them): the cells nearest
the targets are integrated, nearest first, while their points fit; a cell that does not fit keeps its point column,
and the step reports that the pair ran short."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, cast

import jax
import numpy as np
from jax import Array

from mime.core import jnp, quaternion
from mime.nodes.hydro.mrs._node import MRSDragNode, near_wall_image, on_the_cells
from mime.nodes.hydro.mrs._quadrature import CHUNK_BYTES, regularised_stokeslet, turned
from mime.nodes.hydro.mrs._solve import PRECISION
from mime.nodes.hydro.mrs._walls import Corrected, WallCorrection
from mime.nodes.hydro.near_field._contract import BODY_FRAME_CONSTANT, CROSS_KERNEL, PRECONDITIONER, Placed

_lax: Any = cast(Any, jax).lax
_ops: Any = cast(Any, jax).ops
checkpoint: Any = cast(Any, jax).checkpoint

#: The cross-kernel's reach, in each source cell's own size ``sqrt(area)``: a pair of bodies farther apart than it takes
#: the kernel at the source's force points, a pair within half of it the kernel integrated over every cell, a smoothstep
#: between. A tolerance-backed numerical setting: each registered hull kind's conformance case (point against
#: integrated at the reach's edge, within 0.5 % of each grand-resistance entry's diagonal scale) fails a kind that needs
#: a longer one. A labelled default from those cases, measured in W3's pull request (float64): the box needed more than
#: 3 (0.63 % there, 0.41 % at 3.5, 0.28 % at 4), every other kind passed at 3 (the sphere 0.14 %, at 4 0.07 %). Spike
#: S3's sphere data agree: on two spheres squeezing, the point kernel was within 0.3 % of the integrated one beyond a
#: gap of 2.6 force spacings, +4 % at 1.3, -17.5 % at 0.5.
REACH = 4.0
#: The share of a source's quadrature points one cross-kernel integrates at most, fixed at build (its budget), the
#: nearest cells first. All of them by default, from a measurement at MIME's 338-point sphere in float64 (W3's pull
#: request): with each cell switched by its own distance from the targets (integrated within half the reach, at its
#: force point beyond), two spheres a tenth of a radius apart squeezed 5.8 % over Brenner and Maude's series against
#: 3.2 % with every cell integrated (4.8 % at twice the reach), and three in a triangle there had a lagged gain of 1.57
#: against 0.88: near contact the joint system amplifies the point kernel's error in every cell. A lower share is a cost
#: setting, reported where a pair runs short (``cross_cells_short``).
CROSS_SHARE = 1.0


def cross_weight(distance: Array, reach: Any) -> Array:
    """A pair's weight on its integrated cross-block from its least distance, in its cells' own sizes: 1 within half
    the reach, 0 beyond it, a smoothstep between (continuous with its first derivative), as dev1's ``reach_weight`` for
    the wall's image."""
    t = jnp.clip((reach - distance) / (0.5 * reach), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclasses.dataclass(frozen=True)
class Quadrature:
    """A body's separated quadrature placed in the world, for its cross-kernel: its points ``(Q, 3)`` and weights
    ``(Q,)``."""

    points: Array
    weights: Array


class MRSOperator:
    """The MRS near-field body model of one body (``member``, its per-body node as the rung builds it). ``budget`` is
    the count of quadrature points a cross-kernel from it integrates at most (:data:`CROSS_SHARE` of them by
    default)."""

    def __init__(self, member: MRSDragNode, share: float = CROSS_SHARE) -> None:
        if not member.separated:
            raise ValueError("the near field's MRS body model is the separated quadrature's (drag 0.4.0 and later)")
        self.member = member
        owners = np.asarray(cast(np.ndarray, member.owners))
        cells = member.sample_count
        self.order = np.argsort(owners, kind="stable").astype(np.int32)
        self.sizes = np.bincount(owners, minlength=cells).astype(np.int32)
        self.starts = (np.cumsum(self.sizes) - self.sizes).astype(np.int32)
        self.budget = max(int(self.sizes.max()), min(len(owners), math.ceil(share * len(owners))))

    @property
    def count(self) -> int:
        return self.member.sample_count

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({CROSS_KERNEL, BODY_FRAME_CONSTANT, PRECONDITIONER})

    def leaf_names(self) -> tuple[str, ...]:
        """The leaves its discretisation is made from (the kind's numbers, the part's placement, the blob)."""
        return self.member.body_leaf_names()

    def placed(self, leaves: Mapping[str, Any], pose: Array) -> Placed:
        theta = {key: leaves[key] for key in self.leaf_names()}
        body = self.member.body(theta)
        turn = quaternion.rotation_matrix(pose[3:])
        force = pose[:3] + jnp.matmul(body.force, turn.T, precision=PRECISION)
        centroid = pose[:3] + jnp.matmul(body.centroid, turn.T, precision=PRECISION)
        points = pose[:3] + jnp.matmul(body.points, turn.T, precision=PRECISION)
        return Placed(force, centroid, body.area, body.eps_squared, pose[:3], Quadrature(points, body.weights))

    def self_block(self, placed: Placed, leaves: Mapping[str, Any], held: Array | None, pose: Array, viscosity: Any,
                   wall: WallCorrection, parameters: Mapping[str, Any]) -> tuple[Array, Array, Array]:
        """dev1's: the held free part turned to the pose over the viscosity, and the wall's self-image as dev1's step
        adds it."""
        if held is None:
            raise ValueError("the MRS body model's free part is made once per build and held")
        turn = quaternion.rotation_matrix(pose[3:])
        free = turned(held, turn) / viscosity
        if wall.adds_nothing:
            return free, jnp.asarray(True), jnp.asarray(False)
        member = self.member
        corrected = wall.correction(placed.force, parameters, viscosity, eps_squared=placed.eps_squared)
        image = Corrected(corrected.matrix * jnp.repeat(placed.area, 3)[None, :], corrected.accepted)
        short = jnp.asarray(False)
        spacing = leaves.get("force_spacing")
        held_quadrature, capped, chunks = member.wall_quadrature, member.wall_capped, member.wall_chunks
        if held_quadrature is not None and capped is not None and chunks is not None and wall.distance is not None \
                and spacing is not None:
            theta = {key: leaves[key] for key in self.leaf_names()}
            cells = member.quadrature_samples({**theta, "viscosity": jnp.asarray(theta["blob"])}, held_quadrature[0])
            kept, runs, owners = capped
            weights = _ops.segment_sum(cells.weights, jnp.asarray(runs), num_segments=len(kept))
            image, short = near_wall_image(
                wall, image, placed.force,
                pose[:3] + jnp.matmul(cells.points[jnp.asarray(kept)], turn.T, precision=PRECISION),
                on_the_cells(weights, owners, placed.area), wall.distance(placed.force, parameters) / spacing,
                None if member.wall_every else cast(np.ndarray, member.wall_table), member.wall_budget, chunks,
                parameters, viscosity, placed.eps_squared)
        return free + image.matrix, image.accepted, short

    def cross(self, placed: Placed, targets: Array, reach: Any) -> tuple[Array, Array]:
        point = point_block(targets, placed)
        quadrature = cast(Quadrature, placed.own)
        apart = jnp.linalg.norm(targets[:, None, :] - placed.force[None, :, :], axis=-1)
        pairs = apart / jnp.sqrt(placed.area)[None, :]
        held = _lax.stop_gradient(jnp.min(pairs, axis=0))
        return _lax.cond(jnp.any(held < reach),
                         lambda: self._near(point, placed, quadrature, targets, pairs, held, reach),
                         lambda: (point, jnp.asarray(False)))

    def _near(self, point: Array, placed: Placed, quadrature: Quadrature, targets: Array, pairs: Array,
              held: Array, reach: Any) -> tuple[Array, Array]:
        """The cross-kernel of a pair within the reach: the nearest cells integrated while their points fit the
        budget, the whole block blended by the pair's least distance (the least of ``pairs``, each target's from each
        cell in the cell's size, its derivative kept), so that the blend is continuous in every position; ``held``
        (each cell's least distance, held out of every derivative) chooses the cells. Short where a cell was left out."""
        cells = self.count
        nearest = jnp.argsort(held)
        sizes = jnp.asarray(self.sizes)
        filled = jnp.cumsum(sizes[nearest])
        fits = filled <= self.budget
        chosen = jnp.zeros(cells, dtype=bool).at[nearest].set(fits)
        slot = jnp.arange(self.budget)
        rank = jnp.searchsorted(filled, slot, side="right")
        valid = (rank < cells) & fits[jnp.minimum(rank, cells - 1)]
        cell = nearest[jnp.minimum(rank, cells - 1)]
        within = slot - (filled[jnp.minimum(rank, cells - 1)] - sizes[cell])
        index = jnp.asarray(self.order)[jnp.clip(jnp.asarray(self.starts)[cell] + within, 0, len(self.order) - 1)]
        weights = jnp.where(valid, quadrature.weights[index], 0.0)
        integrated = integrated_cells(targets, quadrature.points[index], weights, jnp.where(valid, cell, 0), cells,
                                      placed.eps_squared)
        weights = cross_weight(jnp.min(pairs), reach) * chosen.astype(pairs.dtype)
        blend = jnp.repeat(weights, 3)[None, :]
        return point + blend * (integrated - point), ~jnp.all(chosen)


def point_block(targets: Array, source: Placed) -> Array:
    """``3T x 3N``: the source's regularised Stokeslet at unit viscosity from each cell's force point to each target,
    times the cell's area (each cell's traction placed at its force point)."""
    blocks, _ = regularised_stokeslet(source.eps_squared)(targets, source.force)
    blocks = blocks * source.area[None, :, None, None]
    return blocks.transpose(0, 2, 1, 3).reshape(3 * targets.shape[0], 3 * source.count)


def integrated_cells(targets: Array, points: Array, weights: Array, cell: Array, cells: int, eps_squared: Any) -> Array:
    """``3T x 3C``: the regularised Stokeslet at unit viscosity integrated over each cell's points, ``sum over q in
    cell c of S(x_t, y_q) w_q``, from ``points`` and ``weights`` gathered at the step (each point's cell ``cell``; a
    point of weight zero adds nothing), in chunks of a fixed count, each checkpointed and summed into the cells by
    ``segment_sum``: never an array of every target against every point."""
    count, total = targets.shape[0], points.shape[0]
    size = max(1, min(total, CHUNK_BYTES // (count * 16 * 8)))
    chunks = -(-total // size)
    pad = chunks * size - total
    held_points = jnp.concatenate([points, jnp.broadcast_to(points[:1], (pad, 3))]).reshape(chunks, size, 3)
    held_weights = jnp.concatenate([weights, jnp.zeros(pad, dtype=weights.dtype)]).reshape(chunks, size)
    held_cells = jnp.concatenate([cell, jnp.zeros(pad, dtype=cell.dtype)]).reshape(chunks, size)
    kernel = regularised_stokeslet(eps_squared)

    def piece(at: Array, mass: Array, owner: Array) -> Array:
        blocks, _ = kernel(targets, at)
        weighted = (blocks * mass[None, :, None, None]).transpose(1, 0, 2, 3)
        return _ops.segment_sum(weighted, owner, num_segments=cells)

    made = checkpoint(piece)

    def add(total_so_far: Array, chunk: tuple[Array, Array, Array]) -> tuple[Array, None]:
        return total_so_far + made(*chunk), None

    start = jnp.zeros((cells, count, 3, 3), dtype=targets.dtype)
    by_cell, _ = _lax.scan(add, start, (held_points, held_weights, held_cells))
    return by_cell.transpose(1, 2, 0, 3).reshape(3 * count, 3 * cells)


__all__ = ["CROSS_SHARE", "REACH", "MRSOperator", "Quadrature", "cross_weight", "integrated_cells", "point_block"]
