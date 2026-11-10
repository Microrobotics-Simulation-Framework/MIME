"""The switch of a near-singular evaluation, shared by its two callers (design #308): a kernel summed at each source
cell's collocation point far away and integrated over the cell's own quadrature near it (the boundary-integral
literature's near-singular quadrature, the fast multipole method's near/far split), the two switched by a distance in a
local size. Its callers are the wall's self-image (``mime.nodes.hydro.mrs``'s ``near_wall_image``) and the near field's
cross-kernel (``MRSOperator.cross``).

It owns the mechanics only: the C1 smoothstep the blend is weighted by (:func:`smoothstep`), the ``cond`` that skips the
near work when nothing is within the reach (:func:`within_reach`), a fixed budget chosen nearest first with its
shortfall (:func:`nearest_cells` by cells, :func:`nearest_points` by quadrature points), the blend (:func:`blended`),
and a kernel integrated over cells gathered at the step (:func:`integrated_cells`). The selections are integers and
masks, held out of every derivative; the blend's weight keeps its derivative. Each caller keeps its kernel and near
evaluation, its distance and the size it measures it in, its measured reach and budget, and its granularity (the wall
per cell, the cross-kernel per pair of bodies)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import jax
from jax import Array

from mime.core import jnp
from mime.nodes.hydro.mrs._quadrature import CHUNK_BYTES, Kernel

_lax: Any = cast(Any, jax).lax
_ops: Any = cast(Any, jax).ops
checkpoint: Any = cast(Any, jax).checkpoint


def smoothstep(distance: Any, reach: Any) -> Array:
    """The near evaluation's weight at ``distance``: 1 within half of ``reach``, 0 beyond it, and a smoothstep between
    (continuous with its first derivative)."""
    t = jnp.clip((reach - distance) / (0.5 * reach), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def within_reach[T](distance: Array, reach: Any, near: Callable[[], T], far: Callable[[], T]) -> T:
    """``near()`` where any of ``distance`` lies within ``reach``, else ``far()`` (a ``cond``: a step with nothing in
    reach does none of the near work). The two agree where nothing is in reach, every weight zero."""
    return _lax.cond(jnp.any(distance < reach), near, far)


class Selection(NamedTuple):
    """A budget's choice: ``chosen`` (the cells' indices, nearest first, or a mask over the cells) and ``short``,
    whether a cell within the reach was left out."""

    chosen: Array
    short: Array


def nearest_cells(distance: Array, budget: int, reach: Any) -> Selection:
    """The ``budget`` cells nearest by ``distance`` (``top_k``, so a tie at the budget's edge breaks as it always has),
    and whether the first cell left out lay within ``reach``."""
    count = distance.shape[0]
    nearest = _lax.top_k(-distance, min(budget + 1, count))[1]
    chosen = nearest[:budget]
    short = jnp.asarray(False) if budget >= count else distance[nearest[budget]] < reach
    return Selection(chosen, short)


class PointBudget(NamedTuple):
    """A budget of quadrature points filled by whole cells, nearest first: each slot's cell and its index within the
    cell, whether the slot is filled, and the mask of the cells that fit."""

    cell: Array
    within: Array
    valid: Array
    fits: Array


def nearest_points(distance: Array, counts: Any, budget: int) -> PointBudget:
    """Cells taken nearest first by ``distance`` (held out of every derivative by the caller) while their points
    (``counts``, each cell's, fixed at build) fit ``budget`` slots."""
    cells = distance.shape[0]
    nearest = jnp.argsort(distance)
    sizes = jnp.asarray(counts)
    filled = jnp.cumsum(sizes[nearest])
    fits = filled <= budget
    chosen = jnp.zeros(cells, dtype=bool).at[nearest].set(fits)
    slot = jnp.arange(budget)
    rank = jnp.searchsorted(filled, slot, side="right")
    valid = (rank < cells) & fits[jnp.minimum(rank, cells - 1)]
    cell = nearest[jnp.minimum(rank, cells - 1)]
    within = slot - (filled[jnp.minimum(rank, cells - 1)] - sizes[cell])
    return PointBudget(cell, within, valid, chosen)


def blended(far: Array, near: Array, weight: Array) -> Array:
    """``far``'s columns moved towards ``near``'s by each cell's ``weight`` (three columns a cell)."""
    blend = jnp.repeat(weight, 3)[None, :]
    return far + blend * (near - far)


def integrated_cells(kernel: Kernel, targets: Array, points: Array, weights: Array, cell: Array, cells: int) -> Array:
    """``3T x 3C``: ``kernel`` integrated over each cell's points, ``sum over q in cell c of K(x_t, y_q) w_q``, from
    ``points`` and ``weights`` gathered at the step (each point's cell ``cell``; a point of weight zero adds nothing),
    in chunks of a fixed count, each checkpointed and summed into the cells by ``segment_sum``: never an array of every
    target against every point."""
    count, total = targets.shape[0], points.shape[0]
    size = max(1, min(total, CHUNK_BYTES // (count * 16 * 8)))
    chunks = -(-total // size)
    pad = chunks * size - total
    held_points = jnp.concatenate([points, jnp.broadcast_to(points[:1], (pad, 3))]).reshape(chunks, size, 3)
    held_weights = jnp.concatenate([weights, jnp.zeros(pad, dtype=weights.dtype)]).reshape(chunks, size)
    held_cells = jnp.concatenate([cell, jnp.zeros(pad, dtype=cell.dtype)]).reshape(chunks, size)

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


__all__ = ["PointBudget", "Selection", "blended", "integrated_cells", "nearest_cells", "nearest_points", "smoothstep",
           "within_reach"]
