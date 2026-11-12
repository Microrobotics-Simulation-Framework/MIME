"""MADDENING 0.4's workarounds for the refinement patch's edges, each written to be removed (the record's round 1 item
12, round 2 item 6; the list in ``mime.mappings``' README).

- **A column per MAC component** (:func:`columns`, :func:`from_columns`): a mapping maps ``(n_source, k)`` to
  ``(n_target, k)`` with one ``k``, and a MAC field's three components sit on three lattices, so each goes to its own
  edge as a flat column. Removed by per-column kinds (a candidate MADDENING 0.5.0 ask).
- **The read's points in the patch's frame** (:func:`in_patch_frame`): an edge has one anchor, and the patch's read needs
  two geometries (the patch's origin and the body's points); the patch's node holds the points moved into its own frame,
  so the read's edge anchors there and its map stays linear with an exact transpose. (Carrying the origin in the
  patch's field, the spike's form, made the read's transpose need the origin it cannot see.) Removed by a mapping
  anchored at two nodes' geometries (a candidate ask).
- **The patch's origin as float state holding whole cells** (:func:`origin_geometry`): a geometry is a float array; the
  origin's whole cells are rounded back where they are read."""

from __future__ import annotations

from jax import Array

from mime.core import jnp


def columns(velocity: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
    """Each MAC component flat, one edge's column each."""
    return (velocity[0].reshape(-1), velocity[1].reshape(-1), velocity[2].reshape(-1))


def from_columns(flat: tuple[Array, Array, Array], shapes: tuple[tuple[int, ...], ...]) -> tuple[Array, Array, Array]:
    return (flat[0].reshape(shapes[0]), flat[1].reshape(shapes[1]), flat[2].reshape(shapes[2]))


def in_patch_frame(points: Array, origin: Array, spacing: float) -> Array:
    """Points (m, flow frame) moved into the patch's frame (m from its corner at ``origin`` whole fine cells)."""
    return points - jnp.asarray(origin).astype(points.dtype) * spacing


def origin_geometry(origin: Array, dtype: object) -> Array:
    """The patch's origin as a geometry, ``(1, 3)`` floats holding whole cells."""
    return jnp.asarray(origin).astype(dtype).reshape(1, 3)


__all__ = ["columns", "from_columns", "in_patch_frame", "origin_geometry"]
