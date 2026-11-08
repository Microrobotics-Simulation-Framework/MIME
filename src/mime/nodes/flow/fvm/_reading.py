"""The FVM field's reading at points: each MAC component interpolated trilinearly from its own face lattice, on the
periodic box (the quantity it provides, ``field.flow.velocity``, in m/s). The interpolation is second order in the
fluid and first order across the wall's band; which time level the reading belongs to, and how a body's reading is
weighted, are C5's (SF-2's time level is open)."""

from __future__ import annotations

from typing import Any

from jax import Array

from mime.core import jnp
from mime.nodes.flow.fvm._grid import MACGrid, Velocity

#: Each component's face lattice offset, in cells, from the box's corner: the component's own axis on faces, the
#: others at cell centres.
_OFFSETS = ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))


def _component_at(values: Array, cells: tuple[int, int, int], offset: tuple[float, float, float], spacing: Any,
                  points: Array) -> Array:
    xi = points / spacing - jnp.asarray(offset, dtype=points.dtype)
    base = jnp.floor(xi)
    weight = xi - base
    index = base.astype(jnp.int32)
    total = jnp.zeros(points.shape[0], dtype=values.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corner = (jnp.mod(index[:, 0] + dx, cells[0]), jnp.mod(index[:, 1] + dy, cells[1]),
                          jnp.mod(index[:, 2] + dz, cells[2]))
                w = ((weight[:, 0] if dx else 1.0 - weight[:, 0]) * (weight[:, 1] if dy else 1.0 - weight[:, 1])
                     * (weight[:, 2] if dz else 1.0 - weight[:, 2]))
                total = total + w * values[corner]
    return total


def velocity_at(grid: MACGrid, velocity: Velocity, points: Array) -> Array:
    """The velocity (m/s, shape ``(n, 3)``) at ``points`` (m from the box's corner, shape ``(n, 3)``)."""
    pts = jnp.asarray(points, dtype=velocity[0].dtype)
    return jnp.stack([_component_at(velocity[c], grid.cells, _OFFSETS[c], grid.spacing, pts) for c in range(3)],
                     axis=-1)


__all__ = ["velocity_at"]
