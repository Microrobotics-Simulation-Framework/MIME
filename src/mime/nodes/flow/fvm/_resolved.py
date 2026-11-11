"""Resolved bodies in the FVM's step (F4, ``Drag.Penalty``; the record's B8, body half point 5, and sitting 4's B12).

**The body on the grid.** A resolved body is a solid the field's own band lifts onto its faces, as the walls are: its
indicator ``chi`` rises from 0 to 1 across the band ``W`` about its surface and its stiff interior beyond, both from its
signed distance, so the body is made by the same indicator as the walls (B3). Its distance is its solid region's
registered shape's (:data:`RESOLVED_SHAPES`, exact everywhere: the sphere's and the capsule's), placed by the body's
pose each step. Only a window of faces about the body is evaluated (:func:`body_faces`): a block of fixed size, its
corner at the cell the body's centre lies in (held out of derivatives, as the exchange kernel's stencil is), its values
smooth in the pose through the band. The solid is the union of the wall's and the bodies': the indicators combine as
``1 - (1 - chi_w)(1 - chi_b)`` and the regions' resistivities add (B5.3).

**The penalty.** Inside the band the fluid is driven to the body's rigid velocity ``u_b = V + Omega x r`` by the
resistivity ``nu / K``, implicit in the field's solve: its operator takes ``nu/K`` and its right-hand side ``nu/K u_b``.

**The body's load** (B8, point 5: the field formulation). The fluid's force on the body is the penalty's,
``rho h^3 sum nu/K (u - u_b)``, plus the interior fluid's momentum change taken from the field, ``d/dt rho h^3 sum chi
u`` (and their moments about the body's centre), so the interior fluid, which the grid carries, is counted once. The
rigid-interior shortcut (the interior taken to move with the body) is out: undefined at a density ratio of one,
diverging at 1.2, ringing at 1.4 (the coupling spike's R3). The interior's momentum is held from the step that made it,
so its change telescopes over a run."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
from jax import Array

from mime.core import jnp, shape_named
from mime.core._quaternion import rotation_matrix
from mime.nodes.flow.fvm._band import Band, ramp
from mime.nodes.flow.fvm._grid import MACGrid

#: The solid regions' shapes a resolved body is lifted from: those registered with a signed distance exact everywhere,
#: so the band's contract holds (exact within the band: F2's rows 1, item 1). A kind whose solid is several regions
#: (a cylinder, a box, the screw) or none a registered shape evaluates (the helix, the ellipsoid) is not lifted yet.
RESOLVED_SHAPES = ("sphere", "capsule")
#: The window's margin beyond the body's reach, in cells: the band's outer half, a cell for the window's corner at the
#: body's cell, and one for the Laplacian's reach.
WINDOW_MARGIN = 3


def window_cells(enclosing_radius: float, spacing: float, band: Band) -> int:
    """The window's side in cells for a body of ``enclosing_radius`` (m) on a grid of ``spacing``: every face whose
    distance from the body is under the band's outer half (where its indicator is above zero) lies inside it."""
    reach = enclosing_radius / spacing + 0.5 * band.width + WINDOW_MARGIN
    return 2 * math.ceil(reach) + 1


@dataclass(frozen=True)
class ResolvedBody:
    """A resolved body as the step reads it: its solid region's shape and parameters (in its part's frame), where the
    part sits in the body's frame (its centroid, and the rotation from its frame to the body's, as rows), the window's
    side in cells, and its hull's volume (m^3)."""

    shape: str
    parameters: Mapping[str, Any]
    centre: Any
    rotation: Any
    window: int
    volume: Any

    def solid_distance(self, body_points: Array) -> Array:
        """The signed distance of each of ``body_points`` (``(..., 3)``, the body's frame) from the body's surface,
        positive in the solid (the wall's convention): minus the shape's."""
        local = (body_points - jnp.asarray(self.centre, dtype=body_points.dtype)) @ jnp.asarray(self.rotation,
                                                                                              dtype=body_points.dtype)
        shape = shape_named(self.shape)
        flat = local.reshape(-1, 3)
        def one(point: Array) -> Array:
            return shape.signed_distance(self.parameters, point)

        distance = jax.vmap(one)(flat)
        return -distance.reshape(local.shape[:-1])


class BodyFaces(NamedTuple):
    """One face set's window about a body: each axis's wrapped lattice indices ``(w,)`` (``lattice``), each face's offset from the
    body's centre ``(w, w, w, 3)`` (m), the body's indicator and its resistivity ``1/K`` (m^-2) there."""

    lattice: tuple[Array, Array, Array]
    offset: Array
    chi: Array
    inverse_permeability: Array


def body_faces(grid: MACGrid, component: int, body: ResolvedBody, position: Array, turn: Array, band: Band,
               interior: float) -> BodyFaces:
    """The body's window on the ``component``-th face set, the body's centre at ``position`` (m, the grid's frame) and
    its frame turned by ``turn`` (body to grid, ``3 x 3``): the window's corner the cell the centre lies in less half
    the window, held out of derivatives; the indicator and resistivity the wall's lift of the body's signed distance."""
    h = grid.spacing
    w = body.window
    dtype = position.dtype
    steps = jnp.arange(w) - (w // 2)
    index: list[Array] = []
    coordinates: list[Array] = []
    for axis, count in enumerate(grid.cells):
        shift = 0.0 if axis == component else 0.5
        base = jnp.floor(jax.lax.stop_gradient(position[axis] / h - shift)).astype(jnp.int32)
        lattice = base + steps
        coordinates.append((lattice.astype(dtype) + shift) * h - position[axis])
        index.append(jnp.mod(lattice, count))
    offset = jnp.stack(jnp.meshgrid(*coordinates, indexing="ij"), axis=-1)
    phi = body.solid_distance(offset @ turn)
    width = band.width * h
    stiffening = (band.depth / interior) ** 2
    chi = ramp(phi, width)
    inverse = (ramp(phi, width) + interior_part(phi, band, h) * stiffening) / (band.depth * h) ** 2
    return BodyFaces((index[0], index[1], index[2]), offset, chi, inverse)


def interior_part(phi: Array, band: Band, h: Any) -> Array:
    """The stiff interior's indicator, the wall's own (``mime.nodes.flow.fvm._band.wall_fields``): a second ramp a band
    wide from the band's outer edge."""
    return ramp(phi - band.width * h, max(band.width, 1.0) * h)


def placed(field: Array, faces: BodyFaces) -> Array:
    """``field``'s values on the window (``(w, w, w)``)."""
    ix, iy, iz = faces.lattice
    return field[ix[:, None, None], iy[None, :, None], iz[None, None, :]]


def scattered(field: Array, faces: BodyFaces, values: Array) -> Array:
    """``field`` with ``values`` added on the window."""
    ix, iy, iz = faces.lattice
    return field.at[ix[:, None, None], iy[None, :, None], iz[None, None, :]].add(values)


def united(field: Array, faces: BodyFaces) -> Array:
    """The wall's indicator ``field`` united with the body's on the window: ``1 - (1 - chi_w)(1 - chi_b)``."""
    ix, iy, iz = faces.lattice
    at = (ix[:, None, None], iy[None, :, None], iz[None, None, :])
    return field.at[at].set(1.0 - (1.0 - field[at]) * (1.0 - faces.chi))


def rigid_velocity(faces: BodyFaces, component: int, twist: Array) -> Array:
    """The body's rigid velocity's ``component`` on its window, ``V + Omega x r`` (m/s)."""
    r = faces.offset
    a, b = (component + 1) % 3, (component + 2) % 3
    return twist[component] + twist[3 + a] * r[..., b] - twist[3 + b] * r[..., a]


def moments(faces: BodyFaces, component: int, density: Array, volume: Any) -> tuple[Array, Array]:
    """A density on the ``component``-th face set's window integrated: its sum (the ``component``'s) and its moment
    about the body's centre, ``sum r x (density e_c)``, as a 3-vector."""
    r = faces.offset
    total = volume * jnp.sum(density)
    a, b = (component + 1) % 3, (component + 2) % 3
    moment = jnp.zeros(3, dtype=density.dtype)
    moment = moment.at[a].add(volume * jnp.sum(r[..., b] * density))
    moment = moment.at[b].add(-volume * jnp.sum(r[..., a] * density))
    return total, moment


def pose_turn(pose: Array) -> tuple[Array, Array]:
    """A pose's position (m) and its rotation body to world (``3 x 3``)."""
    return pose[:3], rotation_matrix(pose[3:7])


__all__ = ["RESOLVED_SHAPES", "WINDOW_MARGIN", "BodyFaces", "ResolvedBody", "body_faces", "interior_part", "moments", "placed",
           "pose_turn", "rigid_velocity", "scattered", "united", "window_cells"]
