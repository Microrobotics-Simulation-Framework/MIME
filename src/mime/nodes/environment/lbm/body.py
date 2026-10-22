"""Lattice-body geometry hook for :class:`~mime.nodes.environment.lbm.fluid_node.IBLBMFluidNode`.

This is the LBM half of the shared fluid-node body abstraction (see
``src/mime/nodes/environment/FLUID_NODE_CONTRACT.md``). It mirrors the FVM
node's :class:`~mime.nodes.environment.fvm.ibm.IBMBody` /
``BodyFactory`` pair so the two solvers are wired the same way:

* FVM: ``BodyFactory = Callable[[pose_inputs], IBMBody]``; the node rebuilds
  the body inside ``update()`` from ``body_position`` / ``body_velocity`` /
  ``body_angular_velocity``.
* LBM: ``LatticeBodyFactory = Callable[[pose_inputs, LatticeGrid], LatticeBody]``;
  same idea, plus the lattice the occupancy mask must be rasterised onto.

**Why a parallel dataclass rather than reusing** :class:`IBMBody` **directly.**
The two solvers need genuinely different things from a body:

* FVM's diffuse-penalty IBM only ever evaluates ``sdf(x)`` at cell centres —
  an SDF *is* the complete body description.
* LBM bounce-back needs a **boolean occupancy mask** on the lattice, because
  the missing-link stencil is defined by which *neighbouring lattice node* is
  solid. An SDF can produce a mask (``phi < 0``), but the reverse is false, and
  the shipped UMR mask (``create_umr_mask``) is an analytic region test that is
  **not** the zero-set of ``umr_sdf`` — thresholding the SDF gives a different
  (and unvalidated) staircase. Forcing LBM through an SDF-only body would
  silently change the validated ``umr_confinement`` result.
* LBM additionally needs a per-lattice-node **wall velocity** field and a
  ``ref_point`` in *lattice* units, whereas FVM works in SI.

So ``LatticeBody`` carries ``mask`` (required) and ``sdf`` (optional, only
needed for Bouzidi IBB). What *is* shared is the layer that matters for
swappability: the pose **input names** (``body_position`` / ``body_velocity`` /
``body_angular_velocity`` / ``body_orientation``), the factory *shape*, and —
for the sphere — the actual SDF (:func:`mime.nodes.environment.fvm.sdf.sphere_sdf`
is imported here, not re-derived).

Units
-----
Everything in this module is in **lattice units**: positions and radii in
lattice cells, velocities in cells/step, angular velocities in rad/step. The
``IBLBMFluidNode`` edge transforms convert from SI (see
``fluid_node.make_iblbm_rigid_body_edges``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax.numpy as jnp

from mime.nodes.environment.fvm.sdf import sphere_sdf
from mime.nodes.robot.helix_geometry import (
    create_sphere_mask,
    create_umr_mask,
    umr_sdf,
)


# ---------------------------------------------------------------------------
# Grid + body descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatticeGrid:
    """The lattice a body must be rasterised onto.

    Attributes
    ----------
    shape : (nx, ny, nz)
        Extent of the (sub)lattice, in cells. Under multi-GPU sharding this
        is the *padded slab* extent, not the global lattice.
    origin : (ox, oy, oz)
        Global lattice coordinate of local cell ``(0, 0, 0)``. Non-zero only
        on the sharded path, where each slab builds its piece of the global
        mask. Static Python floats (never traced).
    """
    shape: tuple[int, int, int]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class LatticeBody:
    """A rigid body rasterised onto the LBM lattice.

    Attributes
    ----------
    name : str
        Identifier (e.g. ``"umr"``, ``"sphere"``).
    mask : (nx, ny, nz) bool
        Solid occupancy on the lattice. **Required** — bounce-back is defined
        by lattice-node occupancy.
    ref_point : (3,) float32
        Body reference point in *lattice* units (global coordinates). Used as
        the torque lever-arm origin and the rigid-body velocity pole.
    sdf : callable or None
        ``(N, 3) lattice points -> (N,)`` signed distance in lattice units,
        negative inside. Required only when the node runs Bouzidi IBB
        (``use_bouzidi=True``); ``None`` means the body supports simple
        bounce-back only.
    velocity : (3,) float32
        Body linear velocity in cells/step.
    angular_velocity : (3,) float32
        Body angular velocity in rad/step. A geometry whose mask is only
        parameterised about one axis (the UMR: a z-angle) **must** project
        the incoming angular velocity onto that axis here, so the imposed
        wall velocity stays consistent with the mask's actual motion.
    """
    name: str
    mask: jnp.ndarray
    ref_point: jnp.ndarray
    sdf: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    velocity: jnp.ndarray = field(
        default_factory=lambda: jnp.zeros(3, dtype=jnp.float32))
    angular_velocity: jnp.ndarray = field(
        default_factory=lambda: jnp.zeros(3, dtype=jnp.float32))


#: ``(pose, grid) -> LatticeBody``. ``pose`` is the dict assembled by
#: ``IBLBMFluidNode._pose()``: ``position`` / ``velocity`` /
#: ``angular_velocity`` (lattice units), ``orientation`` (quaternion, w-first)
#: and ``angle`` (the node's integrated z-rotation, rad). Every entry is a
#: traced JAX array — a factory must not branch on their values.
LatticeBodyFactory = Callable[[dict, LatticeGrid], LatticeBody]


# ---------------------------------------------------------------------------
# Rigid-body wall velocity on the lattice
# ---------------------------------------------------------------------------

def rigid_body_velocity_field(
    shape: tuple[int, int, int],
    ref_point: jnp.ndarray,
    velocity: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> jnp.ndarray:
    """``u(x) = v + omega x (x - ref_point)`` at every lattice node.

    The lattice-units analogue of
    :func:`mime.nodes.environment.fvm.sdf.rigid_body_velocity`, evaluated on
    the whole grid rather than at a point cloud.

    Like the ``_rotation_velocity_field`` it generalises, this does **not**
    zero the field outside the solid: :func:`apply_bounce_back` reads the wall
    velocity at *fluid* boundary nodes, so it must be defined everywhere.

    ``origin`` shifts the per-axis lattice coordinates so a sharded slab can
    build its piece of the global field (default ``(0, 0, 0)`` == whole grid).

    Returns
    -------
    wall_velocity : (nx, ny, nz, 3) float32
    """
    nx, ny, nz = shape
    ix = jnp.arange(nx, dtype=jnp.float32) + origin[0]
    iy = jnp.arange(ny, dtype=jnp.float32) + origin[1]
    iz = jnp.arange(nz, dtype=jnp.float32) + origin[2]
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing="ij")

    rx = gx - ref_point[0]
    ry = gy - ref_point[1]
    rz = gz - ref_point[2]

    w = angular_velocity
    # Written out (rather than jnp.cross) to keep the exact arithmetic of the
    # `_rotation_velocity_field` this replaces — the umr_confinement Bouzidi
    # regression is a 0.1%-tolerance check on a near-cancelling torque sum.
    ux = w[1] * rz - w[2] * ry
    uy = w[2] * rx - w[0] * rz
    uz = w[0] * ry - w[1] * rx

    return jnp.stack([ux, uy, uz], axis=-1) + velocity


# ---------------------------------------------------------------------------
# Shipped geometries
# ---------------------------------------------------------------------------

def make_umr_body_factory(
    name: str = "umr", **geom_lu,
) -> LatticeBodyFactory:
    """Factory for the de Boer rotating finned UMR (body + helical fins).

    This is the geometry ``IBLBMFluidNode`` was originally hard-wired to; it
    is now one implementation behind the geometry hook. Behaviour is
    unchanged: the mask is :func:`create_umr_mask` (``body | fins``) evaluated
    at the node's integrated z-rotation ``pose["angle"]``, and the Bouzidi SDF
    is :func:`umr_sdf`.

    ``geom_lu`` are ``create_umr_mask``'s geometry keywords **in lattice
    units** (``body_radius``, ``body_length``, ``fin_outer_radius``, ...);
    ``nx`` / ``ny`` / ``nz`` must not be passed — the grid supplies them.

    Rotation axis
    -------------
    ``create_umr_mask`` parameterises the body by a single **z-rotation
    angle**, so this body projects the incoming angular velocity onto z
    (``omega -> (0, 0, omega_z)``). Feeding the full vector would impose a
    wall velocity the mask's motion does not reproduce. A UMR tumbling off-axis
    needs a mask parameterised by the full orientation — not supported.
    """
    for k in ("nx", "ny", "nz", "center", "rotation_angle", "origin"):
        if k in geom_lu:
            raise ValueError(
                f"make_umr_body_factory: {k!r} is supplied per-step by the "
                "node (grid extent / pose); pass geometry keywords only."
            )
    sdf_kwargs = dict(geom_lu)

    def factory(pose: dict, grid: LatticeGrid) -> LatticeBody:
        nx, ny, nz = grid.shape
        pos = pose["position"]
        centre = (pos[0], pos[1], pos[2])
        angle = pose["angle"]

        mask = create_umr_mask(
            nx=nx, ny=ny, nz=nz,
            center=centre, rotation_angle=angle,
            origin=grid.origin, **geom_lu,
        )

        def sdf(pts):
            return umr_sdf(
                pts, center=centre, rotation_angle=angle, **sdf_kwargs,
            )

        omega = pose["angular_velocity"]
        omega_z = jnp.stack([
            jnp.zeros((), dtype=jnp.float32),
            jnp.zeros((), dtype=jnp.float32),
            omega[2],
        ])
        return LatticeBody(
            name=name, mask=mask, ref_point=pos, sdf=sdf,
            velocity=pose["velocity"], angular_velocity=omega_z,
        )

    return factory


def make_sphere_body_factory(
    name: str = "sphere", *, radius: float,
) -> LatticeBodyFactory:
    """Factory for a translating / rotating sphere of ``radius`` lattice cells.

    The LBM twin of :func:`mime.nodes.environment.fvm.fluid_node.make_sphere_body_factory`
    — same name, same role, same pose inputs; it returns a ``LatticeBody``
    instead of an ``IBMBody`` and works in lattice units.

    The occupancy mask (:func:`create_sphere_mask`, ``|x - c| < R``) and the
    Bouzidi SDF (:func:`mime.nodes.environment.fvm.sdf.sphere_sdf`,
    ``|x - c| - R``) are the *same* surface, so the two bounce-back paths agree
    on where the body is. Unlike the UMR the sphere is rotationally symmetric,
    so the **full** angular-velocity vector is passed through — a spinning
    sphere's mask is pose-independent, and only the imposed wall velocity
    carries the spin.
    """
    R = float(radius)

    def factory(pose: dict, grid: LatticeGrid) -> LatticeBody:
        nx, ny, nz = grid.shape
        pos = pose["position"]

        mask = create_sphere_mask(
            nx, ny, nz, center=(pos[0], pos[1], pos[2]), radius=R,
            origin=grid.origin,
        )

        def sdf(pts):
            return sphere_sdf(pts, center=pos, radius=R)

        return LatticeBody(
            name=name, mask=mask, ref_point=pos, sdf=sdf,
            velocity=pose["velocity"],
            angular_velocity=pose["angular_velocity"],
        )

    return factory
