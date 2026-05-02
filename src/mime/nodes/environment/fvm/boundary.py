"""Boundary-condition specification helpers.

The FVM operators (``laplacian_orthogonal``, ``convection_upwind_blend``,
``divergence_face_flux``) accept boundary specifications as plain dicts
keyed by patch name. This module provides convenience builders that
keep call sites in the SIMPLE/PISO/IBM solvers tidy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import jax.numpy as jnp

from mime.nodes.environment.fvm.mesh import FVMMesh, BoundaryPatch


@dataclass(frozen=True)
class VelocityBC:
    """Velocity boundary condition for one patch.

    Combines:
      * a *kinematic* prescription for diffusion (Dirichlet wall
        velocity) and Rhie-Chow consistency,
      * a *flux* prescription for convection (mass through the face).

    For an impermeable wall both ``u_wall`` (a vector) and ``F_through``
    (zero) are supplied. For an inlet, both ``u_wall`` and a non-zero
    ``F_through = u_wall · Sf_outward`` are supplied. For an outlet,
    typically a zero-gradient extrapolation is used (caller passes
    ``u_wall=None`` so default zero-gradient applies).
    """

    u_wall: Optional[jnp.ndarray] = None  # [N_bf, dim] or None
    F_through: Optional[jnp.ndarray] = None  # [N_bf]   or None


def velocity_diffusion_specs(
    mesh: FVMMesh,
    bcs: Dict[str, VelocityBC],
    *,
    mu: float,
) -> dict:
    """Build a `boundary_specs` dict for laplacian_orthogonal of velocity.

    Boundary values are cast to ``mesh.V.dtype`` to keep the entire
    fori_loop in a single dtype (otherwise an x64-enabled test session
    can promote float32 state to float64 and break the carry).
    """
    dt = mesh.V.dtype
    specs = {}
    for patch in mesh.patches:
        bc = bcs.get(patch.name)
        if bc is None or bc.u_wall is None:
            specs[patch.name] = {"type": "zero_gradient"}
        else:
            specs[patch.name] = {
                "type": "dirichlet",
                "value": bc.u_wall.astype(dt),
                "mu": mu,
            }
    return specs


def velocity_convection_boundaries(
    mesh: FVMMesh,
    bcs: Dict[str, VelocityBC],
):
    """Build (boundary_F, boundary_phi) dicts for convection_upwind_blend."""
    dt = mesh.V.dtype
    bF = {}
    bphi = {}
    for patch in mesh.patches:
        bc = bcs.get(patch.name)
        if bc is None:
            continue
        if bc.F_through is not None:
            bF[patch.name] = bc.F_through.astype(dt)
        if bc.u_wall is not None:
            bphi[patch.name] = bc.u_wall.astype(dt)
    return bF, bphi


# ---------------------------------------------------------------------------
# Inlet/outlet helpers for pipe geometry
# ---------------------------------------------------------------------------

def poiseuille_inlet_velocity(
    mesh: FVMMesh, patch_name: str, *, R_pipe: float,
    U_mean: float, axis: int = 2,
) -> jnp.ndarray:
    """Cell-face velocity vectors for a Poiseuille inlet patch.

    Returns an ``[N_bf, dim]`` array suitable for ``VelocityBC.u_wall``.
    Velocity in the +``axis`` direction is ``2 U_mean (1 − r²/R²)`` for
    cells where ``r ≤ R_pipe`` and 0 outside (so the IBM cylinder wall
    naturally damps any leakage).
    """
    p = mesh.patch(patch_name)
    fx = p.face_x      # [N_bf, dim]
    cross_axes = [a for a in range(mesh.dim) if a != axis]
    rho = jnp.sqrt(sum(fx[:, a] ** 2 for a in cross_axes))
    u_z = jnp.where(rho < R_pipe, 2.0 * U_mean * (1.0 - (rho / R_pipe) ** 2), 0.0)
    u = jnp.zeros((p.owner.size, mesh.dim), dtype=mesh.V.dtype)
    u = u.at[:, axis].set(u_z.astype(mesh.V.dtype))
    return u


def poiseuille_inlet_F_through(
    mesh: FVMMesh, patch_name: str, *, R_pipe: float,
    U_mean: float, axis: int = 2,
) -> jnp.ndarray:
    """Mass-flux ``u·Sf_outward`` for a Poiseuille inlet patch.

    Sign convention: F = u · Sf where Sf is the OUTWARD-from-domain face
    normal. For an inlet at z_min the outward normal is −z so F is
    *negative* (mass flowing IN through this face).
    """
    p = mesh.patch(patch_name)
    fx = p.face_x
    cross_axes = [a for a in range(mesh.dim) if a != axis]
    rho = jnp.sqrt(sum(fx[:, a] ** 2 for a in cross_axes))
    u_z = jnp.where(rho < R_pipe, 2.0 * U_mean * (1.0 - (rho / R_pipe) ** 2), 0.0)
    # F = u · Sf with Sf already containing the outward normal
    # For an inlet patch with normal = -axis, Sf[:, axis] is negative,
    # so F = u_z * Sf[:, axis] (negative for inflow at z_min, positive
    # for outflow at z_max — both consistent with our patch convention).
    return (u_z * p.Sf[:, axis]).astype(mesh.V.dtype)
