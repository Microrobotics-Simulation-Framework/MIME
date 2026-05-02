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
