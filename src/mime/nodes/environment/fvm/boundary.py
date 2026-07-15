"""Boundary-condition specification helpers.

The FVM operators (``laplacian_orthogonal``, ``convection_upwind_blend``,
``divergence_face_flux``) accept boundary specifications as plain dicts
keyed by patch name. This module provides convenience builders that
keep call sites in the SIMPLE/PISO/IBM solvers tidy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import jax
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
    ``F_through = u_wall · Sf_outward`` are supplied.

    ``None`` means "this patch contributes nothing", **not** "extrapolate":

      * ``u_wall=None`` ⇒ zero-gradient for *diffusion* (the patch adds no
        Laplacian flux) — this one really is an extrapolation.
      * ``F_through=None`` ⇒ zero mass flux for *continuity*.
        :func:`~mime.nodes.environment.fvm.operators.divergence_face_flux`
        skips patches absent from ``boundary_F``, and a skipped patch is
        **impermeable** — an implicit wall, not an outlet. There is no
        outflow extrapolation anywhere in the solver.

    So an **outlet must be given an explicit** ``F_through``. Leaving it
    ``None`` on a patch that carries flow closes the domain; if flow is
    prescribed elsewhere, the continuity constraint then has no solution
    consistent with the intended field and the pressure projection will
    stall the flow to satisfy it. That failure looks like a boundary-layer
    or resolution artefact and is neither. (Diagnosed via
    ``tests/verification/test_fvm_freestream_bc.py``, whose own rig carried
    exactly this bug.)
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


def velocity_helmholtz_bc_source(
    mesh: FVMMesh,
    bcs: Dict[str, VelocityBC],
    spectral_bc: str | tuple,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reconcile the per-patch velocity BCs with the spectral Helmholtz basis.

    The diagonalised (DST/DCT/FFT) Helmholtz solvers in :mod:`pressure`
    invert ``(I − α ∇²_s)``, where ``∇²_s`` is a cell-centred Laplacian
    whose boundary behaviour is baked into the *basis* from a per-axis BC
    **type** string (``cfg.velocity_bc``). They can therefore represent
    neither

      * boundary **data** — a non-zero :attr:`VelocityBC.u_wall` is
        silently replaced by ``u = 0`` (``"dirichlet"`` means
        *homogeneous* Dirichlet at the physical face), nor
      * a per-patch BC **type** that disagrees with its axis' string
        (e.g. a zero-gradient outlet on an axis declared ``"dirichlet"``,
        which the basis pins to ``u = 0``).

    Both are repaired by adding the difference between the *physical*
    boundary flux and the one the basis already applies onto the RHS. Per
    boundary face, with ``gA = |Sf| / |d_b|`` and
    :func:`~mime.nodes.environment.fvm.operators.laplacian_orthogonal`'s
    Dirichlet flux ``gA (u_wall − u_P)``::

        physical flux  =  gA (u_wall − u_P)   [dirichlet]   or  0  [zero-gradient]
        basis flux     =  gA (0      − u_P)   [dirichlet axis] or 0  [neumann/periodic axis]

    so the RHS correction is ``(const + lag · u) / V`` with

        const_P = Σ_{bf ∈ P, u_wall given}  gA · u_wall
        lag_P   = Σ_{bf ∈ P, dirichlet axis} gA  −  Σ_{bf ∈ P, u_wall given} gA

    and the solve becomes::

        (I − α ∇²) u = b   ⟺   (I − α ∇²_s) u = b + α (const + lag · u) / V

    Returns ``(const, lag)`` of shapes ``[N_cells, dim]`` and
    ``[N_cells]`` (both already divided by ``V_P``).

    Exactness
    ---------
    For the dominant case — a Dirichlet patch on a Dirichlet axis — the
    two ``gA`` sums **cancel** (``lag = 0``) and ``const = gA · u_wall``
    is independent of the solution, so the correction is *exact*, not a
    lagged iteration.

    ``lag`` is non-zero only where a patch's type disagrees with its
    axis' basis. There the term is evaluated at the incoming iterate
    (deferred correction), which converges because the implicit operator's
    own boundary diagonal dominates it (the local amplification is
    ``2α/dx² / (1 + 3α/dx²) < 2/3`` for every ``α``). A basis that
    represents mixed Dirichlet/Neumann on one axis exactly would need a
    quarter-wave (DST-IV) transform; that is not implemented.

    A homogeneous no-slip wall (``u_wall = 0`` on a ``"dirichlet"`` axis)
    gives ``const = 0`` **and** ``lag = 0`` — a bit-exact no-op, which is
    what every existing wall-bounded caller passes.
    """
    dtype = mesh.V.dtype
    dim = mesh.dim
    if isinstance(spectral_bc, str):
        types = (spectral_bc,) * dim
    else:
        types = tuple(spectral_bc)
    if len(types) != dim:
        raise ValueError(
            f"velocity_bc must have length {dim}; got {types!r}"
        )

    const = jnp.zeros((mesh.N_cells, dim), dtype=dtype)
    lag = jnp.zeros((mesh.N_cells,), dtype=dtype)

    # Static Python floats — deliberately NOT jnp. This function is traced
    # (sometimes abstractly, e.g. under eval_shape), so anything derived from a
    # jnp array here cannot be read back out with float()/int().
    is_dir_flags = [1.0 if t == "dirichlet" else 0.0 for t in types]
    uniform = all(t == types[0] for t in types)

    for patch in mesh.patches:
        d_mag_b = jnp.linalg.norm(patch.d, axis=-1)           # [N_bf]
        gA_bf = (patch.area / d_mag_b).astype(dtype)          # [N_bf]
        # Does the basis treat this patch as homogeneous-Dirichlet? The BC type
        # is declared per *axis*, so we need the axis each patch faces — which
        # cannot be read out concretely here (mesh arrays are tracers). When
        # every axis agrees the answer is axis-independent; otherwise project
        # the per-axis flags onto the outward normal, which on a Cartesian mesh
        # (n = ±e_a — and the diagonalised solvers accept no other kind) picks
        # out exactly that axis' flag.
        if uniform:
            w_dir = is_dir_flags[0]                            # scalar
        else:
            w_dir = jnp.abs(patch.n.astype(dtype)) @ jnp.asarray(
                is_dir_flags, dtype=dtype,
            )                                                  # [N_bf]
        if not (uniform and w_dir == 0.0):
            lag = lag + jax.ops.segment_sum(
                gA_bf * w_dir, patch.owner, num_segments=mesh.N_cells,
            )
        bc = bcs.get(patch.name)
        if bc is None or bc.u_wall is None:
            continue
        u_w = jnp.broadcast_to(
            jnp.asarray(bc.u_wall, dtype=dtype),
            (int(patch.owner.size), dim),
        )
        const = const + jax.ops.segment_sum(
            gA_bf[:, None] * u_w, patch.owner, num_segments=mesh.N_cells,
        )
        lag = lag - jax.ops.segment_sum(
            gA_bf, patch.owner, num_segments=mesh.N_cells,
        )

    return const / mesh.V[:, None], lag / mesh.V


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
