"""Graph-native FVM operators (gather → compute → scatter).

Every operator here is fully vectorised over faces and reduces back to
cells via :func:`jax.ops.segment_sum`. This is the structural pattern
identified by the DiFVM paper (Du et al., arXiv:2603.15920) and is what
makes the entire solver fusible into a single XLA kernel.

Operators implemented
---------------------
- :func:`face_interp` — linear interpolation owner→face
- :func:`grad_green_gauss` — Green-Gauss cell gradient
- :func:`laplacian_orthogonal` — orthogonal Laplacian (diffusion flux)
- :func:`convection_upwind_blend` — upwind/linear-blended convection
- :func:`divergence_face_flux` — divergence of a face-mass-flux
- :func:`face_velocity_rhie_chow` — Rhie-Chow corrected face velocity
- :func:`momentum_diagonal` — assemble momentum-equation diagonal
  coefficient ``a_P`` (used by Rhie-Chow and pressure Poisson scaling)

Conventions
-----------
``Sf`` points from owner toward neighbour. Volumetric face flux
``F_f = u_f · Sf`` is positive when fluid leaves the owner cell.
Divergence is therefore ``segment_sum(F_f, owner) - segment_sum(F_f, neighbour)``
on a per-cell basis (signs flip for the neighbour).

References
----------
- Moukalled, Mangani & Darwish (2016), Ch. 8 (gradients), Ch. 11
  (convection schemes), Ch. 15 (Rhie-Chow on collocated meshes).
- Du et al. (2024) "DiFVM: A differentiable finite volume method",
  arXiv:2603.15920 — operator-as-message-passing pattern.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from mime.nodes.environment.fvm.mesh import FVMMesh, BoundaryPatch


# ---------------------------------------------------------------------------
# Linear face interpolation
# ---------------------------------------------------------------------------

def face_interp(phi: jnp.ndarray, mesh: FVMMesh) -> jnp.ndarray:
    """Linear interpolate a cell-centred scalar/vector to interior faces.

    ``phi_f = w * phi_owner + (1 - w) * phi_neighbour``. Works for any
    trailing array shape (scalar [N_cells], vector [N_cells, dim],
    tensor, ...).
    """
    phi_o = phi[mesh.owner]
    phi_n = phi[mesh.neighbour]
    # Broadcast w over trailing dims.
    w = mesh.w.reshape(mesh.w.shape + (1,) * (phi.ndim - 1))
    return w * phi_o + (1.0 - w) * phi_n


# ---------------------------------------------------------------------------
# Green-Gauss cell gradient (one message-pass)
# ---------------------------------------------------------------------------

def grad_green_gauss(
    phi: jnp.ndarray,
    mesh: FVMMesh,
    boundary_face_values: dict[str, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Cell-centred gradient via Green-Gauss reconstruction.

    ``∇φ_P = (1 / V_P) Σ_f (φ_f * Sf_f)``.

    Boundary faces contribute ``φ_f * Sf_outward``. ``boundary_face_values``
    maps patch name → face-valued ``[N_bf, ...]`` array. Patches missing
    from the dict use a zero-gradient (Neumann) extrapolation: ``φ_f =
    φ_owner``.

    Returns
    -------
    grad : ``[N_cells, ...trailing, dim]``
        Gradient of ``phi``. For scalar ``phi`` the trailing dim is
        absent and result is ``[N_cells, dim]``. For vector ``phi
        [N_cells, k]`` the result is ``[N_cells, k, dim]``.
    """
    # Interior contribution
    phi_f = face_interp(phi, mesh)            # [N_faces, ...]
    # Outer product φ_f ⊗ Sf
    contrib = phi_f[..., None] * mesh.Sf.reshape(
        (mesh.N_faces,) + (1,) * (phi.ndim - 1) + (mesh.dim,)
    )

    grad = jax.ops.segment_sum(contrib, mesh.owner, num_segments=mesh.N_cells)
    grad = grad - jax.ops.segment_sum(
        contrib, mesh.neighbour, num_segments=mesh.N_cells,
    )

    # Boundary contributions (each contributes outward Sf only, owner cell)
    bvals = boundary_face_values or {}
    for patch in mesh.patches:
        if patch.name in bvals:
            phi_bf = bvals[patch.name]
        else:
            # Zero-gradient extrapolation: φ_f = φ_owner
            phi_bf = phi[patch.owner]
        Sf_bf = patch.Sf.reshape(
            (patch.owner.size,) + (1,) * (phi.ndim - 1) + (mesh.dim,)
        )
        bcontrib = phi_bf[..., None] * Sf_bf
        grad = grad + jax.ops.segment_sum(
            bcontrib, patch.owner, num_segments=mesh.N_cells,
        )

    V = mesh.V.reshape((mesh.N_cells,) + (1,) * (phi.ndim))
    return grad / V


# ---------------------------------------------------------------------------
# Diffusion (orthogonal Laplacian)
# ---------------------------------------------------------------------------

def laplacian_orthogonal(
    phi: jnp.ndarray,
    mesh: FVMMesh,
    *,
    mu_face: jnp.ndarray | float = 1.0,
    boundary_specs: dict | None = None,
) -> jnp.ndarray:
    """Cell-centred Laplacian flux (∫ μ ∇φ · dS) via the orthogonal scheme.

    Interior face flux:
        flux_f = μ_f * (φ_N − φ_P) * |Sf| / |d|

    For a uniform Cartesian mesh ``Sf · d = |Sf| * |d|`` so this is
    exact (no non-orthogonal correction needed). Stretched / unstructured
    meshes can add a deferred-correction term later by overlaying a
    Green-Gauss gradient.

    ``boundary_specs`` maps patch name → dict with one of:
        * ``{"type": "dirichlet", "value": [N_bf, ...] }`` — flux uses
          ``μ * (φ_b − φ_P) * |Sf| / |d_b|`` (face-normal distance to
          face centroid).
        * ``{"type": "neumann", "flux": [N_bf, ...] }`` — directly add
          flux value (with units of φ * area).
        * ``{"type": "zero_gradient"}`` — no contribution (default).

    The default for any patch missing from ``boundary_specs`` is
    zero-gradient.

    Returns
    -------
    out : ``[N_cells, ...]`` — sum of fluxes per cell, *not* divided by
    volume. The caller decides whether to divide.
    """
    phi_o = phi[mesh.owner]
    phi_n = phi[mesh.neighbour]
    delta = phi_n - phi_o                       # [N_faces, ...]

    # μ_f * |Sf| / |d|
    if jnp.isscalar(mu_face) or getattr(mu_face, "ndim", 1) == 0:
        gA = (mu_face * mesh.area / mesh.d_mag)
    else:
        gA = mu_face * mesh.area / mesh.d_mag
    # Broadcast geometry coefficient over trailing dims.
    gA_b = gA.reshape((mesh.N_faces,) + (1,) * (phi.ndim - 1))
    flux_f = gA_b * delta                       # [N_faces, ...]

    out = jax.ops.segment_sum(flux_f, mesh.owner, num_segments=mesh.N_cells)
    out = out - jax.ops.segment_sum(
        flux_f, mesh.neighbour, num_segments=mesh.N_cells,
    )

    # Boundary contributions (no neighbour subtraction; outward sign)
    boundary_specs = boundary_specs or {}
    for patch in mesh.patches:
        spec = boundary_specs.get(patch.name, {"type": "zero_gradient"})
        ttype = spec["type"]
        if ttype == "zero_gradient":
            continue
        if ttype == "dirichlet":
            phi_b = spec["value"]               # [N_bf, ...]
            phi_P = phi[patch.owner]
            d_mag_b = jnp.linalg.norm(patch.d, axis=-1)
            mu = spec.get("mu", 1.0)
            gA_bf = mu * patch.area / d_mag_b
            gA_bf_b = gA_bf.reshape(
                (patch.owner.size,) + (1,) * (phi.ndim - 1)
            )
            bflux = gA_bf_b * (phi_b - phi_P)
            out = out + jax.ops.segment_sum(
                bflux, patch.owner, num_segments=mesh.N_cells,
            )
        elif ttype == "neumann":
            # Prescribed flux density (per unit area)
            qn = spec["flux"]                    # [N_bf, ...]
            mu = spec.get("mu", 1.0)
            area_b = patch.area.reshape(
                (patch.owner.size,) + (1,) * (phi.ndim - 1)
            )
            bflux = mu * qn * area_b
            out = out + jax.ops.segment_sum(
                bflux, patch.owner, num_segments=mesh.N_cells,
            )
        else:
            raise ValueError(f"unknown boundary type {ttype!r}")

    return out


# ---------------------------------------------------------------------------
# Convection
# ---------------------------------------------------------------------------

def convection_upwind_blend(
    phi: jnp.ndarray,
    F_face: jnp.ndarray,
    mesh: FVMMesh,
    *,
    gamma: float = 0.0,
    boundary_phi: dict[str, jnp.ndarray] | None = None,
    boundary_F: dict[str, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Convection flux ∫ φ (u · n) dS via blended upwind/central scheme.

    ``F_face`` is the face mass flux (volumetric flux × density, but in
    incompressible flow we conventionally use the volumetric flux ``u_f
    · Sf`` and absorb density into ``φ`` for momentum). Sign convention:
    ``F_face > 0`` means flow from owner to neighbour.

    The convected face value is

        φ_f = γ * φ_central  +  (1 − γ) * φ_upwind

    with ``γ ∈ [0, 1]``. ``γ = 0`` is pure upwind (stable, diffusive),
    ``γ = 1`` is pure linear central (accurate, oscillatory at high Pe).
    For initial milestones use γ = 0; raise to 0.5+ once stable.

    Boundary face contribution:
        flux_b = F_b * φ_b
    where ``F_b = u_b · Sf_outward`` is supplied per patch in
    ``boundary_F`` (defaulting to ``0`` — no through-flow). ``φ_b`` is
    supplied in ``boundary_phi`` (defaulting to upwind extrapolation:
    ``φ_owner`` when outflow, value-required when inflow — caller must
    pass it).

    Returns
    -------
    out : ``[N_cells, ...]`` — convection flux summed per cell, not
    divided by volume.
    """
    phi_o = phi[mesh.owner]
    phi_n = phi[mesh.neighbour]
    phi_central = 0.5 * (phi_o + phi_n)         # uniform mesh; could use w
    # Upwind: pick owner if F>=0, else neighbour
    F = F_face
    F_b = F.reshape((mesh.N_faces,) + (1,) * (phi.ndim - 1))
    phi_upwind = jnp.where(F_b >= 0, phi_o, phi_n)
    phi_f = gamma * phi_central + (1.0 - gamma) * phi_upwind
    flux_f = F_b * phi_f                        # [N_faces, ...]

    out = jax.ops.segment_sum(flux_f, mesh.owner, num_segments=mesh.N_cells)
    out = out - jax.ops.segment_sum(
        flux_f, mesh.neighbour, num_segments=mesh.N_cells,
    )

    # Boundaries
    bphi = boundary_phi or {}
    bF = boundary_F or {}
    for patch in mesh.patches:
        # Default no-through-flow (wall): F_b = 0 ⇒ no contribution
        if patch.name not in bF and patch.name not in bphi:
            continue
        F_bf = bF.get(patch.name, jnp.zeros((patch.owner.size,)))
        # Upwind for outflow (F_b > 0): use owner-cell phi.
        # For inflow (F_b < 0) the user must supply φ_b (Dirichlet).
        if patch.name in bphi:
            phi_b = bphi[patch.name]
        else:
            phi_b = phi[patch.owner]
        F_bf_b = F_bf.reshape(
            (patch.owner.size,) + (1,) * (phi.ndim - 1)
        )
        bflux = F_bf_b * phi_b
        out = out + jax.ops.segment_sum(
            bflux, patch.owner, num_segments=mesh.N_cells,
        )

    return out


# ---------------------------------------------------------------------------
# Divergence of a face mass flux
# ---------------------------------------------------------------------------

def divergence_face_flux(
    F_face: jnp.ndarray,
    mesh: FVMMesh,
    *,
    boundary_F: dict[str, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Compute ∫ ∇·u dV per cell from interior + boundary face fluxes.

    Sign convention: ``F = u_f · Sf`` (Sf outward from owner). Therefore

        (∫∇·u dV)_P = Σ_{f∈∂P} F_f * sign(P)

    with sign +1 if P is owner of f, −1 if neighbour.
    """
    out = jax.ops.segment_sum(F_face, mesh.owner, num_segments=mesh.N_cells)
    out = out - jax.ops.segment_sum(
        F_face, mesh.neighbour, num_segments=mesh.N_cells,
    )
    bF = boundary_F or {}
    for patch in mesh.patches:
        F_bf = bF.get(patch.name)
        if F_bf is None:
            continue
        out = out + jax.ops.segment_sum(
            F_bf, patch.owner, num_segments=mesh.N_cells,
        )
    return out


# ---------------------------------------------------------------------------
# Rhie-Chow face velocity (collocated grid)
# ---------------------------------------------------------------------------

def face_velocity_rhie_chow(
    u_cell: jnp.ndarray,                  # [N_cells, dim]
    p_cell: jnp.ndarray,                  # [N_cells]
    grad_p_cell: jnp.ndarray,             # [N_cells, dim] — same source as in momentum
    a_p_cell: jnp.ndarray,                # [N_cells] — momentum diagonal coefficient
    mesh: FVMMesh,
) -> jnp.ndarray:
    """Rhie-Chow corrected interior face velocity (m/s vector).

    On a collocated mesh, simply averaging cell-centred velocity to faces
    decouples pressure from velocity (checkerboard mode). Rhie-Chow
    introduces a face-level pressure gradient sense:

        u_f = avg(u_P, u_N) − D_f * [(p_N − p_P)/|d| − avg(∇p)_f · n̂]
            * |Sf|

    rearranged so that the volumetric flux

        F_f = u_f · Sf
            = avg(u) · Sf − D_f' * [(p_N − p_P) − avg(∇p)_f · d_PN]

    where ``D_f' = avg(V/a_p) * |Sf|/|d|``. This is the practical form
    used in OpenFOAM and Moukalled §15.6.

    Returns the face *velocity vector* (3 components or 2). The caller
    forms ``F_f = u_f · Sf`` to get the flux. We return the vector form
    because it is what is needed by the momentum corrector step.
    """
    u_o = u_cell[mesh.owner]
    u_n = u_cell[mesh.neighbour]
    u_avg = 0.5 * (u_o + u_n)                       # [N_faces, dim]

    p_o = p_cell[mesh.owner]
    p_n = p_cell[mesh.neighbour]
    grad_p_avg = 0.5 * (grad_p_cell[mesh.owner] + grad_p_cell[mesh.neighbour])

    # Use precomputed V_owner / V_neighbour to avoid XLA constant-folding
    # the static-by-static gather (multi-second compile cost at large N).
    V_o = mesh.V_owner if mesh.V_owner is not None else mesh.V[mesh.owner]
    V_n = mesh.V_neighbour if mesh.V_neighbour is not None else mesh.V[mesh.neighbour]
    aP_o = a_p_cell[mesh.owner]
    aP_n = a_p_cell[mesh.neighbour]
    # Avoid division by zero when a_P is small (e.g. far from convergence)
    safe = lambda a: jnp.where(jnp.abs(a) < 1e-30, 1e-30, a)
    D_o = V_o / safe(aP_o)
    D_n = V_n / safe(aP_n)
    D_face = 0.5 * (D_o + D_n)                      # [N_faces]

    # Pressure-gradient correction term, projected along d̂ direction.
    # "True" face gradient: (p_N - p_P) / |d|
    # "Interpolated" face gradient · d̂: (avg_grad_p · d) / |d|
    n_hat = mesh.d / mesh.d_mag[:, None]            # unit owner→neighbour
    Δp = (p_n - p_o)                                # [N_faces]
    grad_p_along = jnp.einsum("fd,fd->f", grad_p_avg, n_hat)  # [N_faces]
    correction_scalar = D_face * (
        Δp / mesh.d_mag - grad_p_along
    )                                               # [N_faces]

    # u_f = u_avg - correction_scalar * n_hat
    u_face = u_avg - correction_scalar[:, None] * n_hat
    return u_face


def momentum_diagonal_uniform_cartesian(
    mesh: FVMMesh,
    *,
    nu: float,
    rho: float,
    F_face: jnp.ndarray,
    dt: float | None = None,
) -> jnp.ndarray:
    """Approximate momentum-equation diagonal a_P for uniform Cartesian.

    Used by Rhie-Chow and as a scaling coefficient. For uniform spacing
    the diagonal is dominated by:

        a_P = ρ V / dt  +  Σ_f max(F_f, 0) sign  +  μ Σ_f |Sf|/|d|

    with the sign chosen so the diagonal is positive (upwind discretisation
    is positive-coefficient by construction). For ``dt is None`` (steady
    SIMPLE) the transient term is dropped.

    This is a *lumped* approximation valid on uniform meshes. For
    non-uniform / unstructured meshes the full per-cell assembly should
    be used; this function is a fast path for M0–M2.
    """
    mu = rho * nu
    # Diffusion contribution: μ |Sf| / |d| collected per cell from both ends
    diff_per_face = mu * mesh.area / mesh.d_mag    # [N_faces]
    diff_owner = jax.ops.segment_sum(
        diff_per_face, mesh.owner, num_segments=mesh.N_cells,
    )
    diff_neigh = jax.ops.segment_sum(
        diff_per_face, mesh.neighbour, num_segments=mesh.N_cells,
    )
    a_p = diff_owner + diff_neigh

    # Boundary diffusion contributions
    for patch in mesh.patches:
        d_mag_b = jnp.linalg.norm(patch.d, axis=-1)
        a_p = a_p + jax.ops.segment_sum(
            mu * patch.area / d_mag_b, patch.owner, num_segments=mesh.N_cells,
        )

    # Convection contribution (pure upwind diagonal)
    F_pos = jnp.maximum(F_face, 0.0)
    F_neg = jnp.maximum(-F_face, 0.0)
    a_p = a_p + jax.ops.segment_sum(
        F_pos, mesh.owner, num_segments=mesh.N_cells,
    )
    a_p = a_p + jax.ops.segment_sum(
        F_neg, mesh.neighbour, num_segments=mesh.N_cells,
    )

    if dt is not None:
        a_p = a_p + rho * mesh.V / dt

    return a_p
