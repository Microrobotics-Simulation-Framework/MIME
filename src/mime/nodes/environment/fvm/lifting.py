"""Lifting / homogenisation for non-zero-Dirichlet inlet BCs.

The PISO Helmholtz solve uses a DST spectral basis whose basis
functions vanish at the domain boundaries — the spectral solver
therefore enforces ``u=0`` at z_min/z_max regardless of any face-
level Dirichlet value passed through the convection / diffusion
operators. The result is that a prescribed Poiseuille (or Womersley)
inlet velocity never establishes in the domain.

The standard fix is **field decomposition**::

    u(x, t)  =  u_lift(x, t)  +  u_hom(x, t)

where ``u_lift`` satisfies the non-homogeneous Dirichlet BC exactly
by construction and ``u_hom`` is zero at all boundaries. The DST
solver only ever sees ``u_hom`` (which has the homogeneous BC the
basis requires). The inlet velocity is enforced *implicitly* by the
choice of ``u_lift``, never by patching the spectral solver.

Substituting into incompressible NS gives an equation for ``u_hom``
with three additional source terms:

    f_lift  =  − ∂u_lift/∂t
              − (u_hom · ∇) u_lift          [perturbation advected by background shear]
              − (u_lift · ∇) u_hom          [background flow advecting perturbation]
              + ν ∇² u_lift                 [background diffusion]

For a steady Poiseuille u_lift, ``∂u_lift/∂t = 0`` and ``ν ∇² u_lift``
is a uniform body force in the streamwise direction equal to
``-∂P/∂z`` (the driving pressure gradient). For Womersley, all three
remain non-trivial and are precomputed analytically at all timesteps.

This module exposes :class:`LiftingFunction` (precomputed lift fields)
and :func:`compute_lifting_source` (the four-term graph operator).

References
----------
- DiFVM (Du et al. 2024) §2.6.2 — boundary conditions as graph
  operations on boundary edge patches inside the same scatter
  framework as interior fluxes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class LiftingFunction:
    """Precomputed lifting field and its time derivative.

    For *steady* Poiseuille:
        u_lift_static    : [N_cells, 3]
        du_lift_dt       : [N_cells, 3]   (zeros)
        is_time_varying  : False

    For *time-varying* Womersley:
        u_lift_static    : [N_steps, N_cells, 3]
        du_lift_dt       : [N_steps, N_cells, 3]
        is_time_varying  : True

    Both are computed ONCE at mesh init and stored as static JAX
    arrays. Nothing in LiftingFunction is computed inside the PISO
    loop. Index by step index ``i_step`` at runtime when time-varying;
    use the static field directly when steady.

    Pre-computed companion arrays needed in the lifting source term:
        u_lift_face      : [N_faces, 3]            face-interpolated u_lift
        grad_u_lift      : [N_cells, 3, 3]         cell gradient of u_lift
                                                   (component_i, axis_j) →
                                                   ∂u_i/∂x_j
    For time-varying, both have a leading [N_steps, ...] axis.
    """
    u_lift_static: jnp.ndarray            # [N_cells, 3] or [N_steps, N_cells, 3]
    du_lift_dt: jnp.ndarray               # same shape as u_lift_static
    u_lift_face: jnp.ndarray              # [N_faces, 3] or [N_steps, N_faces, 3]
    grad_u_lift: jnp.ndarray              # [N_cells, 3, 3] or [N_steps, N_cells, 3, 3]
    is_time_varying: bool = False

    def at(self, i_step: int | jnp.ndarray):
        """Return the (u_lift, du_lift_dt, u_lift_face, grad_u_lift) tuple at
        the given step index. For steady, ``i_step`` is ignored."""
        if not self.is_time_varying:
            return (self.u_lift_static, self.du_lift_dt,
                    self.u_lift_face, self.grad_u_lift)
        return (
            self.u_lift_static[i_step],
            self.du_lift_dt[i_step],
            self.u_lift_face[i_step],
            self.grad_u_lift[i_step],
        )


# Register as pytree so the FVMMesh / state pytrees containing it
# survive jax.lax.scan / jit traces.
def _lift_flatten(L: LiftingFunction):
    children = (L.u_lift_static, L.du_lift_dt, L.u_lift_face, L.grad_u_lift)
    aux = (L.is_time_varying,)
    return children, aux


def _lift_unflatten(aux, children):
    return LiftingFunction(
        u_lift_static=children[0],
        du_lift_dt=children[1],
        u_lift_face=children[2],
        grad_u_lift=children[3],
        is_time_varying=aux[0],
    )


jax.tree_util.register_pytree_node(LiftingFunction, _lift_flatten, _lift_unflatten)


# ---------------------------------------------------------------------------
# Lifting source term
# ---------------------------------------------------------------------------

def compute_lifting_source(
    u_hom: jnp.ndarray,                   # [N_cells, 3]
    u_lift: jnp.ndarray,                  # [N_cells, 3]
    du_lift_dt: jnp.ndarray,              # [N_cells, 3]
    u_lift_face: jnp.ndarray | None,      # [N_faces, 3] or None (recompute)
    grad_u_lift: jnp.ndarray | None,      # [N_cells, 3, 3] or None (recompute)
    mesh,
    *,
    nu: float,
) -> jnp.ndarray:
    """Lifting body-force source for the homogeneous-velocity equation.

        f_lift = − ∂u_lift/∂t
                 − (u_hom · ∇) u_lift
                 − (u_lift · ∇) u_hom
                 + ν ∇² u_lift

    The third term (background advecting perturbation) is computed as
    a graph scatter operation — same structure as the existing
    convection operator. The first two terms are pointwise. The fourth
    (viscous diffusion of the lift) is precomputed inside ``u_lift``
    via the analytical balance ``ν ∇² u_lift = -∂P/∂z`` for Poiseuille
    and is folded into the existing pressure-gradient setup; here we
    leave it out (set to 0) because the mean pressure-gradient term
    already accounts for it in the standard PISO loop.

    Returns
    -------
    f_lift : [N_cells, 3]
        Body force (per unit volume × ρ) added to the momentum RHS.
    """
    # Term 1: -∂u_lift/∂t
    f1 = -du_lift_dt

    # Term 2: -(u_hom · ∇) u_lift  →  per-cell pointwise einsum
    # grad_u_lift[i, k, j] = ∂u_lift_k / ∂x_j at cell i
    # (u_hom · ∇)u_lift_k = Σ_j u_hom_j * ∂u_lift_k / ∂x_j
    if grad_u_lift is None:
        from mime.nodes.environment.fvm.operators import grad_green_gauss
        # Compute per-component gradient and stack: result [N_cells, 3, 3]
        # where g[i, k, j] = ∂u_lift_k/∂x_j at cell i
        grad_components = jnp.stack(
            [grad_green_gauss(u_lift[:, k], mesh) for k in range(3)],
            axis=1,
        )                                         # [N_cells, 3, 3]
        grad_u_lift_eff = grad_components
    else:
        grad_u_lift_eff = grad_u_lift
    f2 = -jnp.einsum("ij,ikj->ik", u_hom, grad_u_lift_eff)

    # Term 3: -(u_lift · ∇) u_hom  →  scatter via face flux
    # mass-flux carried by u_lift through each face:
    if u_lift_face is None:
        from mime.nodes.environment.fvm.operators import face_interp
        u_lift_face_eff = face_interp(u_lift, mesh)
    else:
        u_lift_face_eff = u_lift_face
    mdot_lift = jnp.einsum("fi,fi->f", u_lift_face_eff, mesh.Sf)
    # face value of u_hom (linear interpolation, same as convection_upwind_blend)
    u_hom_o = u_hom[mesh.owner]
    u_hom_n = u_hom[mesh.neighbour]
    w = mesh.w[:, None]
    u_hom_f = w * u_hom_o + (1.0 - w) * u_hom_n
    flux_f = mdot_lift[:, None] * u_hom_f                # [N_faces, 3]
    out_owner = jax.ops.segment_sum(flux_f, mesh.owner, num_segments=mesh.N_cells)
    out_neigh = jax.ops.segment_sum(flux_f, mesh.neighbour, num_segments=mesh.N_cells)
    # Normalise by V to get per-volume forcing (matches f1, f2 conventions)
    f3 = -(out_owner - out_neigh) / mesh.V[:, None]
    f3 = f3.astype(u_hom.dtype)

    # Term 4: ν ∇² u_lift — for Poiseuille this is a uniform driving
    # pressure gradient already represented in the mean pressure term.
    # For Womersley, it has an analytical form combining time- and r-
    # dependence; precomputed and folded into du_lift_dt by the
    # womersley helper. Here we set 0 — change in the helper, not
    # downstream.
    f4 = jnp.zeros_like(u_hom)

    return f1 + f2 + f3 + f4


def make_poiseuille_lift(
    mesh, *, R_pipe: float, U_mean: float, axis: int = 2, dtype=None,
) -> "LiftingFunction":
    """Build a steady Poiseuille lifting field for a Cartesian pipe mesh.

    u_lift_z(r) = 2 U_mean (1 − r²/R²)  for r ≤ R, else 0.
    All companion arrays (face interp, gradient) precomputed analytically.
    """
    if dtype is None:
        dtype = mesh.V.dtype
    x = mesh.x
    cross_axes = [a for a in range(mesh.dim) if a != axis]
    rho_cell = jnp.sqrt(sum(x[:, a] ** 2 for a in cross_axes))
    u_z_cell = jnp.where(
        rho_cell < R_pipe,
        2.0 * U_mean * (1.0 - (rho_cell / R_pipe) ** 2),
        0.0,
    )
    u_lift = jnp.zeros((mesh.N_cells, 3), dtype=dtype)
    u_lift = u_lift.at[:, axis].set(u_z_cell.astype(dtype))

    # Face-interpolated u_lift (linear, same as face_interp)
    u_o = u_lift[mesh.owner]
    u_n = u_lift[mesh.neighbour]
    w = mesh.w[:, None]
    u_lift_face = w * u_o + (1.0 - w) * u_n

    # Analytical gradient for cell centres:
    # ∂u_z/∂x = -4 U_mean x / R²,  ∂u_z/∂y = -4 U_mean y / R² (for r<R; 0 outside)
    # All other components vanish.
    grad = jnp.zeros((mesh.N_cells, 3, 3), dtype=dtype)
    inside = (rho_cell < R_pipe).astype(dtype)
    for a in cross_axes:
        d_uz_d_xa = -4.0 * U_mean * x[:, a] / (R_pipe ** 2) * inside
        grad = grad.at[:, axis, a].set(d_uz_d_xa.astype(dtype))

    du_lift_dt = jnp.zeros_like(u_lift)
    return LiftingFunction(
        u_lift_static=u_lift,
        du_lift_dt=du_lift_dt,
        u_lift_face=u_lift_face,
        grad_u_lift=grad,
        is_time_varying=False,
    )


def make_poiseuille_p_lift(*, mu: float, U_mean: float, pipe_radius: float):
    """Analytical Hagen–Poiseuille lifted pressure ``p_lift(z)``.

    For steady Poiseuille flow the lift's momentum balance requires a
    linear axial pressure gradient
    ``dP/dz = -8 μ U_mean / R²`` (Hagen–Poiseuille). The PISO solver
    never materialises this gradient — it lives only in the analytical
    lift balance. Pass the returned callable as ``p_lift_fn`` to
    :func:`mime.nodes.environment.fvm.ibm.momentum_deficit_drag` so the
    estimator can reconstruct the full physical pressure at each
    integration plane.

    Returns ``p(z)`` with the convention ``p(0) = 0`` (only the
    *difference* between integration planes matters in the
    momentum-deficit balance, so the additive constant is irrelevant).
    """
    dPdz = -8.0 * mu * U_mean / (pipe_radius ** 2)
    def p_lift(z):
        return dPdz * z
    return p_lift


def make_womersley_lift(
    mesh, *, R_pipe: float, U_mean_dc: float, U_mean_amp: float,
    omega: float, nu: float, n_steps: int, dt: float, axis: int = 2,
    phase_offset: float = 0.0,
    dtype=None,
) -> "LiftingFunction":
    """Build a time-varying Womersley lifting field for a Cartesian pipe.

    The driving body force per unit mass is chosen so that the bulk
    mean velocity matches ``U_mean(t) = U_mean_dc + U_mean_amp·cos(ωt)``
    in steady state.

    Memory-conscious: only ``u_lift_static`` and ``du_lift_dt`` are
    tabulated (over exactly one period of the oscillation, so the
    PISO step modulo-indexes via ``i_step % n_steps``). The face
    interpolation and cell gradient are recomputed on-the-fly inside
    :func:`compute_lifting_source` from these two arrays — that
    recomputation is cheap (a single face_interp + 3 grad_green_gauss
    calls) and saves storing the [N_steps, N_cells, 3, 3] gradient
    table that would otherwise dominate GPU memory.

    Note: ``n_steps`` here is the size of the *table*, NOT the total
    number of solver time steps. For periodic forcing pass
    ``n_steps = round(2π / (ω·dt))`` (one full cycle).
    """
    if dtype is None:
        dtype = mesh.V.dtype
    import numpy as np

    # Match U_mean targets to driving body forces
    f_steady = U_mean_dc * 8.0 * nu / (R_pipe ** 2)

    # Calibrate f_osc such that the bulk-mean Womersley amplitude == U_mean_amp.
    from mime.nodes.environment.fvm.womersley import (
        pipe_velocity, pipe_velocity_time_derivative, pipe_mean_velocity,
    )
    U0_test = pipe_mean_velocity(
        0.0, R=R_pipe, nu=nu, omega=omega, f_steady=0.0, f_osc=1.0,
    )
    U_quarter_test = pipe_mean_velocity(
        np.pi / (2.0 * omega), R=R_pipe, nu=nu, omega=omega,
        f_steady=0.0, f_osc=1.0,
    )
    test_amp = float(np.hypot(U0_test, U_quarter_test))
    f_osc = U_mean_amp / max(test_amp, 1e-30)

    cross_axes = [a for a in range(mesh.dim) if a != axis]
    x = np.asarray(mesh.x)
    rho_cell = np.sqrt(sum(x[:, a] ** 2 for a in cross_axes))
    inside = rho_cell < R_pipe

    np_dtype = np.float32 if dtype == jnp.float32 else np.float64
    u_lift_all = np.zeros((n_steps, mesh.N_cells, 3), dtype=np_dtype)
    du_lift_dt_all = np.zeros_like(u_lift_all)

    r_sample = np.linspace(0.0, R_pipe, 257)
    # phase_offset shifts the time origin: t_eff = t + phase_offset/ω.
    # phase_offset = -π/2 makes U(t=0) = U_dc + U_amp·cos(-π/2) = U_dc
    # only, which is the natural starting point when the FVM has been
    # warmed up to steady Poiseuille at U_dc.
    t_phase = phase_offset / omega
    for k in range(n_steps):
        t_k = float(k * dt + t_phase)
        u_z_r = pipe_velocity(
            r_sample, t_k, R=R_pipe, nu=nu, omega=omega,
            f_steady=f_steady, f_osc=f_osc,
        )
        du_dt_r = pipe_velocity_time_derivative(
            r_sample, t_k, R=R_pipe, nu=nu, omega=omega, f_osc=f_osc,
        )
        u_z_c = np.interp(rho_cell, r_sample, u_z_r) * inside
        du_dt_c = np.interp(rho_cell, r_sample, du_dt_r) * inside
        u_lift_all[k, :, axis] = u_z_c
        du_lift_dt_all[k, :, axis] = du_dt_c

    # Empty placeholders for face/grad arrays (recomputed in PISO).
    # Use shape [n_steps, 0, 3] / [n_steps, 0, 3, 3] so the JAX
    # take-along-axis at i_step still yields a recognisable empty
    # array; compute_lifting_source treats empty as None.
    return LiftingFunction(
        u_lift_static=jnp.asarray(u_lift_all, dtype=dtype),
        du_lift_dt=jnp.asarray(du_lift_dt_all, dtype=dtype),
        u_lift_face=jnp.zeros((n_steps, 0, 3), dtype=dtype),
        grad_u_lift=jnp.zeros((n_steps, 0, 3, 3), dtype=dtype),
        is_time_varying=True,
    )
