"""Steady-state SIMPLE solver for incompressible Navier-Stokes.

Implements the SIMPLE algorithm (Patankar 1980) on a collocated
Cartesian face graph with Rhie-Chow face-velocity correction
(Rhie & Chow 1983, Moukalled §15.6) and an FFT-diagonalised pressure
correction. Designed for steady benchmarks (M0: lid-driven cavity).

The algorithm here uses a *Jacobi-style* momentum predictor with
under-relaxation rather than an inner linear solver — this is the
standard practical SIMPLE form (Versteeg & Malalasekera §6.4) and keeps
the entire iteration inside ``jax.lax.fori_loop`` for JIT fusion.

Loop body (per outer iteration)
-------------------------------
1. ``∇p`` from current pressure (Green-Gauss).
2. Cell residual ``r = -conv + diff − V ∇p`` (steady momentum equation).
3. ``a_p`` (momentum diagonal) and Jacobi update
   ``u* = u + α_u · r / a_p``.
4. Rhie-Chow face velocity ``u_f^*`` and mass flux ``F_f^* = u_f^* · Sf``.
5. Pressure correction Poisson ``∇²p' = ∇·F^* / D̄`` solved by FFT
   under all-Neumann BCs; ``D̄ = mean(V/a_p)`` (constant-coefficient
   approximation valid for moderate-Re benchmarks).
6. Update pressure ``p ← p + α_p · p'`` and velocity
   ``u ← u* − (V/a_p) ∇p'``; correct flux ``F ← F* − D̄ |Sf|/|d| Δp'``.

References
----------
- Patankar (1980) Numerical Heat Transfer and Fluid Flow.
- Rhie & Chow (1983) AIAA J. 21(11) 1525–1532.
- Versteeg & Malalasekera (2007) An Introduction to CFD, 2nd ed., §6.4.
- Moukalled, Mangani, Darwish (2016) The Finite Volume Method in CFD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from mime.nodes.environment.fvm.mesh import FVMMesh
from mime.nodes.environment.fvm.boundary import (
    VelocityBC, velocity_diffusion_specs, velocity_convection_boundaries,
)
from mime.nodes.environment.fvm.operators import (
    grad_green_gauss,
    laplacian_orthogonal,
    convection_upwind_blend,
    divergence_face_flux,
    face_velocity_rhie_chow,
    momentum_diagonal_uniform_cartesian,
)
from mime.nodes.environment.fvm.pressure import (
    make_pressure_solver, make_pressure_solver_fft,
)


@dataclass(frozen=True)
class SimpleConfig:
    nu: float                    # kinematic viscosity
    rho: float = 1.0
    alpha_u: float = 0.7         # velocity under-relaxation
    alpha_p: float = 0.3         # pressure under-relaxation
    gamma_conv: float = 0.0      # 0 = pure upwind, 1 = central
    n_outer: int = 2000          # outer iteration cap


def initial_state(mesh: FVMMesh) -> dict:
    """Zero-velocity, zero-pressure, zero-flux initial condition."""
    dim = mesh.dim
    return {
        "u": jnp.zeros((mesh.N_cells, dim), dtype=mesh.V.dtype),
        "p": jnp.zeros((mesh.N_cells,), dtype=mesh.V.dtype),
        "F": jnp.zeros((mesh.N_faces,), dtype=mesh.V.dtype),
    }


def make_simple_step(
    mesh: FVMMesh,
    bcs: Dict[str, VelocityBC],
    cfg: SimpleConfig,
):
    """Build a JIT-friendly single SIMPLE iteration.

    The returned function maps ``state -> new_state`` and is fully
    pure: pass it to ``jax.lax.fori_loop`` to run multiple iterations.
    """
    mu = cfg.rho * cfg.nu
    diff_specs = velocity_diffusion_specs(mesh, bcs, mu=mu)
    bF, bphi = velocity_convection_boundaries(mesh, bcs)
    dtype = mesh.V.dtype

    # Keep dense matmul for SIMPLE — cuFFT batched plan fails for 2D
    # solver fori_loops on this hardware/driver. The 3D PISO path uses
    # FFT via PisoConfig.transform_backend="fft".
    pressure_solver = make_pressure_solver(mesh, bc="neumann")

    def step(state):
        u = state["u"].astype(dtype)            # [N_cells, dim]
        p = state["p"].astype(dtype)            # [N_cells]
        F = state["F"].astype(dtype)            # [N_faces]

        # ---- 1. Pressure gradient (cell-centred) ------------------
        grad_p = grad_green_gauss(p, mesh)         # [N_cells, dim]

        # ---- 2. Momentum residual ---------------------------------
        # convection per cell: ∫ ρ u (u · n) dS
        #    we work with F = u·Sf already (volumetric flux); for
        #    incompressible momentum the convected quantity is ρ u.
        rhoF = cfg.rho * F
        conv = convection_upwind_blend(
            u, rhoF, mesh,
            gamma=cfg.gamma_conv,
            boundary_phi=bphi,
            boundary_F={k: cfg.rho * v for k, v in bF.items()},
        )                                          # [N_cells, dim]
        diff = laplacian_orthogonal(
            u, mesh, mu_face=mu, boundary_specs=diff_specs,
        )                                          # [N_cells, dim]

        body = jnp.zeros_like(u)                   # no body forces yet
        residual = -conv + diff - mesh.V[:, None] * grad_p \
                   + mesh.V[:, None] * body

        # ---- 3. Momentum diagonal + Jacobi update ----------------
        a_p = momentum_diagonal_uniform_cartesian(
            mesh, nu=cfg.nu, rho=cfg.rho, F_face=rhoF,
        )                                          # [N_cells]
        a_p_safe = jnp.where(a_p < 1e-30, 1e-30, a_p)
        u_star = u + cfg.alpha_u * residual / a_p_safe[:, None]

        # ---- 4. Rhie-Chow face velocity → mass flux --------------
        u_face = face_velocity_rhie_chow(
            u_star, p, grad_p, a_p_safe, mesh,
        )                                          # [N_faces, dim]
        F_star = jnp.einsum("fd,fd->f", u_face, mesh.Sf)

        # Boundary fluxes: prescribed mass through-flow per patch.
        F_b_dict = bF  # already mapped name → [N_bf]

        # ---- 5. Pressure correction Poisson ----------------------
        div_F = divergence_face_flux(
            F_star, mesh, boundary_F=F_b_dict,
        )                                          # [N_cells]
        D_bar = jnp.mean(mesh.V / a_p_safe)        # uniform-D̄ surrogate
        # rhs has units of [Volume / time]; divide by D̄ to convert to
        # the units expected by the FFT-discretised Poisson operator.
        # Solver expects ∫ ∇²p' dV = b ⇒ pass div_F / D̄.
        rhs_p = div_F / D_bar
        p_prime = pressure_solver(rhs_p)           # [N_cells]
        # Subtract mean to keep gauge stable
        p_prime = p_prime - jnp.mean(p_prime)

        # ---- 6. Update p, u, F -----------------------------------
        p_new = p + cfg.alpha_p * p_prime
        grad_pp = grad_green_gauss(p_prime, mesh)
        u_new = u_star - (mesh.V / a_p_safe)[:, None] * grad_pp

        # Face flux correction:  F_new = F* − D_face * (p'_N − p'_P) |Sf|/|d|
        dpp = p_prime[mesh.neighbour] - p_prime[mesh.owner]
        F_new = F_star - D_bar * mesh.area / mesh.d_mag * dpp

        return {
            "u": u_new.astype(dtype),
            "p": p_new.astype(dtype),
            "F": F_new.astype(dtype),
        }

    return step


def run_simple(
    mesh: FVMMesh,
    bcs: Dict[str, VelocityBC],
    cfg: SimpleConfig,
    *,
    n_iter: int | None = None,
    initial: dict | None = None,
) -> dict:
    """Run ``n_iter`` SIMPLE outer iterations from ``initial``.

    The whole loop runs inside ``jax.lax.fori_loop`` and is JIT-compiled
    on first call.
    """
    if n_iter is None:
        n_iter = cfg.n_outer
    if initial is None:
        initial = initial_state(mesh)
    step = make_simple_step(mesh, bcs, cfg)

    @jax.jit
    def _run(state):
        return jax.lax.fori_loop(0, n_iter, lambda i, s: step(s), state)

    return _run(initial)


def continuity_residual_l2(
    state: dict, mesh: FVMMesh, bcs: Dict[str, VelocityBC],
) -> jnp.ndarray:
    """Compute L2 norm of cell continuity residual ∇·F per unit volume."""
    bF, _ = velocity_convection_boundaries(mesh, bcs)
    div = divergence_face_flux(state["F"], mesh, boundary_F=bF)
    return jnp.sqrt(jnp.mean((div / mesh.V) ** 2))


def momentum_residual_l2(
    state: dict, mesh: FVMMesh, bcs: Dict[str, VelocityBC], cfg: SimpleConfig,
) -> jnp.ndarray:
    """Compute L2 norm of cell momentum residual."""
    mu = cfg.rho * cfg.nu
    diff_specs = velocity_diffusion_specs(mesh, bcs, mu=mu)
    bF, bphi = velocity_convection_boundaries(mesh, bcs)
    u = state["u"]; p = state["p"]; F = state["F"]
    grad_p = grad_green_gauss(p, mesh)
    rhoF = cfg.rho * F
    conv = convection_upwind_blend(
        u, rhoF, mesh, gamma=cfg.gamma_conv,
        boundary_phi=bphi,
        boundary_F={k: cfg.rho * v for k, v in bF.items()},
    )
    diff = laplacian_orthogonal(
        u, mesh, mu_face=mu, boundary_specs=diff_specs,
    )
    res = -conv + diff - mesh.V[:, None] * grad_p
    return jnp.sqrt(jnp.mean(jnp.sum(res ** 2, axis=-1)))
