"""Semi-implicit 6DOF particle integrator for IBM-coupled simulations.

The Maxey-Riley equation for a small rigid sphere in a fluid:

    m_p dv/dt = F_drag(u_f − v) + F_pressure_gradient + F_added_mass + F_lift

with linear Stokes drag F_drag = 6πμa(u_f − v). For a particle with
``ρ_p ≈ ρ_f`` the relaxation time τ_p = ρ_p (2a)² / (18 μ) is short
compared to the fluid PISO time step, and an explicit Euler position
update overshoots wildly. This module gives an implicit-drag update
that is unconditionally stable in dt:

    (1 + dt / τ_p) v_new = v_old + (dt / τ_p) u_f + dt * F_other / m_p
    x_new = x_old + dt * v_new

For the Segré-Silberberg validation we treat F_other as the
hydrodynamic force from the surface integral *minus* the linear
Stokes-drag component (which is already absorbed into the implicit
update). In practice we use F_other = F_total directly when F_total
is small relative to the Stokes restoring force (lift-only regime),
which is the case once the sphere is close to its terminal axial
velocity.

Sub-stepping is supported via ``n_sub``: the fluid force is held
fixed and the position equation is integrated with N small substeps
of length dt_sub = dt / n_sub. This further suppresses overshoot
without re-running PISO.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ParticleState:
    """Position + velocity for a single 6DOF rigid particle."""
    position: jnp.ndarray         # [dim]
    velocity: jnp.ndarray         # [dim]


jax.tree_util.register_pytree_node(
    ParticleState,
    lambda s: ((s.position, s.velocity), None),
    lambda _, ch: ParticleState(position=ch[0], velocity=ch[1]),
)


def implicit_drag_step(
    state: ParticleState,
    F_external: jnp.ndarray,
    u_fluid_at_particle: jnp.ndarray,
    *,
    m_p: float,
    drag_coeff: float,
    dt: float,
    n_sub: int = 1,
) -> ParticleState:
    """Semi-implicit Euler with linear-drag damping.

    Per substep dt_sub = dt / n_sub:
        v <- (v_old + (dt_sub/τ) u_f + (dt_sub/m) F_ext) / (1 + dt_sub/τ)
        x <- x_old + dt_sub * v

    where τ = m_p / drag_coeff is the particle relaxation time.

    ``drag_coeff`` is the Stokes-drag coefficient ``6πμa`` (or whatever
    linearised drag the user wants implicit). ``F_external`` is the
    *non-drag* component of the hydrodynamic force (lift, pressure-
    gradient, body force) — held fixed over the substeps.
    """
    dt_sub = dt / n_sub
    tau = m_p / drag_coeff

    def step(carry, _):
        p, v = carry
        v_new = (
            (v + (dt_sub / tau) * u_fluid_at_particle
             + (dt_sub / m_p) * F_external)
            / (1.0 + dt_sub / tau)
        )
        p_new = p + dt_sub * v_new
        return (p_new, v_new), None

    (p_final, v_final), _ = jax.lax.scan(
        step, (state.position, state.velocity), jnp.arange(n_sub),
    )
    return ParticleState(position=p_final, velocity=v_final)


def trilinear_interp(field: jnp.ndarray, x: jnp.ndarray,
                     mesh) -> jnp.ndarray:
    """Trilinear interpolation of a cell-centred field at a point.

    ``field`` has shape ``[N_cells, k]`` (or ``[N_cells]``).
    ``x`` is a single point ``[dim]``. Returns ``[k]`` (or scalar).
    Assumes mesh is Cartesian-structured (uses cartesian_shape /
    cartesian_spacing / cartesian_origin).
    """
    shape = mesh.cartesian_shape
    spacing = mesh.cartesian_spacing
    origin = mesh.cartesian_origin
    dim = len(shape)

    # Local index into the grid (in floating cell coordinates)
    idx_f = jnp.stack([
        (x[a] - (origin[a] + 0.5 * spacing[a])) / spacing[a]
        for a in range(dim)
    ])
    # Clip into valid interpolation range
    idx_f = jnp.clip(idx_f, 0.0, jnp.array([s - 1.0 for s in shape]))
    idx_lo = jnp.floor(idx_f).astype(jnp.int32)
    frac = idx_f - idx_lo

    field_3d = field.reshape(shape + field.shape[1:])

    if dim == 3:
        # 8 corner samples
        i0, j0, k0 = idx_lo[0], idx_lo[1], idx_lo[2]
        i1 = jnp.minimum(i0 + 1, shape[0] - 1)
        j1 = jnp.minimum(j0 + 1, shape[1] - 1)
        k1 = jnp.minimum(k0 + 1, shape[2] - 1)
        fx, fy, fz = frac[0], frac[1], frac[2]
        result = (
            field_3d[i0, j0, k0] * (1-fx)*(1-fy)*(1-fz)
            + field_3d[i1, j0, k0] *    fx *(1-fy)*(1-fz)
            + field_3d[i0, j1, k0] * (1-fx)*   fy *(1-fz)
            + field_3d[i0, j0, k1] * (1-fx)*(1-fy)*   fz
            + field_3d[i1, j1, k0] *    fx *   fy *(1-fz)
            + field_3d[i1, j0, k1] *    fx *(1-fy)*   fz
            + field_3d[i0, j1, k1] * (1-fx)*   fy *   fz
            + field_3d[i1, j1, k1] *    fx *   fy *   fz
        )
        return result
    elif dim == 2:
        i0, j0 = idx_lo[0], idx_lo[1]
        i1 = jnp.minimum(i0 + 1, shape[0] - 1)
        j1 = jnp.minimum(j0 + 1, shape[1] - 1)
        fx, fy = frac[0], frac[1]
        return (
            field_3d[i0, j0] * (1-fx)*(1-fy)
            + field_3d[i1, j0] *    fx *(1-fy)
            + field_3d[i0, j1] * (1-fx)*   fy
            + field_3d[i1, j1] *    fx *   fy
        )
    else:
        raise NotImplementedError(f"dim={dim} unsupported")
