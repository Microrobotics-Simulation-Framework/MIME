"""Wall correction methods for defect correction coupling.

Three methods for computing the wall-induced flow at the body surface:

1. **Richardson**: Inline linear Richardson extrapolation from 3 eval radii.
   Proven at 0.1% for axial translation at 48³. Stable iteration.

2. **Lamb**: 6-term Lamb series basis fit from 8 eval radii.
   Proven at 5.5% for transverse translation at 48³ (one pass, no iteration).

3. **Representation**: Stokes boundary integral representation formula.
   Uses velocity AND stress on a control surface. Theoretically exact.
   Needs ≥64³ for accurate LBM stress. Geometry-general.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.lbm.immersed_boundary import interpolate_velocity
from mime.nodes.environment.stokeslet.flow_field import evaluate_velocity_field
from mime.nodes.environment.stokeslet.stresslet import stresslet_tensor_contracted


# ── Method 1: Inline Richardson ──────────────────────────────────────

def wall_correction_richardson(
    u_lbm, traction, body_pts, body_wts,
    eval_stencils, eval_d_vals,
    epsilon, mu, dx, dt,
):
    """Linear Richardson extrapolation from closest and farthest eval radii.

    Returns (3,) uniform correction in physical units.
    """
    du_means = []
    for es in eval_stencils:
        u_w = interpolate_velocity(u_lbm, es['idx'], es['wts']) * dx / dt
        u_fs = evaluate_velocity_field(es['pts_phys'], body_pts, body_wts,
                                        traction, epsilon, mu)
        du_means.append(jnp.mean(u_w - u_fs, axis=0))

    d_inner = eval_d_vals[0]
    d_outer = eval_d_vals[-1]
    return (du_means[0] * d_outer - du_means[-1] * d_inner) / (d_outer - d_inner)


# ── Method 2: 6-term Lamb basis ──────────────────────────────────────

def wall_correction_lamb(
    u_lbm, traction, body_pts, body_wts,
    eval_stencils, eval_R_phys, body_radius,
    epsilon, mu, dx, dt,
    motion_axis,
):
    """6-term Lamb series extrapolation. One-pass only — do not iterate.

    Returns (3,) uniform correction in physical units.
    """
    R = eval_R_phys
    a = body_radius

    # Build 6-term Lamb Vandermonde matrix
    if motion_axis in [0, 1]:  # transverse (m=1)
        V = jnp.column_stack([R, R**3, R**5, 1/R**2, 1/R**4, 1/R**6])
        V_body = jnp.array([a, a**3, a**5, 1/a**2, 1/a**4, 1/a**6])
    else:  # axial (m=0)
        V = jnp.column_stack([jnp.ones_like(R), R**2, R**4, 1/R, 1/R**3, 1/R**5])
        V_body = jnp.array([1, a**2, a**4, 1/a, 1/a**3, 1/a**5])

    # Normalise columns
    norms = jnp.max(jnp.abs(V), axis=0)
    norms = jnp.where(norms < 1e-30, 1.0, norms)
    V_n = V / norms
    V_body_n = V_body / norms

    # Sample Δu at each eval radius
    du_means = []
    for es in eval_stencils:
        u_w = interpolate_velocity(u_lbm, es['idx'], es['wts']) * dx / dt
        u_fs = evaluate_velocity_field(es['pts_phys'], body_pts, body_wts,
                                        traction, epsilon, mu)
        du_means.append(jnp.mean(u_w - u_fs, axis=0))
    du_means = jnp.stack(du_means)

    # Fit and extrapolate per component
    result = jnp.zeros(3)
    for c in range(3):
        coeffs = jnp.linalg.lstsq(V_n, du_means[:, c])[0]
        result = result.at[c].set(V_body_n @ coeffs)
    return result


# ── Method 3: Boundary integral representation ───────────────────────

def wall_correction_representation(
    u_lbm, f_dist, traction, body_pts, body_wts,
    repr_mesh, repr_stencil, repr_f_stencil,
    tau, epsilon, mu, dx, dt, rho_phys,
):
    """Wall correction via Stokes representation formula on a control surface.

    Computes the wall-induced velocity at body surface points using
    the boundary integral: u_wall = -∫ T·Δu·n dS + ∫ G·Δt dS

    Returns (N_body, 3) per-point correction in physical units.
    """
    N_body = len(body_pts)
    ctrl_pts = repr_mesh['pts_phys']
    ctrl_normals = repr_mesh['normals']
    ctrl_weights = repr_mesh['weights']
    N_ctrl = len(ctrl_pts)

    # 1. LBM velocity at control surface
    u_lbm_ctrl = interpolate_velocity(
        u_lbm, repr_stencil['idx'], repr_stencil['wts'],
    ) * dx / dt

    # 2. BEM free-space velocity at control surface
    u_bem_ctrl = evaluate_velocity_field(
        ctrl_pts, body_pts, body_wts, traction, epsilon, mu,
    )

    # 3. Velocity difference (wall-only contribution)
    delta_u_ctrl = u_lbm_ctrl - u_bem_ctrl  # (N_ctrl, 3)

    # 4. LBM stress → traction at control surface
    t_lbm_ctrl = _compute_lbm_traction(
        f_dist, tau, repr_f_stencil, ctrl_normals, dx, dt, rho_phys,
    )

    # 5. BEM free-space traction at control surface
    t_bem_ctrl = _compute_bem_traction(
        ctrl_pts, ctrl_normals, body_pts, body_wts,
        traction, epsilon, mu,
    )

    # 6. Traction difference
    delta_t_ctrl = t_lbm_ctrl - t_bem_ctrl  # (N_ctrl, 3)

    # 7. Double-layer integral: -∫ T·Δu·n dS at body points
    dl = _double_layer_integral(
        body_pts, ctrl_pts, ctrl_normals, ctrl_weights,
        delta_u_ctrl, epsilon,
    )

    # 8. Single-layer integral: +∫ G·Δt dS at body points
    sl = evaluate_velocity_field(
        body_pts, ctrl_pts, ctrl_weights, delta_t_ctrl, epsilon, mu,
    )

    return -dl + sl  # (N_body, 3)


def _compute_lbm_traction(f_dist, tau, f_stencil, normals, dx, dt, rho_phys):
    """Compute traction t = σ·n at interpolation points from LBM f^{neq}.

    σ_αβ = -(1 - 1/(2τ)) Σ_i f_i^{neq} c_{iα} c_{iβ}
    """
    from mime.nodes.environment.lbm.d3q19 import E, equilibrium

    idx = f_stencil['idx']
    wts = f_stencil['wts']
    N_pts = len(idx)

    # Interpolate each of the 19 distribution components
    f_flat = f_dist.reshape(-1, 19)
    f_at_pts = jnp.zeros((N_pts, 19))
    for q in range(19):
        fq = f_flat[:, q:q+1]
        fq_3d = fq.reshape(*f_dist.shape[:3], 1)
        f_at_pts = f_at_pts.at[:, q].set(
            interpolate_velocity(fq_3d, idx, wts)[:, 0]
        )

    # Macroscopic quantities at interpolation points
    e = jnp.array(E, dtype=jnp.float32)
    rho = jnp.sum(f_at_pts, axis=-1)
    mom = jnp.sum(f_at_pts[:, :, None] * e[None, :, :], axis=1)
    u = mom / rho[:, None]

    # f_eq and f_neq
    f_eq = equilibrium(rho, u)
    f_neq = f_at_pts - f_eq

    # Stress tensor: σ_αβ = -(1-1/(2τ)) Σ_i f_neq_i c_iα c_iβ
    pref = -(1.0 - 0.5 / tau)
    # Compute σ·n directly (avoid full 3x3 tensor)
    # t_α = σ_αβ n_β = pref * Σ_i f_neq_i c_iα (c_iβ n_β)
    c_dot_n = jnp.sum(e[None, :, :] * normals[:, None, :], axis=2)  # (N_pts, 19)
    t_lu = pref * jnp.sum(
        f_neq * c_dot_n[:, :, None] * e[None, :, :],
        axis=1,
    )  # Wait, this isn't right. Let me be more careful.

    # t_α = Σ_β σ_αβ n_β = pref Σ_i f_neq_i c_iα Σ_β c_iβ n_β
    #      = pref Σ_i f_neq_i c_iα (c_i · n)
    t_lu = pref * jnp.sum(
        f_neq[:, :, None] * e[None, :, :] * c_dot_n[:, :, None],
        axis=1,
    )  # (N_pts, 3)

    # Convert to physical: t_phys = t_lu × ρ₀_phys × dx / dt²
    # Actually: σ_lu has units of ρ_lu × c_s² (pressure). Physical: σ_phys = σ_lu × ρ₀ × dx²/dt²
    # t = σ·n, so t_phys = σ_lu × ρ₀ × dx²/dt² (same as σ since n is dimensionless)
    t_phys = t_lu * (rho_phys * dx**2 / dt**2)

    return t_phys


def _compute_bem_traction(eval_pts, eval_normals, body_pts, body_wts,
                           traction, epsilon, mu):
    """BEM free-space traction at eval points.

    The stress of the Stokeslet field:
    σ_jk(x) = Σ_n [T_jkl(x, y_n) / (8π)] f_l(y_n) w_n
    t_j(x) = σ_jk n_k = Σ_n [T_jkl n_k / (8π)] f_l w_n
           = Σ_n K_jl(x, y_n, n) f_l w_n / (8π)

    Where K_jl = T_jkl n_k is stresslet_tensor_contracted.
    """
    prefactor = 3.0 / (4.0 * jnp.pi)  # double-layer prefactor for Stokes

    def _traction_at_point(x, n_x):
        def _contrib(y, w, f):
            K = stresslet_tensor_contracted(x, y, n_x, epsilon)
            return prefactor * w * K @ f

        contribs = jax.vmap(_contrib)(body_pts, body_wts, traction)
        return jnp.sum(contribs, axis=0)

    return jax.vmap(_traction_at_point)(eval_pts, eval_normals)


def _double_layer_integral(body_pts, ctrl_pts, ctrl_normals, ctrl_weights,
                            u_ctrl, epsilon):
    """Double-layer integral: ∫ T_ijk(x₀, y) u_j(y) n_k(y) dS(y) at body points.

    Prefactor: 3/(4π) for the Stokes double-layer kernel.
    """
    prefactor = 3.0 / (4.0 * jnp.pi)

    def _dl_at_point(x):
        def _contrib(y, n, w, u):
            K = stresslet_tensor_contracted(x, y, n, epsilon)
            return prefactor * w * K @ u

        contribs = jax.vmap(_contrib)(ctrl_pts, ctrl_normals, ctrl_weights, u_ctrl)
        return jnp.sum(contribs, axis=0)

    return jax.vmap(_dl_at_point)(body_pts)


# ── Dispatcher ───────────────────────────────────────────────────────

def compute_wall_correction(
    method, u_lbm, traction, body_pts, body_wts,
    eval_data, epsilon, mu, dx, dt,
    f_dist=None, tau=None, rho_phys=None,
    motion_axis=None, body_radius=None,
):
    """Dispatch to the appropriate wall correction method."""
    if method == "richardson":
        return wall_correction_richardson(
            u_lbm, traction, body_pts, body_wts,
            eval_data['stencils_close'], eval_data['d_vals_close'],
            epsilon, mu, dx, dt,
        )
    elif method == "lamb":
        return wall_correction_lamb(
            u_lbm, traction, body_pts, body_wts,
            eval_data['stencils_all'], eval_data['R_phys_all'], body_radius,
            epsilon, mu, dx, dt, motion_axis,
        )
    elif method == "representation":
        return wall_correction_representation(
            u_lbm, f_dist, traction, body_pts, body_wts,
            eval_data['repr_mesh'], eval_data['repr_stencil'],
            eval_data['repr_f_stencil'],
            tau, epsilon, mu, dx, dt, rho_phys,
        )
    else:
        raise ValueError(f"Unknown wall correction method: {method}")
