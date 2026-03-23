"""Convergence monitoring for steady-state LBM simulations.

Provides residual computation and a run-to-convergence driver that
iterates until velocity residuals fall below a threshold.

Uses a Python loop (not jax.lax.scan) so that residuals can be
checked periodically without recompiling.
"""

from __future__ import annotations

import jax.numpy as jnp

from mime.nodes.environment.lbm.d3q19 import lbm_step_split, compute_macroscopic
from mime.nodes.environment.lbm.bounce_back import (
    apply_bounce_back,
    apply_bouzidi_bounce_back,
)


def compute_velocity_residual(
    u_new: jnp.ndarray,
    u_old: jnp.ndarray,
    fluid_mask: jnp.ndarray,
    norm_type: str = "L2",
) -> float:
    """Compute velocity residual over fluid nodes only.

    Parameters
    ----------
    u_new : (nx, ny, nz, 3) float32
    u_old : (nx, ny, nz, 3) float32
    fluid_mask : (nx, ny, nz) bool
        True at fluid nodes (where residual is evaluated).
    norm_type : str
        "L2": sqrt(sum((u_new - u_old)^2 on fluid) / sum(u_new^2 on fluid))
        "Linf": max |u_new - u_old| on fluid

    Returns
    -------
    residual : float
    """
    diff = u_new - u_old  # (nx, ny, nz, 3)

    if norm_type == "Linf":
        # max |u_new - u_old| over fluid nodes
        diff_mag = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))  # (nx, ny, nz)
        return float(jnp.max(jnp.where(fluid_mask, diff_mag, 0.0)))

    # L2 norm: sqrt(sum(diff^2 on fluid) / sum(u_new^2 on fluid))
    diff_sq = jnp.sum(diff ** 2, axis=-1)  # (nx, ny, nz)
    u_sq = jnp.sum(u_new ** 2, axis=-1)    # (nx, ny, nz)

    num = jnp.sum(jnp.where(fluid_mask, diff_sq, 0.0))
    den = jnp.sum(jnp.where(fluid_mask, u_sq, 0.0))

    # Avoid division by zero when u_new is zero everywhere
    den = jnp.maximum(den, 1e-30)
    return float(jnp.sqrt(num / den))


def run_to_convergence(
    f_init: jnp.ndarray,
    tau: float,
    solid_mask: jnp.ndarray,
    missing_mask: jnp.ndarray,
    q_values: jnp.ndarray | None = None,
    wall_velocity: jnp.ndarray | None = None,
    wall_correction: jnp.ndarray | None = None,
    wall_feq: jnp.ndarray | None = None,
    max_steps: int = 50000,
    check_interval: int = 100,
    rtol: float = 1e-6,
    norm_type: str = "L2",
    use_bouzidi: bool = False,
    force: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, int, list[float]]:
    """Run LBM to steady state with residual monitoring.

    Python loop (not jax.lax.scan) with residual check every
    check_interval steps. Each step: lbm_step_split -> apply BB or
    Bouzidi BB -> track residual.

    Parameters
    ----------
    f_init : (nx, ny, nz, Q) float32
        Initial distribution functions.
    tau : float
        BGK relaxation time.
    solid_mask : (nx, ny, nz) bool
    missing_mask : (Q, nx, ny, nz) bool
    q_values : (Q, nx, ny, nz) float32, optional
        Required when use_bouzidi=True.
    wall_velocity : (nx, ny, nz, 3) float32, optional
    wall_correction : (Q, nx, ny, nz) float32, optional
    wall_feq : (Q, nx, ny, nz) float32, optional
    max_steps : int
    check_interval : int
    rtol : float
    norm_type : str
    use_bouzidi : bool
    force : (nx, ny, nz, 3) float32, optional

    Returns
    -------
    f_final : (nx, ny, nz, Q) float32
    n_steps_actual : int
    residual_history : list[float]
        Residual at each check_interval.
    """
    fluid_mask = ~solid_mask
    f = f_init
    residual_history: list[float] = []

    # Get initial velocity for residual tracking
    _, u_old = compute_macroscopic(f, force=force)

    n_steps_actual = 0
    for step in range(1, max_steps + 1):
        # Collision + streaming (no BC)
        f_pre, f_post, rho, u = lbm_step_split(f, tau, force=force)

        # Apply boundary conditions
        if use_bouzidi and q_values is not None:
            f = apply_bouzidi_bounce_back(
                f_post, f_pre, missing_mask, solid_mask,
                q_values, wall_velocity=wall_velocity,
                wall_correction=wall_correction, wall_feq=wall_feq,
            )
        else:
            f = apply_bounce_back(
                f_post, f_pre, missing_mask, solid_mask,
                wall_velocity=wall_velocity,
            )

        n_steps_actual = step

        # Residual check
        if step % check_interval == 0:
            _, u_new = compute_macroscopic(f, force=force)
            residual = compute_velocity_residual(u_new, u_old, fluid_mask, norm_type)
            residual_history.append(residual)
            u_old = u_new

            if residual < rtol:
                break

    return f, n_steps_actual, residual_history
