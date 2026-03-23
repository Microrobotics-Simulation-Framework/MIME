"""Per-step rotating solid mask update for LBM simulations.

At each LBM step, the solid body is rotated by angular_velocity * dt_lbm,
a new solid mask is generated, and boundary conditions are recomputed.

This module wraps the geometry, boundary condition, and LBM step logic
into a single rotating_body_step function and a simulation driver.
"""

from __future__ import annotations

import jax.numpy as jnp

from mime.nodes.environment.lbm.d3q19 import (
    lbm_step_split,
    init_equilibrium,
    compute_macroscopic,
)
from mime.nodes.environment.lbm.bounce_back import (
    compute_missing_mask,
    apply_bounce_back,
    apply_bouzidi_bounce_back,
    compute_q_values_sdf,
    compute_momentum_exchange_force,
    compute_momentum_exchange_torque,
)
from mime.nodes.robot.helix_geometry import (
    create_umr_mask,
    umr_sdf,
)


def _rotation_velocity_field(
    shape: tuple[int, int, int],
    angular_velocity: float,
    rotation_axis: tuple[float, float, float],
    center: tuple[float, float, float],
) -> jnp.ndarray:
    """Compute omega x r at every grid node.

    Unlike compute_helix_wall_velocity, this does NOT zero outside the solid.
    apply_bounce_back reads wall_velocity at *fluid* boundary nodes, so the
    field must be defined everywhere on the grid.

    Returns
    -------
    wall_velocity : (nx, ny, nz, 3) float32
    """
    nx, ny, nz = shape
    omega_vec = jnp.array(rotation_axis, dtype=jnp.float32)
    omega_vec = omega_vec / jnp.maximum(jnp.linalg.norm(omega_vec), 1e-30)
    omega_vec = omega_vec * angular_velocity

    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    iz = jnp.arange(nz, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')

    rx = gx - center[0]
    ry = gy - center[1]
    rz = gz - center[2]

    ux = omega_vec[1] * rz - omega_vec[2] * ry
    uy = omega_vec[2] * rx - omega_vec[0] * rz
    uz = omega_vec[0] * ry - omega_vec[1] * rx

    return jnp.stack([ux, uy, uz], axis=-1)


def rotating_body_step(
    f: jnp.ndarray,
    tau: float,
    angular_velocity: float,
    dt_lbm: float,
    current_angle: float,
    geometry_params: dict,
    center: tuple[float, float, float],
    rotation_axis: tuple[float, float, float] = (0, 0, 1),
    use_bouzidi: bool = True,
    force: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """One LBM step with rotating solid body.

    Steps:
    1. angle_new = current_angle + angular_velocity * dt_lbm
    2. Generate solid_mask at new angle
    3. Compute missing_mask and wall velocity
    4. If Bouzidi: compute q_values via SDF bisection
    5. LBM collision + streaming
    6. Apply bounce-back
    7. Compute momentum exchange force/torque

    Parameters
    ----------
    f : (nx, ny, nz, Q) float32
    tau : float
    angular_velocity : float (rad per dt_lbm)
    dt_lbm : float
    current_angle : float (radians)
    geometry_params : dict
        Keyword arguments for create_umr_mask (body_radius, body_length, etc.)
    center : (cx, cy, cz)
    rotation_axis : (ax, ay, az)
    use_bouzidi : bool
    force : (nx, ny, nz, 3) float32, optional

    Returns
    -------
    f_new : (nx, ny, nz, Q) float32
    force_on_body : (3,) float32
    torque_on_body : (3,) float32
    velocity_field : (nx, ny, nz, 3) float32
    new_angle : float
    """
    # 1. Update angle
    new_angle = current_angle + angular_velocity * dt_lbm

    # 2. Generate solid mask at new angle
    solid_mask = create_umr_mask(
        **geometry_params,
        center=center,
        rotation_angle=new_angle,
    )

    # 3. Compute missing mask and wall velocity
    missing_mask = compute_missing_mask(solid_mask)
    # Wall velocity = omega x r at ALL nodes (not zeroed outside solid).
    # apply_bounce_back reads wall_velocity at fluid boundary nodes, so the
    # velocity field must be defined there for the Ladd correction.
    wall_vel = _rotation_velocity_field(
        solid_mask.shape, angular_velocity, rotation_axis, center,
    ) if angular_velocity != 0.0 else None

    # 4. If Bouzidi: compute q_values via SDF bisection
    q_values = None
    if use_bouzidi:
        def sdf_func(pts):
            return umr_sdf(pts, rotation_angle=new_angle, center=center,
                           **{k: v for k, v in geometry_params.items()
                              if k not in ('nx', 'ny', 'nz')})
        q_values = compute_q_values_sdf(missing_mask, sdf_func)

    # 5. LBM collision + streaming
    f_pre, f_post, rho, u = lbm_step_split(f, tau, force=force)

    # 6. Apply bounce-back
    if use_bouzidi and q_values is not None:
        f_new = apply_bouzidi_bounce_back(
            f_post, f_pre, missing_mask, solid_mask,
            q_values, wall_velocity=wall_vel,
        )
    else:
        f_new = apply_bounce_back(
            f_post, f_pre, missing_mask, solid_mask,
            wall_velocity=wall_vel,
        )

    # 7. Compute momentum exchange force/torque
    body_center = jnp.array(center, dtype=jnp.float32)
    force_on_body = compute_momentum_exchange_force(f_pre, f_new, missing_mask)
    torque_on_body = compute_momentum_exchange_torque(f_pre, f_new, missing_mask, body_center)

    # Velocity field (from the current step)
    _, velocity_field = compute_macroscopic(f_new, force=force)

    return f_new, force_on_body, torque_on_body, velocity_field, new_angle


def run_rotating_body_simulation(
    geometry_params: dict,
    tau: float,
    angular_velocity: float,
    n_steps: int,
    nx: int,
    ny: int,
    nz: int,
    center: tuple[float, float, float] | None = None,
    use_bouzidi: bool = True,
) -> dict:
    """Run n_steps of rotating body simulation.

    Uses a Python loop (not scan) since geometry changes each step.

    Parameters
    ----------
    geometry_params : dict
        Keyword arguments for create_umr_mask.
    tau : float
    angular_velocity : float
    n_steps : int
    nx, ny, nz : int
    center : (cx, cy, cz), optional
    use_bouzidi : bool

    Returns
    -------
    results : dict with keys:
        'force_history' : (n_steps, 3) float32
        'torque_history' : (n_steps, 3) float32
        'final_velocity' : (nx, ny, nz, 3) float32
        'final_f' : (nx, ny, nz, Q) float32
        'final_angle' : float
    """
    if center is None:
        center = (nx / 2.0, ny / 2.0, nz / 2.0)

    dt_lbm = 1.0
    f = init_equilibrium(nx, ny, nz)
    angle = 0.0

    force_history = []
    torque_history = []

    for step in range(n_steps):
        f, force_on_body, torque_on_body, velocity, angle = rotating_body_step(
            f, tau, angular_velocity, dt_lbm, angle,
            geometry_params, center,
            use_bouzidi=use_bouzidi,
        )
        force_history.append(force_on_body)
        torque_history.append(torque_on_body)

    return {
        'force_history': jnp.stack(force_history, axis=0),
        'torque_history': jnp.stack(torque_history, axis=0),
        'final_velocity': velocity,
        'final_f': f,
        'final_angle': angle,
    }
