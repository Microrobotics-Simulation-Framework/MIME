"""IB-LBM Coupled Solver — Immersed Boundary Lattice Boltzmann.

Combines D2Q9 LBM with Multi-Direct Forcing IB method.

Coupling loop per timestep:
1. Compute IB kernels for current boundary positions
2. Multi-Direct Forcing: iteratively correct velocity at boundary
3. LBM collision + streaming + bounce-back (with IB body force)
4. Compute drag force/torque on immersed body

The solver is stateless — all state is passed in and returned.
This makes it compatible with JAX's functional paradigm.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mime.nodes.environment.lbm.d2q9 import (
    lbm_step,
    init_equilibrium,
    compute_macroscopic,
    tau_from_viscosity,
    create_channel_walls,
    CS2,
)
from mime.nodes.environment.lbm.ib import (
    compute_kernels,
    multi_direct_forcing,
    interpolate_velocity,
    compute_marker_forces,
    compute_drag_force,
    compute_drag_torque,
    generate_circle_points,
    compute_boundary_velocities,
)


@dataclass
class IBLBMConfig:
    """Configuration for the IB-LBM solver.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions (lattice units).
    tau : float
        BGK relaxation time. nu = (tau - 0.5) / 3.
    mdf_iterations : int
        Number of Multi-Direct Forcing iterations per timestep.
    dx : float
        Physical lattice spacing [m].
    dt : float
        Physical LBM timestep [s].
    """
    nx: int = 100
    ny: int = 50
    tau: float = 0.8
    mdf_iterations: int = 5
    dx: float = 1.0
    dt: float = 1.0


@dataclass
class IBLBMState:
    """State of the IB-LBM solver (all JAX arrays)."""
    f: jnp.ndarray             # (nx, ny, 9) distribution functions
    density: jnp.ndarray       # (nx, ny)
    velocity: jnp.ndarray      # (nx, ny, 2)
    wall_mask: jnp.ndarray     # (nx, ny) bool
    step_count: int = 0


def create_state(
    config: IBLBMConfig,
    initial_density: float = 1.0,
    initial_velocity: tuple[float, float] = (0.0, 0.0),
    wall_mask: jnp.ndarray | None = None,
) -> IBLBMState:
    """Create initial IB-LBM state."""
    f = init_equilibrium(config.nx, config.ny, initial_density, initial_velocity)
    density = jnp.ones((config.nx, config.ny)) * initial_density
    velocity = jnp.zeros((config.nx, config.ny, 2))
    velocity = velocity.at[..., 0].set(initial_velocity[0])
    velocity = velocity.at[..., 1].set(initial_velocity[1])

    if wall_mask is None:
        wall_mask = create_channel_walls(config.nx, config.ny)

    return IBLBMState(
        f=f, density=density, velocity=velocity,
        wall_mask=wall_mask, step_count=0,
    )


@dataclass
class IBResult:
    """Result of one IB-LBM step with immersed boundary."""
    drag_force: jnp.ndarray      # (2,) total drag on body (lattice units)
    drag_torque: jnp.ndarray     # scalar torque on body
    marker_forces: jnp.ndarray   # (N, 2) per-marker forces
    u_markers: jnp.ndarray       # (N, 2) interpolated velocity at markers


def step(
    state: IBLBMState,
    config: IBLBMConfig,
    boundary_points: jnp.ndarray | None = None,
    target_velocities: jnp.ndarray | None = None,
    ds: float = 1.0,
    body_center: jnp.ndarray | None = None,
    external_force: jnp.ndarray | None = None,
) -> tuple[IBLBMState, IBResult | None]:
    """Perform one IB-LBM step.

    Parameters
    ----------
    state : IBLBMState
    config : IBLBMConfig
    boundary_points : (N, 2), optional
        Lagrangian boundary positions (lattice units).
    target_velocities : (N, 2), optional
        Desired velocity at boundary points (from rigid body).
    ds : float
        Arc length element between Lagrangian points.
    body_center : (2,), optional
        Body centre for torque computation.
    external_force : (nx, ny, 2), optional
        Additional body force (e.g., pressure gradient for Poiseuille).

    Returns
    -------
    new_state : IBLBMState
    ib_result : IBResult or None
    """
    total_force = external_force
    ib_result = None

    # IB coupling via Multi-Direct Forcing
    if boundary_points is not None and target_velocities is not None:
        kernels = compute_kernels(
            boundary_points, config.nx, config.ny, h=1.0,
        )

        corrected_vel, ib_body_force = multi_direct_forcing(
            state.velocity, kernels, target_velocities,
            ds=ds, n_iter=config.mdf_iterations,
        )

        if total_force is None:
            total_force = ib_body_force
        else:
            total_force = total_force + ib_body_force

        marker_f = compute_marker_forces(corrected_vel, kernels, target_velocities)
        drag_f = compute_drag_force(marker_f, ds)

        if body_center is not None:
            drag_t = compute_drag_torque(marker_f, boundary_points, body_center, ds)
        else:
            drag_t = jnp.array(0.0)

        u_markers = interpolate_velocity(corrected_vel, kernels)

        ib_result = IBResult(
            drag_force=drag_f, drag_torque=drag_t,
            marker_forces=marker_f, u_markers=u_markers,
        )

    f_new, density, velocity = lbm_step(
        state.f, config.tau,
        wall_mask=state.wall_mask,
        force=total_force,
    )

    new_state = IBLBMState(
        f=f_new, density=density, velocity=velocity,
        wall_mask=state.wall_mask, step_count=state.step_count + 1,
    )

    return new_state, ib_result


def run(
    state: IBLBMState,
    config: IBLBMConfig,
    n_steps: int,
    external_force: jnp.ndarray | None = None,
) -> IBLBMState:
    """Run multiple LBM steps without IB coupling (pure fluid).

    Uses jax.lax.scan for efficiency.
    """
    def scan_step(f, _):
        f_new, density, velocity = lbm_step(
            f, config.tau,
            wall_mask=state.wall_mask,
            force=external_force,
        )
        return f_new, None

    f_final, _ = jax.lax.scan(scan_step, state.f, None, length=n_steps)
    density, velocity = compute_macroscopic(f_final, force=external_force, tau=config.tau)

    return IBLBMState(
        f=f_final, density=density, velocity=velocity,
        wall_mask=state.wall_mask, step_count=state.step_count + n_steps,
    )
