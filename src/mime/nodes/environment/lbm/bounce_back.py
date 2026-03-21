"""Halfway bounce-back for rigid bodies in D3Q19 with momentum exchange.

Uses the missing_mask pattern (inspired by Autodesk/XLB):
- missing_mask: (Q, nx, ny, nz) bool — True for directions at boundary
  nodes that point into the solid (i.e., the neighboring node in that
  direction is solid). These are the "missing" populations that must be
  reconstructed via bounce-back.
- Bounce-back: f_post[opp_q, x] = f_pre[q, x] at missing links,
  with optional wall velocity correction (Ladd 1994).
- Momentum exchange: force on body = sum over missing links of
  (f_pre[opp_q] + f_post[q]) * e_q, contracted via tensordot.

All operations are vectorised JAX — no Python loops over boundary nodes.

Moving wall support (Ladd):
    f_post[opp_q, x] = f_pre[q, x] + 2 * w_q * rho * (e_q . u_wall) / cs^2
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.lbm.d3q19 import E, W, OPP, CS2, Q


# ── Missing mask computation ─────────────────────────────────────────────

def compute_missing_mask(
    solid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the missing_mask from a solid mask.

    A direction q is "missing" at fluid node x if the neighbor
    x + e_q is solid. These are the directions where the population
    streamed from inside the solid and must be reconstructed.

    Parameters
    ----------
    solid_mask : (nx, ny, nz) bool
        True at solid nodes.

    Returns
    -------
    missing_mask : (Q, nx, ny, nz) bool
        True at (q, x, y, z) if direction q at fluid node (x,y,z) is missing.
    """
    nx, ny, nz = solid_mask.shape
    fluid_mask = ~solid_mask

    # For each direction q, check if the neighbor in that direction is solid
    masks = []
    for q in range(Q):
        ex, ey, ez = int(E[q, 0]), int(E[q, 1]), int(E[q, 2])
        # Shift solid mask by -e_q: if the cell at x+e_q is solid,
        # then direction q at x is missing
        neighbor_solid = jnp.roll(solid_mask, (-ex, -ey, -ez), axis=(0, 1, 2))
        # Missing only at fluid nodes whose neighbor is solid
        masks.append(fluid_mask & neighbor_solid)

    return jnp.stack(masks, axis=0)  # (Q, nx, ny, nz)


# ── Bounce-back application ─────────────────────────────────────────────

def apply_bounce_back(
    f_post_stream: jnp.ndarray,
    f_pre_stream: jnp.ndarray,
    missing_mask: jnp.ndarray,
    solid_mask: jnp.ndarray,
    wall_velocity: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Apply halfway bounce-back with optional moving wall velocity.

    At boundary nodes where missing_mask[q] is True:
        f_post[opp_q, x] = f_pre[q, x]  (static wall)
        f_post[opp_q, x] = f_pre[q, x] + 2*w_q*(e_q.u_wall)/cs^2  (moving wall)

    Parameters
    ----------
    f_post_stream : (nx, ny, nz, Q)
        Distribution functions after streaming.
    f_pre_stream : (nx, ny, nz, Q)
        Distribution functions before streaming (post-collision).
    missing_mask : (Q, nx, ny, nz) bool
        Missing directions at boundary nodes.
    solid_mask : (nx, ny, nz) bool
        Solid nodes (distributions zeroed).
    wall_velocity : (nx, ny, nz, 3) float32, optional
        Wall velocity at each node. Only used at boundary nodes.

    Returns
    -------
    f_corrected : (nx, ny, nz, Q) float32
    """
    opp = jnp.array(OPP)  # (Q,)

    # f_pre with opposite directions: f_pre[..., opp[q]] for each q
    f_pre_opp = f_pre_stream[..., opp]  # (nx, ny, nz, Q)

    # Transpose missing_mask to (nx, ny, nz, Q) for broadcasting
    mm = jnp.moveaxis(missing_mask, 0, -1)  # (nx, ny, nz, Q)

    # Base bounce-back: replace missing directions with opposite pre-stream value
    f_bb = jnp.where(mm, f_pre_opp, f_post_stream)

    # Ladd wall velocity correction
    if wall_velocity is not None:
        e_float = jnp.array(E, dtype=jnp.float32)  # (Q, 3)
        w_arr = jnp.array(W)                        # (Q,)

        # e_q . u_wall for each direction at each node: (nx, ny, nz, Q)
        e_dot_u = wall_velocity @ e_float.T

        # Velocity correction: 2 * w_q * (e_q . u_wall) / cs^2
        correction = 2.0 * w_arr * e_dot_u / CS2  # (nx, ny, nz, Q)

        # Apply correction only at missing links
        f_bb = f_bb + jnp.where(mm, correction, 0.0)

    # Do NOT zero solid nodes — they must retain distributions for
    # correct streaming in the next step. Solid nodes participate in
    # streaming (populations stream from solid to fluid and are then
    # bounce-backed). Zeroing them causes mass leakage.

    return f_bb


# ── Momentum exchange force computation ──────────────────────────────────

def compute_momentum_exchange_force(
    f_pre_collision: jnp.ndarray,
    f_post_stream_bb: jnp.ndarray,
    missing_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute force on the solid body via momentum exchange.

    Following XLB's approach:
        phi_q(x) = f_pre_collision[opp_q, x] + f_post_stream_bb[q, x]
    only at missing links, then:
        F = sum_x sum_q phi_q * e_q  (contracted via tensordot)

    Parameters
    ----------
    f_pre_collision : (nx, ny, nz, Q)
        Pre-collision distributions (= post-stream from previous step).
    f_post_stream_bb : (nx, ny, nz, Q)
        Post-stream + bounce-back distributions.
    missing_mask : (Q, nx, ny, nz) bool
        Missing directions.

    Returns
    -------
    force : (3,) float32
        Total force on the body (lattice units).
    """
    opp = jnp.array(OPP)
    e_float = jnp.array(E, dtype=jnp.float32)  # (Q, 3)

    # phi = f_pre[opp_q] + f_post[q] at each node and direction
    f_pre_opp = f_pre_collision[..., opp]  # (nx, ny, nz, Q)
    phi = f_pre_opp + f_post_stream_bb     # (nx, ny, nz, Q)

    # Mask: only at missing links
    mm = jnp.moveaxis(missing_mask, 0, -1)  # (nx, ny, nz, Q)
    phi = jnp.where(mm, phi, 0.0)

    # Contract with lattice velocities: sum over x,y,z and q
    # phi: (nx, ny, nz, Q), e: (Q, 3)
    # Force = sum_{x,y,z} sum_q phi_q * e_q
    force = jnp.tensordot(phi, e_float, axes=([-1], [0]))  # (nx, ny, nz, 3)
    force = jnp.sum(force, axis=(0, 1, 2))  # (3,)

    return force


def compute_momentum_exchange_torque(
    f_pre_collision: jnp.ndarray,
    f_post_stream_bb: jnp.ndarray,
    missing_mask: jnp.ndarray,
    body_center: jnp.ndarray,
) -> jnp.ndarray:
    """Compute torque on the solid body via momentum exchange.

    T = sum_x sum_q (x - x_center) x (phi_q * e_q)

    Parameters
    ----------
    f_pre_collision : (nx, ny, nz, Q)
    f_post_stream_bb : (nx, ny, nz, Q)
    missing_mask : (Q, nx, ny, nz) bool
    body_center : (3,) float32

    Returns
    -------
    torque : (3,) float32 (lattice units)
    """
    opp = jnp.array(OPP)
    e_float = jnp.array(E, dtype=jnp.float32)

    f_pre_opp = f_pre_collision[..., opp]
    phi = f_pre_opp + f_post_stream_bb

    mm = jnp.moveaxis(missing_mask, 0, -1)
    phi = jnp.where(mm, phi, 0.0)

    # Per-node force: (nx, ny, nz, 3)
    node_force = jnp.tensordot(phi, e_float, axes=([-1], [0]))

    # Position vectors relative to body center
    nx, ny, nz, _ = f_pre_collision.shape
    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    iz = jnp.arange(nz, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
    r = jnp.stack([gx - body_center[0],
                    gy - body_center[1],
                    gz - body_center[2]], axis=-1)  # (nx, ny, nz, 3)

    # Torque = sum r x F
    torque_field = jnp.cross(r, node_force)  # (nx, ny, nz, 3)
    return jnp.sum(torque_field, axis=(0, 1, 2))  # (3,)
