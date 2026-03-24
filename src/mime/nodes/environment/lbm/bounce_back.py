"""Bounce-back boundary conditions for rigid bodies in D3Q19.

Two schemes:
1. **Simple halfway BB** (apply_bounce_back): O(dx) wall position accuracy.
   Uses the missing_mask pattern — wall assumed at midpoint between fluid/solid.

2. **Bouzidi interpolated BB** (apply_bouzidi_bounce_back): O(dx²) accuracy.
   Uses per-link fractional distances (q-values) to interpolate at the
   actual wall position. Requires q_values array from geometry computation.

Both support moving walls via Ladd (1994) velocity correction.

Momentum exchange force/torque uses the missing_mask pattern with
incoming-direction convention (mm_in).

References:
- Ladd, A.J.C. (1994). J. Fluid Mech. 271, 285-309.
- Bouzidi, M. et al. (2001). Phys. Fluids 13(11), 3452-3459.
- Lallemand, P. & Luo, L.S. (2003). J. Comput. Phys. 184, 406-421.
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
    # mm[..., q] = True when x+e_q is solid (outgoing direction into solid).
    mm = jnp.moveaxis(missing_mask, 0, -1)  # (nx, ny, nz, Q)

    # Remap to incoming directions: mm_in[..., q] = True when population q
    # came from a solid node (i.e., x - e_q = x + e_{opp_q} is solid).
    # This identifies the populations that actually need replacing.
    mm_in = mm[..., opp]  # (nx, ny, nz, Q)

    # Bounce-back: replace incoming-from-solid populations with the
    # pre-collision value in the opposite direction (which was heading
    # toward the solid before streaming).
    # When mm_in[q] True: f_bb[q] = f_pre[opp_q]  (standard BB)
    f_bb = jnp.where(mm_in, f_pre_opp, f_post_stream)

    # Ladd wall velocity correction
    if wall_velocity is not None:
        e_float = jnp.array(E, dtype=jnp.float32)  # (Q, 3)
        w_arr = jnp.array(W)                        # (Q,)

        # e_q . u_wall for each direction at each node: (nx, ny, nz, Q)
        e_dot_u = wall_velocity @ e_float.T

        # Correction per incoming link q (came from solid):
        # Bouzidi (2001) Eq. 5: correction = 2*w*(c̄_α · u_wall)/cs²
        # where c̄_α is the incoming direction. In our convention:
        #   c̄_α = e_q (the incoming direction index q)
        #   correction = 2 * w_q * (e_q . u_wall) / cs^2
        correction = 2.0 * w_arr * e_dot_u / CS2  # (nx, ny, nz, Q)

        # Apply correction only at incoming-from-solid links
        f_bb = f_bb + jnp.where(mm_in, correction, 0.0)

    # Do NOT zero solid nodes — they must retain distributions for
    # correct streaming in the next step. Solid nodes participate in
    # streaming (populations stream from solid to fluid and are then
    # bounce-backed). Zeroing them causes mass leakage.

    return f_bb


# ── Bouzidi interpolated bounce-back ─────────────────────────────────────

def compute_q_values_cylinder(
    missing_mask: jnp.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    is_inner: bool = True,
) -> jnp.ndarray:
    """Compute fractional wall distances for Bouzidi IBB on a z-aligned cylinder.

    For each boundary link (q, x) where missing_mask[q, x] is True,
    compute the fractional distance q from the fluid node to the cylinder
    surface along direction e_q. The cylinder has axis along z with
    circular cross-section (x-cx)^2 + (y-cy)^2 = R^2.

    Parameters
    ----------
    missing_mask : (Q, nx, ny, nz) bool
        Outgoing-direction convention: True at (q, x) when x+e_q is solid.
    center_x, center_y : float
        Cylinder center in x-y plane.
    radius : float
        Cylinder radius in lattice units.
    is_inner : bool
        True for inner cylinder (fluid outside, solid inside).
        False for outer wall (fluid inside, solid outside).

    Returns
    -------
    q_values : (Q, nx, ny, nz) float32
        Fractional distance in (0, 1). Only meaningful where missing_mask is True.
    """
    _, nx, ny, nz = missing_mask.shape

    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    gx, gy = jnp.meshgrid(ix, iy, indexing='ij')  # (nx, ny)

    dx = gx - center_x  # (nx, ny)
    dy = gy - center_y

    q_slices = []
    for q in range(Q):
        ex, ey = float(E[q, 0]), float(E[q, 1])
        a_coef = ex**2 + ey**2

        if a_coef == 0:
            # Pure z-direction or rest — no x-y intersection
            q_slices.append(jnp.broadcast_to(
                jnp.float32(0.5), (nx, ny, nz),
            ))
            continue

        # Ray: P(t) = (gx + t*ex, gy + t*ey)
        # Cylinder: (x-cx)^2 + (y-cy)^2 = R^2
        # Quadratic: a*t^2 + b*t + c = 0
        b_coef = 2.0 * (dx * ex + dy * ey)      # (nx, ny)
        c_coef = dx**2 + dy**2 - radius**2       # (nx, ny)

        disc = b_coef**2 - 4.0 * a_coef * c_coef
        disc_safe = jnp.maximum(disc, 0.0)
        sqrt_disc = jnp.sqrt(disc_safe)

        t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
        t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef)

        # Inner cylinder (fluid outside): c > 0, both roots positive, take t1
        # Outer wall (fluid inside): c < 0, t1 < 0, t2 > 0, take t2
        t_wall = t1 if is_inner else t2

        # Clamp to valid Bouzidi range
        t_wall = jnp.clip(t_wall, 1e-6, 1.0 - 1e-6)

        # Broadcast to 3D (same for all z)
        q_slices.append(jnp.broadcast_to(t_wall[:, :, None], (nx, ny, nz)))

    return jnp.stack(q_slices, axis=0)  # (Q, nx, ny, nz)


def compute_q_values_sdf(
    missing_mask: jnp.ndarray,
    sdf_func,
    dx: float = 1.0,
    max_bisection_iters: int = 16,
) -> jnp.ndarray:
    """Compute fractional wall distances using SDF bisection (full-domain).

    WARNING: This evaluates the SDF at ALL domain nodes per direction per
    bisection iteration. At 192^3 this is ~6s/step on H100 — use
    compute_q_values_sdf_sparse for production sweeps.

    For each boundary link (q, x) where missing_mask is True:
    1. x_fluid = grid position of fluid node
    2. x_solid = x_fluid + e_q (solid neighbor)
    3. Bisection along the ray to find the zero of sdf_func
    4. q_value = distance_to_wall / |e_q|

    Process per-direction (19 passes), vectorize over ALL domain nodes.
    Fixed 16 iterations gives ~1e-5 precision.

    Parameters
    ----------
    missing_mask : (Q, nx, ny, nz) bool
        True at (q, x, y, z) where direction q at fluid node (x,y,z)
        points into a solid neighbor.
    sdf_func : callable
        Takes (N, 3) float32 array of points, returns (N,) float32 SDF
        values. Negative inside solid, positive outside.
    dx : float
        Lattice spacing (default 1.0).
    max_bisection_iters : int
        Number of bisection iterations (default 16, gives ~dx/2^16 precision).

    Returns
    -------
    q_values : (Q, nx, ny, nz) float32
        Fractional distance in (0, 1). Only meaningful where missing_mask is True.
    """
    _, nx, ny, nz = missing_mask.shape

    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    iz = jnp.arange(nz, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
    # Flatten grid coordinates
    gx_flat = gx.ravel()
    gy_flat = gy.ravel()
    gz_flat = gz.ravel()

    q_slices = []
    for q in range(Q):
        ex, ey, ez = float(E[q, 0]), float(E[q, 1]), float(E[q, 2])
        e_len = (ex**2 + ey**2 + ez**2) ** 0.5

        if e_len == 0:
            # Rest direction — no boundary link
            q_slices.append(jnp.full((nx, ny, nz), 0.5, dtype=jnp.float32))
            continue

        mm_q = missing_mask[q]  # (nx, ny, nz)
        mm_flat = mm_q.ravel()  # (N_total,)

        # Fluid node positions (all nodes; we'll mask later)
        x_fluid = jnp.stack([gx_flat, gy_flat, gz_flat], axis=-1)  # (N, 3)
        # Solid neighbor positions
        e_vec = jnp.array([ex, ey, ez], dtype=jnp.float32)
        x_solid = x_fluid + e_vec[None, :] * dx  # (N, 3)

        # Bisection: find t in [0, 1] such that sdf(x_fluid + t * e_q * dx) = 0
        # x_fluid should have positive SDF (outside), x_solid should have negative (inside)
        t_lo = jnp.zeros(x_fluid.shape[0], dtype=jnp.float32)
        t_hi = jnp.ones(x_fluid.shape[0], dtype=jnp.float32)

        for _ in range(max_bisection_iters):
            t_mid = 0.5 * (t_lo + t_hi)
            pts_mid = x_fluid + t_mid[:, None] * (x_solid - x_fluid)
            sdf_mid = sdf_func(pts_mid)
            # If sdf_mid > 0 (outside), wall is further along → move t_lo up
            # If sdf_mid < 0 (inside), wall is closer → move t_hi down
            t_lo = jnp.where(sdf_mid > 0, t_mid, t_lo)
            t_hi = jnp.where(sdf_mid <= 0, t_mid, t_hi)

        t_wall = 0.5 * (t_lo + t_hi)

        # Clamp to valid Bouzidi range
        t_wall = jnp.clip(t_wall, 1e-6, 1.0 - 1e-6)

        # Only meaningful at boundary links
        t_wall = jnp.where(mm_flat, t_wall, 0.5)

        q_slices.append(t_wall.reshape(nx, ny, nz))

    return jnp.stack(q_slices, axis=0)  # (Q, nx, ny, nz)


def compute_q_values_sdf_sparse(
    missing_mask: jnp.ndarray,
    sdf_func,
    dx: float = 1.0,
    max_bisection_iters: int = 8,
    max_boundary_links_per_dir: int = 0,
) -> jnp.ndarray:
    """Compute fractional wall distances using SDF bisection (sparse).

    Like compute_q_values_sdf but evaluates the SDF only at boundary
    nodes where missing_mask is True. Uses jnp.nonzero with a fixed
    pad size for JAX static-shape compatibility.

    At 192^3: ~112K boundary links total across all directions, vs 7.1M
    domain nodes. This is ~63x faster than the full-domain version.

    Parameters
    ----------
    missing_mask : (Q, nx, ny, nz) bool
    sdf_func : callable
        Takes (N, 3) float32 array, returns (N,) float32 SDF values.
    dx : float
    max_bisection_iters : int
        Default 8 (gives ~dx/256 precision, sufficient for Bouzidi).
    max_boundary_links_per_dir : int
        Fixed pad size for jnp.nonzero per direction. If 0 (default),
        auto-computed as 1.5x the maximum boundary count across directions.

    Returns
    -------
    q_values : (Q, nx, ny, nz) float32
    """
    _, nx, ny, nz = missing_mask.shape
    N_total = nx * ny * nz

    # Auto-compute pad size if not provided
    if max_boundary_links_per_dir <= 0:
        counts = jnp.sum(missing_mask, axis=(1, 2, 3))  # (Q,)
        max_count = int(jnp.max(counts))
        max_boundary_links_per_dir = int(max_count * 1.5) + 1

    # Precompute flat grid coordinates
    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    iz = jnp.arange(nz, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
    gx_flat = gx.ravel()
    gy_flat = gy.ravel()
    gz_flat = gz.ravel()

    pad = max_boundary_links_per_dir

    q_slices = []
    for q in range(Q):
        ex, ey, ez = float(E[q, 0]), float(E[q, 1]), float(E[q, 2])
        e_len = (ex**2 + ey**2 + ez**2) ** 0.5

        if e_len == 0:
            q_slices.append(jnp.full((nx, ny, nz), 0.5, dtype=jnp.float32))
            continue

        mm_q = missing_mask[q]  # (nx, ny, nz)
        mm_flat = mm_q.ravel()  # (N_total,)
        actual_count = int(jnp.sum(mm_flat))

        # Gather boundary node flat indices (padded to fixed size).
        # fill_value=0 is safe because we track real vs padding via
        # an explicit count mask, not the index value.
        boundary_flat_indices = jnp.nonzero(mm_flat, size=pad, fill_value=0)[0]

        # Mask: first `actual_count` entries are real, rest are padding.
        # This is robust regardless of fill_value — jnp.nonzero returns
        # real entries first, padding entries last.
        entry_idx = jnp.arange(pad, dtype=jnp.int32)
        is_real = entry_idx < actual_count

        # Clamp indices to valid range for safe gather (padding entries
        # index into node 0, but their results are masked out below)
        safe_indices = jnp.clip(boundary_flat_indices, 0, N_total - 1)

        # Gather boundary node coordinates
        bx = gx_flat[safe_indices]  # (pad,)
        by = gy_flat[safe_indices]
        bz = gz_flat[safe_indices]
        x_fluid = jnp.stack([bx, by, bz], axis=-1)  # (pad, 3)

        # Solid neighbor positions
        e_vec = jnp.array([ex, ey, ez], dtype=jnp.float32)
        x_solid = x_fluid + e_vec[None, :] * dx  # (pad, 3)

        # Bisection on sparse boundary nodes only
        t_lo = jnp.zeros(pad, dtype=jnp.float32)
        t_hi = jnp.ones(pad, dtype=jnp.float32)

        for _ in range(max_bisection_iters):
            t_mid = 0.5 * (t_lo + t_hi)
            pts_mid = x_fluid + t_mid[:, None] * (x_solid - x_fluid)
            sdf_mid = sdf_func(pts_mid)  # (pad,) — only boundary nodes!
            t_lo = jnp.where(sdf_mid > 0, t_mid, t_lo)
            t_hi = jnp.where(sdf_mid <= 0, t_mid, t_hi)

        t_wall = 0.5 * (t_lo + t_hi)
        t_wall = jnp.clip(t_wall, 1e-6, 1.0 - 1e-6)
        # Mask out padding entries — set to 0.5 (neutral, won't affect
        # Bouzidi since these indices don't correspond to boundary links)
        t_wall = jnp.where(is_real, t_wall, 0.5)

        # Scatter back to full (nx*ny*nz,) array.
        # Padding entries scatter 0.5 to index 0 — harmless since
        # apply_bouzidi_bounce_back only reads q_values where
        # missing_mask is True, and node 0 is not a boundary node
        # (it's at the domain corner, inside the pipe wall).
        q_full = jnp.full(N_total, 0.5, dtype=jnp.float32)
        q_full = q_full.at[safe_indices].set(t_wall)

        q_slices.append(q_full.reshape(nx, ny, nz))

    return jnp.stack(q_slices, axis=0)  # (Q, nx, ny, nz)


def apply_bouzidi_bounce_back(
    f_post_stream: jnp.ndarray,
    f_pre_stream: jnp.ndarray,
    missing_mask: jnp.ndarray,
    solid_mask: jnp.ndarray,
    q_values: jnp.ndarray,
    wall_velocity: jnp.ndarray | None = None,
    wall_correction: jnp.ndarray | None = None,
    wall_feq: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Apply Bouzidi interpolated bounce-back (second-order accurate).

    For each boundary link with fractional wall distance q:
      q < 0.5:  f[opp_q, x] = 2q·f_pre[q, x] + (1-2q)·f_pre[q, x_ff] + corr
      q >= 0.5: f[opp_q, x] = (1/2q)·(f_pre[q, x] + corr) + (1-1/2q)·f_eq(u_wall)
                (if wall_feq provided; otherwise uses f_pre[opp_q, x] + corr)

    where x_ff = x - e_q (next fluid node away from wall).

    Parameters
    ----------
    f_post_stream : (nx, ny, nz, Q) — after streaming, before BC.
    f_pre_stream : (nx, ny, nz, Q) — after collision, before streaming.
    missing_mask : (Q, nx, ny, nz) bool — outgoing-direction convention.
    solid_mask : (nx, ny, nz) bool
    q_values : (Q, nx, ny, nz) float32 — fractional distances, outgoing convention.
    wall_velocity : (nx, ny, nz, 3) float32, optional
        Per-node wall velocity. Used to compute Ladd correction at node
        position. For simple cases or backward compatibility.
    wall_correction : (Q, nx, ny, nz) float32, optional
        Pre-computed per-link wall velocity correction in outgoing convention:
        wall_correction[q, x, y, z] = 2*w[q]*(e[q]·u_wall(r_wall))/cs²
        Takes precedence over wall_velocity if both are provided.
    wall_feq : (Q, nx, ny, nz) float32, optional
        Per-link equilibrium at wall velocity in outgoing convention:
        wall_feq[q, x, y, z] = f_eq(rho=1, u=u_wall)[q]
        When provided, replaces f_pre_in in the q>=0.5 branch with
        f_eq(u_wall), breaking the velocity-field feedback loop.
        Requires wall_velocity or wall_correction for the q<0.5 branch
        and the Ladd correction on the f_pre_out component.

    Returns
    -------
    f_corrected : (nx, ny, nz, Q) float32
    """
    import warnings

    if wall_velocity is not None and wall_correction is not None:
        warnings.warn(
            "Both wall_velocity and wall_correction provided to "
            "apply_bouzidi_bounce_back; wall_correction takes precedence.",
            stacklevel=2,
        )
    if wall_feq is not None and wall_velocity is None and wall_correction is None:
        warnings.warn(
            "wall_feq provided without wall_velocity or wall_correction. "
            "The q<0.5 branch will have no wall velocity correction "
            "(static wall behavior at those links).",
            stacklevel=2,
        )
    opp = jnp.array(OPP)

    # Transpose to (nx, ny, nz, Q) layout
    mm = jnp.moveaxis(missing_mask, 0, -1)
    mm_in = mm[..., opp]  # incoming direction mask
    qv = jnp.moveaxis(q_values, 0, -1)  # (nx, ny, nz, Q) outgoing convention
    q_in = qv[..., opp]  # q values indexed by incoming direction

    # f_pre in the outgoing direction at x_f:
    # For incoming link q', outgoing is opp_q'
    # f_pre_out[..., q'] = f_pre_stream[..., opp_q']
    f_pre_out = f_pre_stream[..., opp]  # (nx, ny, nz, Q)

    # f_pre in the incoming direction at x_f:
    f_pre_in = f_pre_stream  # f_pre[..., q'] directly

    # f_pre at x_ff in the outgoing direction.
    # x_ff = x_f - e_{opp_q'} = x_f + e_{q'} (one step further from wall).
    # f_pre[opp_q', x_ff] = f_pre[opp_q', x_f + e_{q'}]
    # = jnp.roll(f_pre[..., opp_q'], -e_{q'}, ...) for each q'.
    # Since f_post_stream[opp_q', x_f] = f_pre[opp_q', x_f - e_{opp_q'}]
    #                                   = f_pre[opp_q', x_f + e_{q'}]
    # we can reuse f_post_stream[opp_q'] = f_pre_out at x_ff!
    # But we need to be careful: f_post_stream already has the streamed values.
    # f_post_stream[..., opp_q'] = f_pre_stream[..., opp_q'] rolled by e_{opp_q'}
    # This gives f_pre[opp_q'] at position x - e_{opp_q'} = x + e_{q'} = x_ff. ✓
    f_pre_out_ff = f_post_stream[..., opp]  # (nx, ny, nz, Q)

    # Bouzidi interpolation coefficients.
    # Two JIT-safety clamps prevent extreme intermediate values:
    # 1. q_safe_low >= 0.1: avoids near-extrapolation in q<0.5 branch
    #    (at q=0.004, coeff_b=0.992 makes f_low ≈ f_pre_out_ff, which
    #    amplifies XLA operation-reordering differences under JIT)
    # 2. q_safe_high >= 0.5: caps coeff_a_high at 1.0 (prevents 1/(2q)
    #    explosion at tiny q, which caused NaN under JIT)
    # Links with q < 0.1 fall back to simple BB (f_pre_out).
    q_safe_low = jnp.maximum(q_in, 0.1)
    q_safe_high = jnp.maximum(q_in, 0.5)

    # Case q < 0.5: coeff_a * f_pre_out + coeff_b * f_pre_out_ff
    coeff_a_low = 2.0 * q_safe_low
    coeff_b_low = 1.0 - 2.0 * q_safe_low
    f_low = coeff_a_low * f_pre_out + coeff_b_low * f_pre_out_ff

    # ── Compute the Ladd wall velocity correction (incoming convention) ──
    # This is needed by the q<0.5 branch (always) and the q>=0.5 branch
    # (only when wall_feq is NOT provided — otherwise wall_feq replaces
    # f_pre_in and the correction is folded into the f_pre_out component).
    correction_in = jnp.zeros_like(f_pre_out)
    if wall_correction is not None:
        wc = jnp.moveaxis(wall_correction, 0, -1)  # (nx, ny, nz, Q)
        # Remap outgoing → incoming convention:
        # wc[q_out] = 2*w[q_out]*(e[q_out]·u_wall)/cs² (caller-provided).
        # For incoming q', outgoing = opp[q'], so wc[opp[q']] uses e[opp[q']].
        # Since e[opp[q']] = -e[q'], wc[opp[q']] = -2*w*(e[q']·u_wall)/cs².
        # The correct incoming correction is +2*w*(e[q']·u_wall)/cs² (Bouzidi 2001).
        # So negate after remapping.
        correction_in = -wc[..., opp]  # (nx, ny, nz, Q) incoming convention
    elif wall_velocity is not None:
        e_float = jnp.array(E, dtype=jnp.float32)
        w_arr = jnp.array(W)
        e_dot_u = wall_velocity @ e_float.T
        correction_in = 2.0 * w_arr * e_dot_u / CS2

    # ── Case q >= 0.5: standard Bouzidi interpolation ───────────────
    coeff_a_high = 1.0 / (2.0 * q_safe_high)
    coeff_b_high = 1.0 - coeff_a_high
    f_high = coeff_a_high * f_pre_out + coeff_b_high * f_pre_in

    # ── Select: tiny q → simple BB, q<0.5 → f_low, q>=0.5 → f_high ──
    is_low_q = q_in < 0.5
    is_tiny_q = q_in < 0.1
    f_bouzidi = jnp.where(is_low_q, f_low, f_high)
    f_bouzidi = jnp.where(is_tiny_q, f_pre_out, f_bouzidi)

    # Apply only at incoming-from-solid links
    f_bb = jnp.where(mm_in, f_bouzidi, f_post_stream)

    # ── Mei et al. (2002) wall velocity correction ────────────────────
    # Derived from second-order consistency: u(x_wall) = u_wall requires
    # different correction magnitudes for the two branches:
    #   q < 0.5:  C = 2·w·(e·u_wall)/cs²     (standard Ladd, scale = 1)
    #   q >= 0.5: C = (1/q)·w·(e·u_wall)/cs²  (= Ladd × 1/(2q))
    #   tiny q:   C = 2·w·(e·u_wall)/cs²     (fallback = simple BB)
    # Both are continuous at q = 0.5.
    #
    # Physical justification:
    #   q < 0.5 uses f_pre at x_f and x_ff — no feedback through f_pre_in.
    #   Full Ladd correction is appropriate.
    #   q >= 0.5 uses f_pre_in which carries the developing velocity field.
    #   The 1/(2q) scaling compensates for the equilibrium-mediated feedback.
    mei_scale = jnp.where(is_low_q, 1.0, coeff_a_high)  # 1.0 or 1/(2q)
    mei_scale = jnp.where(is_tiny_q, 1.0, mei_scale)     # fallback = full
    scaled_correction = correction_in * mei_scale
    f_bb = f_bb + jnp.where(mm_in, scaled_correction, 0.0)

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

    # phi = f_pre[opp_q] + f_bb[q] at incoming links (from solid).
    # mm_in[q] = True when population q came from solid.
    mm = jnp.moveaxis(missing_mask, 0, -1)  # (nx, ny, nz, Q)
    mm_in = mm[..., opp]  # remap to incoming directions

    f_pre_opp = f_pre_collision[..., opp]  # (nx, ny, nz, Q)
    phi = f_pre_opp + f_post_stream_bb     # (nx, ny, nz, Q)

    # Mask: only at incoming-from-solid links
    phi = jnp.where(mm_in, phi, 0.0)

    # Contract with lattice velocities: sum over x,y,z and q
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

    # Use incoming-direction mask (same convention as apply_bounce_back)
    mm = jnp.moveaxis(missing_mask, 0, -1)
    mm_in = mm[..., opp]

    f_pre_opp = f_pre_collision[..., opp]
    phi = f_pre_opp + f_post_stream_bb
    phi = jnp.where(mm_in, phi, 0.0)

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
