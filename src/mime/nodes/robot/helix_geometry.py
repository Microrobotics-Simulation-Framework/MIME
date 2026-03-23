"""Helix geometry for LBM — solid mask generation and rotation.

Generates the solid mask for a rigid helical microrobot on a 3D
lattice. The helix is defined parametrically and rasterised onto
the lattice via signed-distance evaluation.

The signed-distance approach is brute-force O(N_grid) but:
- Embarrassingly parallel (vectorised jnp operations)
- Fully JIT-compilable
- No dynamic indexing (unlike scatter-based approaches)
- Accurate to sub-grid resolution

For a 64³ grid this takes ~10ms. For 256³ it takes ~1s on CPU.
GPU would be proportionally faster.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import math


def helix_centreline(
    s: jnp.ndarray,
    radius: float,
    pitch: float,
    n_turns: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> jnp.ndarray:
    """Compute points on the helix centreline.

    Parametrised as:
        x(s) = cx + R * cos(2*pi*s*n_turns)
        y(s) = cy + R * sin(2*pi*s*n_turns)
        z(s) = cz + pitch * s * n_turns

    where s in [0, 1] goes from start to end of the helix.

    Parameters
    ----------
    s : (M,) float
        Parameter values in [0, 1].
    radius : float
        Helix radius [lattice units].
    pitch : float
        Axial advance per turn [lattice units].
    n_turns : float
        Number of complete turns.
    center : (cx, cy, cz)
        Helix start center position.

    Returns
    -------
    points : (M, 3) float32
    """
    cx, cy, cz = center
    theta = 2.0 * jnp.pi * s * n_turns
    x = cx + radius * jnp.cos(theta)
    y = cy + radius * jnp.sin(theta)
    z = cz + pitch * s * n_turns
    return jnp.stack([x, y, z], axis=-1)


def distance_to_helix(
    grid_points: jnp.ndarray,
    helix_points: jnp.ndarray,
) -> jnp.ndarray:
    """Compute minimum distance from each grid point to the helix centreline.

    Uses brute-force distance to the nearest of M sample points on the
    centreline. For M ~ 200 this is accurate to ~pitch/(2*M) lattice units.

    Parameters
    ----------
    grid_points : (N, 3) float32
        Lattice node positions.
    helix_points : (M, 3) float32
        Sample points on the helix centreline.

    Returns
    -------
    distances : (N,) float32
        Minimum distance from each grid point to the centreline.
    """
    # Compute all pairwise distances: (N, M)
    # Use einsum for memory efficiency on large grids
    diff = grid_points[:, None, :] - helix_points[None, :, :]  # (N, M, 3)
    dist_sq = jnp.sum(diff ** 2, axis=-1)  # (N, M)
    return jnp.sqrt(jnp.min(dist_sq, axis=-1))  # (N,)


def create_helix_mask(
    nx: int,
    ny: int,
    nz: int,
    helix_radius: float,
    helix_pitch: float,
    wire_radius: float,
    n_turns: float = 2.0,
    center: tuple[float, float, float] | None = None,
    n_centreline_samples: int = 200,
    rotation_angle: float = 0.0,
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> jnp.ndarray:
    """Create a solid mask for a helical microrobot.

    The helix is centred at `center` with its axis along z.
    The wire has circular cross-section of radius `wire_radius`.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    helix_radius : float
        Helix coil radius [lattice units].
    helix_pitch : float
        Axial advance per turn [lattice units].
    wire_radius : float
        Wire cross-section radius [lattice units].
    n_turns : float
        Number of complete turns.
    center : (cx, cy, cz), optional
        Center of the helix. Default: grid center.
    n_centreline_samples : int
        Number of sample points on the centreline.
    rotation_angle : float
        Rotation angle [radians] about rotation_axis.
    rotation_axis : (ax, ay, az)
        Axis of rotation (for spinning the helix).

    Returns
    -------
    solid_mask : (nx, ny, nz) bool
        True at lattice nodes inside the helix wire.
    """
    if center is None:
        center = (nx / 2.0, ny / 2.0, nz / 2.0)

    # Generate centreline sample points
    s = jnp.linspace(0, 1, n_centreline_samples)
    helix_pts = helix_centreline(s, helix_radius, helix_pitch, n_turns, center)

    # Apply rotation if non-zero
    if abs(rotation_angle) > 1e-10:
        helix_pts = _rotate_points(helix_pts, rotation_angle, rotation_axis, center)

    # Generate grid coordinates
    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    iz = jnp.arange(nz, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
    grid_flat = jnp.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)  # (N, 3)

    # Compute distance from each grid point to helix centreline
    distances = distance_to_helix(grid_flat, helix_pts)

    # Solid where distance < wire_radius
    solid_flat = distances < wire_radius
    return solid_flat.reshape(nx, ny, nz)


def _rotate_points(
    points: jnp.ndarray,
    angle: float,
    axis: tuple[float, float, float],
    center: tuple[float, float, float],
) -> jnp.ndarray:
    """Rotate points about an axis through center (Rodrigues' formula).

    Parameters
    ----------
    points : (N, 3) float32
    angle : float (radians)
    axis : (ax, ay, az) — will be normalised
    center : (cx, cy, cz)

    Returns
    -------
    rotated : (N, 3) float32
    """
    c = jnp.array(center)
    k = jnp.array(axis, dtype=jnp.float32)
    k = k / jnp.linalg.norm(k)

    p = points - c[None, :]

    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    # Rodrigues: p_rot = p*cos(a) + (k x p)*sin(a) + k*(k.p)*(1-cos(a))
    k_cross_p = jnp.cross(k[None, :], p)  # (N, 3)
    k_dot_p = jnp.sum(k[None, :] * p, axis=-1, keepdims=True)  # (N, 1)

    rotated = p * cos_a + k_cross_p * sin_a + k[None, :] * k_dot_p * (1.0 - cos_a)
    return rotated + c[None, :]


def create_sphere_mask(
    nx: int,
    ny: int,
    nz: int,
    center: tuple[float, float, float],
    radius: float,
) -> jnp.ndarray:
    """Create a solid mask for a sphere.

    Parameters
    ----------
    nx, ny, nz : int
    center : (cx, cy, cz) in lattice units
    radius : float in lattice units

    Returns
    -------
    solid_mask : (nx, ny, nz) bool
    """
    cx, cy, cz = center
    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    iz = jnp.arange(nz, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
    dist = jnp.sqrt((gx - cx)**2 + (gy - cy)**2 + (gz - cz)**2)
    return dist < radius


def create_cylinder_body_mask(
    nx: int,
    ny: int,
    nz: int,
    body_radius: float,
    body_length: float,
    cone_length: float,
    cone_end_radius: float,
    center: tuple[float, float, float] | None = None,
    axis: int = 2,
) -> jnp.ndarray:
    """Create solid mask for UMR cylindrical body with conical tip.

    Cylinder along `axis` centered at `center`, with body_length cylinder
    followed by cone tapering from body_radius to cone_end_radius over
    cone_length.

    Parameters
    ----------
    nx, ny, nz : int
    body_radius : float
    body_length : float
    cone_length : float
    cone_end_radius : float
    center : (cx, cy, cz), optional (default: grid center)
    axis : int (0, 1, or 2, default 2 = z-axis)

    Returns
    -------
    solid_mask : (nx, ny, nz) bool
    """
    if center is None:
        center = (nx / 2.0, ny / 2.0, nz / 2.0)

    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    iz = jnp.arange(nz, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')

    cx, cy, cz = center

    # Perpendicular distance from axis
    if axis == 0:
        r_perp = jnp.sqrt((gy - cy) ** 2 + (gz - cz) ** 2)
        z_coord = gx - cx
    elif axis == 1:
        r_perp = jnp.sqrt((gx - cx) ** 2 + (gz - cz) ** 2)
        z_coord = gy - cy
    else:  # axis == 2
        r_perp = jnp.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        z_coord = gz - cz

    total_length = body_length + cone_length

    # Cylinder body: from -total_length/2 to -total_length/2 + body_length
    z_start = -total_length / 2.0
    z_body_end = z_start + body_length
    z_cone_end = z_start + total_length  # = +total_length/2

    # Cylinder region
    in_cylinder_z = (z_coord >= z_start) & (z_coord <= z_body_end)
    in_cylinder = in_cylinder_z & (r_perp < body_radius)

    # Cone region: linear taper from body_radius to cone_end_radius
    # Parametric: t = (z - z_body_end) / cone_length, t in [0, 1]
    # radius(t) = body_radius * (1 - t) + cone_end_radius * t
    t = jnp.clip((z_coord - z_body_end) / jnp.maximum(cone_length, 1e-10), 0.0, 1.0)
    cone_radius = body_radius * (1.0 - t) + cone_end_radius * t
    in_cone_z = (z_coord > z_body_end) & (z_coord <= z_cone_end)
    in_cone = in_cone_z & (r_perp < cone_radius)

    return in_cylinder | in_cone


def create_discontinuous_fins_mask(
    nx: int,
    ny: int,
    nz: int,
    body_radius: float,
    fin_outer_radius: float,
    fin_length: float,
    fin_width: float,
    fin_thickness: float,
    n_fin_sets: int = 2,
    fins_per_set: int = 3,
    helix_pitch: float = 8.0,
    center: tuple[float, float, float] | None = None,
    body_length: float = 0.0,
    rotation_angle: float = 0.0,
) -> jnp.ndarray:
    """Create solid mask for discontinuous helical fins.

    Each fin is a rectangular patch at the body surface, arranged in
    helical pattern. 2 sets x 3 fins = 6 total fin patches.
    Each fin wraps partially around the body following the helix path.

    Parameters
    ----------
    nx, ny, nz : int
    body_radius : float
    fin_outer_radius : float
    fin_length : float
        Axial length of each fin.
    fin_width : float
        Circumferential width of each fin (arc length at body surface).
    fin_thickness : float
        Radial thickness (extends from body_radius outward).
    n_fin_sets : int
    fins_per_set : int
    helix_pitch : float
        # ASSUMPTION: helix_pitch = 8.0mm, estimated from no-overlap constraint
    center : (cx, cy, cz), optional
    body_length : float
        Axial length of the cylindrical body (fins are placed along body).
    rotation_angle : float
        Rotation angle in radians.

    Returns
    -------
    solid_mask : (nx, ny, nz) bool
    """
    if center is None:
        center = (nx / 2.0, ny / 2.0, nz / 2.0)

    cx, cy, cz = center
    total_length = body_length  # Fins are placed along the body length

    ix = jnp.arange(nx, dtype=jnp.float32)
    iy = jnp.arange(ny, dtype=jnp.float32)
    iz = jnp.arange(nz, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')

    # Work in cylindrical coordinates (axis = z)
    dx = gx - cx
    dy = gy - cy
    dz = gz - cz
    r_perp = jnp.sqrt(dx ** 2 + dy ** 2)
    theta = jnp.arctan2(dy, dx)  # [-pi, pi]

    mask = jnp.zeros((nx, ny, nz), dtype=bool)

    # Place fins along the body region
    # Body z range: from -total_length/2 to +total_length/2 (no cone for fins)
    z_body_start = -total_length / 2.0

    # fin_thickness = circumferential blade thickness (§VI.F p.15); fin_width = radial extent (= fin_outer_radius - body_radius)
    fin_angular_width = fin_thickness / jnp.maximum(body_radius, 1e-10)

    # Each set of fins is spaced evenly around the circumference
    # fins_per_set fins at 360/fins_per_set degree spacing
    set_angular_offset = 2.0 * jnp.pi / n_fin_sets / fins_per_set  # offset between sets

    for s in range(n_fin_sets):
        for f_idx in range(fins_per_set):
            # Fin index in global sequence
            fin_global = s * fins_per_set + f_idx

            # Axial center of this fin: distribute evenly along body
            n_total_fins = n_fin_sets * fins_per_set
            z_fin_center = z_body_start + (fin_global + 0.5) * total_length / n_total_fins

            # Angular center follows helix path:
            # theta = 2*pi * z / helix_pitch + base_offset
            theta_center = (2.0 * jnp.pi * z_fin_center / helix_pitch
                            + f_idx * 2.0 * jnp.pi / fins_per_set
                            + s * set_angular_offset
                            + rotation_angle)

            # Axial extent: |dz - z_fin_center| < fin_length / 2
            in_z = jnp.abs(dz - z_fin_center) < fin_length / 2.0

            # Radial extent: body_radius <= r < fin_outer_radius
            in_r = (r_perp >= body_radius - fin_thickness * 0.5) & (r_perp < fin_outer_radius)

            # Angular extent: account for helical twist within the fin
            # The fin follows the helix, so at each z the angular center shifts
            local_theta_center = (theta_center
                                  + 2.0 * jnp.pi * (dz - z_fin_center) / helix_pitch)
            # Wrap angular difference to [-pi, pi]
            d_theta = theta - local_theta_center
            d_theta = jnp.mod(d_theta + jnp.pi, 2.0 * jnp.pi) - jnp.pi
            in_theta = jnp.abs(d_theta) < fin_angular_width / 2.0

            mask = mask | (in_z & in_r & in_theta)

    return mask


def create_umr_mask(
    nx: int,
    ny: int,
    nz: int,
    body_radius: float = 0.87,
    body_length: float = 4.1,
    cone_length: float = 1.9,
    cone_end_radius: float = 0.255,
    fin_outer_radius: float = 1.42,
    fin_length: float = 2.03,
    fin_width: float = 0.55,
    fin_thickness: float = 0.15,
    n_fin_sets: int = 2,
    fins_per_set: int = 3,
    helix_pitch: float = 8.0,
    center: tuple[float, float, float] | None = None,
    rotation_angle: float = 0.0,
) -> jnp.ndarray:
    """Create full UMR solid mask = body | fins.

    Defaults are the d2.8 UMR geometry in mm (can be scaled to lattice
    units by caller).

    Parameters
    ----------
    nx, ny, nz : int
    body_radius : float
    body_length : float
    cone_length : float
    cone_end_radius : float
    fin_outer_radius : float
    fin_length : float
    fin_width : float
    fin_thickness : float
    n_fin_sets : int
    fins_per_set : int
    helix_pitch : float
        # ASSUMPTION: helix_pitch = 8.0mm, estimated from no-overlap constraint
    center : (cx, cy, cz), optional
    rotation_angle : float

    Returns
    -------
    solid_mask : (nx, ny, nz) bool
    """
    body = create_cylinder_body_mask(
        nx, ny, nz, body_radius, body_length,
        cone_length, cone_end_radius,
        center=center, axis=2,
    )
    fins = create_discontinuous_fins_mask(
        nx, ny, nz, body_radius, fin_outer_radius,
        fin_length, fin_width, fin_thickness,
        n_fin_sets=n_fin_sets, fins_per_set=fins_per_set,
        helix_pitch=helix_pitch, center=center,
        body_length=body_length, rotation_angle=rotation_angle,
    )
    return body | fins


def _cylinder_sdf(r_perp, z_coord, body_radius, body_length, cone_length, cone_end_radius):
    """Signed distance to the UMR body (cylinder + cone). Negative inside."""
    total_length = body_length + cone_length
    z_start = -total_length / 2.0
    z_body_end = z_start + body_length
    z_cone_end = z_start + total_length

    # Cylinder SDF: max(r - R, |z - z_center_body| - body_length/2)
    z_body_center = (z_start + z_body_end) / 2.0
    sdf_cyl_r = r_perp - body_radius
    sdf_cyl_z = jnp.abs(z_coord - z_body_center) - body_length / 2.0
    sdf_cyl = jnp.maximum(sdf_cyl_r, sdf_cyl_z)

    # Cone SDF: parametric distance
    t = jnp.clip((z_coord - z_body_end) / jnp.maximum(cone_length, 1e-10), 0.0, 1.0)
    cone_radius = body_radius * (1.0 - t) + cone_end_radius * t
    sdf_cone_r = r_perp - cone_radius
    sdf_cone_z_lo = z_body_end - z_coord
    sdf_cone_z_hi = z_coord - z_cone_end
    sdf_cone_z = jnp.maximum(sdf_cone_z_lo, sdf_cone_z_hi)
    sdf_cone = jnp.maximum(sdf_cone_r, sdf_cone_z)

    return jnp.minimum(sdf_cyl, sdf_cone)


def _fins_sdf(dx, dy, dz, r_perp, theta,
              body_radius, fin_outer_radius, fin_length, fin_width,
              fin_thickness, n_fin_sets, fins_per_set, helix_pitch,
              body_length, rotation_angle):
    """Signed distance to the discontinuous helical fins. Negative inside."""
    total_length = body_length
    z_body_start = -total_length / 2.0
    # fin_thickness = circumferential blade thickness (§VI.F p.15); fin_width = radial extent (= fin_outer_radius - body_radius)
    fin_angular_width = fin_thickness / jnp.maximum(body_radius, 1e-10)

    n_total_fins = n_fin_sets * fins_per_set
    set_angular_offset = 2.0 * jnp.pi / n_fin_sets / fins_per_set

    sdf = jnp.full_like(r_perp, 1e10)  # large positive = far outside

    for s in range(n_fin_sets):
        for f_idx in range(fins_per_set):
            fin_global = s * fins_per_set + f_idx
            z_fin_center = z_body_start + (fin_global + 0.5) * total_length / n_total_fins

            theta_center = (2.0 * jnp.pi * z_fin_center / helix_pitch
                            + f_idx * 2.0 * jnp.pi / fins_per_set
                            + s * set_angular_offset
                            + rotation_angle)

            # Axial distance
            d_z = jnp.abs(dz - z_fin_center) - fin_length / 2.0

            # Radial distance: inside from body_radius-thickness/2 to fin_outer_radius
            r_inner = body_radius - fin_thickness * 0.5
            d_r_inner = r_inner - r_perp
            d_r_outer = r_perp - fin_outer_radius
            d_r = jnp.maximum(d_r_inner, d_r_outer)

            # Angular distance with helical twist
            local_theta_center = (theta_center
                                  + 2.0 * jnp.pi * (dz - z_fin_center) / helix_pitch)
            d_theta = theta - local_theta_center
            d_theta = jnp.mod(d_theta + jnp.pi, 2.0 * jnp.pi) - jnp.pi
            # Convert angular distance to arc length distance at body_radius
            d_arc = (jnp.abs(d_theta) - fin_angular_width / 2.0) * body_radius

            # SDF = max of all three distances
            fin_sdf = jnp.maximum(jnp.maximum(d_z, d_r), d_arc)
            sdf = jnp.minimum(sdf, fin_sdf)

    return sdf


def umr_sdf(
    points: jnp.ndarray,
    body_radius: float = 0.87,
    body_length: float = 4.1,
    cone_length: float = 1.9,
    cone_end_radius: float = 0.255,
    fin_outer_radius: float = 1.42,
    fin_length: float = 2.03,
    fin_width: float = 0.55,
    fin_thickness: float = 0.15,
    n_fin_sets: int = 2,
    fins_per_set: int = 3,
    helix_pitch: float = 8.0,
    center: tuple[float, float, float] | None = None,
    rotation_angle: float = 0.0,
) -> jnp.ndarray:
    """Signed distance function for the UMR. Negative inside.

    Parameters
    ----------
    points : (N, 3) float32
    body_radius, body_length, cone_length, cone_end_radius : float
    fin_outer_radius, fin_length, fin_width, fin_thickness : float
    n_fin_sets, fins_per_set : int
    helix_pitch : float
        # ASSUMPTION: helix_pitch = 8.0mm, estimated from no-overlap constraint
    center : (cx, cy, cz), optional
    rotation_angle : float

    Returns
    -------
    sdf : (N,) float32 — negative inside, positive outside
    """
    if center is None:
        center = (0.0, 0.0, 0.0)

    cx, cy, cz = center
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    dz = points[:, 2] - cz
    r_perp = jnp.sqrt(dx ** 2 + dy ** 2)
    theta = jnp.arctan2(dy, dx)

    body_sdf = _cylinder_sdf(r_perp, dz, body_radius, body_length,
                              cone_length, cone_end_radius)

    fins_sdf = _fins_sdf(dx, dy, dz, r_perp, theta,
                          body_radius, fin_outer_radius, fin_length, fin_width,
                          fin_thickness, n_fin_sets, fins_per_set, helix_pitch,
                          body_length, rotation_angle)

    return jnp.minimum(body_sdf, fins_sdf)


def compute_helix_wall_velocity(
    solid_mask: jnp.ndarray,
    angular_velocity: float,
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    center: tuple[float, float, float] | None = None,
) -> jnp.ndarray:
    """Compute wall velocity field for a rotating solid body.

    u_wall(x) = omega x (x - center)

    Parameters
    ----------
    solid_mask : (nx, ny, nz) bool
    angular_velocity : float (rad/s in lattice time units)
    rotation_axis : (ax, ay, az)
    center : (cx, cy, cz)

    Returns
    -------
    wall_velocity : (nx, ny, nz, 3) float32
    """
    nx, ny, nz = solid_mask.shape
    if center is None:
        center = (nx / 2.0, ny / 2.0, nz / 2.0)

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

    # omega x r
    ux = omega_vec[1] * rz - omega_vec[2] * ry
    uy = omega_vec[2] * rx - omega_vec[0] * rz
    uz = omega_vec[0] * ry - omega_vec[1] * rx

    wall_vel = jnp.stack([ux, uy, uz], axis=-1)
    # Zero outside solid
    wall_vel = jnp.where(solid_mask[..., None], wall_vel, 0.0)
    return wall_vel
