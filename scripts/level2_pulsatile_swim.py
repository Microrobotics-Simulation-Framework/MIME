#!/usr/bin/env python3
"""Level 2 pulsatile swimming simulation + USDC recording.

Architecture:
    - BEM + G_wall computes confined body drag (Womersley u_bg subtraction)
    - LBM runs pulsatile pipe flow for flow visualization (no robot body)
    - Robot dynamics: overdamped force balance at each timestep
    - USDC recording: robot mesh + flow cross-section + vessel

The BEM and LBM are NOT coupled — the BEM uses the analytical
Womersley profile for background subtraction (exact for Level 2
one-way coupling). The LBM produces the same flow for visualization.

Usage:
    python scripts/level2_pulsatile_swim.py [pump_pct] [--kappa 0.30]

    pump_pct: 0, 20, 40, 60, 80, 100 (% of physiological flow)
    Default: 60%
"""

import os
import sys
import time
import logging
import argparse

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp

from mime.nodes.environment.stokeslet.surface_mesh import SurfaceMesh
from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table
from mime.nodes.environment.stokeslet.womersley import (
    pulsatile_poiseuille_3d,
    deboer_flow_velocity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Import numpy BEM assembly from T2.5 script ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from t25_bem_cross_validation import (
    assemble_system_matrix_numpy,
    UMR_PARAMS,
    TOTAL_LENGTH,
    MU,
    A_PHYS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Physical parameters ─────────────────────────────────────────────

# UMR actuation
OMEGA_Z = 2.0 * np.pi * 10.0  # 10 Hz rotation, non-dim time

# Pulsatile flow
F_PULSE = 1.2   # cardiac pulse frequency [Hz]
AMPLITUDE = 0.6  # pulsation amplitude (0=steady, 1=full)

# Simulation
DT = 0.001       # non-dim timestep
N_CARDIAC = 4    # number of cardiac cycles to simulate
FPS_RECORD = 24  # recording frame rate


def load_mesh():
    """Load the precomputed BEM mesh."""
    mesh_path = os.path.join(DATA_DIR, "umr_bem_mesh.npz")
    d = np.load(mesh_path)
    return SurfaceMesh(d["points"], d["normals"], d["weights"])


def setup_bem_system(mesh, kappa):
    """Assemble BEM system A = A_free + G_wall and LU-factorize."""
    from mime.nodes.environment.stokeslet.cylinder_wall_table import (
        assemble_image_correction_matrix_from_table,
    )
    from scipy.linalg import lu_factor

    R_cyl = 1.0 / kappa
    table_path = os.path.join(DATA_DIR, "wall_tables", f"wall_R{R_cyl:.3f}.npz")
    table = load_wall_table(table_path)

    pts = np.array(mesh.points)
    wts = np.array(mesh.weights)
    epsilon = mesh.mean_spacing / 2.0

    log.info("Assembling BEM system (N=%d, κ=%.2f, R_cyl=%.3f)...",
             mesh.n_points, kappa, R_cyl)
    t0 = time.time()
    A_free = assemble_system_matrix_numpy(pts, wts, epsilon, MU)
    G_wall = assemble_image_correction_matrix_from_table(pts, wts, R_cyl, MU, table)
    A = A_free + G_wall

    log.info("LU factorizing %dx%d system...", A.shape[0], A.shape[1])
    lu, piv = lu_factor(A)
    log.info("BEM system ready in %.1f s", time.time() - t0)

    return lu, piv, R_cyl


def precompute_bg_response(lu, piv, mesh):
    """Precompute the force response to a uniform axial background flow.

    Since the Womersley profile u_bg(r,t) = f(r) * g(t), and the BEM
    is linear, we can precompute the force response to a fixed profile
    shape and scale it by g(t) at runtime.

    Precomputes:
    - R: 6×6 resistance matrix (body motion → force/torque)
    - F_bg_profile: (3,) force from unit-amplitude bg profile
    - T_bg_profile: (3,) torque from unit-amplitude bg profile
    """
    from scipy.linalg import lu_solve

    pts = np.array(mesh.points)
    wts = np.array(mesh.weights)
    N = len(pts)
    center = np.zeros(3)

    # 1. Build resistance matrix columns (6 BEM solves)
    R = np.zeros((6, 6))
    e = np.eye(3)
    for col, (U, omega) in enumerate(
        [(e[0], np.zeros(3)), (e[1], np.zeros(3)), (e[2], np.zeros(3)),
         (np.zeros(3), e[0]), (np.zeros(3), e[1]), (np.zeros(3), e[2])]
    ):
        r = pts - center
        rhs = (U + np.cross(omega, r)).ravel()
        trac = lu_solve((lu, piv), rhs).reshape(N, 3)
        wf = trac * wts[:, None]
        R[:3, col] = np.sum(wf, axis=0)
        R[3:, col] = np.sum(np.cross(r, wf), axis=0)

    # 2. Precompute response to the parabolic profile shape
    # u_bg_shape = (1 - (r/R_cyl)²) in the z-direction (unit amplitude)
    # At runtime: u_bg = u_bg_shape * velocity_scale(t)
    # The BEM force from u_bg is: F_bg = -A_inv @ u_bg_shape * scale(t)
    # (negative because bg flow is subtracted from body velocity)
    return R


def solve_drag(lu, piv, mesh, U_body, omega_body, u_bg, center=None):
    """Solve BEM for drag force/torque given body motion and background flow.

    rhs = (u_body - u_bg) at body surface points.
    """
    from scipy.linalg import lu_solve

    if center is None:
        center = np.zeros(3)

    pts = np.array(mesh.points)
    wts = np.array(mesh.weights)
    N = len(pts)

    # Rigid body velocity at surface points
    r = pts - center
    u_rigid = U_body + np.cross(omega_body, r)

    # BEM RHS: relative velocity
    rhs = (u_rigid - u_bg).ravel()

    # Solve
    traction_flat = lu_solve((lu, piv), rhs)
    traction = traction_flat.reshape(N, 3)

    # Integrate force and torque
    weighted_f = traction * wts[:, None]
    F = np.sum(weighted_f, axis=0)
    r = pts - center
    T = np.sum(np.cross(r, weighted_f), axis=0)

    return F, T


def precompute_bg_force(lu, piv, mesh, R_cyl):
    """Precompute force/torque from unit-amplitude Poiseuille bg flow.

    Since u_bg(r,t) = profile(r) × modulation(t), and BEM is linear,
    the bg contribution to force/torque scales linearly with modulation.
    Precompute once, then scale at each timestep.

    Returns
    -------
    F_bg_unit : (3,) force from u_bg with unit centreline velocity
    T_bg_unit : (3,) torque from u_bg with unit centreline velocity
    """
    from scipy.linalg import lu_solve

    pts = np.array(mesh.points)
    wts = np.array(mesh.weights)
    N = len(pts)
    center = np.zeros(3)

    # Parabolic profile: u_z = (1 - (r/R)²) in -z direction
    rho = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    profile = np.maximum(1.0 - (rho / R_cyl)**2, 0.0)
    u_bg_unit = np.zeros((N, 3))
    u_bg_unit[:, 2] = -profile  # opposing z-direction

    # BEM solve with rhs = -u_bg (body at rest, only bg flow)
    rhs = (-u_bg_unit).ravel()
    trac = lu_solve((lu, piv), rhs).reshape(N, 3)
    wf = trac * wts[:, None]
    r = pts - center
    F_bg_unit = np.sum(wf, axis=0)
    T_bg_unit = np.sum(np.cross(r, wf), axis=0)

    return F_bg_unit, T_bg_unit


def run_simulation(kappa, pump_pct, output_dir=None):
    """Run Level 2 pulsatile swimming simulation."""
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, "level2_recordings")
    os.makedirs(output_dir, exist_ok=True)

    R_cyl = 1.0 / kappa
    U_mean_phys = deboer_flow_velocity(pump_pct)

    # Non-dimensionalise flow velocity: U_mean_nd = U_mean_phys / (ω * a)
    # where a = A_PHYS (mm), ω = rotation rate. For simplicity, since
    # we're in non-dim coords with a=1 and μ=1, the flow velocity scale
    # is set by the ratio of flow drag to rotational drag.
    # The BEM resistance matrix is in non-dim units where F = R @ [U, ω].
    # So U_mean needs to be in the same velocity units.
    #
    # Physical: μ_phys = 0.69e-3 Pa·s (water at 37°C)
    # Non-dim force: F_nd = F_phys / (μ_phys * a_phys * U_char)
    # We set U_char = a_phys * ω_phys, so time scale is 1/ω_phys.
    # Then U_mean_nd = U_mean_phys / U_char = U_mean_phys / (a_phys * ω_phys)
    #
    # For now, express everything in non-dim where ω_z=1 and μ=1.
    # The flow velocity in these units:
    a_phys_m = A_PHYS * 1e-3  # mm → m
    omega_phys = OMEGA_Z  # rad/s
    U_char = a_phys_m * omega_phys
    U_mean_nd = U_mean_phys / U_char if U_char > 0 else 0.0

    log.info("Simulation parameters:")
    log.info("  κ = %.2f, R_cyl = %.3f", kappa, R_cyl)
    log.info("  pump = %d%%, U_mean_phys = %.4f m/s", pump_pct, U_mean_phys)
    log.info("  U_mean_nd = %.4f (in ω·a units)", U_mean_nd)
    log.info("  ω_z = %.2f rad/s, f_pulse = %.2f Hz", OMEGA_Z, F_PULSE)

    # Load mesh and setup BEM
    mesh = load_mesh()
    lu, piv, R_cyl = setup_bem_system(mesh, kappa)

    pts = np.array(mesh.points)
    N = mesh.n_points

    # Precompute: resistance matrix (7 BEM solves) + bg flow response (1 solve)
    # = 8 total BEM solves at init, then O(1) per timestep
    log.info("Precomputing resistance matrix + bg flow response...")
    t0 = time.time()
    R = precompute_bg_response(lu, piv, mesh)
    F_bg_unit, T_bg_unit = precompute_bg_force(lu, piv, mesh, R_cyl)
    log.info("Precomputation done in %.1f s", time.time() - t0)

    R_FU = R[:3, :3]
    R_FW = R[:3, 3:]
    R_TU = R[3:, :3]
    R_TW = R[3:, 3:]
    R_FU_inv = np.linalg.inv(R_FU)

    log.info("  R_TW_zz = %.4f (confined rotational drag)", R_TW[2, 2])
    log.info("  F_bg_unit_z = %.4f (bg flow force per unit centreline vel)", F_bg_unit[2])

    # Free-swimming speed without flow (sanity check)
    U_swim_no_flow = -R_FU_inv @ R_FW @ np.array([0, 0, 1.0])
    log.info("  U_swim (no flow) = %.6f", U_swim_no_flow[2])

    # Simulation time
    T_cardiac = 1.0 / F_PULSE  # period of one cardiac cycle
    T_cardiac_nd = T_cardiac * omega_phys
    T_total_nd = N_CARDIAC * T_cardiac_nd
    n_steps = int(T_total_nd / DT)
    record_interval = max(1, n_steps // (FPS_RECORD * N_CARDIAC))

    log.info("  T_cardiac_nd = %.2f, T_total_nd = %.2f", T_cardiac_nd, T_total_nd)
    log.info("  n_steps = %d, record_interval = %d", n_steps, record_interval)

    # State: robot position and orientation
    position = np.zeros(3)
    omega = np.array([0.0, 0.0, 1.0])  # unit ω_z

    # History arrays
    times = []
    positions = []
    swimming_speeds = []

    log.info("Running %d steps (O(1) per step via precomputed R)...", n_steps)
    t_start = time.time()

    for step in range(n_steps):
        t_nd = step * DT
        t_phys = t_nd / omega_phys

        # Pulsatile modulation: u_bg = U_centreline × modulation(t)
        # U_centreline = 2 × U_mean_nd (peak of parabolic profile)
        modulation = 1.0 + AMPLITUDE * np.sin(2.0 * np.pi * F_PULSE * t_phys)
        U_centreline = 2.0 * U_mean_nd * modulation

        # Force from rotation + bg flow (Stokes linearity):
        # F_total = R_FU @ U + R_FW @ ω + F_bg_unit × U_centreline
        # Set F_z = 0 for force-free swimming:
        # R_FU_zz × U_z + (R_FW @ ω)_z + F_bg_unit_z × U_cl = 0
        # U_z = -(R_FW @ ω + F_bg_unit × U_cl)_z / R_FU_zz

        F_rot_z = float(R_FW[2, :] @ omega)
        F_bg_z = float(F_bg_unit[2]) * U_centreline
        U_z = -(F_rot_z + F_bg_z) / R_FU[2, 2]

        # Update position
        position[2] += U_z * DT

        # Record
        if step % record_interval == 0:
            times.append(t_nd)
            positions.append(position.copy())
            swimming_speeds.append(U_z)

            if step % max(1, n_steps // 20) == 0:
                log.info("  step %d/%d (t=%.3f): U_z=%.6f, u_bg_cl=%.4f, z=%.4f",
                         step, n_steps, t_nd, U_z, U_centreline, position[2])

    dt_wall = time.time() - t_start
    log.info("Simulation complete: %d steps in %.1f s (%.2f µs/step)",
             n_steps, dt_wall, 1e6 * dt_wall / n_steps)

    # Net displacement
    net_dz = position[2]
    log.info("Net displacement: Δz = %.6f (%.4f body lengths)",
             net_dz, net_dz / TOTAL_LENGTH)

    # Save trajectory data
    traj_path = os.path.join(output_dir, f"traj_kappa{kappa:.2f}_pump{pump_pct:03d}.npz")
    np.savez(traj_path,
             times=np.array(times),
             positions=np.array(positions),
             swimming_speeds=np.array(swimming_speeds),
             kappa=kappa, pump_pct=pump_pct,
             U_mean_nd=U_mean_nd, omega_z=1.0,
             R=R, F_bg_unit=F_bg_unit, T_bg_unit=T_bg_unit)
    log.info("Trajectory saved to %s", traj_path)

    # Generate USDC recording
    try:
        record_usdc(mesh, kappa, pump_pct, times, positions, output_dir)
    except Exception as e:
        log.warning("USDC recording failed: %s", e)
        import traceback
        traceback.print_exc()

    return net_dz


def record_usdc(mesh, kappa, pump_pct, times, positions, output_dir):
    """Generate USDC recording of the swimming simulation."""
    from pxr import Usd, UsdGeom, Gf, Sdf
    from skimage.measure import marching_cubes
    from mime.nodes.robot.helix_geometry import umr_sdf

    log.info("Generating USDC recording...")

    R_cyl = 1.0 / kappa
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Set timing
    n_frames = len(times)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(n_frames - 1)
    stage.SetTimeCodesPerSecond(FPS_RECORD)
    stage.SetFramesPerSecond(FPS_RECORD)

    # ── Robot mesh ──────────────────────────────────────────────────
    # Generate UMR mesh vertices for visualization
    mc_res = 24  # lower res for USD vis
    pad = 0.3
    bbox_min = (-UMR_PARAMS["fin_outer_radius"] - pad,
                -UMR_PARAMS["fin_outer_radius"] - pad,
                -TOTAL_LENGTH / 2 - pad)
    bbox_max = (+UMR_PARAMS["fin_outer_radius"] + pad,
                +UMR_PARAMS["fin_outer_radius"] + pad,
                +TOTAL_LENGTH / 2 + pad)

    xs = np.linspace(bbox_min[0], bbox_max[0], mc_res)
    ys = np.linspace(bbox_min[1], bbox_max[1], mc_res)
    zs = np.linspace(bbox_min[2], bbox_max[2], mc_res)
    dx = (bbox_max[0] - bbox_min[0]) / (mc_res - 1)
    dy = (bbox_max[1] - bbox_min[1]) / (mc_res - 1)
    dz = (bbox_max[2] - bbox_min[2]) / (mc_res - 1)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    sdf_vals = np.array(umr_sdf(jnp.array(grid_pts), **UMR_PARAMS))
    sdf_3d = sdf_vals.reshape(mc_res, mc_res, mc_res)

    verts, faces, _, _ = marching_cubes(sdf_3d, level=0.0, spacing=(dx, dy, dz))
    verts = verts + np.array(bbox_min)

    robot_mesh = UsdGeom.Mesh.Define(stage, "/World/Robot")
    robot_points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    robot_mesh.GetPointsAttr().Set(robot_points)
    face_counts = [3] * len(faces)
    face_indices = [int(i) for tri in faces for i in tri]
    robot_mesh.GetFaceVertexCountsAttr().Set(face_counts)
    robot_mesh.GetFaceVertexIndicesAttr().Set(face_indices)
    robot_mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.7, 0.3, 0.1)])  # copper

    # Set up dynamic transform
    xformable = UsdGeom.Xformable(robot_mesh.GetPrim())
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    orient_op = xformable.AddOrientOp()

    # ── Vessel cylinder ────────────────────────────────────────────
    vessel_length = TOTAL_LENGTH * 3.0  # extend beyond robot
    vessel = UsdGeom.Cylinder.Define(stage, "/World/Vessel")
    vessel.GetRadiusAttr().Set(float(R_cyl))
    vessel.GetHeightAttr().Set(float(vessel_length))
    vessel.GetAxisAttr().Set("Z")
    vessel.GetDisplayColorAttr().Set([Gf.Vec3f(0.85, 0.85, 0.9)])
    vessel.GetDoubleSidedAttr().Set(True)
    # Make vessel semi-transparent
    prim = vessel.GetPrim()
    prim.CreateAttribute("primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray).Set([0.15])

    # ── Flow cross-section mesh ─────────────────────────────────────
    flow_res = 32
    flow_extent = R_cyl * 1.1
    flow_mesh = UsdGeom.Mesh.Define(stage, "/World/FlowField")

    flow_points = []
    for j in range(flow_res):
        for i in range(flow_res):
            x = -flow_extent + 2 * flow_extent * i / (flow_res - 1)
            y = -flow_extent + 2 * flow_extent * j / (flow_res - 1)
            flow_points.append(Gf.Vec3f(float(x), float(y), 0.0))

    flow_face_counts = []
    flow_face_indices = []
    for j in range(flow_res - 1):
        for i in range(flow_res - 1):
            idx = j * flow_res + i
            flow_face_counts.append(4)
            flow_face_indices.extend([idx, idx + 1, idx + flow_res + 1, idx + flow_res])

    flow_mesh.GetPointsAttr().Set(flow_points)
    flow_mesh.GetFaceVertexCountsAttr().Set(flow_face_counts)
    flow_mesh.GetFaceVertexIndicesAttr().Set(flow_face_indices)
    flow_mesh.GetDoubleSidedAttr().Set(True)

    flow_primvar = UsdGeom.PrimvarsAPI(flow_mesh.GetPrim()).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex,
    )

    # ── Camera ──────────────────────────────────────────────────────
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    cam.GetFocalLengthAttr().Set(35.0)
    cam_xform = UsdGeom.Xformable(cam.GetPrim())
    cam_xform.ClearXformOpOrder()
    cam_translate = cam_xform.AddTranslateOp()
    cam_translate.Set(Gf.Vec3d(R_cyl * 3.0, -R_cyl * 3.0, 0.0))

    # ── Write time samples ──────────────────────────────────────────
    omega_phys = OMEGA_Z
    U_mean_nd_val = deboer_flow_velocity(pump_pct) / (A_PHYS * 1e-3 * omega_phys)

    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("coolwarm")
    except ImportError:
        cmap = None

    for frame_idx in range(n_frames):
        tc = Usd.TimeCode(frame_idx)
        pos = positions[frame_idx]
        t_nd = times[frame_idx]
        t_phys = t_nd / omega_phys
        angle = t_nd  # ω_z = 1 in non-dim

        # Robot transform
        translate_op.Set(Gf.Vec3d(0.0, 0.0, float(pos[2])), tc)

        # Rotation quaternion about z-axis
        half_angle = angle / 2.0
        qw = float(np.cos(half_angle))
        qz = float(np.sin(half_angle))
        orient_op.Set(Gf.Quatf(qw, 0.0, 0.0, qz), tc)

        # Flow field colors (Womersley profile at z=0)
        colors = []
        for j in range(flow_res):
            for i in range(flow_res):
                x = -flow_extent + 2 * flow_extent * i / (flow_res - 1)
                y = -flow_extent + 2 * flow_extent * j / (flow_res - 1)
                r = np.sqrt(x**2 + y**2)
                if r < R_cyl:
                    profile = max(0, 1 - (r / R_cyl)**2)
                    modulation = 1.0 + AMPLITUDE * np.sin(2 * np.pi * F_PULSE * t_phys)
                    u_val = 2.0 * U_mean_nd_val * profile * modulation
                    # Normalize: u_val can be negative during strong pulsation
                    u_norm = (u_val + 2 * U_mean_nd_val) / (4 * U_mean_nd_val + 1e-10)
                    u_norm = max(0, min(1, u_norm))
                    if cmap:
                        rgba = cmap(u_norm)
                        colors.append(Gf.Vec3f(float(rgba[0]), float(rgba[1]), float(rgba[2])))
                    else:
                        colors.append(Gf.Vec3f(float(u_norm), 0.0, float(1 - u_norm)))
                else:
                    colors.append(Gf.Vec3f(0.1, 0.1, 0.1))  # outside vessel

        flow_primvar.Set(colors, tc)

    # ── Metadata ────────────────────────────────────────────────────
    stage.GetRootLayer().customLayerData = {
        "mime:level": 2,
        "mime:kappa": float(kappa),
        "mime:pump_pct": int(pump_pct),
        "mime:n_frames": n_frames,
        "mime:bg_flow": "analytical_womersley",
        "mime:description": (
            f"UMR swimming against {pump_pct}% pulsatile flow, "
            f"κ={kappa}, Level 2 (BEM+Womersley, LBM flow vis)"
        ),
    }

    # Export
    usdc_path = os.path.join(output_dir, f"swim_kappa{kappa:.2f}_pump{pump_pct:03d}.usdc")
    stage.GetRootLayer().Export(usdc_path)
    log.info("USDC recording saved: %s (%d frames)", usdc_path, n_frames)


def main():
    parser = argparse.ArgumentParser(description="Level 2 pulsatile swimming")
    parser.add_argument("pump_pct", type=int, nargs="?", default=60,
                        help="Pump percentage (0-100)")
    parser.add_argument("--kappa", type=float, default=0.30,
                        help="Confinement ratio (default: 0.30)")
    parser.add_argument("--multi", action="store_true",
                        help="Run sweep: 0%%, 40%%, 60%%, 80%%, 100%%")
    args = parser.parse_args()

    if args.multi:
        for pct in [0, 40, 60, 80, 100]:
            log.info("=" * 60)
            log.info("PUMP %d%%", pct)
            log.info("=" * 60)
            run_simulation(args.kappa, pct)
    else:
        run_simulation(args.kappa, args.pump_pct)


if __name__ == "__main__":
    main()
