#!/usr/bin/env python3
"""de Jongh (2025) confined swimming benchmark.

Usage:
    python scripts/dejongh_benchmark.py tables     # Step 2: precompute wall tables
    python scripts/dejongh_benchmark.py sweep      # Step 4: centered speed sweep
    python scripts/dejongh_benchmark.py offcenter  # Step 6: off-center sweep
    python scripts/dejongh_benchmark.py all        # All steps
"""

import os, sys, time, json, logging
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from mime.nodes.environment.stokeslet.dejongh_geometry import (
    dejongh_umr_surface, dejongh_fl_mesh, dejongh_fw_mesh,
    FL_TABLE, FW_TABLE, FL_L_UMR, R_CYL_DEFAULT,
)
from mime.nodes.environment.stokeslet.cylinder_wall_table import (
    precompute_wall_table, save_wall_table, load_wall_table,
    assemble_image_correction_matrix_from_table,
)
from t25_bem_cross_validation import assemble_system_matrix_numpy
from scipy.linalg import lu_factor, lu_solve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

MU = 1.0  # non-dimensional viscosity
R_CYL_UMR = R_CYL_DEFAULT  # 1.56 mm, UMR cylinder radius (length scale)

# Vessel diameters from paper (inner diameter in mm)
VESSELS = {
    '1/2"':   {"D_mm": 12.70, "R_ves_mm": 6.350},
    '3/8"':   {"D_mm":  9.53, "R_ves_mm": 4.765},
    '1/4"':   {"D_mm":  6.35, "R_ves_mm": 3.175},
    '3/16"':  {"D_mm":  4.76, "R_ves_mm": 2.380},
}

# Non-dimensionalise vessel radii by R_CYL_UMR
for v in VESSELS.values():
    v["R_ves_nd"] = v["R_ves_mm"] / R_CYL_UMR
    v["kappa"] = R_CYL_UMR / v["R_ves_mm"]  # confinement ratio 1/L

# Actuation: 10 Hz
OMEGA_PHYS = 2.0 * np.pi * 10.0  # rad/s

# Design subset for first pass
FL_DESIGNS = [3, 5, 7, 9]  # ν = 1.0, 1.4, 1.8, 2.33
FW_DESIGNS = [1, 3, 5]      # ν = 1.0, 1.4, 2.0

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'dejongh_benchmark')
TABLE_DIR = os.path.join(DATA_DIR, 'wall_tables')

# BEM mesh resolution
N_THETA = 40  # ~3000 pts per mesh — feasible for CPU BEM
N_ZETA = 40


def nd_to_mm_per_s(U_nd):
    """Convert non-dim velocity to mm/s."""
    return U_nd * R_CYL_UMR * OMEGA_PHYS  # mm/s (since R_CYL_UMR in mm)


# ── Step 2: Wall table precomputation ────────────────────────────────

def step_tables():
    """Precompute wall tables for all 4 vessel diameters."""
    os.makedirs(TABLE_DIR, exist_ok=True)

    for vname, vdata in VESSELS.items():
        R_nd = vdata["R_ves_nd"]
        path = os.path.join(TABLE_DIR, f"wall_R{R_nd:.3f}.npz")

        if os.path.exists(path):
            log.info("%s: table exists at %s, skipping", vname, path)
            continue

        log.info("%s: precomputing wall table (R_nd=%.3f, κ=%.3f)...",
                 vname, R_nd, vdata["kappa"])
        t0 = time.time()
        table = precompute_wall_table(
            R_cyl=R_nd, mu=MU,
            n_rho=40, n_dphi=64, n_dz=128,
            L_max_factor=5.0,
            n_max=15, n_k=80, n_phi=64,
            n_jobs=0,
            rho_clustering=1.5,  # critical for off-center accuracy
        )
        dt = time.time() - t0
        save_wall_table(table, path)
        log.info("%s: done in %.1f min, saved to %s (%.0f MB)",
                 vname, dt / 60, path, os.path.getsize(path) / 1e6)


# ── Shared BEM solve ─────────────────────────────────────────────────

def compute_R_matrix(mesh, table, R_ves_nd, offset_xy=(0.0, 0.0)):
    """Compute 6×6 resistance matrix for mesh in vessel, optionally off-center.

    The mesh is in mm. We non-dimensionalise all lengths by R_CYL_UMR
    (1.56 mm) before BEM assembly. The resistance matrix R is in
    non-dimensional units: [F] = μ·a·U, [T] = μ·a²·ω.
    """
    # Non-dimensionalise mesh from mm to R_CYL_UMR units
    pts = np.array(mesh.points) / R_CYL_UMR
    wts = np.array(mesh.weights) / R_CYL_UMR**2  # area scales as length²
    N = len(pts)
    eps = (mesh.mean_spacing / R_CYL_UMR) / 2.0

    # Offset in non-dim units
    offset_nd = np.array([offset_xy[0], offset_xy[1], 0.0])  # already non-dim
    pts_shifted = pts + offset_nd
    center = np.array([offset_xy[0], offset_xy[1], pts[:, 2].mean()])

    # Check body inside vessel
    rho = np.sqrt(pts_shifted[:, 0]**2 + pts_shifted[:, 1]**2)
    if np.any(rho >= R_ves_nd):
        raise ValueError(f"Body outside vessel: max(ρ)={rho.max():.3f} ≥ R_ves={R_ves_nd:.3f}")

    # A_body (translation-invariant — uses unshifted body points)
    t0 = time.time()
    A_body = assemble_system_matrix_numpy(pts, wts, eps, MU)
    log.info("    A_body: %.1fs", time.time() - t0)

    # G_wall (uses shifted points)
    t0 = time.time()
    G_wall = assemble_image_correction_matrix_from_table(
        pts_shifted, wts, R_ves_nd, MU, table)
    log.info("    G_wall: %.1fs", time.time() - t0)
    A = A_body + G_wall
    # Note: do NOT symmetrize A — the BEM matrix A[3i:3j] = w_j·K(x_i,x_j)
    # is intentionally asymmetric (different weights). The resistance matrix R
    # should be symmetric by Stokes reciprocity, but Fourier-Bessel truncation
    # introduces ~1% per-pair error. We symmetrize R after extraction instead.

    # Solve 6 RHS
    e = np.eye(3)
    rhs_cols = []
    for i in range(3):
        vel = np.tile(e[i], N)  # unit translation
        rhs_cols.append(vel)
    for i in range(3):
        r = pts  # body-frame for rotation
        vel = np.cross(e[i], r).ravel()
        rhs_cols.append(vel)

    t0 = time.time()
    lu, piv = lu_factor(A)
    solutions = lu_solve((lu, piv), np.column_stack(rhs_cols))
    log.info("    LU+solve: %.1fs", time.time() - t0)

    R_raw = np.zeros((6, 6))
    for col in range(6):
        trac = solutions[:, col].reshape(N, 3)
        wf = trac * wts[:, None]
        R_raw[:3, col] = np.sum(wf, axis=0)
        r = pts  # body-frame torque
        R_raw[3:, col] = np.sum(np.cross(r, wf), axis=0)

    # Symmetrize R: Stokes reciprocity guarantees R = R^T exactly.
    # Fourier-Bessel truncation + table interpolation introduce ~1% per-pair
    # asymmetry in G_image that accumulates. R_sym is the nearest SPD matrix.
    pre_sym_error = float(np.max(np.abs(R_raw - R_raw.T)))
    R = (R_raw + R_raw.T) / 2.0
    return R, pre_sym_error


def swimming_velocity(R):
    """Extract full 3D force-free swimming velocity from R."""
    R_FU = R[:3, :3]
    R_FW = R[:3, 3:]
    omega = np.array([0.0, 0.0, 1.0])  # unit ω_z
    U = -np.linalg.inv(R_FU) @ R_FW @ omega
    return U


def compute_freespace_R(mesh):
    """Compute 6×6 R for body in INFINITE fluid (no wall correction).

    Just A_body, no G_wall. Used as κ→0 anchor for MLP training.
    """
    pts = np.array(mesh.points) / R_CYL_UMR
    wts = np.array(mesh.weights) / R_CYL_UMR**2
    N = len(pts)
    eps = (mesh.mean_spacing / R_CYL_UMR) / 2.0

    A = assemble_system_matrix_numpy(pts, wts, eps, MU)

    e = np.eye(3)
    rhs_cols = []
    for i in range(3):
        rhs_cols.append(np.tile(e[i], N))
    for i in range(3):
        rhs_cols.append(np.cross(e[i], pts).ravel())

    lu, piv = lu_factor(A)
    solutions = lu_solve((lu, piv), np.column_stack(rhs_cols))

    R_raw = np.zeros((6, 6))
    for col in range(6):
        trac = solutions[:, col].reshape(N, 3)
        wf = trac * wts[:, None]
        R_raw[:3, col] = np.sum(wf, axis=0)
        R_raw[3:, col] = np.sum(np.cross(pts, wf), axis=0)

    pre_sym = float(np.max(np.abs(R_raw - R_raw.T)))
    R = (R_raw + R_raw.T) / 2.0
    return R, pre_sym


def step_freespace_anchors():
    """T2.7 Step 1: free-space anchors for MLP training (κ→0 limit)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {}

    for fid in FL_DESIGNS:
        mesh = dejongh_fl_mesh(fid, n_theta=N_THETA, n_zeta=N_ZETA)
        nu_val = FL_TABLE[fid]["nu"]
        log.info("FL-%d (ν=%.3f, N=%d) free-space...", fid, nu_val, mesh.n_points)
        t0 = time.time()
        R, pre = compute_freespace_R(mesh)
        dt_ = time.time() - t0
        U = swimming_velocity(R)
        v_z = nd_to_mm_per_s(U[2])
        log.info("  R in %.1fs, v_z=%.3f mm/s, pre_sym=%.2e", dt_, v_z, pre)

        key = f"FL-{fid}_freespace"
        results[key] = {
            "design": f"FL-{fid}", "vessel": "freespace",
            "kappa": 0.0, "R_ves_nd": 1e6,
            "v_z_mm_s": float(v_z),
            "v_lateral_mm_s": float(nd_to_mm_per_s(np.sqrt(U[0]**2 + U[1]**2))),
            "U_nd": [float(x) for x in U],
            "reciprocity_error": 0.0, "pre_symmetrize_error": pre,
            "positive_definite": bool(np.all(np.linalg.eigvalsh(R) > 0)),
            "R_TW_zz": float(R[5, 5]),
            "R_FU_diag": [float(R[i, i]) for i in range(3)],
            "compute_time_s": dt_,
            "R_matrix": R.tolist(),
            "nu": nu_val, "L_UMR_mm": FL_L_UMR,
            "offset_x_nd": 0.0, "offset_y_nd": 0.0,
            "min_gap_nd": 1e6, "log_min_gap": 5.0,  # "far from wall"
            "N_mesh": mesh.n_points,
            "epsilon_bem": float(mesh.mean_spacing / R_CYL_UMR / 2.0),
        }

    for fid in FW_DESIGNS:
        mesh = dejongh_fw_mesh(fid, n_theta=N_THETA, n_zeta=N_ZETA)
        params = FW_TABLE[fid]
        log.info("FW-%d (ν=%.3f, L=%.2f, N=%d) free-space...",
                 fid, params["nu"], params["L_UMR"], mesh.n_points)
        t0 = time.time()
        R, pre = compute_freespace_R(mesh)
        dt_ = time.time() - t0
        U = swimming_velocity(R)
        v_z = nd_to_mm_per_s(U[2])
        log.info("  R in %.1fs, v_z=%.3f mm/s, pre_sym=%.2e", dt_, v_z, pre)

        key = f"FW-{fid}_freespace"
        results[key] = {
            "design": f"FW-{fid}", "vessel": "freespace",
            "kappa": 0.0, "R_ves_nd": 1e6,
            "v_z_mm_s": float(v_z),
            "v_lateral_mm_s": float(nd_to_mm_per_s(np.sqrt(U[0]**2 + U[1]**2))),
            "U_nd": [float(x) for x in U],
            "reciprocity_error": 0.0, "pre_symmetrize_error": pre,
            "positive_definite": bool(np.all(np.linalg.eigvalsh(R) > 0)),
            "R_TW_zz": float(R[5, 5]),
            "R_FU_diag": [float(R[i, i]) for i in range(3)],
            "compute_time_s": dt_,
            "R_matrix": R.tolist(),
            "nu": params["nu"], "L_UMR_mm": params["L_UMR"],
            "offset_x_nd": 0.0, "offset_y_nd": 0.0,
            "min_gap_nd": 1e6, "log_min_gap": 5.0,
            "N_mesh": mesh.n_points,
            "epsilon_bem": float(mesh.mean_spacing / R_CYL_UMR / 2.0),
        }

    with open(f'{DATA_DIR}/swimming_speeds_freespace.json', 'w') as f:
        json.dump(results, f, indent=2)
    log.info("Free-space anchors saved: %d configs", len(results))


# ── Step 4: Centered speed sweep ────────────────────────────────────

def step_sweep():
    """Compute swimming speed for all designs × vessels (centered)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {}

    # Generate meshes (cached)
    meshes = {}
    for fid in FL_DESIGNS:
        name = f"FL-{fid}"
        meshes[name] = dejongh_fl_mesh(fid, n_theta=N_THETA, n_zeta=N_ZETA)
        log.info("Mesh %s: N=%d", name, meshes[name].n_points)
    for fid in FW_DESIGNS:
        name = f"FW-{fid}"
        meshes[name] = dejongh_fw_mesh(fid, n_theta=N_THETA, n_zeta=N_ZETA)
        log.info("Mesh %s: N=%d", name, meshes[name].n_points)

    for vname, vdata in VESSELS.items():
        R_nd = vdata["R_ves_nd"]
        kappa = vdata["kappa"]
        path = os.path.join(TABLE_DIR, f"wall_R{R_nd:.3f}.npz")

        log.info("Loading table for %s (R_nd=%.3f)...", vname, R_nd)
        table = load_wall_table(path)

        for design_name, mesh in meshes.items():
            log.info("  %s × %s (κ=%.3f)...", design_name, vname, kappa)
            t0 = time.time()

            try:
                R, pre_sym_err = compute_R_matrix(mesh, table, R_nd)
            except ValueError as e:
                log.warning("  SKIP: %s", e)
                continue

            U = swimming_velocity(R)
            U_z = U[2]
            U_lat = np.sqrt(U[0]**2 + U[1]**2)
            recip = np.max(np.abs(R - R.T))
            eigvals = np.linalg.eigvalsh(R)
            pd = bool(np.all(eigvals > 0))

            v_z_mm = nd_to_mm_per_s(U_z)
            v_lat_mm = nd_to_mm_per_s(U_lat)
            dt = time.time() - t0

            log.info("    v_z=%.2f mm/s, v_lat=%.4f mm/s, |R-R^T|=%.2e, PD=%s, %.1fs",
                     v_z_mm, v_lat_mm, recip, pd, dt)

            # Compute gap: min distance from any body point to vessel wall
            pts_shifted = np.array(mesh.points) / R_CYL_UMR  # non-dim
            rho_pts = np.sqrt(pts_shifted[:, 0]**2 + pts_shifted[:, 1]**2)
            min_gap = R_nd - rho_pts.max()

            key = f"{design_name}_{vname}"
            results[key] = {
                "design": design_name,
                "vessel": vname,
                "kappa": kappa,
                "R_ves_nd": R_nd,
                "v_z_mm_s": float(v_z_mm),
                "v_lateral_mm_s": float(v_lat_mm),
                "U_nd": [float(x) for x in U],
                "reciprocity_error": float(recip),
                "pre_symmetrize_error": float(pre_sym_err),
                "positive_definite": pd,
                "R_TW_zz": float(R[5, 5]),
                "R_FU_diag": [float(R[i, i]) for i in range(3)],
                "compute_time_s": dt,
                # Surrogate training data
                "R_matrix": R.tolist(),
                "nu": FL_TABLE.get(int(design_name.split('-')[1]), {}).get("nu") if "FL" in design_name else FW_TABLE.get(int(design_name.split('-')[1]), {}).get("nu"),
                "L_UMR_mm": FL_L_UMR if "FL" in design_name else FW_TABLE.get(int(design_name.split('-')[1]), {}).get("L_UMR"),
                "offset_x_nd": 0.0,
                "offset_y_nd": 0.0,
                "min_gap_nd": float(min_gap),
                "log_min_gap": float(np.log(max(min_gap, 1e-10))),
                "N_mesh": mesh.n_points,
                "epsilon_bem": float(mesh.mean_spacing / R_CYL_UMR / 2.0),
            }

    # Summary table
    log.info("\n" + "=" * 90)
    log.info("CENTERED SWIMMING SPEED RESULTS")
    log.info("=" * 90)
    log.info("%-10s  %-6s  %6s  %10s  %10s  %10s  %8s",
             "Design", "Vessel", "κ", "v_z[mm/s]", "v_lat[mm/s]", "|R-R^T|", "PD")

    for key in sorted(results.keys()):
        r = results[key]
        log.info("%-10s  %-6s  %6.3f  %10.2f  %10.4f  %10.2e  %8s",
                 r["design"], r["vessel"], r["kappa"],
                 r["v_z_mm_s"], r["v_lateral_mm_s"],
                 r["reciprocity_error"], r["positive_definite"])

    log.info("=" * 90)

    # Save
    out_path = os.path.join(DATA_DIR, "swimming_speeds_centered.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)


# ── Step 6: Off-center sweep ────────────────────────────────────────

def step_offcenter():
    """Compute off-center swimming speed for FL-3 and FL-9."""
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {}

    meshes = {
        "FL-3": dejongh_fl_mesh(3, n_theta=N_THETA, n_zeta=N_ZETA),
        "FL-9": dejongh_fl_mesh(9, n_theta=N_THETA, n_zeta=N_ZETA),
    }

    # Max feasible offset = R_ves - R_max_UMR (body must stay inside)
    R_max_UMR = R_CYL_DEFAULT * (1 + 0.33)  # 2.08 mm
    R_max_nd = R_max_UMR / R_CYL_UMR  # non-dim

    for vname, vdata in VESSELS.items():
        R_nd = vdata["R_ves_nd"]
        max_offset_nd = R_nd - R_max_nd - 0.05  # small safety margin
        if max_offset_nd <= 0:
            log.info("%s: no room for off-center (R_ves_nd=%.3f, R_max_nd=%.3f)", vname, R_nd, R_max_nd)
            continue

        path = os.path.join(TABLE_DIR, f"wall_R{R_nd:.3f}.npz")
        table = load_wall_table(path)

        # Offset fractions: 0, 0.1, 0.2, 0.3 of R_ves (or max feasible)
        offset_fracs = [0.0, 0.1, 0.2, 0.3]
        offsets_nd = [f * R_nd for f in offset_fracs if f * R_nd <= max_offset_nd]

        for design_name, mesh in meshes.items():
            for off_nd in offsets_nd:
                off_frac = off_nd / R_nd
                log.info("  %s × %s, offset=%.2f R_ves (%.3f nd)...",
                         design_name, vname, off_frac, off_nd)
                t0 = time.time()

                try:
                    R, pre_sym_err = compute_R_matrix(mesh, table, R_nd, offset_xy=(off_nd, 0.0))
                except ValueError as e:
                    log.warning("    SKIP: %s", e)
                    continue

                U = swimming_velocity(R)
                U_z = U[2]
                U_lat = np.sqrt(U[0]**2 + U[1]**2)
                recip = np.max(np.abs(R - R.T))

                v_z_mm = nd_to_mm_per_s(U_z)
                v_lat_mm = nd_to_mm_per_s(U_lat)
                drift_deg = np.degrees(np.arctan2(U[1], U[0]))
                dt = time.time() - t0

                log.info("    v_z=%.2f, v_lat=%.4f, drift=%.1f°, |R-R^T|=%.2e, %.1fs",
                         v_z_mm, v_lat_mm, drift_deg, recip, dt)

                # Compute gap for off-center body
                pts_oc = np.array(mesh.points) / R_CYL_UMR + np.array([off_nd, 0.0, 0.0])
                rho_oc = np.sqrt(pts_oc[:, 0]**2 + pts_oc[:, 1]**2)
                min_gap = R_nd - rho_oc.max()
                nu_val = FL_TABLE.get(int(design_name.split('-')[1]), {}).get("nu")

                key = f"{design_name}_{vname}_off{off_frac:.2f}"
                results[key] = {
                    "design": design_name,
                    "vessel": vname,
                    "kappa": vdata["kappa"],
                    "offset_frac": float(off_frac),
                    "offset_nd": float(off_nd),
                    "v_z_mm_s": float(v_z_mm),
                    "v_lateral_mm_s": float(v_lat_mm),
                    "drift_angle_deg": float(drift_deg),
                    "reciprocity_error": float(recip),
                "pre_symmetrize_error": float(pre_sym_err),
                    "R_FU_diag": [float(R[i, i]) for i in range(3)],
                    "R_FU_xx_over_yy": float(R[0, 0] / R[1, 1]) if R[1, 1] > 0 else float('nan'),
                    "compute_time_s": dt,
                    # Surrogate training data
                    "R_matrix": R.tolist(),
                    "nu": nu_val,
                    "L_UMR_mm": FL_L_UMR,
                    "offset_x_nd": float(off_nd),
                    "offset_y_nd": 0.0,
                    "min_gap_nd": float(min_gap),
                    "log_min_gap": float(np.log(max(min_gap, 1e-10))),
                    "N_mesh": len(mesh.points),
                    "epsilon_bem": float(mesh.mean_spacing / R_CYL_UMR / 2.0),
                }

    out_path = os.path.join(DATA_DIR, "swimming_speeds_offcenter.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info("Off-center results saved to %s", out_path)


# ── Step bonus: LHS sampling for surrogate training ──────────────────

def step_lhs(n_samples=30, time_budget_s=2700):
    """Generate additional configs via Latin Hypercube Sampling.

    Fills gaps in the (ν, κ) parameter space. Only centered configs
    (off-center adds complexity and requires reciprocity-fix first).
    Stops when time budget exhausted.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {}

    # LHS in (ν, vessel_index) space
    # ν: continuous in [0.5, 3.0] (covers FL range)
    # vessel: pick from the 4 available tables
    rng = np.random.default_rng(42)

    # Simple LHS: stratified random in each dimension
    nu_bins = np.linspace(0.5, 3.0, n_samples + 1)
    nu_samples = np.array([rng.uniform(nu_bins[i], nu_bins[i+1]) for i in range(n_samples)])
    rng.shuffle(nu_samples)

    vessel_keys = list(VESSELS.keys())
    vessel_indices = np.arange(n_samples) % len(vessel_keys)
    rng.shuffle(vessel_indices)

    log.info("LHS sampling: %d configs, time budget %ds", n_samples, time_budget_s)
    t_start = time.time()
    n_done = 0

    for idx in range(n_samples):
        if time.time() - t_start > time_budget_s:
            log.info("Time budget exhausted after %d/%d configs", n_done, n_samples)
            break

        nu = float(nu_samples[idx])
        vname = vessel_keys[vessel_indices[idx]]
        vdata = VESSELS[vname]
        R_nd = vdata["R_ves_nd"]
        kappa = vdata["kappa"]

        table_path = os.path.join(TABLE_DIR, f"wall_R{R_nd:.3f}.npz")
        if not os.path.exists(table_path):
            log.info("  LHS-%03d: skip (no table for %s)", idx, vname)
            continue

        table = load_wall_table(table_path)

        # Generate mesh at this ν (FL-style, fixed length)
        mesh = dejongh_umr_surface(nu, FL_L_UMR, n_theta=N_THETA, n_zeta=N_ZETA)

        log.info("  LHS-%03d: ν=%.3f × %s (κ=%.3f)...", idx, nu, vname, kappa)
        t0 = time.time()

        try:
            R, pre_sym_err = compute_R_matrix(mesh, table, R_nd)
        except ValueError as e:
            log.warning("    SKIP: %s", e)
            continue

        U = swimming_velocity(R)
        recip = np.max(np.abs(R - R.T))
        v_z = nd_to_mm_per_s(U[2])
        dt = time.time() - t0
        n_done += 1

        pts_nd = np.array(mesh.points) / R_CYL_UMR
        rho_pts = np.sqrt(pts_nd[:, 0]**2 + pts_nd[:, 1]**2)
        min_gap = R_nd - rho_pts.max()

        log.info("    v_z=%.2f mm/s, |R-R^T|=%.2e, %.1fs", v_z, recip, dt)

        key = f"LHS-{idx:03d}_nu{nu:.3f}_{vname}"
        results[key] = {
            "design": f"LHS-{idx:03d}",
            "vessel": vname,
            "kappa": kappa,
            "R_ves_nd": R_nd,
            "v_z_mm_s": float(v_z),
            "v_lateral_mm_s": float(nd_to_mm_per_s(np.sqrt(U[0]**2 + U[1]**2))),
            "U_nd": [float(x) for x in U],
            "reciprocity_error": float(recip),
                "pre_symmetrize_error": float(pre_sym_err),
            "positive_definite": bool(np.all(np.linalg.eigvalsh(R) > 0)),
            "compute_time_s": dt,
            "R_matrix": R.tolist(),
            "nu": nu,
            "L_UMR_mm": FL_L_UMR,
            "offset_x_nd": 0.0,
            "offset_y_nd": 0.0,
            "min_gap_nd": float(min_gap),
            "log_min_gap": float(np.log(max(min_gap, 1e-10))),
            "N_mesh": mesh.n_points,
            "epsilon_bem": float(mesh.mean_spacing / R_CYL_UMR / 2.0),
        }

    elapsed = time.time() - t_start
    log.info("LHS complete: %d configs in %.0f min", n_done, elapsed / 60)

    out_path = os.path.join(DATA_DIR, "swimming_speeds_lhs.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info("LHS results saved to %s", out_path)


def step_coverage_report():
    """Print parameter space coverage from all collected data."""
    all_configs = []
    for fname in ["swimming_speeds_centered.json", "swimming_speeds_offcenter.json", "swimming_speeds_lhs.json"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                all_configs.append(v)

    if not all_configs:
        log.info("No data files found for coverage report")
        return

    nus = [c.get("nu", 0) for c in all_configs if c.get("nu") is not None]
    kappas = [c["kappa"] for c in all_configs]
    offsets = [c.get("offset_x_nd", 0) for c in all_configs]
    gaps = [c.get("min_gap_nd", 0) for c in all_configs if c.get("min_gap_nd")]
    recips = [c["reciprocity_error"] for c in all_configs]

    log.info("\n" + "=" * 60)
    log.info("PARAMETER SPACE COVERAGE REPORT")
    log.info("=" * 60)
    log.info("Total configs: %d", len(all_configs))
    log.info("  ν range: [%.3f, %.3f] (%d unique)", min(nus), max(nus), len(set(f"{x:.3f}" for x in nus)))
    log.info("  κ range: [%.3f, %.3f] (%d unique)", min(kappas), max(kappas), len(set(f"{x:.3f}" for x in kappas)))
    log.info("  offset range: [%.3f, %.3f]", min(offsets), max(offsets))
    if gaps:
        log.info("  min_gap range: [%.4f, %.4f]", min(gaps), max(gaps))
        log.info("  log(gap) range: [%.2f, %.2f]", np.log(max(min(gaps), 1e-10)), np.log(max(gaps)))
    log.info("  reciprocity: median=%.2e, max=%.2e", np.median(recips), max(recips))
    n_good = sum(1 for r in recips if r < 0.1)
    log.info("  configs with |R-R^T| < 0.1: %d/%d (%.0f%%)", n_good, len(recips), 100*n_good/len(recips))
    log.info("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "tables":
        step_tables()
    elif mode == "sweep":
        step_sweep()
    elif mode == "offcenter":
        step_offcenter()
    elif mode == "lhs":
        step_lhs()
    elif mode == "freespace":
        step_freespace_anchors()
    elif mode == "coverage":
        step_coverage_report()
    elif mode == "all":
        step_tables()
        step_sweep()
        step_offcenter()
        step_lhs(n_samples=30, time_budget_s=2700)  # 45 min bonus
        step_coverage_report()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
