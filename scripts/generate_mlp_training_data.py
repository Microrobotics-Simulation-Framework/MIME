#!/usr/bin/env python3
"""Generate dense BEM training data for Cholesky MLP surrogate.

Strategy (target ~310 new configs in 10 hours, ~1 hour on GPU):
- Phase A: fill ν gaps (8 new values × 4 κ + 2 × 2 offsets = 64 configs)
- Phase B: new wall tables at κ = 0.12, 0.40 + sweep (46 configs + 2 tables)
- Phase C: LHS space-filling (150 configs)

Usage:
    python scripts/generate_mlp_training_data.py phaseA
    python scripts/generate_mlp_training_data.py phaseB
    python scripts/generate_mlp_training_data.py phaseC
    python scripts/generate_mlp_training_data.py all
"""

from __future__ import annotations

import os
import sys
import json
import time
import datetime as dt
from pathlib import Path
import numpy as np

# Leave JAX default backend (GPU if available)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

from mime.nodes.environment.stokeslet.dejongh_geometry import (
    dejongh_umr_surface, FL_L_UMR, R_CYL_DEFAULT, EPSILON_DEFAULT, N_STARTS_DEFAULT,
)
from mime.nodes.environment.stokeslet.cylinder_wall_table import (
    load_wall_table, precompute_wall_table, save_wall_table,
)
from mime.nodes.environment.stokeslet.resistance import compute_gcyl_confined_R_gpu

DATA_DIR = Path(__file__).parent.parent / "data" / "dejongh_benchmark"
TABLE_DIR = DATA_DIR / "wall_tables"
OUTPUT = DATA_DIR / "swimming_speeds_dense.json"
PROGRESS = DATA_DIR / "dense_progress.json"

R_CYL_UMR = R_CYL_DEFAULT  # 1.56 mm
MU = 1.0
OMEGA_PHYS = 2 * np.pi * 10.0
R_MAX_UMR_ND = (1.0 + EPSILON_DEFAULT)  # 1.33

# Wall tables: map κ → (R_nd, filename)
VESSELS_EXISTING = {
    0.246: "wall_R4.071.npz",  # 1/2"
    0.327: "wall_R3.054.npz",  # 3/8"
    0.491: "wall_R2.035.npz",  # 1/4"
    0.655: "wall_R1.526.npz",  # 3/16"
}

# Phase B new tables
VESSELS_NEW = {
    0.120: "wall_R8.333.npz",  # R_ves_nd = 1/0.12 = 8.333
    0.400: "wall_R2.500.npz",  # R_ves_nd = 1/0.40 = 2.500
}

# Existing ν (don't duplicate)
EXISTING_NUS_FL = {1.0, 1.4, 1.8, 2.0, 2.33}
EXISTING_NUS_FW = {1.0, 1.4, 2.0}

# Phase A: new FL ν values
NEW_NUS_FL_PHASE_A = [0.5, 0.75, 1.2, 1.6, 2.7, 3.0, 3.5]

N_THETA = 40
N_ZETA = 40


def nd_to_mm_per_s(U_nd):
    return U_nd * R_CYL_UMR * OMEGA_PHYS


def make_mesh_and_wts(nu, L_UMR_mm=FL_L_UMR):
    mesh = dejongh_umr_surface(nu, L_UMR_mm, n_theta=N_THETA, n_zeta=N_ZETA)
    pts = np.array(mesh.points) / R_CYL_UMR
    wts = np.array(mesh.weights) / R_CYL_UMR**2
    eps = (mesh.mean_spacing / R_CYL_UMR) / 2.0
    return mesh, pts, wts, eps


def load_existing():
    all_ = {}
    for f in ["swimming_speeds_centered.json", "swimming_speeds_offcenter.json",
              "swimming_speeds_lhs.json", "swimming_speeds_freespace.json",
              "swimming_speeds_dense.json"]:
        path = DATA_DIR / f
        if path.exists():
            all_.update(json.load(open(path)))
    return all_


def save_dense(results):
    # Merge with any existing dense
    existing = {}
    if OUTPUT.exists():
        existing = json.load(open(OUTPUT))
    existing.update(results)
    with open(OUTPUT, "w") as f:
        json.dump(existing, f, indent=2)


def format_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h > 0 else f"{m}m{s:02d}s"


class ProgressTracker:
    def __init__(self, total_configs, phase_name):
        self.total = total_configs
        self.phase_name = phase_name
        self.done = 0
        self.start = dt.datetime.now()
        self.t_per_config = []

    def log(self, config_label, extra=""):
        now = dt.datetime.now()
        self.done += 1
        elapsed = (now - self.start).total_seconds()
        if self.done > 1:
            self.t_per_config.append(elapsed / self.done)
        avg = elapsed / self.done
        remaining = (self.total - self.done) * avg
        eta = now + dt.timedelta(seconds=remaining)
        msg = (f"[{self.phase_name} {self.done:3d}/{self.total:3d}] {config_label} | "
               f"elapsed: {format_duration(elapsed)} | "
               f"avg: {avg:.1f}s | "
               f"remaining: {format_duration(remaining)} | "
               f"ETA: {eta.strftime('%H:%M:%S')}")
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)

        # Save progress
        with open(PROGRESS, "w") as f:
            json.dump({
                "phase": self.phase_name,
                "done": self.done,
                "total": self.total,
                "elapsed_s": elapsed,
                "eta": eta.isoformat(),
                "avg_seconds_per_config": avg,
            }, f, indent=2)


def run_single_config(nu, L_UMR_mm, kappa, offset_x_nd, offset_y_nd, table):
    """Compute R for one config, return results dict."""
    mesh, pts, wts, eps = make_mesh_and_wts(nu, L_UMR_mm)

    R_ves_nd = table.R_cyl

    # Check feasibility: body must be inside vessel after offset
    pts_shifted = pts + np.array([offset_x_nd, offset_y_nd, 0.0])
    rho_max = np.sqrt(pts_shifted[:, 0]**2 + pts_shifted[:, 1]**2).max()
    if rho_max >= R_ves_nd:
        raise ValueError(f"Body outside vessel: max(ρ)={rho_max:.3f} ≥ R_ves={R_ves_nd:.3f}")

    R, pre_sym = compute_gcyl_confined_R_gpu(
        pts.astype(np.float32),
        wts.astype(np.float32),
        float(eps),
        float(R_ves_nd),
        float(MU),
        table,
        offset_nd=(offset_x_nd, offset_y_nd),
    )

    # Swimming velocity
    R_FU = R[:3, :3]
    R_FW = R[:3, 3:]
    U = -np.linalg.solve(R_FU, R_FW @ np.array([0, 0, 1.0]))
    v_z_mm = nd_to_mm_per_s(U[2])
    v_lat_mm = nd_to_mm_per_s(np.sqrt(U[0]**2 + U[1]**2))

    # Gap calculation
    min_gap_nd = R_ves_nd - rho_max
    log_min_gap = np.log(max(min_gap_nd, 1e-4))

    # Quality
    eigs = np.linalg.eigvalsh(R)
    pd = bool(np.all(eigs > 0))

    return {
        "nu": float(nu),
        "L_UMR_mm": float(L_UMR_mm),
        "kappa": float(1.0 / R_ves_nd),
        "R_ves_nd": float(R_ves_nd),
        "offset_x_nd": float(offset_x_nd),
        "offset_y_nd": float(offset_y_nd),
        "min_gap_nd": float(min_gap_nd),
        "log_min_gap": float(log_min_gap),
        "R_matrix": R.tolist(),
        "U_nd": [float(x) for x in U],
        "v_z_mm_s": float(v_z_mm),
        "v_lateral_mm_s": float(v_lat_mm),
        "reciprocity_error": 0.0,  # symmetrized
        "pre_symmetrize_error": float(pre_sym),
        "positive_definite": pd,
        "min_eigenvalue": float(eigs.min()),
        "N_mesh": mesh.n_points,
        "epsilon_bem": float(eps),
    }


def phase_A():
    """ν gap-filling: 8 new ν × 4 κ centered + 2 κ × 2 offsets."""
    tracker = ProgressTracker(
        len(NEW_NUS_FL_PHASE_A) * 4 + len(NEW_NUS_FL_PHASE_A) * 2 * 2,
        "PhaseA"
    )
    results = {}

    # Load existing tables
    tables = {}
    for kappa, fname in VESSELS_EXISTING.items():
        path = TABLE_DIR / fname
        if not path.exists():
            print(f"WARNING: wall table missing: {path}")
            continue
        tables[kappa] = load_wall_table(str(path))

    centered_kappas = list(VESSELS_EXISTING.keys())  # all 4
    offset_kappas = [0.327, 0.491]  # mid-range κ = 0.33, 0.49
    offset_fracs = [0.1, 0.2]

    for nu in NEW_NUS_FL_PHASE_A:
        # Centered at each existing κ
        for kappa in centered_kappas:
            if kappa not in tables:
                continue
            table = tables[kappa]
            try:
                r = run_single_config(nu, FL_L_UMR, kappa, 0.0, 0.0, table)
                key = f"phaseA_FL_nu{nu:.2f}_k{kappa:.3f}_off0.00"
                results[key] = r
                r["source"] = "phaseA_centered"
                tracker.log(f"ν={nu:.2f} κ={kappa:.3f} off=0.00",
                             f"v_z={r['v_z_mm_s']:.2f} mm/s, pre_sym={r['pre_symmetrize_error']:.1e}")
            except ValueError as e:
                tracker.log(f"ν={nu:.2f} κ={kappa:.3f} off=0.00", f"SKIP: {e}")

        # Off-center at mid κ
        for kappa in offset_kappas:
            if kappa not in tables:
                continue
            table = tables[kappa]
            R_ves_nd = table.R_cyl
            max_off_nd = (R_ves_nd - R_MAX_UMR_ND) * 0.95
            for off_frac in offset_fracs:
                off_nd = off_frac * R_ves_nd
                if off_nd > max_off_nd:
                    tracker.log(f"ν={nu:.2f} κ={kappa:.3f} off={off_frac:.2f}",
                                 "SKIP: offset too large")
                    continue
                try:
                    r = run_single_config(nu, FL_L_UMR, kappa, off_nd, 0.0, table)
                    key = f"phaseA_FL_nu{nu:.2f}_k{kappa:.3f}_off{off_frac:.2f}"
                    r["source"] = "phaseA_offcenter"
                    results[key] = r
                    tracker.log(f"ν={nu:.2f} κ={kappa:.3f} off={off_frac:.2f}",
                                 f"v_z={r['v_z_mm_s']:.2f} v_lat={r['v_lateral_mm_s']:.3f} mm/s")
                except ValueError as e:
                    tracker.log(f"ν={nu:.2f} κ={kappa:.3f} off={off_frac:.2f}", f"SKIP: {e}")

        save_dense(results)  # save after each ν

    print(f"\nPhase A complete: {len(results)} configs saved to {OUTPUT}")


def phase_B():
    """Compute 2 new wall tables + sweep at those κ values."""
    # Check which tables need computing
    tables_to_compute = []
    for kappa, fname in VESSELS_NEW.items():
        path = TABLE_DIR / fname
        if not path.exists():
            R_ves_nd = 1.0 / kappa
            tables_to_compute.append((kappa, R_ves_nd, path))

    # Compute new tables (long-running, uses multiprocessing on CPU)
    for kappa, R_ves_nd, path in tables_to_compute:
        print(f"\nComputing wall table κ={kappa} (R_ves_nd={R_ves_nd:.3f})...")
        t0 = dt.datetime.now()
        table = precompute_wall_table(
            R_cyl=R_ves_nd, mu=MU,
            n_rho=40, n_dphi=64, n_dz=128,
            L_max_factor=5.0,
            n_max=15, n_k=80, n_phi=64,
            n_jobs=0,
            rho_clustering=1.5,
        )
        save_wall_table(table, str(path))
        elapsed = dt.datetime.now() - t0
        print(f"  saved {path.name} ({path.stat().st_size/1e6:.0f} MB) in {elapsed}")

    # Now sweep all ν × new κ
    all_nus_fl = sorted(EXISTING_NUS_FL | set(NEW_NUS_FL_PHASE_A))  # 12 total
    sweep_configs = len(all_nus_fl) * len(VESSELS_NEW)  # centered
    sweep_configs += 4 * len(VESSELS_NEW) * 2  # 4 ν × 2 κ × 2 offsets
    tracker = ProgressTracker(sweep_configs, "PhaseB")
    results = {}

    tables_new = {}
    for kappa, fname in VESSELS_NEW.items():
        path = TABLE_DIR / fname
        tables_new[kappa] = load_wall_table(str(path))

    for nu in all_nus_fl:
        for kappa, table in tables_new.items():
            try:
                r = run_single_config(nu, FL_L_UMR, kappa, 0.0, 0.0, table)
                key = f"phaseB_FL_nu{nu:.2f}_k{kappa:.3f}_off0.00"
                r["source"] = "phaseB_centered"
                results[key] = r
                tracker.log(f"ν={nu:.2f} κ={kappa:.3f} off=0.00",
                             f"v_z={r['v_z_mm_s']:.2f} mm/s")
            except ValueError as e:
                tracker.log(f"ν={nu:.2f} κ={kappa:.3f} off=0.00", f"SKIP: {e}")

    # Off-center at 4 mid-range ν
    offset_nus = [1.0, 1.4, 2.0, 2.33]
    for nu in offset_nus:
        for kappa, table in tables_new.items():
            R_ves_nd = table.R_cyl
            max_off = (R_ves_nd - R_MAX_UMR_ND) * 0.95
            for off_frac in [0.1, 0.2]:
                off_nd = off_frac * R_ves_nd
                if off_nd > max_off:
                    tracker.log(f"ν={nu:.2f} κ={kappa:.3f} off={off_frac:.2f}",
                                 "SKIP: too large")
                    continue
                try:
                    r = run_single_config(nu, FL_L_UMR, kappa, off_nd, 0.0, table)
                    key = f"phaseB_FL_nu{nu:.2f}_k{kappa:.3f}_off{off_frac:.2f}"
                    r["source"] = "phaseB_offcenter"
                    results[key] = r
                    tracker.log(f"ν={nu:.2f} κ={kappa:.3f} off={off_frac:.2f}",
                                 f"v_z={r['v_z_mm_s']:.2f} mm/s")
                except ValueError as e:
                    tracker.log(f"ν={nu:.2f} κ={kappa:.3f} off={off_frac:.2f}", f"SKIP: {e}")

    save_dense(results)
    print(f"\nPhase B complete: {len(results)} configs saved")


def phase_C(n_samples=150):
    """LHS space-filling in (ν, κ, offset)."""
    # Load all available tables
    tables = {}
    for kappa, fname in list(VESSELS_EXISTING.items()) + list(VESSELS_NEW.items()):
        path = TABLE_DIR / fname
        if path.exists():
            tables[kappa] = load_wall_table(str(path))
    available_kappas = sorted(tables.keys())
    print(f"Available κ values for LHS: {available_kappas}")

    # LHS
    rng = np.random.default_rng(12345)
    # Stratified uniform in each dim
    nu_bins = np.linspace(0.5, 3.5, n_samples + 1)
    kappa_bins = np.linspace(0, 0.66, n_samples + 1)
    off_bins = np.linspace(0, 0.25, n_samples + 1)

    nus = np.array([rng.uniform(nu_bins[i], nu_bins[i+1]) for i in range(n_samples)])
    kappas = np.array([rng.uniform(kappa_bins[i], kappa_bins[i+1]) for i in range(n_samples)])
    offsets = np.array([rng.uniform(off_bins[i], off_bins[i+1]) for i in range(n_samples)])
    rng.shuffle(nus); rng.shuffle(kappas); rng.shuffle(offsets)

    def snap_kappa(k):
        return min(available_kappas, key=lambda x: abs(x - k))

    tracker = ProgressTracker(n_samples, "PhaseC")
    results = {}

    for idx in range(n_samples):
        nu = float(nus[idx])
        requested_k = float(kappas[idx])
        off_frac = float(offsets[idx])
        snapped_k = snap_kappa(requested_k)
        table = tables[snapped_k]
        R_ves_nd = table.R_cyl
        max_off = (R_ves_nd - R_MAX_UMR_ND) * 0.95
        off_nd = min(off_frac * R_ves_nd, max_off)

        # Random direction for offset (LHS over the radial axis only — otherwise we'd waste samples)
        # Use a quick rotation to distribute over full angle
        theta = rng.uniform(0, 2 * np.pi)
        off_x = off_nd * np.cos(theta)
        off_y = off_nd * np.sin(theta)

        try:
            r = run_single_config(nu, FL_L_UMR, snapped_k, off_x, off_y, table)
            key = f"phaseC_LHS{idx:03d}_nu{nu:.2f}_k{snapped_k:.3f}_off{off_frac:.2f}"
            r["source"] = "phaseC_lhs"
            r["requested_kappa"] = requested_k
            r["offset_angle_rad"] = float(theta)
            results[key] = r
            tracker.log(f"ν={nu:.2f} κ={snapped_k:.3f}(req {requested_k:.3f}) off={off_nd:.3f}",
                         f"v_z={r['v_z_mm_s']:.2f} mm/s")
        except ValueError as e:
            tracker.log(f"ν={nu:.2f} κ={snapped_k:.3f} off={off_nd:.3f}", f"SKIP: {e}")

        if (idx + 1) % 10 == 0:
            save_dense(results)

    save_dense(results)
    print(f"\nPhase C complete: {len(results)} LHS configs saved")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"\n=== Starting mode={mode} at {dt.datetime.now().isoformat(timespec='seconds')} ===\n")
    if mode == "phaseA":
        phase_A()
    elif mode == "phaseB":
        phase_B()
    elif mode == "phaseC":
        phase_C()
    elif mode == "all":
        phase_A()
        phase_B()
        phase_C()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    print(f"\n=== Done at {dt.datetime.now().isoformat(timespec='seconds')} ===")
