#!/usr/bin/env python3
"""Minimal end-to-end test of the de Jongh benchmark pipeline.

Runs one small wall table + one centered + one off-center BEM solve.
Reports per-step timings to estimate the full overnight run.
"""

import os, sys, time, json
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PYTHONUNBUFFERED"] = "1"
# Prevent OpenBLAS deadlock in background/forked processes
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from mime.nodes.environment.stokeslet.dejongh_geometry import (
    dejongh_fl_mesh, R_CYL_DEFAULT,
)
from mime.nodes.environment.stokeslet.cylinder_wall_table import (
    precompute_wall_table, save_wall_table, load_wall_table,
)
from dejongh_benchmark import (
    compute_R_matrix, swimming_velocity, nd_to_mm_per_s,
    VESSELS, R_CYL_UMR, MU, N_THETA, N_ZETA,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'dejongh_benchmark')
TABLE_DIR = os.path.join(DATA_DIR, 'wall_tables')
os.makedirs(TABLE_DIR, exist_ok=True)

timings = {}


def timed(label):
    """Context manager that prints elapsed time."""
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n[START] {label}")
            return self
        def __exit__(self, *args):
            dt = time.time() - self.t0
            timings[label] = dt
            print(f"[DONE]  {label} — {dt:.1f}s ({dt/60:.1f} min)")
    return Timer()


def main():
    print("=" * 70)
    print("DE JONGH PIPELINE — MINIMAL TEST")
    print("=" * 70)

    # ── 1. Mesh generation ───────────────────────────────────────
    with timed("Mesh FL-9 (n_theta=40, n_zeta=40)"):
        mesh = dejongh_fl_mesh(9, n_theta=N_THETA, n_zeta=N_ZETA)
        print(f"  N={mesh.n_points}, area={mesh.total_area:.1f} mm², spacing={mesh.mean_spacing:.4f} mm")

    # ── 2. Wall table — use 1/4" vessel (moderate confinement, representative) ──
    vname = '1/4"'
    vdata = VESSELS[vname]
    R_nd = vdata["R_ves_nd"]
    table_path = os.path.join(TABLE_DIR, f"wall_R{R_nd:.3f}.npz")

    # Reduced params for speed: n_rho=20 (vs 40 production), n_dphi=32, n_dz=64
    with timed(f"Wall table {vname} (R_nd={R_nd:.3f}, n_rho=20, REDUCED)"):
        table = precompute_wall_table(
            R_cyl=R_nd, mu=MU,
            n_rho=20, n_dphi=32, n_dz=64,
            n_max=10, n_k=40, n_phi=32,
            n_jobs=0, rho_clustering=1.5,
        )
        save_wall_table(table, table_path)
        sz_mb = os.path.getsize(table_path) / 1e6
        print(f"  Saved: {table_path} ({sz_mb:.0f} MB)")
        print(f"  rho_grid: [{table.rho_grid[0]:.4f} .. {table.rho_grid[-1]:.4f}], {len(table.rho_grid)} pts")

    # Estimate production table time
    # Production uses: n_rho=40 (vs 20), n_dphi=64 (vs 32), n_dz=128 (vs 64),
    #                  n_max=15 (vs 10), n_k=80 (vs 40), n_phi=64 (vs 32)
    # Scaling: n_rho² (slices) × (n_dphi×n_dz) (pts/slice) × (n_max×n_k×n_phi) (Bessel cost)
    # Ratio: (40/20)² × (64×128)/(32×64) × (15×80×64)/(10×40×32) ≈ 4 × 4 × 6 = 96
    # But parallelism over n_jobs cores absorbs the pts/slice factor.
    # Effective ratio: (40/20)² × (15×80×64)/(10×40×32) ≈ 4 × 6 = 24
    t_table_reduced = timings[f"Wall table {vname} (R_nd={R_nd:.3f}, n_rho=20, REDUCED)"]
    t_table_prod_est = t_table_reduced * 24
    print(f"\n  ** Production estimate (full params): {t_table_prod_est/60:.0f} min per table")
    print(f"  ** 4 tables total: {4*t_table_prod_est/60:.0f} min ({4*t_table_prod_est/3600:.1f} hr)")

    # ── 3. Centered BEM solve ────────────────────────────────────
    with timed(f"Centered R: FL-9 × {vname}"):
        try:
            R = compute_R_matrix(mesh, table, R_nd)
            U = swimming_velocity(R)
            recip = np.max(np.abs(R - R.T))
            eigvals = np.linalg.eigvalsh(R)
            pd = np.all(eigvals > 0)
            v_z = nd_to_mm_per_s(U[2])
            v_lat = nd_to_mm_per_s(np.sqrt(U[0]**2 + U[1]**2))
            print(f"  v_z = {v_z:.2f} mm/s, v_lat = {v_lat:.4f} mm/s")
            print(f"  |R-R^T| = {recip:.2e}, PD = {pd}")
            print(f"  R_FU diag: [{R[0,0]:.2f}, {R[1,1]:.2f}, {R[2,2]:.2f}]")
            print(f"  R_TW_zz = {R[5,5]:.2f}")
            print(f"  Eigenvalues: {np.array2string(eigvals, precision=2)}")
        except ValueError as e:
            print(f"  FAILED: {e}")

    t_centered = timings[f"Centered R: FL-9 × {vname}"]
    print(f"\n  ** 28 centered configs estimate: {28*t_centered/60:.0f} min")

    # ── 4. Off-center BEM solve ──────────────────────────────────
    R_max_nd = R_CYL_DEFAULT * (1 + 0.33) / R_CYL_UMR
    max_off = R_nd - R_max_nd - 0.05
    off_nd = min(0.1 * R_nd, max_off) if max_off > 0 else 0

    if off_nd > 0:
        with timed(f"Off-center R: FL-9 × {vname}, offset={off_nd:.3f}"):
            try:
                R_oc = compute_R_matrix(mesh, table, R_nd, offset_xy=(off_nd, 0.0))
                U_oc = swimming_velocity(R_oc)
                recip_oc = np.max(np.abs(R_oc - R_oc.T))
                v_z_oc = nd_to_mm_per_s(U_oc[2])
                v_lat_oc = nd_to_mm_per_s(np.sqrt(U_oc[0]**2 + U_oc[1]**2))
                drift_deg = np.degrees(np.arctan2(U_oc[1], U_oc[0]))

                print(f"  v_z = {v_z_oc:.2f} mm/s (Δ = {v_z_oc - v_z:.2f} from centered)")
                print(f"  v_lateral = {v_lat_oc:.4f} mm/s, drift = {drift_deg:.1f}°")
                print(f"  |R-R^T| = {recip_oc:.2e}")
                print(f"  R_FU diag: [{R_oc[0,0]:.2f}, {R_oc[1,1]:.2f}, {R_oc[2,2]:.2f}]")
                print(f"  R_FU_xx/yy = {R_oc[0,0]/R_oc[1,1]:.3f} (1.0 = centered)")

                # Compare coupling terms
                print(f"  New couplings R[0,5],R[1,5] = [{R_oc[0,5]:.4f}, {R_oc[1,5]:.4f}]")
                print(f"  (centered were: [{R[0,5]:.4f}, {R[1,5]:.4f}])")
            except ValueError as e:
                print(f"  FAILED: {e}")

        t_offcenter = timings[f"Off-center R: FL-9 × {vname}, offset={off_nd:.3f}"]
        print(f"\n  ** Note: off-center reuses A_body, only G_wall changes")
        print(f"  ** 32 off-center configs estimate: {32*t_offcenter/60:.0f} min")
    else:
        print(f"\n  SKIP off-center: vessel too tight (max_offset={max_off:.3f})")
        t_offcenter = t_centered  # fallback estimate

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    for label, dt in timings.items():
        print(f"  {dt:7.1f}s  {label}")

    print("\n" + "-" * 70)
    print("PRODUCTION ESTIMATES (n_rho=40, full params)")
    print("-" * 70)
    print(f"  Wall tables (4):     {4*t_table_prod_est/60:6.0f} min  ({4*t_table_prod_est/3600:.1f} hr)")
    print(f"  Centered sweep (28): {28*t_centered/60:6.0f} min")
    print(f"  Off-center (32):     {32*t_offcenter/60:6.0f} min")
    total_est = 4*t_table_prod_est + 28*t_centered + 32*t_offcenter
    print(f"  TOTAL:               {total_est/60:6.0f} min  ({total_est/3600:.1f} hr)")
    print("=" * 70)

    # Check pass/fail
    ok = True
    if recip > 1.0:
        print("\n⚠ WARNING: reciprocity error > 1.0 — table resolution may be too low")
    if not pd:
        print("\n✗ FAIL: R not positive definite")
        ok = False
    if v_z <= 0:
        print("\n✗ FAIL: swimming speed ≤ 0")
        ok = False
    if ok:
        print("\n✓ Pipeline test PASSED — ready for overnight run")


if __name__ == "__main__":
    main()
