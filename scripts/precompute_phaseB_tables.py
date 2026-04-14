#!/usr/bin/env python3
"""Precompute Phase B wall tables (κ=0.12 and κ=0.40).

Runs on CPU with multiprocessing (16 workers). Can run in parallel with
Phase A GPU BEM sweep.
"""
import os, sys
import datetime as dt
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mime.nodes.environment.stokeslet.cylinder_wall_table import (
    precompute_wall_table, save_wall_table,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "dejongh_benchmark"
TABLE_DIR = DATA_DIR / "wall_tables"
TABLE_DIR.mkdir(exist_ok=True)

NEW_TABLES = [
    (0.120, 1.0 / 0.120, "wall_R8.333.npz"),
    (0.400, 1.0 / 0.400, "wall_R2.500.npz"),
]

for kappa, R_ves_nd, fname in NEW_TABLES:
    path = TABLE_DIR / fname
    if path.exists():
        print(f"{fname} exists, skipping")
        continue
    print(f"\n{dt.datetime.now().isoformat(timespec='seconds')} "
          f"Computing {fname} (κ={kappa}, R_nd={R_ves_nd:.3f})...")
    t0 = dt.datetime.now()
    table = precompute_wall_table(
        R_cyl=R_ves_nd, mu=1.0,
        n_rho=40, n_dphi=64, n_dz=128,
        L_max_factor=5.0,
        n_max=15, n_k=80, n_phi=64,
        n_jobs=0, rho_clustering=1.5,
    )
    save_wall_table(table, str(path))
    elapsed = dt.datetime.now() - t0
    print(f"  saved {fname} ({path.stat().st_size/1e6:.0f} MB) in {elapsed}")

print(f"\n{dt.datetime.now().isoformat(timespec='seconds')} All Phase B tables ready")
