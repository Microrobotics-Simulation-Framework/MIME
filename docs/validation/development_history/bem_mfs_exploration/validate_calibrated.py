#!/usr/bin/env python3
"""Validate calibrated IB-BEM wall correction at 48³ and 64³.

Uses wall_correction_method="auto" which now triggers the calibrated
IB-BEM subtraction. All 6 R matrix directions use the same method.
"""

import os
import sys
import time

os.environ["XLA_FLAGS"] = " ".join([
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_enable_triton_gemm=false",
])

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_mime")

import jax.numpy as jnp
import numpy as np

from mime.nodes.environment.stokeslet.surface_mesh import (
    sphere_surface_mesh, cylinder_surface_mesh,
)
from mime.nodes.environment.stokeslet.resistance import (
    compute_nn_confined_resistance_matrix,
)
from mime.nodes.environment.defect_correction import DefectCorrectionFluidNode

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", default="48,64",
                        help="Comma-separated grid sizes")
    args = parser.parse_args()

    resolutions = [int(n) for n in args.resolutions.split(",")]

    print(f"Device: {jax.devices()[0]}")

    a = 1.0
    mu = 1.0
    rho = 1.0
    lam = 0.3
    R_cyl = a / lam
    eps_nn = min(0.05, 0.02 * (R_cyl - a))

    labels = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]

    # NN-BEM reference
    print("Computing NN-BEM reference...")
    bc = sphere_surface_mesh(radius=a, n_refine=2)
    bf = sphere_surface_mesh(radius=a, n_refine=4)
    cl = 12.0 * R_cyl
    wc = cylinder_surface_mesh(radius=R_cyl, length=cl, n_circ=48, n_axial=16,
                                cluster_center=True)
    wf = cylinder_surface_mesh(radius=R_cyl, length=cl, n_circ=192, n_axial=64,
                                cluster_center=True)
    Rn = compute_nn_confined_resistance_matrix(
        jnp.array(bc.points), jnp.array(bc.weights),
        jnp.array(bf.points), jnp.array(bf.weights),
        jnp.array(wc.points), jnp.array(wc.weights),
        jnp.array(wf.points), jnp.array(wf.weights),
        jnp.zeros(3), eps_nn, mu,
    )
    nn_diag = [float(Rn[i, i]) for i in range(6)]
    print("NN-BEM reference:")
    for i, l in enumerate(labels):
        print(f"  {l}: {nn_diag[i]:.4f}")
    print()

    results = {}
    for N_lbm in resolutions:
        dx = 2 * R_cyl / (N_lbm * 0.8)
        body = sphere_surface_mesh(radius=a, n_refine=2)

        print(f"=== N={N_lbm}³, method=auto (calibrated) ===")
        t0 = time.time()

        node = DefectCorrectionFluidNode(
            "dc", timestep=0.001, mu=mu, rho=rho,
            body_mesh=body, body_radius=a,
            vessel_radius=R_cyl, dx=dx,
            wall_correction_method="auto",
            open_bc_axis=2,
            max_defect_iter=25,
            alpha=0.3,
        )

        state = node.initial_state()
        R = node.compute_resistance_matrix(state)
        elapsed = time.time() - t0

        diag = [float(R[i, i]) for i in range(6)]
        errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100
                for i in range(6)]
        results[N_lbm] = {"diag": diag, "errors": errs, "elapsed": elapsed}

        for i, l in enumerate(labels):
            ok = "PASS" if errs[i] < 5 else "FAIL"
            print(f"  {l}: {diag[i]:>10.2f}  (err {errs[i]:>5.1f}%) [{ok}]")
        print(f"  Time: {elapsed:.0f}s")
        print(flush=True)

    # Summary
    print("\n" + "=" * 70)
    print("CALIBRATED WALL CORRECTION RESULTS")
    print("=" * 70)
    header = f"{'':>6}"
    for N in resolutions:
        header += f"  {N}³/cal"
    header += f"  {'NN-BEM':>10}"
    print(header)
    print("-" * 70)

    all_pass = True
    for i, l in enumerate(labels):
        row = f"{l:>6}"
        for N in resolutions:
            val = results[N]["diag"][i]
            err = results[N]["errors"][i]
            if err >= 5:
                all_pass = False
            row += f"  {val:>6.1f}({err:.0f}%)"
        row += f"  {nn_diag[i]:>10.2f}"
        print(row)

    print("-" * 70)
    if all_pass:
        print("✓ ALL PASS: calibrated method < 5% for all directions")
    else:
        print("✗ SOME FAILURES: see table above")


if __name__ == "__main__":
    main()
