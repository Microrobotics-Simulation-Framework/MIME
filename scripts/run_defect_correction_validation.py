#!/usr/bin/env python3
"""Defect correction resolution convergence study.

Runs the DefectCorrectionFluidNode at multiple resolutions (48³, 64³, 96³, 128³)
with both "auto" and "representation" methods, comparing against NN-BEM reference.

Usage:
    python3 scripts/run_defect_correction_validation.py [--output results.json]
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", " ".join([
    "--xla_gpu_autotune_level=1",
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_autotune_max_solutions=4",
]))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="defect_correction_validation.json")
    parser.add_argument("--resolutions", default="48,64,96,128",
                        help="Comma-separated grid sizes")
    parser.add_argument("--methods", default="auto",
                        help="Comma-separated methods: auto,representation")
    args = parser.parse_args()

    resolutions = [int(n) for n in args.resolutions.split(",")]
    methods = args.methods.split(",")

    a = 1.0
    mu = 1.0
    rho = 1.0
    lam = 0.3
    R_cyl = a / lam
    eps_nn = min(0.05, 0.02 * (R_cyl - a))

    print(f"Device: {jax.devices()[0]}")
    print(f"Resolutions: {resolutions}")
    print(f"Methods: {methods}")
    print()

    # NN-BEM reference (resolution-independent)
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
    labels = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]
    print("NN-BEM reference:")
    for i, l in enumerate(labels):
        print(f"  {l}: {nn_diag[i]:.4f}")
    print()

    results = {"nn_bem": nn_diag, "runs": []}

    for N_lbm in resolutions:
        dx = 2 * R_cyl / (N_lbm * 0.8)
        body = sphere_surface_mesh(radius=a, n_refine=2)

        for method in methods:
            print(f"--- N={N_lbm}³, method={method} ---")
            t0 = time.time()

            try:
                node = DefectCorrectionFluidNode(
                    "dc", timestep=0.001, mu=mu, rho=rho,
                    body_mesh=body, body_radius=a,
                    vessel_radius=R_cyl, dx=dx,
                    wall_correction_method=method,
                    open_bc_axis=2,
                    n_lbm_spinup=500,
                    n_lbm_warmstart=200,
                    max_defect_iter=25,
                    alpha=0.3,
                )

                state = node.initial_state()
                R = node.compute_resistance_matrix(state)
                elapsed = time.time() - t0

                diag = [float(R[i, i]) for i in range(6)]
                errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100
                        for i in range(6)]

                run = {
                    "N": N_lbm, "method": method,
                    "R_diag": diag, "errors": errs,
                    "elapsed_s": elapsed,
                }
                results["runs"].append(run)

                for i, l in enumerate(labels):
                    ok = "✓" if errs[i] < 5 else "✗"
                    print(f"  {l}: {diag[i]:>10.2f}  (err {errs[i]:>5.1f}%) {ok}")
                print(f"  Time: {elapsed:.0f}s")

            except Exception as e:
                print(f"  FAILED: {e}")
                results["runs"].append({
                    "N": N_lbm, "method": method, "error": str(e),
                })
            print(flush=True)

    # Summary table
    print("\n" + "=" * 70)
    print("RESOLUTION CONVERGENCE TABLE")
    print("=" * 70)
    header = f"{'':>6}"
    for N in resolutions:
        for m in methods:
            header += f"  {N}³/{m[:4]:>4}"
    header += f"  {'NN-BEM':>8}"
    print(header)

    for i, l in enumerate(labels):
        row = f"{l:>6}"
        for N in resolutions:
            for m in methods:
                run = next((r for r in results["runs"]
                           if r.get("N") == N and r.get("method") == m
                           and "R_diag" in r), None)
                if run:
                    val = run["R_diag"][i]
                    err = run["errors"][i]
                    row += f"  {val:>6.1f}({err:.0f}%)"
                else:
                    row += f"  {'FAIL':>10}"
        row += f"  {nn_diag[i]:>8.2f}"
        print(row)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
