#!/usr/bin/env python3
"""Task 2: Validate 3-kernel Triton vs JAX gather at 48³ on A40.

Runs DefectCorrectionFluidNode with both backends, compares against NN-BEM.
Both backends should give identical drag (within ~0.5% of each other),
proving the 3-kernel split fixed the Triton compiler reordering.
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


def main():
    print(f"Device: {jax.devices()[0]}")

    a = 1.0
    mu = 1.0
    rho = 1.0
    lam = 0.3
    R_cyl = a / lam
    N_lbm = 48
    dx = 2 * R_cyl / (N_lbm * 0.8)
    eps_nn = min(0.05, 0.02 * (R_cyl - a))

    labels = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]

    # ── NN-BEM reference ──────────────────────────────────────────
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

    # ── Run both backends at 48³ ──────────────────────────────────
    results = {}
    body = sphere_surface_mesh(radius=a, n_refine=2)

    for backend_name in ["triton_3kernel", "jax_gather"]:
        print(f"=== {backend_name} @ {N_lbm}³ ===")
        t0 = time.time()

        if backend_name == "jax_gather":
            # Temporarily disable Triton to force JAX gather fallback
            import mime.nodes.environment.lbm.triton_kernels as tk
            orig = tk.TRITON_AVAILABLE
            tk.TRITON_AVAILABLE = False

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

        if backend_name == "jax_gather":
            tk.TRITON_AVAILABLE = orig

        diag = [float(R[i, i]) for i in range(6)]
        errs = [abs(diag[i] - nn_diag[i]) / abs(nn_diag[i]) * 100
                for i in range(6)]
        results[backend_name] = {"diag": diag, "errors": errs, "elapsed": elapsed}

        # Force new node on next iteration to re-trigger backend selection
        del node
        if hasattr(DefectCorrectionFluidNode, '_logged_backend'):
            delattr(DefectCorrectionFluidNode, '_logged_backend')

        for i, l in enumerate(labels):
            ok = "PASS" if errs[i] < 5 else "FAIL"
            print(f"  {l}: {diag[i]:>10.2f}  (err {errs[i]:>5.1f}%) [{ok}]")
        print(f"  Time: {elapsed:.0f}s")
        print(flush=True)

    # ── Comparison table ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON TABLE: 48³")
    print("=" * 70)
    print(f"{'':>6}  {'Triton 3K':>10}  {'JAX gather':>10}  {'ΔBackend':>10}  {'NN-BEM':>10}")
    print("-" * 70)

    all_pass = True
    for i, l in enumerate(labels):
        t = results["triton_3kernel"]["diag"][i]
        j = results["jax_gather"]["diag"][i]
        nn = nn_diag[i]
        delta_backend = abs(t - j) / abs(j) * 100
        err_t = results["triton_3kernel"]["errors"][i]
        err_j = results["jax_gather"]["errors"][i]
        status = "OK" if delta_backend < 0.5 else "WARN"
        if err_t >= 5 or err_j >= 5:
            all_pass = False
        print(f"{l:>6}  {t:>8.2f}({err_t:.0f}%)  {j:>8.2f}({err_j:.0f}%)  "
              f"{delta_backend:>8.2f}%  {nn:>10.2f}  [{status}]")

    print("-" * 70)
    print(f"Triton time: {results['triton_3kernel']['elapsed']:.0f}s  "
          f"JAX gather time: {results['jax_gather']['elapsed']:.0f}s")

    if all_pass:
        print("\n✓ ALL PASS: Both backends < 5% error vs NN-BEM")
    else:
        print("\n✗ SOME FAILURES: See table above")

    backend_diff = max(
        abs(results["triton_3kernel"]["diag"][i] - results["jax_gather"]["diag"][i])
        / abs(results["jax_gather"]["diag"][i]) * 100
        for i in range(6)
    )
    if backend_diff < 0.5:
        print(f"✓ BACKENDS MATCH: max inter-backend difference = {backend_diff:.3f}%")
    else:
        print(f"✗ BACKEND MISMATCH: max inter-backend difference = {backend_diff:.3f}%")


if __name__ == "__main__":
    main()
