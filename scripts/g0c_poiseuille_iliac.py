#!/usr/bin/env python3
"""G0-c accuracy anchor (CORRECT one) — vessel-flow fidelity at iliac Re.

WHY THIS, NOT SPHERE DRAG: the schwarz two-scale far-node carries the VESSEL
flow (Poiseuille/Womersley) + the IBM vessel WALL only; the screw is NOT a sharp
IBM body — it couples via regularized forcing (forcing_sigma=1.5dx) and the BEM
owns its drag. So the FVM never resolves an immersed bluff-body boundary layer,
and sphere-drag-vs-SN (which needs cpr~16 to resolve a BL at Re=100) tests
capability production does not use. What the FVM must do at iliac Re (the
against-the-current scenario, Re~390-620) is carry the vessel flow accurately.

Fully-developed Poiseuille is an EXACT steady NS solution at any Re (u·∇u=0 for
unidirectional flow), so the test is: does the FVM-IBM PRESERVE it at iliac Re?
This + TG (2nd-order convection/diffusion) + the stability sweep (stable to
Re~1100) + the swim/Schwarz coupling evidence is the G0-c accuracy case.

Reuses r6_inlet_outlet.setup_pipe (Dirichlet inlet/outlet Poiseuille + IBM wall),
SI-matched to the schwarz vessel (R_ves=3.175 mm, nu=1e-6), at Re=400 and 600.
Pass: profile error < 3% AND finite (stable). Run: python scripts/g0c_poiseuille_iliac.py
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fvm_validation"))
import numpy as np

from mime.nodes.environment.fvm.piso import run_piso
from r6_inlet_outlet import setup_pipe

OUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "experiments", "schwarz_vessel_helix",
    "output", "g0c_poiseuille_iliac.jsonl"))


def run_case(*, Re, R_pipe=3.175e-3, nu=1e-6, cpr=4, a=1.56e-3,
             L_over_R=4.0, dt=5e-4, n_chunks=8, n_per_chunk=200):
    U_mean = Re * nu / (2 * R_pipe)             # Re = U_mean·2R/nu
    L_pipe = L_over_R * R_pipe
    dx_t = a / cpr
    margin = 1.2
    N_cross = int(np.ceil(2 * margin * R_pipe / dx_t))
    N_axial = int(np.ceil(L_pipe / dx_t))
    mesh, bcs, cfg, bodies, dx, _ = setup_pipe(
        R_pipe=R_pipe, L_pipe=L_pipe, U_mean=U_mean, nu=nu,
        with_sphere=False, N_cross=N_cross, N_axial=N_axial)

    state = None
    for _ in range(n_chunks):
        state = run_piso(mesh, bcs, cfg, n_steps=n_per_chunk, dt=dt,
                         body_force_fn=None, ibm_bodies=bodies, initial=state)
    state["u"].block_until_ready()

    # profile error vs analytic Poiseuille at 3 axial stations
    u = np.asarray(state["u"]).reshape(mesh.cartesian_shape + (3,))
    Nx, Ny, Nz = mesh.cartesian_shape
    iy = Ny // 2
    xline = np.asarray(mesh.x).reshape(mesh.cartesian_shape + (3,))[:, iy, 0, 0]
    errs = []
    finite = bool(np.all(np.isfinite(np.asarray(state["u"]))))
    for frac in (0.25, 0.5, 0.75):
        iz = int(frac * Nz)
        uz = u[:, iy, iz, 2]
        ana = np.where(np.abs(xline) < R_pipe,
                       2 * U_mean * (1 - (xline / R_pipe) ** 2), 0.0)
        interior = np.abs(xline) < R_pipe - 1.5 * dx
        errs.append(float(np.max(np.abs(uz[interior] - ana[interior])) / (2 * U_mean)))
    return U_mean, max(errs), finite, int(mesh.N_cells)


def main():
    done = {}
    if os.path.exists(OUT):
        for l in open(OUT):
            if l.strip():
                r = json.loads(l); done[r["key"]] = r
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"results -> {OUT}", flush=True)
    for Re in (400.0, 600.0):
        key = f"Re{int(Re)}"
        if key in done:
            print(f"  [skip] {key}", flush=True); continue
        t0 = time.time()
        try:
            U_mean, err, finite, ncell = run_case(Re=Re)
        except Exception as e:
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        passed = finite and err < 0.03
        rec = {"key": key, "Re": Re, "U_mean": U_mean, "profile_err": err,
               "finite": finite, "n_cells": ncell, "passed": passed,
               "wall_s": round(time.time() - t0, 1)}
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  Re={Re:.0f} (U_mean={U_mean:.4f}): max profile err={err*100:.2f}%  "
              f"finite={finite}  -> {'PASS' if passed else 'FAIL'}  ({rec['wall_s']:.0f}s)",
              flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
