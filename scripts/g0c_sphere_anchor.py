#!/usr/bin/env python3
"""G0-c accuracy anchor — finite-Re sphere drag vs Schiller-Naumann, PRODUCTION config.

Replaces the broken g0c_drag_benchmark.py (which reused the Stokes a2 harness and
never reached terminal velocity -> Re=1.2 instead of ~100, a viscous spin-up /
slow-equilibration artifact, NOT a physics error).

This reuses the purpose-built finite-Re harness `p2_unconfined_sn.run_unconfined`
(periodic box L=12a so wall images negligible; nu set from target Re; body force
from the SN balance so U_inf converges to ~0.1). Crucially that harness already
runs at gamma_conv=1.0 (production) — we sweep cpr=4 (the schwarz far-node target)
and cpr=6 (convergence) at Re=100 and Re=200 (steady, axisymmetric wake; no
shedding below ~210, so no coarse-mesh-damps-shedding trap).

Pass: err = |C_D_FVM - C_D_SN(Re_measured)| / C_D_SN < 15% at cpr=4 (gate),
tightening toward cpr=6. If U_inf lands far from the 0.1 target the periodic box
did not equilibrate -> fall back to the driven-inflow r6_inlet_outlet harness.

Resumable jsonl. Run: python scripts/g0c_sphere_anchor.py
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fvm_validation"))
import numpy as np

OUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "experiments", "schwarz_vessel_helix",
    "output", "g0c_sphere_anchor.jsonl"))


def _done(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line); d[r["key"]] = r
    return d


def _save(rec):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def main():
    from p2_unconfined_sn import run_unconfined, schiller_naumann

    a = 0.05
    configs = [(4, 100.0), (6, 100.0), (4, 200.0), (6, 200.0)]
    done = _done(OUT)
    print(f"results -> {OUT} ({len(done)} done)", flush=True)
    for cpr, Re in configs:
        key = f"cpr{cpr}:Re{int(Re)}"
        if key in done:
            print(f"  [skip] {key}", flush=True); continue
        t0 = time.time()
        try:
            F_x, U_inf, F_bal, dx, n_cells, elapsed, nu = run_unconfined(
                Re_p=Re, cells_per_radius=cpr, L_over_a=12.0,
                n_chunks=15, n_per_chunk=200, dt=0.05)
        except Exception as e:
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        Re_meas = U_inf * 2 * a / nu
        rho = 1.0
        C_D_FVM = F_x / (0.5 * rho * U_inf ** 2 * np.pi * a ** 2)
        C_D_SN = schiller_naumann(Re_meas)
        err = abs(C_D_FVM - C_D_SN) / C_D_SN
        # U_inf far from the 0.1 target => box did not equilibrate (the a2 trap)
        equilibrated = abs(U_inf - 0.1) / 0.1 < 0.5
        rec = {"key": key, "cpr": cpr, "Re_target": Re, "Re_meas": Re_meas,
               "U_inf": U_inf, "nu": nu, "F_si": F_x, "F_balance": F_bal,
               "C_D_FVM": C_D_FVM, "C_D_SN": C_D_SN, "err": err,
               "equilibrated": bool(equilibrated), "n_cells": int(n_cells),
               "wall_s": round(time.time() - t0, 1)}
        _save(rec)
        print(f"  cpr={cpr} Re_meas={Re_meas:6.1f} (U={U_inf:.3e}): "
              f"C_D_FVM={C_D_FVM:.3f} C_D_SN={C_D_SN:.3f}  err={err*100:5.1f}%  "
              f"{'EQUIL' if equilibrated else 'NOT-EQUIL'}  ({rec['wall_s']:.0f}s)",
              flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
