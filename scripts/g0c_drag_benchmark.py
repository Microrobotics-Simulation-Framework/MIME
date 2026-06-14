#!/usr/bin/env python3
"""G0-c accuracy anchor: finite-Re sphere drag vs Schiller-Naumann (Re<=200).

Reuses the VALIDATED a2_sphere_uniform_stokes harness (periodic box, body-force
driven, surface-integral drag) at finite Re. Two errors per config:
  err_num  = |F_si - F_balance| / F_balance   (surface-integral vs exact momentum
             balance F=ρ f V_box — numerical accuracy of the drag extraction)
  err_phys = |F_balance - F_SN(Re)| / F_SN    (does the flow carry the right drag;
             F_SN = 6πμaU(1+0.15 Re^0.687), Schiller-Naumann, sphere, steady ≤Re210)
Mesh convergence via cpr=4,6. Re capped at ~200 (steady, axisymmetric wake — no
shedding, so no coarse-mesh-damps-shedding trap). Re>210 is a separate open item.

Pass: at the finer mesh (cpr=6), err_num AND err_phys < 15% across Re≤200.
Resumable jsonl. Run: python scripts/g0c_drag_benchmark.py [--smoke]
"""
from __future__ import annotations
import os, sys, json, argparse, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fvm_validation"))
import numpy as np

OUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "experiments", "schwarz_vessel_helix",
    "output", "g0c_drag_results.jsonl"))


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    from a2_sphere_uniform_stokes import run  # the validated Stokes harness

    a, L_over_a, nu = 0.05, 8.0, 1e-3            # Re = U_inf*2a/nu = 100*U_inf
    dt = 1e-3                                     # CFL-safe for U_max~2 at cpr<=6
    if args.smoke:
        cprs, fs, nch, npc = [4], [0.015], 4, 200
    else:
        cprs, fs, nch, npc = [4, 6], [0.005, 0.015, 0.03], 15, 200

    done = _done(OUT)
    print(f"results → {OUT} ({len(done)} done)", flush=True)
    for cpr in cprs:
        for f in fs:
            key = f"cpr{cpr}:f{f}"
            if key in done:
                print(f"  [skip] {key}", flush=True); continue
            t0 = time.time()
            try:
                F_si, F_bal, U_inf, F_stokes, dx, N = run(
                    cells_per_radius=cpr, a=a, L_over_a=L_over_a, nu=nu,
                    f_body=f, n_chunks=nch, dt=dt, n_per_chunk=npc)
            except Exception as e:
                print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
                continue
            Re = abs(U_inf) * 2 * a / nu
            F_SN = abs(F_stokes) * (1 + 0.15 * Re ** 0.687)   # Schiller-Naumann
            err_num = abs(F_si - F_bal) / abs(F_bal)
            err_phys = abs(abs(F_bal) - F_SN) / F_SN
            rec = {"key": key, "cpr": cpr, "f": f, "N": int(N), "Re": Re,
                   "U_inf": U_inf, "F_si": F_si, "F_balance": F_bal,
                   "F_SN": F_SN, "err_num": err_num, "err_phys": err_phys,
                   "wall_s": round(time.time() - t0, 1)}
            _save(rec)
            print(f"  cpr={cpr} Re={Re:6.1f}: F_si={F_si:.3e} F_SN={F_SN:.3e}  "
                  f"err_num={err_num*100:5.1f}% err_phys={err_phys*100:5.1f}%  "
                  f"({rec['wall_s']:.0f}s)", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
