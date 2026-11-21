"""Arm A: sphere (Sneddon/Hertz) and flat punch on a bond zone blended into a graded Q1 block (quarter), and the plant.

    python probe/w7fib/arm_a.py --h 0.2 --nu 0.4 --corr ghost --case sphere --out FILE
corr: fe (continuum only) | none | volume | ghost.  Lengths in mm in the arguments, SI inside."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lat3d  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--h", type=float, required=True)
ap.add_argument("--nu", type=float, default=0.4)
ap.add_argument("--corr", default="none", choices=("fe", "none", "volume", "ghost", "skin"))
ap.add_argument("--case", default="sphere", choices=("sphere", "punch", "plant"))
ap.add_argument("--L", type=float, default=20.0)
ap.add_argument("--zone", type=float, default=1.7)
ap.add_argument("--band", type=float, default=6.0, help="in spacings")
ap.add_argument("--shape", default="quintic_c2")
ap.add_argument("--stencil", default="iso")
ap.add_argument("--R", type=float, default=5.0)
ap.add_argument("--depths", default="0.01,0.02,0.03,0.04,0.05", help="d/R")
ap.add_argument("--punch", type=float, default=1.0)
ap.add_argument("--cap", type=float, default=16.0)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--frac", type=float, default=0.02)
ap.add_argument("--out", required=True)
a = ap.parse_args()

mm = 1e-3
E = 50e3
mu = E / (2 * (1 + a.nu))
h = a.h * mm
band = a.band * h
zone = a.zone * mm
core = zone + band + 5 * h
t0 = time.time()
rec = dict(args=vars(a))


def build(plant=None):
    return lat3d.Block(h, core, a.L * mm, zone, band, mu, a.nu, corr=("none" if a.corr == "fe" else a.corr),
                       stencil_kind=a.stencil, shape=a.shape, bonds=(a.corr != "fe"), plant=plant, cap=a.cap)


B = build()
rec.update(nodes=int(B.N), shape=list(B.shape3), core_cells=int(B.nc), terms=int(getattr(B, "n_terms", 0)),
           ghosts=int(getattr(B, "n_ghosts", 0)), Kp_over_mu=B.Kp / mu, build_s=time.time() - t0)
print(json.dumps({k: v for k, v in rec.items() if k != "args"}), flush=True)
Estar = E / (1 - a.nu ** 2)
if a.case == "sphere":
    depths = [float(x) * a.R * mm for x in a.depths.split(",")]
    _, amax = lat3d.sneddon_sphere(max(depths), a.R * mm, Estar)
    rows, u = lat3d.indent_sphere(B, a.R * mm, depths, r_cand=1.35 * amax)
    rec["rows"] = rows
elif a.case == "punch":
    d = 0.01 * mm
    F, u, ncg, res, nact = lat3d.punch(B, a.punch * mm, d)
    rec.update(F=F, F_boussinesq=2 * a.punch * mm * Estar * d, cg=ncg, res=res, punch_nodes=nact)
    parts = [float(x) for x in B.parts(lat3d.jnp.asarray(u.reshape(-1)))]
    rec["energy_parts"] = parts
    print(json.dumps(dict(F=F, ratio_boussinesq=F / rec["F_boussinesq"], cg=ncg, res=res, parts=parts)), flush=True)
else:  # broken-bond plant under the punch: first-order prediction dF/F = -(energy of dropped bonds) / (total energy)
    d = 0.01 * mm
    F0, u0, ncg, res, nact = lat3d.punch(B, a.punch * mm, d)
    import jax.numpy as jnp
    T = {k: np.asarray(v) for k, v in B.terms.items()}
    U = u0.reshape(-1, 3)
    s = np.sum(T["n"] * (T["S"] * U[T["J"]] - U[T["I"]]), axis=1)
    e_term = T["w"] * 0.5 * T["k"] * s ** 2
    E0 = float(B.energy(jnp.asarray(u0.reshape(-1))))
    rows = []
    for seed in range(a.seeds):
        Bp = build(plant=(a.frac, a.punch * mm, 3, seed))
        F1, u1, ncg1, res1, _ = lat3d.punch(Bp, a.punch * mm, d)
        pred = -float(np.sum(e_term[Bp.planted_mask])) / E0
        rows.append(dict(seed=seed, planted_terms=Bp.planted_terms, dF_over_F=F1 / F0 - 1, predicted=pred, cg=ncg1,
                         res=res1))
        print(json.dumps(rows[-1]), flush=True)
    rec.update(F0=F0, cg0=ncg, res0=res, plant_rows=rows)
rec["wall_s"] = time.time() - t0
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(rec, open(a.out, "w"), indent=1)
print("wall", rec["wall_s"])
