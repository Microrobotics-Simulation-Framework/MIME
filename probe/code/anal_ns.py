"""Reduce NS arms: per (gap, Re) the wrench at each twist, F_0 (alpha = 0), the motion part M = F(U) - F_0, the regime label and
the near-cancellation ratios. Usage: python -I anal_ns.py out.json SG1.json f1.jsonl [f2.jsonl ...]
- regime: F_ext = -F(U) (steady motion, body inertia zero). N if flow = 0. Else V < 0: U; V = 0: hold (B); V > 0 with F_ext,x < 0
  (thrust upstream): B; F_ext,x > 0: W (slower / equal / faster than u_c by alpha).
- kappa_x = |F_0,x| / |M_x| (axial thrust-to-net); r = |F_0|_(S^-1) / |U|_S on the (x, z, ry) block of the Stokes S of the same
  gap (REFUTE_W31 5.4), U = (V, Vz, Om).
- the discretisation error of each value: |P4 - P3| where both exist (the two levels of the spec family)."""
import sys, json
import numpy as np
from collections import defaultdict


def main():
    out, sfile, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    SG = json.load(open(sfile)) if sfile != "-" else {}
    rows = defaultdict(dict)
    for fn in files:
        for line in open(fn):
            r = json.loads(line)
            if "Fx" not in r:
                continue
            rows[(r["gap"], round(r["Re"], 3), r["label"])][r["order"]] = r
    res = []
    f0 = {}
    for (gap, Re, lab), d in rows.items():
        if lab.endswith("alpha+0.00"):
            top = max(d)
            f0[(gap, Re)] = np.array([d[top]["Fx"], d[top]["Fz"], d[top]["Ty"]])
    for (gap, Re, lab), d in sorted(rows.items()):
        top = max(d)
        r = d[top]
        F = np.array([r["Fx"], r["Fz"], r["Ty"]])
        err = np.abs(F - np.array([d[min(d)]["Fx"], d[min(d)]["Fz"], d[min(d)]["Ty"]])) if len(d) > 1 else np.full(3, np.nan)
        flow = r.get("flow", 1.0)
        U = np.array([r["V"], r.get("Vz", 0.0), r.get("Om", 0.0)])
        F0 = f0.get((gap, Re), np.zeros(3)) if flow else np.zeros(3)
        M = F - F0
        if flow == 0:
            regime = "N"
        elif r["V"] < 0:
            regime = "U"
        elif r["V"] == 0:
            regime = "B-hold"
        else:
            regime = "W" if -F[0] > 0 else "B"
        kap = abs(F0[0]) / abs(M[0]) if abs(M[0]) > 0 else np.inf
        rr = np.nan
        S = SG.get(str(gap))
        if S is not None and np.any(U):
            S = np.array(S)
            idx = [0, 2, 4]
            Sb = S[np.ix_(idx, idx)]
            nu = np.sqrt(U @ Sb @ U)
            nf = np.sqrt(F0 @ np.linalg.solve(Sb, F0))
            rr = nf / nu if nu > 0 else np.inf
        res.append(dict(gap=gap, Re=Re, label=lab, order=top, regime=regime, V=r["V"], Vz=r.get("Vz", 0.0), Om=r.get("Om", 0.0),
                        alpha=r.get("alpha"), F=F.tolist(), err_P3P4=err.tolist(), F0=F0.tolist(), M=M.tolist(), kappa_x=kap,
                        r_W31=rr, err_over_Mx=float(err[0] / abs(M[0])) if abs(M[0]) > 0 else None, newton=r.get("newton_its"),
                        converged=r.get("converged")))
    json.dump(res, open(out, "w"), indent=1, default=float)
    for q in res:
        print(f"g{q['gap']} Re{q['Re']:.2f} {q['label']:22s} {q['regime']:6s} F=({q['F'][0]:+9.3f},{q['F'][1]:+9.3f},{q['F'][2]:+8.3f})"
              f" errP3P4=({q['err_P3P4'][0]:.2e},{q['err_P3P4'][1]:.2e},{q['err_P3P4'][2]:.2e}) M_x={q['M'][0]:+9.3f} "
              f"kappa_x={q['kappa_x']:.3g} r={q['r_W31']:.3g}")


if __name__ == "__main__":
    main()
