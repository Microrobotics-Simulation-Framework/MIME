"""Arm D1: a homogeneous bond network (the horizon-3 stencil, Cauchy-Born: a simple lattice deforms affinely) in
uniaxial tension along z, lateral faces traction free (drained: no pressure) or incompressible (a shared pressure, J = 1).
Reports the effective Poisson ratio, J, densification 1/J and fibre order parameters against the stretch.

Bond laws (force on elongation x = l - l0, k the bond stiffness):
  sym        f = k x
  tension    f = k x (x > 0), beta k x (x < 0)               (fibres buckle: beta << 1)
  stiffen    tension, and slope x zeta beyond stretch lam_s  (single-fibre hardening, about x3 above stretch 2)
"""
import argparse
import itertools
import json

import numpy as np
from scipy.optimize import minimize_scalar

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()

m = 3
offs = np.array([o for o in itertools.product(range(-m, m + 1), repeat=3) if 0 < sum(np.square(o)) <= m * m], float)
half = offs[[tuple(o) > (0, 0, 0) for o in offs]]
r = np.linalg.norm(half, axis=1)
n = half / r[:, None]
A_ = n[:, 0] ** 4 - 3 * n[:, 0] ** 2 * n[:, 1] ** 2
sh = np.round(r ** 2).astype(int)
ids = np.unique(sh)
a_s = np.array([A_[sh == s].mean() for s in ids])
a_b = a_s[np.searchsorted(ids, sh)]
beta_iso = -np.sum(r * A_) / np.sum(r * a_b * A_)
wt = r * (1 + beta_iso * a_b)
k = wt / r ** 2  # k l0^2 = wt (scale irrelevant: ratios only)
l0 = r


def bond_energy(l, law, beta=0.01, zeta=3.0, lam_s=2.0):
    x = l - l0
    if law == "sym":
        return 0.5 * k * x ** 2
    e = np.where(x > 0, 0.5 * k * x ** 2, 0.5 * beta * k * x ** 2)
    if law == "stiffen":
        xs = (lam_s - 1) * l0
        extra = np.where(x > xs, 0.5 * (zeta - 1) * k * (x - xs) ** 2, 0.0)
        e = e + extra
    return e


def W(lam, lt, law, **kw):
    F = np.array([lt, lt, lam])
    l = np.linalg.norm(half * F, axis=1)
    return np.sum(bond_energy(l, law, **kw))


def orient(lam, lt, law, **kw):
    F = np.array([lt, lt, lam])
    y = half * F
    l = np.linalg.norm(y, axis=1)
    c = y[:, 2] / l
    w = wt * l  # fibre length density (deformed), reference-isotropic weights
    P2 = np.sum(w * (1.5 * c ** 2 - 0.5)) / np.sum(w)
    # 2-D projection onto the x-z plane (an SEM image): cos 2 theta from the stretch axis
    pz, px = y[:, 2], y[:, 0]
    keep = (pz ** 2 + px ** 2) > 0
    cos2 = (pz ** 2 - px ** 2) / np.maximum(pz ** 2 + px ** 2, 1e-30)
    S2d = np.sum((w * cos2)[keep]) / np.sum(w[keep])
    # load-bearing (tension) alignment, force weighted
    x = l - l0
    f = np.where(x > 0, k * x, 0.0)
    S2d_f = np.sum((f * cos2)[keep]) / max(np.sum(f[keep]), 1e-30)
    return P2, S2d, S2d_f


cases = [("sym", {}), ("tension", dict(beta=0.1)), ("tension", dict(beta=0.01)), ("tension", dict(beta=0.001)),
         ("stiffen", dict(beta=0.01)), ("stiffen", dict(beta=0.001))]
lams = [1.001, 1.01, 1.05, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5]
out = dict(stencil_neighbours=len(offs), rows=[])
for law, kw in cases:
    for lam in lams:
        res = minimize_scalar(lambda lt: W(lam, lt, law, **kw), bounds=(0.02, 1.5), method="bounded",
                              options=dict(xatol=1e-12))
        lt = res.x
        J = lam * lt * lt
        P2, S2d, S2df = orient(lam, lt, law, **kw)
        lt_i = lam ** -0.5
        P2i, S2di, _ = orient(lam, lt_i, law, **kw)
        out["rows"].append(dict(law=law, **kw, stretch=lam, lateral=lt, nu_eff=-np.log(lt) / np.log(lam), J=J,
                                densification=1 / J, P2=P2, S2d=S2d, S2d_force=S2df, incompressible_S2d=S2di,
                                incompressible_P2=P2i, at_bound=bool(abs(lt - 0.02) < 1e-6)))
for r_ in out["rows"]:
    print("%-8s %-14s lam %.3f  nu_eff %.3f  J %.3f  1/J %.2f  S2d %.3f (incompr %.3f)  P2 %.3f" % (
        r_["law"], str({k_: v for k_, v in r_.items() if k_ in ("beta",)}), r_["stretch"], r_["nu_eff"], r_["J"],
        r_["densification"], r_["S2d"], r_["incompressible_S2d"], r_["P2"]))
json.dump(out, open(a.out, "w"), indent=1)
