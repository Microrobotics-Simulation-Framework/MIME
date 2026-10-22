#!/usr/bin/env python3
"""Axisymmetric magnetostatic oracle for the EP-B1 bench coils — the FEM that generates
`FIELD_TABLES` in `plans/epb1_model.py`.

NOT part of MIME and NOT run in CI. It exists so the plan's field numbers are
reproducible rather than asserted: every `B(z)` the EP-B1 document quotes comes from
here, and `epb1_model.py` refuses to evaluate any geometry it has no table for.

scikit-fem is NOT a MIME dependency. Install it into a throwaway venv:

    python -m venv /tmp/femvenv && /tmp/femvenv/bin/pip install scikit-fem numpy scipy
    /tmp/femvenv/bin/python scripts/epb1_field_oracle.py --gaps 16 20 24 28

Usage:
  python scripts/epb1_field_oracle.py [--gaps MM ...] [--single] [--json OUT]

Physics: weak form of the axisymmetric vector potential in u = r*A_phi,
    integral( (1/(mu*r)) grad(u) . grad(v) ) = integral( J * v ),
with Bz on the axis recovered from u/r^2 as r -> 0 by a two-term fit in r^2.

BIQUADRATIC ELEMENTS ARE REQUIRED. Bilinear sits 32% low and does NOT improve with
refinement, because it cannot represent u ~ r^2 near the axis. This is the single
easiest way to get a confidently wrong answer out of this problem.

The `--gaps` runs place a SECOND, UNEXCITED core of the same geometry facing the first
across the given tip-to-tip gap. That passive core is worth +16% on |B| at 8 mm and
reverses grad|B| at ~0.742 of the gap; modelling the array as one coil in free space --
which is what `geom_key` did until v9, by keying on how many coils were DRIVEN rather
than on the presence of passive cores -- misses a force null inside the workspace.

An axisymmetric solve holds only the COAXIAL passive core. The two perpendicular cores
of the real 4-coil array cannot be represented here at all, so these numbers are a lower
bound on the perturbation and the assembled array must be measured, not computed.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from skfem import (MeshQuad, ElementQuad2, ElementQuad0, Basis, BilinearForm,
                   LinearForm, asm, solve, condense)
from skfem.helpers import grad, dot

MU0 = 4 * np.pi * 1e-7

# shipped geometry: 8 mm bundle core (20 x 1.5 mm rods + 20 um coating -> mu_r 436.9),
# 144 mm rod, 7 layers x 280 turns of 0.25 mm wire on a 0.8 mm bobbin wall.
DEFAULT = dict(mur=436.9, Dc=8e-3, Lrod=144e-3, Lw=79.8e-3, b=1.995e-3,
               NI=490.0, wall=0.8e-3)


def solve_axi(gap: float | None = None, *, mur=436.9, Dc=8e-3, Lrod=144e-3,
              Lw=79.8e-3, b=1.995e-3, NI=490.0, wall=0.8e-3,
              rmax=0.25, zfar=0.35, probes=()) -> dict:
    """Solve one coil (gap=None) or a coil facing a passive core at `gap` tip-to-tip.

    Returns {standoff: (|B|, d|B|/d(standoff) toward the coil)}. The gradient sign is
    kept: past the null it genuinely reverses, and abs() would hide the whole point.
    """
    a = Dc / 2
    ri = a + wall
    ro = ri + b
    r = np.unique(np.concatenate([np.linspace(0, 2e-3, 21),
                                  np.linspace(2e-3, 1.25 * ro, 37),
                                  np.geomspace(1.25 * ro, rmax, 26)]))
    if gap is None:
        z = np.unique(np.concatenate([-np.geomspace(1e-4, zfar, 24),
                                      np.linspace(0.0, Lrod, 49),
                                      Lrod + np.geomspace(1e-4, zfar, 18)]))
    else:
        z = np.unique(np.concatenate([
            -(gap + Lrod) - np.geomspace(1e-4, zfar, 18),
            np.linspace(-(gap + Lrod), -gap, 49),
            np.linspace(-gap, 0.0, 81),
            np.linspace(0.0, Lrod, 49),
            Lrod + np.geomspace(1e-4, zfar, 18)]))
    m = MeshQuad.init_tensor(r, z)
    basis = Basis(m, ElementQuad2())                 # see the docstring: NOT Quad1
    p0 = Basis(m, ElementQuad0(), quadrature=basis.quadrature)
    rc, zc = m.p[:, m.t].mean(axis=1)

    incore = (rc <= a) & (zc >= 0) & (zc <= Lrod)
    if gap is not None:
        incore = incore | ((rc <= a) & (zc <= -gap) & (zc >= -(gap + Lrod)))
    mu = np.where(incore, mur * MU0, MU0)
    J = np.where((rc > ri) & (rc <= ro) & (zc >= 0) & (zc <= Lw), NI / (b * Lw), 0.0)

    @BilinearForm
    def A(u, v, w):
        return dot(grad(u), grad(v)) / (w["mu"] * w.x[0])

    @LinearForm
    def L(v, w):
        return w["J"] * v

    K = asm(A, basis, mu=p0.interpolate(mu))
    f = asm(L, basis, J=p0.interpolate(J))
    u = solve(*condense(K, f, D=basis.get_dofs()))

    def Bz(zq: float) -> float:
        rs = r[1:6]
        vals = [2.0 * float(np.asarray(basis.probes(np.array([[rr], [zq]])) @ u).ravel()[0])
                / rr ** 2 for rr in rs]
        # u/r^2 -> Bz as r->0; fit c0 + c1 r^2 and take c0
        return float(np.linalg.lstsq(np.vstack([np.ones_like(rs), rs ** 2]).T,
                                     np.array(vals), rcond=None)[0][0])

    out = {}
    for d in probes:
        h = 1e-4
        b0, bm, bp = Bz(-d), Bz(-(d - h)), Bz(-(d + h))
        out[d] = (abs(b0), (abs(bm) - abs(bp)) / (2 * h))
    return out


PROBES = [4e-3, 5e-3, 6e-3, 6.5e-3, 7e-3, 8e-3, 9e-3, 10e-3, 11e-3, 12e-3, 13e-3,
          14e-3, 15e-3, 16e-3, 18e-3, 20e-3, 22e-3, 24e-3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", type=float, nargs="*", default=[16, 20, 24, 28],
                    help="tip-to-tip gaps in mm for the two-core runs")
    ap.add_argument("--single", action="store_true",
                    help="also solve the single coil in free space")
    ap.add_argument("--json", default=None, help="write results here")
    a = ap.parse_args()

    res: dict = {}
    if a.single or not a.gaps:
        print("single coil in free space ...", flush=True)
        one = solve_axi(None, probes=PROBES)
        res["single"] = {str(z): one[z] for z in PROBES}
        print("  " + "  ".join(f"{z*1e3:.0f}:{one[z][0]*1e3:.2f}" for z in PROBES))

    for gmm in a.gaps:
        gap = gmm * 1e-3
        P = [z for z in PROBES if z <= gap - 1.5e-3]
        print(f"gap {gmm:.0f} mm ({len(P)} stations) ...", flush=True)
        two = solve_axi(gap, probes=P)
        res[f"gap_{gmm:g}mm"] = {str(z): two[z] for z in P}
        g = [two[z][1] for z in P]
        null = next((P[i] + (P[i+1]-P[i]) * g[i]/(g[i]-g[i+1])
                     for i in range(len(P)-1) if g[i]*g[i+1] < 0), None)
        print("  " + " ".join(f"{z*1e3:.0f}:{two[z][0]*1e3:.2f}/{two[z][1]:+.2f}" for z in P))
        if null:
            print(f"  FORCE NULL at {null*1e3:.2f} mm = {null/gap:.3f} of the gap; "
                  f"usable radius about the array centre {(null - gap/2)*1e3:.2f} mm")
        else:
            print("  no null inside the gap")

    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
