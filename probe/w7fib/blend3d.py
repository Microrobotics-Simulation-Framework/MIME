"""Arm B: residual forces of the shared-node blend in 3-D (no solve: the energy's gradient at an imposed field).

Fields symmetric under the quarter's mirrors: affine diag(1, -0.3, 0.5) e-3; quadratic u_z = kappa x^2 / 2.
The band runs along x (y, z inside the zone, away from the free top). Residual = force minus the pure carriers' force,
in units of mu eps h^2 (affine) or of the pure carriers' force (quadratic). Also the free surface's traction test
(in-plane uniaxial stress) for each surface treatment."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lat3d  # noqa: E402
import jax.numpy as jnp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
mu, nu, h = 1.0, 0.4, 1.0
zone = 8.0
rows = []
offs, n_, r_, wt_pmb = lat3d.stencil("pmb")
s_p = lat3d.bond_scale(wt_pmb, n_, mu, h)
C = np.einsum("b,bi,bj,bk,bl->ijkl", s_p * wt_pmb, n_, n_, n_, n_) / (2 * h ** 3)
C11_pmb = C[0, 0, 0, 0]
for stencil, partner, pname in (("iso", None, "iso"), ("pmb", None, "isotropic(mismatch)"),
                                ("pmb", (C11_pmb, mu, mu), "cauchy_born")):
    for shape in ("linear", "quad_c1", "cubic_c1", "quintic_c2", "step"):
        for w in (6, 12):
            band = w * h
            core = zone + band + 5 * h
            B = lat3d.Block(h, core, core, zone, band, mu, nu, corr="none", stencil_kind=stencil, shape=shape,
                            partner=partner)
            X = np.stack([B.X, B.Y, B.Z], -1)
            ix0, ix1 = int(zone) - 3, int(zone + band) + 4
            sl = (slice(ix0, ix1), slice(1, 5), slice(4, int(zone) - 1))
            eps = 1e-3 * np.array([1.0, -0.3, 0.5])
            fa = np.asarray(B.grad(jnp.asarray((X * eps).reshape(-1)))).reshape(B.shape3 + (3,))
            aff = float(np.abs(fa[sl]).max() / (mu * 1e-3 * h ** 2))
            kap = 1e-3
            uq = np.zeros_like(X)
            uq[..., 2] = 0.5 * kap * X[..., 0] ** 2
            fq = np.asarray(B.grad(jnp.asarray(uq.reshape(-1)))).reshape(B.shape3 + (3,))
            ref_in = fq[2:4, 1:5, 4:int(zone) - 1].mean(axis=(0, 1, 2))
            ref_out = fq[ix1 + 1:ix1 + 3, 1:5, 4:int(zone) - 1].mean(axis=(0, 1, 2))
            refmag = float(np.linalg.norm(ref_in))
            quad = float(np.abs(fq[sl] - ref_in).max() / refmag)
            rows.append(dict(stencil=stencil, partner=pname, shape=shape, w=w, affine=aff, quad=quad,
                             pure_in=ref_in.tolist(), pure_out=ref_out.tolist()))
            print(json.dumps(rows[-1]), flush=True)
surf = []
for corr in ("none", "volume", "ghost", "skin"):
    B = lat3d.Block(h, 22 * h, 22 * h, 10 * h, 4 * h, mu, nu, corr=corr)
    X = np.stack([B.X, B.Y, B.Z], -1)
    f = np.asarray(B.grad(jnp.asarray((X * (1e-3 * np.array([1.0, -nu, -nu]))).reshape(-1)))).reshape(B.shape3 + (3,))
    f = f / (mu * 1e-3 * h ** 2)
    surf.append(dict(corr=corr, top_layers_max=[float(np.abs(f[3:7, 3:7, k, :]).max()) for k in range(5)],
                     stress_force_scale=2 * (1 + nu)))
    print(json.dumps(surf[-1]), flush=True)
json.dump(dict(rows=rows, surface=surf, C11_pmb_over_mu=C11_pmb / mu), open(a.out, "w"), indent=1)
