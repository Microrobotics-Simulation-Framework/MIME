"""Arm C: an edge-notched strip (plane strain) on a 2-D bond lattice, horizon m spacings, geometrically exact bonds.

Bond energy w = k_eff/2 (l - l0)^2 with k_eff = k r(s_max) in tension, k in compression (a broken bond still resists
interpenetration). Laws: critical stretch (r = 1 below s0, k_res above) or bilinear softening (peak at s1, zero at sc;
secant unloading). Calibrated on the discrete stencil: the energy to break every bond crossing a lattice line equals Gc.
An optional shared pressure lam_p/2 (J - 1)^2 on cells (undamaged, or its tension part degraded by the cell's bonds).
Quasi-static, grip displacement steps; at each step Newton iterations on the frozen-damage energy, damage updated after
each iteration, until the residual and the damage settle. Outputs the grip force, work, stored energy, dissipation and
crack tip, and near-tip fields (1/J and the bond order parameter) at the peak.

    python probe/w7fib/strip2d.py --h 0.3 --law bilinear --regime fibrin --out FILE
"""
import argparse
import itertools
import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ap = argparse.ArgumentParser()
ap.add_argument("--h", type=float, default=0.3, help="mm")
ap.add_argument("--m", type=int, default=3)
ap.add_argument("--law", default="bilinear", choices=("bilinear", "critical", "elastic"))
ap.add_argument("--regime", default="fibrin", choices=("fibrin", "small"))
ap.add_argument("--notch", type=float, default=10.5, help="mm; 0 = unnotched")
ap.add_argument("--nu", type=float, default=0.25, help="> 0.25 adds a shared pressure")
ap.add_argument("--pressure", default="undamaged", choices=("undamaged", "tension_degraded"))
ap.add_argument("--compressive", type=float, default=1.0, help="compressive bond stiffness fraction (fibres: < 1)")
ap.add_argument("--gc_factor", type=float, default=1.0)
ap.add_argument("--dmax", type=float, default=0.0, help="mm grip displacement; 0 = regime default")
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--max_iter", type=int, default=400)
ap.add_argument("--elastic_at", type=float, default=0.0, help="elastic run: report U at this grip displacement (mm)")
ap.add_argument("--grip", default="clamped", choices=("clamped", "free_x"),
                help="free_x: grips move only in y (uniform uniaxial stress for the strength runs)")
ap.add_argument("--protect", type=int, default=0, help="rows next to each grip whose bonds never damage (0: none)")
ap.add_argument("--grip_rows", type=int, default=1, help="rows held rigidly by each grip (m: a grip a horizon thick)")
ap.add_argument("--wall", type=float, default=17000.0, help="s; stop and write before the runner's limit")
ap.add_argument("--out", required=True)
a = ap.parse_args()
t_start = time.time()
mm = 1e-3
W, HH, T = 30 * mm, 10 * mm, 6.5 * mm
E = 5.7e3
if a.regime == "fibrin":
    GC, SIGMA_U, DMAX = 7.6, 4.1e3, 14 * mm
else:
    GC, SIGMA_U, DMAX = 0.076, 0.41e3, 1.4 * mm
GC *= a.gc_factor
if a.dmax > 0:
    DMAX = a.dmax * mm
nu = a.nu
mu = E / (2 * (1 + nu))
lam = 2 * mu * nu / (1 - 2 * nu)  # plane strain, 3-D Lame
lam_p = lam - mu  # the lattice carries lambda = mu
h = a.h * mm

# ---------------------------------------------------------------- lattice
nx = int(round(W / h)) + 1
ny_half = int(round(HH / h + 0.5))
xs = np.arange(nx) * h
ys = (np.arange(2 * ny_half) - ny_half + 0.5) * h  # rows at half spacings; y = 0 between rows
X, Y = np.meshgrid(xs, ys, indexing="ij")
N = X.size
idx = np.arange(N).reshape(X.shape)
P0 = np.stack([X.ravel(), Y.ravel()], 1)
grip_spacing = ys[-1] - ys[0]
m = a.m
offs = np.array([o for o in itertools.product(range(-m, m + 1), repeat=2) if 0 < o[0] ** 2 + o[1] ** 2 <= m * m])
half = offs[(offs[:, 1] > 0) | ((offs[:, 1] == 0) & (offs[:, 0] > 0))]
r = np.linalg.norm(half, axis=1)
n = half / r[:, None]
A_ = n[:, 0] ** 4 - 3 * n[:, 0] ** 2 * n[:, 1] ** 2
sh = np.round(r ** 2).astype(int)
ids = np.unique(sh)
a_s = np.array([A_[sh == s].mean() for s in ids])
a_b = a_s[np.searchsorted(ids, sh)]
beta = -np.sum(r * A_) / np.sum(r * a_b * A_)
wt = r * (1 + beta * a_b)
cst = mu * h ** 2 / np.sum(wt * n[:, 0] ** 2 * n[:, 1] ** 2)  # k_o l0^2 = cst wt_o (per unit thickness)
BI, BJ, BK, BL0, BO = [], [], [], [], []
for oi, o in enumerate(half):
    i0 = np.arange(nx)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(2 * ny_half), indexing="ij")
    i2, j2 = ii + o[0], jj + o[1]
    ok = (i2 >= 0) & (i2 < nx) & (j2 >= 0) & (j2 < 2 * ny_half)
    BI.append(idx[ii[ok], jj[ok]])
    BJ.append(idx[i2[ok], j2[ok]])
    l0 = r[oi] * h
    BK.append(np.full(ok.sum(), cst * wt[oi] / l0 ** 2))
    BL0.append(np.full(ok.sum(), l0))
    BO.append(np.full(ok.sum(), oi))
BI, BJ, BK, BL0, BO = map(np.concatenate, (BI, BJ, BK, BL0, BO))
nb = len(BI)
# fracture calibration: G = (1/h) sum_{oy > 0} oy Gamma_o, Gamma_o = (k_o l0^2 / 2) s0^2 = cst wt_o s0^2 / 2
cross_sum = np.sum(half[:, 1].clip(min=0) * wt)
s0 = np.sqrt(2 * GC * h / (cst * cross_sum))
# strength: under uniaxial stress the bonds along the load reach s1 at the peak; small strain: sigma_u = E' s1 with
# E' the plane-strain uniaxial modulus of the lattice (+ pressure)
Eps = 4 * mu * (mu + lam) / (2 * mu + lam)  # plane-strain modulus under uniaxial stress, eps_zz = 0
s1 = SIGMA_U / Eps
sc = s0 ** 2 / s1
k_res = 1e-6
law = a.law
if law == "bilinear" and sc <= s1:
    raise SystemExit("bilinear needs sc > s1")

# notch: bonds crossing y = 0 with crossing point x < a are cut from the start
yi, yj = P0[BI, 1], P0[BJ, 1]
crosses = (yi < 0) & (yj > 0)
xc = P0[BI, 0] + (P0[BJ, 0] - P0[BI, 0]) * (0 - yi) / np.where(crosses, yj - yi, 1.0)
cut0 = crosses & (xc < a.notch * mm)
vert = crosses & (BO == np.where((half == [0, 1]).all(1))[0][0])  # nearest vertical bonds across y = 0
_row = np.broadcast_to(np.arange(2 * ny_half)[None, :], X.shape).ravel()
_near = (_row < a.protect) | (_row >= 2 * ny_half - a.protect)
protected = _near[BI] | _near[BJ]

# cells for the shared pressure (bilinear quads), J at the centroid
ci, cj = np.meshgrid(np.arange(nx - 1), np.arange(2 * ny_half - 1), indexing="ij")
C4 = np.stack([idx[ci, cj], idx[ci + 1, cj], idx[ci + 1, cj + 1], idx[ci, cj + 1]], -1).reshape(-1, 4)
ncell = len(C4)
mid = 0.5 * (P0[BI] + P0[BJ])
bcell = (np.clip((mid[:, 0] // h).astype(int), 0, nx - 2) * (2 * ny_half - 1)
         + np.clip(((mid[:, 1] - ys[0]) // h).astype(int), 0, 2 * ny_half - 2))
bcount = np.bincount(bcell, minlength=ncell).astype(float)

# boundary conditions: grips clamped (u_x = 0, u_y = +-delta/2)
top = idx[:, -a.grip_rows:].ravel()
bot = idx[:, :a.grip_rows].ravel()
top_row = idx[:, -1]
fixed = np.zeros((N, 2), bool)
if a.grip == "clamped":
    fixed[top] = True
    fixed[bot] = True
else:
    fixed[top, 1] = True
    fixed[bot, 1] = True
    fixed[bot[0], 0] = True
freedofs = np.where(~fixed.ravel())[0]
fmap = -np.ones(2 * N, int)
fmap[freedofs] = np.arange(len(freedofs))
nf = len(freedofs)


class Pattern:
    """static CSR structure over free dofs for a fixed list of (row, col) contributions"""

    def __init__(self, rows, cols):
        fr, fc = fmap[rows], fmap[cols]
        self.keep = (fr >= 0) & (fc >= 0)
        key = fr[self.keep].astype(np.int64) * nf + fc[self.keep]
        uk, self.inv = np.unique(key, return_inverse=True)
        self.indices = (uk % nf).astype(np.int32)
        counts = np.bincount((uk // nf).astype(np.int64), minlength=nf)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        self.nnz = len(uk)

    def matrix(self, vals):
        data = np.bincount(self.inv, weights=vals[self.keep], minlength=self.nnz)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(nf, nf))


def _bond_rc():
    rows, cols = [], []
    for (p, q) in ((BI, BI), (BJ, BJ), (BI, BJ), (BJ, BI)):
        for c1 in range(2):
            for c2 in range(2):
                rows.append(2 * p + c1)
                cols.append(2 * q + c2)
    return np.concatenate(rows), np.concatenate(cols)


BOND_PAT = Pattern(*_bond_rc())


def ratio(smax):
    return np.where(protected, 1.0, _ratio(smax))


def _ratio(smax):
    if law == "elastic":
        return np.ones_like(smax)
    if law == "critical":
        return np.where(smax < s0, 1.0, k_res)
    rr = np.where(smax <= s1, 1.0, s1 * (sc - smax) / (np.maximum(smax, 1e-30) * (sc - s1)))
    return np.clip(np.where(smax >= sc, k_res, rr), k_res, 1.0)


def state(U, rb, gcell):
    """energy, gradient, PSD Hessian with frozen damage"""
    Pc = P0 + U
    y = Pc[BJ] - Pc[BI]
    l = np.linalg.norm(y, axis=1)
    e = y / l[:, None]
    ten = l > BL0
    keff = BK * np.where(ten, rb, a.compressive)
    dl = l - BL0
    w = 0.5 * keff * dl ** 2
    wp = keff * dl
    g = np.zeros((N, 2))
    np.add.at(g, BJ, wp[:, None] * e)
    np.add.at(g, BI, -wp[:, None] * e)
    Hb = keff[:, None, None] * e[:, :, None] * e[:, None, :] + (np.maximum(wp, 0) / l)[:, None, None] * (
        np.eye(2)[None] - e[:, :, None] * e[:, None, :])
    vals = []
    for sgn in (1, 1, -1, -1):
        for c1 in range(2):
            for c2 in range(2):
                vals.append(sgn * Hb[:, c1, c2])
    Hf = BOND_PAT.matrix(np.concatenate(vals))
    Ep = 0.0
    if lam_p > 1e-12:
        Ue = U[C4]  # (C, 4, 2)
        # centroid gradient of the bilinear quad: d/dx = (u1 + u2 - u0 - u3) / 2h, d/dy = (u2 + u3 - u0 - u1) / 2h
        Gx = np.array([-1, 1, 1, -1]) / (2 * h)
        Gy = np.array([-1, -1, 1, 1]) / (2 * h)
        F11 = 1 + Ue[:, :, 0] @ Gx
        F12 = Ue[:, :, 0] @ Gy
        F21 = Ue[:, :, 1] @ Gx
        F22 = 1 + Ue[:, :, 1] @ Gy
        J = F11 * F22 - F12 * F21
        gfac = np.where(J > 1, gcell, 1.0)
        Ac = h * h
        Ep = float(np.sum(0.5 * lam_p * Ac * gfac * (J - 1) ** 2))
        # dJ/du: dJ/dF11 = F22, dJ/dF22 = F11, dJ/dF12 = -F21, dJ/dF21 = -F12
        dJx = F22[:, None] * Gx[None] - F21[:, None] * Gy[None]  # w.r.t. u_x of the 4 nodes
        dJy = -F12[:, None] * Gx[None] + F11[:, None] * Gy[None]
        coef = lam_p * Ac * gfac * (J - 1)
        np.add.at(g[:, 0], C4, coef[:, None] * dJx)
        np.add.at(g[:, 1], C4, coef[:, None] * dJy)
        dJ = np.stack([dJx, dJy], -1).reshape(-1, 8)  # node-major: (n0x, n0y, n1x, ...)
        dofs = (2 * C4[:, :, None] + np.arange(2)[None, None]).reshape(-1, 8)
        Hc = (lam_p * Ac * gfac)[:, None, None] * dJ[:, :, None] * dJ[:, None, :]
        global CELL_PAT
        if CELL_PAT is None:
            CELL_PAT = Pattern(np.repeat(dofs, 8, axis=1).ravel(), np.tile(dofs, (1, 8)).ravel())
        Hf = Hf + CELL_PAT.matrix(Hc.ravel())
    return float(np.sum(w)) + Ep, g, Hf, l


CELL_PAT = None


def energy_only(U, rb, gcell):
    Pc = P0 + U
    l = np.linalg.norm(Pc[BJ] - Pc[BI], axis=1)
    keff = BK * np.where(l > BL0, rb, a.compressive)
    e = float(np.sum(0.5 * keff * (l - BL0) ** 2))
    if lam_p > 1e-12:
        Ue = U[C4]
        Gx = np.array([-1, 1, 1, -1]) / (2 * h)
        Gy = np.array([-1, -1, 1, 1]) / (2 * h)
        J = (1 + Ue[:, :, 0] @ Gx) * (1 + Ue[:, :, 1] @ Gy) - (Ue[:, :, 0] @ Gy) * (Ue[:, :, 1] @ Gx)
        e += float(np.sum(0.5 * lam_p * h * h * np.where(J > 1, gcell, 1.0) * (J - 1) ** 2))
    return e


def cell_factor(rb):
    if lam_p <= 1e-12 or a.pressure == "undamaged":
        return np.ones(ncell)
    return np.bincount(bcell, weights=rb, minlength=ncell) / np.maximum(bcount, 1)


FACTOR = dict(lu=None, uses=0, refactors=0, cg_its=0)


def lin_solve(Hf, b):
    """PCG preconditioned by the last stored LU factor (damage changes are local, so an old factor is a close SPD
    preconditioner); refactor when PCG needs more than 60 iterations"""
    if FACTOR["lu"] is not None:
        M = spla.LinearOperator(Hf.shape, matvec=FACTOR["lu"].solve)
        its = [0]

        def cb(_):
            its[0] += 1
        x, info = spla.cg(Hf, b, rtol=1e-11, atol=0.0, maxiter=60, M=M, callback=cb)
        FACTOR["cg_its"] += its[0]
        if info == 0:
            FACTOR["uses"] += 1
            return x
    FACTOR["lu"] = spla.splu(Hf.tocsc(), permc_spec="MMD_AT_PLUS_A")
    FACTOR["refactors"] += 1
    return FACTOR["lu"].solve(b)


def equilibrium(U, smax, delta, tol=1e-9, freeze=False):
    """Newton iterations on the frozen-damage energy, the damage updated after each iteration, until the residual and
    the damage settle (the same final state as a staggered scheme converged at each damage update, measured to 1e-11 on
    the smoke strip, at half the solves)"""
    U = U.copy()
    if a.grip == "clamped":
        U[top, 0] = 0.0
        U[bot, 0] = 0.0
    else:
        U[bot[0], 0] = 0.0
    U[top, 1] = delta / 2
    U[bot, 1] = -delta / 2
    its = 0
    damage_moves = 0
    while its < a.max_iter:
        rb = np.where(cut0, k_res, ratio(smax))
        gcell = cell_factor(rb)
        e0, g, Hf, l = state(U, rb, gcell)
        gf = g.ravel()[freedofs]
        du = lin_solve(Hf, -gf)
        step = 1.0
        for _ in range(30):
            Un = U.ravel().copy()
            Un[freedofs] += step * du
            en = energy_only(Un.reshape(N, 2), rb, gcell)
            if en <= e0 + 1e-12 * abs(e0) + 1e-30:
                break
            step *= 0.5
        U = Un.reshape(N, 2)
        its += 1
        moved = False
        if not freeze and law != "elastic":
            Pc = P0 + U
            s_now = np.linalg.norm(Pc[BJ] - Pc[BI], axis=1) / BL0 - 1
            new = np.maximum(smax, s_now)
            moved = bool(np.any((ratio(new) < ratio(smax) - 1e-12) & ~cut0))
            smax = new
            damage_moves += moved
        if (not moved) and step == 1.0 and np.max(np.abs(du)) < 1e-8 * h:
            break
    rb = np.where(cut0, k_res, ratio(smax))
    _, g, _, _ = state(U, rb, cell_factor(rb))
    F = float(np.sum(g[top, 1])) * T  # reaction on the top grip (N): the sum over the grip's held rows
    Ustore = energy_only(U, rb, cell_factor(rb)) * T
    return U, smax, F, Ustore, its, damage_moves


def dissipated(smax):
    """exact bond-level dissipation: envelope work minus the energy stored on the secant, at the largest stretch"""
    dm = np.maximum(smax, 0.0) * BL0
    if law == "critical":
        d0 = s0 * BL0
        Db = np.where(smax >= s0, 0.5 * BK * d0 ** 2, 0.0)
    elif law == "bilinear":
        d1, dc = s1 * BL0, sc * BL0
        env_work = np.where(dm <= d1, 0.5 * BK * dm ** 2,
                            0.5 * BK * d1 ** 2 + BK * d1 * ((dm - d1) - 0.5 * (np.minimum(dm, dc) - d1) ** 2 / (dc - d1)
                                                          * 0 + 0) )
        # envelope force beyond d1: k d1 (dc - x)/(dc - d1); its integral from d1 to x
        xx = np.minimum(dm, dc)
        tail = BK * d1 * ((xx - d1) * dc - 0.5 * (xx ** 2 - d1 ** 2)) / (dc - d1)
        env_work = np.where(dm <= d1, 0.5 * BK * dm ** 2, 0.5 * BK * d1 ** 2 + tail)
        fenv = np.where(dm <= d1, BK * dm, np.where(dm >= dc, 0.0, BK * d1 * (dc - dm) / (dc - d1)))
        Db = env_work - 0.5 * fenv * dm
    else:
        Db = np.zeros_like(smax)
    return float(np.sum(np.where(cut0 | protected, 0.0, Db))) * T


def tip_x(smax):
    rb = np.where(cut0, k_res, ratio(smax))
    broken = vert & (rb <= 1e-3)
    xv = P0[BI[vert], 0]
    bv = broken[vert]
    order = np.argsort(xv)
    xv, bv = xv[order], bv[order]
    tip = 0.0
    for x_, b_ in zip(xv, bv):
        if b_:
            tip = x_
        else:
            if x_ > a.notch * mm + h:
                break
    return tip


def near_tip(U, smax, xt):
    Pc = P0 + U
    Ue = U[C4]
    Gx = np.array([-1, 1, 1, -1]) / (2 * h)
    Gy = np.array([-1, -1, 1, 1]) / (2 * h)
    J = (1 + Ue[:, :, 0] @ Gx) * (1 + Ue[:, :, 1] @ Gy) - (Ue[:, :, 0] @ Gy) * (Ue[:, :, 1] @ Gx)
    cc = P0[C4].mean(1)
    y = Pc[BJ] - Pc[BI]
    l = np.linalg.norm(y, axis=1)
    rb = np.where(cut0, k_res, ratio(smax))
    intact = rb > 0.5
    cos2 = (y[:, 1] ** 2 - y[:, 0] ** 2) / l ** 2  # cos 2 theta from the loading axis (y)
    wgt = wt[BO] * l * intact
    out = []
    for r0, r1 in ((0, 2), (2, 4), (4, 8), (8, 16), (16, 40)):
        dc = np.hypot(cc[:, 0] - xt, cc[:, 1])
        sel = (dc >= r0 * h) & (dc < r1 * h) & (cc[:, 0] > xt - r1 * h)
        db = np.hypot(mid[:, 0] - xt, mid[:, 1])
        selb = (db >= r0 * h) & (db < r1 * h)
        out.append(dict(r_mm=[r0 * a.h, r1 * a.h], inv_J=float(np.mean(1 / J[sel])) if sel.any() else None,
                        min_J=float(np.min(J[sel])) if sel.any() else None,
                        order=float(np.sum(wgt[selb] * cos2[selb]) / np.sum(wgt[selb])) if selb.any() else None))
    return out


rec = dict(args=vars(a), h_mm=a.h, horizon_mm=m * a.h, nodes=N, bonds=nb, s0=s0, s1=s1, sc=sc, beta=beta, Gc=GC,
           lam_p_over_mu=lam_p / mu, grip_spacing_mm=grip_spacing / mm, cut_bonds=int(cut0.sum()))
print(json.dumps(rec), flush=True)
U = np.zeros((N, 2))
smax = np.zeros(nb)
deltas = np.linspace(0, DMAX, a.steps + 1)[1:]
if a.elastic_at > 0:
    deltas = np.linspace(0, a.elastic_at * mm, 9)[1:]
rows = []
Wext, Fprev, dprev = 0.0, 0.0, 0.0
peak = (0.0, None)
U_at_peak = None
for k, d in enumerate(deltas):
    # the increment with damage frozen first (the work's trapezoid), then damage at fixed grips (no work)
    U, _, F_el, _, its0, _ = equilibrium(U, smax, d, freeze=True)
    Wext += 0.5 * (Fprev + F_el) * (d - dprev)
    U, smax, F, Us, its, dm = equilibrium(U, smax, d)
    Fprev, dprev = F, d
    xt = tip_x(smax)
    D = Wext - Us
    rows.append(dict(delta_mm=d / mm, F=F, F_el=F_el, W=Wext, U=Us, D=D, D_bonds=dissipated(smax), tip_mm=xt / mm,
                     its=its0 + its, damage_moves=dm, refactors=FACTOR["refactors"], cg_its=FACTOR["cg_its"],
                     wall=time.time() - t_start))
    print(json.dumps(rows[-1]), flush=True)
    if F > peak[0]:
        peak = (F, k)
        U_at_peak, smax_at_peak = U.copy(), smax.copy()
    if law != "elastic" and k > 5 and F < 0.15 * peak[0]:
        break
    if time.time() - t_start > a.wall:
        rec["stopped"] = "wall"
        break
rec["rows"] = rows
if law != "elastic" and U_at_peak is not None:
    rec["peak"] = dict(F=peak[0], delta_mm=rows[peak[1]]["delta_mm"], tip_mm=rows[peak[1]]["tip_mm"])
    rec["near_tip_at_peak"] = near_tip(U_at_peak, smax_at_peak, max(rows[peak[1]]["tip_mm"] * mm, a.notch * mm))
    # dissipation per crack area over the propagation (tip advanced 1 to 8 mm past the notch)
    adv = np.array([r_["tip_mm"] for r_ in rows]) - a.notch
    Dd = np.array([r_["D_bonds"] for r_ in rows])
    sel = (adv >= 1.0) & (adv <= 8.0)
    if sel.sum() >= 3:
        slope = np.polyfit(adv[sel] * mm * T, Dd[sel], 1)[0]
        rec["G_dissipation"] = float(slope)
    rec["G_total_over_advance"] = float(Dd[-1] / max(adv[-1] * mm * T, 1e-30)) if adv[-1] > 0.5 else None
if a.elastic_at > 0:
    rec["U_at"] = rows[-1]["U"]
rec["wall_s"] = time.time() - t_start
json.dump(rec, open(a.out, "w"), indent=1)
print("done", rec.get("peak"), rec.get("G_dissipation"), rec["wall_s"])
