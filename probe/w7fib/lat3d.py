"""A bond zone (horizon 3 spacings) on shared nodes with a Q1 continuum, in a quarter block, linear (small strain).

Energy (quarter = full / 4):
  bonds:     sum over each zone node's full family, weight alpha(mid) * orb/4 * 1/2 * corr, energy k/2 (n . (S u_j - u_i))^2
  continuum: (1 - alpha(centroid)) * [mu eps:eps + (mu/2) tr(eps)^2] at 2x2x2 Gauss points (the bonds' own tensor, lambda = mu)
  pressure:  K_p/2 tr(eps_centroid)^2 on every cell (the shared volumetric part; K = 5 mu/3 + K_p)
Bonds use an isotropic-calibrated stencil (|o|^2 <= 9, 122 neighbours) unless stencil="pmb". Symmetry planes x = 0, y = 0 by
mirrored neighbours; the top z = 0 is free (or in contact), the bottom fixed. Surface corrections: none, volume
(2 V0 / (V_i + V_j)), ghost (missing bonds evaluated with the node's finite-difference gradient: exact under uniform strain).
"""
import itertools
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------------------------------------------- stencil
def stencil(kind="iso", m=3):
    offs = np.array([o for o in itertools.product(range(-m, m + 1), repeat=3)
                     if 0 < o[0] ** 2 + o[1] ** 2 + o[2] ** 2 <= m * m], float)
    r = np.linalg.norm(offs, axis=1)
    n = offs / r[:, None]
    wt = r.copy()
    if kind == "iso":
        A = n[:, 0] ** 4 - 3 * n[:, 0] ** 2 * n[:, 1] ** 2
        sh = np.round(r ** 2).astype(int)
        ids = np.unique(sh)
        a_s = np.array([A[sh == s].mean() for s in ids])
        a_b = a_s[np.searchsorted(ids, sh)]
        beta = -np.sum(r * A) / np.sum(r * a_b * A)
        wt = r * (1 + beta * a_b)
    return offs.astype(int), n, r, wt


def bond_scale(wt, n, mu, h):
    """k_o |xi_o|^2 = s wt_o with C1212 = mu (full-family sum, per-node share 1/2)."""
    return 2 * mu * h ** 3 / np.sum(wt * n[:, 0] ** 2 * n[:, 1] ** 2)


# ----------------------------------------------------------------------------------------------------------- grid
def graded_axis(core, h, L, ratio=1.25, cap=16.0):
    """uniform cells of h over [0, core], then cells growing by `ratio` up to cap*h, the graded part scaled to end at L"""
    n = int(round(core / h))
    sizes, dx, tot = [], h, 0.0
    while tot < L - n * h - 1e-12:
        dx = min(dx * ratio, cap * h)
        sizes.append(dx)
        tot += dx
    sizes = np.array(sizes) * (L - n * h) / tot
    xs = np.concatenate([np.arange(n + 1) * h, n * h + np.cumsum(sizes)])
    return xs, n


def smooth(t, shape):
    t = np.clip(t, 0.0, 1.0)
    if shape == "linear":
        return t
    if shape == "quad_c1":
        return np.where(t < 0.5, 2 * t ** 2, 1 - 2 * (1 - t) ** 2)
    if shape == "cubic_c1":
        return 3 * t ** 2 - 2 * t ** 3
    if shape == "quintic_c2":
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
    if shape == "step":
        return (t > 0.5).astype(float)
    raise ValueError(shape)


class Block:
    """Quarter block, nodes (ix, iy, iz), z = -zs[iz] (iz = 0 the top)."""

    def __init__(self, h, core, L, zone, band, mu, nu, corr="none", stencil_kind="iso", shape="quintic_c2",
                 bonds=True, plant=None, ratio=1.25, cap=16.0):
        self.h, self.mu, self.nu = h, mu, nu
        K = 2 * mu * (1 + nu) / (3 * (1 - 2 * nu))
        self.Kp = K - 5 * mu / 3
        if self.Kp < -1e-12:
            raise ValueError("nu below 1/4 needs a negative pressure modulus")
        self.xs, self.nc = graded_axis(core, h, L, ratio, cap)
        self.zs, self.ncz = graded_axis(core, h, L, ratio, cap)
        nx, nz = len(self.xs), len(self.zs)
        self.shape3 = (nx, nx, nz)
        self.N = nx * nx * nz
        idx = np.arange(self.N).reshape(self.shape3)
        self.idx = idx
        X, Y, Z = np.meshgrid(self.xs, self.xs, -self.zs, indexing="ij")
        self.X, self.Y, self.Z = X, Y, Z
        zone_z = zone

        def alpha(x, y, z):  # 1 inside the zone, falls to 0 over the band
            if not bonds:
                return np.zeros_like(np.asarray(x, float))
            fx = smooth(1 - (np.abs(x) - zone) / band, shape)
            fy = smooth(1 - (np.abs(y) - zone) / band, shape)
            fz = smooth(1 - (np.maximum(-z, 0.0) - zone_z) / band, shape)
            if corr == "skin":  # bonds fade out towards the free surface over one horizon; the continuum carries the skin
                fz = fz * smooth(np.asarray(-z, float) / (3 * h), shape)
            return fx * fy * fz

        self.alpha = alpha
        # ---- elements
        ex, ey, ez = np.meshgrid(np.arange(nx - 1), np.arange(nx - 1), np.arange(nz - 1), indexing="ij")
        ex, ey, ez = ex.ravel(), ey.ravel(), ez.ravel()
        corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        self.econn = np.stack([idx[ex + a, ey + b, ez + c] for a, b, c in corners], 1)
        dx = np.diff(self.xs)[ex]
        dy = np.diff(self.xs)[ey]
        dz = np.diff(self.zs)[ez]  # positive; z axis points up, iz down: handle sign in the gradient
        self.esize = np.stack([dx, dy, dz], 1)
        cx = 0.5 * (self.xs[ex] + self.xs[ex + 1])
        cy = 0.5 * (self.xs[ey] + self.xs[ey + 1])
        cz = -0.5 * (self.zs[ez] + self.zs[ez + 1])
        self.ealpha = alpha(cx, cy, cz)
        # reference-cell shape-function derivatives on [0,1]^3 at the 8 Gauss points and the centroid; local z = depth
        # coordinate (iz grows downward), so d/dz = -d/d(depth)
        g = 0.5 + np.array([-1, 1]) / (2 * np.sqrt(3))
        gps = np.array(list(itertools.product(g, g, g)))
        cs = np.array(corners, float)

        def dN(p):
            out = np.zeros((8, 3))
            for a, c in enumerate(cs):
                fx = c[0] * p[0] + (1 - c[0]) * (1 - p[0])
                fy = c[1] * p[1] + (1 - c[1]) * (1 - p[1])
                fz = c[2] * p[2] + (1 - c[2]) * (1 - p[2])
                sx, sy, sz = 2 * c - 1
                out[a] = [sx * fy * fz, fx * sy * fz, -fx * fy * sz]  # last: d/dz = -d/d(depth)
            return out

        self.dNg = np.stack([dN(p) for p in gps])  # (8 gp, 8 nodes, 3), per unit cell size
        self.dNc = dN(np.array([0.5, 0.5, 0.5]))
        # ---- constraints
        free = np.ones(self.shape3 + (3,))
        free[0, :, :, 0] = 0.0
        free[:, 0, :, 1] = 0.0
        free[:, :, -1, :] = 0.0
        self.free0 = free
        # ---- bonds
        self.corr, self.stencil_kind = corr, stencil_kind
        self.terms = None
        self.ghosts = None
        if bonds:
            self._build_bonds(zone + band, zone_z + band, plant)
        self._build_diag()
        self._jit()

    # ------------------------------------------------------------------------------------------------------- bonds
    def _build_bonds(self, reach, reach_z, plant):
        h, nc, ncz = self.h, self.nc, self.ncz
        offs, n, r, wt = stencil(self.stencil_kind)
        self.offs, self.n_o = offs, n
        s = bond_scale(wt, n, self.mu, h)
        k_o = s * wt / (r * h) ** 2
        # zone nodes: uniform core, within reach (+ margin), every end inside the core
        lim = min(nc - 3, int(np.ceil(reach / h)) + 1)
        limz = min(ncz - 3, int(np.ceil(reach_z / h)) + 1)
        if reach / h > nc - 4 or reach_z / h > ncz - 4:
            raise ValueError("zone + band does not fit inside the uniform core with a horizon to spare")
        ii, jj, kk = np.meshgrid(np.arange(lim + 1), np.arange(lim + 1), np.arange(limz + 1), indexing="ij")
        ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()
        orb = np.where(ii > 0, 2, 1) * np.where(jj > 0, 2, 1)
        exist = lambda kz: kz >= 0
        # family counts (existing neighbours): only the top surface removes neighbours
        def count(kz):
            return np.sum((kz[:, None] + offs[None, :, 2]) >= 0, axis=1)
        V0 = len(offs)
        Vi = count(kk)
        Ti, Tj, Ts, Tn, Tk, Tw, Tkey = [], [], [], [], [], [], []
        Gi, Gn, Gk, Gw, Gxi = [], [], [], [], []
        for o_i, (o, no, ko) in enumerate(zip(offs, n, k_o)):
            jx, jy, jz = ii + o[0], jj + o[1], kk + o[2]
            sx = np.where(jx < 0, -1.0, 1.0)
            sy = np.where(jy < 0, -1.0, 1.0)
            jx, jy = np.abs(jx), np.abs(jy)
            mid = (h * (ii + o[0] / 2), h * (jj + o[1] / 2), -h * (kk + o[2] / 2))
            a = self.alpha(*mid)
            base = a * orb / 4.0 * 0.5
            present = jz >= 0
            # real (incl. mirrored) neighbours
            sel = present & (base > 0)
            if np.any(sel):
                if self.corr == "volume":
                    Vj = count(jz[sel])
                    c = 2 * V0 / (Vi[sel] + Vj)
                else:
                    c = 1.0
                I = self.idx[ii[sel], jj[sel], kk[sel]]
                J = self.idx[jx[sel], jy[sel], jz[sel]]
                Ti.append(I); Tj.append(J)
                Ts.append(np.stack([sx[sel], sy[sel], np.ones(sel.sum())], 1))
                Tn.append(np.repeat(no[None], sel.sum(), 0))
                Tk.append(np.full(sel.sum(), ko))
                Tw.append(base[sel] * c)
                mirrored = (ii[sel] + o[0] < 0) | (jj[sel] + o[1] < 0)
                Tkey.append(np.where(mirrored, -1, np.minimum(I, J) * self.N + np.maximum(I, J)))
            gsel = (~present) & (base > 0)
            if self.corr == "ghost" and np.any(gsel):
                Gi.append(np.stack([ii[gsel], jj[gsel], kk[gsel]], 1))
                Gn.append(np.repeat(no[None], gsel.sum(), 0))
                Gk.append(np.full(gsel.sum(), ko))
                Gw.append(base[gsel])
                Gxi.append(np.repeat((o * np.array([1, 1, -1]) * h)[None].astype(float), gsel.sum(), 0))
        T = dict(I=np.concatenate(Ti), J=np.concatenate(Tj), S=np.concatenate(Ts), n=np.concatenate(Tn),
                 k=np.concatenate(Tk), w=np.concatenate(Tw), key=np.concatenate(Tkey))
        # the offsets' z is in index space (down); the bond direction n in physical space has z flipped
        T["n"] = T["n"] * np.array([1, 1, -1.0])
        if plant is not None:  # break a fraction of whole real bonds in a region (both directed terms)
            frac, radius, layers, seed = plant
            rr = np.hypot(self.X, self.Y).ravel()
            dep = np.broadcast_to(np.arange(self.shape3[2])[None, None, :], self.shape3).ravel()
            region = lambda q: (rr[q] <= radius) & (dep[q] < layers)
            I, J = T["I"], T["J"]
            inreg = region(I) & region(J) & (T["key"] >= 0)
            key = T["key"].astype(np.uint64)
            hsh = ((key * np.uint64(2654435761) + np.uint64(seed * 97531 + 12345)) % np.uint64(1000003)).astype(float) / 1000003.0
            drop = inreg & (hsh < frac)
            self.planted_terms = int(drop.sum())
            self.planted_mask = drop
            T["w"] = np.where(drop, 0.0, T["w"])
        self.terms = {k: jnp.asarray(v) for k, v in T.items() if k != "key"}
        self.n_terms = len(T["I"])
        if Gi:
            G = dict(ijk=np.concatenate(Gi), n=np.concatenate(Gn) * np.array([1, 1, -1.0]), k=np.concatenate(Gk),
                     w=np.concatenate(Gw), xi=np.concatenate(Gxi))
            self.ghosts = {k: jnp.asarray(v) for k, v in G.items()}
            self.n_ghosts = len(G["k"])
        else:
            self.n_ghosts = 0

    # ------------------------------------------------------------------------------------------------------- energy
    def energy_parts(self, u):
        mu, Kp = self.mu, self.Kp
        U = u.reshape(self.N, 3)
        Ue = U[self.econn]  # (E, 8, 3)
        inv = 1.0 / jnp.asarray(self.esize)  # (E, 3)
        grad = jnp.einsum("eai,gad->egid", Ue, jnp.asarray(self.dNg)) * inv[:, None, None, :]  # du_i/dx_d
        eps = 0.5 * (grad + jnp.swapaxes(grad, -1, -2))
        tr = jnp.trace(eps, axis1=-2, axis2=-1)
        vol = jnp.prod(jnp.asarray(self.esize), axis=1)
        dens = mu * jnp.sum(eps * eps, axis=(-1, -2)) + 0.5 * mu * tr ** 2
        e_partner = jnp.sum((1.0 - jnp.asarray(self.ealpha)) * vol * jnp.mean(dens, axis=1))
        Gc = jnp.asarray(self.dNc)[None] * inv[:, None, :]
        trc = jnp.einsum("eai,eai->e", Ue, Gc)
        e_press = 0.5 * Kp * jnp.sum(vol * trc ** 2)
        e_bond = 0.0
        if self.terms is not None:
            T = self.terms
            d = T["S"] * U[T["J"]] - U[T["I"]]
            s = jnp.sum(T["n"] * d, axis=1)
            e_bond = jnp.sum(T["w"] * 0.5 * T["k"] * s ** 2)
        e_ghost = 0.0
        if self.ghosts is not None:
            Gh = self.ghosts
            Gm = self.nodal_gradient(u, Gh["ijk"])  # (n, 3, 3) du_i/dx_d
            s = jnp.einsum("ni,nid,nd->n", Gh["n"], Gm, Gh["xi"])
            e_ghost = jnp.sum(Gh["w"] * 0.5 * Gh["k"] * s ** 2)
        return e_partner, e_press, e_bond, e_ghost

    def nodal_gradient(self, u, ijk):
        h = self.h
        U = u.reshape(self.shape3 + (3,))
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        Sx = jnp.array([-1.0, 1.0, 1.0])
        Sy = jnp.array([1.0, -1.0, 1.0])
        up = U[i + 1, j, k]
        um = jnp.where((i > 0)[:, None], U[jnp.maximum(i - 1, 0), j, k], Sx * U[i + 1, j, k])
        gx = (up - um) / (2 * h)
        vp = U[i, j + 1, k]
        vm = jnp.where((j > 0)[:, None], U[i, jnp.maximum(j - 1, 0), k], Sy * U[i, j + 1, k])
        gy = (vp - vm) / (2 * h)
        # depth derivative, then d/dz = -d/d(depth)
        dd_c = (U[i, j, k + 1] - U[i, j, jnp.maximum(k - 1, 0)]) / (2 * h)
        dd_0 = (-3 * U[i, j, k] + 4 * U[i, j, k + 1] - U[i, j, k + 2]) / (2 * h)
        gz = -jnp.where((k > 0)[:, None], dd_c, dd_0)
        return jnp.stack([gx, gy, gz], axis=-1)  # (n, 3 comp, 3 dir)

    def energy(self, u):
        return sum(self.energy_parts(u))

    # ------------------------------------------------------------------------------------------------------- diag
    def _build_diag(self):
        mu, Kp = self.mu, self.Kp
        D = np.zeros((self.N, 3))
        inv = 1.0 / self.esize
        vol = np.prod(self.esize, 1)
        for gp in range(8):
            Gg = self.dNg[gp][None] * inv[:, None, :]  # (E, 8, 3)
            sq = np.sum(Gg ** 2, axis=2)  # |grad N|^2
            for d in range(3):
                val = (1 - self.ealpha)[:, None] * vol[:, None] / 8 * (mu * (sq + Gg[:, :, d] ** 2) + mu * Gg[:, :, d] ** 2)
                np.add.at(D[:, d], self.econn, val)
        Gc = self.dNc[None] * inv[:, None, :]
        for d in range(3):
            np.add.at(D[:, d], self.econn, Kp * vol[:, None] * Gc[:, :, d] ** 2)
        if self.terms is not None:
            T = {k: np.asarray(v) for k, v in self.terms.items()}
            same = T["I"] == T["J"]
            for d in range(3):
                kn = T["w"] * T["k"] * T["n"][:, d] ** 2
                np.add.at(D[:, d], T["I"][~same], kn[~same])
                np.add.at(D[:, d], T["J"][~same], kn[~same])
                np.add.at(D[:, d], T["I"][same], (kn * (T["S"][:, d] - 1) ** 2)[same])
        self.diag = np.maximum(D, 1e-30 * D.max())

    # ------------------------------------------------------------------------------------------------------- solve
    def _jit(self):
        self.grad = jax.jit(jax.grad(self.energy))
        self.parts = jax.jit(self.energy_parts)

        @jax.jit
        def pcg(rhs, x0, free, diag, tol, maxit):
            def A(v):
                return free * jax.grad(self.energy)(free * v)
            Minv = free / diag
            r = rhs - A(x0)
            z = Minv * r
            p = z
            rz = jnp.sum(r * z)
            b2 = jnp.sum(rhs * rhs)

            def cond(c):
                x, r, p, rz, it = c
                return (jnp.sum(r * r) > tol * tol * b2) & (it < maxit)

            def body(c):
                x, r, p, rz, it = c
                Ap = A(p)
                al = rz / jnp.sum(p * Ap)
                x = x + al * p
                r = r - al * Ap
                z = Minv * r
                rz2 = jnp.sum(r * z)
                p = z + (rz2 / rz) * p
                return x, r, p, rz2, it + 1

            x, r, p, rz, it = jax.lax.while_loop(cond, body, (x0, r, p, rz, 0))
            return x, it, jnp.sqrt(jnp.sum(r * r) / b2)

        self._pcg = pcg

    def solve(self, up, free, x0=None, tol=1e-10, maxit=40000):
        """u = up on prescribed dofs (free = 0), equilibrium elsewhere."""
        free = jnp.asarray(free.reshape(-1))
        up = jnp.asarray(up.reshape(-1))
        rhs = -free * self.grad(up)
        x0 = jnp.zeros_like(up) if x0 is None else jnp.asarray(x0.reshape(-1)) * free
        x, it, res = self._pcg(rhs, x0, free, jnp.asarray(self.diag.reshape(-1)), tol, maxit)
        u = up * (1 - free) + x
        return np.asarray(u).reshape(self.shape3 + (3,)), int(it), float(res)


# --------------------------------------------------------------------------------------------------------------- contact
def sneddon_sphere(d, R, Estar):
    """Frictionless rigid sphere on a half-space (Sneddon 1965): d = (a/2) ln((R+a)/(R-a)),
    F = (E*/2) [(a^2 + R^2) ln((R+a)/(R-a)) - 2 a R]."""
    lo, hi = 0.0, R * (1 - 1e-12)
    for _ in range(200):
        a = 0.5 * (lo + hi)
        if 0.5 * a * np.log((R + a) / (R - a)) < d:
            lo = a
        else:
            hi = a
    a = 0.5 * (lo + hi)
    F = 0.5 * Estar * ((a * a + R * R) * np.log((R + a) / (R - a)) - 2 * a * R)
    return F, a


def indent_sphere(B, R, depths, r_cand, log=print):
    top = (B.X[..., 0] ** 2 + B.Y[..., 0] ** 2) ** 0.5  # (nx, nx) at iz = 0
    cand = top <= r_cand
    gap = R - np.sqrt(np.maximum(R * R - top ** 2, 0.0))
    out = []
    u = None
    active = None
    E = 2 * B.mu * (1 + B.nu)
    Estar = E / (1 - B.nu ** 2)
    for d in depths:
        Fs, a_s = sneddon_sphere(d, R, Estar)
        if active is None:
            active = cand & (top <= a_s)
        target = -d + gap
        hist = []
        for it in range(40):
            free = B.free0.copy()
            free[:, :, 0, 2][active] = 0.0
            up = np.zeros(B.shape3 + (3,))
            up[:, :, 0, 2] = np.where(active, target, 0.0)
            x0 = None if u is None else u
            u, ncg, res = B.solve(up, free, x0=x0)
            f = np.asarray(B.grad(jnp.asarray(u.reshape(-1)))).reshape(B.shape3 + (3,))
            fz = f[:, :, 0, 2]
            release = active & (fz > 0)
            pen = cand & (~active) & (u[:, :, 0, 2] > target + 1e-9 * d)
            hist.append((int(active.sum()), int(ncg), res, int(release.sum()), int(pen.sum())))
            if not release.any() and not pen.any():
                break
            active = (active & ~release) | pen
        F = -4.0 * float(np.sum(fz[active]))
        a_num = float(np.max(top[active])) if active.any() else 0.0
        out.append(dict(d=d, F=F, F_sneddon=Fs, a_sneddon=a_s, a_max_node=a_num, active=int(active.sum()),
                        iterations=hist))
        log(f"  d {d:.4e}  F {F:.6e}  sneddon {Fs:.6e}  ratio {F / Fs:.5f}  active {int(active.sum())}  its {len(hist)}"
            f"  cg {[h[1] for h in hist]}")
    return out, u


def punch(B, a, d):
    top = (B.X[..., 0] ** 2 + B.Y[..., 0] ** 2) ** 0.5
    act = top <= a * (1 + 1e-9)
    free = B.free0.copy()
    free[:, :, 0, 2][act] = 0.0
    up = np.zeros(B.shape3 + (3,))
    up[:, :, 0, 2] = np.where(act, -d, 0.0)
    u, ncg, res = B.solve(up, free)
    f = np.asarray(B.grad(jnp.asarray(u.reshape(-1)))).reshape(B.shape3 + (3,))
    F = -4.0 * float(np.sum(f[:, :, 0, 2][act]))
    return F, u, ncg, res, int(act.sum())
