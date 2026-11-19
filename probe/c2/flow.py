"""The unsteady collocated step on a face mesh whose boundary faces are all no-slip walls (a periodic tube round a held
body): explicit SSP-RK3 momentum (central convection of the face-mean velocity by a face flux, the over-relaxed diffusive
flux with its Gauss-gradient cross term), and a projection onto face fluxes that sum to zero in every cell.

Projection schemes (``scheme``):
  ``p3``  three projections a step: every Euler substep of the RK3 is projected, no pressure in the momentum stage;
  ``l1``  one projection a step: the three stages carry the gradient of the lagged pressure ``p^n``;
  ``x1``  one projection a step: the stages carry the gradient of the extrapolated pressure ``2 p^n - p^(n-1)``.
Face flux (``rc``, ``carry``):
  ``carry``  the first stage is convected by the projected face flux kept from the last projection (otherwise by the
             flux interpolated from the cells);
  ``rc``     Rhie-Chow: every flux interpolated from the cells carries ``dt [(grad p*)_f . S - g (p*_n - p*_o)]`` of the
             stage pressure ``p*`` (the compact face gradient in place of the interpolated cell gradient).
The pressure equation is the compact one, ``sum_f g (psi_i - psi_nb) = - sum_f F*_f`` with ``psi = dt (p - p*)``.

All fields are kinematic (pressure over density). Tables are passed as arguments."""
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import lax

from gfv import amg, prims


class Flow:
    def __init__(self, mesh, impl):
        self.mesh, self.impl, self.n = mesh, impl, mesh.n
        n, inner = mesh.n, mesh.inner
        own, nei = mesh.own[inner], mesh.nei[inner]
        S, d, w = mesh.S[inner], mesh.d[inner], mesh.w[inner]
        g = np.einsum("fi,fi->f", S, S) / np.einsum("fi,fi->f", S, d)
        k = S - g[:, None] * d
        bo, Sb, xfb, pb = mesh.own[~inner], mesh.S[~inner], mesh.xf[~inner], mesh.patch[~inner]
        areab = np.linalg.norm(Sb, axis=1)
        dn = np.einsum("fi,fi->f", xfb - mesh.xc[bo], Sb) / areab
        gb = areab / dn
        wall_g = np.bincount(bo, gb, n)
        wallS = np.zeros((n, 3))
        np.add.at(wallS, bo, Sb)
        body = pb == 1
        bodyS = np.zeros((n, 3))
        np.add.at(bodyS, bo[body], Sb[body])
        self.m = own.size
        self.own, self.nei, self.g = own, nei, g
        rows = np.arange(self.m)
        B = sp.csr_matrix((np.concatenate([-np.ones(self.m), np.ones(self.m)]), (np.concatenate([rows, rows]), np.concatenate([own, nei]))),
                          shape=(self.m, n))
        self.A = (B.T @ sp.diags(g) @ B).tocsr()            # (A psi)_i = sum_f g (psi_i - psi_nb): positive semidefinite
        tab = {"vol": mesh.V, "S": S, "k": k, "g": g, "w": w, "wall_g": wall_g, "wallS": wallS, "bodyS": bodyS,
               "body_g": np.bincount(bo[body], gb[body], n), "adiag": self.A.diagonal()}
        tab["gu"], self._gather = prims.make_gather(np.stack([own, nei]), n, impl)
        tab["net"], self._net = prims.make_net(own, nei, n, impl)
        self.tab = tab
        lam_v = 2.0 * (np.bincount(own, g, n) + np.bincount(nei, g, n) + wall_g) / mesh.V
        area = np.linalg.norm(S, axis=1)
        lam_c = 0.5 * (np.bincount(own, area, n) + np.bincount(nei, area, n)) / mesh.V
        self.lam_v, self.lam_c = lam_v, lam_c

    def stable_dt(self, nu, speed, safety=0.6):
        """The step under SSP-RK3's limits (2.51 on the real axis, 1.73 on the imaginary) from each cell's Gershgorin
        radii, and which limit binds where."""
        rate = nu * self.lam_v / 2.51 + speed * self.lam_c / 1.73
        i = int(np.argmax(rate))
        rho = np.hypot(self.mesh.xc[:, 0], self.mesh.xc[:, 1])
        return safety / rate[i], {"binding_cell_rho": float(rho[i]), "binding_cell_z": float(self.mesh.xc[i, 2]),
                                  "viscous_share": float(nu * self.lam_v[i] / 2.51 / rate[i]),
                                  "dt_viscous_only": float(safety * 2.51 / (nu * self.lam_v.max())),
                                  "dt_convective_only": float(safety * 1.73 / (speed * self.lam_c.max())),
                                  "dt_viscous_p50": float(safety * 2.51 / (nu * np.median(self.lam_v))),
                                  "dt_convective_p50": float(safety * 1.73 / (speed * np.median(self.lam_c)))}

    # ---- operators ---------------------------------------------------------------------------------------------------
    def ends(self, tab, x):
        e = self._gather(tab["gu"], x)
        return e[0], e[1]

    def cell_flux(self, tab, u):
        uo, un = self.ends(tab, u)
        w = tab["w"][:, None]
        return ((w * uo + (1.0 - w) * un) * tab["S"]).sum(axis=-1)

    def grad_and_jump(self, tab, p):
        """The cells' Gauss gradient of a cell scalar (its wall value the cell's) and ``g (p_n - p_o)`` on faces."""
        po, pn = self.ends(tab, p)
        w = tab["w"]
        grad = (self._net(tab["net"], (w * po + (1.0 - w) * pn)[:, None] * tab["S"]) + p[:, None] * tab["wallS"]) / tab["vol"][:, None]
        return grad, tab["g"] * (pn - po)

    def face_dot(self, tab, vec):
        """A cell vector interpolated to faces, dotted with the area vector."""
        vo, vn = self.ends(tab, vec)
        w = tab["w"][:, None]
        return ((w * vo + (1.0 - w) * vn) * tab["S"]).sum(axis=-1)

    def rhs(self, tab, par, u, flux, extra, nonorth=True):
        """``du/dt`` of the momentum stage without its pressure, and the flux interpolated from ``u``. ``flux`` convects
        if given; otherwise the interpolated flux plus ``extra`` does."""
        nu, force = par["nu"], par["force"]
        uo, un = self.ends(tab, u)
        w = tab["w"][:, None]
        ufl = w * uo + (1.0 - w) * un
        fc = (ufl * tab["S"]).sum(axis=-1)
        F = fc + extra if flux is None else flux
        msg = nu * tab["g"][:, None] * (un - uo) - F[:, None] * (0.5 * (uo + un))
        if nonorth:
            S = tab["S"]
            grad = self._net(tab["net"], ufl[:, :, None] * S[:, None, :]) / tab["vol"][:, None, None]
            go, gn = self.ends(tab, grad.reshape(self.n, 9))
            gf = (w * go + (1.0 - w) * gn).reshape(-1, 3, 3)
            k = tab["k"]
            msg = msg + nu * (gf[:, :, 0] * k[:, 0:1] + gf[:, :, 1] * k[:, 1:2] + gf[:, :, 2] * k[:, 2:3])
        R = (self._net(tab["net"], msg) - nu * tab["wall_g"][:, None] * u) / tab["vol"][:, None] + force
        return R, fc

    def body_force(self, tab, par, u, p):
        """The fluid's force on the body over the density: pressure and viscous parts, each (3,)."""
        return (p[:, None] * tab["bodyS"]).sum(axis=0), par["nu"] * (tab["body_g"][:, None] * u).sum(axis=0)

    def divergence(self, tab, F):
        return self._net(tab["net"], F) / tab["vol"]


# ---------------------------------------------------------------------------------------------------------------------
# the pressure solve: preconditioned CG from a first guess, with a residual or an energy-norm stop
# ---------------------------------------------------------------------------------------------------------------------
def pcg(apply, precondition, r0, tol_abs, maxiter, energy=None):
    """CG for ``A dx = r0`` from zero (``r0`` mean-free; ``A`` singular on constants). Stops at ``|r| <= tol_abs`` or,
    with ``energy = (eta, delay, e0)``, when the Hestenes-Stiefel estimate of the squared energy-norm error ``delay``
    iterations ago is under ``eta^2`` times the solution's squared energy norm so far (``e0`` the first guess's).
    Returns ``(dx, r, iterations)``."""
    fix = lambda v: v - jnp.mean(v)
    z = fix(precondition(r0))
    dtype = r0.dtype
    delay = 1 if energy is None else int(energy[1])
    ring = jnp.zeros(delay, dtype)
    state = (jnp.zeros_like(r0), r0, z, jnp.vdot(r0, z), jnp.zeros((), jnp.int32), ring, jnp.zeros((), dtype), z)

    def cond(s):
        _, r, _, rz, k, ring, total, _ = s
        go = (k < maxiter) & (jnp.linalg.norm(r) > tol_abs) & (rz > 0)
        if energy is not None:
            done = (k >= delay) & (jnp.sum(ring) <= energy[0] ** 2 * (energy[2] + total))
            go = go & ~done
        return go

    def body(s):
        x, r, p, rz, k, ring, total, _ = s
        q = apply(p)
        alpha = rz / jnp.vdot(p, q)
        x, r = x + alpha * p, fix(r - alpha * q)
        z = fix(precondition(r))
        rz_new = jnp.vdot(r, z)
        ring = ring.at[k % delay].set(alpha * rz)
        return (x, r, z + (rz_new / rz) * p, rz_new, k + 1, ring, total + alpha * rz, z)

    x, r, _, _, k, _, _, _ = lax.while_loop(cond, body, state)
    return x, r, k


class Solver:
    """``guess``: ``zero``, ``prev`` (the last solution) or ``proj``: the best combination in the energy norm of past
    solutions (Fischer 1998), in one of two forms. ``fold="solutions"`` keeps the last ``depth`` solutions with their
    images and orthonormalises them, newest first, at every solve (``depth (depth + 1) / 2`` inner products and twice
    as many updates). ``fold="restart"`` is Fischer's own: an A-orthonormal basis that grows by each solve's correction
    and, when full, becomes the current solution alone. Two forms tried first and dropped [measured, local]: dropping
    the oldest correction outright (12 iterations against 14 from zero), and merging the two oldest (10): the span then
    no longer holds the past solutions, only the current one."""

    def __init__(self, A, dtype, guess="zero", depth=0, tol=1e-8, stop="res", eta=1e-3, delay=4, cycle32=False, theta=0.05,
                 maxiter=400, fold="solutions"):
        self.guess, self.depth, self.tol, self.stop, self.eta, self.delay = guess, int(depth), tol, stop, eta, int(delay)
        self.cycle32, self.maxiter, self.n, self.fold = cycle32, maxiter, A.shape[0], fold
        H, self.facts = amg.hierarchy(A, theta=theta)
        self.E = amg.ell(A)
        self.H = H
        self.dtype = dtype

    def tables(self):
        dev = lambda t, dt: jax.tree_util.tree_map(
            lambda a: jnp.asarray(a, dtype=dt if np.issubdtype(np.asarray(a).dtype, np.floating) else np.int32), t)
        return {"E": dev(self.E, self.dtype), "H": dev(self.H, jnp.float32 if self.cycle32 else self.dtype)}

    def history(self):
        k = max(self.depth, 1)
        z = jnp.zeros((k, self.n), self.dtype)
        return {"W": z, "AW": z, "ptr": jnp.zeros((), jnp.int32), "prev": jnp.zeros(self.n, self.dtype),
                "count": jnp.zeros((), jnp.int32)}

    def solve(self, T, b, hist, tol=None, skip=False, use=True):
        """``A x = b``; returns ``(x, iterations, |b - A x| / |b|, |b|, hist)``. ``skip``: no iteration at all (a timing arm)."""
        tol = self.tol if tol is None else tol
        dtype = b.dtype
        apply = lambda v: amg.matvec(T["E"], v)
        if self.cycle32:
            def precondition(r):
                s = jnp.maximum(jnp.linalg.norm(r), jnp.finfo(dtype).tiny)
                return amg.vcycle(T["H"], (r / s).astype(jnp.float32)).astype(dtype) * s
        else:
            precondition = lambda r: amg.vcycle(T["H"], r)
        b = b - jnp.mean(b)
        bnorm = jnp.linalg.norm(b)
        e0 = jnp.zeros((), dtype)
        if self.guess == "zero":
            x0, r0 = jnp.zeros_like(b), b
        elif self.guess == "prev":
            x0 = jnp.where(use, hist["prev"], 0.0)
            ax = apply(x0)
            r0 = b - ax
            e0 = jnp.vdot(x0, ax)
        elif self.fold == "restart":
            c = jnp.where(use, hist["W"] @ b, 0.0)
            x0, r0 = c @ hist["W"], b - c @ hist["AW"]
            e0 = jnp.vdot(c, c)
        else:
            Q, AQ = [], []
            x0, r0 = jnp.zeros_like(b), b
            for i in range(self.depth):                               # newest first
                idx = (hist["ptr"] - 1 - i) % self.depth
                q, aq = hist["W"][idx], hist["AW"][idx]
                ref = jnp.vdot(q, aq)
                for qj, aqj in zip(Q, AQ):
                    hcoef = jnp.vdot(qj, aq)
                    q, aq = q - hcoef * qj, aq - hcoef * aqj
                n2 = jnp.vdot(q, aq)
                ok = (n2 > 1e-10 * ref) & (ref > 0) & use      # below this a direction is rounding, amplified
                sc = jnp.where(ok, 1.0 / jnp.sqrt(jnp.where(ok, n2, 1.0)), 0.0)
                q, aq = q * sc, aq * sc
                Q.append(q), AQ.append(aq)
                ci = jnp.vdot(q, b)
                x0, r0, e0 = x0 + ci * q, r0 - ci * aq, e0 + ci * ci
        r0 = r0 - jnp.mean(r0)
        energy = (self.eta, self.delay, e0) if self.stop == "energy" else None
        dx, r, its = pcg(apply, precondition, r0, tol * bnorm, 0 if skip else self.maxiter, energy)
        x = x0 + dx
        if self.guess == "proj" and self.fold == "solutions":
            ptr = hist["ptr"]
            W = hist["W"].at[ptr].set(jnp.where(use, x, hist["W"][ptr]))
            AW = hist["AW"].at[ptr].set(jnp.where(use, b - r, hist["AW"][ptr]))
            hist = dict(hist, W=W, AW=AW, ptr=jnp.where(use, (ptr + 1) % self.depth, ptr))
        elif self.guess == "proj":
            adx = r0 - r
            c1 = hist["W"] @ adx
            wv, aw = dx - c1 @ hist["W"], adx - c1 @ hist["AW"]
            nrm2 = jnp.vdot(wv, aw)
            ok = (nrm2 > 1e-24 * jnp.maximum(jnp.vdot(x, b), jnp.finfo(dtype).tiny)) & use
            scale = jnp.where(ok, 1.0 / jnp.sqrt(jnp.where(ok, nrm2, 1.0)), 0.0)
            W, AW = hist["W"], hist["AW"]
            full = hist["count"] >= self.depth
            ax = b - r
            xs = 1.0 / jnp.sqrt(jnp.maximum(jnp.vdot(x, ax), jnp.finfo(dtype).tiny))
            slot = jnp.where(full, 0, hist["count"])
            W = jnp.where(full & ok, jnp.zeros_like(W), W).at[slot].set(jnp.where(ok, jnp.where(full, x * xs, wv * scale), W[slot]))
            AW = jnp.where(full & ok, jnp.zeros_like(AW), AW).at[slot].set(jnp.where(ok, jnp.where(full, ax * xs, aw * scale), AW[slot]))
            hist = dict(hist, W=W, AW=AW, count=jnp.where(ok, jnp.where(full, 1, hist["count"] + 1), hist["count"]))
        elif self.guess == "prev":
            hist = dict(hist, prev=jnp.where(use, x, hist["prev"]))
        return x, its, jnp.linalg.norm(r) / jnp.maximum(bnorm, jnp.finfo(dtype).tiny), bnorm, hist


# ---------------------------------------------------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------------------------------------------------
def initial_state(flow, solver, u0, dtype):
    n, m = flow.n, flow.m
    z = jnp.zeros(n, dtype)
    return {"u": jnp.asarray(u0, dtype), "p": z, "gp": jnp.zeros((n, 3), dtype), "p_old": z, "gp_old": jnp.zeros((n, 3), dtype),
            "F": jnp.zeros(m, dtype), "hist": solver.history(), "k": jnp.zeros((), jnp.int32)}


def make_step(flow, solver, scheme, rc=False, carry=True, nonorth=True, skip_solve=False, stride=1):
    """``step(tab, T, par, state) -> (state, out)``; ``par``: ``nu``, ``force`` (3,), ``dt``. ``out`` holds each solve's
    iterations, relative residual and right-hand-side norm."""

    def project(tab, T, us, Fs, hist, use):
        b = -flow._net(tab["net"], Fs)
        psi, its, res, bnorm, hist = solver.solve(T, b, hist, skip=skip_solve, use=use)
        gpsi, jump = flow.grad_and_jump(tab, psi)
        return us - gpsi, Fs - jump, psi, gpsi, hist, (its, res, bnorm)

    if scheme == "p3":
        def step(tab, T, par, state):
            dt = par["dt"]
            u, F, hist = state["u"], state["F"], state["hist"]
            zero = jnp.zeros((), u.dtype)
            use = state["k"] % stride == 0

            def euler(v, Fv, hist):
                R, fc = flow.rhs(tab, par, v, Fv if carry else None, zero, nonorth)
                vs = v + dt * R
                return project(tab, T, vs, flow.cell_flux(tab, vs), hist, use)

            v1, F1, _, _, hist, o1 = euler(u, F, hist)
            w2, G2, _, _, hist, o2 = euler(v1, F1, hist)
            v2, F2 = 0.75 * u + 0.25 * w2, 0.75 * F + 0.25 * G2
            w3, G3, psi, gpsi, hist, o3 = euler(v2, F2, hist)
            un, Fn = u / 3.0 + (2.0 / 3.0) * w3, F / 3.0 + (2.0 / 3.0) * G3
            new = {"u": un, "p": psi / dt, "gp": gpsi / dt, "p_old": state["p"], "gp_old": state["gp"], "F": Fn, "hist": hist,
                   "k": state["k"] + 1}
            out = {"its": jnp.stack([o1[0], o2[0], o3[0]]), "res": jnp.stack([o1[1], o2[1], o3[1]]),
                   "bnorm": jnp.stack([o1[2], o2[2], o3[2]]), "use": use}
            return new, out
        return step

    def step(tab, T, par, state):
        dt = par["dt"]
        u, F, hist = state["u"], state["F"], state["hist"]
        if scheme == "l1":
            ps, gps = state["p"], state["gp"]
        else:
            ps, gps = 2.0 * state["p"] - state["p_old"], 2.0 * state["gp"] - state["gp_old"]
        if rc:
            _, jump = flow.grad_and_jump(tab, ps)
            extra = dt * (flow.face_dot(tab, gps) - jump)
        else:
            extra = jnp.zeros((), u.dtype)
        R1, fc0 = flow.rhs(tab, par, u, F if carry else None, extra, nonorth)
        v1 = u + dt * (R1 - gps)
        R2, _ = flow.rhs(tab, par, v1, None, extra, nonorth)
        v2 = 0.75 * u + 0.25 * (v1 + dt * (R2 - gps))
        R3, _ = flow.rhs(tab, par, v2, None, extra, nonorth)
        us = u / 3.0 + (2.0 / 3.0) * (v2 + dt * (R3 - gps))
        use = state["k"] % stride == 0
        un, Fn, psi, gpsi, hist, o = project(tab, T, us, flow.cell_flux(tab, us) + extra, hist, use)
        new = {"u": un, "p": ps + psi / dt, "gp": gps + gpsi / dt, "p_old": state["p"], "gp_old": state["gp"], "F": Fn, "hist": hist,
               "k": state["k"] + 1}
        return new, {"its": o[0][None], "res": o[1][None], "bnorm": o[2][None], "use": use}
    return step


def diagnostics(flow, tab, par, state, sectors, rc):
    """Scalars of a state: the body's force, the flow rate's mean axial velocity, the kinetic energy, the divergence of
    the projected flux and of the flux interpolated from the cells (with its Rhie-Chow term if ``rc``), and the two
    checkerboard measures of the pressure."""
    u, p, F, vol = state["u"], state["p"], state["F"], tab["vol"]
    fp, fv = flow.body_force(tab, par, u, p)
    total = jnp.sum(vol)
    div = flow.divergence(tab, F)
    fc = flow.cell_flux(tab, u)
    if rc:
        _, jump = flow.grad_and_jump(tab, p)
        fc = fc + par["dt"] * (flow.face_dot(tab, state["gp"]) - jump)
    divc = flow.divergence(tab, fc)
    l2 = lambda a: jnp.sqrt(jnp.sum(vol * a * a) / total)
    return {"fp": fp, "fv": fv, "uz_mean": jnp.sum(vol * u[:, 2]) / total, "ke": 0.5 * jnp.sum(vol[:, None] * u * u),
            "div_l2": l2(div), "div_max": jnp.max(jnp.abs(div)), "divc_l2": l2(divc), "divc_max": jnp.max(jnp.abs(divc)),
            **checkerboard(flow, tab, p, sectors)}


def checkerboard(flow, tab, p, sectors):
    """``cb_rough``: the pressure less its neighbours' weighted mean, over the pressure's own variation (volume-weighted
    norms). ``cb_odd``: the same ratio for the pressure's odd-even mode round the sweep, ``(-1)^k`` over the sectors."""
    vol = tab["vol"]
    total = jnp.sum(vol)
    pm = p - jnp.sum(vol * p) / total
    scale = jnp.sqrt(jnp.sum(vol * pm * pm) / total) + jnp.finfo(p.dtype).tiny
    po, pn = flow.ends(tab, p)
    rough = flow._net(tab["net"], tab["g"] * (po - pn)) / tab["adiag"]
    sign = jnp.where(jnp.arange(sectors) % 2 == 0, 1.0, -1.0).astype(p.dtype)
    odd = (p.reshape(-1, sectors) * sign).sum(axis=1, keepdims=True) / sectors * sign
    v2 = vol.reshape(-1, sectors)
    return {"cb_rough": jnp.sqrt(jnp.sum(vol * rough * rough) / total) / scale,
            "cb_odd": jnp.sqrt(jnp.sum(v2 * odd * odd) / total) / scale, "p_scale": scale}
