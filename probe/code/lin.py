"""r39 linear reference solver: Stokes (s = 0) and Brinkman (s > 0) resistance S (6x6) and F_0 on body-fitted curved Taylor-Hood
P_k/P_{k-1} tetrahedra (NGSolve + Netgen OCC). Units mu = a = 1, U_m = 1.

Problem for each column of S: s u - div(2 eps(u)) + grad p = 0, div u = 0; u = rigid twist on the body; 0 on walls, window
ends and containers. F_0: the disturbance problem with u = -u_bg on the body (u_bg the body-free vessel flow): it is F_0 for
Stokes and, for s > 0, one step of the step operator from u^n = u_bg with the body held (declared in work/PLAN.md section 2).
Forces: the residual (reaction) of the Galerkin system on the body's Dirichlet dofs, tested with the rigid modes; for a
Galerkin solve this equals the energy form a(u_i, u_j). Mirror planes: one factorisation per parity class.
"""
import os, sys, time, json, resource
import numpy as np
from ngsolve import (Mesh, VectorH1, H1, NumberSpace, BilinearForm, GridFunction, CoefficientFunction, InnerProduct, Sym,
                     Grad, div, dx, x, y, z, SetNumThreads, TaskManager, BND, Integrate)
from netgen.occ import OCCGeometry
from netgen.meshing import MeshingParameters

AX = {"x": 0, "y": 1, "z": 2}


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def mode_cf(i, c):
    """rigid mode i (0-2 translation x,y,z; 3-5 rotation about x,y,z through c) as a vector CF."""
    if i < 3:
        e = [0.0, 0.0, 0.0]
        e[i] = 1.0
        return CoefficientFunction(tuple(e))
    rx, ry, rz = x - c[0], y - c[1], z - c[2]
    k = i - 3
    if k == 0:
        return CoefficientFunction((0, -rz, ry))
    if k == 1:
        return CoefficientFunction((rz, 0, -rx))
    return CoefficientFunction((-ry, rx, 0))


def mode_parity(i, n):
    """parity of rigid mode i under the mirror with normal axis n: +1 even, -1 odd."""
    if i < 3:
        return -1 if i == n else +1
    return +1 if (i - 3) == n else -1


def sym_flags(par, planes):
    """component Dirichlet flags on the mirror planes for a class with parities par[n] (dict n -> +-1)."""
    comp = {0: [], 1: [], 2: []}
    for pl in planes:
        n = AX[pl]
        name = "sym" + pl
        if par[n] > 0:
            comp[n].append(name)
        else:
            for j in range(3):
                if j != n:
                    comp[j].append(name)
    return comp


def grade_points_sphere(wall, c, gap, cfac, h_body, gmax=0.6, hmin=None):
    """RestrictH cloud for a sphere: rings round the contact direction, spacing set by the local gap."""
    eps = 1e-6
    g0 = wall.sdf(np.atleast_2d(c))[0]
    grad = np.array([(wall.sdf(np.atleast_2d(c + eps * e))[0] - wall.sdf(np.atleast_2d(c - eps * e))[0]) / (2 * eps)
                     for e in np.eye(3)])
    n = -grad / np.linalg.norm(grad)
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    pts = []
    psi = 0.0
    while psi < np.pi:
        nph = max(1, int(np.ceil(2 * np.pi * np.sin(psi) / max(1e-9, step if psi > 0 else 1))) if psi > 0 else 1)
        ph = np.linspace(0, 2 * np.pi, nph, endpoint=False)
        ring = c + np.cos(psi) * n + np.sin(psi) * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)
        g = wall.sdf(ring)
        gr = float(np.min(g))
        near = g < gmax
        for p in ring[~near]:
            pts.append((p, h_body))
        if near.any():
            R_ = ring[near]
            gw = np.stack([(wall.sdf(R_ + eps * e) - wall.sdf(R_ - eps * e)) / (2 * eps) for e in np.eye(3)], -1)
            gw /= np.linalg.norm(gw, axis=1, keepdims=True)
            for p, gg, gv in zip(R_, g[near], gw):
                hl = min(max(cfac * gg, hmin if hmin else 0.0), h_body)
                pts.append((p, hl))
                pts.append((p - 0.5 * gg * gv, hl))
                pts.append((p - gg * gv, hl))
        step = min(0.5 * max(cfac * gr, hmin if hmin else 0.0), 0.7 * h_body) if gr < gmax else 0.7 * h_body
        step = max(step, 1e-5)
        psi += step
    return pts


def make_mesh(shape, pts, maxh, grading=0.3, order=2):
    geo = OCCGeometry(shape)
    mp = MeshingParameters(maxh=maxh, grading=grading)
    for p, h in pts:
        mp.RestrictH(float(p[0]), float(p[1]), float(p[2]), float(h))
    t0 = time.time()
    ng = geo.GenerateMesh(mp=mp)
    mesh = Mesh(ng)
    tm = time.time() - t0
    mesh.Curve(order)
    return mesh, tm


def background(mesh, order, inlet, R, planes, inverse="pardiso", log=print):
    """Body-free Stokes flow on the whole two-material mesh (fluid + the body's interior as fluid): Poiseuille 2(1 - r^2/R^2)
    on the inlet (r measured from the inlet's centre line), no slip on the walls, do-nothing outlets (Laplacian form), mirror
    planes even. Returns (velocity GridFunction, info)."""
    t0 = time.time()
    p0, tdir = np.asarray(inlet[0], float), np.asarray(inlet[1], float)
    comp = sym_flags({AX[pl]: +1 for pl in planes}, planes)
    V = VectorH1(mesh, order=order, dirichlet="wall|carina|inlet", dirichletx="|".join(comp[0]), dirichlety="|".join(comp[1]),
                 dirichletz="|".join(comp[2]))
    Q = H1(mesh, order=order - 1)
    Xs = V * Q
    (u, p), (v, q) = Xs.TnT()
    a = BilinearForm(Xs, symmetric=True)
    a += (InnerProduct(Grad(u), Grad(v)) - div(u) * q - div(v) * p) * dx
    with TaskManager():
        a.Assemble()
    gf = GridFunction(Xs)
    rx, ry, rz = x - p0[0], y - p0[1], z - p0[2]
    along = rx * tdir[0] + ry * tdir[1] + rz * tdir[2]
    r2 = rx * rx + ry * ry + rz * rz - along * along
    prof = 2.0 * (1 - r2 / R ** 2)
    gf.components[0].Set(CoefficientFunction(tuple(prof * float(tdir[i]) for i in range(3))), definedon=mesh.Boundaries("inlet"))
    r = gf.vec.CreateVector()
    r.data = -a.mat * gf.vec
    gf.vec.data += a.mat.Inverse(Xs.FreeDofs(), inverse=inverse) * r
    ub = gf.components[0]
    flux_in = Integrate(InnerProduct(ub, CoefficientFunction(tuple(float(t) for t in tdir))), mesh, BND, definedon=mesh.Boundaries("inlet"))
    info = dict(ndof=Xs.ndof, t=time.time() - t0, rss_gib=rss_gib(), inlet_flux=float(flux_in))
    log(json.dumps({"background": info}))
    return ub, info


def solve_linear(mesh, order, s, c, planes, mult, modes=range(6), bg=None, bg_parity=None, dirichlet="wall|carina|body|inlet|outlet|outer",
                 inverse="pardiso", log=print, fluid_only=False):
    """Returns dict with S (6x6, full-domain values), F0 (6) if bg, ndof per class, timings."""
    planes = list(planes)
    classes = {}
    for i in modes:
        par = tuple(mode_parity(i, AX[pl]) for pl in planes)
        classes.setdefault(par, []).append(("m", i))
    if bg is not None:
        par = tuple(bg_parity[pl] for pl in planes)
        classes.setdefault(par, []).append(("bg", None))
    S = np.zeros((6, 6))
    F0 = np.zeros(6)
    info = {"classes": [], "order": order, "s": s}
    for par, items in classes.items():
        t0 = time.time()
        pard = {AX[pl]: par[k] for k, pl in enumerate(planes)}
        comp = sym_flags(pard, planes)
        dirich = dirichlet + "|outletA|outletB"
        kw = dict(definedon="fluid") if fluid_only else {}
        V = VectorH1(mesh, order=order, dirichlet=dirich, dirichletx="|".join(comp[0]), dirichlety="|".join(comp[1]),
                     dirichletz="|".join(comp[2]), **kw)
        Q = H1(mesh, order=order - 1, **kw)
        needs_num = not any(p < 0 for p in par)
        if needs_num:
            N = NumberSpace(mesh, **kw)
            Xs = V * Q * N
            (u, p, lam), (v, q, mu_) = Xs.TnT()
        else:
            Xs = V * Q
            (u, p), (v, q) = Xs.TnT()
        form = 2 * InnerProduct(Sym(Grad(u)), Sym(Grad(v))) * dx - div(u) * q * dx - div(v) * p * dx
        if s > 0:
            form += s * InnerProduct(u, v) * dx
        if needs_num:
            form += (p * mu_ + q * lam) * dx
        a = BilinearForm(Xs, symmetric=True, condense=False)
        a += form
        with TaskManager():
            a.Assemble()
        t_as = time.time() - t0
        inv = a.mat.Inverse(Xs.FreeDofs(), inverse=inverse)
        t_fac = time.time() - t0 - t_as
        sols = {}
        for kind, i in items:
            gf = GridFunction(Xs)
            cf = mode_cf(i, c) if kind == "m" else -bg
            gf.components[0].Set(cf, definedon=mesh.Boundaries("body"))
            r = gf.vec.CreateVector()
            r.data = -a.mat * gf.vec
            gf.vec.data += inv * r
            sols[(kind, i)] = gf
        # reactions: r = A x on the body's Dirichlet dofs; force on the body along mode j = -<A x, w_j>
        Ax = {}
        for key, gf in sols.items():
            v_ = gf.vec.CreateVector()
            v_.data = a.mat * gf.vec
            Ax[key] = v_
        ws = {}
        for j in modes:
            if all(mode_parity(j, AX[pl]) == par[k] for k, pl in enumerate(planes)):
                w = GridFunction(Xs)
                w.components[0].Set(mode_cf(j, c), definedon=mesh.Boundaries("body"))
                ws[j] = w
        for key, axv in Ax.items():
            for j, w in ws.items():
                val = -mult * InnerProduct(axv, w.vec)
                if key[0] == "m":
                    S[j, key[1]] = -val          # resistance: force = -S U
                else:
                    F0[j] = val
        cinfo = {"parity": par, "items": [str(k) for k in items], "ndof": Xs.ndof, "free": int(sum(Xs.FreeDofs())),
                 "t_assemble": t_as, "t_factor_solve": time.time() - t0 - t_as, "rss_gib": rss_gib()}
        info["classes"].append(cinfo)
        log(json.dumps(cinfo))
        del inv, a
    info["S"] = S.tolist()
    if bg is not None:
        info["F0"] = F0.tolist()
    return info


def helix_surface(H, t, ps):
    """points on H's tube: X(t) + rho (cos ps N + sin ps Bn), N, Bn the Frenet-free normal pair of the helix."""
    X = H.X(t); T = H.dX(t); T = T / np.linalg.norm(T, axis=-1, keepdims=True)
    N = np.stack([-np.cos(t), -np.sin(t), np.zeros_like(t)], -1)      # principal normal of a helix
    Bn = np.cross(T, N)
    return X + H.rho * (np.cos(ps)[..., None] * N + np.sin(ps)[..., None] * Bn)


def grade_points_param(wall, surf, u_rng, v_rng, scale_u, scale_v, cfac, h_body, gmax=0.6, n0=(400, 24), maxlev=14, hmin=0.0):
    """Adaptive quadtree in surface parameters (u, v) -> lab points surf(u, v). scale_u, scale_v convert parameter steps to
    arc length. A cell is split while its arc-length size exceeds 0.5 * min(cfac * gap, h_body) at its centre."""
    U = np.linspace(u_rng[0], u_rng[1], n0[0] + 1)
    V = np.linspace(v_rng[0], v_rng[1], n0[1] + 1)
    du0, dv0 = U[1] - U[0], V[1] - V[0]
    uc, vc = np.meshgrid(0.5 * (U[:-1] + U[1:]), 0.5 * (V[:-1] + V[1:]), indexing="ij")
    cells = np.stack([uc.ravel(), vc.ravel()], -1)
    du, dv = du0, dv0
    pts = []
    for lev in range(maxlev):
        P = surf(cells[:, 0], cells[:, 1])
        g = wall.sdf(P)
        target = np.where(g < gmax, np.maximum(np.minimum(cfac * g, h_body), hmin), h_body)
        size = np.maximum(du * scale_u, dv * scale_v)
        split = size > 0.5 * target
        for p, gg, tt in zip(P[~split], g[~split], target[~split]):
            pts.append((p, float(tt)))
        if not split.any():
            break
        cs = cells[split]
        du, dv = du / 2, dv / 2
        cells = np.concatenate([cs + np.array([sx * du / 2, sy * dv / 2]) for sx in (-1, 1) for sy in (-1, 1)])
    # mid-gap and wall-foot companions where the gap is small
    out = []
    eps = 1e-6
    for p, h in pts:
        out.append((p, h))
    P = np.array([p for p, _ in pts]); H_ = np.array([h for _, h in pts])
    g = wall.sdf(P)
    m = g < gmax
    if m.any():
        Pm = P[m]
        grad = np.stack([(wall.sdf(Pm + eps * e) - wall.sdf(Pm - eps * e)) / (2 * eps) for e in np.eye(3)], -1)
        grad /= np.linalg.norm(grad, axis=1, keepdims=True)
        for fr in (0.5, 1.0):
            for p, gg, hh in zip(Pm - fr * g[m][:, None] * grad, g[m], H_[m]):
                out.append((p, float(hh)))
    return out
