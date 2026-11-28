"""r39 steady Navier-Stokes in G1, body frame (Type B; with Vz != 0 the body also moves along z on a frozen geometry, Type L): Re_a (u . grad u) - Lap u + grad p = 0, div u = 0; u = Omega x (x - c)
on the body, -V e_x on the wall, u_P - V e_x at the inlet (u_P = flow * 2 (1 - r^2/R^2)), do-nothing outlet (Laplacian form,
consistent with developed Poiseuille). Units a = mu = U_m = 1; forces in mu U_m a. Half domain y >= 0 (the class even in y:
translations x, z and rotation y); forces doubled.
Force on the body along mode j: F_j = -int [2 eps(u):eps(w_j) + Re_a (grad u u) . w_j - p div w_j] with w_j the rigid mode's
lift on the body dofs (the symmetric stress: the Laplacian form's residual misses (grad u)^T, which a rotating body needs)."""
import sys, os, json, time
import numpy as np
from ngsolve import (VectorH1, H1, BilinearForm, GridFunction, CoefficientFunction, InnerProduct, Grad, div, dx, x, y, z,
                     Integrate, TaskManager, solvers, Sym)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lin


def solve_ns(mesh, order, R, c, points, log=print, maxit=25, tol=1e-10):
    """points: list of dicts {Re, V, Om (y-rotation rate), flow (0 or 1)}; solved in the given order, each Newton started from
    the previous solution. Returns a list of result dicts."""
    V_ = VectorH1(mesh, order=order, dirichlet="wall|inlet|body", dirichlety="symy")
    Q = H1(mesh, order=order - 1)
    X = V_ * Q
    (u, p), (v, q) = X.TnT()
    gfu = GridFunction(X)
    Re_par = __import__("ngsolve").Parameter(0.0)
    a = BilinearForm(X)
    a += (InnerProduct(Grad(u), Grad(v)) + Re_par * InnerProduct(Grad(u) * u, v) - div(v) * p - div(u) * q) * dx
    ws = {}
    for j in (0, 2, 4):
        w = GridFunction(V_)
        w.Set(lin.mode_cf(j, c), definedon=mesh.Boundaries("body"))
        ws[j] = w
    out = []
    for pt in points:
        t0 = time.time()
        Re, Vb, Om, flow, Vz = pt["Re"], pt["V"], pt.get("Om", 0.0), pt.get("flow", 1.0), pt.get("Vz", 0.0)
        uP = flow * 2.0 * (1 - (y * y + z * z) / R ** 2)
        bc = mesh.BoundaryCF({"wall": CoefficientFunction((-Vb, 0, 0)), "inlet": CoefficientFunction((uP - Vb, 0, 0)),
                              "body": CoefficientFunction((Om * (z - c[2]), 0, -Om * (x - c[0]) + Vz))}, default=CoefficientFunction((0, 0, 0)))
        # keep the interior, reset the Dirichlet values
        gu = gfu.components[0]
        tmp = GridFunction(V_)
        tmp.Set(bc, definedon=mesh.Boundaries("wall|inlet|body"))
        fd = V_.FreeDofs()
        for i in range(V_.ndof):
            if not fd[i]:
                gu.vec[i] = tmp.vec[i]
        # continuation in Re from the last solved Re
        Re0 = float(Re_par.Get())
        steps = [Re] if abs(Re - Re0) <= 30 else list(np.linspace(Re0, Re, int(np.ceil(abs(Re - Re0) / 30)) + 1)[1:])
        conv = True
        its = []
        for Rs in steps:
            Re_par.Set(float(Rs))
            with TaskManager():
                ret = solvers.Newton(a, gfu, freedofs=X.FreeDofs(), maxit=maxit, maxerr=tol, inverse="pardiso", printing=False)
            its.append(int(ret[1]))
            conv = conv and ret[0] == 0
        uh, ph = gfu.components
        F = {}
        for j, w in ws.items():
            val = Integrate(2 * InnerProduct(Sym(Grad(uh)), Sym(Grad(w))) + Re * InnerProduct(Grad(uh) * uh, w) - ph * div(w),
                            mesh, order=3 * order)
            F[j] = -2.0 * val
        rec = dict(pt, Fx=F[0], Fz=F[2], Ty=F[4], newton_its=its, converged=bool(conv), ndof=X.ndof, t=time.time() - t0,
                   rss_gib=lin.rss_gib())
        log(json.dumps(rec))
        out.append(rec)
    return out
