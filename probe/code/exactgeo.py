"""Exact-geometry correction for surfaces Netgen-OCC only approximates (the helix tube; the G4 carina fillet).
The curved mesh's boundary is moved onto the analytic surface of cases/geom.py by a displacement d = proj(p) - p, set on the named
boundary at order k and extended harmonically inside (mesh.SetDeformation). The projections are unrolled Newton iterations
written as NGSolve coefficient functions, so they are evaluated at the curved mesh's own points."""
import numpy as np
from ngsolve import (CoefficientFunction, IfPos, sqrt, sin, cos, atan2, x, y, z, VectorH1, GridFunction, BilinearForm,
                     InnerProduct, Grad, dx, Integrate, BND)


def _clamp(t, lo, hi):
    t = IfPos(t - lo, t, lo)
    return IfPos(t - hi, hi, t)


def helix_correction(H, c, Rm, newton=8):
    Rm = np.asarray(Rm, float)
    P = [x - float(c[0]), y - float(c[1]), z - float(c[2])]
    pb = [sum(float(Rm[j, i]) * P[j] for j in range(3)) for i in range(3)]
    rh, cp, t1 = H.rh, H.pitch / (2 * np.pi), H.t1
    th = atan2(pb[1], pb[0])
    Zt = lambda t: cp * t - 1.5
    best = th - 2 * np.pi
    for k in range(0, 4):
        tk = th + 2 * np.pi * k
        best = IfPos((pb[2] - Zt(best)) ** 2 - (pb[2] - Zt(tk)) ** 2, tk, best)
    t = _clamp(best, 0.0, t1)
    for _ in range(newton):
        ct, st = cos(t), sin(t)
        rx, ry, rz = pb[0] - rh * ct, pb[1] - rh * st, pb[2] - Zt(t)
        f = -(rx * (-rh * st) + ry * (rh * ct) + rz * cp)
        fp = rh * rh + cp * cp + rx * rh * ct + ry * rh * st
        t = _clamp(t - f / fp, 0.0, t1)
    ct, st = cos(t), sin(t)
    rv = [pb[0] - rh * ct, pb[1] - rh * st, pb[2] - Zt(t)]
    d = sqrt(rv[0] ** 2 + rv[1] ** 2 + rv[2] ** 2)
    cb = [-(d - H.rho) * rv[i] / d for i in range(3)]
    return CoefficientFunction(tuple(sum(float(Rm[i, j]) * cb[j] for j in range(3)) for i in range(3)))


def carina_correction(g, newton=10):
    A, B, rc = g.Ae, g.Be, g.rc
    phi = _clamp(atan2(A * y, B * x), -np.pi / 2, np.pi / 2)
    for _ in range(newton):
        ex, ey = A * cos(phi), B * sin(phi)
        f = (x - ex) * A * sin(phi) - (y - ey) * B * cos(phi)
        fp = A * A * sin(phi) ** 2 + B * B * cos(phi) ** 2 + (x - ex) * ex + (y - ey) * ey
        phi = _clamp(phi - f / fp, -np.pi / 2, np.pi / 2)
    ex, ey = A * cos(phi), B * sin(phi)
    rv = [x - ex, y - ey, z]
    d = sqrt(rv[0] ** 2 + rv[1] ** 2 + rv[2] ** 2)
    return CoefficientFunction(tuple((rc - d) * rv[i] / d for i in range(3)))


def apply(mesh, order, corrections, inverse="sparsecholesky"):
    """corrections: {boundary name: CF}. Returns (max |d| on those boundaries before, the deformation GridFunction)."""
    names = "|".join(corrections)
    allb = "|".join(sorted(set(mesh.GetBoundaries())))
    V = VectorH1(mesh, order=order, dirichlet=allb)
    d = GridFunction(V)
    for nm, cf in corrections.items():
        d.Set(cf, definedon=mesh.Boundaries(nm))
    u, v = V.TnT()
    a = BilinearForm(InnerProduct(Grad(u), Grad(v)) * dx).Assemble()
    r = d.vec.CreateVector()
    r.data = -a.mat * d.vec
    d.vec.data += a.mat.Inverse(V.FreeDofs(), inverse=inverse) * r
    mag = Integrate(InnerProduct(d, d), mesh, BND, definedon=mesh.Boundaries(names))
    mesh.SetDeformation(d)
    return float(mag), d
