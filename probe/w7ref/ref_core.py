"""REF (w7ref): steady incompressible Navier-Stokes around a held sphere, body-fitted, NGSolve Taylor-Hood.

Units inside: sphere radius a = 1, velocity scale U (case-dependent), density 1, so the code's kinematic viscosity is
nu = 1 / Re_a with Re_a = U a / nu_phys. Forces come out in rho U^2 a^2 and torques in rho U^2 a^3; callers divide by nu to
get mu U a and mu U a^2.

Discretisation: P(k)/P(k-1) Taylor-Hood on a curved (order k) Netgen tetrahedral mesh, Galerkin (no stabilisation),
gradient (Laplacian) viscous form so the natural outlet is the do-nothing condition consistent with developed flow.
Newton with a direct factorisation (MKL PARDISO through NGSolve) and continuation in Re.
Forces by the residual (variational) method: F . e = -R(u)(w_e), w_e = e on the sphere's boundary dofs and 0 elsewhere,
cross-checked against the surface traction integral.

Mesh kinds (half domains y >= 0 unless full=True; mirror plane y = 0 with u_y = 0):
  tube   tube radius R along x in [-Lu, Ld], wall at r = R; sphere centre (0, 0, -(R - 1 - gap)); Poiseuille inflow.
  shear  box x in [-Lu, Ld], y in [0, Ly], z in [0, H]; plane wall z = 0; sphere centre (0, 0, 1 + gap); u = G z e_x.
  eka    as shear, all outer faces Dirichlet (the validation case: the wall and far field move, zero slip at the centre).
  free   cylinder radius Rf along x in [-Lu, Ld]; sphere at the origin; uniform inflow.
"""
import math
import time

import numpy as np
import ngsolve as ng
from netgen.occ import Box, Cylinder, Sphere, Pnt, X, Y, Z, OCCGeometry
from netgen.meshing import MeshingParameters

# Base mesh sizes (scale s = 1); every length is multiplied by s for a level, the gap grading k is divided by s.
BASE = dict(k=2.0, h_sph=0.15, h_wall=0.6, h_far=3.0, h_w=0.3, grading=0.3)


def _lattice(mp, x0, x1, y0, y1, z0, z1, h, inside=None):
    n = 0
    for xx in np.arange(x0, x1 + 1e-9, h):
        for yy in np.arange(y0, y1 + 1e-9, h):
            for zz in np.arange(z0, z1 + 1e-9, h):
                if inside is not None and not inside(xx, yy, zz):
                    continue
                mp.RestrictH(x=float(xx), y=float(yy), z=float(zz), h=h)
                n += 1
    return n


def build_mesh(kind, gap, s=1.0, order=2, full=False, R=2.5925925925925926, Lu=10.0, Ld=30.0, Ly=12.0, H=12.0, Rf=20.0,
               wake=None, hmin=2e-4, hfar=None, base=None):
    """Netgen mesh of the fluid around a unit sphere. Returns (mesh, info). base overrides entries of BASE."""
    B = dict(BASE, **(base or {}))
    k = B["k"] / s
    h_sph, h_wall, h_far, h_w = (B[n] * s for n in ("h_sph", "h_wall", "h_far", "h_w"))
    if hfar is not None:
        h_far = hfar * s
    big = 1000.0
    if kind == "tube":
        zc = -(R - 1.0 - gap)
        body = Cylinder(Pnt(-Lu, 0, 0), X, r=R, h=Lu + Ld)
        body.faces.name = "wall"
        body.faces.Min(X).name = "inlet"
        body.faces.Max(X).name = "outlet"
        centre = (0.0, 0.0, zc)
    elif kind in ("shear", "eka", "res"):
        zc = 1.0 + gap
        body = Box(Pnt(-Lu, -Ly, 0.0), Pnt(Ld, Ly, H))
        body.faces.name = "box"
        body.faces.Min(Z).name = "wall"
        if kind == "shear":
            body.faces.Max(X).name = "outlet"
        centre = (0.0, 0.0, zc)
    elif kind == "free":
        body = Cylinder(Pnt(-Lu, 0, 0), X, r=Rf, h=Lu + Ld)
        body.faces.name = "box"
        body.faces.Max(X).name = "outlet"
        centre = (0.0, 0.0, 0.0)
    else:
        raise ValueError(kind)
    sph = Sphere(Pnt(*centre), 1.0)
    sph.faces.name = "sphere"
    sph.faces.maxh = h_sph
    dom = body - sph
    if not full:
        half = Box(Pnt(-big, 0, -big), Pnt(big, big, big))
        half.faces.Min(Y).name = "sym"
        dom = dom * half
    names = {"wall", "inlet", "outlet", "box", "sphere", "sym"}
    for f in dom.faces:
        if f.name not in names:
            f.name = "sym"
    geo = OCCGeometry(dom)
    mp = MeshingParameters(maxh=h_far, grading=B["grading"])
    n_restrict = 0
    ylo = -R if full else 0.0
    if kind in ("tube", "shear", "eka", "res"):
        cx, cy, cz = centre
        th = 0.0
        while th < math.pi / 2:  # rings around the point nearest the wall, h = local gap / k
            rho = math.sin(th)
            pz = cz - math.cos(th)
            g = (R - abs(pz)) if kind == "tube" else pz
            if g / k >= h_sph:
                break
            phi = 0.0
            phimax = 2 * math.pi if full else math.pi
            while phi < phimax + 1e-12:
                h = max(min(g / k, h_sph), hmin)
                px, py = rho * math.cos(phi), rho * math.sin(phi)
                if kind == "tube":
                    rr = math.hypot(py, pz)
                    wy, wz = py * R / rr, pz * R / rr
                    pts = ((px, py, pz), (px, wy, wz), (px, 0.5 * (py + wy), 0.5 * (pz + wz)))
                else:
                    pts = ((px, py, pz), (px, py, 0.0), (px, py, 0.5 * pz))
                for q in pts:
                    mp.RestrictH(x=q[0], y=q[1] if full else max(q[1], 0.0), z=q[2], h=h)
                n_restrict += 1
                if rho < 1e-12:
                    break
                phi += h / rho
            th += max(min(g / k, h_sph), hmin)
        r = 0.0
        while r <= 3.0:  # a wall disc around the contact point
            nphi = max(1, int(math.ceil((2 if full else 1) * math.pi * r / h_wall)))
            for j in range(nphi + 1):
                ang = (2 if full else 1) * math.pi * j / max(nphi, 1)
                dx, dt = r * math.cos(ang), r * math.sin(ang)
                if kind == "tube":
                    psi = dt / R
                    if abs(psi) > math.pi:
                        continue
                    mp.RestrictH(x=dx, y=R * math.sin(psi), z=-R * math.cos(psi), h=h_wall)
                else:
                    mp.RestrictH(x=dx, y=dt, z=0.0, h=h_wall)
            r += h_wall
    wk = wake or {}
    x_near, x_mid = wk.get("x_near", 8.0), wk.get("x_mid", 20.0)
    if kind == "tube":
        inside = lambda xx, yy, zz: yy * yy + zz * zz <= R * R
        n_restrict += _lattice(mp, -3.0, x_near, ylo, R, -R, R, h_w, inside)
        n_restrict += _lattice(mp, x_near, min(x_mid, Ld), ylo, R, -R, R, 2 * h_w, inside)
    elif kind == "free":
        n_restrict += _lattice(mp, -2.0, 6.0, ylo if full else 0.0, 2.5, -2.5, 2.5, h_w)
        n_restrict += _lattice(mp, 6.0, min(14.0, Ld), 0.0 if not full else -3.5, 3.5, -3.5, 3.5, 2 * h_w)
        if Ld > 14.0:
            n_restrict += _lattice(mp, 14.0, min(Ld, 40.0), 0.0 if not full else -5.0, 5.0, -5.0, 5.0, 4 * h_w)
    else:
        z_top = centre[2]
        y0 = 0.0 if not full else -3.0
        xn, xm = wk.get("x_near", 6.0), wk.get("x_mid", 16.0)
        n_restrict += _lattice(mp, -3.0, xn, y0, 3.0, 0.0, z_top + 2.5, h_w)
        n_restrict += _lattice(mp, xn, min(xm, Ld), -4.5 if full else 0.0, 4.5, 0.0, z_top + 4.0, 2 * h_w)
        if Ld > 16.0 and wk.get("far", True):
            n_restrict += _lattice(mp, 16.0, min(Ld, 40.0), -6.0 if full else 0.0, 6.0, 0.0, z_top + 5.5, 4 * h_w)
    t0 = time.time()
    ngm = geo.GenerateMesh(mp=mp)
    t_mesh = time.time() - t0
    mesh = ng.Mesh(ngm)
    mesh.Curve(order)
    info = dict(kind=kind, gap=gap, s=s, order=order, full=full, centre=centre, R=R, Lu=Lu, Ld=Ld, Ly=Ly, H=H, Rf=Rf,
                k=k, h_sph=h_sph, h_wall=h_wall, h_far=h_far, h_w=h_w, n_restrict=n_restrict, t_mesh=t_mesh, ne=mesh.ne,
                nv=mesh.nv, ngsolve=ng.__version__)
    return mesh, info


class _PyPardiso:
    """MKL PARDISO (real nonsymmetric) through pypardiso on the free block; applied as ``inv * r`` like an NGSolve inverse."""

    def __init__(self, mat, free_mask):
        import scipy.sparse as sp
        import pypardiso
        try:
            data, indices, indptr = mat.CSR()
            A = sp.csr_matrix((np.array(data), np.array(indices), np.array(indptr)), shape=(mat.height, mat.width))
        except Exception:  # older NGSolve: COO
            rows, cols, vals = mat.COO()
            A = sp.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))), shape=(mat.height, mat.width))
        self.idx = np.flatnonzero(free_mask)
        self.A = A[self.idx][:, self.idx].tocsr()
        self.A.sort_indices()
        self.ps = pypardiso.PyPardisoSolver()
        self.ps.set_matrix_type(11)
        self.ps.factorize(self.A)
        self.n = mat.height

    def __mul__(self, r):
        return _Applied(self, r)

    def solve_into(self, r, out):
        rf = r.FV().NumPy()[self.idx]
        x = self.ps.solve(self.A, rf)
        arr = out.FV().NumPy()
        arr[:] = 0.0
        arr[self.idx] = x

    def __del__(self):
        try:
            self.ps.free_memory(everything=True)
        except Exception:
            pass


class _Applied:
    def __init__(self, inv, r):
        self.inv, self.r = inv, r


class Flow:
    """Steady NS on a mesh from build_mesh. bc: dict boundary-name -> CoefficientFunction (velocity) for Dirichlet parts."""

    def __init__(self, mesh, info, order=2, dirichlet_cf=None, gauge=False, log=print):
        self.mesh, self.info, self.log = mesh, info, log
        self.order = order
        names = set(mesh.GetBoundaries())
        self.dir_names = [n for n in ("wall", "inlet", "sphere", "box") if n in names]
        full = info["full"]
        kw = {}
        if not full:
            kw["dirichlety"] = "sym"
        V = ng.VectorH1(mesh, order=order, dirichlet="|".join(self.dir_names), **kw)
        Q = ng.H1(mesh, order=order - 1)
        self.gauge = gauge  # all-Dirichlet velocity: a tiny pressure penalty fixes the constant (a dense multiplier row
        # would ruin the sparse factorisation)
        self.X = ng.FESpace([V, Q])
        self.V, self.Q = V, Q
        self.gfu = ng.GridFunction(self.X)
        self.nu = ng.Parameter(1.0)
        self.conv = ng.Parameter(1.0)  # 0 gives Stokes flow
        tr = self.X.TrialFunction()
        te = self.X.TestFunction()
        u, p = tr[0], tr[1]
        v, q = te[0], te[1]
        form = (self.nu * ng.InnerProduct(ng.grad(u), ng.grad(v)) + self.conv * ng.InnerProduct(ng.grad(u) * u, v)
                - ng.div(u) * q - ng.div(v) * p)
        if gauge:
            form += -1e-10 * p * q
        self.a = ng.BilinearForm(self.X, symmetric=False)
        self.a += form * ng.dx
        # Stokes form (for the Re = 0 start and the Stokes controls)
        self.dirichlet_cf = dirichlet_cf or {}
        self.set_bc()
        self.res = self.gfu.vec.CreateVector()
        self.centre = np.array(info["centre"], dtype=float)
        fd = self.X.FreeDofs()
        self.free_mask = np.array(list(fd), dtype=bool)
        self._weights()

    def set_bc(self):
        """Dirichlet values: each named boundary's dofs from its own interpolant (later names win on shared dofs)."""
        vec = np.zeros(self.V.ndof)
        tmp = ng.GridFunction(self.V)
        for name in self.dir_names:  # boundaries not named in dirichlet_cf are at rest
            cf = self.dirichlet_cf.get(name, ng.CoefficientFunction((0, 0, 0)))
            tmp.vec[:] = 0.0
            tmp.Set(cf, definedon=self.mesh.Boundaries(name))
            mask = np.array(list(self.V.GetDofs(self.mesh.Boundaries(name))), dtype=bool)
            vec[mask] = tmp.vec.FV().NumPy()[mask]
        self.gfu.components[0].vec.FV().NumPy()[:] = vec

    def _weights(self):
        """w_e: e on the sphere's boundary dofs, zero elsewhere (velocity component only)."""
        self.w = {}
        cx, cy, cz = self.centre
        cfs = {"Fx": ng.CoefficientFunction((1, 0, 0)), "Fy": ng.CoefficientFunction((0, 1, 0)),
               "Fz": ng.CoefficientFunction((0, 0, 1)),
               # e_i x (x - c): torque about the centre
               "Tx": ng.CoefficientFunction((0, -(ng.z - cz), ng.y - cy)),
               "Ty": ng.CoefficientFunction((ng.z - cz, 0, -(ng.x - cx))),
               "Tz": ng.CoefficientFunction((-(ng.y - cy), ng.x - cx, 0))}
        mask = np.array(list(self.V.GetDofs(self.mesh.Boundaries("sphere"))), dtype=bool)
        tmp = ng.GridFunction(self.V)
        for key, cf in cfs.items():
            g = ng.GridFunction(self.X)
            tmp.vec[:] = 0.0
            tmp.Set(cf, definedon=self.mesh.Boundaries("sphere"))
            arr = np.zeros(self.V.ndof)
            arr[mask] = tmp.vec.FV().NumPy()[mask]
            g.components[0].vec.FV().NumPy()[:] = arr
            self.w[key] = g

    def residual(self):
        self.a.Apply(self.gfu.vec, self.res)
        return self.res

    def free_norm(self):
        r = self.residual()
        return float(np.linalg.norm(r.FV().NumPy()[self.free_mask]))

    def _factor(self, inverse):
        t0 = time.time()
        self.a.AssembleLinearization(self.gfu.vec)
        t1 = time.time()
        if inverse == "pypardiso":
            self._inv = _PyPardiso(self.a.mat, self.free_mask)
        else:
            self._inv = self.a.mat.Inverse(self.X.FreeDofs(), inverse=inverse)
        self.n_factor += 1
        return t1 - t0, time.time() - t1

    def _apply(self, r, out):
        if isinstance(self._inv, _PyPardiso):
            self._inv.solve_into(r, out)
        else:
            out.data = self._inv * r

    def newton(self, nu, conv=1.0, rtol=1e-7, max_it=25, inverse="pypardiso", reuse=False, gmres_steps=40):
        """Newton at viscosity nu from the current state; stops when the update is under rtol of the state. With reuse,
        the first iteration factorises the Jacobian and later ones solve with GMRES preconditioned by that factor
        (refactorising when GMRES does not reach 1e-10 in gmres_steps)."""
        self.nu.Set(nu)
        self.conv.Set(conv)
        if not hasattr(self, "n_factor"):
            self.n_factor = 0
        hist, steps = [], []
        du = self.gfu.vec.CreateVector()
        old = self.gfu.vec.CreateVector()
        t_ass = t_fac = t_sol = 0.0
        conv_ok = False
        fd = self.X.FreeDofs()
        for it in range(max_it):
            r = self.residual()
            rn = self.free_norm()
            hist.append(rn)
            t0 = time.time()
            n_g = 0
            if it == 0 or not reuse:
                ta, tf = self._factor(inverse)
                t_ass += ta
                t_fac += tf
                t0 = time.time()
                self._apply(r, du)
            else:
                ta0 = time.time()
                self.a.AssembleLinearization(self.gfu.vec)
                t_ass += time.time() - ta0
                t0 = time.time()
                cnt = [0]
                du[:] = 0.0
                ng.solvers.GMRes(self.a.mat, r, pre=self._inv, x=du, maxsteps=gmres_steps, tol=1e-11,
                                 printrates=False, callback=lambda *a_: cnt.__setitem__(0, cnt[0] + 1))
                n_g = cnt[0]
                chk = r.CreateVector()
                chk.data = self.a.mat * du
                chk.data -= r
                lin = float(np.linalg.norm(chk.FV().NumPy()[self.free_mask])) / max(rn, 1e-300)
                if lin > 1e-8:
                    ta, tf = self._factor(inverse)
                    t_ass += ta - (time.time() - ta0) * 0
                    t_fac += tf
                    self._apply(r, du)
                    n_g = -1
            t_sol += time.time() - t0
            dn = float(np.linalg.norm(du.FV().NumPy()[self.free_mask]))
            un = float(np.linalg.norm(self.gfu.vec.FV().NumPy())) + 1e-300
            lam = 1.0
            old.data = self.gfu.vec
            self.gfu.vec.data = old - du
            while self.free_norm() > rn and lam > 1.0 / 16 and dn / un > 1e-6:
                lam *= 0.5
                self.gfu.vec.data = old - lam * du
            steps.append(dict(du=dn / un, lam=lam, gmres=n_g))
            self.log(f"    newton it {it} |R| {rn:.3e} |du|/|u| {dn/un:.3e} lam {lam} gmres {n_g}")
            # stop once the update is small inside the quadratic phase: after an update of e (relative) the error left is
            # about e^2 there, so 1e-4 after a step under 5e-2 leaves about 1e-8
            if dn / un < rtol or (dn / un < 1e-4 and len(steps) >= 2 and steps[-2]["du"] < 5e-2 and lam == 1.0):
                conv_ok = True
                break
        hist.append(self.free_norm())
        return dict(hist=hist, steps=steps, iters=len(steps), t_assemble=t_ass, t_factor=t_fac, t_solve=t_sol,
                    converged=conv_ok, n_factor=self.n_factor)

    def forces(self):
        """Forces (rho U^2 a^2) and torques (rho U^2 a^3) on the body in the mesh: residual method and traction integral."""
        r = self.residual()
        out = {}
        u = self.gfu.components[0]
        nu = self.nu.Get()
        for key, g in self.w.items():
            out[key] = -float(ng.InnerProduct(r, g.vec))
            # the symmetric-stress residual adds nu (grad u)^T : grad w; it vanishes on a held no-slip body as h -> 0
            # (there (grad u)^T n = (div u) n), not on a rotating one, where the Laplacian form misses a third of the torque
            gw = ng.grad(g.components[0])
            extra = ng.Integrate(nu * ng.InnerProduct(ng.grad(u).trans, gw), self.mesh, order=2 * self.order)
            out[key + "_sym"] = out[key] - float(extra)
        # traction integral, gradient form and symmetric form
        p = self.gfu.components[1]
        n = ng.specialcf.normal(3)  # outward from the fluid: into the sphere
        gu = ng.BoundaryFromVolumeCF(ng.grad(u))  # the volume gradient, not the surface gradient
        tg = p * n - nu * (gu * n)  # traction on the sphere (gradient form), n_s = -n
        ts = p * n - nu * ((gu + gu.trans) * n)
        bnd = self.mesh.Boundaries("sphere")
        cx, cy, cz = self.centre
        rvec = ng.CoefficientFunction((ng.x - cx, ng.y - cy, ng.z - cz))
        for tag, t in (("g", tg), ("s", ts)):
            F = ng.Integrate(t, self.mesh, ng.BND, definedon=bnd, order=2 * self.order + 2)
            T = ng.Integrate(ng.Cross(rvec, t), self.mesh, ng.BND, definedon=bnd, order=2 * self.order + 2)
            out[f"Fx_{tag}"], out[f"Fy_{tag}"], out[f"Fz_{tag}"] = (float(F[i]) for i in range(3))
            out[f"Tx_{tag}"], out[f"Ty_{tag}"], out[f"Tz_{tag}"] = (float(T[i]) for i in range(3))
        return out

    def ndof(self):
        return int(self.free_mask.sum()), int(self.X.ndof)
