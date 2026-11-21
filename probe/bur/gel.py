"""An Eulerian gel on cells with a rigid body moving through them (spike w7bur; numpy/scipy, float64).

A periodic MAC grid (nx by ny, spacing h): u on x-faces, v on y-faces, the antiplane velocity w and the pressure at centres. The gel's
history is a reference map X = x + xi at centres (xi1, xi2 in the plane, zeta out of it), convected by a semi-Lagrangian step. One
step is the incremental potential's linear solve: the elastic increment enters as a viscosity kappa G dt, the stress carried from the
map as a load, incompressibility by the pressure. Rigid cells (the body, the clamp) are Dirichlet velocities; the body's load is the
reaction on its rows. A band of cells on the body carries a wall law in the slip speed; a failed cell (damage 1, stored on material
coordinates or convected, by choice) carries a paste law.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy import ndimage


def _shift(n):
    """(P f)[i] = f[i-1], periodic."""
    return sp.csr_matrix((np.ones(n), (np.arange(n), (np.arange(n) - 1) % n)), shape=(n, n))


class Grid:
    def __init__(s, nx, ny, h):
        s.nx, s.ny, s.h, s.n = nx, ny, h, nx * ny
        Ix, Iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
        Px, Py = _shift(nx), _shift(ny)
        s.Dmx = sp.kron((Ix - Px) / h, Iy, format="csr")      # f[i] - f[i-1]
        s.Dpx = sp.kron((Px.T - Ix) / h, Iy, format="csr")    # f[i+1] - f[i]
        s.Dmy = sp.kron(Ix, (Iy - Py) / h, format="csr")
        s.Dpy = sp.kron(Ix, (Py.T - Iy) / h, format="csr")
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        s.xc, s.yc = (i + 0.5) * h, (j + 0.5) * h              # centres
        s.xu, s.yu = i * h, (j + 0.5) * h                      # x-faces
        s.xv, s.yv = (i + 0.5) * h, j * h                      # y-faces
        s.xn, s.yn = i * h, j * h                              # corners
        s.Lx, s.Ly = nx * h, ny * h

    # averages between staggered places (periodic), on (nx, ny) arrays
    @staticmethod
    def n2c(a):   # corners (i,j) -> centre (i+1/2, j+1/2)
        return 0.25 * (a + np.roll(a, -1, 0) + np.roll(a, -1, 1) + np.roll(np.roll(a, -1, 0), -1, 1))

    @staticmethod
    def n2u(a):   # corners (i,j),(i,j+1) -> x-face (i, j+1/2)
        return 0.5 * (a + np.roll(a, -1, 1))

    @staticmethod
    def n2v(a):   # corners (i,j),(i+1,j) -> y-face (i+1/2, j)
        return 0.5 * (a + np.roll(a, -1, 0))

    @staticmethod
    def c2n(a):   # centres -> corner (i,j)
        return 0.25 * (a + np.roll(a, 1, 0) + np.roll(a, 1, 1) + np.roll(np.roll(a, 1, 0), 1, 1))

    @staticmethod
    def c2u(a):
        return 0.5 * (a + np.roll(a, 1, 0))

    @staticmethod
    def c2v(a):
        return 0.5 * (a + np.roll(a, 1, 1))

    def sample(s, a, x, y, order):
        """A centre field at points (x, y), periodic."""
        return ndimage.map_coordinates(a, [x / s.h - 0.5, y / s.h - 0.5], order=order, mode="grid-wrap")


class Gel:
    """geom(t) -> dict(phi=f(x,y) signed distance to the body (positive in the gel), vel=f(x,y)->(ub, vb, wb), clamp=f(x,y)->bool)."""

    def __init__(s, grid, G, geom, dt, band=0.0, wall_law=None, tau_y=None, paste=None, damage="material", order=3,
                 kappa_max=8.0, n_picard=4, antiplane=False, plant_wall=1.0, plant_G=1.0, d_order=1):
        s.g, s.G, s.geom, s.dt = grid, G, geom, dt
        s.band, s.wall_law, s.tau_y, s.paste, s.damage, s.order = band, wall_law, tau_y, paste, damage, order
        s.kappa_max, s.n_picard, s.antiplane, s.plant_wall = kappa_max, n_picard, antiplane, plant_wall
        s.plant_G, s.d_order = plant_G, d_order
        n = grid.n
        s.xi = np.zeros((3, grid.nx, grid.ny))          # xi1, xi2, zeta at centres
        s.d_ref = np.zeros((grid.nx, grid.ny))          # damage on material coordinates (a reference grid, same cells)
        s.d_adv = np.zeros((grid.nx, grid.ny))          # damage as a convected scalar (the comparison)
        s.t = 0.0
        s.eta_c_prev = None
        s.U = np.zeros(2 * n); s.W = np.zeros(n)
        s.log = []

    # ---- kinematics of the map -------------------------------------------------------------------------------------------
    def left_cauchy_green(s):
        """B = (M^T M)^-1 at centres, corners, x-faces and y-faces; M = dX/dx = I + grad xi (third row (zeta_x, zeta_y, 1)).
        The map is staggered like the velocity (xi1 on x-faces, xi2 on y-faces, zeta at centres), so each strain is taken where
        the velocity's is, with the same differences: the carried stress and the step's viscosity are one operator."""
        g = s.g
        sh = (g.nx, g.ny)
        a = (g.Dpx @ s.xi[0].ravel()).reshape(sh); d = (g.Dpy @ s.xi[1].ravel()).reshape(sh)          # centres
        b = (g.Dmy @ s.xi[0].ravel()).reshape(sh); c = (g.Dmx @ s.xi[1].ravel()).reshape(sh)          # corners
        e = (g.Dmx @ s.xi[2].ravel()).reshape(sh); f = (g.Dmy @ s.xi[2].ravel()).reshape(sh)          # x-faces, y-faces
        r = np.roll
        def inv(a, b, c, d, e, f):
            M = np.zeros(sh + (3, 3))
            M[..., 0, 0] = 1 + a; M[..., 0, 1] = b; M[..., 1, 0] = c; M[..., 1, 1] = 1 + d
            M[..., 2, 0] = e; M[..., 2, 1] = f; M[..., 2, 2] = 1.0
            C = np.einsum("...ki,...kj->...ij", np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0), M)
            det = np.linalg.det(C)
            bad = ~np.isfinite(det) | (det < 1e-10) | (det > 1e10)     # a map with no meaning there (deep in the body, a wound paste)
            C[bad] = np.eye(3)
            return np.clip(np.linalg.inv(C), -1e6, 1e6)
        Bc = inv(a, g.n2c(b), g.n2c(c), d, 0.5 * (e + r(e, -1, 0)), 0.5 * (f + r(f, -1, 1)))
        Bn = inv(g.c2n(a), b, c, g.c2n(d), 0.5 * (e + r(e, 1, 1)), 0.5 * (f + r(f, 1, 0)))
        if not s.antiplane:
            return Bc, Bn, None, None
        f_u = 0.25 * (f + r(f, 1, 0) + r(f, -1, 1) + r(r(f, 1, 0), -1, 1))
        e_v = 0.25 * (e + r(e, 1, 1) + r(e, -1, 0) + r(r(e, -1, 0), 1, 1))
        Bu = inv(g.c2u(a), g.n2u(b), g.n2u(c), g.c2u(d), e, f_u)
        Bv = inv(g.c2v(a), g.n2v(b), g.n2v(c), g.c2v(d), e_v, f)
        return Bc, Bn, Bu, Bv

    def map_at_centres(s):
        g = s.g
        return (g.xc + 0.5 * (s.xi[0] + np.roll(s.xi[0], -1, 0)), g.yc + 0.5 * (s.xi[1] + np.roll(s.xi[1], -1, 1)))

    def damage_at_centres(s):
        g = s.g
        if s.damage == "material":
            X, Y = s.map_at_centres()
            i = np.floor(X / g.h).astype(int) % g.nx; j = np.floor(Y / g.h).astype(int) % g.ny
            return s.d_ref[i, j]
        return np.clip(s.d_adv, 0.0, 1.0)

    # ---- one step --------------------------------------------------------------------------------------------------------
    def step(s, commit=True):
        g, dt, G, n = s.g, s.dt, s.G, s.g.n
        geo = s.geom(s.t + dt)
        phi = lambda x, y: geo["phi"](x, y)
        ph_c, ph_n, ph_u, ph_v = phi(g.xc, g.yc), phi(g.xn, g.yn), phi(g.xu, g.yu), phi(g.xv, g.yv)
        cl = geo["clamp"]
        # fixed velocities
        ub, vb, wb = geo["vel"]
        fix_u_body, fix_v_body, fix_w_body = ph_u < 0, ph_v < 0, ph_c < 0
        fix_u = fix_u_body | cl(g.xu, g.yu); fix_v = fix_v_body | cl(g.xv, g.yv); fix_w = fix_w_body | cl(g.xc, g.yc)
        Uc = np.zeros(2 * n)
        Uc[:n][fix_u_body.ravel()] = ub(g.xu, g.yu)[fix_u_body]
        Uc[n:][fix_v_body.ravel()] = vb(g.xv, g.yv)[fix_v_body]
        fixed = np.concatenate([fix_u.ravel(), fix_v.ravel()])
        # material at each stress place: gel (elastic), band (wall law), rigid
        def kind(ph, x, y):
            rigid = (ph < 0) | cl(x, y)
            bandm = (~rigid) & (ph < s.band)
            return (~rigid) & (~bandm), bandm
        gel_n, band_n = kind(ph_n, g.xn, g.yn)
        gel_c, band_c = kind(ph_c, g.xc, g.yc)
        # carried elastic stress, G (1 - d) (B - I), where the place is gel
        # where the map's stress acts: everywhere when bonded (a rigid place's map is rigid, its stress zero by itself, and the
        # places that straddle the body's surface must act, as the viscosity's do); outside the band when the wall slips
        el = (lambda ph: ph >= s.band) if s.band > 0 else (lambda ph: np.ones_like(ph, dtype=bool))
        el_c, el_n, el_u, el_v = el(ph_c), el(ph_n), el(ph_u), el(ph_v)
        soft = (gel_c | band_c).astype(float)                     # a place between rigid cells only carries nothing
        el_n = el_n & (g.c2n(soft) > 0); el_u = el_u & (g.c2u(soft) > 0); el_v = el_v & (g.c2v(soft) > 0)
        Bc, Bn, Bu, Bv = s.left_cauchy_green()
        d_c = s.damage_at_centres() if s.tau_y is not None else np.zeros((g.nx, g.ny))
        hit = lambda a: (a > 0.2).astype(float)                    # a place fails with any failed neighbour
        Gc = s.plant_G * G * el_c * (1.0 - d_c); Gn = s.plant_G * G * el_n * (1.0 - hit(g.c2n(d_c)))
        sxx_c, syy_c, Sxy = Gc * (Bc[..., 0, 0] - 1), Gc * (Bc[..., 1, 1] - 1), Gn * Bn[..., 0, 1]
        kap_c = np.clip(np.maximum(np.maximum(Bc[..., 0, 0], Bc[..., 1, 1]), Bc[..., 2, 2]), 1.0, s.kappa_max)
        eta_el_c = kap_c * G * dt                       # the elastic increment as a viscosity, on the stiff side
        f_u = -(g.Dpx.T @ sxx_c.ravel() + g.Dmy.T @ Sxy.ravel())
        f_v = -(g.Dpy.T @ syy_c.ravel() + g.Dmx.T @ Sxy.ravel())
        f = np.concatenate([f_u, f_v])
        if s.antiplane:
            Sxz = G * el_u * (1.0 - hit(g.c2u(d_c))) * Bu[..., 0, 2]; Syz = G * el_v * (1.0 - hit(g.c2v(d_c))) * Bv[..., 1, 2]
        Div = sp.hstack([g.Dpx, g.Dpy], format="csr")
        # the stress the intact gel carries, for the failure law (centres)
        D = Bc - np.eye(3) * (np.trace(Bc, axis1=-2, axis2=-1) / 3.0)[..., None, None]
        teq_c = G * np.sqrt(0.5 * np.einsum("...ij,...ij->...", D, D))
        failed_c = (d_c > 0.5) & gel_c
        eta_c_law = None if s.eta_c_prev is None else s.eta_c_prev.copy()
        nonlin = (s.wall_law is not None and s.band > 0) or (s.paste is not None and s.tau_y is not None)
        U = s.U; W = s.W
        for it in range(s.n_picard if nonlin else 1):
            # viscosity at centres: elastic increment in intact gel; laws in band and failed cells
            eta_c = eta_el_c.copy()
            if nonlin:
                if eta_c_law is None:
                    eta_c_law = eta_c.copy()
                eta_c = np.where(band_c | failed_c, eta_c_law, eta_c)
            mc = (gel_c | band_c).astype(float)                    # rigid places carry no law: averages leave them out
            def mavg(op):
                wgt = op(mc)
                return np.where(wgt > 0, op(eta_c * mc) / np.maximum(wgt, 1e-300), 0.0)   # rigid against rigid: no coupling
            eta_n, eta_u, eta_v = mavg(g.c2n), mavg(g.c2u), mavg(g.c2v)
            eta_c = np.where(mc > 0, eta_c, G * dt)
            Ec, En = sp.diags(eta_c.ravel()), sp.diags(eta_n.ravel())
            Auu = 2 * g.Dpx.T @ Ec @ g.Dpx + g.Dmy.T @ En @ g.Dmy
            Avv = 2 * g.Dpy.T @ Ec @ g.Dpy + g.Dmx.T @ En @ g.Dmx
            Auv = g.Dmy.T @ En @ g.Dmx
            A = sp.bmat([[Auu, Auv], [Auv.T, Avv]], format="csr")
            free = ~fixed
            eps = 1e-9 / (G * dt)
            K = sp.bmat([[A[free][:, free], -Div[:, free].T], [-Div[:, free], -eps * sp.identity(n)]], format="csc")
            rhs = np.concatenate([(f - A @ Uc)[free], Div @ Uc])
            sol = spl.splu(K).solve(rhs)
            U = Uc.copy(); U[free] = sol[:free.sum()]; p = sol[free.sum():]
            react = A @ U - Div.T @ p - f                        # the constraint's force on the medium, per cell area
            if s.antiplane:
                fw = -(g.Dmx.T @ Sxz.ravel() + g.Dmy.T @ Syz.ravel())
                Aw = (g.Dmx.T @ sp.diags(eta_u.ravel()) @ g.Dmx + g.Dmy.T @ sp.diags(eta_v.ravel()) @ g.Dmy).tocsr()
                Wc = np.zeros(n); Wc[fix_w_body.ravel()] = wb(g.xc, g.yc)[fix_w_body]
                fr = ~fix_w.ravel()
                W = Wc.copy(); W[fr] = spl.splu(Aw[fr][:, fr].tocsc()).solve((fw - Aw @ Wc)[fr])
                react_w = Aw @ W - fw
            if not nonlin:
                break
            # the laws: equivalent shear rate at centres, then the secant viscosity
            u2, v2 = U[:n], U[n:]
            exx = (g.Dpx @ u2).reshape(g.nx, g.ny); eyy = (g.Dpy @ v2).reshape(g.nx, g.ny)
            gxy = (g.Dmy @ u2 + g.Dmx @ v2).reshape(g.nx, g.ny)
            gam2 = 2 * exx ** 2 + 2 * eyy ** 2 + g.n2c(gxy ** 2)
            if s.antiplane:
                gxz = (g.Dmx @ W).reshape(g.nx, g.ny); gyz = (g.Dmy @ W).reshape(g.nx, g.ny)
                gam2 = gam2 + 0.5 * (gxz ** 2 + np.roll(gxz, -1, 0) ** 2) + 0.5 * (gyz ** 2 + np.roll(gyz, -1, 1) ** 2)
            gam = np.sqrt(gam2) + 1e-300
            new = eta_c_law.copy()
            if s.wall_law is not None and s.band > 0:
                V = gam * s.band
                new = np.where(band_c, s.plant_wall * s.wall_law(V) * s.band / V, new)
            if s.paste is not None and s.tau_y is not None:
                new = np.where(failed_c, s.paste(gam) / gam, new)
            eta_c_law = np.exp(0.5 * np.log(new) + 0.5 * np.log(eta_c_law)) if it > 0 else new
        if commit:
            s.eta_c_prev = eta_c_law
            s.U, s.W = U, W
        # loads on the body (per unit depth): minus the reaction on its rows
        ru, rv = react[:n].reshape(g.nx, g.ny), react[n:].reshape(g.nx, g.ny)
        h2 = g.h ** 2
        cx, cy = geo.get("centre", (0.0, 0.0))
        out = dict(t=s.t + dt,
                   Fx=-h2 * ru[fix_u_body].sum(), Fy=-h2 * rv[fix_v_body].sum(),
                   Tz=-h2 * (((g.xv - cx) * rv)[fix_v_body].sum() - ((g.yu - cy) * ru)[fix_u_body].sum()))
        if s.antiplane:
            out["Fz"] = -h2 * react_w.reshape(g.nx, g.ny)[fix_w_body].sum()
        # failure: mark the material point (both stores), then convect
        if not commit:                      # a trial solve: the load at this body velocity, nothing moved or marked
            return out
        if s.tau_y is not None:
            fail = (teq_c > s.tau_y) & gel_c
            X, Y = s.map_at_centres()
            i = np.floor(X / g.h).astype(int) % g.nx; j = np.floor(Y / g.h).astype(int) % g.ny
            np.maximum.at(s.d_ref, (i[fail], j[fail]), 1.0)
            s.d_adv = np.maximum(s.d_adv, fail.astype(float))
            out["failed_cells"] = int(((s.damage_at_centres() > 0.5) & gel_c).sum())
        intact = gel_c & el_c & (d_c < 0.5)
        out["teq_max"] = float(teq_c[intact].max()) if intact.any() else 0.0
        s.last = dict(U=U, W=W, gel_c=gel_c, band_c=band_c, d_c=d_c, teq_c=teq_c, ph_c=ph_c)
        s.convect(U, W, (ph_u, ph_v, ph_c))
        s.t += dt
        out["kappa_max"] = float(kap_c[gel_c].max()) if gel_c.any() else 1.0
        s.log.append(out)
        return out

    # ---- convection of the map (and of the comparison damage scalar) ----------------------------------------------------
    def convect(s, U, W, phis):
        """Semi-Lagrangian, midpoint departure points; each component at its own place."""
        g, dt, n, h = s.g, s.dt, s.g.n, s.g.h
        u = U[:n].reshape(g.nx, g.ny); v = U[n:].reshape(g.nx, g.ny); w = W.reshape(g.nx, g.ny)
        S = lambda a, x, y, off, order: ndimage.map_coordinates(a, [x / h - off[0], y / h - off[1]], order=order, mode="grid-wrap")
        OU, OV, OC = (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)
        def depart(x, y):
            um, vm = S(u, x, y, OU, 1), S(v, x, y, OV, 1)
            xm, ym = x - 0.5 * dt * um, y - 0.5 * dt * vm
            um, vm = S(u, xm, ym, OU, 1), S(v, xm, ym, OV, 1)
            return x - dt * um, y - dt * vm, um, vm, xm, ym
        new = np.empty_like(s.xi)
        xd, yd, um, vm, xm, ym = depart(g.xu, g.yu)
        new[0] = S(s.xi[0], xd, yd, OU, s.order) - dt * um
        xd, yd, um, vm, xm, ym = depart(g.xv, g.yv)
        new[1] = S(s.xi[1], xd, yd, OV, s.order) - dt * vm
        xd, yd, um, vm, xm, ym = depart(g.xc, g.yc)
        new[2] = S(s.xi[2], xd, yd, OC, s.order) - (dt * S(w, xm, ym, OC, 1) if s.antiplane else 0.0)
        s.xi = new
        s.d_adv = S(s.d_adv, xd, yd, OC, s.d_order)
        if s.band > 0:
            s.fill_from_gel(phis, S, (OU, OV, OC))

    def fill_from_gel(s, phis, S, offs):
        """With a slipping wall the map is discontinuous at the body: fill the band and the body's rim from the gel side, linearly
        along the normal, so no stencil reads across the slip."""
        g, h = s.g, s.g.h
        places = ((g.xu, g.yu), (g.xv, g.yv), (g.xc, g.yc))
        for k in range(3):
            ph = phis[k]
            nxg = (np.roll(ph, -1, 0) - np.roll(ph, 1, 0)) / (2 * h); nyg = (np.roll(ph, -1, 1) - np.roll(ph, 1, 1)) / (2 * h)
            nrm = np.sqrt(nxg ** 2 + nyg ** 2) + 1e-300
            nxg, nyg = nxg / nrm, nyg / nrm
            m = (ph < s.band + 0.5 * h) & (ph > -6 * h)
            d1 = (s.band + 1.5 * h - ph)[m]; step = 1.5 * h
            x1, y1 = places[k][0][m] + nxg[m] * d1, places[k][1][m] + nyg[m] * d1
            x2, y2 = x1 + nxg[m] * step, y1 + nyg[m] * step
            a1, a2 = S(s.xi[k], x1, y1, offs[k], 1), S(s.xi[k], x2, y2, offs[k], 1)
            s.xi[k][m] = a1 + (a1 - a2) * d1 / step
