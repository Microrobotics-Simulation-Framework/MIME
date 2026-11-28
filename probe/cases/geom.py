"""r39 cases: the analytic shared definition of every geometry, body and pose (numpy only).

Units: lengths in a (sphere radius; outer radius of H and F). Flow along +x; the pressed wall is on -z; y = 0 is the mirror
plane of every wall. Every wall function is a signed distance, positive in the fluid (exact distance for G0, G0b, G1, G2, G3,
and G4 inside the lumen). Body functions are positive outside the body (an exact distance for SPH and H; for F an implicit
function whose zero set is the exact surface). See cases/README.md.
"""
import numpy as np

# ---------------------------------------------------------------- tube radii (declared)
R_SPH = 1.0 / 0.386        # sphere cases: a/R_T = 0.386 (W7REF's tube, flagship basilar)
R_HF = 2.0                 # helix and fin cases: R_T = 2a (w35m, REFUTE_W35)
R_SPH5 = 2.0               # sphere at a/R_T = 0.5 (cross-check against Caldag et al. 2022's table)


def _norm(v):
    return np.sqrt(np.sum(v * v, axis=-1))


# ---------------------------------------------------------------- walls
class Plane:                # G0
    name = "G0"
    def sdf(self, p):
        return p[..., 2]


class HeldSphere:           # G0b: fixed sphere of radius b centred (0,0,-b); the wall bulges towards the body
    name = "G0b"
    def __init__(self, b=4.0):
        self.b = b
    def sdf(self, p):
        return _norm(p - np.array([0.0, 0.0, -self.b])) - self.b


class Tube:                 # G1
    name = "G1"
    def __init__(self, R):
        self.R = R
    def sdf(self, p):
        return self.R - np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)


class Bend:                 # G2: straight inlet, 90-degree arc (centreline radius Rc = 3R), straight outlet
    """Centre of curvature C = (0,0,-Rc); the arc runs phi in [-45, 45] degrees, phi = atan2(x, z + Rc); the body station is
    phi = 0 (the origin, tangent +x). Inner wall at the body station: z = -R, bulge radius Rc - R along the flow."""
    name = "G2"
    def __init__(self, R, L_in=None, L_out=None, half_angle=np.pi / 4):
        self.R, self.Rc, self.ph = R, 3.0 * R, half_angle
        self.L_in = 4.0 * R if L_in is None else L_in
        self.L_out = 12.0 * R if L_out is None else L_out
        self.C = np.array([0.0, 0.0, -self.Rc])
        ph = self.ph
        self.P_in = self.C + self.Rc * np.array([np.sin(-ph), 0.0, np.cos(-ph)])
        self.t_in = np.array([np.cos(-ph), 0.0, -np.sin(-ph)])
        self.P_out = self.C + self.Rc * np.array([np.sin(ph), 0.0, np.cos(ph)])
        self.t_out = np.array([np.cos(ph), 0.0, -np.sin(ph)])

    @staticmethod
    def _dseg(p, A, B):
        d = B - A
        t = np.clip(((p - A) @ d) / (d @ d), 0.0, 1.0)
        return _norm(p - (A + t[..., None] * d))

    def centreline_distance(self, p):
        q = p - self.C
        rho = np.sqrt(q[..., 0] ** 2 + q[..., 2] ** 2)
        phi = np.arctan2(q[..., 0], q[..., 2])
        darc = np.where(np.abs(phi) <= self.ph, np.sqrt((rho - self.Rc) ** 2 + q[..., 1] ** 2), np.inf)
        d1 = self._dseg(p, self.P_in - self.L_in * self.t_in, self.P_in)
        d2 = self._dseg(p, self.P_out, self.P_out + self.L_out * self.t_out)
        return np.minimum(darc, np.minimum(d1, d2))

    def sdf(self, p):
        return self.R - self.centreline_distance(p)


class Stenosis:             # G3: axisymmetric about x; circular-arc profile, C1
    """Throat radius Rs at x = 0 with axial radius of curvature rho_s (circle centre (0, Rs + rho_s) in the (x, r) plane),
    joined tangentially to r = R by concave arcs of radius rho_j centred (+-x_j, R - rho_j); x_j^2 = 2 d (rho_s + rho_j) - d^2,
    d = R - Rs."""
    name = "G3"
    def __init__(self, R, Rs, rho_s, rho_j, L_in=None, L_out=None):
        self.R, self.Rs, self.rs, self.rj = R, Rs, rho_s, rho_j
        d = R - Rs
        self.xj = np.sqrt(2 * d * (rho_s + rho_j) - d * d)
        self.Os = np.array([0.0, Rs + rho_s])
        self.Oj = np.array([self.xj, R - rho_j])
        w = rho_s / (rho_s + rho_j)
        self.T = self.Os + w * (self.Oj - self.Os)          # tangent point (x_t, r_t), x_t > 0
        self.L_in = 4.0 * R if L_in is None else L_in
        self.L_out = 12.0 * R if L_out is None else L_out

    def radius(self, x):
        x = np.abs(np.asarray(x, float))
        r = np.full_like(x, self.R)
        m1 = x <= self.T[0]
        r[m1] = self.Os[1] - np.sqrt(self.rs ** 2 - x[m1] ** 2)
        m2 = (x > self.T[0]) & (x < self.xj)
        r[m2] = self.Oj[1] + np.sqrt(self.rj ** 2 - (x[m2] - self.xj) ** 2)
        return r

    def _dist2d(self, X, Rr):
        X = np.abs(X)
        # throat arc: points of the circle (Os, rs) with angle range from straight down to the tangent point
        v = np.stack([X - self.Os[0], Rr - self.Os[1]], -1)
        ang = np.arctan2(v[..., 0], -v[..., 1])                  # 0 = straight down
        ang_T = np.arctan2(self.T[0] - self.Os[0], -(self.T[1] - self.Os[1]))
        d_th = np.where((ang >= 0) & (ang <= ang_T), np.abs(_norm(v) - self.rs), np.inf)
        w = np.stack([X - self.Oj[0], Rr - self.Oj[1]], -1)
        angj = np.arctan2(-w[..., 0], w[..., 1])                 # 0 = straight up (the point x = xj, r = R)
        angj_T = np.arctan2(-(self.T[0] - self.Oj[0]), self.T[1] - self.Oj[1])
        d_j = np.where((angj >= 0) & (angj <= angj_T), np.abs(_norm(w) - self.rj), np.inf)
        d_line = np.where(X >= self.xj, np.abs(Rr - self.R), np.inf)
        d_pts = np.minimum(np.hypot(X - self.T[0], Rr - self.T[1]), np.hypot(X - self.xj, Rr - self.R))
        d_pts = np.minimum(d_pts, np.hypot(X, Rr - self.Rs))
        return np.minimum(np.minimum(d_th, d_j), np.minimum(d_line, d_pts))

    def sdf(self, p):
        r = np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)
        d = self._dist2d(p[..., 0], r)
        return np.where(r < self.radius(p[..., 0]), d, -d)


class Bifurcation:          # G4
    """Parent C: radius R along -x (x <= 0), closed by a bulb sphere of radius R at the origin. Daughters A, B: radius Rd along
    e_A = (cos th, 0, sin th), e_B = (cos th, 0, -sin th) from the origin. The lumen is the closing of the sharp union by a ball
    of radius rc (every tissue ridge rounded by a rolling ball of radius rc): sdf = dist(p, outside of L') - rc, L' = the same
    union with radii R + rc, Rd + rc. Symmetry planes y = 0 and z = 0."""
    name = "G4"
    def __init__(self, R, Rd_frac=0.76, th_deg=40.0, rc_frac=0.15, L_in=None, L_d=None, rounding="carina"):
        self.rounding = rounding
        self.R, self.Rd, self.th, self.rc = R, Rd_frac * R, np.radians(th_deg), rc_frac * R
        self.eA = np.array([np.cos(self.th), 0.0, np.sin(self.th)])
        self.eB = np.array([np.cos(self.th), 0.0, -np.sin(self.th)])
        self.R1, self.R2 = self.R + self.rc, self.Rd + self.rc
        self.tstar = np.sqrt(self.R1 ** 2 - self.R2 ** 2)
        self.L_in = 4.0 * R if L_in is None else L_in
        self.L_d = 8.0 * R if L_d is None else L_d
        # carina ellipse (dilated) in z = 0: x^2 sin^2 th + y^2 = R2^2, outside the dilated bulb
        self.Ae, self.Be = self.R2 / np.sin(self.th), self.R2
        xs = self.tstar / np.cos(self.th)
        self.phistar = np.arccos(np.clip(xs / self.Ae, -1, 1))

    def apex_x(self):
        """x of the rounded apex on the x axis."""
        return self.Ae - self.rc

    def _inside(self, q, which, tol=1e-12):
        """Strict interior of the dilated pieces, for candidate validity."""
        out = np.zeros(q.shape[:-1], bool)
        if "C" in which:
            out |= (q[..., 0] < -tol) & (np.hypot(q[..., 1], q[..., 2]) < self.R1 - tol)
        if "S" in which:
            out |= _norm(q) < self.R1 - tol
        for k, e in (("A", self.eA), ("B", self.eB)):
            if k in which:
                t = q @ e
                d = np.sqrt(np.maximum(np.sum(q * q, -1) - t * t, 0))
                out |= (t > tol) & (d < self.R2 - tol)
        return out

    def g(self, p):
        """distance from p (inside the dilated union) to its complement."""
        p = np.atleast_2d(p).astype(float)
        best = np.full(p.shape[0], np.inf)
        # C lateral
        rr = np.hypot(p[:, 1], p[:, 2])
        u = np.stack([np.zeros_like(rr), p[:, 1], p[:, 2]], -1) / np.maximum(rr, 1e-300)[:, None]
        q = np.stack([p[:, 0], np.zeros_like(rr), np.zeros_like(rr)], -1) + self.R1 * u
        ok = (q[:, 0] <= 0) & ~self._inside(q, "SAB")
        best = np.where(ok, np.minimum(best, self.R1 - rr), best)
        # bulb
        n = _norm(p)
        q = self.R1 * p / np.maximum(n, 1e-300)[:, None]
        ok = ~self._inside(q, "CAB")
        best = np.where(ok, np.minimum(best, self.R1 - n), best)
        for k, e, other in (("A", self.eA, "CSB"), ("B", self.eB, "CSA")):
            t = p @ e
            radial = p - t[:, None] * e
            d = _norm(radial)
            ur = radial / np.maximum(d, 1e-300)[:, None]
            q = t[:, None] * e + self.R2 * ur
            ok = (t >= 0) & ~self._inside(q, other)
            best = np.where(ok, np.minimum(best, self.R2 - d), best)
            # crease with the bulb: circle at t*, radius R2 about axis k
            q = self.tstar * e + self.R2 * ur
            ok = ~self._inside(q, "CB" if k == "A" else "CA")
            best = np.where(ok, np.minimum(best, np.hypot(t - self.tstar, d - self.R2)), best)
        # carina crease: ellipse arc in z = 0, phi in [-phistar, phistar]
        ph = np.linspace(-self.phistar, self.phistar, 401)
        E = np.stack([self.Ae * np.cos(ph), self.Be * np.sin(ph), np.zeros_like(ph)], -1)
        D2 = np.sum((p[:, None, :] - E[None]) ** 2, -1)
        i = np.argmin(D2, 1)
        phi = ph[i]
        for _ in range(30):     # Newton on d/dphi |p - E(phi)|^2 / 2
            ex, ey = self.Ae * np.cos(phi), self.Be * np.sin(phi)
            dx, dy = -self.Ae * np.sin(phi), self.Be * np.cos(phi)
            ddx, ddy = -ex, -ey
            f1 = -(p[:, 0] - ex) * dx - (p[:, 1] - ey) * dy
            f2 = dx * dx + dy * dy - (p[:, 0] - ex) * ddx - (p[:, 1] - ey) * ddy
            phi = np.clip(phi - f1 / np.where(np.abs(f2) > 1e-300, f2, 1e-300), -self.phistar, self.phistar)
        ex, ey = self.Ae * np.cos(phi), self.Be * np.sin(phi)
        dcar = np.sqrt((p[:, 0] - ex) ** 2 + (p[:, 1] - ey) ** 2 + p[:, 2] ** 2)
        best = np.minimum(best, dcar)
        return best

    # ---- carina-only rounding (the declared default): L = C u bulb u closing(A u B)
    def sdf_closeAB(self, p):
        p = np.atleast_2d(p).astype(float)
        best = np.full(p.shape[0], np.inf)
        inside = np.zeros(p.shape[0], bool)
        for e, eo in ((self.eA, self.eB), (self.eB, self.eA)):
            t = p @ e
            radial = p - t[:, None] * e
            d = _norm(radial)
            inside |= (t >= 0) & (d < self.R2)
            ur = radial / np.maximum(d, 1e-300)[:, None]
            q = t[:, None] * e + self.R2 * ur
            to = q @ eo
            do = np.sqrt(np.maximum(np.sum(q * q, -1) - to * to, 0))
            ok = (t >= 0) & ~((to > 1e-12) & (do < self.R2 - 1e-12))
            best = np.where(ok & (d < self.R2), np.minimum(best, self.R2 - d), best)
        ph = np.linspace(-np.pi / 2, np.pi / 2, 721)
        E = np.stack([self.Ae * np.cos(ph), self.Be * np.sin(ph), np.zeros_like(ph)], -1)
        D2 = np.sum((p[:, None, :] - E[None]) ** 2, -1)
        phi = ph[np.argmin(D2, 1)]
        for _ in range(30):
            ex, ey = self.Ae * np.cos(phi), self.Be * np.sin(phi)
            dx, dy = -self.Ae * np.sin(phi), self.Be * np.cos(phi)
            f1 = -(p[:, 0] - ex) * dx - (p[:, 1] - ey) * dy
            f2 = dx * dx + dy * dy + (p[:, 0] - ex) * ex + (p[:, 1] - ey) * ey
            phi = np.clip(phi - f1 / np.where(np.abs(f2) > 1e-300, f2, 1e-300), -np.pi / 2, np.pi / 2)
        ex, ey = self.Ae * np.cos(phi), self.Be * np.sin(phi)
        best = np.minimum(best, np.sqrt((p[:, 0] - ex) ** 2 + (p[:, 1] - ey) ** 2 + p[:, 2] ** 2))
        return np.where(inside, best - self.rc, -1.0)

    def sdf_carina_only(self, p):
        sh = p.shape[:-1]
        p2 = p.reshape(-1, 3)
        rho = np.hypot(p2[:, 1], p2[:, 2])
        sC = np.where(p2[:, 0] <= 0, self.R - rho, np.minimum(self.R - rho, -p2[:, 0]))
        sS = self.R - _norm(p2)
        sAB = np.concatenate([self.sdf_closeAB(p2[i:i + 4000]) for i in range(0, len(p2), 4000)]) if len(p2) else np.zeros(0)
        return np.maximum(np.maximum(sC, sS), sAB).reshape(sh)

    def in_dilated(self, p):
        p = np.atleast_2d(p)
        return self._inside(p, "CSAB", tol=0.0)

    def sdf(self, p):
        return self.sdf_carina_only(p) if self.rounding == "carina" else self.sdf_closing_all(p)

    def sdf_closing_all(self, p):
        sh = p.shape[:-1]
        p2 = p.reshape(-1, 3)
        inside = self.in_dilated(p2)
        out = np.full(p2.shape[0], -1.0)
        if inside.any():
            out[inside] = self.g(p2[inside]) - self.rc
        return out.reshape(sh)


# ---------------------------------------------------------------- bodies (body frame; axis z for H and F)
class Sphere:
    name = "SPH"
    def sdf_body(self, pb):
        return _norm(pb) - 1.0


class Helix:
    """w35m's H: centreline X(t) = (0.8 cos t, 0.8 sin t, 1.2 t/(2 pi) - 1.5), t in [0, 5 pi]; body = points within 0.2 of it."""
    name = "H"
    rh, pitch, rho, t1 = 0.8, 1.2, 0.2, 5 * np.pi

    def X(self, t):
        return np.stack([self.rh * np.cos(t), self.rh * np.sin(t), self.pitch * t / (2 * np.pi) - 1.5], -1)

    def dX(self, t):
        return np.stack([-self.rh * np.sin(t), self.rh * np.cos(t), np.full_like(t, self.pitch / (2 * np.pi))], -1)

    def curve_distance(self, pb, nt=4001):
        pb = np.atleast_2d(pb)
        tt = np.linspace(0, self.t1, nt)
        Xs = self.X(tt)
        out = np.empty(pb.shape[0])
        for i0 in range(0, pb.shape[0], 2000):
            P = pb[i0:i0 + 2000]
            D2 = np.sum((P[:, None, :] - Xs[None]) ** 2, -1)
            t = tt[np.argmin(D2, 1)]
            for _ in range(20):
                r = P - self.X(t)
                d1 = self.dX(t)
                d2 = np.stack([-self.rh * np.cos(t), -self.rh * np.sin(t), np.zeros_like(t)], -1)
                f1 = -np.sum(r * d1, -1)
                f2 = np.sum(d1 * d1, -1) - np.sum(r * d2, -1)
                t = np.clip(t - f1 / f2, 0.0, self.t1)
            out[i0:i0 + 2000] = _norm(P - self.X(t))
        return out

    def sdf_body(self, pb):
        return self.curve_distance(pb) - self.rho


class Fins:
    """Three fins on a core. Core: points within 0.3 of the axis segment |z| <= 1.5. Fin k: for |z| <= 1.5, rotate (x, y) by
    -th_k(z), th_k = 2 pi k/3 + pi z/3; the planar stadium {|v| <= 0.04, 0 <= u <= 0.96} U disc((0.96, 0), 0.04). End faces of
    the fins are flat. sdf_body is an implicit function: exact zero set, not a distance."""
    name = "F"
    t, L, zh, rcore = 0.04, 0.96, 1.5, 0.3

    def th(self, z, k):
        return 2 * np.pi * k / 3 + np.pi * z / 3

    def stadium2d(self, u, v):
        du = np.clip(u, 0.0, self.L)
        d = np.hypot(u - du, v) - self.t
        return d

    def sdf_body(self, pb):
        pb = np.atleast_2d(pb)
        x, y, z = pb[:, 0], pb[:, 1], pb[:, 2]
        zc = np.clip(z, -self.zh, self.zh)
        core = np.sqrt(x * x + y * y + (z - zc) ** 2) - self.rcore
        out = core
        for k in range(3):
            a = self.th(z, k)
            u = np.cos(a) * x + np.sin(a) * y
            v = -np.sin(a) * x + np.cos(a) * y
            fin = np.maximum(self.stadium2d(u, v), np.abs(z) - self.zh)
            out = np.minimum(out, fin)
        return out

    def tip_points(self, nz=3001, ns=181):
        """points exactly on the fins' rounded tip (the semicircle of the stadium), for gap searches."""
        z = np.linspace(-self.zh, self.zh, nz)
        s = np.linspace(-np.pi / 2, np.pi / 2, ns)
        Z, S = np.meshgrid(z, s, indexing="ij")
        pts = []
        for k in range(3):
            a = self.th(Z, k)
            u = self.L + self.t * np.cos(S)
            v = self.t * np.sin(S)
            pts.append(np.stack([np.cos(a) * u - np.sin(a) * v, np.sin(a) * u + np.cos(a) * v, Z], -1).reshape(-1, 3))
        return np.concatenate(pts)


# ---------------------------------------------------------------- poses
def rot_body_to_lab(spin=0.0, axis="x"):
    """Body axis z mapped to lab x (G1-G3 poses), after a spin about the body axis."""
    c, s = np.cos(spin), np.sin(spin)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])     # z -> x, x -> -z
    return Ry @ Rz


def body_gap(wall, body, c, Rm=None):
    """smallest wall distance minus the body's own radius term; exact for SPH and H (wall sdf is a distance)."""
    if body.name == "SPH":
        return float(wall.sdf(np.atleast_2d(c))[0] - 1.0)
    if body.name == "H":
        t = np.linspace(0, body.t1, 20001)
        pts = c + body.X(t) @ Rm.T
        g = wall.sdf(pts)
        i = np.argmin(g)
        lo, hi = t[max(i - 2, 0)], t[min(i + 2, len(t) - 1)]
        for _ in range(80):     # golden section on the smooth min
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            g1 = wall.sdf((c + body.X(np.array([m1])) @ Rm.T))[0]
            g2 = wall.sdf((c + body.X(np.array([m2])) @ Rm.T))[0]
            lo, hi = (lo, m2) if g1 < g2 else (m1, hi)
        tm = 0.5 * (lo + hi)
        return float(wall.sdf(c + body.X(np.array([tm])) @ Rm.T)[0] - body.rho)
    if body.name == "F":
        from scipy.optimize import minimize
        pts = c + body.tip_points() @ Rm.T
        g = wall.sdf(pts)
        order = np.argsort(g)[:5]
        best = np.inf
        Zs = np.linspace(-body.zh, body.zh, 3001)
        for j in order:
            k = j // (3001 * 181)
            rem = j % (3001 * 181)
            z0, s0 = Zs[rem // 181], np.linspace(-np.pi / 2, np.pi / 2, 181)[rem % 181]
            def f(q, k=k):
                z, s = np.clip(q[0], -body.zh, body.zh), q[1]
                a = body.th(z, k)
                u, v = body.L + body.t * np.cos(s), body.t * np.sin(s)
                pb = np.array([np.cos(a) * u - np.sin(a) * v, np.sin(a) * u + np.cos(a) * v, z])
                return float(wall.sdf(np.atleast_2d(c + Rm @ pb))[0])
            r = minimize(f, [z0, s0], method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 4000})
            best = min(best, r.fun)
        return float(best)
    raise ValueError(body.name)


def press(wall, body, c0, direction, gap, Rm=None, lo=0.0, hi=None):
    """move the body centre from c0 along direction until its gap to the wall equals `gap` (bisection to 1e-13)."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    if hi is None:
        hi = 10.0
    f = lambda s: body_gap(wall, body, c0 + s * d, Rm) - gap
    flo, fhi = f(lo), f(hi)
    assert flo > 0 > fhi or flo < 0 < fhi, (flo, fhi)
    for _ in range(100):
        m = 0.5 * (lo + hi)
        fm = f(m)
        if (fm > 0) == (flo > 0):
            lo, flo = m, fm
        else:
            hi = m
        if hi - lo < 1e-14:
            break
    return c0 + 0.5 * (lo + hi) * d


# ---------------------------------------------------------------- the declared case table
def sphere_centre(geom, gap, pose=None):
    """centre of SPH for each geometry and pose (closed forms where they exist)."""
    if geom.name == "G0":
        return np.array([0.0, 0.0, 1.0 + gap])
    if geom.name == "G0b":
        return np.array([0.0, 0.0, 1.0 + gap])
    if geom.name in ("G1", "G2"):
        return np.array([0.0, 0.0, -(geom.R - 1.0 - gap)])
    if geom.name == "G3":
        return np.array([0.0, 0.0, -(geom.Rs - 1.0 - gap)])
    if geom.name == "G4":
        if pose == "P1":     # on the x axis, nearest point the rounded apex (checked: beyond the arch's centre of curvature)
            return np.array([geom.apex_x() - 1.0 - gap, 0.0, 0.0])
        if pose == "P2":     # flank: nearest point on the carina fillet, 30 degrees round from the apex towards A
            ps = np.radians(30.0)
            Cb = np.array([geom.Ae, 0.0, 0.0])
            return Cb + (geom.rc + 1.0 + gap) * np.array([-np.cos(ps), 0.0, np.sin(ps)])
        if pose == "P3":     # entering A: centre at t = 1.5 R on A's axis line, pressed towards A's carina-side wall
            nin = np.array([np.sin(geom.th), 0.0, -np.cos(geom.th)])
            return press(geom, Sphere(), 1.5 * geom.R * geom.eA, nin, gap)
    raise ValueError(geom.name)
