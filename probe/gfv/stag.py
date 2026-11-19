"""G0: the Cartesian MAC grid's operators as stencils (a copy of the MAC grid kind's, checked against it where the package
is installed) and the same operators on the same cells and faces held as a graph.

The graph has two levels. The primal one (cells, faces) carries the divergence (cells from faces) and the gradient (faces
from cells). The dual one (faces as nodes; a dual edge joins two neighbouring faces of the same component) carries the
Laplacian and the convection of the face-normal velocity; a dual edge also names the two faces whose average advects
across it. Face ``(a, c)`` is the face below cell ``c`` along axis ``a`` (the MAC arrays' layout)."""
import jax.numpy as jnp
import numpy as np

from . import prims


# ---------------------------------------------------------------------------------------------------------------------
# the MAC stencils (periodic box, cubic cells)
# ---------------------------------------------------------------------------------------------------------------------
def _roll(a, s, ax):
    return jnp.roll(a, s, ax)


def mac_divergence(u, h):
    return sum(((_roll(c, -1, ax) - c) / h for ax, c in enumerate(u)), start=jnp.zeros_like(u[0]))


def mac_gradient(p, h):
    return tuple((p - _roll(p, 1, ax)) / h for ax in range(3))


def mac_laplacian(u, h):
    def of(a):
        return sum(((_roll(a, 1, ax) + _roll(a, -1, ax) - 2.0 * a) for ax in range(3)), start=jnp.zeros_like(a)) / h ** 2
    return tuple(of(c) for c in u)


def mac_convection(vel, h):
    u, v, w = vel
    to_face = lambda a, ax: 0.5 * (_roll(a, 1, ax) + a)
    to_centre = lambda a, ax: 0.5 * (a + _roll(a, -1, ax))
    d_face = lambda a, ax: (a - _roll(a, 1, ax)) / h
    d_centre = lambda a, ax: (_roll(a, -1, ax) - a) / h
    centred = [to_centre(c, ax) for ax, c in enumerate(vel)]
    normal = [d_face(c * c, ax) for ax, c in enumerate(centred)]
    xy = to_face(u, 1) * to_face(v, 0)
    xz = to_face(u, 2) * to_face(w, 0)
    yz = to_face(v, 2) * to_face(w, 1)
    cu = normal[0] + d_centre(xy, 1) + d_centre(xz, 2)
    cv = normal[1] + d_centre(xy, 0) + d_centre(yz, 2)
    cw = normal[2] + d_centre(xz, 0) + d_centre(yz, 1)
    return (-cu, -cv, -cw)


def mac_stage(par, u, p):
    """One explicit stage without its Poisson solve: momentum's right-hand side, the update, the divergence."""
    h, nu, dt, beta = par
    lap, conv, grad = mac_laplacian(u, h), mac_convection(u, h), mac_gradient(p, h)
    u1 = tuple(c + dt * (nu * l + n - g) for c, l, n, g in zip(u, lap, conv, grad))
    d = mac_divergence(u1, h)
    return u1, p - beta * d


# ---------------------------------------------------------------------------------------------------------------------
# the same cells and faces as a graph
# ---------------------------------------------------------------------------------------------------------------------
class StaggeredGraph:
    def __init__(self, cells, h, order="native", impl="scatter", plant=False):
        self.cells, self.h, self.impl, self.order = tuple(cells), h, impl, order
        n = int(np.prod(cells))
        self.n = n
        grid = np.arange(n).reshape(cells)
        dn = [np.roll(grid, 1, ax).ravel() for ax in range(3)]
        up = [np.roll(grid, -1, ax).ravel() for ax in range(3)]
        c = grid.ravel()
        p_own = np.concatenate(dn)
        p_nei = np.concatenate([c, c, c])
        own, nei, q1, q2 = [], [], [], []
        for a in range(3):
            for d in range(3):
                if d == a:
                    o, m = a * n + c, a * n + up[a]
                    own.append(o), nei.append(m), q1.append(o), q2.append(m)
                else:
                    own.append(a * n + dn[d]), nei.append(a * n + c)
                    q1.append(d * n + dn[a]), q2.append(d * n + c)
        own, nei, q1, q2 = (np.concatenate(x) for x in (own, nei, q1, q2))
        if plant:  # one advecting face of one dual edge named wrongly
            q2 = q2.copy()
            q2[len(q2) // 3] = (q2[len(q2) // 3] + 1) % (3 * n)
        # orders: new position -> old index
        idx = np.stack(np.meshgrid(*[np.arange(k) for k in cells], indexing="ij"), -1).reshape(-1, 3).astype(float)
        cell_pts = idx + 0.5
        face_pts = np.concatenate([cell_pts - 0.5 * np.eye(3)[a] for a in range(3)])
        self.pc = prims.node_order(order, cell_pts, p_own, p_nei, n)
        self.pf = prims.node_order(order, face_pts, own, nei, 3 * n)
        inv_c, inv_f = np.argsort(self.pc), np.argsort(self.pf)
        p_own, p_nei = inv_c[p_own[self.pf]], inv_c[p_nei[self.pf]]
        own, nei, q1, q2 = inv_f[own], inv_f[nei], inv_f[q1], inv_f[q2]
        e = np.lexsort((nei, own))
        own, nei, q1, q2 = own[e], nei[e], q1[e], q2[e]
        self.tab = {}
        gather_impl = "padded" if impl == "slots" else impl
        self.tab["gp"], self._gp = prims.make_gather(np.stack([p_own, p_nei]), n, gather_impl)
        self.tab["np"], self._np = prims.make_net(p_own, p_nei, n, "padded" if impl == "slots" else impl)
        if impl == "slots":
            both = np.concatenate([own, nei])
            table, ragged, _, _ = prims.transposed_table(both, 3 * n)
            assert not ragged
            edge = table % own.size
            is_own = table < own.size
            nb = np.where(is_own, nei[edge], own[edge])
            self.tab["sg"] = np.where(is_own, 1.0, -1.0)
            self.tab["gs"], self._gs = prims.make_gather(np.stack([nb, q1[edge], q2[edge]]), 3 * n, "padded")
        else:
            self.tab["gd"], self._gd = prims.make_gather(np.stack([own, nei, q1, q2]), 3 * n, impl)
            self.tab["nd"], self._nd = prims.make_net(own, nei, 3 * n, impl)

    # mac arrays <-> graph vectors
    def to_graph(self, u, p):
        return jnp.concatenate([c.ravel() for c in u])[self.pf], p.ravel()[self.pc]

    def to_mac(self, u, p):
        inv_f, inv_c = np.argsort(self.pf), np.argsort(self.pc)
        flat = u[inv_f]
        return tuple(flat[a * self.n:(a + 1) * self.n].reshape(self.cells) for a in range(3)), p[inv_c].reshape(self.cells)

    def divergence(self, tab, u):
        return self._np(tab["np"], u) / self.h

    def gradient(self, tab, p):
        g = self._gp(tab["gp"], p)
        return (g[1] - g[0]) / self.h

    def _flux(self, tab, u, nu):
        """The momentum right-hand side's Laplacian and convection parts, ``(nu lap, conv)`` weights ``(nu, 1)``."""
        h = self.h
        if self.impl == "slots":
            g = self._gs(tab["gs"], u)
            mine = u[:, None]
            sg = tab["sg"].astype(u.dtype)
            return (nu * (g[0] - mine) / h ** 2 - sg * (0.5 * (mine + g[0])) * (0.5 * (g[1] + g[2])) / h).sum(axis=1)
        g = self._gd(tab["gd"], u)
        return self._nd(tab["nd"], nu * (g[1] - g[0]) / h ** 2 - (0.5 * (g[0] + g[1])) * (0.5 * (g[2] + g[3])) / h)

    def laplacian(self, tab, u):
        if self.impl == "slots":
            return (self._gs(tab["gs"], u)[0] - u[:, None]).sum(axis=1) / self.h ** 2
        g = self._gd(tab["gd"], u)
        return self._nd(tab["nd"], g[1] - g[0]) / self.h ** 2

    def convection(self, tab, u):
        if self.impl == "slots":
            g = self._gs(tab["gs"], u)
            sg = tab["sg"].astype(u.dtype)
            return -(sg * (0.5 * (u[:, None] + g[0])) * (0.5 * (g[1] + g[2]))).sum(axis=1) / self.h
        g = self._gd(tab["gd"], u)
        return -self._nd(tab["nd"], (0.5 * (g[0] + g[1])) * (0.5 * (g[2] + g[3]))) / self.h

    def stage(self, tab, par, u, p):
        h, nu, dt, beta = par
        u1 = u + dt * (self._flux(tab, u, nu) - self.gradient(tab, p))
        return u1, p - beta * self.divergence(tab, u1)


def fields(cells, h, dtype, seed=0):
    """A smooth velocity of amplitude 0.1 m/s with 10 % noise, and a pressure, on the MAC arrays."""
    rng = np.random.default_rng(seed)
    x = np.stack(np.meshgrid(*[np.arange(k) / k for k in cells], indexing="ij"))
    two_pi = 2 * np.pi
    u = [0.1 * np.sin(two_pi * (x[(a + 1) % 3] + 0.3 * a)) * np.cos(two_pi * x[(a + 2) % 3]) + 0.01 * rng.standard_normal(cells)
         for a in range(3)]
    p = 0.01 * np.cos(two_pi * x[0]) * np.sin(two_pi * x[2]) + 0.001 * rng.standard_normal(cells)
    return tuple(jnp.asarray(c, dtype) for c in u), jnp.asarray(p, dtype)
