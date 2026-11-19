"""Sparse operators as padded (ELL) gathers, a smoothed-aggregation V-cycle whose hierarchy pyamg sets up outside JAX and
JAX runs, a preconditioned CG, a restarted right-preconditioned GMRES, and the coupled Stokes solve built from them."""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def ell(A):
    """A scipy sparse matrix as ``{"idx": (rows, K), "val": (rows, K)}`` (rows padded with zeros)."""
    A = A.tocsr()
    A.sum_duplicates()
    rows = A.shape[0]
    counts = np.diff(A.indptr)
    width = max(1, int(counts.max()))
    idx, val = np.zeros((rows, width), np.int32), np.zeros((rows, width))
    r = np.repeat(np.arange(rows), counts)
    slot = np.arange(A.nnz) - A.indptr[r]
    idx[r, slot], val[r, slot] = A.indices, A.data
    return {"idx": idx, "val": val}


def matvec(E, x):
    val = E["val"]
    if x.ndim == 2:
        val = val[..., None]
    return (val * x[E["idx"]]).sum(axis=1)


def hierarchy(A, theta=0.0, strength="symmetric", max_coarse=300, omega=4.0 / 3.0, smooth="jacobi"):
    """pyamg's smoothed-aggregation hierarchy of the SPD (or semidefinite) ``A`` as static arrays, its damped-Jacobi
    smoother's weights, the coarsest level's dense pseudo-inverse, and its set-up facts."""
    import time

    import pyamg
    from pyamg.util.linalg import approximate_spectral_radius
    t0 = time.time()
    ml = pyamg.smoothed_aggregation_solver(A.tocsr(), strength=(strength, {"theta": theta}) if strength == "symmetric" else strength,
                                           max_coarse=max_coarse, max_levels=12)
    levels = []
    for lv in ml.levels[:-1]:
        D = lv.A.diagonal()
        rho = approximate_spectral_radius((lv.A.multiply(1.0 / D[:, None])).tocsr(), maxiter=15)
        levels.append({"A": ell(lv.A), "P": ell(lv.P), "R": ell(lv.R), "dinv": omega / (rho * D)})
    coarse = np.linalg.pinv(ml.levels[-1].A.toarray(), hermitian=True)
    facts = {"levels": len(ml.levels), "sizes": [int(lv.A.shape[0]) for lv in ml.levels],
             "operator_complexity": float(ml.operator_complexity()), "grid_complexity": float(ml.grid_complexity()),
             "ell_widths": [int(L["A"]["idx"].shape[1]) for L in levels], "setup_s": time.time() - t0, "theta": theta,
             "strength": str(strength)}
    return {"levels": levels, "coarse": coarse}, facts


def vcycle(H, b, level=0):
    if level == len(H["levels"]):
        return H["coarse"] @ b
    L = H["levels"][level]
    dinv = L["dinv"] if b.ndim == 1 else L["dinv"][:, None]
    x = dinv * b
    r = b - matvec(L["A"], x)
    x = x + matvec(L["P"], vcycle(H, matvec(L["R"], r), level + 1))
    return x + dinv * (b - matvec(L["A"], x))


def pcg(apply, precondition, b, tol, maxiter, mean_free=False):
    """Preconditioned CG to ``|r| <= tol |b|``; returns ``(x, iterations, |r| / |b|)``."""
    fix = (lambda v: v - jnp.mean(v)) if mean_free else (lambda v: v)
    b = fix(b)
    bnorm = jnp.linalg.norm(b)
    z = fix(precondition(b))
    state = (jnp.zeros_like(b), b, z, z, jnp.vdot(b, z), jnp.zeros((), jnp.int32))

    def cond(s):
        return (s[5] < maxiter) & (jnp.linalg.norm(s[1]) > tol * bnorm)

    def body(s):
        x, r, z, p, rz, k = s
        q = apply(p)
        alpha = rz / jnp.vdot(p, q)
        x, r = x + alpha * p, fix(r - alpha * q)
        z = fix(precondition(r))
        rz_new = jnp.vdot(r, z)
        return (x, r, z, z + (rz_new / rz) * p, rz_new, k + 1)

    x, r, _, _, _, k = lax.while_loop(cond, body, state)
    return x, k, jnp.linalg.norm(r) / bnorm


def gmres(apply, precondition, b, tol, restart, max_cycles):
    """Right-preconditioned GMRES(restart), whole cycles; returns ``(x, cycles, |b - A x| / |b|)``."""
    bnorm = jnp.linalg.norm(b)
    size = b.shape[0]

    def cycle(x):
        r = b - apply(x)
        beta = jnp.linalg.norm(r)
        V = jnp.zeros((restart + 1, size), b.dtype).at[0].set(r / beta)
        Hm = jnp.zeros((restart + 1, restart), b.dtype)

        def arnoldi(carry, j):
            V, Hm = carry
            w = apply(precondition(V[j]))
            h = V @ w
            w = w - h @ V
            h2 = V @ w
            w = w - h2 @ V
            norm = jnp.linalg.norm(w)
            V = V.at[j + 1].set(w / jnp.maximum(norm, jnp.finfo(b.dtype).tiny))
            return (V, Hm.at[:, j].set((h + h2).at[j + 1].set(norm))), None

        (V, Hm), _ = lax.scan(arnoldi, (V, Hm), jnp.arange(restart))
        y = jnp.linalg.lstsq(Hm, jnp.zeros(restart + 1, b.dtype).at[0].set(beta))[0]
        x = x + precondition(y @ V[:restart])
        return x, jnp.linalg.norm(b - apply(x)) / bnorm

    def cond(s):
        return (s[2] < max_cycles) & (s[1] > tol)

    def body(s):
        x, res = cycle(s[0])
        return (x, res, s[2] + 1)

    x, res, cycles = lax.while_loop(cond, body, (jnp.zeros_like(b), jnp.ones((), b.dtype), jnp.zeros((), jnp.int32)))
    return x, cycles, res


class StokesSolver:
    """The coupled system of :class:`stokes.Stokes` as ELL blocks, and its solve: GMRES under the block lower-triangular
    preconditioner ``u = AMG(A0) r_u``, ``p = (r_p - D u) / (V / mu + diag C)``."""

    def __init__(self, st, dtype, theta=0.0, restart=40, sweeps=1, strength="symmetric"):
        self.n = st.n
        H, self.facts = hierarchy(st.A0, theta=theta, strength=strength)
        tab = {"A": ell(st.A), "G": [ell(g) for g in st.Gp], "D": [ell(d) for d in st.Dv], "C": ell(st.C), "H": H,
               "schur": 1.0 / st.schur_diag}
        self.tab = jax.tree_util.tree_map(lambda t: jnp.asarray(t, dtype=dtype if np.issubdtype(np.asarray(t).dtype, np.floating) else np.int32), tab)
        self.restart, self.sweeps = restart, sweeps
        n = self.n

        def apply(tab, x):
            u, p = x[:3 * n].reshape(3, n).T, x[3 * n:]
            ru = matvec(tab["A"], u) + jnp.stack([matvec(g, p) for g in tab["G"]], axis=1)
            rp = sum(matvec(d, u[:, c]) for c, d in enumerate(tab["D"])) + matvec(tab["C"], p)
            return jnp.concatenate([ru.T.reshape(-1), rp])

        def precondition(tab, r):
            ru, rp = r[:3 * n].reshape(3, n).T, r[3 * n:]
            u = vcycle(tab["H"], ru)
            for _ in range(sweeps - 1):
                u = u + vcycle(tab["H"], ru - matvec(tab["A"], u))
            rp = rp - sum(matvec(d, u[:, c]) for c, d in enumerate(tab["D"]))
            rp = rp - jnp.mean(rp)
            return jnp.concatenate([u.T.reshape(-1), tab["schur"] * rp])

        self.apply = apply
        self._solve = jax.jit(lambda tab, b, tol, cycles: gmres(lambda x: apply(tab, x), lambda r: precondition(tab, r), b, tol,
                                                               restart, cycles))

    def solve(self, b, tol=1e-9, max_cycles=50):
        x, cycles, res = self._solve(self.tab, b, tol, max_cycles)
        return x, int(cycles) * self.restart, float(res)
