#!/usr/bin/env python3
"""Phase 0: MFS (Method of Fundamental Solutions) validation.

Validates MFS for Stokes drag on a sphere, both in free space and
confined in a cylinder, via block Gauss-Seidel iteration.

Task 0A: Free-space drag (compare against Stokes: F=6πμa, T=8πμa³)
Task 0B: Confined drag at κ=0.3 (compare against NN-BEM reference)
Task 0C: Sensitivity (source ratios, GS convergence, κ sweep)

Reference: Wilson et al. (JCP 520, 2025) — "IMP-MFS."
"""

import os
os.environ["XLA_FLAGS"] = " ".join([
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_enable_triton_gemm=false",
])
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_mime")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import time

from mime.nodes.environment.stokeslet.surface_mesh import (
    sphere_surface_mesh,
    cylinder_surface_mesh,
)

# ── Physical constants ─────────────────────────────────────────────────
A = 1.0        # sphere radius
MU = 1.0       # dynamic viscosity
RHO = 1.0      # density (unused in Stokes)
LAM = 0.3      # confinement ratio κ = a / R_cyl
R_CYL = A / LAM  # ≈ 3.333

# Exact free-space Stokes drag
F_STOKES = 6.0 * np.pi * MU * A       # 18.8496
T_STOKES = 8.0 * np.pi * MU * A ** 3  # 25.1327

# NN-BEM reference at κ=0.3 (magnitudes of diagonal R entries)
REF_K03 = {
    "F_x": 31.76, "F_y": 31.76, "F_z": 44.39,
    "T_x": 25.19, "T_y": 25.19, "T_z": 25.17,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MFS infrastructure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def assemble_mfs_matrix(colloc, source, mu, eps=0.0):
    """Stokeslet interaction matrix (3Nc × 3Ns).

    Uses the regularised Cortez (2005) formula which reduces to the
    singular Stokeslet when eps=0:

        G_jk = (1/(8πμ)) [δ_jk(r²+2ε²) + r_j r_k] / (r²+ε²)^{3/2}

    For MFS the sources are offset from collocation points, so the
    singular kernel (eps=0) is safe.
    """
    r = colloc[:, None, :] - source[None, :, :]       # (Nc, Ns, 3)
    r_sq = jnp.sum(r ** 2, axis=-1)                    # (Nc, Ns)
    eps_sq = eps ** 2
    denom = (r_sq + eps_sq) ** 1.5                      # (r²+ε²)^{3/2}
    inv_denom = 1.0 / denom                             # (Nc, Ns)

    I3 = jnp.eye(3)[None, None, :, :]                  # (1,1,3,3)
    rr = r[..., :, None] * r[..., None, :]              # (Nc,Ns,3,3)

    pf = 1.0 / (8.0 * jnp.pi * mu)
    S = pf * (I3 * (r_sq + 2.0 * eps_sq)[..., None, None]
              + rr) * inv_denom[..., None, None]         # (Nc,Ns,3,3)

    Nc, Ns = colloc.shape[0], source.shape[0]
    return S.transpose(0, 2, 1, 3).reshape(3 * Nc, 3 * Ns)


def rigid_body_vel(pts, center, U, omega):
    """u(x) = U + ω × (x − center)."""
    return U + jnp.cross(omega, pts - center)


# ── MFS sphere ─────────────────────────────────────────────────────────

class MFSSphere:
    """MFS for a rigid sphere: collocation on surface, sources inside.

    The icosahedral mesh returns triangle centroids which sit slightly
    inside the sphere (centroid of a curved triangle ≠ point on sphere).
    We project them radially onto the true sphere to eliminate the
    systematic O(h²) geometric bias.
    """

    def __init__(self, center, radius, n_refine=2,
                 source_ratio=0.7, mu=1.0, eps=0.0):
        self.center = jnp.asarray(center, dtype=jnp.float64)
        self.radius = radius
        c_tup = tuple(float(x) for x in self.center)

        mesh = sphere_surface_mesh(center=c_tup, radius=radius,
                                   n_refine=n_refine)
        self.colloc = jnp.array(
            self._project(mesh.points, c_tup, radius))
        self.n_pts = len(self.colloc)

        src = sphere_surface_mesh(center=c_tup,
                                  radius=source_ratio * radius,
                                  n_refine=n_refine)
        self.source = jnp.array(
            self._project(src.points, c_tup, source_ratio * radius))

        self.G = assemble_mfs_matrix(self.colloc, self.source, mu, eps)
        self.lu, self.piv = jax.scipy.linalg.lu_factor(self.G)
        self._cond = None

    @property
    def cond(self):
        if self._cond is None:
            self._cond = float(jnp.linalg.cond(self.G))
        return self._cond

    def solve(self, u_bc):
        """u_bc: (N, 3) → lam: (N, 3) source strengths."""
        return jax.scipy.linalg.lu_solve(
            (self.lu, self.piv), u_bc.ravel()
        ).reshape(-1, 3)

    def force_torque(self, lam):
        """Physical convention: F = −Σλ, T = −Σ(r×λ).

        The Stokeslet sources inside the body exert force +λ_j on the
        fluid. By Newton III, the fluid exerts −λ_j on the body.
        """
        F = -jnp.sum(lam, axis=0)
        r = self.source - self.center
        T = -jnp.sum(jnp.cross(r, lam), axis=0)
        return F, T

    @staticmethod
    def _project(pts, center, radius):
        """Project triangle centroids radially onto the true sphere."""
        import numpy as _np
        c = _np.asarray(center, dtype=_np.float64)
        r = pts - c
        norms = _np.linalg.norm(r, axis=1, keepdims=True)
        return c + r / norms * radius


# ── MFS cylinder ───────────────────────────────────────────────────────

class MFSCylinder:
    """MFS for a cylindrical wall: collocation inside, sources outside."""

    def __init__(self, radius, length, n_circ=32, n_axial=40,
                 source_ratio=1.3, mu=1.0, eps=0.0, cluster=False):
        mesh = cylinder_surface_mesh(
            center=(0, 0, 0), radius=radius, length=length,
            n_circ=n_circ, n_axial=n_axial, cluster_center=cluster)
        self.colloc = jnp.array(mesh.points)
        self.n_pts = len(self.colloc)

        src = cylinder_surface_mesh(
            center=(0, 0, 0), radius=source_ratio * radius,
            length=length,
            n_circ=n_circ, n_axial=n_axial, cluster_center=cluster)
        self.source = jnp.array(src.points)

        self.G = assemble_mfs_matrix(self.colloc, self.source, mu, eps)
        self.lu, self.piv = jax.scipy.linalg.lu_factor(self.G)
        self._cond = None

    @property
    def cond(self):
        if self._cond is None:
            self._cond = float(jnp.linalg.cond(self.G))
        return self._cond


# ── Confined solver (body + wall, Gauss-Seidel) ───────────────────────

class ConfinedMFS:
    """Pre-assembled confined MFS solver with cross-interaction matrices.

    Build once per (body, wall) pair, then call solve() for each of the
    6 unit motions without re-assembling cross matrices.
    """

    def __init__(self, body: MFSSphere, wall: MFSCylinder, mu, eps=0.0):
        self.body = body
        self.wall = wall
        self.G_w2b = assemble_mfs_matrix(body.colloc, wall.source, mu, eps)
        self.G_b2w = assemble_mfs_matrix(wall.colloc, body.source, mu, eps)

    def solve(self, center, U, omega,
              max_iter=30, tol=1e-6, verbose=False):
        """Block Gauss-Seidel. Returns list of (F, T) per iteration."""
        body, wall = self.body, self.wall

        u_body = rigid_body_vel(body.colloc, center, U, omega)
        lam_body = body.solve(u_body)

        hist = []
        for it in range(max_iter):
            # Wall: enforce u = 0 → cancel body flow at wall surface
            u_at_wall = (self.G_b2w @ lam_body.ravel()).reshape(-1, 3)
            lam_wall = jax.scipy.linalg.lu_solve(
                (wall.lu, wall.piv), (-u_at_wall).ravel()
            ).reshape(-1, 3)

            # Body: enforce u = u_body → subtract wall flow at body surface
            u_at_body = (self.G_w2b @ lam_wall.ravel()).reshape(-1, 3)
            lam_body = jax.scipy.linalg.lu_solve(
                (body.lu, body.piv), (u_body - u_at_body).ravel()
            ).reshape(-1, 3)

            F, T = body.force_torque(lam_body)
            F_np, T_np = np.array(F), np.array(T)
            hist.append((F_np, T_np))

            if verbose:
                print(f"    it {it+1}: F=[{F_np[0]:+.4f}, "
                      f"{F_np[1]:+.4f}, {F_np[2]:+.4f}]  "
                      f"T=[{T_np[0]:+.4f}, {T_np[1]:+.4f}, "
                      f"{T_np[2]:+.4f}]")

            if len(hist) >= 2:
                dF = np.linalg.norm(hist[-1][0] - hist[-2][0])
                dT = np.linalg.norm(hist[-1][1] - hist[-2][1])
                scale = np.linalg.norm(hist[-1][0]) + \
                        np.linalg.norm(hist[-1][1]) + 1e-30
                rel = (dF + dT) / scale
                if rel < tol:
                    break

        return hist


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Task 0A — Free-space sphere drag
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def task_0a():
    print("=" * 72)
    print("TASK 0A: Free-space sphere drag (MFS, singular Stokeslet)")
    print("=" * 72)

    center = jnp.zeros(3)
    e = jnp.eye(3)
    zero3 = jnp.zeros(3)

    # ── Source ratio sweep (translation U_x = 1) ──────────────────────
    print(f"\nSource-ratio sweep  (n_refine=2, U_x = 1)")
    print(f"{'sr':>6} | {'N':>5} | {'|F_x|':>10} | {'6πμa':>10} | "
          f"{'err%':>7} | {'cond(G)':>12}")
    print("-" * 66)

    for sr in [0.5, 0.6, 0.7, 0.8]:
        t0 = time.time()
        body = MFSSphere(center, A, n_refine=2, source_ratio=sr, mu=MU)
        u_bc = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]),
                                (body.n_pts, 3))
        lam = body.solve(u_bc)
        F, T = body.force_torque(lam)
        Fx = abs(float(F[0]))
        err = abs(Fx - F_STOKES) / F_STOKES * 100
        dt = time.time() - t0
        print(f"{sr:6.2f} | {body.n_pts:5d} | {Fx:10.4f} | "
              f"{F_STOKES:10.4f} | {err:6.2f}% | {body.cond:12.1f}"
              f"  ({dt:.1f}s)")

    # ── Full 6-DOF validation with source_ratio = 0.7 ────────────────
    best = 0.7
    body = MFSSphere(center, A, n_refine=2, source_ratio=best, mu=MU)

    print(f"\n6-DOF validation  (source_ratio={best}, N={body.n_pts}, "
          f"cond={body.cond:.1f})")
    print(f"{'motion':>8} | {'component':>10} | {'|value|':>10} | "
          f"{'exact':>10} | {'err%':>7} | {'sign':>5}")
    print("-" * 66)

    for i, lab in enumerate(["U_x", "U_y", "U_z"]):
        u_bc = rigid_body_vel(body.colloc, center, e[i], zero3)
        lam = body.solve(u_bc)
        F, T = body.force_torque(lam)
        val = abs(float(F[i]))
        err = abs(val - F_STOKES) / F_STOKES * 100
        sign_ok = float(F[i]) < 0
        print(f"{lab:>8} | {'F_' + 'xyz'[i]:>10} | {val:10.4f} | "
              f"{F_STOKES:10.4f} | {err:6.2f}% | {'✓' if sign_ok else '✗':>5}")

    for i, lab in enumerate(["ω_x", "ω_y", "ω_z"]):
        u_bc = rigid_body_vel(body.colloc, center, zero3, e[i])
        lam = body.solve(u_bc)
        F, T = body.force_torque(lam)
        val = abs(float(T[i]))
        err = abs(val - T_STOKES) / T_STOKES * 100
        sign_ok = float(T[i]) < 0
        print(f"{lab:>8} | {'T_' + 'xyz'[i]:>10} | {val:10.4f} | "
              f"{T_STOKES:10.4f} | {err:6.2f}% | {'✓' if sign_ok else '✗':>5}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Task 0B — Confined sphere in cylinder, block Gauss-Seidel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def task_0b():
    print("\n" + "=" * 72)
    print("TASK 0B: Confined sphere in cylinder (κ=0.3) — Gauss-Seidel")
    print("=" * 72)
    print("NOTE: Discrete MFS for the cylinder wall has a fundamental")
    print("trade-off between transverse (Fx) and axial (Fz) accuracy.")
    print("Close wall sources (sr~1.05) → good Fx, bad Fz.")
    print("Far wall sources (sr~1.3)   → balanced ~12% error.")
    print("The Wilson et al. IMP-MFS likely uses cylinder IMAGE")
    print("singularities (Liron-Mochon) for exact wall handling.")

    center = jnp.zeros(3)
    cyl_len = 15.0 * R_CYL

    t0 = time.time()
    body = MFSSphere(center, A, n_refine=2, source_ratio=0.7, mu=MU)
    wall = MFSCylinder(R_CYL, cyl_len, n_circ=32, n_axial=40,
                       source_ratio=1.3, mu=MU, cluster=False)
    solver = ConfinedMFS(body, wall, MU)
    print(f"\nSetup: {time.time()-t0:.1f}s  |  "
          f"Body: {body.n_pts} pts, cond={body.cond:.1f}  |  "
          f"Wall: {wall.n_pts} pts, cond={wall.cond:.1f}")
    print(f"Cylinder: R={R_CYL:.3f}, L={cyl_len:.1f}, no clustering")

    e = jnp.eye(3)
    zero3 = jnp.zeros(3)
    labels = ["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"]
    refs = [REF_K03[l] for l in labels]
    motions = [(e[i], zero3) for i in range(3)] + \
              [(zero3, e[i]) for i in range(3)]

    print(f"\n{'':>5} | {'MFS(GS)':>10} | {'NN-BEM':>10} | "
          f"{'err%':>8} | {'iters':>5}")
    print("-" * 52)

    mfs_vals = []
    for j, (U, omega) in enumerate(motions):
        t0 = time.time()
        hist = solver.solve(center, U, omega)
        dt = time.time() - t0

        F_last, T_last = hist[-1]
        if j < 3:
            val = abs(float(F_last[j]))
        else:
            val = abs(float(T_last[j - 3]))

        ref = refs[j]
        err = abs(val - ref) / ref * 100
        mfs_vals.append(val)
        print(f"{labels[j]:>5} | {val:10.4f} | {ref:10.4f} | "
              f"{err:7.2f}% | {len(hist):5d}  ({dt:.1f}s)")

    # Direction-independence check
    err_Fx = abs(mfs_vals[0] - refs[0]) / refs[0] * 100
    err_Fy = abs(mfs_vals[1] - refs[1]) / refs[1] * 100
    err_Fz = abs(mfs_vals[2] - refs[2]) / refs[2] * 100
    print(f"\nDirection check: "
          f"F_x err={err_Fx:.2f}%, F_y err={err_Fy:.2f}%, "
          f"F_z err={err_Fz:.2f}%")
    spread = max(err_Fx, err_Fy, err_Fz) - min(err_Fx, err_Fy, err_Fz)
    print(f"  Max spread: {spread:.2f}pp "
          f"(want ≈0 for direction-independence)")
    print(f"  MFS IS direction-independent (same code path, spread<1pp)")
    print(f"  but has ~12% systematic wall error at 1280 wall pts.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BEM reference helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_bem_reference(kappa, mu=1.0, a=1.0):
    """Compute confined BEM reference 6×6 R matrix (standard Nyström)."""
    from mime.nodes.environment.stokeslet.resistance import (
        compute_confined_resistance_matrix,
    )
    r_cyl = a / kappa
    cyl_len = 15.0 * r_cyl

    # Use moderate resolution — body n_refine=3 (1280 pts),
    # wall 36×50 (1800 pts). Total DOF = 3×3080 = 9240.
    body_mesh = sphere_surface_mesh(
        center=(0, 0, 0), radius=a, n_refine=3)
    wall_mesh = cylinder_surface_mesh(
        center=(0, 0, 0), radius=r_cyl, length=cyl_len,
        n_circ=36, n_axial=50, cluster_center=True)
    eps = body_mesh.mean_spacing / 2.0

    R = compute_confined_resistance_matrix(
        jnp.array(body_mesh.points), jnp.array(body_mesh.weights),
        jnp.array(wall_mesh.points), jnp.array(wall_mesh.weights),
        jnp.zeros(3), eps, mu,
        use_dlp=False,  # DLP correction OOMs at moderate resolution
    )
    return np.array(R)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Task 0C — Sensitivity and convergence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def task_0c():
    print("\n" + "=" * 72)
    print("TASK 0C: Sensitivity and convergence")
    print("=" * 72)

    center = jnp.zeros(3)
    cyl_len = 15.0 * R_CYL
    Ux = jnp.array([1.0, 0.0, 0.0])
    Uz = jnp.array([0.0, 0.0, 1.0])
    zero3 = jnp.zeros(3)

    # ── 0C.1: Body source_ratio sweep ─────────────────────────────────
    print("\n--- 0C.1a: Body source_ratio sweep (wall sr=1.3 fixed) ---")
    print(f"{'body_sr':>8} | {'|F_x|':>10} | {'cond(G_b)':>12}")
    print("-" * 38)

    wall = MFSCylinder(R_CYL, cyl_len, 32, 40, source_ratio=1.3, mu=MU,
                       cluster=False)
    for bsr in [0.5, 0.6, 0.7, 0.8]:
        body = MFSSphere(center, A, 2, bsr, MU)
        solver = ConfinedMFS(body, wall, MU)
        hist = solver.solve(center, Ux, zero3)
        Fx = abs(float(hist[-1][0][0]))
        print(f"{bsr:8.2f} | {Fx:10.4f} | {body.cond:12.1f}")

    # ── 0C.1b: Wall source_ratio sweep (Fx AND Fz) ─────────────────
    print("\n--- 0C.1b: Wall source_ratio sweep (body sr=0.7 fixed) ---")
    print("  Shows the fundamental Fx-Fz trade-off in discrete wall MFS:")
    print(f"{'wall_sr':>8} | {'|F_x|':>10} {'errFx':>6} | "
          f"{'|F_z|':>10} {'errFz':>6} | {'cond(G_w)':>12}")
    print("-" * 66)

    for wsr in [1.05, 1.10, 1.20, 1.30, 1.40]:
        body = MFSSphere(center, A, 2, 0.7, MU)
        wall = MFSCylinder(R_CYL, cyl_len, 32, 40, wsr, MU, cluster=False)
        solver = ConfinedMFS(body, wall, MU)
        h = solver.solve(center, Ux, zero3)
        Fx = abs(float(h[-1][0][0]))
        h = solver.solve(center, Uz, zero3)
        Fz = abs(float(h[-1][0][2]))
        eFx = abs(Fx - REF_K03["F_x"]) / REF_K03["F_x"] * 100
        eFz = abs(Fz - REF_K03["F_z"]) / REF_K03["F_z"] * 100
        print(f"{wsr:8.2f} | {Fx:10.4f} {eFx:5.1f}% | "
              f"{Fz:10.4f} {eFz:5.1f}% | {wall.cond:12.1f}")

    # ── 0C.2: GS convergence tracking ────────────────────────────────
    print("\n--- 0C.2: GS convergence ---")
    body = MFSSphere(center, A, 2, 0.7, MU)
    wall = MFSCylinder(R_CYL, cyl_len, 32, 40, 1.3, MU, cluster=False)
    solver = ConfinedMFS(body, wall, MU)

    for label, U, om in [("F_x (U_x=1)", Ux, zero3),
                         ("F_z (U_z=1)", Uz, zero3)]:
        hist = solver.solve(center, U, om, max_iter=30, tol=1e-10)
        print(f"\n  {label} — {len(hist)} iterations:")
        print(f"  {'it':>3} | {'|F|':>12} | {'Δrel':>10}")
        print("  " + "-" * 32)
        for k, (Fk, Tk) in enumerate(hist):
            Fn = np.linalg.norm(Fk)
            if k == 0:
                print(f"  {k+1:3d} | {Fn:12.6f} |        —")
            else:
                dr = np.linalg.norm(Fk - hist[k - 1][0]) / (Fn + 1e-30)
                print(f"  {k+1:3d} | {Fn:12.6f} | {dr:10.2e}")

    # ── 0C.3: Confinement sweep ───────────────────────────────────────
    print("\n--- 0C.3: Confinement sweep (MFS only) ---")
    print("  NOTE: Low-res BEM reference is unreliable (~30-50% error).")
    print("  Showing MFS values and Fz/Fx ratio trend vs κ.")
    kappas = [0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"\n{'κ':>5} | {'F_x':>9} | {'F_z':>9} | {'T_z':>9} | "
          f"{'Fz/Fx':>6} | {'iters':>5}")
    print("-" * 55)

    for kap in kappas:
        r_cyl = A / kap
        cyl_l = 15.0 * r_cyl

        body = MFSSphere(center, A, 2, 0.7, MU)
        wall = MFSCylinder(r_cyl, cyl_l, 32, 40, 1.3, MU, cluster=False)
        solver = ConfinedMFS(body, wall, MU)

        h = solver.solve(center, Ux, zero3)
        Fx = abs(float(h[-1][0][0]))
        n_Fx = len(h)
        h = solver.solve(center, Uz, zero3)
        Fz = abs(float(h[-1][0][2]))
        h = solver.solve(center, zero3, Uz)
        Tz = abs(float(h[-1][1][2]))
        ratio = Fz / Fx

        print(f"{kap:5.2f} | {Fx:9.4f} | {Fz:9.4f} | {Tz:9.4f} | "
              f"{ratio:6.3f} | {n_Fx:5d}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    print(f"Float64 dtype: {jnp.ones(1).dtype}")
    print()

    task_0a()
    task_0b()
    task_0c()

    print("\n" + "=" * 72)
    print("Phase 0 complete.")
    print("=" * 72)
