"""The builder's check, local and light: equal to ``tests/tube_fourier_bessel.image`` at drawn points, the planted
defect fails that every run (and the builder with the defect planted reproduces the defective series), the identities
on the builder's own output, Blake's image and its transform, the remainder series, the cubic read against
``mime.core.tables``, and the cofactor solve against LAPACK.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 nice -n 10 .venv/bin/python scripts/tube_table/check.py [out.json]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

spec = importlib.util.spec_from_file_location("fb", HERE.parents[1] / "tests" / "tube_fourier_bessel.py")
assert spec is not None and spec.loader is not None
fb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fb)

SMALL = dict(modes=15, wavenumbers=80, angles=64)
#: Fewer wavenumbers: the smallest node stays where the series' own basis is well conditioned.
COARSE = dict(modes=15, wavenumbers=20, angles=64)
RUNS = 5
#: Rounding: the two compute the same sums in a different order; a planted error must exceed this by orders.
EQUAL = 1e-9
DEFECTIVE = ((1.0, 1.0), (1.0, 1.0), (-1.0, 1.0))


def series_system(kk, nn, r1, r2, p, sn):
    """The series' own system (``tests/tube_fourier_bessel``'s ``system``, divided by its ``i_wall``)."""
    dw, q = nn / kk + r1, (kk * kk + nn * nn) / kk
    system = np.zeros((*r1.shape, 3, 3))
    system[..., 0, 0], system[..., 0, 1], system[..., 0, 2] = sn * nn, p * kk * kk * dw, p * q
    system[..., 1, 0], system[..., 1, 1], system[..., 1, 2] = -kk * dw, -p * sn * nn * kk, -p * sn * nn * dw
    system[..., 2, 1], system[..., 2, 2] = -kk * kk, -(2.0 + kk * dw)
    return system


def series_terms(xp, rho, side_k, coefficients, log_wall, log_target, ratio1, ratio2, top):
    """The series' own basis at the target, by the same ratios."""
    k = side_k[None, None, :]
    n = xp.arange(top + 1, dtype=xp.float64)[:, None, None]
    rho3 = rho[None, :, None]
    on_axis = rho3 <= 0.0
    safe_rho = xp.where(on_axis, 1.0, rho3)
    x = k * safe_rho
    lam = xp.exp(log_target - log_wall[:, None, :])
    d_lam = (n / x + ratio1) * lam
    half = 0.5 * xp.exp(-log_wall[1])[None, None, :]
    first = n == 1.0
    a1 = xp.where(on_axis, xp.where(first, k * half, 0.0), n * lam / safe_rho)
    a2 = xp.where(on_axis, xp.where(first, half, 0.0), d_lam)
    a3 = xp.where(on_axis, xp.where(first, half, 0.0), (x * x + n * n) / x * lam)
    x_d_lam = xp.where(on_axis, 0.0, x * d_lam)
    out = [[None] * 3 for _ in range(3)]
    for col, (p, sn) in enumerate(build.COLUMNS):
        c = [xp.transpose(coefficients[:, :, b, col])[:, None, :] for b in range(3)]
        out[0][col] = sn * a1 * c[0] + p * k * k * a2 * c[1] + p * a3 * c[2]
        out[1][col] = -k * a2 * c[0] - p * sn * k * a1 * c[1] - p * sn * n * a2 * c[2]
        out[2][col] = -k * k * lam * c[1] - (2.0 * lam + x_d_lam) * c[2]
    return [out[row][col] for row in range(3) for col in range(3)]


def by_inverse(matrix, rhs):
    """As the series solves: each matrix inverted by LAPACK, a singular one skipped."""
    out = np.zeros(rhs.shape)
    for index in np.ndindex(*matrix.shape[:-2]):
        try:
            out[index] = np.linalg.inv(matrix[index]) @ rhs[index]
        except np.linalg.LinAlgError:
            pass
    return out


def with_series_basis(call):
    saved = build.wall_system, build.target_terms, build.solve3
    build.wall_system, build.target_terms = series_system, series_terms
    defaults = build.wall_side.__defaults__
    build.wall_side.__defaults__ = (None, by_inverse)
    try:
        return call()
    finally:
        build.wall_system, build.target_terms, build.solve3 = saved
        build.wall_side.__defaults__ = defaults


def draw(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    radius = np.round(rng.uniform(0.1, 0.85, 3) / 1e-6) * 1e-6
    angle = rng.uniform(-np.pi, np.pi, 3)
    sources = np.stack([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(-0.3, 0.3, 3)], axis=1)
    rt, at = np.sqrt(rng.uniform(0.0, 0.9 ** 2, 6)), rng.uniform(-np.pi, np.pi, 6)
    targets = np.stack([rt * np.cos(at), rt * np.sin(at), rng.uniform(-1.0, 1.0, 6)], axis=1)
    targets[0] = [0.0, 0.0, rng.uniform(-0.5, 0.5)]  # exactly on the axis
    return targets, sources


def miss(a: np.ndarray, b: np.ndarray) -> float:
    """The largest entry miss over each block's largest entry."""
    return float(np.max(np.max(np.abs(a - b), axis=(-2, -1)) / np.max(np.abs(b), axis=(-2, -1))))


def against_oracle(counts: dict) -> dict:
    found, planted, defect_reproduced, jitter_oracle, jitter_ours, replica = [], [], [], [], [], []
    for seed in range(RUNS):
        targets, sources = draw(seed)
        truncation = build.series_truncation(**counts, largest_source_radius=float(np.max(np.hypot(*sources[:, :2].T))))
        ours = build.image_points(targets, sources, truncation)
        oracle = fb.image(targets, sources, 1.0, 1.0, **counts)
        defective = fb.image(targets, sources, 1.0, 1.0, **counts, as_built=True)
        found.append(miss(ours, oracle))
        replica.append(miss(with_series_basis(lambda: build.image_points(targets, sources, truncation)), oracle))
        planted.append(miss(ours, defective))
        # each one's own rounding floor: the change under a shift of the targets by 1e-12 along the axis
        moved = targets + np.array([0.0, 0.0, 1e-12])
        jitter_oracle.append(miss(fb.image(moved, sources, 1.0, 1.0, **counts), oracle))
        jitter_ours.append(miss(build.image_points(moved, sources, truncation), ours))
        build.COLUMNS = DEFECTIVE
        try:
            defect_reproduced.append(miss(build.image_points(targets, sources, truncation), defective))
        finally:
            build.COLUMNS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0))
    return {"builder_vs_series": found, "builder_in_the_series_basis_vs_series": replica, "builder_vs_defective_series": planted,
            "defect_planted_in_builder_vs_defective_series": defect_reproduced,
            "series_own_jitter_under_1e-12_shift": jitter_oracle, "builder_own_jitter_under_1e-12_shift": jitter_ours,
            "equal_runs": sum(v <= EQUAL for v in found), "planted_fails_runs": sum(v > 1e3 * EQUAL for v in planted),
            "runs": RUNS}


def identities(defect: bool) -> dict:
    """Divergence (4th-order stencil, step 0.02, over the gradient's largest entry), reciprocity, the axis's symmetry
    and its limit, on the builder's output at the small truncation."""
    if defect:
        build.COLUMNS = DEFECTIVE
    try:
        tr = build.series_truncation(**SMALL, largest_source_radius=0.6)
        sources = np.array([[0.5, 0.0, 0.0], [0.2, -0.35, 0.3]])
        points = np.array([[0.3, 0.2, 0.1], [-0.4, 0.1, -0.2], [0.1, -0.6, 0.4]])
        step, worst_div = 0.02, 0.0
        for point in points:
            grad = np.zeros((3, 3, len(sources), 3))  # d/dx_d of u_i: [d, i, source, force]
            for d in range(3):
                e = np.zeros(3)
                e[d] = step
                stencil = [build.image_points((point + c * e)[None], sources, tr)[0] for c in (-2, -1, 1, 2)]
                grad[d] = np.transpose((stencil[0] - 8 * stencil[1] + 8 * stencil[2] - stencil[3]) / (12 * step),
                                       (1, 0, 2))
            div = sum(grad[d, d] for d in range(3))
            worst_div = max(worst_div, float(np.max(np.abs(div) / np.max(np.abs(grad), axis=(0, 1)))))
        pts = np.concatenate([sources, points])
        g = build.image_points(pts, pts, tr)
        recip = float(np.max(np.abs(g - np.transpose(g, (1, 0, 3, 2)))) / np.max(np.abs(g)))
        s = fb.image(pts, pts, 1.0, 1.0, **SMALL, as_built=defect)
        series_recip = float(np.max(np.abs(s - np.transpose(s, (1, 0, 3, 2)))) / np.max(np.abs(s)))
        axis = np.array([[0.0, 0.0, 0.0]])
        near = np.array([[1e-6, 0.0, 0.0]])
        at, off = build.image_points(axis, axis, tr)[0, 0], build.image_points(near, near, tr)[0, 0]
        return {"divergence": worst_div, "reciprocity": recip, "series_reciprocity_same_points": series_recip,
                "axis_xx_yy": [float(at[0, 0]), float(at[1, 1])],
                "axis_symmetry": abs(float(at[0, 0] - at[1, 1])) / abs(float(at[0, 0])),
                "axis_limit": float(np.max(np.abs(at - off)) / np.max(np.abs(off)))}
    finally:
        build.COLUMNS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0))


def blake_checks() -> dict:
    rho_s = 0.9
    rng = np.random.default_rng(7)
    y, zz = rng.uniform(-0.5, 0.5, 40), rng.uniform(-0.5, 0.5, 40)
    out = {}
    for name, sign in (("nearest", 1.0), ("control_flipped", -1.0)):
        # in space, on the plane x = 1: the free Stokeslet plus the image vanishes
        worst = 0.0
        for yy, z in zip(y, zz, strict=True):
            rho, phi = float(np.hypot(1.0, yy)), float(np.arctan2(yy, 1.0))
            b = build.blake_cylindrical(rho_s, np.array([rho]), np.array([phi]), np.array([z]), (0.0, sign))
            f = build.free_cylindrical(rho_s, np.array([rho]), np.array([phi]), np.array([z]))
            worst = max(worst, float(np.max(np.abs(b + f)) / np.max(np.abs(f))))
        out[f"plane_no_slip_{name}"] = worst
        # its transform along z, on the plane: the free Stokeslet's plus the image's vanishes at each k
        h, normal, mirror = build.plane_geometry(rho_s, (0.0, sign))
        k = np.array([[0.3], [3.0], [30.0]])
        fx, fy = np.full((1, 40), 1.0 - rho_s), y[None, :]
        mx, my = np.full((1, 40), 1.0 - mirror[0]), y[None, :]
        free = build.stokeslet_kernels((fx, fy))
        image = build.blake_kernels((mx, my), h, normal)
        worst = 0.0
        for j in range(3):
            for c in range(3):
                fk = free[j][c].transformed(k, np.hypot(fx, fy))
                bk = sign * image[j][c].transformed(k, np.hypot(mx, my))
                worst = max(worst, float(np.max(np.abs(fk + bk)) / np.max(np.abs(fk) + 1e-300)))
        out[f"plane_no_slip_transformed_{name}"] = worst
    # the kernels' transform of the free Stokeslet against the series' closed forms on the wall
    theta = np.linspace(0, 2 * np.pi, 33)[None, :]
    k = np.array([[0.5], [5.0]])
    dx, dy = np.cos(theta) - rho_s, np.sin(theta)
    wall = np.hypot(dx, dy)
    k0, k1 = build.special.k0(k * wall), build.special.k1(k * wall)
    closed = {(0, 0): 2 * k0 + 2 * k / wall * dx ** 2 * k1, (0, 2): 2 * k * dx * k0, (2, 2): 4 * k0 - 2 * k * wall * k1}
    free = build.stokeslet_kernels((dx, dy))
    out["free_transform_vs_closed_form"] = max(
        float(np.max(np.abs(free[j][c].transformed(k, wall) - v)) / np.max(np.abs(v))) for (j, c), v in closed.items())
    # Blake's image is divergence-free inside the tube
    p = np.array([0.6, 0.2, 0.15])
    step = 1e-3
    div = 0.0
    scale = 0.0
    for d in range(3):
        e = np.zeros(3)
        e[d] = step
        vals = []
        for c in (-2, -1, 1, 2):
            q = p + c * e
            rho, phi = float(np.hypot(q[0], q[1])), float(np.arctan2(q[1], q[0]))
            cyl = build.blake_cylindrical(rho_s, np.array([rho]), np.array([phi]), np.array([q[2]]), (0.0, 1.0))[0, 0, 0]
            vals.append(build._turn(phi).T @ cyl)
        grad = (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * step)
        div = div + grad[d]
        scale = max(scale, float(np.max(np.abs(grad))))
    out["blake_divergence"] = float(np.max(np.abs(div))) / scale
    return out


def remainder_check() -> dict:
    """The remainder series plus the plane image against the image's series, source at 0.9, at two truncations: the
    two must close as both converge."""
    rho, phi, z = np.array([0.5, 0.85, 0.9]), np.array([0.0, 0.1, 1.0]), np.array([0.0, 0.13, 1.1])
    out = {}
    for label, (m, w, r) in (("coarse", (6.0, 30.0, 12.0)), ("fine", (12.0, 60.0, 24.0))):
        tr = build.scaled_truncation(0.9, 0.9, modes=m, wavenumbers=w, reach=r)
        full = build.image_grid(0.9, rho, phi, z, tr)
        rest = build.image_grid(0.9, rho, phi, z, tr, plane=build.NEAREST)
        plane = build.blake_cylindrical(0.9, rho, phi, z, build.NEAREST)
        out[f"remainder_plus_plane_vs_image_{label}"] = miss(rest + plane, full)
        out[f"control_remainder_alone_vs_image_{label}"] = miss(rest, full)
    return out


def conditioning_check() -> dict:
    """Each basis's solve, cofactors against LAPACK, at the series' truncation (15/80/64, source at 0.5): where the
    two solvers disagree the solve is rounding-limited."""
    tr = build.series_truncation(**SMALL, largest_source_radius=0.5)
    k, w = build.wavenumber_nodes(tr)
    lapack = lambda a, b: np.linalg.solve(a, b[..., None])[..., 0]  # noqa: E731
    out = {}
    for label, system in (("recombined", build.wall_system), ("series_basis", series_system)):
        saved = build.wall_system
        build.wall_system = system
        try:
            ours = build.wall_side(0.5, k, w, tr)
            ref = build.wall_side(0.5, k, w, tr, solver=lapack)
        finally:
            build.wall_system = saved
        # compare the image each gives on the wall's nearest point instead of the coefficients (bases differ)
        out[label] = float(np.max(np.abs(ours.coefficients - ref.coefficients))
                           / np.max(np.abs(ref.coefficients)))
    return out


def read_check() -> dict:
    """The numpy cubic read equals ``mime.core.tables``' read of the tube's table along each axis."""
    from mime.core import jnp
    from mime.core.tables import evaluate
    from mime.nodes.hydro.mrs._tube import TABLE, unit_tube_table

    rho = np.array([0.0, 0.3, 0.55, 0.7, 0.8, 0.86, 0.9])
    angle = np.array([0.0, 0.05, 0.15, 0.4, 0.9, 1.7, 2.5, np.pi])
    along = np.array([0.0, 0.04, 0.12, 0.3, 0.8, 1.6, 3.0, 5.0])
    rt, rs, a, zz = np.meshgrid(rho, rho, angle, along, indexing="ij")
    data = np.empty((*rt.shape, 3, 3))
    for i in range(3):
        for j in range(3):
            fa = np.sin if build.SIN_IN_ANGLE[i, j] else np.cos
            fz = np.sin if build.SIN_ALONG[i, j] else np.cos
            data[..., i, j] = (1 + i + 2 * j + rt * rs) * fa((1 + i) * a) * fz(0.7 * (1 + j) * zz) / (1.2 - rt * rs)
    table = unit_tube_table(1.0, rho, angle, along, data, scheme="cubic")
    rng = np.random.default_rng(3)
    worst = 0.0
    qa, qz, qr = rng.uniform(0, np.pi, 50), rng.uniform(0, 5.0, 50), rng.uniform(0, 0.9, 50)
    it, js, ka, kz = 4, 5, 3, 2
    for axis in range(3):
        if axis == 0:
            ref = np.stack([build.hermite_read(rho, data[:, js, ka, kz, i, j], qr) for i in range(3) for j in range(3)],
                           -1)
            read = evaluate(table, jnp.asarray(qr), jnp.full(50, rho[js]), jnp.full(50, angle[ka]),
                            jnp.full(50, along[kz]), owner=TABLE).value
        elif axis == 1:
            ref = []
            for i in range(3):
                for j in range(3):
                    x, f = build.mirrored(angle, data[it, js, :, kz, i, j], -1.0 if build.SIN_IN_ANGLE[i, j] else 1.0,
                                          wraps=True)
                    ref.append(build.hermite_read(x, f, qa))
            ref = np.stack(ref, -1)
            read = evaluate(table, jnp.full(50, rho[it]), jnp.full(50, rho[js]), jnp.asarray(qa),
                            jnp.full(50, along[kz]), owner=TABLE).value
        else:
            ref = []
            for i in range(3):
                for j in range(3):
                    x, f = build.mirrored(along, data[it, js, ka, :, i, j], -1.0 if build.SIN_ALONG[i, j] else 1.0,
                                          wraps=False)
                    ref.append(build.hermite_read(x, f, qz))
            ref = np.stack(ref, -1)
            read = evaluate(table, jnp.full(50, rho[it]), jnp.full(50, rho[js]), jnp.full(50, angle[ka]),
                            jnp.asarray(qz), owner=TABLE).value
        read = np.asarray(read).reshape(50, 9)
        worst = max(worst, float(np.max(np.abs(read - ref)) / np.max(np.abs(ref))))
    return {"numpy_read_vs_table_read": worst}


def solve_check() -> dict:
    """The cofactor solve against LAPACK's on the hardest system: a source at 0.997, many modes and wavenumbers."""
    tr = build.scaled_truncation(0.997, 0.997, modes=3.0, wavenumbers=4.0, reach=12.0)
    k, w = build.wavenumber_nodes(tr)
    keep = k > 1e-3  # LAPACK refuses the systems singular to rounding at the smallest wavenumbers
    k, w = k[keep], w[keep]
    pick = slice(None, None, max(1, len(k) // 40))
    ours = build.wall_side(0.997, k[pick], w[pick], tr)
    lapack = build.wall_side(0.997, k[pick], w[pick], tr, solver=lambda a, b: np.linalg.solve(a, b[..., None])[..., 0])
    scale = np.max(np.abs(lapack.coefficients), axis=(1, 2), keepdims=True)
    return {"cofactor_vs_lapack": float(np.max(np.abs(ours.coefficients - lapack.coefficients) / scale)),
            "modes": tr.modes, "k_max": tr.k_max}


def main() -> None:
    report = {"oracle_15_80_64": against_oracle(SMALL), "oracle_15_20_64": against_oracle(COARSE),
              "conditioning": conditioning_check(), "identities": identities(False), "identities_defect_planted": identities(True),
              "blake": blake_checks(), "remainder": remainder_check(), "read": read_check(), "solve": solve_check()}
    text = json.dumps(report, indent=1)
    print(text)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)


if __name__ == "__main__":
    main()
