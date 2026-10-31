"""The surface blobs scored on A's helix (FL-9, free space): a diagnostic in scratch code, not a register entry, nothing
in MIME changed. No measured speed enters. Predictions: ``predictions_blob_helix.json``, committed before any solve.

- **Kernels** (the blob spike's closed forms, ``S = A(r) I + B(r) x x^T``, velocity ``S g / (8 pi mu)``, ``A = a(r/eps) /
  eps``, ``B = b(r/eps) / eps^3``): Cortez's as MIME's own; ``surf3 = (1+x+2x^2) e^-x`` and ``surf5 = (1+x+4x^3/3)
  e^-x`` with their 14-term series in ``rho^2`` (``blob_helix_series.json``) below ``rho = 0.5`` (float64) or ``1.0``
  (float32), the closed form fed a safe ``rho^2`` (the double ``where``).
- **The quasi-uniform sampler** (``quasi_uniform``): a body's own sampler at a third of the spacing (about nine times
  the target count), thinned to the target count by farthest-point selection (Euclidean), then discrete Lloyd steps on
  that dense set (each sample moved to the dense point nearest its cell's area-weighted centroid). Each sample's weight
  is its cell's area on the dense set.
- **The solve** (``solve``): point forces at the samples, ``eps = fraction x mean spacing`` (MIME's rule), the matrix
  assembled in place by row blocks, Cholesky (LU where the factor is not finite), ``R = M^T A^-1 M``; at chosen levels
  the condition (largest eigenvalue by power iteration, least by inverse iteration with the factor: a lower bound).
- **Bodies**: FL-9 as the helix reference sets it up (``helix_ref.Body``); the unit sphere (MIME's Gauss-Legendre x
  midpoint sampling, the spike's); for the edge test, a prolate spheroid of FL-9's size (no edges; Chwang & Wu's axial
  drag and torque) and a capped circular cylinder of FL-9's radius and length (the caps' edges, no thread).
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
Save = Callable[[dict[str, Any]], None]
SQRT_PI = math.sqrt(math.pi)
#: Below this rho the series branch (the spike's switches: float64 0.5, float32 1.0).
RHO_SERIES = {"float64": 0.5, "float32": 1.0}
ROW_BLOCK = 256
COND_ITERATIONS = 60
DENSE_FACTOR = 3.0
LLOYD_STEPS = 6

# ---- kernels ---------------------------------------------------------------------------------------------------------


def _series() -> dict[str, dict[str, list[float]]]:
    raw = json.loads((HERE / "blob_helix_series.json").read_text())
    out = {}
    for name in ("surf3", "surf5"):
        out[name] = {part: [float(eval(c, {"pi": math.pi, "sqrt": math.sqrt})) for c in raw[name][part]]  # noqa: S307
                     for part in ("A", "B")}
    return out


SERIES = _series()


def _horner(coefficients: list[float], x: Any) -> Any:
    acc = coefficients[-1] + 0.0 * x
    for c in reversed(coefficients[:-1]):
        acc = acc * x + c
    return acc


def kernel_ab(name: str, rho2: Any, xp: Any, erf: Any, rho_series: float) -> tuple[Any, Any]:
    """``a(rho), b(rho)`` at ``eps = 1`` from ``rho^2`` (array module ``xp``, its ``erf``)."""
    if name == "cortez":
        t = 1.0 / xp.sqrt(rho2 + 1.0)
        t3 = t * t * t
        return (rho2 + 2.0) * t3, t3
    near = rho2 < rho_series * rho_series
    safe = xp.where(near, 1.0, rho2)
    rho = xp.sqrt(safe)
    e = xp.exp(-safe) / SQRT_PI
    f = erf(rho)
    if name == "surf3":
        a = f / rho + (6.0 - 4.0 * safe) * e
        b = f / (rho * safe) + (4.0 - 2.0 / safe) * e
    elif name == "surf5":
        a = f / rho + (26.0 / 3.0 - 12.0 * safe + 8.0 * safe * safe / 3.0) * e
        b = f / (rho * safe) + (28.0 / 3.0 - 8.0 * safe / 3.0 - 2.0 / safe) * e
    else:
        raise ValueError(name)
    sa, sb = _horner(SERIES[name]["A"], rho2), _horner(SERIES[name]["B"], rho2)
    return xp.where(near, sa, a), xp.where(near, sb, b)


# ---- the solve -------------------------------------------------------------------------------------------------------
_PROGRAMS: dict[Any, Any] = {}


def programs(name: str, dtype: str) -> dict[str, Any]:
    key = (name, dtype)
    if key in _PROGRAMS:
        return _PROGRAMS[key]
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.scipy.special import erf

    from mime.nodes.hydro.bem._solve import rigid_motions

    rho_series = RHO_SERIES[dtype]

    def assemble(points: Any, eps2: Any, viscosity: Any) -> Any:
        n = points.shape[0]
        rows = min(ROW_BLOCK, n)
        eps = jnp.sqrt(eps2)

        def body(k: Any, matrix: Any) -> Any:
            start = jnp.minimum(k * rows, n - rows)
            x = lax.dynamic_slice_in_dim(points, start, rows, 0)
            r = x[:, None, :] - points[None, :, :]
            r2 = jnp.sum(r * r, axis=-1)
            if name == "cortez":  # MIME's own form
                t = lax.rsqrt(r2 + eps2)
                big_a, big_b = (r2 + 2.0 * eps2) * t * t * t, t * t * t
            else:
                a, b = kernel_ab(name, r2 / eps2, jnp, erf, rho_series)
                big_a, big_b = a / eps, b / (eps * eps2)
            scale = 1.0 / (8.0 * math.pi * viscosity)
            eye = jnp.eye(3, dtype=points.dtype)
            blocks = (eye * big_a[..., None, None] + r[..., :, None] * r[..., None, :] * big_b[..., None, None]) * scale
            block = blocks.transpose(0, 2, 1, 3).reshape(3 * rows, 3 * n)
            return lax.dynamic_update_slice_in_dim(matrix, block, 3 * start, 0)

        count = -(-n // rows)
        return lax.fori_loop(0, count, body, jnp.zeros((3 * n, 3 * n), dtype=points.dtype))

    def factor(matrix: Any) -> Any:
        return lax.linalg.cholesky(matrix, symmetrize_input=False)

    def finish(lower: Any, points: Any, centre: Any) -> Any:
        motions = rigid_motions(points, centre)
        y = lax.linalg.triangular_solve(lower, motions, left_side=True, lower=True)
        return jnp.matmul(y.T, y, precision="highest")

    def lu_finish(matrix: Any, points: Any, centre: Any) -> Any:
        motions = rigid_motions(points, centre)
        found = jnp.matmul(motions.T, jnp.linalg.solve(matrix, motions), precision="highest")
        return (found + found.T) / 2.0

    def unit(v: Any) -> Any:
        return v / jnp.linalg.norm(v)

    def largest(matrix: Any, v0: Any) -> Any:
        def body(_: Any, carry: Any) -> Any:
            v, _q = carry
            w = jnp.matmul(matrix, v, precision="highest")
            return unit(w), jnp.dot(v, w)

        v, q_prev = lax.fori_loop(0, COND_ITERATIONS - 1, body, (unit(v0), jnp.zeros((), v0.dtype)))
        return jnp.dot(v, jnp.matmul(matrix, v, precision="highest")), q_prev

    def smallest(lower: Any, v0: Any) -> Any:
        def inverse_quotient(v: Any) -> Any:
            y = lax.linalg.triangular_solve(lower, v[:, None], left_side=True, lower=True)
            return y, jnp.sum(y * y)

        def body(_: Any, carry: Any) -> Any:
            v, _q = carry
            y, q = inverse_quotient(v)
            w = lax.linalg.triangular_solve(lower, y, left_side=True, lower=True, transpose_a=True)[:, 0]
            return unit(w), q

        v, q_prev = lax.fori_loop(0, COND_ITERATIONS - 1, body, (unit(v0), jnp.ones((), v0.dtype)))
        return 1.0 / inverse_quotient(v)[1], 1.0 / q_prev

    found = {"assemble": jax.jit(assemble), "factor": jax.jit(factor, donate_argnums=0), "finish": jax.jit(finish),
             "lu_finish": jax.jit(lu_finish, donate_argnums=0), "largest": jax.jit(largest),
             "smallest": jax.jit(smallest), "eigvalsh": jax.jit(jnp.linalg.eigvalsh)}
    _PROGRAMS[key] = found
    return found


def solve(name: str, points: np.ndarray, eps: float, viscosity: float, centre: np.ndarray, *,
          dtype: str = "float64", condition: bool = False, spectrum: bool = False) -> dict[str, Any]:
    """``R`` (float64 numpy) and the phases' seconds; with ``condition`` the power and inverse iterations' bound."""
    import jax
    import jax.numpy as jnp

    run = programs(name, dtype)
    t = getattr(jnp, dtype)
    pts, c = jnp.asarray(points, dtype=t), jnp.asarray(centre, dtype=t)
    eps2, mu = t(eps * eps), t(viscosity)
    out: dict[str, Any] = {"kernel": name, "dtype": dtype, "eps_m": eps, "samples": len(points)}
    began = time.perf_counter()
    matrix = jax.block_until_ready(run["assemble"](pts, eps2, mu))
    out["assemble_s"] = time.perf_counter() - began
    if spectrum:
        values = np.asarray(run["eigvalsh"](matrix), dtype=np.float64)
        out["eigen"] = {"least": float(values[0]), "largest": float(values[-1]),
                        "ratio": float(values[-1] / values[0]), "negative": int(np.sum(values < 0))}
    v0 = jnp.asarray(np.random.default_rng(11).standard_normal(3 * len(points)), dtype=t)
    if condition:
        top, top_prev = (float(x) for x in run["largest"](matrix, v0))
        out.update(largest=top, largest_change=abs(top - top_prev) / abs(top))
    start = time.perf_counter()
    lower = jax.block_until_ready(run["factor"](matrix))
    del matrix
    out["factor_s"] = time.perf_counter() - start
    finite = bool(jnp.all(jnp.isfinite(jnp.diagonal(lower))))
    out["cholesky_finite"] = finite
    if condition and finite:
        low, low_prev = (float(x) for x in run["smallest"](lower, v0))
        out.update(least=low, least_change=abs(low - low_prev) / abs(low), condition_at_least=out["largest"] / low)
    start = time.perf_counter()
    r = np.asarray(jax.block_until_ready(run["finish"](lower, pts, c)), dtype=np.float64)
    del lower
    out["finish_s"] = time.perf_counter() - start
    out["method"] = "cholesky"
    if not (finite and np.all(np.isfinite(r))):
        start = time.perf_counter()
        matrix = run["assemble"](pts, eps2, mu)
        r = np.asarray(jax.block_until_ready(run["lu_finish"](matrix, pts, c)), dtype=np.float64)
        out["method"] = "lu (cholesky not finite)"
        out["lu_s"] = time.perf_counter() - start
    out["seconds"] = time.perf_counter() - began
    out["resistance"] = r.tolist()
    out["finite"] = bool(np.all(np.isfinite(r)))
    if out["finite"]:
        out["asymmetry"] = float(np.max(np.abs(r - r.T)) / np.max(np.abs(r)))
    return out


# ---- the quasi-uniform sampler ---------------------------------------------------------------------------------------
_FPS: dict[int, Any] = {}


def farthest_points(dense: np.ndarray, n: int, start: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from jax import lax

    if n not in _FPS:
        def fps(points: Any, first: Any) -> Any:
            d0 = jnp.sum((points - points[first]) ** 2, axis=-1)
            chosen = jnp.zeros(n, dtype=jnp.int32).at[0].set(first)

            def body(i: Any, carry: Any) -> Any:
                sel, low = carry
                j = jnp.argmax(low).astype(jnp.int32)
                return sel.at[i].set(j), jnp.minimum(low, jnp.sum((points - points[j]) ** 2, axis=-1))

            return lax.fori_loop(1, n, body, (chosen, d0))[0]

        _FPS[n] = jax.jit(fps)
    return np.asarray(_FPS[n](jnp.asarray(dense), jnp.int32(start)))


def quasi_uniform(dense: np.ndarray, weights: np.ndarray, n: int, steps: int = LLOYD_STEPS) -> dict[str, Any]:
    """``n`` of the ``dense`` points: farthest-point selection, then ``steps`` discrete Lloyd steps. Returns the chosen
    indices, each sample's cell area and the Lloyd steps' record."""
    from scipy.spatial import cKDTree

    centroid = (weights[:, None] * dense).sum(0) / weights.sum()
    start = int(np.argmax(np.sum((dense - centroid) ** 2, axis=-1)))
    began = time.perf_counter()
    chosen = farthest_points(dense, n, start)
    fps_s = time.perf_counter() - began
    tree = cKDTree(dense)
    record = []
    for _ in range(steps):
        _, owner = cKDTree(dense[chosen]).query(dense, workers=-1)
        area = np.bincount(owner, weights=weights, minlength=n)
        safe = np.where(area > 0, area, 1.0)
        target = np.stack([np.bincount(owner, weights=weights * dense[:, i], minlength=n) for i in range(3)], 1) \
            / safe[:, None]
        _, moved = tree.query(target, workers=-1)
        moved = np.where(area > 0, moved, chosen)
        _, first = np.unique(moved, return_index=True)
        keep = np.zeros(n, dtype=bool)
        keep[first] = True
        moved = np.where(keep, moved, chosen)
        if len(np.unique(moved)) < n:
            record.append({"skipped": "a collision"})
            continue
        record.append({"moved": int(np.sum(moved != chosen))})
        chosen = moved
    _, owner = cKDTree(dense[chosen]).query(dense, workers=-1)
    area = np.bincount(owner, weights=weights, minlength=n)
    return {"chosen": chosen, "area": area, "lloyd": record, "fps_s": fps_s, "empty_cells": int(np.sum(area == 0))}


def nearest(points: np.ndarray, mean_h: float) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    d, _ = cKDTree(points).query(points, k=2)
    nn = d[:, 1] / mean_h
    return {"nn_over_h_min": float(nn.min()), "nn_over_h_p1": float(np.percentile(nn, 1)),
            "nn_over_h_p5": float(np.percentile(nn, 5)), "nn_over_h_median": float(np.median(nn)),
            "share_below_half_h": float(np.mean(nn < 0.5))}


# ---- bodies ------------------------------------------------------------------------------------------------------------
class Sampled:
    """A body's samples (placed), their weights, region labels and the mean spacing; ``dense`` keeps the dense set's
    labels for the area shares."""

    def __init__(self, points: np.ndarray, weights: np.ndarray, labels: dict[str, np.ndarray], centre: np.ndarray,
                 viscosity: float, mean_h: float) -> None:
        self.points, self.weights, self.labels = points, weights, labels
        self.centre, self.viscosity, self.mean_h = centre, viscosity, mean_h


def regions_report(sample_labels: dict[str, np.ndarray], dense_labels: dict[str, np.ndarray],
                   dense_weights: np.ndarray, sample_points: np.ndarray, mean_h: float) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    out = {}
    d, _ = cKDTree(sample_points).query(sample_points, k=2)
    nn = d[:, 1] / mean_h
    total = dense_weights.sum()
    for region, mask in sample_labels.items():
        area_share = float(dense_weights[dense_labels[region]].sum() / total)
        sample_share = float(np.mean(mask))
        out[region] = {"area_share": area_share, "sample_share": sample_share,
                       "ratio": sample_share / area_share if area_share else None,
                       "nn_over_h_median": float(np.median(nn[mask])) if mask.any() else None,
                       "share_below_half_h": float(np.mean(nn[mask] < 0.5)) if mask.any() else None}
    return out


class Helix:
    """FL-9 at one spacing as the helix reference sets it up, with region labels for its own samples and for a dense
    sampling of its own kind."""

    def __init__(self, spacing_mm: float) -> None:
        import helix_ref as hr

        self.body = hr.Body("helix", spacing_mm)
        settings = self.body.settings()
        self.radius = float(np.asarray(settings["radius"]))
        self.length = float(np.asarray(settings["length"]))
        self.modulation = float(np.asarray(settings["modulation"]))
        self.spacing_mm = spacing_mm

    def labelled(self, counts: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        import jax.numpy as jnp

        body = self.body
        constants = body.node.constants()
        kind = body.node._kind  # noqa: SLF001
        traced = body.node._traced  # noqa: SLF001
        parameters = {name: body.p[f"shape_{name}"] if name in traced else constants[f"shape_{name}"]
                      for name in body.settings()}
        turn = np.asarray(body.p["part_rotation"], dtype=np.float64).reshape(3, 3)
        centre = np.asarray(body.p["part_centre"], dtype=np.float64)
        pts, wts, patch_of = [], [], []
        for patch, count in zip(constants["patches"], counts, strict=True):
            s = kind.samples(parameters, patch, tuple(count))
            p = np.asarray(jnp.asarray(s.points), dtype=np.float64)
            pts.append(p)
            wts.append(np.asarray(jnp.asarray(s.weights), dtype=np.float64))
            patch_of += [patch] * len(p)
        local = np.concatenate(pts)
        weights = np.concatenate(wts)
        patch_of = np.asarray(patch_of)
        lateral = patch_of == "lateral"
        rho = np.hypot(local[:, 1], local[:, 2])
        s = (rho / self.radius - 1.0) / self.modulation
        half = self.length / 2.0
        labels = {"caps": ~lateral, "lateral": lateral, "crest_band": lateral & (s > 0.6),
                  "trough_band": lateral & (s < -0.6)}
        placed = body.place(local @ turn.T + centre)
        return placed, weights, {**labels, "_axial": local[:, 0], "_half": np.full(len(local), half)}

    def rim_labels(self, points: np.ndarray, labels: dict[str, np.ndarray], rim_points: np.ndarray,
                   mean_h: float) -> np.ndarray:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(rim_points).query(points, workers=-1)
        return d < mean_h

    def own(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        counts = [tuple(c) for c in self.body.node.constants()["counts"]]
        return self.labelled(counts)

    def dense(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        return self.labelled(self.body.counts_at(self.spacing_mm / DENSE_FACTOR))


def sphere_samples(polar: int, around: int) -> tuple[np.ndarray, np.ndarray]:
    """MIME's sphere sampling (the spike's ``mrs.sphere_samples``), unit radius."""
    cosine, weight = np.polynomial.legendre.leggauss(polar)
    angle = 2.0 * math.pi * (np.arange(around) + 0.5) / around
    sine = np.sqrt(1.0 - cosine ** 2)
    unit = np.stack([np.outer(sine, np.cos(angle)), np.outer(sine, np.sin(angle)),
                     np.outer(cosine, np.ones(around))], axis=-1).reshape(-1, 3)
    return unit, np.outer(weight, np.full(around, 2.0 * math.pi / around)).ravel()


def spheroid_dense(c: float, a: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """A prolate spheroid along x (semi-axes ``c`` along, ``a`` round): rings in the polar angle by Gauss-Legendre,
    each ring's count proportional to its radius."""
    rings = max(8, math.ceil(math.pi * (a + c) / 2.0 / h))
    t, w = np.polynomial.legendre.leggauss(rings)
    theta = (t + 1.0) * math.pi / 2.0
    w = w * math.pi / 2.0
    pts, wts = [], []
    for th, wt in zip(theta, w, strict=True):
        r = a * math.sin(th)
        around = max(6, math.ceil(2.0 * math.pi * r / h))
        phi = 2.0 * math.pi * (np.arange(around) + 0.5) / around
        element = a * math.sin(th) * math.sqrt((a * math.cos(th)) ** 2 + (c * math.sin(th)) ** 2)
        pts.append(np.stack([np.full(around, c * math.cos(th)), r * np.cos(phi), r * np.sin(phi)], -1))
        wts.append(np.full(around, wt * element * 2.0 * math.pi / around))
    return np.concatenate(pts), np.concatenate(wts)


def spheroid_exact(c: float, a: float, viscosity: float) -> dict[str, float]:
    """Chwang & Wu (the spike's ``s4b_spheroid``): axial drag and axial torque of a prolate spheroid."""
    e = math.sqrt(c * c - a * a) / c
    big_l = math.log((1 + e) / (1 - e))
    return {"drag": 16 * math.pi * viscosity * c * e ** 3 / ((1 + e * e) * big_l - 2 * e),
            "torque": 32 * math.pi * viscosity * c * a * a * e ** 3 / (3 * (2 * e - (1 - e * e) * big_l))}


def cylinder_dense(radius: float, length: float, h: float) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """A capped circular cylinder along x: the side by midpoints in x and round; each cap in rings by midpoints in the
    radius, each ring's count proportional to its radius."""
    along = max(4, math.ceil(length / h))
    around = max(8, math.ceil(2.0 * math.pi * radius / h))
    x = -length / 2.0 + length * (np.arange(along) + 0.5) / along
    phi = 2.0 * math.pi * (np.arange(around) + 0.5) / around
    xx, pp = np.meshgrid(x, phi, indexing="ij")
    side = np.stack([xx.ravel(), radius * np.cos(pp.ravel()), radius * np.sin(pp.ravel())], -1)
    side_w = np.full(len(side), length / along * 2.0 * math.pi * radius / around)
    rings = max(2, math.ceil(radius / h))
    caps, caps_w = [], []
    for end in (-1.0, 1.0):
        for k in range(rings):
            r = radius * (k + 0.5) / rings
            m = max(3, math.ceil(2.0 * math.pi * r / h))
            ang = 2.0 * math.pi * (np.arange(m) + 0.5) / m
            caps.append(np.stack([np.full(m, end * length / 2.0), r * np.cos(ang), r * np.sin(ang)], -1))
            caps_w.append(np.full(m, 2.0 * math.pi * r * (radius / rings) / m))
    pts = np.concatenate([side, *caps])
    wts = np.concatenate([side_w, *caps_w])
    cap = np.arange(len(pts)) >= len(side)
    return pts, wts, {"caps": cap, "lateral": ~cap}


# ---- readings ------------------------------------------------------------------------------------------------------------
def helix_reading(r: np.ndarray) -> dict[str, Any]:
    import helix_ref as hr

    if not np.all(np.isfinite(r)):
        return {"finite": False}
    out = hr.reading(r, "helix")
    out.pop("resistance_si", None)
    sym = (r + r.T) / 2.0
    out.update(R_xx=float(sym[0, 0]), R_xOx=float(sym[0, 3]), R_OxOx=float(sym[3, 3]))
    return out


def axis_reading(r: np.ndarray) -> dict[str, Any]:
    if not np.all(np.isfinite(r)):
        return {"finite": False}
    sym = (r + r.T) / 2.0
    return {"F_x": float(sym[0, 0]), "F_y": float(sym[1, 1]), "F_z": float(sym[2, 2]), "T_x": float(sym[3, 3]),
            "T_y": float(sym[4, 4]), "T_z": float(sym[5, 5])}
