"""A converged free-space reference for A's body (FL-9 as the frozen model builds it): a diagnostic, not a register entry.
No measured speed enters. Predictions: ``predictions_helix_ref.json``, committed before any solve.

The body is A's free-space arm of the error budget (``bem_budget_grid.a_level``'s ``free``): ``a0f_solves.Floored`` at a
spacing, phase 0's floored pose, the wall left out. The quantity is its axial speed (``operating_point.speeds``) and
its 6 x 6 resistance. The controls are the error budget's sphere (0.3 of A's tube radius) in free space.

- **Route (a), the node as it is** (``nystrom_level``): the node's discretisation (point forces at its samples, ``eps =
  blob x mean spacing``) at a spacing, every blob of ``BLOBS``, and at the same level each absolute ``eps`` of
  ``EPS_M`` (the fixed-``eps`` lines). The matrix is assembled in place by row blocks and factorised by Cholesky (the
  regularised Stokeslet's matrix is symmetric positive definite), so that one matrix fits where the node's own path,
  with its intermediates and its LU copy, does not; ``R = Y^T Y``, ``Y = L^-1 M``. The node's path is run beside it at
  the levels where it fits, as the control (``node_path``).
- **Route (b), quadrature separated from regularisation** (``nn_level``): Gallagher & Smith's nearest-neighbour
  discretisation. A fixed force set (the node's samples at a spacing, collocation there, one traction per sample's
  cell) and a much finer quadrature (the body's own sampler at a finer spacing, its axial count raised so that the
  quadrature is about isotropic on the screw's flanks): each quadrature point belongs to its nearest force point, and
  ``A_mn = sum over q in cell n of S_eps(x_m, y_q) w_q``. The wrench of the tractions is taken on the quadrature
  (``M_q^T diag(a) f``, the cells' areas and centroids). At a fixed ``eps`` the quadrature is refined at a fixed force
  set, then the force set refined at the finest quadrature.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import bem_budget as bb  # noqa: E402
import operating_point as op  # noqa: E402

Save = Callable[[dict[str, Any]], None]
#: Route (a)'s blobs (the error budget's five; the brief names 0.35 to 1.0).
BLOBS = (0.25, 0.35, 0.5, 0.7, 1.0)
#: The frozen point: the mean spacing at 0.16 mm nominal (the error budget's record, recomputed in every level) and its
#: blob. Route (b)'s eps: the frozen spacing's five blobs' eps, then two larger, where the node's own fixed-eps line
#: converges within the levels reached (the cross-check of route (b)).
FROZEN_SPACING_MM, FROZEN_BLOB = 0.16, 0.5
FROZEN_MEAN_H_M = 1.4571595021224538e-4
EPS_M = tuple(b * FROZEN_MEAN_H_M for b in BLOBS) + (2.0e-4, 2.9e-4)
#: Route (a)'s spacings (nominal, mm): the error budget's four, then finer, as far as one matrix fits.
A_SPACINGS_MM = (0.2, 0.16, 0.128, 0.1024, 0.09, 0.08, 0.072, 0.069, 0.066)
#: Route (b): the force sets (nominal, mm) and the quadrature's spacings (nominal, mm), coarse to fine; the axial count
#: of the lateral patch is raised by ``ALONG_FACTOR`` (the mid-flank's axial spacing is about 2.3 times the round's).
NN_FORCE_MM = (0.2, 0.16, 0.128, 0.1024, 0.09, 0.08)
NN_QUAD_MM = (0.03, 0.021, 0.015, 0.0106)
#: The force sets finer than 0.1024 mm take the finest quadrature alone (their quadrature change is the 0.1024 set's).
NN_FINEST_ONLY_BELOW_MM = 0.1
ALONG_FACTOR = 2.5
#: The sphere controls: route (a)'s spacings (nominal, mm) and route (b)'s force sets and quadrature.
SPHERE_A_MM = (0.2, 0.16, 0.128, 0.1024, 0.08, 0.064, 0.05, 0.04, 0.032)
SPHERE_NN_FORCE_MM = (0.2, 0.16, 0.128, 0.1024)
SPHERE_NN_QUAD_MM = (0.03, 0.021, 0.015, 0.0106)
#: Device memory a level may plan for (bytes): the device's own limit where it reports one (else the A100-80GB's share
#: at a fraction of 0.95), less a margin for the row blocks and the factor's workspace.
MARGIN_BYTES = 3.0e9
ROW_BLOCK = 256
#: Quadrature slots (cells x padded points) per block of route (b)'s assembly.
NN_BLOCK = 16384


def device_budget() -> float:
    limit = device_memory().get("bytes_limit")
    return (float(limit) if limit else 0.95 * 80 * 1024 ** 3) - MARGIN_BYTES


def device_memory() -> dict[str, int]:
    import jax

    stats = jax.devices()[0].memory_stats() or {}
    return {key: int(stats[key]) for key in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit") if key in stats}


# ---- the bodies --------------------------------------------------------------------------------------------------------
class Body:
    """A built experiment's drag node, its samples at the pose its arm uses (A: phase 0's floored pose; the sphere: its
    initial pose), and the same body sampled at any other counts by its own kind."""

    def __init__(self, which: str, spacing_mm: float) -> None:
        import jax.numpy as jnp

        from mime.core import quaternion

        self.which, self.spacing_mm = which, spacing_mm
        if which == "helix":
            import a0f_solves

            rung = a0f_solves.Floored(spacing_mm, 0.5)
            self.built, self.node_name = rung.built, op.DRAG_NODE
            self.pose = np.asarray(rung.pose(op.PHASES[0]), dtype=np.float64)
        elif which == "sphere":
            self.built, self.node_name = bb.sphere_experiment(spacing_mm, False), "drag:ball"
            self.pose = np.asarray(bb.sphere_pose(self.built), dtype=np.float64)
        else:
            raise ValueError(which)
        self.node = self.built.graph.get_node(self.node_name)
        self.p = self.node._step_constants(op.leaves_of(self.built)[self.node_name])  # noqa: SLF001
        self.viscosity = float(np.asarray(self.p["viscosity"]))
        self.rotation = np.asarray(quaternion.rotation_matrix(jnp.asarray(self.pose[3:])), dtype=np.float64)
        self.centre = self.pose[:3].copy()
        held = self.node.samples(self.p)
        self.points = self.place(np.asarray(held.points, dtype=np.float64))
        self.weights = np.asarray(held.weights, dtype=np.float64)
        self.mean_h = math.sqrt(self.weights.sum() / len(self.weights))

    def place(self, body_points: np.ndarray) -> np.ndarray:
        return self.centre + body_points @ self.rotation.T

    def settings(self) -> dict[str, Any]:
        return {key.removeprefix("shape_"): value for key, value in self.node.constants().items()
                if key.startswith("shape_")}

    def counts_at(self, spacing_mm: float, along_factor: float = 1.0) -> list[tuple[int, ...]]:
        kind, settings = self.node._kind, self.settings()  # noqa: SLF001 - the node's own kind
        found = []
        for patch in self.node.constants()["patches"]:
            counts = tuple(kind.sample_counts(settings, spacing_mm * 1e-3, patch))
            if self.which == "helix" and patch == "lateral":
                counts = (counts[0], int(math.ceil(along_factor * counts[1])))
            found.append(counts)
        return found

    def resampled(self, counts: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
        """The body sampled by its own kind at ``counts`` (one per patch), placed as the node places its samples."""
        import jax.numpy as jnp

        constants = self.node.constants()
        kind = self.node._kind  # noqa: SLF001
        traced = self.node._traced  # noqa: SLF001
        parameters = {name: self.p[f"shape_{name}"] if name in traced else constants[f"shape_{name}"]
                      for name in self.settings()}
        found = [kind.samples(parameters, patch, tuple(count)) for patch, count in zip(constants["patches"], counts,
                                                                                       strict=True)]
        turn = np.asarray(self.p["part_rotation"], dtype=np.float64).reshape(3, 3)
        centre = np.asarray(self.p["part_centre"], dtype=np.float64)
        points = np.concatenate([np.asarray(jnp.asarray(s.points), dtype=np.float64) for s in found])
        weights = np.concatenate([np.asarray(jnp.asarray(s.weights), dtype=np.float64) for s in found])
        return self.place(points @ turn.T + centre), weights

    def describe(self) -> dict[str, Any]:
        return {"which": self.which, "spacing_mm": self.spacing_mm, "samples": len(self.weights),
                "mean_spacing_m": self.mean_h, "area_m2": float(self.weights.sum()),
                "counts": [list(c) for c in self.node.constants()["counts"]], "pose": self.pose.tolist()}


def reading(r: np.ndarray, which: str, **extra: Any) -> dict[str, Any]:
    finite = bool(np.all(np.isfinite(r)))
    out: dict[str, Any] = {"resistance_si": r.tolist(), "finite": finite, **extra}
    if finite:
        sym = (r + r.T) / 2.0
        out["asymmetry"] = float(np.max(np.abs(r - r.T)) / np.max(np.abs(r)))
        if which == "helix":
            out["axial_mm_s"] = 1e3 * float(op.speeds(sym)["axial"])
            out["lateral_free_mm_s"] = 1e3 * float(op.speeds(sym)["lateral_free"])
            out["axial_raw_mm_s"] = 1e3 * float(-r[0, 3] * 2 * math.pi * op.FREQUENCY_HZ / r[0, 0])
        else:
            a = bb.SPHERE_RATIO * bb.TUBE_RADIUS_M
            out["drag_over_stokes"] = float(np.trace(sym[:3, :3]) / 3.0 / (6 * math.pi * bb.VISCOSITY * a))
            out["drag_zz_over_stokes"] = float(sym[2, 2] / (6 * math.pi * bb.VISCOSITY * a))
    return out


# ---- route (a): the node's discretisation, the matrix in place ------------------------------------------------------
def _programs() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from jax import lax

    from mime.nodes.hydro.bem._solve import PRECISION, rigid_motions

    def assemble(points: Any, eps2: Any, viscosity: Any) -> Any:
        n = points.shape[0]
        rows = min(ROW_BLOCK, n)

        def body(k: Any, matrix: Any) -> Any:
            start = jnp.minimum(k * rows, n - rows)
            x = lax.dynamic_slice_in_dim(points, start, rows, 0)
            r = x[:, None, :] - points[None, :, :]
            r2 = jnp.sum(r * r, axis=-1)
            t = lax.rsqrt(r2 + eps2)
            scale = t * t * t / (8.0 * math.pi * viscosity)
            eye = jnp.eye(3, dtype=points.dtype)
            blocks = (eye * (r2 + 2.0 * eps2)[..., None, None] + r[..., :, None] * r[..., None, :]) \
                * scale[..., None, None]
            block = blocks.transpose(0, 2, 1, 3).reshape(3 * rows, 3 * n)
            return lax.dynamic_update_slice_in_dim(matrix, block, 3 * start, 0)

        count = -(-n // rows)
        return lax.fori_loop(0, count, body, jnp.zeros((3 * n, 3 * n), dtype=points.dtype))

    def factor(matrix: Any) -> Any:
        return lax.linalg.cholesky(matrix, symmetrize_input=False)

    def finish(lower: Any, points: Any, centre: Any) -> Any:
        motions = rigid_motions(points, centre)
        y = lax.linalg.triangular_solve(lower, motions, left_side=True, lower=True)
        return jnp.matmul(y.T, y, precision=PRECISION)

    def lu_finish(matrix: Any, points: Any, centre: Any) -> Any:
        motions = rigid_motions(points, centre)
        found = jnp.matmul(motions.T, jnp.linalg.solve(matrix, motions), precision=PRECISION)
        return (found + found.T) / 2.0

    return {"assemble": jax.jit(assemble), "factor": jax.jit(factor, donate_argnums=0), "finish": jax.jit(finish),
            "lu_finish": jax.jit(lu_finish, donate_argnums=0)}


_PROGRAMS: dict[str, Any] = {}


def programs() -> dict[str, Any]:
    if not _PROGRAMS:
        _PROGRAMS.update(_programs())
        _PROGRAMS.update(_nn_programs())
    return _PROGRAMS


def solve_in_place(points: np.ndarray, eps2: float, viscosity: float, centre: np.ndarray) -> tuple[np.ndarray, str]:
    """``R = M^T A^-1 M`` with ``A`` assembled in place and factorised by Cholesky (donated); a non-finite factor falls
    back to LU where two matrices fit (``method`` says which ran)."""
    import jax
    import jax.numpy as jnp

    run = programs()
    pts, c = jnp.asarray(points), jnp.asarray(centre)
    matrix = run["assemble"](pts, eps2, viscosity)
    lower = run["factor"](matrix)
    del matrix
    found = np.asarray(jax.block_until_ready(run["finish"](lower, pts, c)), dtype=np.float64)
    del lower
    if np.all(np.isfinite(found)):
        return found, "cholesky"
    n3 = 3 * len(points)
    if 2 * 8 * n3 * n3 > device_budget():
        return found, "cholesky (not finite; LU does not fit)"
    matrix = run["assemble"](pts, eps2, viscosity)
    found = np.asarray(jax.block_until_ready(run["lu_finish"](matrix, pts, c)), dtype=np.float64)
    return found, "lu (cholesky not finite)"


def node_path(body: Body, blob: float) -> np.ndarray:
    """The node's own assembly and solve (``bem_budget.Case``'s free-space solve, the grid's): the control."""
    case = bb.Case(body.built, body.node_name)
    parts = case.parts(body.pose, with_wall=False)
    return case.solve(parts, blob, False)


def matrix_bytes(samples: int) -> int:
    return 8 * (3 * samples) ** 2


def nystrom_level(which: str, spacing_mm: float, save: Save, held: dict[str, Any] | None = None,
                  deadline: float | None = None, control_blob: float | None = 0.5) -> dict[str, Any]:
    """Route (a) at one level: every blob of ``BLOBS``, then every ``eps`` of ``EPS_M`` (blob ``eps / h``), each
    saved as found; then the node's own path at ``control_blob`` where two matrices and its intermediates fit."""
    out: dict[str, Any] = dict(held or {})
    out.pop("stopped", None)
    begun = time.perf_counter()
    body = Body(which, spacing_mm)
    out.setdefault("level", body.describe())
    out["bytes_one_matrix"] = matrix_bytes(len(body.weights))
    if matrix_bytes(len(body.weights)) > device_budget():
        out.update(stopped="does not fit: one matrix exceeds the device budget", complete=True)
        save(out)
        return out
    out["memory_before"] = device_memory()
    blobs = out.setdefault("blobs", {})
    lines = out.setdefault("eps_lines", {})
    jobs = [("blobs", f"{b}", b) for b in BLOBS] + [("eps_lines", f"{e:.6e}", e / body.mean_h) for e in EPS_M]
    for group, key, blob in jobs:
        table = blobs if group == "blobs" else lines
        if key in table:
            continue
        if deadline is not None and time.time() > deadline:
            out["stopped"] = "deadline"
            save(out)
            return out
        start = time.perf_counter()
        eps2 = blob * blob * body.mean_h ** 2
        try:
            r, method = solve_in_place(body.points, eps2, body.viscosity, body.centre)
        except Exception as error:  # memory at the finest level: the finding, and the level's end
            table[key] = {"error": repr(error)[:600], "blob": blob, "memory": device_memory()}
            out.update(stopped=f"does not fit: {repr(error)[:200]}", complete=True)
            save(out)
            return out
        table[key] = reading(r, which, blob=blob, eps_m=math.sqrt(eps2), method=method,
                             seconds=time.perf_counter() - start)
        out["memory_after"] = device_memory()
        save(out)
    if control_blob is not None and "node_path" not in out:
        # the node's path holds the blocks, their transpose, the matrix and LU's copy: about four matrices
        if 4 * matrix_bytes(len(body.weights)) < device_budget():
            start = time.perf_counter()
            try:
                r = node_path(body, control_blob)
            except Exception as error:  # memory: the finding
                out["node_path"] = {"error": repr(error)[:600]}
            else:
                mine = np.asarray(blobs[f"{control_blob}"]["resistance_si"])
                out["node_path"] = {"blob": control_blob, "largest_relative": float(
                    np.max(np.abs(mine - r)) / np.max(np.abs(r))), **reading(r, which),
                    "seconds": time.perf_counter() - start, "memory_after": device_memory()}
        else:
            out["node_path"] = {"not_run": "about four matrices exceed the device budget"}
        save(out)
    out["complete"] = True
    out["seconds"] = time.perf_counter() - begun
    save(out)
    return out


# ---- route (b): forces separated from quadrature (nearest neighbour) ------------------------------------------------
def _nn_programs() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from jax import lax

    def assemble(force: Any, quad: Any, wq: Any, eps2: Any, viscosity: Any, owners_per_block: int) -> Any:
        """``A`` (3N x 3N, rows the collocation points, columns the cells): ``quad`` (N, L, 3) and ``wq`` (N, L) each
        cell's quadrature points and weights, padded with weight zero."""
        n, width = quad.shape[0], owners_per_block

        def body(k: Any, matrix: Any) -> Any:
            start = jnp.minimum(k * width, n - width)
            y = lax.dynamic_slice_in_dim(quad, start, width, 0)  # (c, L, 3)
            w = lax.dynamic_slice_in_dim(wq, start, width, 0)  # (c, L)
            r = force[:, None, None, :] - y[None]  # (N, c, L, 3)
            r2 = jnp.sum(r * r, axis=-1)
            t = lax.rsqrt(r2 + eps2)
            s = w[None] * t * t * t / (8.0 * math.pi * viscosity)  # (N, c, L)
            diagonal = jnp.sum(s * (r2 + 2.0 * eps2), axis=-1)  # (N, c)
            outer = jnp.einsum("mcl,mcli,mclj->micj", s, r, r)  # (N, 3, c, 3)
            eye = jnp.eye(3, dtype=force.dtype)
            block = outer + diagonal[:, None, :, None] * eye[None, :, None, :]
            return lax.dynamic_update_slice(matrix, block, (0, 0, start, 0))

        count = -(-n // width)
        matrix = lax.fori_loop(0, count, body, jnp.zeros((n, 3, n, 3), dtype=force.dtype))
        return matrix.reshape(3 * n, 3 * n)

    def solve(matrix: Any, colloc_motions: Any, wrench_motions: Any) -> Any:
        return jnp.matmul(wrench_motions.T, jnp.linalg.solve(matrix, colloc_motions), precision="highest")

    return {"nn_assemble": jax.jit(assemble, static_argnums=5), "nn_solve": jax.jit(solve, donate_argnums=0)}


class Cells:
    """The force set's cells on a quadrature: each quadrature point to its nearest force point; per cell its points
    and weights padded to the largest cell (weight zero), its area and centroid."""

    def __init__(self, force: np.ndarray, quad: np.ndarray, wq: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        _, owner = cKDTree(force).query(quad, k=1, workers=-1)
        order = np.argsort(owner, kind="stable")
        owner, quad, wq = owner[order], quad[order], wq[order]
        counts = np.bincount(owner, minlength=len(force))
        self.empty = int(np.sum(counts == 0))
        self.counts = counts
        self.width = int(counts.max())
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        slot = np.arange(len(owner)) - starts[owner]
        self.quad = np.repeat(force[:, None, :], self.width, axis=1).copy()
        self.wq = np.zeros((len(force), self.width))
        self.quad[owner, slot] = quad
        self.wq[owner, slot] = wq
        self.area = np.bincount(owner, weights=wq, minlength=len(force))
        safe = np.where(self.area > 0, self.area, 1.0)
        self.centroid = np.stack([np.bincount(owner, weights=wq * quad[:, i], minlength=len(force))
                                  for i in range(3)], axis=1) / safe[:, None]

    def describe(self) -> dict[str, Any]:
        return {"empty_cells": self.empty, "largest_cell": self.width, "mean_cell": float(self.counts.mean()),
                "smallest_cell": int(self.counts.min())}


def nn_solve(force: np.ndarray, cells: Cells, eps2: float, viscosity: float, centre: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    from mime.nodes.hydro.bem._solve import rigid_motions

    run = programs()
    per_block = max(1, min(len(force), NN_BLOCK // cells.width))
    matrix = run["nn_assemble"](jnp.asarray(force), jnp.asarray(cells.quad), jnp.asarray(cells.wq), eps2, viscosity,
                                per_block)
    c = jnp.asarray(centre)
    colloc = rigid_motions(jnp.asarray(force), c)
    wrench = rigid_motions(jnp.asarray(cells.centroid), c) * jnp.repeat(jnp.asarray(cells.area), 3)[:, None]
    return np.asarray(jax.block_until_ready(run["nn_solve"](matrix, colloc, wrench)), dtype=np.float64)


def nn_level(which: str, force_mm: float, quads_mm: tuple[float, ...], eps_set: tuple[float, ...], save: Save,
             held: dict[str, Any] | None = None, deadline: float | None = None) -> dict[str, Any]:
    """Route (b) at one force set: for each quadrature of ``quads_mm`` (coarse to fine) the cells, then each eps of
    ``eps_set``, each solve saved as found."""
    out: dict[str, Any] = dict(held or {})
    out.pop("stopped", None)
    begun = time.perf_counter()
    body = Body(which, force_mm)
    out.setdefault("force_set", body.describe())
    quads = out.setdefault("quadratures", {})
    for q_mm in quads_mm:
        row = quads.setdefault(f"{q_mm}", {"eps": {}})
        if len(row["eps"]) == len(eps_set):
            continue
        counts = body.counts_at(q_mm, ALONG_FACTOR if which == "helix" else 1.0)
        start = time.perf_counter()
        quad, wq = body.resampled(counts)
        cells = Cells(body.points, quad, wq)
        row.update(counts=[list(c) for c in counts], points=len(wq), quad_area_m2=float(wq.sum()),
                   mean_spacing_m=math.sqrt(wq.sum() / len(wq)), cells_seconds=time.perf_counter() - start,
                   **cells.describe())
        if cells.empty:
            row["skipped"] = "a force point owns no quadrature point"
            save(out)
            continue
        for eps in eps_set:
            key = f"{eps:.6e}"
            if key in row["eps"]:
                continue
            if deadline is not None and time.time() > deadline:
                out["stopped"] = "deadline"
                save(out)
                return out
            start = time.perf_counter()
            r = nn_solve(body.points, cells, eps * eps, body.viscosity, body.centre)
            row["eps"][key] = reading(r, which, eps_m=eps, seconds=time.perf_counter() - start)
            out["memory_after"] = device_memory()
            save(out)
        del cells
    out["complete"] = True
    out["seconds"] = time.perf_counter() - begun
    save(out)
    return out


def sphere_references(eps_set: tuple[float, ...] = EPS_M) -> dict[str, float]:
    """The sphere's regularised drag with exact quadrature (``bem_budget.sphere_regularised_drag``) at each eps."""
    a = bb.SPHERE_RATIO * bb.TUBE_RADIUS_M
    return {f"{e:.6e}": bb.sphere_regularised_drag(e / a) for e in eps_set}
