"""The BEM solves for A0f's statics and M1's ladder at A (FL-9 on the floor of the 1/4" tube, the closed surface,
margin 0.1 mm), each a function of the discretisation alone, run on Modal by ``a0f_modal.py`` and here on the CPU at a
coarse spacing for a dry run. No measured speed enters. Predictions: ``predictions_a0f_stage1.json``, committed before
any of these runs.

- ``floored``: the drag node's own step (the model of record) at the floored pose over the eight phases of the half
  turn, at one spacing and blob: the 6 x 6 resistance in SI, the resistance's smallest non-dimensional eigenvalue, and
  the node's ``wall_accepted``, each written through ``save`` as soon as it is found (A0f's statics, stage 1).
- ``level``: one level of M1's ladder at one spacing: for each phase the samples, the blob's ``eps^2`` at both blob
  ratios and the wall's correction ``G`` assembled once; the correction's scale from x1.000 to x1.040 in steps of
  0.0025 at blob 0.5 (the margin scan; its x1.000 row is the level's forward reading); the forward reading at blob 0.35;
  the free-space resistance at both blobs (phase 0's pose: free space has no preferred pose); and a control at phase 0,
  this assembly against the node's own step at blob 0.5. Every solve is saved as it is found, and a rerun skips what the
  saved record already holds.
- ``probe``: the time and the device's peak memory of one assembly and one solve at a spacing (stage 2's cost, measured
  in stage 1); the resistance is not kept and no speed is read from it.
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
import bem  # noqa: E402
import operating_point as op  # noqa: E402

#: M3's fine scan of the wall correction's scale (the register's, section 6 M1).
SCAN_SCALES = tuple(round(1.0 + 0.0025 * k, 4) for k in range(17))
BLOBS = (0.5, 0.35)

Save = Callable[[dict[str, Any]], None]


def _now() -> float:
    return time.time()


def device_memory() -> dict[str, Any]:
    """The first device's memory statistics, where its backend reports them."""
    import jax

    stats = jax.devices()[0].memory_stats() or {}
    return {key: int(stats[key]) for key in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit") if key in stats}


class Floored:
    """The closed rung at A at one discretisation, the node's own step compiled once (resistance and
    ``wall_accepted`` from one call)."""

    def __init__(self, spacing_mm: float, blob: float = op.BLOB, margin_mm: float = op.MARGIN_MM) -> None:
        import jax

        from mime.build._maddening import node_state

        self.spacing_mm, self.blob, self.margin_mm = spacing_mm, blob, margin_mm
        self.built = op.experiment("bem", spacing_mm=spacing_mm, blob=blob, margin_mm=margin_mm).build()
        self.leaves = op.leaves_of(self.built)
        self.node = self.built.graph.get_node(op.DRAG_NODE)
        self.drag = self.leaves[op.DRAG_NODE]
        self.offset = float(op.floored_offset(self.leaves[op.BODY_NODE]))
        state = node_state(self.built.graph, op.DRAG_NODE)

        def step(pose: Any, drag: Any) -> tuple[Any, Any]:
            found = self.node.update(state, {"pose": pose}, 1e-3, params=drag)
            return found["resistance"], found["wall_accepted"]

        self._step = jax.jit(step)

    def pose(self, phase: float) -> Any:
        return op.floored_pose(np.float64(self.offset), phase)

    def step(self, phase: float) -> tuple[np.ndarray, bool]:
        import jax

        r, accepted = jax.block_until_ready(self._step(self.pose(phase), self.drag))
        return np.asarray(r, dtype=np.float64), bool(np.asarray(accepted))

    def describe(self) -> dict[str, Any]:
        held = self.node.samples(self.node._step_constants(self.drag))  # noqa: SLF001 - what the step samples
        weights = np.asarray(held.weights, dtype=np.float64)
        return {"spacing_mm": self.spacing_mm, "blob": self.blob, "margin_mm": self.margin_mm, "offset_m": self.offset,
                "samples": int(weights.shape[0]), "unknowns": int(3 * weights.shape[0]),
                "mean_spacing_m": float(math.sqrt(weights.sum() / weights.shape[0])), "area_m2": float(weights.sum()),
                "counts": [list(c) for c in self.node.constants()["counts"]],
                "kind_version": self.node.constants()["kind_version"]}


def reading(r: np.ndarray, accepted: bool) -> dict[str, Any]:
    return {"resistance_si": r.tolist(), "min_eigenvalue_nd": bem.smallest_eigenvalue(r), "wall_accepted": accepted,
            "finite": bool(np.all(np.isfinite(r))), "axial_mm_s": 1e3 * float(op.speeds(r)["axial"])}


def floored(spacing_mm: float, save: Save, held: dict[str, Any] | None = None, blob: float = op.BLOB
            ) -> dict[str, Any]:
    """Stage 1: the node's step at the eight phases, each saved as it is found, the saved ones skipped."""
    out: dict[str, Any] = dict(held or {})
    started = time.perf_counter()
    rung = Floored(spacing_mm, blob)
    out.setdefault("rung", rung.describe())
    out["build_seconds"] = time.perf_counter() - started
    phases = out.setdefault("phases", {})
    for k, phase in enumerate(op.PHASES):
        if str(k) in phases:
            continue
        begun = time.perf_counter()
        r, accepted = rung.step(phase)
        phases[str(k)] = {"phase": phase, **reading(r, accepted), "seconds": time.perf_counter() - begun,
                          "finished_unix": _now()}
        out["device_memory"] = device_memory()
        save(out)
        if not (accepted and phases[str(k)]["min_eigenvalue_nd"] > 0.0 and phases[str(k)]["finite"]):
            out["stopped"] = f"phase {k}: wall_accepted {accepted}, smallest eigenvalue " \
                             f"{phases[str(k)]['min_eigenvalue_nd']}, finite {phases[str(k)]['finite']}"
            save(out)
            break
    out["complete"] = len(phases) == len(op.PHASES)
    out["seconds"] = time.perf_counter() - started
    save(out)
    return out


# ---- M1's ladder: one level ------------------------------------------------------------------------------------------
class Level:
    """One spacing of the ladder: the samples at the floored pose, the wall's correction ``G`` and the free-space
    matrix ``A`` held once per phase, and the solves at a scale of ``G`` and at either blob."""

    def __init__(self, spacing_mm: float) -> None:
        import jax
        import jax.numpy as jnp

        from mime.core import quaternion
        from mime.nodes.hydro.bem import blob_squared, wall_correction_named
        from mime.nodes.hydro.bem._solve import PRECISION, point_forces, rigid_motions, stokeslet_matrix

        self.rung = Floored(spacing_mm, op.BLOB)
        node, drag = self.rung.node, self.rung.drag
        kind = wall_correction_named(node.constants()["wall_kind"], node.constants()["wall_kind_version"])

        def assemble(pose: Any, drag: Any, with_wall: bool) -> dict[str, Any]:
            p = node._step_constants(drag)  # noqa: SLF001 - what the node's step reads
            held = node.samples(p)
            points = pose[:3] + jnp.matmul(held.points, quaternion.rotation_matrix(pose[3:]).T, precision=PRECISION)
            eps2 = {b: blob_squared(held.weights, b) for b in BLOBS}
            out = {"points": points, "eps2_0.5": eps2[0.5], "eps2_0.35": eps2[0.35], "viscosity": p["viscosity"],
                   "centre": pose[:3]}
            if with_wall:
                corrected = kind.correction(points, {name: p[f"wall_{name}"] for name in kind.parameters},
                                            p["viscosity"])
                out["G"], out["accepted"] = corrected.matrix, corrected.accepted
            return out

        def free_matrix(parts: dict[str, Any], eps2: Any) -> Any:
            return stokeslet_matrix(parts["points"], eps2, parts["viscosity"])

        def solve(matrix: Any, wall: Any, scale: Any, points: Any, centre: Any) -> Any:
            full = matrix if wall is None else matrix + scale * wall
            motions = rigid_motions(points, centre)
            found = jnp.matmul(motions.T, point_forces(full, motions), precision=PRECISION)
            return (found + found.T) / 2.0

        self._assemble = jax.jit(assemble, static_argnums=2)
        self._free_matrix = jax.jit(free_matrix)
        self._solve = jax.jit(solve)
        self._solve_free = jax.jit(lambda matrix, points, centre: solve(matrix, None, 1.0, points, centre))
        self.drag = drag

    def parts(self, phase: float, with_wall: bool = True) -> dict[str, Any]:
        return self._assemble(self.rung.pose(phase), self.drag, with_wall)

    def solve(self, parts: dict[str, Any], blob: float, scale: float | None) -> np.ndarray:
        import jax

        matrix = self._free_matrix(parts, parts[f"eps2_{blob}"])
        if scale is None:
            found = self._solve_free(matrix, parts["points"], parts["centre"])
        else:
            found = self._solve(matrix, parts["G"], scale, parts["points"], parts["centre"])
        return np.asarray(jax.block_until_ready(found), dtype=np.float64)


def level(spacing_mm: float, save: Save, held: dict[str, Any] | None = None, deadline_unix: float | None = None
          ) -> dict[str, Any]:
    """One level of the ladder, every solve saved as it is found and the saved ones skipped. The level stops
    (``stopped``) at a non-finite or indefinite reading at the nominal scale or a refused wall read (the caller goes on
    to the next level), and at ``deadline_unix`` (the caller stops)."""
    import jax

    out: dict[str, Any] = dict(held or {})
    if out.get("stopped") == "deadline":
        out.pop("stopped")
    started = time.perf_counter()
    lev = Level(spacing_mm)
    out.setdefault("rung", lev.rung.describe())
    out["build_seconds"] = time.perf_counter() - started
    phases = out.setdefault("phases", {})
    solves = out.setdefault("solves", 0)

    def late() -> bool:
        return deadline_unix is not None and _now() > deadline_unix

    def checkpoint() -> None:
        out["solves"] = solves
        out["device_memory"] = device_memory()
        save(out)

    for k, phase in enumerate(op.PHASES):
        row = phases.setdefault(str(k), {"phase": phase, "scan": {}, "blob_0.35": None})
        if len(row["scan"]) == len(SCAN_SCALES) and row["blob_0.35"] is not None:
            continue
        begun = time.perf_counter()
        parts = lev.parts(phase)
        accepted = bool(np.asarray(jax.block_until_ready(parts["accepted"])))
        row["wall_accepted"] = accepted
        row["assembly_seconds"] = time.perf_counter() - begun
        if not accepted:
            out["stopped"] = f"phase {k}: a wall read refused"
            checkpoint()
            return out
        for scale in SCAN_SCALES:
            if f"{scale:.4f}" in row["scan"]:
                continue
            if late():
                out["stopped"] = "deadline"
                checkpoint()
                return out
            begun = time.perf_counter()
            r = lev.solve(parts, 0.5, scale)
            solves += 1
            row["scan"][f"{scale:.4f}"] = {"scale": scale, **reading(r, accepted),
                                           "seconds": time.perf_counter() - begun}
            if scale != 1.0:
                row["scan"][f"{scale:.4f}"].pop("resistance_si")
            checkpoint()
            nominal = row["scan"].get("1.0000")
            if scale == 1.0 and not (nominal["finite"] and nominal["min_eigenvalue_nd"] > 0.0):
                out["stopped"] = f"phase {k}: the nominal reading is not finite or not positive definite"
                checkpoint()
                return out
        if row["blob_0.35"] is None:
            if late():
                out["stopped"] = "deadline"
                checkpoint()
                return out
            begun = time.perf_counter()
            r = lev.solve(parts, 0.35, 1.0)
            solves += 1
            row["blob_0.35"] = {**reading(r, accepted), "seconds": time.perf_counter() - begun}
            checkpoint()
        del parts
    if "free_space" not in out:
        parts = lev.parts(op.PHASES[0], with_wall=False)
        out["free_space"] = {}
        for blob in BLOBS:
            begun = time.perf_counter()
            r = lev.solve(parts, blob, None)
            solves += 1
            out["free_space"][str(blob)] = {**reading(r, True), "seconds": time.perf_counter() - begun}
        checkpoint()
        del parts
    if "control_own_against_node" not in out:
        try:
            node_r, node_accepted = lev.rung.step(op.PHASES[0])
        except Exception as error:  # the node's step holds assembly and solve in one program: memory may refuse it
            out["control_own_against_node"] = {"not_run": repr(error)[:600]}
        else:
            solves += 1
            own = np.asarray(phases["0"]["scan"]["1.0000"]["resistance_si"])
            out["control_own_against_node"] = {"largest_relative": float(np.max(np.abs(own - node_r)) /
                                                                         np.max(np.abs(node_r))),
                                               "node_wall_accepted": node_accepted}
        checkpoint()
    out["complete"] = True
    out["seconds_this_call"] = time.perf_counter() - started
    checkpoint()
    return out


def probe(spacing_mm: float, node_step: bool = False) -> dict[str, Any]:
    """The cost of one level's pieces at ``spacing_mm``: the build, one assembly with the wall, and two solves at
    blob 0.5 (the first compiles), with the device's peak memory; with ``node_step``, also the node's own step (the
    level's control). Every resistance is dropped unread."""
    started = time.perf_counter()
    lev = Level(spacing_mm)
    built = time.perf_counter() - started
    begun = time.perf_counter()
    parts = lev.parts(op.PHASES[0])
    import jax

    jax.block_until_ready(parts["G"])
    assembly = time.perf_counter() - begun
    times = []
    for scale in (1.0, 1.0025):
        begun = time.perf_counter()
        lev.solve(parts, 0.5, scale)
        times.append(time.perf_counter() - begun)
    found = {"spacing_mm": spacing_mm, "rung": lev.rung.describe(), "build_seconds": built,
             "assembly_seconds_first": assembly, "solve_seconds": times, "device_memory": device_memory()}
    if node_step:
        del parts
        begun = time.perf_counter()
        lev.rung.step(op.PHASES[0])
        found["node_step_seconds_first"] = time.perf_counter() - begun
        found["device_memory_after_node_step"] = device_memory()
    return found
