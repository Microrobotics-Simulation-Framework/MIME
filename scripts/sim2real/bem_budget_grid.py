"""The BEM error budget's grid (item 2) and its controls (item 3): every solve saved through ``save`` as it is found, a
rerun skipping what the record holds. ``bem_budget_modal.py`` runs A's levels on an A100-80GB; the spheres run here on
the CPU (they are small).

- ``a_level``: A (FL-9 floored in the 1/4 inch tube, closed surface, margin 0.1 mm) at one spacing: at each of the eight
  phases (the least margin's and the lock's first, ``PHASE_ORDER``) the samples and the wall's correction assembled once,
  then at each blob fraction the resistance with the node's correction (``tube``), and with the regularised plane-wall
  image's correction added on the samples near the wall (``tube_regimage``, the lever); free space at phase 0's pose at
  each blob fraction.
- ``sphere_level``: the controls at one spacing, each blob fraction: a sphere (0.3 of A's tube radius) in free space and
  on the axis of A's tube.
- ``sphere_fine``: the sphere's regularisation reference's own control: a free-space solve at ``h << eps``.

No measured speed enters.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python bem_budget_grid.py --spheres --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import bem  # noqa: E402
import bem_budget as bb  # noqa: E402
import operating_point as op  # noqa: E402

Save = Callable[[dict[str, Any]], None]
#: The phase of least margin first (phase 6: the least onset of the wall correction's scale at every level of M1's
#: ladder, alone at 0.2 mm and tied at the finer ones), then the lock's (phase 0: the largest effective turning
#: resistance R_OO - R_xO^2 / R_xx at 0.16 mm, so the largest lag), then the rest.
PHASE_ORDER = (6, 0, 2, 4, 1, 3, 5, 7)
#: The sphere reference's control: eps / a and the nominal spacing (mm) of the fine solves (h about eps / 4 and eps / 3).
FINE_EPS_OVER_A = 0.25
FINE_SPACINGS_MM = (0.0595, 0.079)


def reading(r: np.ndarray, accepted: bool = True, **extra: Any) -> dict[str, Any]:
    finite = bool(np.all(np.isfinite(r)))
    return {"resistance_si": r.tolist(), "min_eigenvalue_nd": bem.smallest_eigenvalue(r) if finite else None,
            "accepted": bool(accepted),
            "finite": finite, "axial_mm_s": 1e3 * float(op.speeds(r)["axial"]), **extra}


class ALevel:
    def __init__(self, spacing_mm: float) -> None:
        import jax
        import jax.numpy as jnp

        from mime.nodes.hydro.bem._solve import PRECISION, point_forces, rigid_motions, stokeslet_matrix

        self.rung = bb.a_rung(spacing_mm)
        self.case = bb.Case(self.rung.built, op.DRAG_NODE)
        p = self.rung.node._step_constants(self.rung.drag)  # noqa: SLF001 - what the node's step reads
        self.frame = np.asarray(p["wall_frame"], dtype=np.float64).reshape(3, 3)
        self.centre = np.asarray(p["wall_centre"], dtype=np.float64)
        self.radius = float(p["wall_radius"])

        def patched(points: Any, eps2: Any, viscosity: Any, centre: Any, wall: Any, rows: Any, patch: Any) -> Any:
            matrix = stokeslet_matrix(points, eps2, viscosity) + wall
            matrix = matrix.at[rows[:, None], rows[None, :]].add(patch)
            motions = rigid_motions(points, centre)
            found = jnp.matmul(motions.T, point_forces(matrix, motions), precision=PRECISION)
            return (found + found.T) / 2.0

        self._patched = jax.jit(patched)

    def corrector(self, parts: dict[str, Any]) -> tuple[np.ndarray, Any]:
        """The samples near the wall at this pose and the correction's program for them, compiled once per pose."""
        import jax

        index = bb.near_wall(np.asarray(parts["points"]), self.frame, self.centre, self.radius)
        program = jax.jit(lambda pts, e2: bb.image_correction_blocks(pts, index, self.frame, self.centre, self.radius,
                                                                     parts["viscosity"], e2))
        return index, program

    def correction(self, parts: dict[str, Any], blob: float, prepared: tuple[np.ndarray, Any] | None = None
                   ) -> tuple[Any, np.ndarray, dict[str, Any]]:
        import jax.numpy as jnp

        index, program = prepared if prepared is not None else self.corrector(parts)
        eps2 = blob * blob * parts["area_per_sample"]
        blocks = program(parts["points"], eps2)
        n = len(index)
        patch = jnp.transpose(blocks, (0, 2, 1, 3)).reshape(3 * n, 3 * n)
        rows = bb.scatter_rows(index)
        g_near = jnp.max(jnp.abs(parts["G"][jnp.asarray(rows)][:, jnp.asarray(rows)]))
        stats = {"near_samples": int(n), "patch_max_over_G_near_max": float(jnp.max(jnp.abs(patch)) / g_near)}
        return patch, rows, stats

    def solve_patched(self, parts: dict[str, Any], blob: float, rows: np.ndarray, patch: Any) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        eps2 = blob * blob * parts["area_per_sample"]
        found = self._patched(parts["points"], eps2, parts["viscosity"], parts["centre"], parts["G"],
                              jnp.asarray(rows), patch)
        return np.asarray(jax.block_until_ready(found), dtype=np.float64)


def a_level(spacing_mm: float, save: Save, held: dict[str, Any] | None = None, deadline_unix: float | None = None,
            regimage: bool = True) -> dict[str, Any]:
    """One spacing of A's grid. Stops at ``deadline_unix`` (``stopped: deadline``); a refused wall read stops the
    level. The regularised-image arm, which needs the near-wall patch beside the full matrix, records an error (an
    out-of-memory at the finest spacing is a finding) and the level goes on."""
    import jax

    out: dict[str, Any] = dict(held or {})
    if out.get("stopped") == "deadline":
        out.pop("stopped")
    started = time.perf_counter()
    level = ALevel(spacing_mm)
    out.setdefault("rung", level.rung.describe())
    out["build_seconds"] = time.perf_counter() - started
    phases = out.setdefault("phases", {})

    def late() -> bool:
        return deadline_unix is not None and time.time() > deadline_unix

    for k in PHASE_ORDER:
        row = phases.setdefault(str(k), {"phase": op.PHASES[k], "tube": {}, "tube_regimage": {}})
        if len(row["tube"]) == len(bb.BLOBS) and (not regimage or len(row["tube_regimage"]) == len(bb.BLOBS)):
            continue
        begun = time.perf_counter()
        parts = level.case.parts(level.rung.pose(op.PHASES[k]))
        accepted = bool(np.asarray(parts["accepted"]))
        row["wall_accepted"], row["assembly_seconds"] = accepted, time.perf_counter() - begun
        row.update(bb.describe(parts))
        if not accepted:
            out["stopped"] = f"phase {k}: a wall read refused"
            save(out)
            return out
        for blob in bb.BLOBS:
            if f"{blob}" in row["tube"]:
                continue
            if late():
                out["stopped"] = "deadline"
                save(out)
                return out
            begun = time.perf_counter()
            r = level.case.solve(parts, blob, True)
            row["tube"][f"{blob}"] = reading(r, accepted, seconds=time.perf_counter() - begun)
            save(out)
        if regimage:
            prepared = level.corrector(parts) if len(row["tube_regimage"]) < len(bb.BLOBS) else None
            for blob in bb.BLOBS:
                if f"{blob}" in row["tube_regimage"]:
                    continue
                if late():
                    out["stopped"] = "deadline"
                    save(out)
                    return out
                begun = time.perf_counter()
                try:
                    patch, rows, stats = level.correction(parts, blob, prepared)
                    r = level.solve_patched(parts, blob, rows, patch)
                    del patch
                except Exception as error:  # memory at the finest spacing: the finding, not a stop
                    row["tube_regimage"][f"{blob}"] = {"error": repr(error)[:800]}
                    save(out)
                    break
                row["tube_regimage"][f"{blob}"] = reading(r, accepted, seconds=time.perf_counter() - begun, **stats)
                save(out)
        del parts
    if "free" not in out or len(out["free"]) < len(bb.BLOBS):
        parts = level.case.parts(level.rung.pose(op.PHASES[0]), with_wall=False)
        free = out.setdefault("free", {})
        for blob in bb.BLOBS:
            if f"{blob}" in free:
                continue
            if late():
                out["stopped"] = "deadline"
                save(out)
                return out
            begun = time.perf_counter()
            free[f"{blob}"] = reading(level.case.solve(parts, blob, False), True,
                                      seconds=time.perf_counter() - begun)
            save(out)
        del parts
    if "control_frozen_against_node" not in out and abs(spacing_mm - bb.FROZEN_SPACING_MM) < 1e-12:
        # the node's own step at the frozen spacing and blob, phase 0, against this assembly's solve
        node_r, node_accepted = level.rung.step(op.PHASES[0])
        own = np.asarray(phases["0"]["tube"][f"{bb.FROZEN_BLOB}"]["resistance_si"])
        out["control_frozen_against_node"] = {"largest_relative": float(np.max(np.abs(own - node_r))
                                                                        / np.max(np.abs(node_r))),
                                              "node_wall_accepted": node_accepted}
        save(out)
    jax.clear_caches()
    out["complete"] = True
    out["seconds_this_call"] = time.perf_counter() - started
    save(out)
    return out


# ---- the controls -----------------------------------------------------------------------------------------------------
def sphere_level(spacing_mm: float, save: Save, held: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = dict(held or {})
    out["spacing_mm"] = spacing_mm
    for arm, tube in (("free", False), ("tube", True)):
        block = out.setdefault(arm, {})
        if len(block.get("solves", {})) == len(bb.BLOBS):
            continue
        built = bb.sphere_experiment(spacing_mm, tube)
        case = bb.Case(built, "drag:ball")
        parts = case.parts(bb.sphere_pose(built), with_wall=tube)
        block.update(bb.describe(parts))
        if tube:
            block["accepted"] = bool(np.asarray(parts["accepted"]))
        solves = block.setdefault("solves", {})
        for blob in bb.BLOBS:
            if f"{blob}" in solves:
                continue
            begun = time.perf_counter()
            r = case.solve(parts, blob, tube)
            solves[f"{blob}"] = {"resistance_si": r.tolist(), "drag_over_stokes": float(r[2, 2] / (
                6 * math.pi * bb.VISCOSITY * bb.SPHERE_RATIO * bb.TUBE_RADIUS_M)), "seconds": time.perf_counter() - begun}
            save(out)
    return out


def sphere_fine(save: Save, held: dict[str, Any] | None = None) -> dict[str, Any]:
    """Free-space solves at ``eps / a = FINE_EPS_OVER_A`` with ``h`` about ``eps / 4`` and ``eps / 3``, against the
    reference at the same ``eps``: the reference's control (they must agree as ``h / eps`` falls)."""
    out: dict[str, Any] = dict(held or {})
    a = bb.SPHERE_RATIO * bb.TUBE_RADIUS_M
    rows = out.setdefault("rows", {})
    for spacing in FINE_SPACINGS_MM:
        if f"{spacing}" in rows:
            continue
        built = bb.sphere_experiment(spacing, False)
        case = bb.Case(built, "drag:ball")
        parts = case.parts(bb.sphere_pose(built), with_wall=False)
        h = math.sqrt(float(np.asarray(parts["area_per_sample"])))
        blob = FINE_EPS_OVER_A * a / h
        r = case.solve(parts, blob, False)
        rows[f"{spacing}"] = {"samples": int(parts["points"].shape[0]), "mean_spacing_m": h, "blob": blob,
                              "eps_over_a": FINE_EPS_OVER_A, "h_over_eps": h / (FINE_EPS_OVER_A * a),
                              "drag_over_stokes": float(r[2, 2] / (6 * math.pi * bb.VISCOSITY * a))}
        save(out)
    out["reference"] = bb.sphere_regularised_drag(FINE_EPS_OVER_A)
    save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spheres", action="store_true")
    parser.add_argument("--fine", action="store_true")
    parser.add_argument("--a-level", type=float, default=None, help="one of A's levels, here (a dry run)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    op.require_float64()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)

    def saver(name: str) -> Save:
        def save(payload: dict[str, Any]) -> None:
            (folder / f"{name}.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n")
        return save

    def held(name: str) -> dict[str, Any] | None:
        path = folder / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    if args.spheres:
        for spacing in bb.SPACINGS_MM:
            name = f"sphere_{spacing:.4f}"
            sphere_level(spacing, saver(name), held(name))
            print(f"done {name}", flush=True)
    if args.fine:
        sphere_fine(saver("sphere_fine"), held("sphere_fine"))
        print("done sphere_fine", flush=True)
    if args.a_level is not None:
        name = f"a_level_{args.a_level:.4f}"
        a_level(args.a_level, saver(name), held(name))
        print(f"done {name}", flush=True)


if __name__ == "__main__":
    main()
