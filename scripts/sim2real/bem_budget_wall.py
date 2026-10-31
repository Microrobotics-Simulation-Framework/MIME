"""Item 1 of the BEM error budget: the wall table's own error near the floor at A, measured directly, and the plane-wall
image beside it. Local CPU (the heavy lock), no Modal. No measured speed enters.

At A's floored pose (FL-9, closed surface, the frozen spacing 0.16 mm, the frozen blob 0.5), for every ordered pair of
the samples nearer the wall than ``bem_budget.NEAR_WALL_M``:

- ``exact``: the image series evaluated directly, by the table builder's own code at its per-source truncation rule
  (``m3_route2.exact_group``, scale 1.5, the table's own: M3 stage 2 measured it 1e-4 under the table's error);
- ``table``: the table's blended read, as the node assembles it (``m3_route2.table_pair``); its plumbing against the
  library's matrix at the same blocks is checked (``plumbing``);
- ``blake0`` and ``blake_eps``: Blake's plane-wall image in the tangent plane at the source's foot, point and
  regularised at each blob fraction's ``eps`` (``bem_budget.blake``).

Reported per pair against the nearer sample's height over the wall: the table's error ``E = |table - exact|`` and the
plane's departure from the tube ``|exact - blake0|`` (which must close as the gap closes), and the regularisation's
size ``|blake_eps - blake0|``, each over the exact block's largest entry. Then the solves: the near blocks replaced by
the exact series, and the regularised image's correction added (the grid's lever, recomputed here with this code's
own Blake and at two other near-wall reaches), each against the library's resistance and its axial speed.

**The control:** a table with a planted error at its near-wall radial knots (random knots in the near samples' range of
radius, ``x(1 + d)``, ``|d|`` from 0.2% to 1%, random sign; ten draws) must raise the measured E above the unplanted
maximum on every draw.

    MIME_ASSETS_DIR=... JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python bem_budget_wall.py --pairs --phase 0 --out DIR
    MIME_ASSETS_DIR=... JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python bem_budget_wall.py --solve --phase 0 --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PAIRS = "--pairs" in sys.argv
if PAIRS:  # one thread per worker
    for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_variable] = "1"
import numpy as np  # noqa: E402

FACTOR = 1.5
PLANTS, PLANT_SEED = 10, 20261031
PLANT_RANGE = (0.002, 0.01)
REACHES_M = (0.4e-3, 0.6e-3, 0.8e-3)
XLA_ONE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def build_samples(phase_index: int, spacing_mm: float) -> Any:
    import jax.numpy as jnp

    import bem_budget as bb
    import m3_route2 as r2
    import operating_point as op
    from mime.core import quaternion

    rung = bb.a_rung(spacing_mm)
    node = rung.node
    p = node._step_constants(rung.drag)  # noqa: SLF001
    held = node.samples(p)
    pose = rung.pose(op.PHASES[phase_index])
    points = np.asarray(pose[:3] + jnp.matmul(held.points, quaternion.rotation_matrix(pose[3:]).T,
                                              precision="highest"))
    frame = np.asarray(p["wall_frame"], dtype=np.float64).reshape(3, 3)
    centre = np.asarray(p["wall_centre"], dtype=np.float64)
    radius = float(p["wall_radius"])
    local = (points - centre) @ frame.T / radius
    weights = np.asarray(held.weights)
    samples = r2.Samples(local, points, frame, centre, radius, float(p["viscosity"]), None, {})
    return samples, float(math.sqrt(weights.sum() / weights.shape[0]))


# ---- workers -----------------------------------------------------------------------------------------------------------
_W: dict[str, Any] = {}


def _init(rho: np.ndarray, phi: np.ndarray, z: np.ndarray, near: np.ndarray, readable: float) -> None:
    os.environ["XLA_FLAGS"] = XLA_ONE_THREAD
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "1"
    _W.update(rho=rho, phi=phi, z=z, near=near, readable=readable)


def _source(s: int) -> tuple[int, np.ndarray, float]:
    import m3_route2 as r2

    started = time.perf_counter()
    rho, phi, z, near = _W["rho"], _W["phi"], _W["z"], _W["near"]
    block = r2.exact_group(float(rho[s]), rho[near], phi[near] - phi[s], z[near] - z[s],
                           r2.pair_truncation(float(rho[s]), factor=FACTOR, readable=_W["readable"]))
    return s, r2.cartesian_block(block, phi[near], np.full(len(near), phi[s])), time.perf_counter() - started


def table_blocks(table: Any, samples: Any, near: np.ndarray) -> np.ndarray:
    """``[t, s]`` blended table reads (tube axes, unit tube) for every ordered pair of ``near``."""
    import m3_route2 as r2

    t, s = np.meshgrid(near, near, indexing="ij")
    pairs = samples.pairs(t.ravel(), s.ravel())
    return r2.table_pair(table, pairs)["blended"].reshape(len(near), len(near), 3, 3)


def blake_blocks(samples: Any, near: np.ndarray, eps_unit: float) -> np.ndarray:
    """``[t, s]`` Blake's image (``G - S``, over ``8 pi``, tube axes, unit tube) in the tangent plane at each source's
    foot, at ``eps_unit`` (0: the point form)."""
    import jax
    import jax.numpy as jnp

    import bem_budget as bb

    local = samples.local[near]
    planes = np.stack([bb.tangent_plane(x)[0] for x in local])
    origins = np.stack([bb.tangent_plane(x)[1] for x in local])
    one = jax.jit(jax.vmap(lambda x, y: bb.blake_image(x, y, eps_unit * eps_unit), in_axes=(0, None)))
    out = np.empty((len(near), len(near), 3, 3))
    for j in range(len(near)):
        q, o = planes[j], origins[j]
        y = q @ (local[j] - o)
        xs = (local - o) @ q.T
        blocks = np.asarray(one(jnp.asarray(xs), jnp.asarray(y)))
        out[:, j] = np.einsum("ai,tab,bj->tij", q, blocks, q) / (8.0 * math.pi)
    return out


def relative(miss: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.max(np.abs(miss), axis=(-2, -1)) / np.max(np.abs(reference), axis=(-2, -1))


def planted_table(rng: np.random.Generator, near_rho: tuple[float, float]) -> tuple[Any, dict[str, Any]]:
    """The shipped table with ``x(1 + d)`` at a random run of radial knots (target radius) inside the near samples'
    range of radius."""
    from mime import assets
    from mime.nodes.hydro.bem._tube import unit_tube_table

    path = assets.locate(assets.reference("unit_tube_R0.997"))
    with np.load(path) as held:
        radius = float(held["R_cyl"])
        rho = held["rho_grid"] / radius
        data = np.array(held["data"])
        stated = held["readable_rho"] if "readable_rho" in held.files else None
        args = (radius, held["rho_grid"], held["dphi_grid"], held["dz_grid"])
    inside = np.flatnonzero((rho >= near_rho[0]) & (rho <= near_rho[1]))
    start = int(rng.integers(0, max(1, len(inside) - 2)))
    knots = inside[start:start + int(rng.integers(1, 4))]
    size = float(rng.uniform(*PLANT_RANGE)) * float(rng.choice((-1.0, 1.0)))
    data[knots] *= 1.0 + size
    readable = None if stated is None else tuple(float(r) / radius for r in np.asarray(stated).ravel())
    table = unit_tube_table(*args, data, readable_rho=readable)
    return table, {"knots": knots.tolist(), "knot_rho": rho[knots].tolist(), "size": size}


# ---- --pairs -----------------------------------------------------------------------------------------------------------
def run_pairs(phase: int, out: Path, workers: int, spacing_mm: float) -> dict[str, Any]:
    import multiprocessing as mp

    import bem_budget as bb
    import m3_route2 as r2

    started = time.time()
    samples, h = build_samples(phase, spacing_mm)
    near = bb.near_wall(samples.points, samples.frame, samples.centre, samples.radius, max(REACHES_M))
    heights = (1.0 - samples.rho) * samples.radius
    readable = max(r2.RHO_MAX_A, math.ceil(float(samples.rho.max()) * 1e4) / 1e4)
    print(f"{len(near)} samples nearer the wall than {max(REACHES_M)} m of {samples.count}; rho max "
          f"{samples.rho.max():.5f}, readable rule {readable}", flush=True)
    exact = np.full((len(near), len(near), 3, 3), np.nan)
    order = sorted(range(len(near)), key=lambda j: -samples.rho[near[j]])
    position = {int(s): j for j, s in enumerate(near)}
    cpu = 0.0
    context = mp.get_context("spawn")
    with context.Pool(workers, initializer=_init, initargs=(samples.rho, samples.phi, samples.z, near, readable)) as pool:
        for done, (s, cart, seconds) in enumerate(pool.imap_unordered(_source, [int(near[j]) for j in order])):
            exact[:, position[s]] = cart
            cpu += seconds
            if done % 100 == 0:
                print(f"{done}/{len(near)} sources, {time.time() - started:.0f} s", flush=True)
    seconds_pairs = time.time() - started
    table = r2.load_table()
    tab = table_blocks(table, samples, near)
    blake0 = blake_blocks(samples, near, 0.0)
    blake_eps = {f"{b}": blake_blocks(samples, near, b * h / samples.radius) for b in bb.BLOBS}
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"pairs_phase{phase}.npz", near=near, heights=heights[near], exact=exact, table=tab,
             blake0=blake0, **{f"blake_{k}": v for k, v in blake_eps.items()}, local=samples.local[near])
    summary = {"phase": phase, "spacing_mm": spacing_mm, "mean_spacing_m": h, "samples": samples.count,
               "near": int(len(near)), "reach_m": max(REACHES_M), "readable_rule": readable,
               "pairs_seconds": seconds_pairs, "cpu_hours": cpu / 3600.0, "workers": workers,
               "finite_exact": bool(np.all(np.isfinite(exact)))}
    summary["planted"] = planted_control(samples, near, exact, tab)
    summary["seconds"] = time.time() - started
    (out / f"pairs_phase{phase}.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    return summary


def planted_control(samples: Any, near: np.ndarray, exact: np.ndarray, tab: np.ndarray) -> dict[str, Any]:
    """On the pairs of the samples within the smallest reach (for the time): the knots are drawn in their radii."""
    rng = np.random.default_rng(PLANT_SEED)
    keep = np.flatnonzero((1.0 - samples.rho[near]) * samples.radius < min(REACHES_M))
    near, exact, tab = near[keep], exact[np.ix_(keep, keep)], tab[np.ix_(keep, keep)]
    base = relative(tab - exact, exact)
    rho = samples.rho[near]
    rows = []
    for _ in range(PLANTS):
        table, plant = planted_table(rng, (float(rho.min()), float(rho.max())))
        planted = relative(table_blocks(table, samples, near) - exact, exact)
        plant.update(max_E=float(planted.max()), unplanted_max_E=float(base.max()),
                     caught=bool(planted.max() > base.max()))
        rows.append(plant)
        del table
    return {"draws": rows, "caught_every_run": all(r["caught"] for r in rows), "near": int(len(near))}


# ---- --solve -----------------------------------------------------------------------------------------------------------
def run_solve(phase: int, out: Path, spacing_mm: float) -> dict[str, Any]:
    import jax.numpy as jnp

    import bem_budget as bb
    import bem_budget_grid as grid
    import operating_point as op

    started = time.time()
    held = np.load(out / f"pairs_phase{phase}.npz")
    near_all, heights = held["near"], held["heights"]
    level = grid.ALevel(spacing_mm)
    parts = level.case.parts(level.rung.pose(op.PHASES[phase]))
    points = np.asarray(parts["points"])
    frame, radius, mu = level.frame, level.radius, float(np.asarray(parts["viscosity"]))
    result: dict[str, Any] = {"phase": phase, "spacing_mm": spacing_mm, "blob": bb.FROZEN_BLOB}

    def world(blocks: np.ndarray) -> np.ndarray:
        """``[t, s]`` unit-tube blocks (tube axes) in the library's layout and units."""
        n = blocks.shape[0]
        turned = np.einsum("ai,tsab,bj->tsij", frame, blocks, frame)
        return turned.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) / (radius * mu)

    def axial(r: np.ndarray) -> float:
        return 1e3 * float(op.speeds(r)["axial"])

    base = level.case.solve(parts, bb.FROZEN_BLOB, True)
    result["library"] = {"axial_mm_s": axial(base), "min_eigenvalue_nd": grid.bem.smallest_eigenvalue(base),
                         "resistance_si": base.tolist()}
    rows = bb.scatter_rows(near_all)
    g_lib = np.asarray(parts["G"])[np.ix_(rows, rows)]
    tab_world, exact_world = world(held["table"]), world(held["exact"])
    result["plumbing_table_vs_library"] = float(np.max(np.abs(tab_world - g_lib)) / np.max(np.abs(g_lib)))
    # the near blocks replaced by the exact series
    replaced = jnp.asarray(parts["G"]).at[rows[:, None], rows[None, :]].set(jnp.asarray(exact_world))
    r = level.case.solve(parts, bb.FROZEN_BLOB, True, wall_matrix=replaced)
    del replaced
    result["exact_near"] = summary_of(r, base, axial)
    # the regularised image's correction, from this script's Blake at the pairs (reach 0.6 mm), and the grid's code
    for reach in REACHES_M:
        inside = np.flatnonzero(heights < reach)
        sub = near_all[inside]
        blocks = held[f"blake_{bb.FROZEN_BLOB}"] - held["blake0"]
        patch = world(blocks[np.ix_(inside, inside)])
        r = level.solve_patched(parts, bb.FROZEN_BLOB, bb.scatter_rows(sub), jnp.asarray(patch))
        result[f"regimage_reach_{reach}"] = summary_of(r, base, axial) | {"near": int(len(sub))}
    patch, prow, stats = level.correction(parts, bb.FROZEN_BLOB)
    r = level.solve_patched(parts, bb.FROZEN_BLOB, prow, patch)
    result["regimage_grid_code"] = summary_of(r, base, axial) | stats
    inside = np.flatnonzero(heights < bb.NEAR_WALL_M)
    mine = world((held[f"blake_{bb.FROZEN_BLOB}"] - held["blake0"])[np.ix_(inside, inside)])
    result["control_grid_code_vs_this_script"] = float(np.max(np.abs(np.asarray(patch) - mine))
                                                       / np.max(np.abs(mine)))
    # the margin to indefiniteness: the wall matrix's scale at which the resistance first loses definiteness (M1's
    # scan, by bisection to 0.0006), for the library, the exact near blocks and the regularised image
    arms = {"library": jnp.asarray(parts["G"]),
            "exact_near": jnp.asarray(parts["G"]).at[rows[:, None], rows[None, :]].set(jnp.asarray(exact_world)),
            "regimage": jnp.asarray(parts["G"]).at[prow[:, None], prow[None, :]].add(patch)}
    result["margin"] = {}
    for name, wall in arms.items():
        result["margin"][name] = onset(level, parts, wall)
        print(name, result["margin"][name], flush=True)
    del arms
    result["seconds"] = time.time() - started
    (out / f"solve_phase{phase}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


MARGIN_TOP, MARGIN_STEPS = 1.08, 7


def onset(level: Any, parts: dict[str, Any], wall: Any) -> dict[str, Any]:
    """The least scale of ``wall`` in ``[1, MARGIN_TOP]`` at which the resistance's smallest non-dimensional
    eigenvalue is not positive, by bisection."""
    import bem
    import bem_budget as bb

    def least(scale: float) -> float:
        return bem.smallest_eigenvalue(level.case.solve(parts, bb.FROZEN_BLOB, True, wall_matrix=scale * wall))

    lo, hi = 1.0, MARGIN_TOP
    at_lo, at_hi = least(lo), least(hi)
    if at_lo <= 0.0:
        return {"onset": 1.0, "indefinite_at_nominal": True}
    if at_hi > 0.0:
        return {"onset": None, "beyond": MARGIN_TOP, "least_at_top": at_hi}
    for _ in range(MARGIN_STEPS):
        mid = 0.5 * (lo + hi)
        if least(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return {"onset": hi, "last_definite": lo, "resolution": hi - lo}


def summary_of(r: np.ndarray, base: np.ndarray, axial: Any) -> dict[str, Any]:
    import bem

    return {"axial_mm_s": axial(r), "axial_change_relative": axial(r) / axial(base) - 1.0,
            "min_eigenvalue_nd": bem.smallest_eigenvalue(r),
            "resistance_change_max_relative": float(np.max(np.abs(r - base) / np.abs(np.where(base == 0, 1, base))
                                                           * (np.abs(base) > 1e-6 * np.abs(base).max()))),
            "R_xx_change": float(r[0, 0] / base[0, 0] - 1.0), "R_xO_change": float(r[0, 3] / base[0, 3] - 1.0),
            "R_OO_change": float(r[3, 3] / base[3, 3] - 1.0), "resistance_si": r.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", action="store_true")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--spacing", type=float, default=0.16)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.pairs:
        print(json.dumps(run_pairs(args.phase, Path(args.out), args.workers, args.spacing), indent=1, default=float))
    if args.solve:
        found = run_solve(args.phase, Path(args.out), args.spacing)
        print(json.dumps({k: v for k, v in found.items() if not isinstance(v, dict)}
                         | {k: {kk: vv for kk, vv in v.items() if kk != "resistance_si"}
                            for k, v in found.items() if isinstance(v, dict)}, indent=1, default=float))


if __name__ == "__main__":
    main()
