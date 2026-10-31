"""M3 route 2, stage 2: the wall table's actual error E at *every* sample pair of operating point A (FL-9, closed surface
now ``screw`` 0.2.0, 0.178 mm, blob 0.5, 1/4 inch tube, 0.1 mm margin) at one phase, propagated through the solve
(``M3_ROUTE2_SCOPE.md`` option (a1)); ``m3_route2.py`` (stage 1: the pairs kernel and its controls) is imported, not edited.

Two programs, so that the heavy numpy part runs on every core with one thread each and the solve runs on all threads:

``--pairs``  for every unordered pair (the sample nearer the axis is the source; the self pairs included) the exact
             block (``m3_route2.exact_group``, the builder's per-source truncation at A's ``rho_max``, scale
             ``FACTOR``) and the table's blended read (``m3_route2.table_pair``, the layout the plumbing control
             certified), a pool of spawned workers over (source, target-chunk) tasks, the most expensive first. Also the
             truncation control (a few sources' pairs at ``FACTOR`` and at 3.0, the scale stage 1 used for E) and the
             knot control (``m3_route2.control_knots``) on this sampling. A deadline saves what is done for ``--resume``.
``--solve``  assembles ``G_exact`` and the table's ``G`` in ``corrected.matrix``'s layout (tube axes turned to the world
             by the frame's rows, over ``R mu``), compares the table's read with the library's matrix at **all** entries
             (the plumbing control), solves for the resistance with the library's G, the layout's G, G_exact, and the
             planted errors, and writes E's statistics per stratum, the axial speed (``-R[0,3] 2 pi f / R[0,0]``, ``op.speeds``'
             form, the one ``m3_wall_read.Study.axial`` computes) and the smallest non-dimensional eigenvalue of each.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu MIME_ASSETS_DIR=... python m3_stage2.py --pairs --work DIR --workers 32
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu MIME_ASSETS_DIR=... python m3_stage2.py --solve --work DIR --out DIR
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

SOLVE = "--solve" in sys.argv
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if SOLVE:  # numpy first, so that its BLAS keeps every thread; route 2's import then sets one thread for the rest
    import numpy as _np  # noqa: F401
import numpy as np  # noqa: E402

import m3_route2 as r2  # noqa: E402  (sets every numerical library to one thread at import)

if SOLVE:
    for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.pop(_variable, None)

#: The truncation scale (the table's own, 1.5; stage 1's E used 3.0, twice it) and the target chunk of a task.
FACTOR = 1.5
CONTROL_FACTOR = 3.0
CHUNK = 1000
RHO_MAX = r2.RHO_MAX_A
SEED = 20261031
#: The planted smooth error: x(1 + PLANT) over the first radial interval (a model-2 field, ``G_ij (phi_i + phi_j) / 2``).
PLANT = 0.005
TRUNCATION_SOURCES, TRUNCATION_PARTNERS = 24, 40
XLA_ONE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


# ---- samples --------------------------------------------------------------------------------------------------------
def build_samples(phase: float = 0.0, spacing_mm: float | None = None) -> r2.Samples:
    """A's samples at ``phase`` (``r2.phase_zero_samples`` with the phase and, for a local test, the spacing as arguments)."""
    import jax.numpy as jnp

    import bem
    import operating_point as op
    from mime.core import quaternion
    from mime.nodes.hydro.bem import wall_correction_named

    rung = bem.Rung("closed", spacing_mm=op.SPACING_MM if spacing_mm is None else spacing_mm)
    node = rung.node
    p = node._step_constants(rung.leaves[op.DRAG_NODE])  # noqa: SLF001
    held = node.samples(p)
    pose = rung.pose(phase)
    points = np.asarray(pose[:3] + jnp.matmul(held.points, quaternion.rotation_matrix(pose[3:]).T, precision="highest"))
    frame = np.asarray(p["wall_frame"], dtype=np.float64).reshape(3, 3)
    centre = np.asarray(p["wall_centre"], dtype=np.float64)
    radius = float(p["wall_radius"])
    wall = wall_correction_named(node.constants()["wall_kind"], node.constants()["wall_kind_version"])
    parameters = {name: p[f"wall_{name}"] for name in wall.parameters}
    local = (points - centre) @ frame.T / radius
    return r2.Samples(local, points, frame, centre, radius, float(p["viscosity"]), wall, parameters)


class Layout:
    """The unordered pairs in task order: pair ``k`` of source rank ``r`` is ``(target order[r + j], source order[r])``,
    ``j = 0`` the self pair, stored at ``base[r] + j``."""

    def __init__(self, rho: np.ndarray) -> None:
        self.order = np.argsort(rho, kind="stable")
        self.n = len(rho)
        self.sizes = self.n - np.arange(self.n)
        self.base = np.concatenate([[0], np.cumsum(self.sizes)])
        self.count = int(self.base[-1])

    def pair_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(target, source)`` sample indices of every pair, in storage order."""
        source = np.repeat(self.order, self.sizes)
        target = np.concatenate([self.order[r:] for r in range(self.n)])
        return target.astype(np.int32), source.astype(np.int32)

    def tasks(self, chunk: int = CHUNK) -> list[tuple[str, int, int, int]]:
        """``("pair", rank, lo, hi)``: source ``order[rank]`` against ``order[lo:hi]``; highest source radius first (the
        costliest)."""
        out = [("pair", r, lo, min(lo + chunk, self.n)) for r in range(self.n) for lo in range(r, self.n, chunk)]
        out.sort(key=lambda task: -task[1])  # rank grows with the source's radius
        return out


# ---- workers ---------------------------------------------------------------------------------------------------------
_W: dict[str, Any] = {}


def _init(rho: np.ndarray, phi: np.ndarray, z: np.ndarray, order: np.ndarray) -> None:
    os.environ["XLA_FLAGS"] = XLA_ONE_THREAD
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "1"
    _W.update(rho=rho, phi=phi, z=z, order=order, table=None)


def _pairs_of(source: int, targets: np.ndarray) -> dict[str, np.ndarray]:
    rho, phi, z = _W["rho"], _W["phi"], _W["z"]
    count = len(targets)
    return {"rho_t": rho[targets], "rho_s": np.full(count, rho[source]), "phi_t": phi[targets],
            "phi_s": np.full(count, phi[source]), "dphi": phi[targets] - phi[source], "dz": z[targets] - z[source]}


def _table() -> Any:
    if _W["table"] is None:
        _W["table"] = r2.load_table()
    return _W["table"]


def _task(task: tuple[Any, ...]) -> tuple[Any, ...]:
    started = time.perf_counter()
    kind = task[0]
    order = _W["order"]
    if kind == "pair":
        _, rank, lo, hi = task
        source, targets = int(order[rank]), order[lo:hi]
        pairs = _pairs_of(source, targets)
        rho_s = float(_W["rho"][source])
        block = r2.exact_group(rho_s, pairs["rho_t"], pairs["dphi"], pairs["dz"],
                               r2.pair_truncation(rho_s, factor=FACTOR, readable=RHO_MAX))
        cart = r2.cartesian_block(block, pairs["phi_t"], pairs["phi_s"])
        table = r2.table_pair(_table(), pairs)["blended"]
        return task, cart, table, time.perf_counter() - started
    _, rank, picks = task  # the truncation control: the same pairs at FACTOR and at CONTROL_FACTOR
    source, targets = int(order[rank]), np.asarray(picks)
    pairs = _pairs_of(source, targets)
    rho_s = float(_W["rho"][source])
    carts = []
    for factor in (FACTOR, CONTROL_FACTOR):
        block = r2.exact_group(rho_s, pairs["rho_t"], pairs["dphi"], pairs["dz"],
                               r2.pair_truncation(rho_s, factor=factor, readable=RHO_MAX))
        carts.append(r2.cartesian_block(block, pairs["phi_t"], pairs["phi_s"]))
    table = r2.table_pair(_table(), pairs)["blended"]
    return task, np.stack(carts), table, time.perf_counter() - started


def truncation_tasks(layout: Layout, rng: np.random.Generator) -> list[tuple[str, int, list[int]]]:
    ranks = np.unique(np.linspace(0, layout.n - 2, TRUNCATION_SOURCES).astype(int))
    return [("trunc", int(r), [int(i) for i in rng.choice(layout.order[r + 1:], min(TRUNCATION_PARTNERS, layout.n - 1 - r),
                                                          replace=False)]) for r in ranks]


# ---- --pairs ---------------------------------------------------------------------------------------------------------
def run_pairs(work: Path, workers: int, deadline_s: float | None, resume: bool, spacing: float | None,
              estimate: int = 0) -> dict[str, Any]:
    import multiprocessing as mp

    started = time.time()
    work.mkdir(parents=True, exist_ok=True)
    print("building A's samples", flush=True)
    samples = build_samples(0.0, spacing)
    layout = Layout(samples.rho)
    print(f"{layout.n} samples, {layout.count} unordered pairs (self included), rho {samples.rho.min():.4f} to "
          f"{samples.rho.max():.4f}, readable {RHO_MAX}", flush=True)
    assert samples.rho.max() <= RHO_MAX + 1e-12, "a sample lies beyond the truncation rule's readable radius"
    exact = np.full((layout.count, 3, 3), np.nan)
    table = np.full((layout.count, 3, 3), np.nan)
    tasks = layout.tasks()
    done = np.zeros(len(tasks), dtype=bool)
    index_of = {t: i for i, t in enumerate(tasks)}
    if resume and (work / "partial.npz").exists():
        held = np.load(work / "partial.npz")
        exact, table, done = held["exact"], held["table"], held["done"]
        print(f"resumed: {int(done.sum())} of {len(tasks)} tasks done", flush=True)
    todo = [t for i, t in enumerate(tasks) if not done[i]]
    if estimate:  # timing only: a random sample of tasks on this process, scaled by the weights
        rng = np.random.default_rng(SEED + 5)
        _init(samples.rho, samples.phi, samples.z, layout.order)
        pick = rng.choice(len(todo), min(estimate, len(todo)), replace=False)
        times = np.array([_task(todo[i])[3] for i in pick])
        total = float(times.mean() * len(todo))
        out = {"tasks": len(todo), "sampled": len(pick), "estimated_core_hours": total / 3600.0,
               "sample_seconds": times.tolist()}
        print(json.dumps(out), flush=True)
        return out
    trunc = truncation_tasks(layout, np.random.default_rng(SEED + 11))
    trunc_results: list[tuple[Any, ...]] = []
    cpu_seconds, finished = 0.0, 0
    context = mp.get_context("spawn")
    complete = True
    with context.Pool(workers, initializer=_init, initargs=(samples.rho, samples.phi, samples.z, layout.order)) as pool:
        for result in pool.imap_unordered(_task, trunc + todo, chunksize=1):
            task, cart, tab, seconds = result
            cpu_seconds += seconds
            if task[0] == "trunc":
                trunc_results.append(result)
                continue
            _, rank, lo, hi = task
            start = int(layout.base[rank]) + (lo - rank)
            exact[start:start + (hi - lo)] = cart
            table[start:start + (hi - lo)] = tab
            done[index_of[task]] = True
            finished += 1
            if finished % 500 == 0:
                print(f"{finished}/{len(todo)} tasks, {time.time() - started:.0f} s, cpu {cpu_seconds / 3600:.2f} h", flush=True)
            if deadline_s is not None and time.time() - started > deadline_s:
                print("deadline: stopping the pool", flush=True)
                complete = False
                pool.terminate()
                break
    seconds_pairs = time.time() - started
    summary: dict[str, Any] = {"samples": layout.n, "pairs": layout.count, "tasks": len(tasks),
                               "tasks_done": int(done.sum()), "workers": workers, "factor": FACTOR,
                               "rho_max_rule": RHO_MAX, "rho_range": [float(samples.rho.min()), float(samples.rho.max())],
                               "wall_seconds_pairs": seconds_pairs, "cpu_hours_pairs": cpu_seconds / 3600.0,
                               "complete": bool(done.all())}
    if not done.all():
        np.savez(work / "partial.npz", exact=exact, table=table, done=done)
        (work / "pairs_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print("saved partial", flush=True)
        return summary
    np.save(work / "exact.npy", exact)
    np.save(work / "table.npy", table)
    summary["truncation_control"] = truncation_control(samples, trunc_results)
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    summary["knots_control"] = r2.control_knots(samples, r2.load_table(), rng)
    summary["knots_control"]["seconds"] = time.perf_counter() - t0
    summary["wall_seconds"] = time.time() - started
    (work / "pairs_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"pairs done in {summary['wall_seconds']:.0f} s", flush=True)
    return summary


def truncation_control(samples: r2.Samples, results: list[tuple[Any, ...]]) -> dict[str, Any]:
    """The truncation's change against E on the control pairs: the exact block at ``FACTOR`` against ``CONTROL_FACTOR``
    (stage 1's scale), over the block's largest entry, beside E (the table's read against the ``CONTROL_FACTOR``
    block). The criterion stage 1 used: E at least ten times the change, at the median and the 95th percentile."""
    change, e_found = [], []
    for task, carts, tab, _ in results:
        change.append(r2.relative(carts[0] - carts[1], carts[1]))
        e_found.append(r2.relative(tab - carts[1], carts[1]))
    change, e_found = np.concatenate(change), np.concatenate(e_found)
    ratio = {q: float(np.quantile(e_found, q) / max(np.quantile(change, q), 1e-300)) for q in (0.5, 0.95)}
    return {"pairs": int(len(change)), "change_median": float(np.median(change)), "change_p95": float(np.quantile(change, 0.95)),
            "change_max": float(change.max()), "E_median": float(np.median(e_found)),
            "E_p95": float(np.quantile(e_found, 0.95)), "E_over_change": ratio,
            "per_source_change_max": [float(r2.relative(c[0] - c[1], c[1]).max()) for _, c, _, _ in results],
            "pass": bool(min(ratio.values()) > r2.TOLERANCE["truncation_ratio"])}


# ---- --solve ---------------------------------------------------------------------------------------------------------
def assemble(layout: Layout, blocks: np.ndarray, transpose_mirror: bool = True) -> np.ndarray:
    """``(n, n, 3, 3)``: block ``[target, source]`` of every ordered pair from the unordered pairs' blocks (the mirrored
    pair transposed; the self block as computed)."""
    n = layout.n
    full = np.empty((n, n, 3, 3))
    for r in range(n):
        targets = layout.order[r:]
        source = layout.order[r]
        block = blocks[layout.base[r]:layout.base[r] + len(targets)]
        full[source, targets] = np.swapaxes(block, -1, -2) if transpose_mirror else block
        full[targets, source] = block
    return full


def to_world(full: np.ndarray, samples: r2.Samples) -> np.ndarray:
    """``corrected.matrix``'s layout and units: tube axes turned to the world (``F^T G F``), over ``R mu``; (3n, 3n)."""
    n = full.shape[0]
    frame = samples.frame
    world = np.matmul(np.matmul(frame.T, full), frame)
    return world.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) / (samples.radius * samples.viscosity)


def strata_of(rho_t: np.ndarray, rho_s: np.ndarray) -> np.ndarray:
    """0 inner (a sample at rho < 0.157), 1 outer (a sample above 0.9, none inside), 2 other."""
    inner = (rho_t < r2.INNER_EDGE) | (rho_s < r2.INNER_EDGE)
    outer = ~inner & ((rho_t > r2.OUTER_EDGE) | (rho_s > r2.OUTER_EDGE))
    return np.where(inner, 0, np.where(outer, 1, 2))


def error_statistics(samples: r2.Samples, layout: Layout, exact: np.ndarray, table: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    target, source = layout.pair_arrays()
    e = r2.relative(table - exact, exact)
    stratum = strata_of(samples.rho[target], samples.rho[source])
    out: dict[str, Any] = {}
    groups = {"all": np.ones(len(e), bool), "diagonal": target == source,
              "inner": (stratum == 0), "outer": (stratum == 1), "other": (stratum == 2)}
    for name, pick in groups.items():
        values = e[pick]
        out[name] = {"pairs": int(pick.sum()), "mean": float(values.mean()), "median": float(np.median(values)),
                     "p95": float(np.quantile(values, 0.95)), "p99": float(np.quantile(values, 0.99)),
                     "p999": float(np.quantile(values, 0.999)), "max": float(values.max()),
                     "share_above_0.002": float(np.mean(values > 0.002)),
                     "share_above_0.0005": float(np.mean(values > 0.0005))}
    worst = np.argsort(e)[-5:][::-1]
    out["largest"] = [{"E": float(e[k]), "rho_target": float(samples.rho[target[k]]), "rho_source": float(samples.rho[source[k]]),
                       "dz": float(samples.z[target[k]] - samples.z[source[k]])} for k in worst]
    return out, e


def run_solve(work: Path, out: Path, spacing: float | None, phase: float = 0.0) -> dict[str, Any]:
    import jax.numpy as jnp

    import bem
    import m3_wall_read as m3
    import operating_point as op

    op.require_float64()
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    samples = build_samples(phase, spacing)
    layout = Layout(samples.rho)
    exact = np.load(work / "exact.npy")
    table = np.load(work / "table.npy")
    assert exact.shape[0] == layout.count and not np.isnan(exact).any() and not np.isnan(table).any()
    study = m3.Study(op.SPACING_MM if spacing is None else spacing, (phase,))
    parts = study.parts(phase)
    library = np.asarray(parts["G"], dtype=np.float64)
    result: dict[str, Any] = {"phase": phase, "samples": layout.n, "pairs": layout.count, "environment": op.environment()}

    def solve(G: np.ndarray) -> dict[str, Any]:
        resistance = study.read(study._solve(parts, jnp.asarray(G)))  # noqa: SLF001
        return {**study.row(resistance), "resistance": resistance}

    def public(entry: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in entry.items() if k != "resistance"}

    # E over all pairs
    stats, e_pairs = error_statistics(samples, layout, exact, table)
    result["E_blend"] = stats
    np.savez_compressed(out / "stage2_E.npz", E_blend=e_pairs.astype(np.float32))
    # the matrices
    G_exact = to_world(assemble(layout, exact), samples)
    G_mine = to_world(assemble(layout, table), samples)
    scale = float(np.abs(library).max())
    result["matrices"] = {
        "plumbing_all_entries_max_over_library_max": float(np.abs(G_mine - library).max() / scale),
        "plumbing_entries": int(library.size),
        "library_asymmetry_over_max": float(np.abs(library - library.T).max() / scale),
        "layout_G_asymmetry_over_max": float(np.abs(G_mine - G_mine.T).max() / scale),
        "exact_G_asymmetry_over_max": float(np.abs(G_exact - G_exact.T).max() / scale),
        "exact_minus_library_max_over_library_max": float(np.abs(G_exact - library).max() / scale),
        "exact_minus_library_fro_over_library_fro": float(np.linalg.norm(G_exact - library) / np.linalg.norm(library)),
        "exact_minus_layout_max_over_library_max": float(np.abs(G_exact - G_mine).max() / scale)}
    # the comparison's planted fault (must miss): the mirrored blocks left untransposed
    wrong = to_world(assemble(layout, table, transpose_mirror=False), samples)
    result["matrices"]["plumbing_planted_untransposed_mirror"] = float(np.abs(wrong - library).max() / scale)
    # the solves
    lib = solve(library)
    mine = solve(G_mine)
    ex = solve(G_exact)
    result["solves"] = {"library_G": public(lib), "layout_G": public(mine), "exact_G": public(ex)}
    speed = lambda row: row["axial"]  # noqa: E731
    result["speed"] = {
        "library_m_s": speed(lib), "layout_m_s": speed(mine), "exact_m_s": speed(ex),
        "layout_vs_library": speed(mine) / speed(lib) - 1.0,
        "exact_vs_library": speed(ex) / speed(lib) - 1.0,
        "min_eigenvalue_nd": {"library": lib["min_eigenvalue_nd"], "layout": mine["min_eigenvalue_nd"],
                              "exact": ex["min_eigenvalue_nd"],
                              "exact_over_library": ex["min_eigenvalue_nd"] / lib["min_eigenvalue_nd"] - 1.0},
        "exact_positive_definite": bool(ex["min_eigenvalue_nd"] > 0), "exact_finite": bool(ex["finite"])}
    # the floor at this phase: 1 to 3 ulp changes of the pose's offset (the cap fix's convention), on the library's G
    ulp = float(np.spacing(study.rung.offset))
    floors = []
    for k in m3.FLOOR_ULPS:
        p = study.parts(phase, study.rung.offset + k * ulp)
        r = study.read(study._solve(p, p["G"]))  # noqa: SLF001
        floors.append(study.axial(r) / speed(lib) - 1.0)
    result["speed"]["floor_changes"] = floors
    result["speed"]["floor_half_width"] = float(np.abs(floors).max())
    # plants: x(1 + PLANT) over the first radial interval (the model-2 form), and a uniform one (a large effect)
    target, source = layout.pair_arrays()
    plants: dict[str, Any] = {}
    kinds = {"first_interval": np.where(samples.rho < r2.INNER_EDGE, PLANT, 0.0),
             "uniform": np.full(layout.n, PLANT),
             "first_interval_wrong_samples": np.where(samples.rho < r2.INNER_EDGE, PLANT, 0.0)[::-1].copy()}
    direct_of = {"first_interval": kinds["first_interval"], "uniform": kinds["uniform"],
                 "first_interval_wrong_samples": kinds["first_interval"]}
    for name, phi in kinds.items():
        factor = 1.0 + 0.5 * (phi[target] + phi[source])
        planted = solve(to_world(assemble(layout, exact * factor[:, None, None]), samples))
        direct_phi = direct_of[name]
        delta = np.asarray(m3.smooth_delta(jnp.asarray(library), jnp.asarray(direct_phi)))
        direct = solve(library + delta)
        pipe = speed(planted) / speed(ex) - 1.0
        dire = speed(direct) / speed(lib) - 1.0
        plants[name] = {"pipeline_change": pipe, "direct_scaled_solve_change": dire,
                        "difference_over_direct": (pipe - dire) / dire if dire != 0 else None,
                        "samples_planted": int((phi != 0).sum()), "min_eigenvalue_nd": planted["min_eigenvalue_nd"]}
    tolerance = 0.01
    plants["tolerance_difference_over_direct"] = tolerance
    plants["recovered"] = {n: bool(abs(plants[n]["difference_over_direct"]) <= tolerance) for n in ("first_interval", "uniform")}
    plants["wrong_samples_detected"] = bool(abs(plants["first_interval_wrong_samples"]["difference_over_direct"]) > 10 * tolerance)
    plants["effect_above_floor"] = {n: bool(abs(plants[n]["direct_scaled_solve_change"]) > 3 * result["speed"]["floor_half_width"])
                                    for n in ("first_interval", "uniform")}
    result["plants"] = plants
    result["stage3_needed"] = bool(abs(result["speed"]["exact_vs_library"]) > 0.001 or not result["speed"]["exact_positive_definite"])
    result["seconds"] = time.time() - started
    (out / "stage2_solve.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", action="store_true")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--deadline-s", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--spacing-mm", type=float, default=None, help="a coarse mesh for a local test")
    parser.add_argument("--estimate", type=int, default=0, help="time this many random tasks on this process and extrapolate")
    args = parser.parse_args()
    if args.pairs:
        run_pairs(Path(args.work), args.workers, args.deadline_s, args.resume, args.spacing_mm, args.estimate)
    if args.solve:
        result = run_solve(Path(args.work), Path(args.out or args.work), args.spacing_mm)
        print(json.dumps({k: result[k] for k in ("speed", "plants", "matrices")}, indent=1, default=float))


if __name__ == "__main__":
    main()
