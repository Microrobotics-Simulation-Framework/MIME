"""The full unit-tube table (MIME-ANO-117): its knots and truncation rule, the specs of the jobs Modal runs, the 4D read
on blocks of knots, the table's assembly in the form ``mime.nodes.hydro.bem._tube.tube_table_from`` reads, and its
gate.

    python full.py specs-blocks SPECS.json
    python full.py blocks RESULTS_DIR          # the 4D read per candidate grid (JAX, CPU)
    python full.py specs-build GRID SPECS.json
    python full.py assemble GRID RESULTS_DIR TABLE.npz [--as-built]
    python full.py gate TABLE.npz DEFECTIVE.npz RESULTS_DIR

The approach is the image itself (the slice's approach a), readable to ``R_MAX`` of the radius. The truncation of a
source radius scales with ``H = (1 - rho_s) + (1 - R_MAX)``: modes ``10 / H``, Gauss nodes ``64 / H`` and ``k_max = 12 /
H`` (the slice's 0.1% level), with floors for the sources far from the wall. The knots are geometric from the
readable radius and from ``dphi = dz = 0``, ``step * (x + 2 (1 - R_MAX))`` apart."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import build

R_MAX = 0.997
NEAR = 2.0 * (1.0 - R_MAX)
AXIAL_REACH = 5.0
#: Candidate grids, steps (radius, angle, axial): a coarse control, the slice's per-axis 1% picks, and two finer.
GRIDS = {"control": (0.4, 0.28, 0.28), "G0": (0.28, 0.19, 0.19), "G1": (0.19, 0.13, 0.13), "G2": (0.13, 0.09, 0.09)}
RULE = {"modes": 10.0, "wavenumbers": 64.0, "reach": 12.0, "floor_modes": 24, "floor_wavenumbers": 256,
        "floor_k_max": 24.0}


def knots(grid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_r, step_a, step_z = GRIDS[grid]
    rho = R_MAX - build.graded_knots(R_MAX, NEAR, step_r)[::-1]
    rho[0], rho[-1] = 0.0, R_MAX
    angle = build.graded_knots(math.pi, NEAR, step_a)
    angle[-1] = math.pi
    along = build.graded_knots(AXIAL_REACH, NEAR, step_z)
    along[-1] = AXIAL_REACH
    return rho, angle, along


def truncation_for(source: float, finer: bool = False) -> build.Truncation:
    h = (1.0 - source) + (1.0 - R_MAX)
    scale = (1.25, 2.0, 1.5) if finer else (1.0, 1.0, 1.0)
    modes = max(int(math.ceil(scale[0] * RULE["modes"] / h)), RULE["floor_modes"])
    nodes = max(int(math.ceil(scale[1] * RULE["wavenumbers"] / h)), RULE["floor_wavenumbers"])
    k_max = max(scale[2] * RULE["reach"] / h, RULE["floor_k_max"])
    angles = 2 * int(math.ceil((modes + 25.0 / max(1.0 - source, 1e-3)) / 2.0))
    return build.Truncation(modes, nodes, k_max, max(angles, 2 * modes + 2))


#: With ``SINGLE`` set, every source radius takes the truncation the readable radius needs (the largest any source
#: needs), so the table's two halves G(x, y) and G(y, x) come from one series and reciprocity holds to rounding.
SINGLE = False


def spec(key: str, source: float, rho, phi, z, *, finer: bool = False, as_built: bool = False) -> dict:
    truncation = truncation_for(R_MAX if SINGLE else source, finer)
    return {"key": key, "source": float(source), "rho": [float(v) for v in rho], "phi": [float(v) for v in phi],
            "z": [float(v) for v in z], "truncation": truncation.__dict__, "as_built": as_built}


def _cost(item: dict) -> float:
    t = item["truncation"]
    return t["wavenumbers"] * (t["angles"] + len(item["rho"]) * t["modes"])


def random_reference(key: str, rng: np.random.Generator, sources: np.ndarray, rho_range, small: bool,
                     count: int = 4) -> list[dict]:
    """Random off-knot points: per source radius a tensor of random target radii, angles and offsets; with ``small``
    half the angles and offsets are drawn log-uniform in [1e-4, 0.1] (the hard corner)."""
    out = []
    for i, source in enumerate(sources):
        rho = rng.uniform(*rho_range, count)
        if small:
            half = count // 2
            phi = np.concatenate([10 ** rng.uniform(-4, -1, half), rng.uniform(0, math.pi, count - half)])
            z = np.concatenate([10 ** rng.uniform(-4, -1, half), rng.uniform(0, AXIAL_REACH, count - half)])
        else:
            phi, z = rng.uniform(0, math.pi, count), rng.uniform(0, AXIAL_REACH, count)
        out.append(spec(f"{key}_{i}", source, np.sort(rho), np.sort(phi), np.sort(z)))
    return out


BLOCKS = {"wall": {"knots": (0.95, R_MAX), "draw": (0.99, R_MAX), "small": True, "sources": 16},
          "interior": {"knots": (0.05, 0.95), "draw": (0.35, 0.65), "small": False, "sources": 12}}


def block_specs() -> list[dict]:
    items = []
    for grid in GRIDS:
        rho, angle, along = knots(grid)
        for block, setting in BLOCKS.items():
            low, high = setting["knots"]
            inside = rho[(rho >= low) & (rho <= high)]
            for source in inside:
                items.append(spec(f"block_{grid}_{block}_{source:.12f}", source, inside, angle, along))
    rng = np.random.default_rng(29)
    for block, setting in BLOCKS.items():
        sources = rng.uniform(*setting["draw"], setting["sources"])
        items += random_reference(f"reference_{block}", rng, sources, setting["draw"], setting["small"])
    rho, angle, along = knots("G1")
    for source in (0.0, 0.5, 0.9, 0.97, 0.99, R_MAX):
        for finer in (False, True):
            items.append(spec(f"truncation_{source}_{'finer' if finer else 'rule'}", source, rho, angle[::4],
                              along[::4], finer=finer))
    return sorted(items, key=_cost, reverse=True)


def build_specs(grid: str) -> list[dict]:
    rho, angle, along = knots(grid)
    items = [spec(f"table_{i:03d}", source, rho, angle, along) for i, source in enumerate(rho)]
    items += [spec(f"defective_{i:03d}", source, rho, angle, along, as_built=True) for i, source in enumerate(rho)]
    for run in range(5):
        rng = np.random.default_rng(100 + run)
        sources = np.concatenate([rng.uniform(0.0, 0.97, 4), rng.uniform(0.97, R_MAX, 4)])
        near = random_reference(f"gate_{run}_wall", rng, sources[4:], (0.97, R_MAX), True, count=3)
        far = random_reference(f"gate_{run}_any", rng, sources[:4], (0.0, R_MAX), False, count=3)
        items += near + far
    return sorted(items, key=_cost, reverse=True)


def load(results: Path, key: str) -> tuple[np.ndarray, dict]:
    held = np.load(results / f"{key}.npz")
    return held["values"], json.loads(str(held["spec"]))


def read_at(table, rho_t, rho_s, phi, z) -> np.ndarray:
    from mime.core import jnp
    from mime.core.tables import evaluate
    from mime.nodes.hydro.bem._tube import TABLE

    value = evaluate(table, jnp.asarray(rho_t), jnp.asarray(rho_s), jnp.asarray(phi), jnp.asarray(z), owner=TABLE)
    return np.asarray(value.value).reshape(len(rho_t), 3, 3)


def reference_points(results: Path, prefix: str) -> tuple[np.ndarray, ...]:
    """The random points of every reference job starting with ``prefix``, flattened: (rho_t, rho_s, phi, z, values)."""
    rows = []
    for path in sorted(results.glob(f"{prefix}*.npz")):
        values, item = load(results, path.stem)
        rt, ph, zz = np.meshgrid(item["rho"], item["phi"], item["z"], indexing="ij")
        rows.append((rt.ravel(), np.full(rt.size, item["source"]), ph.ravel(), zz.ravel(),
                     values.reshape(-1, 3, 3)))
    return tuple(np.concatenate([row[i] for row in rows]) for i in range(5))


def relative(read: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.max(np.abs(read - reference), axis=(-2, -1)) / np.max(np.abs(reference), axis=(-2, -1))


def blocks(results: Path) -> dict:
    from mime.nodes.hydro.bem._tube import unit_tube_table

    out: dict = {}
    for grid in GRIDS:
        rho, angle, along = knots(grid)
        for block, setting in BLOCKS.items():
            low, high = setting["knots"]
            inside = rho[(rho >= low) & (rho <= high)]
            data = np.empty((len(inside), len(inside), len(angle), len(along), 3, 3))
            for j, source in enumerate(inside):
                data[:, j] = load(results, f"block_{grid}_{block}_{source:.12f}")[0]
            table = unit_tube_table(1.0, inside, angle, along, data, scheme="cubic")
            rt, rs, ph, zz, reference = reference_points(results, f"reference_{block}")
            error = relative(read_at(table, rt, rs, ph, zz), reference)
            worst = int(np.argmax(error))
            out[f"{grid}_{block}"] = {"max": float(error.max()), "p99": float(np.quantile(error, 0.99)),
                                      "median": float(np.median(error)), "points": int(error.size),
                                      "worst_at": [float(rt[worst]), float(rs[worst]), float(ph[worst]),
                                                   float(zz[worst])], "knots": [len(rho), len(angle), len(along)],
                                      "size_GB": len(rho) ** 2 * len(angle) * len(along) * 72 / 1e9}
            print(grid, block, out[f"{grid}_{block}"], flush=True)
    for source in (0.0, 0.5, 0.9, 0.97, 0.99, R_MAX):
        rule, _ = load(results, f"truncation_{source}_rule")
        finer, _ = load(results, f"truncation_{source}_finer")
        out[f"truncation_{source}"] = float(np.max(relative(rule, finer)))
        print("truncation", source, out[f"truncation_{source}"], flush=True)
    return out


def assemble(grid: str, results: Path, target: Path, *, as_built: bool, commit: str) -> None:
    rho, angle, along = knots(grid)
    prefix = "defective" if as_built else "table"
    data = np.empty((len(rho), len(rho), len(angle), len(along), 3, 3))
    truncations = []
    for i in range(len(rho)):
        values, item = load(results, f"{prefix}_{i:03d}")
        assert abs(item["source"] - rho[i]) < 1e-15 and item["as_built"] == as_built
        data[:, i] = values
        truncations.append(item["truncation"])
    if not np.all(np.isfinite(data)):
        raise SystemExit("the table holds a non-finite value")
    np.savez(target, R_cyl=np.float64(1.0), rho_grid=rho, dphi_grid=angle, dz_grid=along, data=data,
             readable_rho=np.array([0.0, R_MAX]), axial_reach=np.float64(AXIAL_REACH),
             modes=np.array([t["modes"] for t in truncations]),
             wavenumbers=np.array([t["wavenumbers"] for t in truncations]),
             k_max=np.array([t["k_max"] for t in truncations]), angles=np.array([t["angles"] for t in truncations]),
             builder="scripts/tube_table/build.py (MIME, branch v0.3-tube-table)", builder_commit=commit,
             truncation_rule=json.dumps(RULE), knot_rule=json.dumps({"near": NEAR, "steps": GRIDS[grid]}),
             single_truncation=json.dumps(truncations[0] if all(t == truncations[0] for t in truncations) else None),
             as_built_defect=np.bool_(as_built), anomaly="MIME-ANO-117",
             layout="data[rho_t, rho_s, dphi, dz, row, col]: rows the velocity's cylindrical components at the "
                    "target, columns the force's at the source (radial, azimuthal, axial); unit tube, viscosity 1")


def _mfs(sources: list[np.ndarray], targets: np.ndarray) -> tuple[np.ndarray, float]:
    """``tests/oracles/test_tube_mfs.py``'s method of fundamental solutions: (case, force across the axis, target,
    velocity) and the wall residual at fresh points."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "oracles"))
    import test_tube_mfs as mfs
    import tube_fourier_bessel as fb
    from scipy import linalg

    stokeslets = mfs.ring(mfs.OUTSIDE, mfs.AROUND, mfs.ALONG)
    wall = mfs.ring(1.0, int(1.5 * mfs.AROUND), int(1.5 * mfs.ALONG), shift=0.25)
    rhs = np.column_stack([-fb.free_stokeslet(wall, s[None], 1.0)[:, 0, :, f].ravel() for s in sources for f in (0, 1)])
    strengths = linalg.lstsq(mfs.stokeslet_matrix(wall, stokeslets), rhs, lapack_driver="gelsy", check_finite=False)[0]
    rng = np.random.default_rng(5)
    angle, height = rng.uniform(0, 2 * math.pi, 200), rng.uniform(-1.5, 1.5, 200)
    fresh = np.stack([np.cos(angle), np.sin(angle), height], axis=1)
    residual = 0.0
    on_wall = mfs.stokeslet_matrix(fresh, stokeslets) @ strengths
    for c, s in enumerate(sources):
        for f in (0, 1):
            free = fb.free_stokeslet(fresh, s[None], 1.0)[:, 0, :, f]
            residual = max(residual, float(np.max(np.abs(on_wall[:, 2 * c + f].reshape(-1, 3) + free))
                                         / np.max(np.abs(free))))
    velocity = (mfs.stokeslet_matrix(targets, stokeslets) @ strengths).reshape(len(targets), 3, len(sources), 2)
    return velocity, residual


def cartesian(table_arrays, points: np.ndarray) -> np.ndarray:
    """``(N, N, 3, 3)``: the evaluator's own read (``tube_correction``) of the image between every pair of points."""
    from mime.core import jnp
    from mime.nodes.hydro.bem._tube import frame_across, tube_correction

    parameters = {"table": table_arrays, "frame": frame_across((0.0, 0.0, 1.0)), "radius": 1.0,
                  "centre": (0.0, 0.0, 0.0)}
    found = tube_correction(jnp.asarray(points), parameters, 1.0)
    n = len(points)
    return np.asarray(found.matrix).reshape(n, 3, n, 3).transpose(0, 2, 1, 3)


#: The gate's tolerances, declared before the full table existed: the read within 1%; the axis's coefficients at a
#: knot against Bohlin's 2.10444 and the series' 1.80436; reciprocity at 1e4 times the floor measured on a block table
#: (2.8e-12, the knots reciprocal and both radial axes alike); the read's divergence (its derivative's error) and the
#: MFS (its own 5e-3 at this resolution, plus the read) loosely.
TOLERANCES = {"read": 1e-2, "bohlin": 1e-5, "transverse": 1e-5, "axis_symmetry": 1e-6, "reciprocity": 1e-8,
              "knot_reciprocity": 1e-8, "divergence": 5e-2, "mfs": 1e-2}


def verdicts(entry: dict) -> dict:
    """Each measure against its tolerance, per run where it has runs."""
    t = TOLERANCES
    return {"read": [run["max"] <= t["read"] for run in entry["read_vs_builder"]],
            "bohlin": abs(entry["bohlin_axial"] - 2.10444) <= t["bohlin"],
            "transverse": abs(entry["transverse_axis"][0] - 1.80436) <= t["transverse"],
            "axis_symmetry": abs(entry["transverse_axis"][0] - entry["transverse_axis"][1])
            <= t["axis_symmetry"] * abs(entry["transverse_axis"][0]),
            "reciprocity": [run["reciprocity"] <= t["reciprocity"] for run in entry["identities"]],
            "knot_reciprocity": entry["knot_reciprocity"]["max"] <= t["knot_reciprocity"],
            "divergence": [run["divergence"] <= t["divergence"] for run in entry["identities"]],
            "mfs": entry["mfs"]["distance"] <= t["mfs"]}


def decay(path: Path) -> dict:
    """The image at the table's axial end against at dz = 0, and the tube's whole Green's function (the image plus the
    free Stokeslet) at the end against the free Stokeslet there, for three pairs at dphi = 0."""
    with np.load(path) as held:
        rho, data, along = held["rho_grid"], held["data"], held["dz_grid"]
    out = {}
    for i in (0, len(rho) // 2, len(rho) - 1):
        end, start = data[i, i, 0, -1], data[i, i, 0, 0]
        free = build.free_cylindrical(float(rho[i]), np.array([rho[i]]), np.array([0.0]), along[-1:])[0, 0, 0]
        out[f"{rho[i]:.3f}"] = {"image_end_over_start": float(np.max(np.abs(end)) / np.max(np.abs(start))),
                                "whole_over_free_at_end": float(np.max(np.abs(end + free)) / np.max(np.abs(free)))}
    return out


def gate(table_path: Path, defective_path: Path, results: Path) -> dict:
    from mime.nodes.hydro.bem._tube import tube_table_from

    out: dict = {}
    for label, path in (("table", table_path), ("defective", defective_path)):
        table = tube_table_from(str(path), scheme="cubic")
        arrays = table.arrays(np.float64)
        entry: dict = {}
        # the read against the builder at random off-knot points over the readable range, five runs
        runs = []
        for run in range(5):
            rt, rs, ph, zz, reference = reference_points(results, f"gate_{run}_")
            error = relative(read_at(table, rt, rs, ph, zz), reference)
            runs.append({"max": float(error.max()), "p99": float(np.quantile(error, 0.99)), "points": int(error.size)})
        entry["read_vs_builder"] = runs
        # Bohlin's and the transverse coefficient, on the axis (a knot)
        with np.load(path) as held:
            axis = held["data"][0, 0, 0, 0]
        entry["bohlin_axial"] = -6.0 * math.pi * float(axis[2, 2])
        entry["transverse_axis"] = [-6.0 * math.pi * float(axis[0, 0]), -6.0 * math.pi * float(axis[1, 1])]
        # identities of the evaluator's read at drawn interior points (5 runs)
        identity_runs = []
        for run in range(5):
            rng = np.random.default_rng(300 + run)
            radius, angle = rng.uniform(0.05, 0.85, 4), rng.uniform(-math.pi, math.pi, 4)
            points = np.stack([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(-1.0, 1.0, 4)], axis=1)
            g = cartesian(arrays, points)
            recip = float(np.max(np.abs(g - np.transpose(g, (1, 0, 3, 2)))) / np.max(np.abs(g)))
            step, worst = 0.02, 0.0
            for p in range(2):
                grad = np.zeros((3, 3, 3))
                for d in range(3):
                    e = np.zeros(3)
                    e[d] = step
                    stencil = np.vstack([points[3:4]] + [points[p:p + 1] + c * e for c in (-2, -1, 1, 2)])
                    read = cartesian(arrays, stencil)[1:, 0]  # the image at the stencil per force at point 3
                    grad[d] = (read[0] - 8 * read[1] + 8 * read[2] - read[3]) / (12 * step)
                div = sum(grad[d, d] for d in range(3))
                worst = max(worst, float(np.max(np.abs(div)) / np.max(np.abs(grad))))
            identity_runs.append({"reciprocity": recip, "divergence": worst})
        entry["identities"] = identity_runs
        # reciprocity at every knot of the file, not drawn interior points: the evaluator's blend makes its own reads
        # reciprocal, so the file is the only place the table's reciprocity can be measured
        with np.load(path) as held:
            miss = build.knot_reciprocity(held["data"])
            grids = (held["rho_grid"], held["rho_grid"], held["dphi_grid"], held["dz_grid"])
        corner = np.unravel_index(np.argmax(miss), miss.shape)
        entry["knot_reciprocity"] = {
            "max": float(miss.max()), "at (rho_t, rho_s, dphi, dz)": [float(g[i]) for g, i in zip(grids, corner)],
            "knots_over_tolerance": int(np.sum(miss > TOLERANCES["knot_reciprocity"])), "knots": int(miss.size)}
        # the MFS at a few points (forces across the axis)
        sources = [np.array([1e-6, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), np.array([0.2, -0.35, 0.3])]
        inside = np.array([[0.25, 0.1, 0.2], [-0.3, -0.35, -0.15], [0.05, 0.55, 0.4], [0.45, -0.2, 0.05]])
        velocity, residual = _mfs(sources, inside)
        worst = 0.0
        for c, s in enumerate(sources):
            g = cartesian(arrays, np.vstack([s[None], inside]))[1:, 0]  # image at the points per force at s
            for f in (0, 1):
                scale = float(np.max(np.abs(g[:, :, f])))
                worst = max(worst, float(np.max(np.abs(g[:, :, f] - velocity[:, :, c, f]))) / scale)
        entry["mfs"] = {"distance": worst, "mfs_wall_residual": residual}
        entry["verdicts"] = verdicts(entry)
        out[label] = entry
        print(label, json.dumps(entry), flush=True)
    out["decay"] = decay(table_path)
    out["tolerances"] = TOLERANCES
    return out


def main() -> None:
    command = sys.argv[1]
    if command == "specs-blocks":
        Path(sys.argv[2]).write_text(json.dumps(block_specs()))
    elif command == "specs-build":
        global SINGLE
        SINGLE = "--single" in sys.argv
        Path(sys.argv[3]).write_text(json.dumps(build_specs(sys.argv[2])))
    elif command == "blocks":
        print(json.dumps(blocks(Path(sys.argv[2])), indent=1))
    elif command == "assemble":
        assemble(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), as_built="--as-built" in sys.argv,
                 commit=sys.argv[5] if len(sys.argv) > 5 and not sys.argv[5].startswith("--") else "unknown")
    elif command == "gate":
        print(json.dumps(gate(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])), indent=1))


if __name__ == "__main__":
    main()
