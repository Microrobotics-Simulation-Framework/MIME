"""The Fourier-Bessel check of the 2026-05-21 wall tables, run by hand on the full tables after ``mime assets fetch``
(each about 900 MB in memory: one table a process, under the machine's heavy-job lock):

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/differential/check_wall_tables.py <asset name>...

appends one record per table to ``references/wall_table_check.jsonl``, and ``--summary`` writes
``references/wall_table_check.json`` from them. Not collected by pytest. What each record holds, and the tolerance
declared before the first run:

1. **Provenance**: the file's ``R_cyl`` against the radius in its name, its viscosity, and its grids against the
   v0.2 builder's (radii from 0.02 to 0.98 of ``R_cyl``, the angle on ``[0, pi]``, the axial offset to ``5 R_cyl``).
2. **The knots**: at drawn knots and at the corner nearest the wall, the table times ``R_cyl`` against the series
   as the v0.2 builder computed it (``tube_fourier_bessel.image(..., as_built=True)``) for the tube of radius 1 at
   the table's truncation (``modes=15, wavenumbers=80, angles=64``): that the values are what the builder computed,
   at that radius (so that every table is the unit tube's, scaled), and the component order (rows the target's, the
   first axis the target's radius); then against the series (without the builder's defect) at the table's truncation
   and at a finer one (``40, 160, 128``), binned by the larger of the pair's radii over the tube's.
3. **The lateral patch**: FL-9's lateral samples (the de Jongh kind's at a spacing of 0.9 mm, 135 of them), the
   body's radius 1.56 mm and the tube's ``R_cyl`` times it, centred on the tube's axis and,
   where it fits, moved 0.3 of the tube's radius off it: the wall's correction as the tube's wall kind reads it
   (``mime.nodes.hydro.bem.tube_correction``, cubic on every axis) against the series at the finer truncation at the
   same samples, and the 6 x 6 resistance with each, and the screw's swim along its axis per unit spin about it; the
   antisymmetric part of each correction (reciprocity); and the resistance in free space, for the size of the wall's
   effect.

**Declared before the first run** (``RELIED_ON``): a table is relied on at a configuration when every entry of the
resistance with its correction lies within 1% of that with the finer series, each judged on ``sqrt(R_ii R_jj)``,
beside which the method's own discretisation error at these counts is several per cent (MIME-ANO-116)."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

RECORDS = HERE / "references" / "wall_table_check.jsonl"
SUMMARY = HERE / "references" / "wall_table_check.json"
#: The tolerance on the resistance, declared before the first run.
RELIED_ON = 0.01
#: Knots drawn per table.
DRAWN = 60
#: The series at the tables' truncation, and at the finer one they are checked against.
AS_BUILT, FINER = (15, 80, 64), (40, 160, 128)
#: FL-9's radius, which the tables' lengths are in.
BODY_RADIUS = 1.56e-3
VISCOSITY = 1e-3
BINS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.99)


def provenance(name: str, held: dict[str, np.ndarray]) -> dict[str, object]:
    radius = float(held["R_cyl"])
    written = float(name.rsplit("_R", 1)[1])
    rho, angle, along = held["rho_grid"], held["dphi_grid"], held["dz_grid"]
    return {"R_cyl": radius, "radius_in_name": written, "name_matches": abs(radius - written) < 5e-4,
            "mu": float(held["mu"]), "radii": len(rho), "rho_over_R": [float(rho[0] / radius), float(rho[-1] / radius)],
            "angle": [float(angle[0]), float(angle[-1])], "angles": len(angle),
            "along_over_R": float(along[-1] / radius), "offsets": len(along),
            "grids_match_the_builder": bool(np.isclose(rho[0], 0.02 * radius) and np.isclose(rho[-1], 0.98 * radius)
                                            and angle[0] == 0.0 and np.isclose(angle[-1], math.pi)
                                            and along[0] == 0.0 and np.isclose(along[-1], 5.0 * radius))}


def knots(held: dict[str, np.ndarray], count: int = DRAWN) -> dict[str, object]:
    import tube_fourier_bessel as fb

    radius = float(held["R_cyl"])
    rho, angle, along, data = held["rho_grid"], held["dphi_grid"], held["dz_grid"], held["data"]
    rng = np.random.default_rng(20261029)
    n = len(rho)
    picks = [tuple(int(v) for v in (rng.integers(n), rng.integers(n), rng.integers(len(angle)),
                                    rng.integers(len(along)))) for _ in range(count)]
    picks += [(n - 1, n - 1, 0, 0), (n - 1, n - 2, 0, 1), (n - 2, n - 1, 1, 0), (n - 1, n - 1, 1, 1),
              (0, 0, 0, 0), (0, n - 1, 3, 2), (n - 1, 0, 3, 2)]
    as_built, swapped, bins = 0.0, 0.0, [f"{a}-{b}" for a, b in zip(BINS, BINS[1:], strict=False)]
    defect, total = dict.fromkeys(bins, 0.0), dict.fromkeys(bins, 0.0)
    for i, j, p, q in picks:
        target = np.array([[rho[i] * math.cos(angle[p]), rho[i] * math.sin(angle[p]), along[q]]]) / radius
        source = np.array([[rho[j], 0.0, 0.0]]) / radius
        turn = fb.to_cylindrical(np.array([angle[p]]))[0]

        def series(truncation: tuple[int, int, int], *, built: bool = False,
                   target: np.ndarray = target, source: np.ndarray = source, turn: np.ndarray = turn) -> np.ndarray:
            return turn @ fb.image(target, source, 1.0, 1.0, modes=truncation[0], wavenumbers=truncation[1],
                                   angles=truncation[2], as_built=built)[0, 0]

        built, correct, finer = series(AS_BUILT, built=True), series(AS_BUILT), series(FINER)
        value = data[i, j, p, q] * radius
        as_built = max(as_built, float(np.max(np.abs(value - built)) / np.max(np.abs(built))))
        swapped = max(swapped, float(np.max(np.abs(data[j, i, p, q] * radius - built)) / np.max(np.abs(built))))
        reach = max(rho[i], rho[j]) / radius
        key = next(f"{a}-{b}" for a, b in zip(BINS, BINS[1:], strict=False) if a <= reach < b)
        defect[key] = max(defect[key], float(np.max(np.abs(value - correct)) / np.max(np.abs(correct))))
        total[key] = max(total[key], float(np.max(np.abs(value - finer)) / np.max(np.abs(finer))))
    return {"picks": len(picks), "against_the_builders_series": as_built,
            "with_the_first_two_axes_swapped": swapped,
            "against_the_series_at_the_tables_truncation_by_reach": defect,
            "against_the_series_at_the_finer_truncation_by_reach": total}


def helix_samples() -> np.ndarray:
    import jax.numpy as jnp
    from script_a import fl9

    kind = fl9()
    settings = {"radius": BODY_RADIUS, "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33, "starts": 2}
    counts = kind.sample_counts(settings, 0.9e-3, "lateral")
    held = kind.samples({k: jnp.asarray(v) if isinstance(v, float) else v for k, v in settings.items()}, "lateral",
                        counts)
    return np.asarray(held.points), np.asarray(held.weights)


def patch(table_path: str, radius_cyl: float) -> list[dict[str, object]]:
    import jax
    import jax.numpy as jnp
    import tube_fourier_bessel as fb

    from mime.nodes.hydro.bem import blob_squared, resistance, stokeslet_matrix, tube_correction, tube_table_from

    jax.config.update("jax_enable_x64", True)
    points0, weights = helix_samples()
    tube = radius_cyl * BODY_RADIUS
    eps2 = float(blob_squared(jnp.asarray(weights), 0.5))
    found: list[dict[str, object]] = []
    reach = float(np.max(np.hypot(points0[:, 1], points0[:, 2])))
    for fraction in (0.0, 0.3):
        if reach + fraction * tube > 0.98 * tube:
            continue
        points = points0 + np.array([0.0, fraction * tube, 0.0])
        frame = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        started = time.perf_counter()
        table = tube_table_from(table_path).arrays(jnp.float64)
        loaded = time.perf_counter() - started
        read = tube_correction(jnp.asarray(points), {"centre": (0.0, 0.0, 0.0), "frame": frame, "radius": tube,
                                                     "table": table}, VISCOSITY)
        g_table = np.asarray(read.matrix)
        local = points[:, [1, 2, 0]] / tube
        started = time.perf_counter()
        series = fb.image(local, local, 1.0, 1.0, modes=FINER[0], wavenumbers=FINER[1], angles=FINER[2])
        took = time.perf_counter() - started
        series = series[:, :, [2, 0, 1]][:, :, :, [2, 0, 1]]
        n = len(points)
        g_series = series.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) / (tube * VISCOSITY)
        centre = jnp.asarray(points.mean(axis=0))

        def resisted(correction: np.ndarray | None) -> np.ndarray:
            return np.asarray(resistance(jnp.asarray(points), eps2, VISCOSITY, centre,
                                         None if correction is None else jnp.asarray(correction)))

        free, with_table, with_series = resisted(None), resisted(g_table), resisted(g_series)
        scale = np.sqrt(np.outer(np.diag(with_series), np.diag(with_series)))

        def swim(resisting: np.ndarray) -> float:
            """The force-free screw's speed along its axis (x) per unit spin about it."""
            return float(-np.linalg.solve(resisting[:3, :3], resisting[:3, 3:] @ np.array([1.0, 0.0, 0.0]))[0])

        free_matrix = np.asarray(stokeslet_matrix(jnp.asarray(points), eps2, VISCOSITY))
        found.append({
            "offset_over_R": fraction, "samples": n, "reach_over_R": (reach + fraction * tube) / tube,
            "accepted": bool(read.accepted),
            "correction_difference_frobenius": float(np.linalg.norm(g_table - g_series) / np.linalg.norm(g_series)),
            "correction_over_free_space_frobenius": float(np.linalg.norm(g_series) / np.linalg.norm(free_matrix)),
            "reciprocity_table": float(np.linalg.norm(g_table - g_table.T) / np.linalg.norm(g_table)),
            "reciprocity_series": float(np.linalg.norm(g_series - g_series.T) / np.linalg.norm(g_series)),
            "resistance_worst_over_sqrt_diagonal": float(np.max(np.abs(with_table - with_series) / scale)),
            "resistance_diagonal_relative": [float(v) for v in np.diag(with_table) / np.diag(with_series) - 1.0],
            "wall_effect_on_the_diagonal": [float(v) for v in np.diag(with_series) / np.diag(free) - 1.0],
            "swim_with_the_table_over_the_series": swim(with_table) / swim(with_series),
            "relied_on": bool(np.max(np.abs(with_table - with_series) / scale) <= RELIED_ON),
            "seconds": {"table_load_and_compile": loaded, "series": took},
        })
    return found


def check(name: str) -> dict[str, object]:
    from mime.assets import locate, reference

    path = str(locate(reference(name)))
    started = time.perf_counter()
    with np.load(path) as file:
        held = {key: file[key] for key in ("R_cyl", "mu", "rho_grid", "dphi_grid", "dz_grid", "data")}
    loaded = time.perf_counter() - started
    record: dict[str, object] = {"table": name, "seconds_to_read_the_file": loaded}
    record["provenance"] = provenance(name, held)
    record["knots"] = knots(held)
    radius = float(held["R_cyl"])
    del held
    record["lateral_patch"] = patch(path, radius)
    return record


def main() -> None:
    if sys.argv[1:] == ["--summary"]:
        records = [json.loads(line) for line in RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()]
        SUMMARY.write_text(json.dumps({"written_by": "tests/differential/check_wall_tables.py", "relied_on": RELIED_ON,
                                       "as_built": AS_BUILT, "finer": FINER, "tables": records}, indent=1) + "\n",
                           encoding="utf-8")
        return
    for name in sys.argv[1:]:
        record = check(name)
        with RECORDS.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
