"""Phase 7j-a's breakdown, assembled from the parts' JSON (``bem.py``, ``surrogate.py``, ``modal_app.py``): the speed
at A's operating point by rung, each with its band; the model-internal gaps; the uncertainty budget; every control's
verdict; the bit-for-bit check; and each result beside its preregistered prediction (``predictions_7ja.json``).
Computes nothing new from the model. No measured speed enters it: it reads none, and compares with none.

    python scripts/sim2real/assemble.py --runs DIR [--repeat DIR] [--mesh MESH.json] --out SUMMARY.json

``--mesh`` is the headline round two's mesh sequence (its ``results2.json``: the floored pose at phase 0, six
spacings from 0.178 to 0.100 mm), reused rather than run again.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
MM = 1e3
VARIANTS = ("axial", "lateral_free", "wall_held")
DT_ORDER_RANGE, DT_ORDER_AGREEMENT = (0.7, 1.3), 0.3
DT_PLANTS, DT_PLANT_SEED = 20, 20261031


def load(directory: Path, name: str) -> dict[str, Any] | None:
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else None


def mm(x: float | None) -> float | None:
    return None if x is None else x * MM


# ---- the ladder ----------------------------------------------------------------------------------------------------
def surrogate_static(runs: dict[str, Any], lubrication: bool, margin: float = 0.1) -> dict[str, Any]:
    case = next(c for c in runs["surrogate_static"]["cases"]
                if c["lubrication"] is lubrication and abs(c["margin_mm"] - margin) < 1e-12)
    return {v: mm(case[v]["mean"]) for v in VARIANTS} | {"phase_spread": case["axial"]["spread_over_mean"]}


def bem_static(runs: dict[str, Any], patches: str, spacing: float, key: str = "bem_ladder") -> dict[str, Any] | None:
    held = runs.get(key)
    if held is None:
        return None
    case = next((c for c in held["cases"] if c["label"].startswith(f"{patches}/{spacing:.3f}mm/")), None)
    if case is None:
        return None
    out = {v: mm(case[v]["mean"]) for v in VARIANTS}
    out.update(phase_spread=case["axial"]["spread_over_mean"], gap_over_spacing=case["gap_over_spacing"],
               samples=case["samples"], unknowns=case["unknowns"],
               quadrature=_phase_quadrature(case), per_phase_axial=[mm(r["axial"]) for r in case["per_phase"]])
    if "with_lubrication" in case:
        out["with_lubrication"] = {v: mm(case["with_lubrication"][v]["mean"]) for v in VARIANTS}
    if "control_node_against_own_assembly" in case:
        out["control_node_against_own_assembly"] = case["control_node_against_own_assembly"]
    return out


def _phase_quadrature(case: dict[str, Any]) -> float:
    """The phase mean over eight phases against the mean over every other one: the quadrature's own error, estimated."""
    values = np.asarray([r["axial"] for r in case["per_phase"]])
    return float(abs(values.mean() - values[::2].mean()) / abs(values.mean()))


def dynamics_speed(run: dict[str, Any]) -> float:
    return run["swim_speed_axial_m_s"] * MM


# ---- convergence in dt -----------------------------------------------------------------------------------------------
def dt_series(runs: dict[str, Any]) -> list[tuple[float, float]]:
    found = {0.5: runs["surrogate_dynamics"]["runs"][0]["swim_speed_axial_m_s"]}
    for run in runs["surrogate_dt"]["runs"]:
        if run["settings"].get("lubrication", True):
            found[run["settings"]["dt_ms"]] = run["swim_speed_axial_m_s"]
    return sorted(found.items(), reverse=True)


def richardson(series: list[tuple[float, float]]) -> dict[str, Any]:
    """Observed orders from the coarsest and the finest triples (each step halves dt), and the extrapolation from the
    finest; the verdict "first order, consistent" when both orders lie in ``DT_ORDER_RANGE`` and agree within
    ``DT_ORDER_AGREEMENT``."""
    values = [v for _, v in series]

    def order(a: float, b: float, c: float) -> float:
        ratio = (a - b) / (b - c)
        return math.log2(ratio) if ratio > 0 else float("nan")

    first, last = order(*values[:3]), order(*values[-3:])
    p = last
    limit = values[-1] - (values[-2] - values[-1]) / (2 ** p - 1) if math.isfinite(p) else float("nan")
    consistent = all(math.isfinite(q) and DT_ORDER_RANGE[0] <= q <= DT_ORDER_RANGE[1] for q in (first, last)) \
        and abs(first - last) <= DT_ORDER_AGREEMENT
    at_half = dict(series)[0.5]
    return {"dt_ms": [d for d, _ in series], "speeds_mm_s": [v * MM for v in values], "order_coarse": first,
            "order_fine": last, "limit_mm_s": limit * MM, "excess_at_0p5ms": at_half / limit - 1.0,
            "first_order_consistent": consistent}


def richardson_control(series: list[tuple[float, float]]) -> dict[str, Any]:
    rng = np.random.default_rng(DT_PLANT_SEED)
    plants = []
    for _ in range(DT_PLANTS):
        bump = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.003, 0.02))
        planted = [*series[:-1], (series[-1][0], series[-1][1] * (1.0 + bump))]
        plants.append({"bump": bump, "caught": not richardson(planted)["first_order_consistent"]})
    return {"plants": plants, "caught_every_run": all(p["caught"] for p in plants)}


# ---- the budget ----------------------------------------------------------------------------------------------------
def budget(runs: dict[str, Any]) -> dict[str, Any] | None:
    held = runs.get("bem_budget")
    if held is None:
        return None
    speed = held["speed"]["axial"]
    columns: dict[str, Any] = {}
    for name, linear in held["across_range"]["axial"].items():
        ends = held.get("range_ends", {}).get(name, {})
        secant = ends.get("secant_across_range", {}).get("axial")
        used = max(abs(linear), abs(secant)) if secant is not None else abs(linear)
        columns[name] = {"linear_across_range_relative": linear / speed,
                         "secant_across_range_relative": None if secant is None else secant / speed,
                         "secant_over_linear": ends.get("secant_over_linear", {}).get("axial"),
                         "used_relative": used / abs(speed)}
    without_margin = [c["used_relative"] for n, c in columns.items() if n != "margin"]
    return {"speed_mm_s": speed * MM, "columns": columns,
            "ranking_without_margin": sorted((n for n in columns if n != "margin"),
                                             key=lambda n: -columns[n]["used_relative"]),
            "rss_uniform_relative": float(math.sqrt(sum((c / math.sqrt(12.0)) ** 2 for c in without_margin))),
            "linear_worst_case_relative": float(sum(c / 2.0 for c in without_margin)),
            "fim_rank": held["fim_axial"].get("rank"), "fim_param_names": held["fim_axial"].get("param_names"),
            "fim_diagonal": np.diag(np.asarray(held["fim_axial"]["fim"])).tolist() if "fim" in held["fim_axial"]
            else None,
            "fim_leaves_param_names": held["fim_leaves_axial"].get("param_names"),
            "fim_leaves_diagonal": np.diag(np.asarray(held["fim_leaves_axial"]["fim"])).tolist()
            if "fim" in held["fim_leaves_axial"] else None,
            "fd_check": held["fd_check"], "viscosity_worked_example": held["viscosity_worked_example"]}


def dynamics_fd(runs: dict[str, Any], nominal: float) -> dict[str, Any] | None:
    held = runs.get("surrogate_fd")
    if held is None:
        return None
    out: dict[str, Any] = {}
    for name, ends in held["runs"].items():
        low = ends.get("low", {}).get("swim_speed_axial_m_s", nominal)
        high = ends.get("high", {}).get("swim_speed_axial_m_s", nominal)
        out[name] = {"low_mm_s": low * MM, "high_mm_s": high * MM, "across_range_relative": (high - low) / nominal}
        if "nominal_by_density" in ends:
            out[name]["nominal_by_density_minus_nominal_relative"] = \
                ends["nominal_by_density"]["swim_speed_axial_m_s"] / nominal - 1.0
    width = [abs(v["across_range_relative"]) for v in out.values()]
    return {"inputs": out, "rss_uniform_relative": float(math.sqrt(sum((w / math.sqrt(12.0)) ** 2 for w in width))),
            "linear_worst_case_relative": float(sum(w / 2.0 for w in width))}


# ---- the mesh (reused) and the blob --------------------------------------------------------------------------------
def mesh(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    series = json.loads(path.read_text())["part1"]["series"]
    out: dict[str, Any] = {}
    for key in ("closed/floored/new", "lateral/floored/new", "lateral/free/none"):
        rows = series[key]["rows"]
        speeds = [r["axial_mm_s_10Hz"] for r in rows]
        out[key] = {"spacing_mm": [r["spacing_mm"] for r in rows], "axial_mm_s": speeds,
                    "rise_0p178_to_finest": speeds[-1] / speeds[0] - 1.0,
                    "rise_0p178_to_0p140": speeds[2] / speeds[0] - 1.0,
                    "last_step": speeds[-1] / speeds[-2] - 1.0}
    return out


def blob(runs: dict[str, Any]) -> dict[str, Any] | None:
    held = runs.get("bem_blob")
    if held is None:
        return None
    out: dict[str, Any] = {}
    for case in held["cases"]:
        patches, spacing, blob_label = case["label"].split("/")[:3]
        out.setdefault(f"{patches}/{spacing}", {})[blob_label] = mm(case["axial"]["mean"])
    for key, row in out.items():
        values = [row[f"blob{b}"] for b in (0.35, 0.5, 0.7) if f"blob{b}" in row]
        row["spread_relative"] = (max(values) - min(values)) / row.get("blob0.5", values[len(values) // 2])
        row["change_0p35_to_0p7_relative"] = (row["blob0.7"] - row["blob0.35"]) / row["blob0.5"] \
            if "blob0.7" in row and "blob0.35" in row else None
    return out


# ---- the bit-for-bit check -------------------------------------------------------------------------------------------
def bitwise(runs: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("surrogate_static", "surrogate_dynamics", "surrogate_dt", "surrogate_margin"):
        a, b = runs.get(name), repeat.get(name)
        if a is None or b is None:
            continue
        out[name] = _strip(a) == _strip(b)
    if runs.get("bem_ladder") is not None and runs.get("bem_ladder_repeat") is not None:
        out["bem_ladder_a100"] = _strip(runs["bem_ladder"]) == _strip(runs["bem_ladder_repeat"])
    for patches in ("closed", "lateral"):
        a, b = runs.get(f"bem_dynamics_{patches}_first"), runs.get(f"bem_dynamics_{patches}_second")
        if a is not None and b is not None:
            out[f"bem_dynamics_{patches}"] = a["trajectory_sha256"] == b["trajectory_sha256"]
    dynamics = runs.get("surrogate_dynamics")
    if dynamics is not None and "control_one_ulp" in dynamics:
        out["control_one_ulp_changes_the_hash"] = (dynamics["control_one_ulp"]["trajectory_sha256"]
                                                   != dynamics["runs"][0]["trajectory_sha256"])
    return out


def _strip(payload: Any) -> Any:
    """A part's numbers without its timings and environment."""
    if isinstance(payload, dict):
        return {k: _strip(v) for k, v in payload.items()
                if k not in ("seconds", "run_seconds", "environment", "jacobian_seconds", "fim_seconds")}
    if isinstance(payload, list):
        return [_strip(v) for v in payload]
    return payload


# ---- the whole -------------------------------------------------------------------------------------------------------
def assemble(runs: dict[str, Any], repeat: dict[str, Any], mesh_path: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {"ladder": {}, "gaps": {}, "controls": {}}
    ladder = out["ladder"]
    ladder["surrogate_static_nolub"] = surrogate_static(runs, False)
    ladder["surrogate_static_lub"] = surrogate_static(runs, True)
    dyn = runs["surrogate_dynamics"]["runs"]
    ladder["surrogate_dynamics_lub"] = {"axial": dynamics_speed(dyn[0]), "tilt_max_rad": dyn[0]["tilt_max_rad"],
                                        "rotation_hz": dyn[0]["body_rotation_rate_hz"]}
    ladder["surrogate_dynamics_nolub"] = {"axial": dynamics_speed(dyn[1]), "tilt_max_rad": dyn[1]["tilt_max_rad"],
                                          "rotation_hz": dyn[1]["body_rotation_rate_hz"]}
    for patches in ("closed", "lateral"):
        for spacing in (0.178, 0.140):
            found = bem_static(runs, patches, spacing)
            if found is not None:
                ladder[f"bem_static_{patches}_{spacing:.3f}"] = found
        for tag in ("first", "second"):
            run = runs.get(f"bem_dynamics_{patches}_{tag}")
            if run is not None and tag == "first":
                ladder[f"bem_dynamics_{patches}"] = {"axial": dynamics_speed(run), "tilt_max_rad": run["tilt_max_rad"],
                                                     "rotation_hz": run["body_rotation_rate_hz"],
                                                     "seconds_per_step": run["seconds_per_step"],
                                                     "device": run["environment"]["devices"]}

    gaps = out["gaps"]
    series = dt_series(runs)
    gaps["dt"] = richardson(series)
    out["controls"]["dt_convergence_verdict"] = richardson_control(series)
    short = runs.get("surrogate_short")
    if short is not None:
        gaps["short_window_bias"] = {r["label"]: r["swim_speed_axial_m_s"] / d["swim_speed_axial_m_s"] - 1.0
                                     for r, d in zip(short["runs"], dyn[:2], strict=True)}
    gaps["mesh_reused"] = mesh(mesh_path)
    gaps["blob"] = blob(runs)
    table = runs.get("bem_table")
    if table is not None:
        gaps["table_read"] = {c["label"]: {"band_half_width_relative": c["axial"]["band_half_width"]
                                           / abs(c["axial"]["nominal"])} for c in table["cases"]}
        out["controls"]["table_plants"] = {c["label"]: c["control"]["caught_every_run"] for c in table["cases"]}
    margin = runs.get("bem_margin")
    if margin is not None:
        gaps["margin_bem"] = [{"label": c["label"], "margin_mm": c["margin_mm"], "axial_mm_s": mm(c["axial"]["mean"]),
                               "gap_over_spacing": c["gap_over_spacing"]} for c in margin["cases"]]
    gaps["margin_surrogate_static"] = [{"lubrication": c["lubrication"], "margin_mm": c["margin_mm"],
                                        "axial_mm_s": mm(c["axial"]["mean"])}
                                       for c in runs["surrogate_static"]["cases"]]
    sweep = runs.get("surrogate_margin")
    if sweep is not None:
        gaps["margin_surrogate_dynamics"] = [{"label": r["label"], "axial_mm_s": dynamics_speed(r)}
                                             for r in sweep["runs"]]
    offsets = runs.get("bem_offsets")
    if offsets is not None:
        gaps["surrogate_against_bem_by_offset"] = [
            {"label": c["label"], "offset_over_R": c["offset_over_R"], "bem_axial_mm_s": mm(c["axial"]["mean"]),
             "surrogate_axial_mm_s": mm(c["surrogate"]["axial"]["mean"]),
             "surrogate_over_bem": c["surrogate"]["axial"]["mean"] / c["axial"]["mean"]} for c in offsets["cases"]]
    out["budget"] = budget(runs)
    out["dynamics_fd"] = dynamics_fd(runs, dyn[0]["swim_speed_axial_m_s"])
    out["bitwise"] = bitwise(runs, repeat)
    solves = sum(int(runs[k].get("bem_solves", 0)) for k in runs if k.startswith("bem_") and isinstance(runs[k], dict))
    out["bem_solves"] = {"static": solves, "dynamics_steps": sum(int(runs[k]["steps"]) for k in runs
                                                                 if k.startswith("bem_dynamics_"))}
    groups = runs.get("surrogate_groups")
    out["coupling_groups"] = None if groups is None else {k: v["coupling_groups"] for k, v in groups["rungs"].items()}
    if groups is not None:
        out["controls"]["same_as_script_a"] = groups["same_as_script_a"]
    out["same_physics_control"] = "missing: de Jongh's model values are not digitised (every value null)"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", required=True)
    parser.add_argument("--repeat", default=None)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    directory = Path(args.runs)
    runs = {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))}
    repeat = {path.stem: json.loads(path.read_text()) for path in sorted(Path(args.repeat).glob("*.json"))} \
        if args.repeat else {}
    summary = assemble(runs, repeat, Path(args.mesh) if args.mesh else None)
    summary["predictions"] = json.loads((HERE / "predictions_7ja.json").read_text())["predictions"]
    Path(args.out).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("ladder", "bitwise", "bem_solves")}, indent=1))


if __name__ == "__main__":
    main()
