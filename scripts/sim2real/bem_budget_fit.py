"""The BEM error budget's analysis (items 2 to 4), from the grid's and the controls' records: the speeds, the error
surface's fits and their controls, the split at the frozen point, Richardson in the blob and in the spacing, and the
regularised near-wall image's effect. The rules are ``predictions_bem_budget.json``'s, written before any solve. No
measured speed enters.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python bem_budget_fit.py --grid DIR --controls DIR --stage1 FLOORED.json --out OUT
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import bem_budget as bb  # noqa: E402

K = 2.0
MODELS = {
    "A": lambda h, e: [np.ones_like(h), e, h * h / e],
    "B": lambda h, e: [np.ones_like(h), e, e * e, h * h / e],
    "C": lambda h, e: [np.ones_like(h), e, e * e, h * h / e, h * h],
    # added after the grid's first levels were read (post hoc, labelled so in the report): a lattice's quadrature
    # error for the regularised kernel, h exp(-2 pi eps / h)
    "D": lambda h, e: [np.ones_like(h), e, e * e, h * np.exp(-2.0 * math.pi * e / h)],
}
#: Which columns of each model are the regularisation (terms in eps alone) and which the quadrature.
REGULARISATION = {"A": [1], "B": [1, 2], "C": [1, 2], "D": [1, 2]}
QUADRATURE = {"A": [2], "B": [3], "C": [3, 4], "D": [3]}


def a_speed(matrices: np.ndarray) -> float | None:
    """A0f's reading at A (mm/s) from the eight phases' resistances: the torque variant in script A's uniform field,
    both senses' mean (``a0f_statics``); None where a phase is not finite or not positive definite."""
    import a0f_statics as statics
    import bem

    if not np.all(np.isfinite(matrices)) or min(bem.smallest_eigenvalue(r) for r in matrices) <= 0.0:
        return None
    interp = statics.Interpolant(matrices)
    if not statics.positive_definite_everywhere(interp, 512)["positive_definite"]:
        return None
    values = [statics.readings(interp, np.asarray([0.0]), sense, "torque", 0.0, uniform=True)[0]["u_along_pass_m_s"]
              for sense in (1, -1)]
    return 1e3 * 0.5 * (values[0] + values[1])


def fit(model: str, h: np.ndarray, e: np.ndarray, u: np.ndarray) -> dict[str, Any]:
    x = np.stack(MODELS[model](h, e), axis=1)
    coef, *_ = np.linalg.lstsq(x, u, rcond=None)
    resid = u - x @ coef
    dof = max(len(u) - x.shape[1], 1)
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.inv(x.T @ x)
    return {"model": model, "coef": coef.tolist(), "cov": cov.tolist(), "rms": float(math.sqrt(np.mean(resid ** 2))),
            "max_residual": float(np.max(np.abs(resid))), "points": int(len(u)), "dof": dof,
            "limit": float(coef[0]), "limit_sigma": float(math.sqrt(cov[0, 0]))}


def split_at(result: dict[str, Any], h: float, e: float) -> dict[str, Any]:
    """The regularisation and quadrature terms at ``(h, e)`` with their sigmas, and the sigma of their difference."""
    model = result["model"]
    row = np.asarray([float(v) for v in MODELS[model](np.asarray(h), np.asarray(e))])
    coef, cov = np.asarray(result["coef"]), np.asarray(result["cov"])
    g_reg = np.zeros_like(row)
    g_reg[REGULARISATION[model]] = row[REGULARISATION[model]]
    g_quad = np.zeros_like(row)
    g_quad[QUADRATURE[model]] = row[QUADRATURE[model]]
    reg, quad = float(g_reg @ coef), float(g_quad @ coef)
    s_reg, s_quad = float(math.sqrt(g_reg @ cov @ g_reg)), float(math.sqrt(g_quad @ cov @ g_quad))
    g_diff = np.sign(abs(quad) - abs(reg)) * (np.sign(quad) * g_quad - np.sign(reg) * g_reg)
    s_diff = float(math.sqrt(g_diff @ cov @ g_diff))
    if abs(abs(quad) - abs(reg)) > K * s_diff:
        dominant = "quadrature" if abs(quad) > abs(reg) else "regularisation"
    else:
        dominant = "neither (within 2 sigma)"
    return {"regularisation": reg, "regularisation_sigma": s_reg, "quadrature": quad, "quadrature_sigma": s_quad,
            "difference_sigma": s_diff, "dominant": dominant, "h": h, "eps": e}


def richardson_eps(e: tuple[float, float], u: tuple[float, float]) -> float:
    """The linear extrapolation in eps to zero from two blobs at one spacing."""
    return u[0] - e[0] * (u[1] - u[0]) / (e[1] - e[0])


def table(levels: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Per level and blob: A's speed (eight phases), the axial speeds at phases 6 and 0, the least eigenvalue."""
    import bem

    rows = []
    for level in levels:
        h = level["rung"]["mean_spacing_m"] * 1e3
        for blob in bb.BLOBS:
            key = f"{blob}"
            held = [level["phases"][str(k)][arm].get(key) for k in range(8)]
            if any(x is None or "resistance_si" not in x for x in held):
                rows.append({"h_mm": h, "blob": blob, "eps_mm": blob * h, "missing": True})
                continue
            matrices = np.asarray([x["resistance_si"] for x in held])
            finite = bool(np.all(np.isfinite(matrices)))
            least = min(bem.smallest_eigenvalue(r) for r in matrices) if finite else None
            rows.append({"h_mm": h, "blob": blob, "eps_mm": blob * h, "spacing_mm": level["rung"]["spacing_mm"],
                         "A_mm_s": a_speed(matrices), "axial_phase6_mm_s": held[6]["axial_mm_s"],
                         "axial_phase0_mm_s": held[0]["axial_mm_s"],
                         "axial_phase_mean_mm_s": float(np.mean([x["axial_mm_s"] for x in held])),
                         "least_eigenvalue_nd": least,
                         "indefinite_phases": [k for k in range(8) if held[k]["min_eigenvalue_nd"] is None
                                               or held[k]["min_eigenvalue_nd"] <= 0.0],
                         "near_samples": held[0].get("near_samples")})
    return {"rows": rows}


def fits_for(rows: list[dict[str, Any]], key: str, frozen: tuple[float, float]) -> dict[str, Any]:
    good = [r for r in rows if r.get(key) is not None and not r.get("missing")]
    h = np.asarray([r["h_mm"] for r in good])
    e = np.asarray([r["eps_mm"] for r in good])
    u = np.asarray([r[key] for r in good])
    out: dict[str, Any] = {"points_used": len(good), "points_excluded": [
        (r["h_mm"], r["blob"]) for r in rows if r.get(key) is None or r.get("missing")]}
    for model in MODELS:
        if len(good) <= len(MODELS[model](h, e)):
            out[model] = {"not_fitted": "too few points"}
            continue
        result = fit(model, h, e, u)
        result["split_at_frozen"] = split_at(result, *frozen)
        out[model] = result
    return out


def b5_per_blob(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    import m1_ladder

    out = {}
    for blob in bb.BLOBS:
        chosen = sorted([r for r in rows if r["blob"] == blob and r.get(key) is not None], key=lambda r: -r["h_mm"])
        if len(chosen) < 3:
            out[f"{blob}"] = {"not_run": f"{len(chosen)} values"}
            continue
        verdict = m1_ladder.b5([r["h_mm"] for r in chosen], [r[key] for r in chosen])
        out[f"{blob}"] = {k: verdict.get(k) for k in ("values", "orders", "order_spread", "bounded", "reasons",
                                                      "limit_finest_triplet", "richardson_estimate")}
    return out


def eps_richardson(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out = {}
    for pair in ((0.7, 1.0), (0.5, 0.7), (0.5, 1.0)):
        found = {}
        for spacing in sorted({r["h_mm"] for r in rows}, reverse=True):
            at = {r["blob"]: r for r in rows if r["h_mm"] == spacing}
            if all(b in at and at[b].get(key) is not None for b in pair):
                found[f"{spacing:.5f}"] = richardson_eps((at[pair[0]]["eps_mm"], at[pair[1]]["eps_mm"]),
                                                         (at[pair[0]][key], at[pair[1]][key]))
        out[f"{pair[0]}_{pair[1]}"] = found
    return out


def controls(folder: Path) -> dict[str, Any]:
    rows_free, rows_tube = [], []
    a = bb.SPHERE_RATIO * bb.TUBE_RADIUS_M * 1e3
    for path in sorted(folder.glob("sphere_0.*.json")):
        held = json.loads(path.read_text())
        for arm, rows in (("free", rows_free), ("tube", rows_tube)):
            block = held[arm]
            h = block["mean_spacing_m"] * 1e3
            for blob in bb.BLOBS:
                solve = block["solves"].get(f"{blob}")
                if solve is None:
                    continue
                rows.append({"h_mm": h, "blob": blob, "eps_mm": blob * h, "spacing_mm": held["spacing_mm"],
                             "value": solve["drag_over_stokes"], "samples": block["samples"]})
    k_hs, k_bohlin = bb.haberman_sayre(bb.SPHERE_RATIO), bb.bohlin(bb.SPHERE_RATIO)
    for r in rows_free:
        r["reference_regularised"] = bb.sphere_regularised_drag(r["eps_mm"] / a)
        r["quadrature_measured"] = r["value"] - r["reference_regularised"]
    out: dict[str, Any] = {"free": {"rows": rows_free}, "tube": {"rows": rows_tube}, "K_haberman_sayre": k_hs,
                           "K_bohlin": k_bohlin, "sphere_radius_mm": a}
    frozen_free = [r for r in rows_free if abs(r["spacing_mm"] - bb.FROZEN_SPACING_MM) < 1e-9
                   and r["blob"] == bb.FROZEN_BLOB]
    frozen_tube = [r for r in rows_tube if abs(r["spacing_mm"] - bb.FROZEN_SPACING_MM) < 1e-9
                   and r["blob"] == bb.FROZEN_BLOB]
    if frozen_free:
        point = (frozen_free[0]["h_mm"], frozen_free[0]["eps_mm"])
        out["free"]["fits"] = fits_for(rows_free, "value", point)
        known = bb.sphere_regularised_drag(point[1] / a) - 1.0
        out["free"]["known_regularisation_at_frozen"] = known
        verdicts = {}
        for model in MODELS:
            result = out["free"]["fits"].get(model, {})
            if "split_at_frozen" not in result:
                continue
            s = result["split_at_frozen"]
            verdicts[model] = {
                "regularisation_recovered": abs(s["regularisation"] - known) <= K * s["regularisation_sigma"],
                "regularisation_miss": s["regularisation"] - known,
                "limit_recovered": abs(result["limit"] - 1.0) <= K * result["limit_sigma"],
                "limit_miss": result["limit"] - 1.0}
        out["free"]["control"] = verdicts
        # the measured quadrature error's form, against h^2/eps
        q = np.asarray([r["quadrature_measured"] for r in rows_free])
        hh = np.asarray([r["h_mm"] for r in rows_free])
        ee = np.asarray([r["eps_mm"] for r in rows_free])
        x = (hh * hh / ee)[:, None]
        c, *_ = np.linalg.lstsq(x, q, rcond=None)
        out["free"]["quadrature_form_h2_over_eps"] = {"c": float(c[0]), "rms": float(np.sqrt(np.mean((q - x[:, 0] * c[0]) ** 2))),
                                                      "rms_of_q": float(np.sqrt(np.mean(q ** 2)))}
        # does the measured quadrature error collapse as h Q(eps / h)? Q's spread across the spacings at each blob
        collapse = {}
        for blob in bb.BLOBS:
            values = [r["quadrature_measured"] / r["h_mm"] for r in rows_free if r["blob"] == blob]
            collapse[f"{blob}"] = {"Q_per_mm": values, "spread_over_mean": float((max(values) - min(values))
                                                                                 / abs(np.mean(values)))
                                   if values and np.mean(values) != 0 else None}
        out["free"]["quadrature_collapse_h_Q"] = collapse
        out["free"]["eps_richardson"] = eps_richardson(rows_free, "value")
        out["free"]["b5_per_blob"] = b5_per_blob(rows_free, "value")
    if frozen_tube:
        point = (frozen_tube[0]["h_mm"], frozen_tube[0]["eps_mm"])
        for r in rows_tube:
            r["over_K"] = r["value"] / k_hs
        out["tube"]["fits"] = fits_for(rows_tube, "value", point)
        out["tube"]["control"] = {model: {"limit_recovered": abs(res["limit"] - k_hs) <= K * res["limit_sigma"],
                                          "limit_miss_relative": res["limit"] / k_hs - 1.0}
                                  for model, res in out["tube"]["fits"].items() if "limit" in res}
        out["tube"]["eps_richardson"] = eps_richardson(rows_tube, "value")
        out["tube"]["b5_per_blob"] = b5_per_blob(rows_tube, "value")
    return out


def chosen_model(control: dict[str, Any]) -> str | None:
    """The preregistered order (B, A, C); D, added after the first levels were read, is never chosen here."""
    for model in ("B", "A", "C"):
        free = control["free"].get("control", {}).get(model)
        tube = control["tube"].get("control", {}).get(model)
        if free and tube and free["regularisation_recovered"] and free["limit_recovered"] and tube["limit_recovered"]:
            return model
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--stage1", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    levels = [json.loads(p.read_text()) for p in sorted(Path(args.grid).glob("a_level_*.json"))]
    levels = sorted(levels, key=lambda lv: -lv["rung"]["spacing_mm"])
    out: dict[str, Any] = {"levels": [{"spacing_mm": lv["rung"]["spacing_mm"], "complete": lv.get("complete", False),
                                       "stopped": lv.get("stopped"), "samples": lv["rung"]["samples"],
                                       "mean_spacing_mm": lv["rung"]["mean_spacing_m"] * 1e3} for lv in levels]}
    frozen_level = [lv for lv in levels if abs(lv["rung"]["spacing_mm"] - bb.FROZEN_SPACING_MM) < 1e-9]
    frozen = (frozen_level[0]["rung"]["mean_spacing_m"] * 1e3,
              bb.FROZEN_BLOB * frozen_level[0]["rung"]["mean_spacing_m"] * 1e3) if frozen_level else None
    out["controls"] = controls(Path(args.controls))
    model = chosen_model(out["controls"])
    out["chosen_model"] = model
    for arm in ("tube", "tube_regimage"):
        found = table(levels, arm)
        if frozen is not None:
            found["fits_A"] = fits_for(found["rows"], "A_mm_s", frozen)
            found["fits_axial_phase6"] = fits_for(found["rows"], "axial_phase6_mm_s", frozen)
            found["fits_axial_phase0"] = fits_for(found["rows"], "axial_phase0_mm_s", frozen)
        found["eps_richardson_A"] = eps_richardson(found["rows"], "A_mm_s")
        found["b5_per_blob_A"] = b5_per_blob(found["rows"], "A_mm_s")
        out[arm] = found
    free_rows = []
    for lv in levels:
        h = lv["rung"]["mean_spacing_m"] * 1e3
        for blob in bb.BLOBS:
            got = lv.get("free", {}).get(f"{blob}")
            if got is not None:
                free_rows.append({"h_mm": h, "blob": blob, "eps_mm": blob * h, "axial_mm_s": got["axial_mm_s"],
                                  "spacing_mm": lv["rung"]["spacing_mm"]})
    out["free"] = {"rows": free_rows}
    if frozen is not None:
        out["free"]["fits"] = fits_for(free_rows, "axial_mm_s", frozen)
        out["free"]["eps_richardson"] = eps_richardson(free_rows, "axial_mm_s")
        out["free"]["b5_per_blob"] = b5_per_blob(free_rows, "axial_mm_s")
    # the regularised image against the node's, per level and blob
    effect = []
    for plain, reg in zip(out["tube"]["rows"], out["tube_regimage"]["rows"], strict=True):
        if plain.get("A_mm_s") is not None and reg.get("A_mm_s") is not None:
            effect.append({"h_mm": plain["h_mm"], "blob": plain["blob"], "A_node": plain["A_mm_s"],
                           "A_regimage": reg["A_mm_s"], "relative": reg["A_mm_s"] / plain["A_mm_s"] - 1.0})
        else:
            effect.append({"h_mm": plain["h_mm"], "blob": plain["blob"], "A_node": plain.get("A_mm_s"),
                           "A_regimage": reg.get("A_mm_s"), "least_node": plain.get("least_eigenvalue_nd"),
                           "least_regimage": reg.get("least_eigenvalue_nd")})
    out["regimage_effect"] = effect
    if frozen is not None:
        for arm in ("tube", "tube_regimage"):
            values = [r["A_mm_s"] for r in out[arm]["rows"] if abs(r["h_mm"] - frozen[0]) < 1e-9
                      and r.get("A_mm_s") is not None]
            out[arm]["blob_spread_at_frozen_spacing"] = (max(values) - min(values)) / float(np.mean(values)) \
                if values else None
        values = [r["axial_mm_s"] for r in free_rows if abs(r["h_mm"] - frozen[0]) < 1e-9]
        out["free"]["blob_spread_at_frozen_spacing"] = (max(values) - min(values)) / float(np.mean(values)) \
            if values else None
    # the control: the frozen point against stage 1's record (the node's own step)
    if args.stage1 and frozen_level:
        held = json.loads(Path(args.stage1).read_text())
        level = frozen_level[0]
        diffs = []
        for k in range(8):
            a = np.asarray(level["phases"][str(k)]["tube"][f"{bb.FROZEN_BLOB}"]["resistance_si"])
            b = np.asarray(held["phases"][str(k)]["resistance_si"])
            diffs.append(float(np.max(np.abs(a - b)) / np.max(np.abs(b))))
        out["control_frozen_against_stage1"] = {"largest_relative": max(diffs),
                                                "node_step_control": level.get("control_frozen_against_node")}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")


if __name__ == "__main__":
    main()
