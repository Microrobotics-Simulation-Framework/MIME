"""The helix reference's analysis: the rules of ``predictions_helix_ref.json`` (its definitions, fixed before any solve)
applied to the records ``helix_ref_run`` wrote. Reads model values only.

    python helix_ref_fit.py --records DIR --out analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

BLOBS_LIMIT = ("0.35", "0.5", "0.7", "1.0")
FROZEN_FORCE, FROZEN_BLOB = "0.1600", "0.5"


def quantity(row: dict[str, Any], name: str) -> float:
    r = np.asarray(row["resistance_si"])
    sym = (r + r.T) / 2.0
    if name == "axial":
        return float(row["axial_mm_s"])
    if name == "drag":
        return float(row["drag_zz_over_stokes"])
    if name == "R_xx":
        return float(sym[0, 0])
    if name == "R_xOx":
        return float(sym[0, 3])
    raise KeyError(name)


def richardson(points: list[tuple[float, float]]) -> dict[str, Any]:
    """``points``: (h, U), coarse to fine. L1 (two finest), L1' (second and third finest), Lp (three finest, fitted
    order), and sigma by the predictions' rule."""
    out: dict[str, Any] = {"points": points}
    if len(points) < 2:
        return out
    (h2, u2), (h1, u1) = points[-2], points[-1]
    out["L1"] = u1 - (u2 - u1) / (h2 - h1) * h1
    sigma = []
    if len(points) >= 3:
        h3, u3 = points[-3]
        out["L1_prime"] = u2 - (u3 - u2) / (h3 - h2) * h2
        sigma.append(abs(out["L1"] - out["L1_prime"]))
        ratio = (u3 - u2) / (u2 - u1) if u2 != u1 else float("nan")

        def g(p: float) -> float:
            return (h3 ** p - h2 ** p) / (h2 ** p - h1 ** p) - ratio

        try:
            p = brentq(g, 0.05, 6.0)
        except (ValueError, ZeroDivisionError):
            p = None
        out["order"] = p
        if p is not None:
            c = (u2 - u1) / (h2 ** p - h1 ** p)
            out["Lp"] = u1 - c * h1 ** p
            if 0.5 <= p <= 3.0:
                sigma.append(abs(out["Lp"] - out["L1"]))
    out["sigma"] = max(sigma) if sigma else abs(u1 - u2)
    return out


def load(folder: Path, prefix: str) -> dict[str, dict[str, Any]]:
    return {p.stem.removeprefix(prefix): json.loads(p.read_text()) for p in sorted(folder.glob(f"{prefix}*.json"))}


# ---- route (a) ----------------------------------------------------------------------------------------------------
def route_a(levels: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    by_blob: dict[str, Any] = {}
    ordered = sorted((lv for lv in levels.values() if lv.get("blobs")), key=lambda lv: -lv["level"]["mean_spacing_m"])
    for blob in ("0.25",) + BLOBS_LIMIT:
        pts = [(lv["level"]["mean_spacing_m"] * 1e3, quantity(lv["blobs"][blob], name)) for lv in ordered
               if blob in lv["blobs"] and lv["blobs"][blob].get("finite")]
        by_blob[blob] = richardson(pts)
    l1 = [by_blob[b]["L1"] for b in BLOBS_LIMIT if "L1" in by_blob[b]]
    limit = float(np.mean(l1))
    sigma = max(max(by_blob[b]["sigma"] for b in BLOBS_LIMIT), (max(l1) - min(l1)) / 2.0)
    return {"by_blob": by_blob, "limit": limit, "sigma": sigma, "levels": [lv["level"]["spacing_mm"] for lv in ordered]}


def eps_lines(levels: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    ordered = sorted((lv for lv in levels.values() if lv.get("eps_lines")), key=lambda lv: -lv["level"]["mean_spacing_m"])
    keys = sorted({k for lv in ordered for k in lv["eps_lines"]}, key=float)
    out = {}
    for key in keys:
        pts = [(lv["level"]["mean_spacing_m"] * 1e3, lv["eps_lines"][key]["blob"], quantity(lv["eps_lines"][key], name))
               for lv in ordered if key in lv["eps_lines"] and lv["eps_lines"][key].get("finite")]
        out[key] = {"points": pts, "finest": pts[-1][2] if pts else None,
                    "last_change": abs(pts[-1][2] - pts[-2][2]) if len(pts) >= 2 else None}
    return out


# ---- route (b) ----------------------------------------------------------------------------------------------------
def route_b(sets: dict[str, dict[str, Any]], name: str, reference: dict[str, float] | None = None,
            eps_max: float = math.inf) -> dict[str, Any]:
    """The finest quadrature per force set, its sigma_q; U_reg(eps) by order-1 Richardson in the force sets' mean
    spacing from the two finest, sigma_f; the limit by weighted least squares in eps."""
    table: dict[str, dict[str, Any]] = {}
    sigma_q_by_eps: dict[str, float] = {}
    for force, rec in sorted(sets.items(), key=lambda kv: -kv[1]["force_set"]["mean_spacing_m"]):
        quads = {q: row for q, row in rec.get("quadratures", {}).items() if row.get("eps")}
        if not quads:
            continue
        order = sorted(quads, key=float)  # finest first
        finest = quads[order[0]]
        row: dict[str, Any] = {"h_mm": rec["force_set"]["mean_spacing_m"] * 1e3, "samples": rec["force_set"]["samples"],
                               "finest_quadrature": order[0], "eps": {}}
        for key, reading in finest["eps"].items():
            if float(key) > eps_max:
                continue
            entry = {"U": quantity(reading, name), "asymmetry": reading.get("asymmetry"), "quadrature_ladder": {
                q: quantity(quads[q]["eps"][key], name) for q in order if key in quads[q]["eps"]}}
            if len(order) >= 2 and key in quads[order[1]]["eps"]:
                entry["sigma_q"] = abs(entry["U"] - quantity(quads[order[1]]["eps"][key], name))
            if len(order) >= 3:
                ladder = [(quads[q]["mean_spacing_m"] * 1e3, quantity(quads[q]["eps"][key], name))
                          for q in reversed(order) if key in quads[q]["eps"]]
                entry["quadrature_fit"] = richardson(ladder)
            row["eps"][key] = entry
        table[force] = row
    forces = sorted(table, key=lambda f: -table[f]["h_mm"])
    # sigma_q for the sets with one quadrature: the 0.1024 set's
    for f in forces:
        for key, entry in table[f]["eps"].items():
            if "sigma_q" not in entry:
                donor = table.get("0.1024", {}).get("eps", {}).get(key, {})
                entry["sigma_q"] = donor.get("sigma_q", float("nan"))
                entry["sigma_q_from"] = "0.1024"
            sigma_q_by_eps.setdefault(key, 0.0)
    reg: dict[str, Any] = {}
    for key in sorted(sigma_q_by_eps, key=float):
        pts = [(table[f]["h_mm"], table[f]["eps"][key]["U"]) for f in forces if key in table[f]["eps"]]
        fit = richardson(pts)
        if "L1" not in fit:
            continue
        sq = [table[f]["eps"][key]["sigma_q"] for f in forces[-2:] if key in table[f]["eps"]]
        fit["sigma_f"] = max([abs(fit["L1"] - fit.get("L1_prime", fit["L1"]))] + [s for s in sq if math.isfinite(s)])
        fit["U_reg"] = fit["L1"]
        if reference is not None:
            fit["exact"] = reference[key]
            fit["minus_exact"] = fit["U_reg"] - reference[key]
            fit["finest_minus_exact"] = pts[-1][1] - reference[key]
        reg[key] = fit
    eps = np.array([float(k) * 1e3 for k in reg])
    u = np.array([reg[k]["U_reg"] for k in reg])
    s = np.array([max(reg[k]["sigma_f"], 1e-12) for k in reg])
    limit = _limit(eps, u, s)
    limit["forces"] = forces
    return {"table": table, "U_reg": reg, **limit}


def _limit(eps: np.ndarray, u: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if len(eps) < 3:
        return out
    w = 1.0 / s
    design = np.stack([np.ones_like(eps), eps, eps ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(design * w[:, None], u * w, rcond=None)
    residual = (design @ coef - u) * w
    dof = max(1, len(eps) - 3)
    chi2 = float(np.sum(residual ** 2) / dof)
    cov = np.linalg.inv((design * w[:, None]).T @ (design * w[:, None])) * max(1.0, chi2)
    small = np.argsort(eps)[:4]
    lin, *_ = np.linalg.lstsq((design[small, :2] * w[small, None]), u[small] * w[small], rcond=None)
    out.update(U0=float(coef[0]), slope_per_mm=float(coef[1]), curvature_per_mm2=float(coef[2]),
               se_U0=float(math.sqrt(cov[0, 0])), chi2_per_dof=chi2, U0_linear_four_smallest=float(lin[0]),
               rms=float(np.sqrt(np.mean((design @ coef - u) ** 2))))
    out["sigma"] = max(out["se_U0"], abs(out["U0"] - out["U0_linear_four_smallest"]), float(np.max(s)))
    out["limit"] = out["U0"]
    # the regularisation's order, a three-parameter fit U0 + a eps^p (reported, not a verdict)
    try:
        from scipy.optimize import least_squares

        fit = least_squares(lambda p: (p[0] + p[1] * eps ** p[2] - u) * w, [coef[0], coef[1], 1.0],
                            bounds=([-np.inf, -np.inf, 0.2], [np.inf, np.inf, 4.0]))
        out["power_fit"] = {"U0": float(fit.x[0]), "a": float(fit.x[1]), "order": float(fit.x[2]),
                            "rms": float(np.sqrt(np.mean((fit.x[0] + fit.x[1] * eps ** fit.x[2] - u) ** 2)))}
    except Exception as error:  # noqa: BLE001 - a report, not a verdict
        out["power_fit"] = {"error": repr(error)}
    return out


def combine(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if "limit" not in a or "limit" not in b:
        return {"claimed": False, "why": "a route has no limit"}
    diff = a["limit"] - b["limit"]
    bound = 2.0 * math.hypot(a["sigma"], b["sigma"])
    out: dict[str, Any] = {"a": a["limit"], "sigma_a": a["sigma"], "b": b["limit"], "sigma_b": b["sigma"],
                           "difference": diff, "two_combined_sigma": bound, "agree": abs(diff) <= bound}
    wa, wb = a["sigma"] ** -2, b["sigma"] ** -2
    out["mean"] = (a["limit"] * wa + b["limit"] * wb) / (wa + wb)
    out["sigma"] = max((wa + wb) ** -0.5, abs(diff) / 2.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    folder = Path(args.records)
    out: dict[str, Any] = {}
    refs = json.loads((folder / "sphere_references.json").read_text())["eps"] if (
        folder / "sphere_references.json").exists() else None

    # the sphere controls
    sa, sn = load(folder, "sphere_a_"), load(folder, "sphere_nn_")
    if sa:
        a = route_a(sa, "drag")
        a["passes"] = abs(a["limit"] - 1.0) <= max(2 * a["sigma"], 0.005)
        lines = eps_lines(sa, "drag")
        for key, line in lines.items():
            if refs and key in refs and line["finest"] is not None:
                line["finest_minus_exact"] = line["finest"] - refs[key]
        out["sphere"] = {"route_a": a, "eps_lines": lines}
    if sn and refs:
        b = route_b(sn, "drag", refs)
        if "limit" in b:
            b["passes_limit"] = abs(b["limit"] - 1.0) <= max(2 * b["sigma"], 0.005)
            b["passes_U_reg"] = all(abs(v["minus_exact"]) <= max(2 * v["sigma_f"], 1e-3 * v["exact"])
                                    for v in b["U_reg"].values())
            b["passes"] = b["passes_limit"] and b["passes_U_reg"]
            small = sorted(b["U_reg"], key=float)[:3]
            e = np.array([float(k) for k in small]) / (0.3 * 3.175e-3)
            exact_slope = np.polyfit(e, [refs[k] for k in small], 1)[0]
            got_slope = np.polyfit(e, [b["U_reg"][k]["U_reg"] for k in small], 1)[0]
            b["slope_over_a_exact_three_smallest"], b["slope_over_a_three_smallest"] = float(exact_slope), float(got_slope)
        out.setdefault("sphere", {})["route_b"] = b

    # A's body
    ha, hn = load(folder, "helix_a_"), load(folder, "helix_nn_")
    for name in ("axial", "R_xx", "R_xOx"):
        block: dict[str, Any] = {}
        if ha:
            block["route_a"] = route_a(ha, name)
            block["eps_lines"] = eps_lines(ha, name)
        if hn:
            block["route_b"] = route_b(hn, name)
            # post hoc (chosen after route (b)'s records were read): the eps at which every force set's solve is
            # finite and its R symmetric to 1e-5 (rounding below the discretisation), eps <= 0.146 mm
            block["route_b_post_hoc_eps_le_0.146mm"] = route_b(hn, name, eps_max=1.5e-4)
            if "route_a" in block and "limit" in block["route_b_post_hoc_eps_le_0.146mm"]:
                block["agreement_post_hoc"] = combine(block["route_a"], block["route_b_post_hoc_eps_le_0.146mm"])
        if "route_a" in block and "route_b" in block:
            block["agreement"] = combine(block["route_a"], block["route_b"])
            # the control on the body: the node's fixed-eps lines at the two largest eps against route (b)'s U_reg
            checks = {}
            for key in sorted(block["route_b"]["U_reg"], key=float)[-2:]:
                line = block["eps_lines"].get(key, {})
                if line.get("finest") is None or line.get("last_change") is None:
                    continue
                reg = block["route_b"]["U_reg"][key]
                bound = 2.0 * math.hypot(line["last_change"], reg["sigma_f"])
                checks[key] = {"line": line["finest"], "line_last_change": line["last_change"], "U_reg": reg["U_reg"],
                               "sigma_f": reg["sigma_f"], "difference": line["finest"] - reg["U_reg"],
                               "two_combined_sigma": bound, "agree": abs(line["finest"] - reg["U_reg"]) <= bound}
            block["control_on_body"] = checks
            # item 4 at the frozen point
            frozen = ha.get("0.1600", {}).get("blobs", {}).get(FROZEN_BLOB)
            eps0 = min(block["route_b"]["U_reg"], key=lambda k: abs(float(k) - 0.5 * 1.4571595021224538e-4))
            if frozen is not None:
                node = quantity(frozen, name)
                reg = block["route_b"]["U_reg"][eps0]
                ref = block["agreement"]
                block["item4"] = {"node_at_frozen": node, "U_reg_eps0": reg["U_reg"], "sigma_U_reg": reg["sigma_f"],
                                  "reference": ref["mean"], "sigma_reference": ref["sigma"],
                                  "quadrature_error": node - reg["U_reg"], "sigma_quadrature": reg["sigma_f"],
                                  "regularisation_error": reg["U_reg"] - ref["mean"],
                                  "sigma_regularisation": math.hypot(reg["sigma_f"], ref["sigma"]),
                                  "quadrature_percent": 100 * (node - reg["U_reg"]) / ref["mean"],
                                  "regularisation_percent": 100 * (reg["U_reg"] - ref["mean"]) / ref["mean"],
                                  "regularisation_from_route_b_limit": reg["U_reg"] - block["route_b"].get("limit", math.nan)}
                if "agreement_post_hoc" in block:
                    ph = block["agreement_post_hoc"]
                    block["item4_post_hoc"] = {"reference": ph["mean"], "sigma_reference": ph["sigma"],
                                               "regularisation_error": reg["U_reg"] - ph["mean"],
                                               "sigma_regularisation": math.hypot(reg["sigma_f"], ph["sigma"]),
                                               "regularisation_percent": 100 * (reg["U_reg"] - ph["mean"]) / ph["mean"],
                                               "quadrature_percent": 100 * (node - reg["U_reg"]) / ph["mean"],
                                               "regularisation_from_route_b_limit": reg["U_reg"] - block[
                                                   "route_b_post_hoc_eps_le_0.146mm"]["limit"]}
        out[f"helix_{name}"] = block
    Path(args.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: {kk: (vv.get("limit"), vv.get("sigma")) for kk, vv in v.items()
                          if isinstance(vv, dict) and "limit" in vv} for k, v in out.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
