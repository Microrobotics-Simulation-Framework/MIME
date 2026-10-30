"""Phase 7j-a's surrogate rungs at A's operating point, on the CPU: the surrogate's resistance alone and with its
lubrication at the floored pose (static), the full dynamics (script A's run) with the lubrication on and off, the
time step's convergence, the contact margin's sweep, and the dynamics' finite differences across the budget's ranges
(the surrogate rung is ``piecewise_jump``, so its sensitivities are finite differences, never ``fim``'s derivative).

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/sim2real/surrogate.py --part static --out DIR [--weights NAME]

The surrogate's weights are ``dejongh_mlp_v2`` (trained through the 2026-05-21 tables, MIME-ANO-117) until phase 7i
ships its retrained ones; ``--weights`` re-runs every surrogate row with another asset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import operating_point as op  # noqa: E402

HERE = Path(__file__).resolve().parent
STATIC_MARGINS_MM = (0.02, 0.05, 0.1, 0.125, 0.15, 0.2)
DT_MS = (1.0, 0.5, 0.25, 0.125)
DT_MS_WITHOUT_LUBRICATION = (0.5, 0.25)
MARGINS_MM = (0.02, 0.05, 0.1, 0.2)
#: The BEM's dynamics run is this long (its cost), so the surrogate is also run this long, to read the short
#: window's bias against its own 5 s run.
SHORT_DURATION_S = 1.5


def part_static(args: argparse.Namespace) -> dict[str, Any]:
    import jax

    out: dict[str, Any] = {"weights": args.weights, "cases": []}
    for lubrication in (True, False):
        for margin in STATIC_MARGINS_MM:
            built = op.experiment(weights=args.weights, lubrication=lubrication, margin_mm=margin).build()
            solve = jax.jit(lambda pose, b=built: op.static_resistance(b, pose))
            offset = float(op.floored_offset(op.leaves_of(built)[op.BODY_NODE]))
            rows, matrices = [], []
            for phase in op.PHASES:
                r = np.asarray(solve(op.floored_pose(np.float64(offset), phase)), dtype=np.float64)
                matrices.append(r.tolist())
                rows.append({k: float(v) for k, v in op.speeds(r).items()})
            case: dict[str, Any] = {"lubrication": lubrication, "margin_mm": margin, "offset_m": offset,
                                    "offset_over_R": offset / op.TUBE_RADIUS_M, "per_phase": rows,
                                    "resistance_phase0": matrices[0]}
            for variant in op.VARIANTS:
                case[variant] = op.phase_mean([row[variant] for row in rows])
            out["cases"].append(case)
    return out


def _run(label: str, duration: float = op.DURATION_S, **settings: Any) -> dict[str, Any]:
    dt = settings.get("dt_ms", op.DT_MS)
    built = op.experiment(**settings).build()
    found = op.run_steady(built, dt, duration)
    found.update(label=label, settings=settings, duration_s=duration)
    print(f"{label}: {found['steps']} steps in {found['run_seconds']:.1f} s", flush=True)
    return found


def part_dynamics(args: argparse.Namespace) -> dict[str, Any]:
    """Script A with the lubrication on and off; and the bit-for-bit check's control, script A with its moment one
    unit in the last place larger, whose trajectory's hash must differ from script A's."""
    runs = [_run(f"lubrication {on}", weights=args.weights, lubrication=on) for on in (True, False)]
    plant = _run("control: the moment one ulp larger", weights=args.weights, lubrication=True,
                 moment=float(np.nextafter(2 * 8.4e-4, 1.0)))
    return {"weights": args.weights, "runs": runs, "control_one_ulp": plant}


def part_dt(args: argparse.Namespace) -> dict[str, Any]:
    runs = [_run(f"dt {dt} ms, lubrication on", weights=args.weights, lubrication=True, dt_ms=dt)
            for dt in DT_MS if dt != op.DT_MS]
    runs += [_run(f"dt {dt} ms, lubrication off", weights=args.weights, lubrication=False, dt_ms=dt)
             for dt in DT_MS_WITHOUT_LUBRICATION if dt != op.DT_MS]
    return {"weights": args.weights, "runs": runs}


def part_margin(args: argparse.Namespace) -> dict[str, Any]:
    return {"weights": args.weights, "runs": [
        _run(f"margin {margin} mm, lubrication {on}", weights=args.weights, lubrication=on, margin_mm=margin)
        for on in (True, False) for margin in MARGINS_MM if margin != op.MARGIN_MM]}


def part_short(args: argparse.Namespace) -> dict[str, Any]:
    return {"weights": args.weights, "runs": [
        _run(f"{SHORT_DURATION_S} s, lubrication {on}", SHORT_DURATION_S, weights=args.weights, lubrication=on)
        for on in (True, False)]}


def part_fd(args: argparse.Namespace) -> dict[str, Any]:
    """Each dynamics input at the ends of its range, the others at A's values, lubrication on (script A)."""
    inputs = json.loads((HERE / "budget_inputs.json").read_text())["dynamics"]
    mapping = {"moment": lambda v: {"moment": v}, "amplitude": lambda v: {"amplitude_mT": v * 1e3},
               "body_density": lambda v: {"body_density": v}, "water_density": lambda v: {"water_density": v},
               "viscosity": lambda v: {"viscosity_mPa_s": v * 1e3},
               "transverse_inertia_scale": lambda v: {"inertia_scale": (1.0, v, v)}}
    runs: dict[str, Any] = {}
    for name, settings in mapping.items():
        spec = inputs[name]
        runs[name] = {end: _run(f"{name} at {end} ({spec[end]})", weights=args.weights, **settings(spec[end]))
                      for end in ("low", "high") if spec[end] != spec["nominal"]}
    # body_density's nominal is A's m_eff through a density: the run at A's own declaration is the ladder's.
    runs["body_density"]["nominal_by_density"] = _run("body_density at nominal through the density",
                                                      weights=args.weights,
                                                      body_density=inputs["body_density"]["nominal"])
    return {"weights": args.weights, "inputs": inputs, "runs": runs}


def part_groups(args: argparse.Namespace) -> dict[str, Any]:
    """The coupling groups of every rung's graph at A's operating point, to be checked for convergence before a
    sensitivity is read through them (MADDENING's IFT rule is exact only at a converged fixed point); and that this
    phase's experiment at its defaults is script A's reproduction arm (with its control: a margin moved by 0.1 um
    must differ)."""
    found: dict[str, Any] = {}
    for rung in ("surrogate", "bem"):
        built = op.experiment(rung).build()
        groups = list(getattr(built.graph, "_coupling_groups"))  # MADDENING keeps them private; read only
        found[rung] = {"coupling_groups": len(groups), "groups": [str(g) for g in groups]}
    same = op.same_as_script_a(op.experiment(weights=args.weights).build())
    planted = op.same_as_script_a(op.experiment(weights=args.weights, margin_mm=0.1001).build())
    return {"rungs": found, "same_as_script_a": {"differences": same, "is_script_a": not same,
                                                 "control_margin_plant_caught": bool(planted),
                                                 "control_differences": planted}}


PARTS = {"groups": part_groups, "static": part_static, "dynamics": part_dynamics, "dt": part_dt, "margin": part_margin,
         "short": part_short, "fd": part_fd}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--part", choices=sorted(PARTS), required=True)
    parser.add_argument("--out", required=True, help="the directory the part's JSON is written to")
    parser.add_argument("--weights", default=op.DEFAULT_WEIGHTS, help="the surrogate's weights (an asset's name)")
    parser.add_argument("--tag", default="", help="appended to the output's name (a repeat run for the bit-for-bit "
                                                  "check)")
    args = parser.parse_args()
    op.require_float64()
    started = time.perf_counter()
    out = PARTS[args.part](args)
    out.update(part=args.part, environment=op.environment(), seconds=time.perf_counter() - started)
    op.write_json(Path(args.out) / f"surrogate_{args.part}{args.tag}.json", out)


if __name__ == "__main__":
    main()
