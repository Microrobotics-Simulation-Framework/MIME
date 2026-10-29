"""The slice's table from its results: per readable radius and approach, the truncation, knots, size, time and cost.

    python3 summarize.py RESULTS_DIR [RATES_JSON]

Rates (USD per hour) default to Modal's as read on 2026-10-29: CPU core 0.0473, GiB of memory 0.008, A100-40GB 2.10;
the containers are the app's (8 cores and 16 GiB; the A100 with 8 cores and 32 GiB)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CPU_HOUR = 8 * 0.0473 + 16 * 0.008
GPU_HOUR = 2.10 + 8 * 0.0473 + 32 * 0.008


def first_level(rows: list[dict], key: str, bound: float) -> str:
    for row in rows:
        if row[key] <= bound:
            return str(row.get("count", row.get("k_max", row["level"])))
    return f">{rows[-1].get('count', rows[-1].get('k_max', rows[-1]['level']))}"


def main() -> None:
    results = Path(sys.argv[1])
    lines = ["| radius | approach | modes / nodes / k_max (1%) | (0.1%) | no-slip 1%: modes / nodes / k_max | knots "
             "rho x phi x z | read 2-axis | size GB | per-source CPU s | per-source A100 s | build CPU h (proj.) | "
             "cost CPU $ (proj.) | build A100 h (proj.) | cost A100 $ (proj.) |", "|" + "---|" * 15]
    controls = []
    for path in sorted(results.glob("slice_r*.json")):
        data = json.loads(path.read_text())
        r = data["radius"]
        gpu_path = results / f"gpu_r{r}.json"
        gpu = json.loads(gpu_path.read_text()) if gpu_path.exists() else {}
        for name in ("image", "remainder"):
            sweep = data[f"sweep_{name}"]
            c1, c2 = sweep["chosen"]["0.01"], sweep["chosen"]["0.001"]
            t1, t2 = c1["truncation"], c2["truncation"]
            no_slip = " / ".join(first_level(sweep[key], "no_slip", 1e-2) for key in ("modes", "nodes", "reach"))
            job = data["jobs"][name]
            knots = f"{job['radial']} x {job['angle']} x {job['axial']}"
            size = job["table_bytes_float64"] / 1e9
            cpu_s = job["seconds"]
            gpu_s = min(gpu[name]["seconds"]) if name in gpu else None
            build_cpu = job["radial"] * cpu_s / 3600
            build_gpu = job["radial"] * gpu_s / 3600 if gpu_s is not None else None
            two = data["knots"].get(f"two_axis_{name}")
            lines.append(
                f"| {r} | {name} | {t1['modes']} / {t1['wavenumbers']} / {t1['k_max']:.0f} "
                f"(err {c1['interior']:.1e}) | {t2['modes']} / {t2['wavenumbers']} / {t2['k_max']:.0f} "
                f"(err {c2['interior']:.1e}) | {no_slip} | {knots} | {two:.1e} | {size:.3g} | {cpu_s:.1f} | "
                f"{'' if gpu_s is None else f'{gpu_s:.1f}'} | {build_cpu:.3g} | {build_cpu * CPU_HOUR:.3g} | "
                f"{'' if build_gpu is None else f'{build_gpu:.3g}'} | "
                f"{'' if build_gpu is None else f'{build_gpu * GPU_HOUR:.3g}'} |")
        for axis in ("angle", "axial", "radius"):
            entry = data["knots"][axis]
            row = {"radius": r, "axis": axis,
                   **{label: f"{v['error']:.2g}{'' if v['fails'] else ' (PASSES)'}"
                      for label, v in entry["remainder_controls"].items()},
                   "image_best": entry["image"]["best"], "remainder_best": entry["remainder"]["best"],
                   "agreement": f"{entry['image_vs_remainder_plus_plane']:.1e}"}
            controls.append(row)
        old = data["sweep_image"].get("control_2026_05_21_truncation")
        controls.append({"radius": r, "control_2026_05_21": old,
                         "reference_bound_image": data["sweep_image"]["reference_bound"],
                         "reference_bound_remainder": data["sweep_remainder"]["reference_bound"],
                         "seconds": data["seconds"], "gpu_parity": {k: v.get("gpu_vs_cpu") for k, v in gpu.items()
                                                                   if isinstance(v, dict)}})
    print("\n".join(lines))
    print()
    for row in controls:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
