"""Verify the local 3-config sweep manifest before launching H100."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np


REQUIRED_BASE = [
    "label", "lambda_", "Re", "Wo",
    "K_FVM_fine", "K_FVM_coarse", "K_Happel",
    "wall_time_fine_s", "wall_time_coarse_s",
    "N_cells_fine", "N_cells_coarse",
    "WSS_mean_Pa", "WSS_max_Pa", "WSS_std_Pa",
    "piso_iters_fine", "piso_iters_coarse",
    "u_centreline", "u_mean_cross",
]
REQUIRED_PULSATILE = [
    "Fz_peak", "Fz_mean", "Fr_rms", "phase_lag_rad", "phase_lag_deg",
]
# gnn_correction_* allowed to be None when no local corrector is present


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sweep_manifest_test")
    manifest_path = out_dir / "results_manifest.json"
    print("=" * 78)
    print(f"Manifest verification — {manifest_path}")
    print("=" * 78)
    if not manifest_path.exists():
        print("FAIL — manifest not found")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)
    n = len(manifest)
    print(f"  entries: {n}")

    failures = []
    for entry in manifest:
        label = entry.get("label", "<?>")
        for key in REQUIRED_BASE:
            if key not in entry:
                failures.append(f"{label}: missing {key}")
                continue
            v = entry[key]
            if v is None:
                failures.append(f"{label}: None for {key}")
            elif isinstance(v, float) and np.isnan(v):
                failures.append(f"{label}: NaN for {key}")

        # Pulsatile: required only when Wo > 0
        if entry.get("Wo", 0.0) > 0.0:
            for key in REQUIRED_PULSATILE:
                if key not in entry:
                    failures.append(f"{label}: missing pulsatile {key}")
                elif entry[key] is None:
                    failures.append(f"{label}: None for pulsatile {key}")

        # Physics sanity
        if entry.get("K_FVM_fine") is not None:
            if entry["K_FVM_fine"] <= 0:
                failures.append(f"{label}: K_FVM_fine ≤ 0")
            elif entry["K_FVM_fine"] < 1.0:
                failures.append(f"{label}: K_FVM_fine < 1 (unphysical)")
        if entry.get("K_FVM_coarse") is not None and entry["K_FVM_coarse"] <= 0:
            failures.append(f"{label}: K_FVM_coarse ≤ 0")
        if entry.get("WSS_mean_Pa") is not None and entry["WSS_mean_Pa"] <= 0:
            failures.append(f"{label}: WSS_mean_Pa ≤ 0")
        if entry.get("wall_time_fine_s") is not None and entry["wall_time_fine_s"] <= 0:
            failures.append(f"{label}: wall_time_fine_s ≤ 0")

    # Saved files exist
    for entry in manifest:
        label = entry["label"]
        prof = out_dir / f"{label}_velocity_profile.npy"
        if not prof.exists():
            failures.append(f"{label}: velocity profile npy missing")
        if entry.get("Wo", 0.0) > 0.0:
            for tag in ("Fz_waveform", "Fr_waveform"):
                p = out_dir / f"{label}_{tag}.npy"
                if not p.exists():
                    failures.append(f"{label}: {tag} npy missing")

    if failures:
        print("\nFAIL — checks not met:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(2)

    print("\nManifest verification: ALL CHECKS PASSED")
    print(f"Configs verified: {n}")
    print()
    for entry in manifest:
        wo = entry.get("Wo", 0.0)
        wo_str = f"Wo={wo:>3.1f}" if wo > 0 else "steady"
        print(f"  {entry['label']:50s}  λ={entry['lambda_']:.2f}  "
              f"Re={entry['Re']:4d}  {wo_str}  "
              f"K_fine={entry['K_FVM_fine']:.3f}  "
              f"K_coarse={entry['K_FVM_coarse']:.3f}  "
              f"WSS_mean={entry['WSS_mean_Pa']:.3e} Pa  "
              f"t_fine={entry['wall_time_fine_s']:.1f}s")


if __name__ == "__main__":
    main()
