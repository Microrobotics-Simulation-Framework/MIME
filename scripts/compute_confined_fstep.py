#!/usr/bin/env python3
"""T2.7: ODE-LBM coupling — confined step-out frequency predictions.

Uses T2.6 (simple BB) drag multipliers by default.
Run with --validated to use T2.6b (Bouzidi) multipliers once available.

All outputs are labelled PRELIMINARY when using T2.6 simple BB multipliers.

Usage:
    python3 scripts/compute_confined_fstep.py              # preliminary
    python3 scripts/compute_confined_fstep.py --validated   # after T2.6b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# JAX — CPU is fine for ODE integration
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import jax.numpy as jnp

from mime.nodes.robot.umr_ode import (
    fit_drag_coefficients,
    sweep_frequency,
    compute_step_out_frequency,
    params_dict_to_array,
)

# ---------------------------------------------------------------------------
# Drag multipliers from T2.6 (simple BB, 192^3)
# ---------------------------------------------------------------------------

DRAG_MULTIPLIERS_SIMPLE_BB = {
    0.15: 1.000,  # reference
    0.22: 1.127,
    0.30: 1.194,
    0.35: 1.284,  # held-out
    0.40: 1.394,
}

# Magnet configurations from de Boer 2025 Fig. 12 (d=2.8mm)
MAGNET_CONFIGS = [
    {"n_mag": 1, "f_step_paper": 128.0},
    {"n_mag": 2, "f_step_paper": 181.0},
    {"n_mag": 3, "f_step_paper": 222.0},
]

# Physical constants from deboer2025_params.md
M_SINGLE = 1.07e-3   # A*m^2 per magnet
B_FIELD = 3e-3        # T (simulation value from paper)

# Digitised data directory
DATA_DIR = Path("docs/validation/umr_deboer2025/deboer_fig12_digitised")
OUTPUT_DIR = Path("docs/validation/umr_deboer2025/figures")
JSON_DIR = Path("docs/validation/umr_deboer2025")


def load_drag_multipliers_from_hdf5(path: str) -> dict[float, float]:
    """Load drag multipliers from T2.6b Bouzidi HDF5 results."""
    import h5py
    multipliers = {}
    with h5py.File(path, 'r') as f:
        gt = f['ground_truth']
        # Find reference torque (ratio 0.15)
        ref_torque = None
        for ratio_key in sorted(gt.keys()):
            ratio = float(ratio_key)
            tz = gt[ratio_key]['drag_torque_z'][:]
            # Use first 192^3 sample (index 0)
            if ratio == 0.15 or (ref_torque is None and len(tz) > 0):
                ref_torque = float(tz[0])
                break
        if ref_torque is None or ref_torque == 0:
            raise ValueError("Could not find reference torque at ratio 0.15")
        for ratio_key in sorted(gt.keys()):
            ratio = float(ratio_key)
            tz = gt[ratio_key]['drag_torque_z'][:]
            if len(tz) > 0:
                multipliers[ratio] = float(tz[0]) / ref_torque
    return multipliers


def compute_confined_fstep(
    drag_multipliers: dict[float, float],
    validated: bool,
) -> list[dict]:
    """Compute confined step-out frequencies for all (ratio, n_mag) combos."""
    # Fit baseline drag coefficients from 1-mag digitised data
    data_1 = np.loadtxt(DATA_DIR / "d2.8_1mag.csv", delimiter=",", skiprows=1)
    base_params = fit_drag_coefficients(
        data_1, n_mag=1, m_single=M_SINGLE, B=B_FIELD, f_step=128.0,
    )

    results = []

    for cfg in MAGNET_CONFIGS:
        n_mag = cfg["n_mag"]
        f_step_paper = cfg["f_step_paper"]

        # Unconfined baseline: analytical step-out
        C_rot_base = base_params["C_rot"]
        f_step_unconfined = compute_step_out_frequency(
            n_mag, M_SINGLE, B_FIELD, C_rot_base,
        )

        for ratio, mult_rot in sorted(drag_multipliers.items()):
            # Haberman-Sayre translational correction
            mult_trans = 1.0 / (1.0 - ratio**2)
            # Geometric mean for propulsion
            mult_prop = math.sqrt(mult_rot * mult_trans)

            # Scale drag coefficients
            confined_params = dict(base_params)
            confined_params["n_mag"] = float(n_mag)
            confined_params["C_rot"] = C_rot_base * mult_rot
            confined_params["C_trans"] = base_params["C_trans"] * mult_trans
            confined_params["C_prop"] = base_params["C_prop"] * mult_prop

            # Confined step-out frequency (analytical)
            f_step_confined = compute_step_out_frequency(
                n_mag, M_SINGLE, B_FIELD, confined_params["C_rot"],
            )

            # Frequency sweep to find step-out from phase dynamics
            f_max = f_step_confined * 1.5
            freqs = jnp.linspace(10.0, f_max, 200)
            confined_params["omega_field"] = 2.0 * math.pi * f_step_confined
            p_arr = params_dict_to_array(confined_params)

            dt = 1e-5
            t_settle = 0.5  # 500ms settling time
            speeds = sweep_frequency(p_arr, freqs, dt, t_settle)
            speeds_np = np.array(speeds)

            # Detect step-out: speed drops below 50% of peak
            peak_speed = float(np.max(speeds_np))
            if peak_speed < 1e-6:
                print(f"  WARNING: No measurable speed for n_mag={n_mag}, "
                      f"ratio={ratio}. Step-out may be suppressed.", flush=True)
                f_step_detected = None
            else:
                above_half = speeds_np > 0.5 * peak_speed
                # Find last frequency where speed is above half peak
                indices = np.where(above_half)[0]
                if len(indices) > 0:
                    f_step_detected = float(freqs[indices[-1]])
                else:
                    f_step_detected = None
                    print(f"  WARNING: Could not detect step-out for n_mag={n_mag}, "
                          f"ratio={ratio}", flush=True)

            shift_pct = (1.0 - f_step_confined / f_step_unconfined) * 100.0

            results.append({
                "n_mag": n_mag,
                "ratio": ratio,
                "f_step_unconfined_hz": round(f_step_unconfined, 1),
                "f_step_paper_hz": f_step_paper,
                "f_step_confined_hz": round(f_step_confined, 1),
                "shift_pct": round(shift_pct, 1),
                "mult_rot": round(mult_rot, 3),
                "mult_trans": round(mult_trans, 3),
                "mult_prop": round(mult_prop, 3),
                "peak_speed_m_s": round(peak_speed, 4),
                "validated": validated,
            })

    return results


def print_table(results: list[dict], validated: bool) -> None:
    """Print results table."""
    label = "VALIDATED (Bouzidi IBB)" if validated else "PRELIMINARY (simple BB)"
    print(f"\n{'='*85}")
    print(f"Confined step-out frequency predictions — {label}")
    if not validated:
        print("Bouzidi re-run (T2.6b) pending — predictions will be updated")
    print(f"{'='*85}")
    print(f"{'n_mag':>5} {'ratio':>6} {'f_unconf':>9} {'f_paper':>8} "
          f"{'f_conf':>7} {'shift':>7} {'m_rot':>6} {'m_trans':>8} "
          f"{'peak_U':>8}")
    print(f"{'':>5} {'':>6} {'(Hz)':>9} {'(Hz)':>8} {'(Hz)':>7} {'(%)':>7} "
          f"{'':>6} {'':>8} {'(m/s)':>8}")
    print("-" * 85)
    for r in results:
        f_conf = f"{r['f_step_confined_hz']:.1f}" if r['f_step_confined_hz'] else "N/A"
        print(f"{r['n_mag']:>5} {r['ratio']:>6.2f} {r['f_step_unconfined_hz']:>9.1f} "
              f"{r['f_step_paper_hz']:>8.1f} {f_conf:>7} {r['shift_pct']:>7.1f} "
              f"{r['mult_rot']:>6.3f} {r['mult_trans']:>8.3f} "
              f"{r['peak_speed_m_s']:>8.4f}")


def save_figure(results: list[dict], validated: bool) -> str:
    """Save confined f_step figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping figure")
        return ""

    suffix = "validated" if validated else "preliminary"
    label = "VALIDATED (Bouzidi IBB)" if validated else "PRELIMINARY (simple BB)"

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for cfg in MAGNET_CONFIGS:
        n_mag = cfg["n_mag"]
        subset = [r for r in results if r["n_mag"] == n_mag]
        ratios = [r["ratio"] for r in subset]
        f_conf = [r["f_step_confined_hz"] for r in subset]
        f_unconf = subset[0]["f_step_unconfined_hz"] if subset else 0

        ax.plot(ratios, f_conf, 'o-', label=f'{n_mag} magnet(s)', markersize=6)
        ax.axhline(f_unconf, linestyle='--', alpha=0.3)

    ax.set_xlabel('Confinement ratio (R_umr / R_vessel)')
    ax.set_ylabel('Step-out frequency (Hz)')
    ax.set_title(f'Confined step-out frequency predictions — {label}')
    if not validated:
        ax.text(0.5, 0.02,
                'Preliminary — simple BB boundary conditions. Bouzidi re-run pending.',
                transform=ax.transAxes, ha='center', fontsize=9, style='italic',
                color='red')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = OUTPUT_DIR / f"confined_fstep_{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {path}")
    return str(path)


def save_json(results: list[dict], validated: bool) -> str:
    """Save results JSON."""
    suffix = "validated" if validated else "preliminary"
    os.makedirs(JSON_DIR, exist_ok=True)
    path = JSON_DIR / f"confined_fstep_{suffix}.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {path}")
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="T2.7: Confined step-out predictions")
    parser.add_argument("--validated", action="store_true",
                        help="Use T2.6b Bouzidi multipliers from HDF5")
    parser.add_argument("--hdf5", default="data/umr_training_v2_bouzidi.h5",
                        help="Path to T2.6b Bouzidi HDF5 (used with --validated)")
    args = parser.parse_args()

    if args.validated:
        if not Path(args.hdf5).exists():
            print(f"ERROR: --validated requires {args.hdf5} but file not found")
            sys.exit(1)
        print(f"Loading Bouzidi drag multipliers from {args.hdf5}")
        drag_multipliers = load_drag_multipliers_from_hdf5(args.hdf5)
    else:
        drag_multipliers = DRAG_MULTIPLIERS_SIMPLE_BB

    print(f"Drag multipliers: {drag_multipliers}")
    results = compute_confined_fstep(drag_multipliers, args.validated)
    print_table(results, args.validated)
    save_figure(results, args.validated)
    save_json(results, args.validated)


if __name__ == "__main__":
    main()
