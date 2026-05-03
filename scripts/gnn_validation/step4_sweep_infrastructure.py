"""H100-bound sweep runner with full data collection.

For each (λ, Re, Wo) config in the M2 catalogue:

  - fine + coarse Poiseuille runs (steady, K_FVM_fine/coarse)
  - if Wo > 0: pulsatile fine run with cycle-3 F_z(t), F_r(t),
    U_inlet(t) waveforms; phase lag, Fz_peak, Fz_mean, Fr_rms
  - if Re ≥ 100 (steady): Strouhal number from Fz history FFT
  - WSS mean / max / std at the pipe wall band (steady fine state)
  - radial velocity profile at sphere mid-plane (20 bins)
  - GNN correction norm + max on the coarse state
  - PISO step counts + wall times

Manifest: data/<output-dir>/results_manifest.json (atomic JSON).
Marker files:    data/<output-dir>/<label>_done.txt (legacy resumability)
Per-config arrays:
  <label>_fine.npz           — fine state (u_phys, p, mesh shape)
  <label>_coarse.npz         — coarse state
  <label>_fine_downsampled.npz — fine downsampled to coarse mesh
  <label>_Fz_waveform.npy   (Wo > 0)
  <label>_Fr_waveform.npy   (Wo > 0)
  <label>_velocity_profile.npy

Usage
-----
Local dry-run on 3 configs (cpr_fine=4, cpr_coarse=2 — small enough
to fit on RTX 2060):
    python step4_sweep_infrastructure.py --dry-run --n-configs 3 \\
        --output-dir data/sweep_manifest_test/

H100 full sweep (via SkyPilot — sky_tasks/gnn_sweep.yaml):
    python step4_sweep_infrastructure.py \\
        --output-dir /outputs/gnn_sweep --n-configs 360 \\
        --cpr-fine 8 --cpr-coarse 4
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))

from sweep_helpers import (
    happel_brenner, build_setup, k_FVM,
    run_steady, run_pulsatile, cycle_force_waveform,
    compute_wall_shear_stress, radial_velocity_profile,
    gnn_correction_diag, strouhal_from_history, phase_lag,
    append_to_manifest,
)
from step1_generate_data import save_state_npz, downsample_to_coarse

from mime.nodes.environment.fvm import GNNTrainingSweepConfig


def all_sweep_configs(sweep: GNNTrainingSweepConfig):
    out = []
    for lam in sweep.confinement_lambdas:
        for Re in sweep.reynolds_numbers:
            for ar in sweep.aspect_ratios:
                for Wo in sweep.womersley_numbers:
                    label = f"sweep_lam{lam:.2f}_Re{int(Re)}_ar{ar:.1f}_Wo{int(Wo)}"
                    out.append(dict(label=label, lambda_=lam, Re=Re,
                                     aspect=ar, Wo=Wo))
    return out


def run_single_sweep_config(cfg, output_dir: Path,
                             cpr_fine=8, cpr_coarse=4,
                             n_steps_steady=400,
                             corrector=None,
                             manifest_path: Path | None = None):
    """Idempotent: skip if marker exists. Atomically appends to manifest
    on completion. The state .npz files + waveform .npy + profile .npy
    are written before the marker."""
    label = cfg["label"]
    marker = output_dir / f"{label}_done.txt"
    if marker.exists():
        print(f"  skip {label} (marker exists)")
        return

    nu = 1e-3
    R_pipe = 1e-3 / cfg["lambda_"]
    U_dc = cfg["Re"] * nu / R_pipe

    # ---- Fine + coarse steady ----
    d_fine = build_setup(lambda_=cfg["lambda_"], cpr=cpr_fine,
                          U_dc=U_dc, Wo=0.0)
    state_fine, dt_fine, t_fine = run_steady(d_fine, n_steps=n_steps_steady)
    F_fine, K_fine = k_FVM(d_fine, state_fine)

    d_coarse = build_setup(lambda_=cfg["lambda_"], cpr=cpr_coarse,
                            U_dc=U_dc, Wo=0.0)
    state_coarse, dt_coarse, t_coarse = run_steady(d_coarse,
                                                    n_steps=n_steps_steady)
    F_coarse, K_coarse = k_FVM(d_coarse, state_coarse)

    # ---- Save state pytrees ----
    u_fine_phys = state_fine["u"] + d_fine["lift"].u_lift_static
    u_coarse_phys = state_coarse["u"] + d_coarse["lift"].u_lift_static
    save_state_npz(output_dir / f"{label}_fine.npz",
                    state_fine, u_fine_phys, d_fine["mesh"])
    save_state_npz(output_dir / f"{label}_coarse.npz",
                    state_coarse, u_coarse_phys, d_coarse["mesh"])
    u_fine_ds = downsample_to_coarse(u_fine_phys, d_fine["mesh"], d_coarse["mesh"])
    p_fine_ds = downsample_to_coarse(state_fine["p"], d_fine["mesh"], d_coarse["mesh"])
    np.savez_compressed(
        output_dir / f"{label}_fine_downsampled.npz",
        u=u_fine_ds, p=p_fine_ds,
        cartesian_shape=np.asarray(d_coarse["mesh"].cartesian_shape, dtype=np.int32),
        cartesian_spacing=np.asarray(d_coarse["mesh"].cartesian_spacing, dtype=np.float32),
    )

    # ---- WSS, velocity profile (fine steady) ----
    wss_mean, wss_max, wss_std = compute_wall_shear_stress(d_fine, state_fine)
    profile = radial_velocity_profile(d_fine, state_fine)
    np.save(output_dir / f"{label}_velocity_profile.npy", profile)

    # ---- GNN correction diagnostic on the coarse state ----
    gnn_norm, gnn_max = gnn_correction_diag(
        corrector, d_coarse, state_coarse, dt_coarse,
    )

    # ---- Strouhal (steady, Re ≥ 100) — single-window FFT placeholder ----
    # Without a per-step Fz history captured during the steady run, we
    # report None for now (would require run_piso_with_history of
    # ~last N steps). Cheap to add once steady runs are converted to
    # history-capturing — out of scope for the dry-run validation.
    f_shed = None; St = None

    metrics = dict(
        label=label,
        lambda_=cfg["lambda_"],
        Re=int(cfg["Re"]),
        Wo=float(cfg.get("Wo", 0.0)),
        aspect=cfg.get("aspect"),
        # Steady drag
        K_FVM_fine=K_fine,
        K_FVM_coarse=K_coarse,
        K_Happel=happel_brenner(cfg["lambda_"]),
        F_z_fine=F_fine,
        F_z_coarse=F_coarse,
        # Wall times
        wall_time_fine_s=t_fine,
        wall_time_coarse_s=t_coarse,
        N_cells_fine=d_fine["mesh"].N_cells,
        N_cells_coarse=d_coarse["mesh"].N_cells,
        # PISO step counts (fixed-step PISO; report the n_steps)
        piso_iters_fine=n_steps_steady,
        piso_iters_coarse=n_steps_steady,
        # WSS
        WSS_mean_Pa=wss_mean,
        WSS_max_Pa=wss_max,
        WSS_std_Pa=wss_std,
        # GNN correction
        gnn_correction_norm=gnn_norm,
        gnn_correction_max=gnn_max,
        # Velocity profile summary
        u_centreline=float(profile[0]),
        u_mean_cross=float(np.mean(profile)),
        # Strouhal placeholder
        shedding_freq_hz=f_shed,
        strouhal=St,
    )

    # ---- Pulsatile run for Wo > 0 ----
    # Use the COARSE mesh for pulsatile to keep PISO history within
    # GPU memory. The waveform is for cycle-averaged drag scoring;
    # cpr_coarse is sufficient. Fine pulsatile is a separate H100
    # phase if accuracy demands it.
    if cfg.get("Wo", 0.0) > 0.0:
        d_p = build_setup(lambda_=cfg["lambda_"], cpr=cpr_coarse,
                           U_dc=U_dc, Wo=cfg["Wo"])
        try:
            (state_p, hist_p, dt_p, wall_p,
             sample_every, n_per_cycle) = run_pulsatile(
                d_p, n_cycles=3, samples_per_cycle=20,
            )
            Fz_c3, Fr_c3, t_c3, U_c3 = cycle_force_waveform(
                d_p, hist_p, sample_every, n_per_cycle, cycle_idx=2,
            )
            np.save(output_dir / f"{label}_Fz_waveform.npy", Fz_c3)
            np.save(output_dir / f"{label}_Fr_waveform.npy", Fr_c3)
            phi = phase_lag(Fz_c3, U_c3)
            metrics.update(dict(
                Fz_peak=float(np.max(np.abs(Fz_c3))),
                Fz_mean=float(np.mean(Fz_c3)),
                Fr_rms=float(np.sqrt(np.mean(Fr_c3 ** 2))),
                phase_lag_rad=phi,
                phase_lag_deg=float(np.degrees(phi)),
                wall_time_pulsatile_s=wall_p,
            ))
        except Exception as e:
            print(f"  pulsatile FAIL ({type(e).__name__}: {e}); "
                  "marker will record steady-only metrics")
            metrics["pulsatile_error"] = str(e)

    # ---- Atomic manifest append ----
    if manifest_path is not None:
        append_to_manifest(manifest_path, metrics)

    # ---- Marker (idempotency / legacy) ----
    with open(marker, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-configs", type=int, default=3)
    ap.add_argument("--output-dir", default="data/sweep_manifest_test")
    ap.add_argument("--cpr-fine", type=int, default=8)
    ap.add_argument("--cpr-coarse", type=int, default=4)
    ap.add_argument("--n-steps-steady", type=int, default=400)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "results_manifest.json"

    # Optional GNN corrector
    corrector_path = Path("data/gnn_training/gnn_params_local.pkl")
    corrector = None
    if corrector_path.exists():
        with open(corrector_path, "rb") as f:
            corrector = pickle.load(f)
        print(f"  loaded local corrector: {corrector.param_count()} params")
    else:
        print("  no local corrector found; gnn_correction_norm = None")

    sweep = GNNTrainingSweepConfig()
    all_cfg = all_sweep_configs(sweep)
    print(f"  total catalogue: {len(all_cfg)} configs")

    if args.dry_run:
        cpr_f, cpr_c, n_s = 4, 2, 100
        n = args.n_configs
        print(f"  dry-run: first {n} configs at cpr_fine={cpr_f}, "
              f"cpr_coarse={cpr_c}, n_steps={n_s}")
    else:
        cpr_f, cpr_c, n_s = args.cpr_fine, args.cpr_coarse, args.n_steps_steady
        n = len(all_cfg)
        print(f"  full sweep: {n} configs at cpr_fine={cpr_f}, "
              f"cpr_coarse={cpr_c}, n_steps={n_s}")

    for i, cfg in enumerate(all_cfg[:n]):
        print(f"\n[{i+1}/{n}] {cfg['label']}", flush=True)
        try:
            run_single_sweep_config(
                cfg, out, cpr_fine=cpr_f, cpr_coarse=cpr_c,
                n_steps_steady=n_s, corrector=corrector,
                manifest_path=manifest_path,
            )
        except Exception as e:
            print(f"  FAILED ({type(e).__name__}: {e}); marker NOT written, "
                  "next launch will retry")

    # Final summary
    if manifest_path.exists():
        with open(manifest_path) as f:
            data = json.load(f)
        print(f"\nManifest at {manifest_path}: {len(data)} entries")
        for e in data[-min(len(data), n):]:
            print(f"  {e['label']:50s} K_fine={e['K_FVM_fine']:.3f} "
                  f"K_coarse={e['K_FVM_coarse']:.3f} "
                  f"WSS_mean={e['WSS_mean_Pa']:.3e}Pa "
                  f"t_fine={e['wall_time_fine_s']:.1f}s")


if __name__ == "__main__":
    main()
