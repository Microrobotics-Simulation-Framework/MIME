"""Step 4 — H100 sweep runner with resumability.

Wraps :func:`step1_generate_data.run_one` for the full
:class:`GNNTrainingSweepConfig` (5 λ × 3 aspect × 6 Re × 4 Wo = 360
configs in M2). Each completed config writes a marker file; resuming
skips configs whose marker exists.

Usage
-----
Local dry-run:
    python step4_sweep_infrastructure.py --dry-run --n-configs 3

H100 full sweep (via SkyPilot — see sky_tasks/gnn_sweep.yaml):
    python step4_sweep_infrastructure.py --output-dir /data/gnn_sweep \\
        --cpr-fine 8 --cpr-coarse 4
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

# Reuse the proven Step 1 driver (fine + coarse + downsample + drag)
from step1_generate_data import (
    run_one, save_state_npz, downsample_to_coarse,
)
import numpy as np

from mime.nodes.environment.fvm import GNNTrainingSweepConfig


def all_sweep_configs(sweep: GNNTrainingSweepConfig):
    """Materialise the cartesian product implied by the sweep dataclass.

    Aspect ratio is intended for capsule bodies (a future extension);
    until the sphere-body driver supports capsules we ignore aspect
    and emit one steady-flow config per (λ, Re) pair plus one Wo-tag
    for traceability. Yields dicts matching ``run_one``'s kwargs.
    """
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
                             cpr_fine=8, cpr_coarse=4, n_steps=400):
    """Idempotent: skip if marker exists. Saves fine/coarse states +
    downsampled fine + per-config metrics. Atomically writes the marker
    last, so a crash mid-config triggers a re-run rather than a false
    skip on the next launch."""
    marker = output_dir / f"{cfg['label']}_done.txt"
    if marker.exists():
        print(f"  skip {cfg['label']} (marker exists)")
        return
    t0 = time.time()
    fine = run_one(lambda_=cfg["lambda_"], Re=cfg["Re"], cpr=cpr_fine,
                    label=cfg["label"], n_steps=n_steps)
    coarse = run_one(lambda_=cfg["lambda_"], Re=cfg["Re"], cpr=cpr_coarse,
                      label=cfg["label"], n_steps=n_steps,
                      U_dc=fine["U_dc"])

    save_state_npz(output_dir / f"{cfg['label']}_fine.npz",
                    fine["state"], fine["u_phys"], fine["mesh"])
    save_state_npz(output_dir / f"{cfg['label']}_coarse.npz",
                    coarse["state"], coarse["u_phys"], coarse["mesh"])
    u_fine_ds = downsample_to_coarse(fine["u_phys"], fine["mesh"], coarse["mesh"])
    p_fine_ds = downsample_to_coarse(fine["state"]["p"], fine["mesh"], coarse["mesh"])
    np.savez_compressed(
        output_dir / f"{cfg['label']}_fine_downsampled.npz",
        u=u_fine_ds, p=p_fine_ds,
        cartesian_shape=np.asarray(coarse["mesh"].cartesian_shape, dtype=np.int32),
        cartesian_spacing=np.asarray(coarse["mesh"].cartesian_spacing, dtype=np.float32),
    )

    elapsed = time.time() - t0
    metrics = dict(
        label=cfg["label"], lambda_=cfg["lambda_"], Re=cfg["Re"],
        aspect=cfg.get("aspect"), Wo=cfg.get("Wo"),
        K_FVM_fine=fine["K_FVM"], K_FVM_coarse=coarse["K_FVM"],
        K_Happel=fine["K_Happel"],
        cells_fine=fine["cells"], cells_coarse=coarse["cells"],
        wall_total_s=elapsed,
    )
    with open(marker, "w") as f:
        json.dump(metrics, f, indent=2)


def dry_run_sweep(n_configs: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = GNNTrainingSweepConfig()
    all_cfg = all_sweep_configs(sweep)
    print(f"  total sweep size = {len(all_cfg)} configs (M2 catalogue)")
    print(f"  dry-running first {n_configs}...")
    for i, cfg in enumerate(all_cfg[:n_configs]):
        print(f"\n[{i+1}/{n_configs}] {cfg['label']}")
        run_single_sweep_config(cfg, output_dir,
                                 cpr_fine=4, cpr_coarse=2,    # tiny for dry-run
                                 n_steps=100)
    # Print summary
    markers = sorted(output_dir.glob("sweep_*_done.txt"))
    print(f"\nDry run complete: {len(markers)} markers written in {output_dir}")
    for m in markers:
        with open(m) as f:
            metrics = json.load(f)
        print(f"  {metrics['label']}  K_FVM_fine={metrics['K_FVM_fine']:+.3f}  "
              f"K_Happel={metrics['K_Happel']:.3f}  "
              f"wall={metrics['wall_total_s']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-configs", type=int, default=3)
    ap.add_argument("--output-dir", default="data/gnn_sweep_dryrun")
    ap.add_argument("--cpr-fine", type=int, default=8)
    ap.add_argument("--cpr-coarse", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run_sweep(args.n_configs, out)
        return

    # Full sweep (intended for H100; resumable)
    sweep = GNNTrainingSweepConfig()
    all_cfg = all_sweep_configs(sweep)
    print(f"Full sweep: {len(all_cfg)} configs into {out}")
    for i, cfg in enumerate(all_cfg):
        print(f"\n[{i+1}/{len(all_cfg)}] {cfg['label']}")
        try:
            run_single_sweep_config(cfg, out,
                                     cpr_fine=args.cpr_fine,
                                     cpr_coarse=args.cpr_coarse)
        except Exception as e:
            print(f"  FAILED ({type(e).__name__}: {e}); marker NOT written, "
                  "next launch will retry")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
