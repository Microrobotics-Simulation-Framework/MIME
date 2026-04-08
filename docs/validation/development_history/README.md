# Development History — Verification & Validation Evidence

This directory contains research scripts from the development of MIME's
confined hydrodynamic drag solver. Each subdirectory documents a
specific approach that was explored, validated, and either adopted or
superseded by a better method.

These scripts are **verification evidence**, not dead code. They prove
that alternatives were systematically tested and document the reasoning
behind architectural decisions. When a reviewer asks "did you try X?",
the answer is here.

## Directory Index

### `bem_mfs_exploration/`
Method of Fundamental Solutions (MFS) and Liron-Shahar cylindrical
Green's function development. The MFS approach (discrete wall sources)
was explored first (Phase 0) and found to have a fundamental Fx-Fz
source-ratio trade-off at accessible resolutions. This motivated the
analytical Liron-Shahar approach (Phase 1), which achieves <4% on all
6 resistance matrix entries for a sphere at κ=0.3.

- `experiment_mfs_phase0.py` — Phase 0: discrete MFS sphere-in-cylinder
- `experiment_helix_phase1.py` — Phase 1: helix confined drag + convergence study
- `experiment_wall_matvec.py` — Phase 2: discrete MFS at 5k-10k pts (confirms non-convergence)
- `test_cylinder_greens.py` — Cylindrical Green's function unit tests
- `validate_calibrated.py` — Calibrated IB-BEM wall correction (superseded by twin-LBM)

### `defect_correction/`
IB-BEM defect correction approach: use LBM with immersed boundary to
compute the wall correction, subtract the free-space BEM contribution.
Explored extensively with multiple variants (blob kernels, matched
regularisation, ratio methods). Found to be direction-dependent due to
Peskin delta anisotropy — this was the key finding that motivated the
pivot to BEM+Liron-Shahar.

- `experiment_bb_defect.py` — Bouzidi BB body defect correction
- `experiment_bb_mismatch.py` — Free-space BB-BEM mismatch diagnostic
- `experiment_bb_onepass.py` — One-pass BB (no iteration)
- `experiment_blob_kernel.py` — Matched blob kernel (Cortez regularised)
- `experiment_convergence.py` — Resolution convergence 48³-128³
- `experiment_matched_eps.py` — Matched regularisation experiments
- `experiment_ratio.py` — Drag ratio approach (K=u_walled/u_free)
- `experiment_single_method.py` — Single method for all R entries
- `diagnose_velocity_arrival.py` — Velocity development diagnostics
- `validate_ib_bem_mismatch.py` — IB-BEM mismatch quantification
- `run_defect_correction_validation.py` — Resolution sweep (48³-128³)

### `schwarz_coupling/`
Schwarz decomposition: BEM body + LBM wall with iterative coupling.
Multiple variants tested (Bouzidi IBB, Faxén force extraction, open
BCs, iterated coupling). Works qualitatively but the IB-LBM wall
correction inherits the Peskin delta direction-dependence.

- `experiment_schwarz_bb.py` — Schwarz with BB at rest
- `experiment_schwarz_bouzidi.py` — Schwarz with Bouzidi IBB
- `experiment_schwarz_diag.py` — Diagnostics + Poiseuille test
- `experiment_schwarz_faxen.py` — Faxén force extraction variant
- `experiment_schwarz_iterated.py` — Two-way iterative coupling
- `experiment_schwarz_openbc.py` — Corrected open BCs
- `experiment_schwarz_v2.py` — One-pass no-body Schwarz (cleanest)
- `run_schwarz_sweep.py` — VER-029 drag multiplier sweep

### `lbm_methods/`
LBM-specific method exploration: periodic boundary conditions, twin-LBM
ratio approach, fixed-step convergence.

- `experiment_periodic_corrected.py` — Periodic ratio with mean-velocity correction
- `experiment_periodic_long.py` — Periodic stability at long times
- `experiment_periodic_ratio.py` — Periodic box twin-LBM ratio
- `validate_fixed_steps.py` — Fixed-step defect correction

### `cloud_infrastructure/`
Cloud GPU validation scripts and job configurations from the A40/H100
validation campaigns.

- `validate_3configs.py` — 3 wall correction configs comparison
- `validate_3kernel.py` — Triton vs JAX backend on A40
- `a40_*.yaml` — A40 cloud job configurations
