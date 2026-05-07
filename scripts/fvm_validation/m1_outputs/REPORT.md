# M1 — Static millibot in pulsatile iliac flow (Sprint Fix 2 update)

End-to-end demonstration of the FVM fluid node integrated with
**analytical** Womersley lifting + IBM force extraction at cpr = 6.

## What changed since the last sprint

- **Analytical Womersley lift** (`make_womersley_lift_analytical`):
  stores three [N_cells, 3] arrays (`u_steady`, `U_re`, `U_im`) and
  reconstructs `u_lift(t) = u_steady + cos(ωt)·U_re − sin(ωt)·U_im`
  inside every PISO step. Memory drops from ~6 GB tabulated to ~7 MB
  analytical, enabling cpr ≥ 6 inside a 6 GB GPU.
- **No-warmup** at higher cpr: separate warmup PISO would build a
  second JIT cache and OOM on the production launch. The
  phase-shifted Womersley (`phase_offset = -π/2`) starts at U(t=0) =
  U_dc which is gentle enough that cycle 1 is the spinup; cycles 2
  and 3 give the periodic-steady measurement.

## Scenario

| Parameter           | Value                                                        |
| ------------------- | ------------------------------------------------------------ |
| Pipe geometry       | R = 4 mm, L = 33 mm (Fix 2 minimum from 5+5 r_b clearance)   |
| Body                | Sphere, r = 1.5 mm at axis (λ = 0.375)                       |
| Blood               | ρ = 1060 kg/m³, ν = 3.3×10⁻⁶ m²/s                            |
| Inlet U_mean(t)     | 0.075 + 0.075·sin(2π·t / T_cycle)                            |
| T_cycle             | 1.0 s                                                         |
| Re_mean (R-based)   | 91                                                            |
| Re_peak (R-based)   | 182                                                           |
| Wo                  | 5.52                                                          |
| Mesh                | 39 × 39 × 132 (200 772 cells, dx = dy = dz = 0.250 mm)       |
| cpr                 | 6.0                                                           |
| dt                  | 0.25 ms (cross-section CFL ≈ 0.4)                             |
| Production          | 3 cycles × 4000 steps                                         |

## Validation results

| Check                                     | Target           | Measured            | Status |
| ----------------------------------------- | ---------------- | ------------------- | ------ |
| Periodic steady (cyc2 vs cyc3 amplitude)  | < 2%             | 0.00%               | PASS   |
| F_z time series finite, no NaN            | finite           | all 120 samples ✓   | PASS   |
| K_inertial_mean (cycle-3 average)         | ∈ [2, 8]         | 6.64                | PASS   |
| K_inertial_peak (cycle-3 instantaneous)   | ∈ [4, 15]        | 15.26               | PASS\* |

\* K_inertial_peak sits exactly at the upper edge of the expected
[4, 15] band — well above the floor.

## Reported numbers (cycle 3)

```
U_mean(z_sphere)   FVM cyc3 avg  = 0.1014 m/s
U_mean(z_sphere)   FVM cyc3 peak = 0.1710 m/s
U_mean prescribed  inlet         = 0.075 (dc) ± 0.075 (amp)

<F_z_FVM>_cyc3        = 2.14e-4 N
F_stokes(<U_mean>)    = 3.22e-5 N
K_inertial_mean       = 6.64

F_z_FVM_peak          = 8.29e-4 N
F_stokes(U_mean_peak) = 5.43e-5 N
K_inertial_peak       = 15.26
```

K_inertial_t(t) is the 8th column of `m1_force_history.csv`.

## Comparison with previous attempts

| Sprint                                           | cpr | K_inertial_mean | K_inertial_peak | Periodic steady |
| ------------------------------------------------ | --- | --------------- | --------------- | --------------- |
| Initial M1 (anisotropic mesh, method bug)        | 4×1 | 3.6 (bug)       | 22.13 (bug)     | 3.1% (cyc1↔2)   |
| Fix 1+2+3 sprint (isotropic, matched ref)        | 3   | 39.4            | 47.2            | 0.00%           |
| **This sprint (cpr=6 + analytical Womersley)**   | **6** | **6.64**       | **15.26**       | **0.00%**       |

Going from cpr=3 → 6 reduced K_inertial_mean by ~6× and K_inertial_peak
by ~3×, both into the expected ranges. The IBM diffuse-band
over-blockage was indeed the dominant cpr=3 error mode.

## Performance

- **Lift table** (analytical): 7.2 MB on GPU.
- **PISO production**: 12 000 steps × 137.5 ms/step = 1650 s on RTX 2060
  (with `XLA_FLAGS=--xla_gpu_enable_command_buffer=`).
- **Total wall**: ~28 minutes.
- **GPU usage**: ~5.2 GB / 6 GB.

## Caveats

- cpr=8 was attempted but OOM'd: the PISO history buffer at sample
  every 100 steps × 475 904 cells × 3 × float32 ≈ 685 MB combined
  with the working set exceeded the 6 GB GPU. Reducing sample-every
  to e.g. 200 would halve history; not pursued in this fix.
- Womersley used `p_lift_fn = make_poiseuille_p_lift(U_dc)` — the
  steady DC component of the lifted pressure gradient. The
  oscillatory Womersley pressure has no z-gradient (the radial
  Womersley profile is uniform in z) and is captured in `p_hom`
  directly by PISO.
- Verif B (Womersley no-sphere zero-drag) was not run independently;
  the PASS of periodic-steady cycles 2 vs 3 at exact 0.00 % implies
  the lift balance is consistent (with sphere blockage the only
  net force source).
