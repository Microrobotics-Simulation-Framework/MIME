# M1 — Static millibot in pulsatile iliac flow

End-to-end demonstration of the FVM fluid node integrated with
Womersley lifting + IBM force extraction in a physiologically
representative iliac scenario.

## Scenario

| Parameter           | Value                                     |
| ------------------- | ----------------------------------------- |
| Pipe geometry       | R = 4 mm, L = 18 mm                       |
| Body                | Sphere, r = 1.5 mm at axis (λ = 0.375)    |
| Blood               | ρ = 1060 kg/m³, ν = 3.3×10⁻⁶ m²/s         |
| Inlet U_mean(t)     | 0.15 + 0.15·cos(2π·t / T_cycle)           |
| T_cycle             | 1.0 s                                      |
| Re_mean             | 364 (2R definition)                        |
| Re_peak             | 727 (peak systole)                         |
| Wo                  | 5.52                                       |
| Mesh                | 26 × 26 × 12 (8112 cells, dx_xy=0.37 mm)  |
| dt                  | 0.5 ms (CFL ≈ 0.81 cross at peak)         |
| n_cycles            | 2 (4000 steps total)                       |

## Validation results

| Check                                     | Target           | Measured           | Status |
| ----------------------------------------- | ---------------- | ------------------ | ------ |
| Periodic steady (cyc1 vs cyc2 amplitude)  | < 10%            | 3.1%               | PASS   |
| K_inertial = F_FVM_peak / F_BEM_peak      | > 1.15           | 22.13              | PASS   |
| F_z time series finite, no NaN            | finite           | all finite         | PASS   |

`F_BEM_peak = 6π μ r_b U_centre_peak K_Happel(λ=0.375)`
       = 6π · 3.498×10⁻³ · 1.5×10⁻³ · 0.60 · 3.211 = **1.91×10⁻⁴ N**

`F_FVM_peak` is the maximum |F_z| extracted by the momentum-deficit
estimator over the second cardiac cycle, with the time-dependent
driving body force `f(t) = 8ν U_mean(t) / R²` passed for the F_body /
F_wall cancellation (see `FLUID_NODE_CONTRACT.md` § "Known caveat").

## Notes on K_inertial

`K_inertial = 22` is consistent with the Re_peak = 727 regime where
inertial drag dominates Stokes drag by orders of magnitude. The
Schiller–Naumann correction for unconfined spheres at Re = 200 alone
predicts C_D / C_Stokes ≈ 8; confinement at λ = 0.375 amplifies this
further. The brief's criterion of K_inertial > 1.15 is a binary check
that the FVM solver captures inertial enhancement vs the linear-Stokes
BEM baseline — exceeded here by a factor of 19.

## F_z(t) waveform

See `m1_force_history.csv` (5 columns: t, F_z, F_x, F_y, |F|; 80 rows
sampled at 25 ms intervals over 2 s).

## Performance

- **PISO step**: 38.3 ms/step on RTX 2060 (with `XLA_FLAGS=--xla_gpu_enable_command_buffer=` to avoid CUDA-graph OOM at 8K cells).
- **Total wall-time**: 153 s for 4000 steps + 3.5 s lift table + ~6 s force extraction.
- **Memory**: lift table at 2000 slices × 8112 cells × 3 × float32 ≈ 195 MB on GPU; well within 6 GB budget after disabling CUDA command-buffer pre-allocation.
- **H100 estimate (extrapolation)**: at 256³ the dense pressure solver becomes the bottleneck; FFT backend is ~2× faster there. With native command-buffer support and no memory pressure, expect ~5–10 ms/step at this mesh size, dropping total wall-time to ~25 s.

## Caveats and follow-up

1. **Mesh sized for RTX 2060**: production runs should use cpr ≥ 6
   in cross-section (mesh ≈ 64 × 64 × 24 ≈ 100K cells) to bring the
   IBM diffuse band to under-r_b/3 at the body surface. This is
   feasible on H100; on RTX 2060 host-RAM and JIT working-set push
   us to the cpr = 4 floor used here.
2. **Disable CUDA command buffer** by exporting
   `XLA_FLAGS="--xla_gpu_enable_command_buffer="` when the lift table
   is large; without this we hit a graph-instantiation OOM during JIT.
3. **K_inertial absolute value not validated against high-fidelity
   reference**: the binary "> 1.15" check passes, but tying the
   absolute K to a literature value at this exact Re/Wo/λ requires a
   companion BEM-Stokeslet run with the same confined geometry —
   scoped in M3 / Schwarz-coupling work.
