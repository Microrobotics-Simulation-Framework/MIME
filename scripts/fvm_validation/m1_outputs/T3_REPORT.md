# T3 — Confined-Stokes drag (after Fix 1 p_lift reconstruction + cpr sweep)

Re-run after Fix 1 added `p_lift_fn` and `U_mean_analytical` to
`momentum_deficit_drag`. Verification A (no-sphere zero-drag) now
passes at machine precision (`F_md = -5.7×10⁻¹⁴ N`, ratio 0.0002 %).

## cpr sweep results

| λ    | cpr | mesh                | cells       | K_FVM   | K_Happel | err     |
| ---- | --- | ------------------- | ----------- | ------- | -------- | ------- |
| 0.1  | 4   | 96 × 96 × 88        |   811 008   | 0.0124  | 1.263    | 99.0%   |
| 0.1  | 6   | 144 × 144 × 132     | 2 737 152   | 0.0136  | 1.263    | 98.9%   |
| 0.1  | 8   | 192 × 192 × 176     | 6 488 064   | OOM (~6.4 GB constant alloc) — RTX 2060 |
| 0.3  | 4   | 32 × 32 × 88        |    90 112   | 0.0148  | 2.370    | 99.4%   |
| 0.3  | 6   | 48 × 48 × 132       |   304 128   | 0.0159  | 2.370    | 99.3%   |
| 0.3  | 8   | 64 × 64 × 176       |   720 896   | 0.0180  | 2.370    | 99.2%   |

K_FVM is **positive** (Fix 1 succeeded — sign is right, the missing
lifted-pressure contribution is now reconstructed) but the magnitude
is only ~1 % of K_Happel and refinement from cpr=4 → 8 only doubles
K_FVM. At this convergence rate cpr ≈ 1000 would be needed, which is
nonphysical — pointing to a deeper systematic issue in the PISO +
lifting + IBM interaction, not a resolution problem.

## Diagnosis

Verification A (no-sphere baseline) reads exactly zero, so the
*formula* is correct. The sphere case is not exhibiting a measurable
pressure jump in `state["p"]` (= `p_hom`). Hypotheses:

1. **Pressure projection finds a near-trivial p_hom**: the IBM Brinkman
   suppresses `u_phys` inside the sphere; the projection step solves
   `∇·u_phys = 0` and finds a `p_hom` perturbation, but the choice of
   that perturbation is not unique and the solver picks one with
   minimal axial gradient — the ΔP_hom·A_pipe across the sphere
   integration planes ends up near zero.
2. **Sphere drag absorbed into u_hom kinetic field**: the perturbation
   energy is in the wake (u_hom) rather than the pressure field.
   Momentum-deficit reads (M_in − M_out + ΔP·A); for Stokes the
   M-deficit term is small (Re·F_Stokes), so this hypothesis predicts
   F_md ≈ small × F_Stokes — consistent with what we see.
3. **PISO not converged to sphere-drag steady state**: 800 PISO steps
   × dt = 50 simulation seconds, vs diffusion time R²/ν = 0.1 s, so
   500 diffusion times — should be plenty, but the wake equilibrium
   in the lifted frame may need more.

Hypothesis (1) or (2) is most likely. Resolution: the lifted PISO
step needs an explicit sphere-drag equilibration mechanism, OR the
extraction needs to use the surface integral of viscous stress on the
IBM shell (`surface_integral_force`) rather than the CV momentum
balance.

## Status

- **Verification A**: PASS at machine precision.
- **Verification C**: PASS (K_FVM > 0).
- **Magnitude convergence to K_Happel**: FAIL — out of scope for this
  fix sprint; needs either an alternative force extractor or a
  deeper fix to the PISO + lifting + IBM pressure coupling.
