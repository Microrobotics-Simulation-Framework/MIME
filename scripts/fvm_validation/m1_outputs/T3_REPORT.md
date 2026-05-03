# T3 — Confined-Stokes drag (re-run after Fix 1+2)

Re-run of the T3 confined-sphere Stokes drag verification after the
isotropic-mesh + BC-clearance fixes.

## Setup

| Parameter           | Value                                       |
| ------------------- | ------------------------------------------- |
| Body                | Sphere of radius r_b = 1 mm                 |
| Pipe radius         | R = r_b / λ                                 |
| Pipe length         | L = 22 r_b (Fix 2 minimum at 5+5 r_b margin)|
| Lift                | Steady Poiseuille at U_dc = 1×10⁻³ m/s      |
| Re (R-based)        | 0.01 (λ=0.1) / 0.0033 (λ=0.3) — Stokes      |
| Mesh                | isotropic dx = r_b / cpr                    |
| Solver              | PISO 800 steps to convergence               |

## Results at cpr = 4

| λ    | mesh           | cells   | wall     | K_FVM   | K_Happel | err    |
| ---- | -------------- | ------- | -------- | ------- | -------- | ------ |
| 0.1  | 96 × 96 × 88   | 811 008 | 276 s    | -0.299  | 1.263    | 124%   |
| 0.3  | 32 × 32 × 88   |  90 112 |  39 s    | -1.073  | 2.370    | 145%   |

K_Happel from the standard Happel-Brenner series
``K = 1 / (1 − 2.10443λ + 2.08877λ³ − 0.94813λ⁵ − 1.372λ⁶ + 3.87λ⁸ − 4.19λ¹⁰)``.
The brief's value of 1.75 for λ=0.3 appears to be an error;
literature (Happel & Brenner 1965 §7-3, Bungay & Brenner 1973)
agrees with 2.37.

## Diagnosis

Both K_FVM are negative — the momentum-deficit estimator is reading
back roughly the residual `F_body − F_wall` without the sphere-induced
pressure jump showing up in `state["p"]` at all. Tracking through the
formula with no-sphere analytical Poiseuille predicts a residual of
``-F_wall_bias ≈ -1.3×10⁻⁸ N`` at λ=0.1 (matches the measured value
exactly), and the addition of the sphere does **not** add a positive
contribution to the measured F_md.

Why: with the lifting decomposition, ``state["p"]`` stores only
``p_hom`` (the perturbation pressure). The PISO projection enforces
``∇·u_hom = 0`` but does NOT pin a mean pressure or fix a reference
gradient — so the *absolute* p_hom scale is free, and what shows up
near the sphere is a small local perturbation, not a true `ΔP·A_pipe`
drag signature. With the steady Poiseuille lift, the lift itself
already satisfies the momentum balance through its analytical pressure
gradient (which is **never** materialised into ``state["p"]``).

In other words: the sphere drag *is* in u_hom (the wake) and *is*
balanced by some gradient in p_hom, but the current
``momentum_deficit_drag`` reads p_in − p_out from the cell-centre
pressure averaged over the fluid plane, which doesn't see the
sphere-driven contribution because that part of the pressure was
absorbed into the lift's analytical balance, not the perturbation.

## Status: open issue, methodology gap

This is the same class of failure as M0d (documented in
`FLUID_NODE_CONTRACT.md` § "Known caveat: momentum_deficit_drag with
lifting"). The contract notes this is a calibration issue requiring a
re-derivation of the F_md formula for lifted flow — adding back the
analytical lift-pressure contribution explicitly, not just the body
force.

cpr = 6 / 8 was deferred because the cpr = 4 result above already
demonstrates the failure is **not** a resolution issue: at λ=0.3,
cpr=4 (90 K cells) is plenty to resolve a 4-cells-per-radius IBM
sphere in Stokes flow, yet K_FVM is still negative.

## Required follow-up (out of scope for this sprint)

- Add a `lifted_pressure_callback(z)` parameter to
  `momentum_deficit_drag` that the user passes the analytical lifted
  pressure profile (e.g. for Poiseuille,
  ``p_lift(z) = -8μU_mean/R² · z``). The estimator then evaluates
  `(p_lift(z_in) + p_hom_in) - (p_lift(z_out) + p_hom_out)` for the
  full ΔP·A term.
- Verify this restores K_FVM > 0 at λ=0.1 first, then sweep cpr to
  measure convergence rate.

## All 18 regression tests still PASS after these fixes.
