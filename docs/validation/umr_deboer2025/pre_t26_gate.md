# Pre-T2.6 Gate: Fin Resolution and Stability

## Fin resolution study (DONE)

All 4 tested resolutions (64^3 through 192^3) produce distinguishable fins
with 3 angular bands per z-slice (matching 3 fins per set at 120 degree spacing).

| Resolution | Circ. arc (lu) | Radial extent (lu) | Angular bands |
|---|---|---|---|
| 64^3 | 1.4 | 2.5 | 3 |
| 96^3 | 2.0 | 5.5 | 3 |
| 128^3 | 2.6 | 6.0 | 3 |
| 192^3 | 4.1 | 10.0 | 3 |

Note: these are corrected fin geometry values (MIME-ANO-003 fix applied —
fin_thickness = 0.15mm used as circumferential blade thickness per §VI.F p.15).

**Minimum viable resolution**: 128^3 (circ. arc = 2.6 lu — marginal but usable with Bouzidi IBB).
**Target resolution for T2.6**: 192^3 (circ. arc = 4.1 lu — well-resolved).

## Two-pass bounce-back architecture (CONFIRMED)

Single-pass BB with combined wall velocity for pipe + UMR causes compressibility
instability (Ma > 0.1 at pipe wall radius). The corrected architecture uses
two sequential BB passes with disjoint missing masks:

1. **Pass 1**: Pipe wall (stationary) — `apply_bounce_back(f_post, f_pre, pipe_missing, solid, wall_velocity=None)`
2. **Pass 2**: UMR (rotating) — `apply_bounce_back(f, f_pre, umr_missing, solid, wall_velocity=omega_x_r)`

Both passes use the same `f_pre` (from `lbm_step_split`). The second pass receives
the output of the first as its `f_post_stream` argument. The `solid_mask` parameter
is vestigial in both BB functions (unused in computation, driven by `missing_mask`).

Missing masks are disjoint: pipe wall boundary links and UMR boundary links never
share a node (minimum gap = 19 lu at worst-case confinement ratio 0.40).

For Bouzidi IBB: each pass gets its own q-values computed from its own SDF.
Pipe wall q-values can use analytical `compute_q_values_cylinder`. UMR q-values
use `compute_q_values_sdf` with `umr_sdf`.

## Union SDF (CONFIRMED — optional with two-pass)

With two-pass BB, each pass uses its own SDF for q-value computation. A union SDF
(`min(pipe_sdf, umr_sdf)`) is not required for the two-pass architecture but would
be needed for a single-pass variant.

No pipe wall SDF exists in the codebase. A trivial cylinder SDF should be added to
`helix_geometry.py` if single-pass or combined q-value computation is needed later:
`pipe_sdf(pts) = R_vessel - sqrt(dx^2 + dy^2)` (positive inside pipe = fluid,
negative outside = solid wall).

## Mach number guard (CONFIRMED)

Constraint: `omega * R_fin_lu * sqrt(3) < 0.1` (Ma < 0.1 at fin tips).

| Resolution | R_fin (lu) | Max safe omega | Period (steps) |
|---|---|---|---|
| 64^3 | 9.7 | 0.00596 | 1,054 |
| 128^3 | 19.3 | 0.00299 | 2,101 |
| 192^3 | 29.0 | 0.00199 | 3,156 |

Target Ma = 0.05 (half of limit for safety margin):

| Resolution | Safe omega (Ma=0.05) | Period (steps) |
|---|---|---|
| 64^3 | 0.00299 | 2,104 |
| 128^3 | 0.00149 | 4,209 |
| 192^3 | 0.00100 | 6,283 |

Guard implemented as an assertion at sweep initialisation (checked once, not per-step).

## 128^3 rotating stability check (DONE — PASS, 2026-03-23)

**Setup**: 128x128x128, confinement ratio 0.30, two-pass BB (pipe static + UMR rotating),
omega = 0.00149 rad/step (Ma = 0.05), tau = 0.8, simple BB, 1000 steps.

**Results**:
- NaN: False
- Inf: False
- u_max: 0.027 lu (< 0.05 threshold)
- Ma_max: 0.048 (< 0.1 threshold)
- Density conservation: 0.0037% (< 0.01% threshold)
- Torque sign: Correct (positive — body pumps momentum into fluid)
- Step time: **0.98 s/step** on RTX 2060 GPU

## Convergence rate (measured at 64^3, 2026-03-23)

Confinement ratio 0.30, two-pass BB, omega = 0.003, tau = 0.8, Ma = 0.05.
**Converged at step 4400 (~2.1 rotation periods)**, rel_change = 0.73%.
Convergence criterion: 2% relative change in mean drag torque between
consecutive rotation periods.

Convergence in rotation periods is resolution-independent (same physics).
At 192^3 with period = 6,283 steps: expected convergence at ~13,000 steps.

## GPU choice (DECIDED)

**H100 SXM** at $2.69/hr (RunPod). Rationale:
- 1.68x bandwidth advantage over A100 SXM (3.35 TB/s vs 2.0 TB/s)
- LBM confirmed bandwidth-bound at 98% GPU utilisation at production resolution
- Python dispatch overhead (~30ms) negligible at 192^3 step time (~1-5s)
- Expected sweep time: 8.4 hours (H100) vs 14 hours (A100) — 5.6 hours saved
- Cost: ~$22.60 (expected) vs ~$20.86 (A100) — $1.74 premium for 40% faster completion

## Resolution (DECIDED)

**192^3**. Fin circumferential arc = 4.1 lu (well-resolved).
128^3 (2.6 lu) is viable but marginal. The $50 budget accommodates 192^3 comfortably.

## Revised cost estimate

| Scenario | Steps | H100 step time | Time per ratio | 4 ratios | Cost |
|---|---|---|---|---|---|
| Optimistic (1.5 periods) | 9,400 | ~0.60s | 1.6 hr | 6.2 hr | $16.70 |
| **Expected (2 periods)** | **12,600** | **~0.60s** | **2.1 hr** | **8.4 hr** | **$22.60** |
| Conservative (3 periods) | 18,800 | ~0.60s | 3.1 hr | 12.6 hr | $33.89 |

**Budget**: $50.00. Expected cost: $22.60. Reserve after expected: $27.40.
Reserve covers: one full re-run ($22.60) OR Track C extended collection + held-out test point.

## Decision: PROCEED with T2.6

All pre-launch gate checks pass. Architecture confirmed. Ready for H100 launch.
