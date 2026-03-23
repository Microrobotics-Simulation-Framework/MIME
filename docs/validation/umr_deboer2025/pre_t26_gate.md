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

## Cloud rehearsal (DONE — PASS, 2026-03-23)

**Setup**: A100 SXM 80GB on RunPod (US), 192^3, confinement ratio 0.30,
two-pass BB, omega = 0.001 (Ma = 0.05), simple BB, 500 steps.

**Results**:
- All 6 gates PASS
- NaN: False, Inf: False
- u_max: 0.028 lu (< 0.05)
- Density conservation: 0.001% (< 0.01%)
- Torque sign: Correct
- **Step time: 0.058 s/step on A100 SXM 80GB**

**Issues fixed during rehearsal**:
1. GPU type: `A100-80GB` (PCIe) → `A100-80GB-SXM` (SXM)
2. cuDNN: Docker image CUDA 12.2 incompatible with host driver 570 (CUDA 12.8).
   Fixed by adding `pip3 install --upgrade 'jax[cuda12]'` as first setup step.
3. SkyPilot lifecycle: `stream_and_get` returns at job submission, not completion.
   Fixed with SSH polling in launch script.
4. Git hash: `.git/` not synced to cloud. Fixed by writing hash to file pre-sync.

## GPU choice (REVISED after rehearsal)

**A100 SXM** at $1.49/hr (RunPod). Revised rationale:
- Measured step time at 192^3: **0.058 s/step** — 17x faster than RTX 2060 extrapolation
- The H100 SXM advantage (1.68x bandwidth) gives ~0.035 s/step — only 0.023s faster
- At these step times, the H100 premium ($2.69 vs $1.49/hr) costs more than it saves
- **A100 is cheaper for this job**: $1.21 vs $1.31 on H100

## Resolution (DECIDED)

**192^3**. Fin circumferential arc = 4.1 lu (well-resolved).

## Revised cost estimate (from measured A100 SXM timing)

| Scenario | Steps | A100 SXM step time | Time per ratio | 4 ratios | Cost |
|---|---|---|---|---|---|
| Optimistic (1.5 periods) | 9,400 | 0.058s | 9 min | 36 min | $0.89 |
| **Expected (2 periods)** | **12,600** | **0.058s** | **12 min** | **49 min** | **$1.21** |
| Conservative (3 periods) | 18,800 | 0.058s | 18 min | 73 min | $1.81 |

**Budget**: $50.00. Expected cost: $1.21. Reserve: $48.79.
Massive headroom — enough for multiple re-runs, Track C collection, orientation repeats,
held-out test points, and future resolution escalation if needed.

## Decision: PROCEED with T2.6

All pre-launch gate checks pass. Architecture confirmed. Cloud rehearsal PASS.
Ready for A100 SXM production launch.
