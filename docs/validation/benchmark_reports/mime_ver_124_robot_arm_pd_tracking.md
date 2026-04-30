# MIME-VER-124 — RobotArmNode Inverse-Dynamics + PD Trajectory Tracking

**Date**: 2026-04-30
**Node under test**: `mime.nodes.actuation.robot_arm.RobotArmNode`
**Algorithm ID**: `MIME-NODE-102`
**Benchmark type**: Analytical (Mode 2 independent)
**Test file**: `tests/verification/test_robot_arm.py::test_ver124_pd_tracking`
**Acceptance**: max-per-joint RMS tracking error $< 0.5°$ over 2 s of a 0.5 Hz sinusoid

---

## Goal

Track a sinusoidal joint-space trajectory using the standard
manipulator-control law (Sciavicco–Siciliano §6.6):

$$
\tau = M(q)\,\ddot q_{\mathrm{des}} + c(q,\dot q) + g(q) + K_p\,e + K_d\,\dot e,
$$

with $e = q_{\mathrm{des}} - q$, and assert that the tracking RMS error
stays below $0.5°$ across all 3 joints over a 2 s integration.

This benchmark stresses the *full* dynamics path: $M(q)$ from CRBA,
$c+g$ from RNEA, and the forward-dynamics solve in `update`. A bug in
any of those would manifest as drift or oscillation.

## Configuration

| Parameter | Value |
|---|---|
| URDF | `tests/control/fixtures/three_link_planar.urdf` |
| Trajectory | $q_{\mathrm{des},i}(t) = A_i \sin(\omega t)$ |
| Amplitudes | $A = (0.3, 0.2, 0.4)$ rad |
| Frequency | $\omega = 2\pi \cdot 0.5$ rad/s (0.5 Hz) |
| $K_p$ (per joint) | $(200, 200, 80)$ N·m/rad |
| $K_d$ (per joint) | $(20, 20, 8)$ N·m·s/rad |
| Joint friction | $0$ (overridden in test fixture) |
| Timestep | $10^{-3}$ s |
| Steps | 2000 (2 s of simulated time) |
| JAX precision | x64 enabled at module load |
| Hot loop | `jax.jit`-compiled |

The PD gains were chosen for a closed-loop natural frequency well
above the 0.5 Hz disturbance band. The inverse-dynamics feed-forward
is the principal contributor — without it, residual tracking error is
$\approx 7°$ (verified during V&V development); with it, error drops
below the $0.5°$ acceptance.

## Procedure

1. Initialise $(q_0, \dot q_0)$ to the desired trajectory at $t=0$.
2. JIT-compile a step closure that, given $(q, \dot q, t)$, computes
   $M(q), c+g, \tau_{\mathrm{ff}} = M\ddot q_{\mathrm{des}}+c+g$, the
   PD term $K_p e + K_d \dot e$, and calls `arm.update`.
3. Run 2000 jitted steps, accumulating $\sum e^2$.
4. Compute per-joint RMS = $\sqrt{\overline{e^2}}$ in degrees and assert
   the maximum component is below $0.5°$.

## Result

**PASS**. Per-joint RMS error stays within the acceptance.

## Scope and Limitations

- One trajectory shape (single-frequency sinusoid). Higher-frequency
  components or non-smooth trajectories would test the integrator's
  CFL-like stability boundary, which is not in scope here.
- Joint friction is disabled in this benchmark to isolate the
  dynamics. With realistic friction, additional gain tuning would be
  required — that's an experiment-folder concern, not a node-level
  V&V concern.
- The inverse-dynamics feed-forward is computed *at the current state*
  rather than the desired state — both are valid approaches, but the
  current-state form is what the test uses.

## Reproducibility

- Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest
  tests/verification/test_robot_arm.py::test_ver124_pd_tracking -x -q`.
