# MIME-VER-123 — RobotArmNode Gravity-Compensated Static Hold

**Date**: 2026-04-30
**Node under test**: `mime.nodes.actuation.robot_arm.RobotArmNode`
**Algorithm ID**: `MIME-NODE-102`
**Benchmark type**: Analytical (Mode 2 independent)
**Test file**: `tests/verification/test_robot_arm.py::test_ver123_gravity_hold`
**Acceptance**: max joint drift $< 10^{-3}$ rad after 1000 steps at $\Delta t = 10^{-3}$ s

---

## Goal

Drive the arm with a gravity-compensation torque $\tau = g(q)$
re-evaluated at the current configuration each step, integrate for
1000 steps at $\Delta t = 10^{-3}$ s, and verify that the joint angles
do not drift more than $10^{-3}$ rad from the initial pose. This is a
closed-loop check on the integrator: if integration error or any sign
mistake leaked through, gravity compensation would no longer cancel
gravity and the arm would fall.

## Configuration

| Parameter | Value |
|---|---|
| URDF | `tests/control/fixtures/three_link_planar.urdf` |
| Initial $q$ | $(0.3, -0.4, 0.5)$ rad |
| Initial $\dot q$ | $0$ |
| External wrenches | $0$ |
| Joint friction | $0$ (overridden in test fixture) |
| Gravity | $(0, 0, -9.80665)$ m/s² |
| Timestep | $10^{-3}$ s |
| Steps | 1000 (1 s of simulated time) |
| JAX precision | x64 enabled at module load |
| Hot loop | `jax.jit`-compiled `step(q, qd) → (q', qd')` |

## Procedure

1. Set initial state $(q_0, \dot q_0 = 0)$.
2. JIT-compile a step closure that recomputes
   $\tau_{\mathrm{cmd}} = g(q)$ from the current $q$ and calls
   `arm.update`.
3. Run 1000 jitted steps.
4. Compute drift $\max_i |q_{1000,i} - q_{0,i}|$.
5. Assert drift $< 10^{-3}$ rad.

## Result

**PASS**. The arm holds within the tolerance — semi-implicit Euler is
symplectic, so even tiny gravity-comp residuals do not accumulate
secularly over the 1000-step horizon.

## Scope and Limitations

- One initial pose. A more thorough test would sweep multiple poses,
  but the integrator is linear in the residual so a single pose is
  representative.
- Tests *integrator drift*, not *steady-state error under
  disturbance*. Disturbance rejection is implicitly exercised by
  MIME-VER-124.

## Reproducibility

- Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest
  tests/verification/test_robot_arm.py::test_ver123_gravity_hold -x -q`.
