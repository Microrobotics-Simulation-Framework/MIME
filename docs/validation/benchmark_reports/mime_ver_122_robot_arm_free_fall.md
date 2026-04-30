# MIME-VER-122 — RobotArmNode Free-Fall Round-Trip

**Date**: 2026-04-30
**Node under test**: `mime.nodes.actuation.robot_arm.RobotArmNode`
**Algorithm ID**: `MIME-NODE-102`
**Benchmark type**: Analytical (Mode 2 independent)
**Test file**: `tests/verification/test_robot_arm.py::test_ver122_free_fall_round_trip`
**Acceptance**: $\|\ddot q\|_\infty < 10^{-10}$ when $\tau = g(q)$, $\dot q = 0$, $F_{\mathrm{ext}} = 0$

---

## Goal

Verify the consistency between the RNEA-derived gravity vector and
the forward-dynamics solve. With zero velocity and zero external
wrench, applying a torque equal to the gravity vector $g(q)$ must
produce zero joint acceleration:

$$
M(q)\,\ddot q = \tau - c(q,\dot q) - g(q) = g(q) - 0 - g(q) = 0
\quad \Longrightarrow \quad \ddot q = 0.
$$

Any deviation reveals a numerical inconsistency between
`mime.control.kinematics.rnea.gravity_vector`,
`mime.control.kinematics.rnea.nonlinear_bias`, and
`mime.control.kinematics.crba.mass_matrix` (e.g., a sign error,
inertia-tensor misorientation, or RNEA / CRBA spatial-algebra
mismatch).

## Configuration

| Parameter | Value |
|---|---|
| URDF | `tests/control/fixtures/three_link_planar.urdf` |
| Configurations | 5 samples from $\mathcal{U}[-\pi, \pi]^3$ (seed 20260430) |
| Velocity | $\dot q = 0$ (so bias = $g(q)$ exactly) |
| Gravity | $(0, 0, -9.80665)$ m/s² |
| JAX precision | x64 enabled at module load |

## Procedure

For each random $q$:
1. Compute $g(q)$ via `gravity_vector(tree, q, g_world)`.
2. Compute $c(q, 0) + g(q)$ via `nonlinear_bias(tree, q, 0, g_world)`
   and assert it equals $g(q)$ to machine precision.
3. Compute $M(q)$ via `mass_matrix(tree, q)`.
4. Solve $\ddot q = M^{-1}(g(q) - \mathrm{bias})$.
5. Assert $\|\ddot q\|_\infty < 10^{-10}$.

## Result

**PASS**. All 5 configurations satisfy the round-trip identity to
better than $10^{-10}$ rad/s².

## Scope and Limitations

- Validates *internal* consistency between RNEA and CRBA, not
  agreement with an external reference. External agreement is
  exercised by the kinematics-package tests
  (`tests/control/test_kinematics.py`) which include an RNEA
  self-consistency benchmark on the same fixture.
- Coriolis terms are not exercised here (because $\dot q = 0$). The
  PD-tracking benchmark (MIME-VER-124) implicitly stresses Coriolis
  via the inverse-dynamics feed-forward.

## Reproducibility

- Seed: NumPy `default_rng(20260430)`.
- Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest
  tests/verification/test_robot_arm.py::test_ver122_free_fall_round_trip -x -q`.
