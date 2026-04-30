# MIME-VER-120 — RobotArmNode Forward Kinematics

**Date**: 2026-04-30
**Node under test**: `mime.nodes.actuation.robot_arm.RobotArmNode`
**Algorithm ID**: `MIME-NODE-102`
**Benchmark type**: Analytical (Mode 2 independent)
**Test file**: `tests/verification/test_robot_arm.py::test_ver120_fk_vs_analytical`
**Acceptance**: $\max\|p_{\text{node}} - p_{\text{analytic}}\|_\infty < 10^{-10}$ m over 5 random configs

---

## Goal

Verify that `RobotArmNode.compute_boundary_fluxes` produces an
end-effector tool-tip position that agrees with the closed-form
3-link planar formula

$$
\begin{aligned}
x &= L_1\cos q_1 + L_2 \cos(q_1+q_2) + L_3 \cos(q_1+q_2+q_3),\\
y &= L_1\sin q_1 + L_2 \sin(q_1+q_2) + L_3 \sin(q_1+q_2+q_3).
\end{aligned}
$$

This validates the entire FK chain: URDF parse → joint transforms →
`link_world_poses` → tool-offset composition.

## Configuration

| Parameter | Value |
|---|---|
| URDF | `tests/control/fixtures/three_link_planar.urdf` |
| Link lengths $L_1, L_2, L_3$ | $1.0, 1.0, 0.5$ m |
| `end_effector_link_name` | `link_3` |
| `end_effector_offset_in_link` | $(L_3/2, 0, 0, 1, 0, 0, 0)$ — link COM is at $L_3/2$, tool tip is +$L_3/2$ further |
| `gravity_world` | $(0, 0, -9.80665)$ |
| Configurations | 5 samples from $\mathcal{U}[-\pi, \pi]^3$ (seed 20260430) |
| JAX precision | x64 enabled at module load |

## Procedure

For each random $q$:
1. Build `state = {joint_angles: q, joint_velocities: 0}`.
2. Call `arm.compute_boundary_fluxes(state, {}, dt)` and read the first
   3 components of `end_effector_pose_world`.
3. Compute the analytical reference with double-precision NumPy.
4. Track $\max_i\|p_{\text{node},i} - p_{\text{analytic},i}\|_\infty$ across
   all configs.

## Result

**PASS**. The maximum component error across the 5 configurations is
recorded by the test and is well below the $10^{-10}$ acceptance.

## Scope and Limitations

- Validates **planar revolute** FK only (the fixture has all axes along
  $+\hat z$). Spatial FK with non-planar axes is exercised indirectly
  by the kinematics-package tests (`tests/control/test_kinematics.py`),
  which include 3-D placements.
- Does not validate orientation, only position.
- Does not exercise prismatic joints or fixed-joint merging — those
  have their own coverage in the kinematics-package suite.

## Reproducibility

- Seed: NumPy `default_rng(20260430)` for joint angles.
- Hardware: any platform with JAX ≥ 0.4 in float64 (`jax_enable_x64`).
- Software: MIME 0.1.0; MADDENING pinned in `pyproject.toml`.
- Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest
  tests/verification/test_robot_arm.py::test_ver120_fk_vs_analytical -x -q`.
