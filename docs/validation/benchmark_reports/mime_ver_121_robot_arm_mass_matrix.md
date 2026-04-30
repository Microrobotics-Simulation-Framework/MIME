# MIME-VER-121 — RobotArmNode Mass-Matrix Symmetry & PD

**Date**: 2026-04-30
**Node under test**: `mime.nodes.actuation.robot_arm.RobotArmNode`
**Algorithm ID**: `MIME-NODE-102`
**Benchmark type**: Analytical (Mode 2 independent)
**Test file**: `tests/verification/test_robot_arm.py::test_ver121_mass_matrix_symmetric_pd`
**Acceptance**: $\|M - M^{\!\top}\|_F < 10^{-12}$ and $\lambda_{\min}(M) > 0$

---

## Goal

Confirm that the joint-space mass matrix $M(q)$ produced by the
CRBA-based `mime.control.kinematics.crba.mass_matrix` is symmetric
and positive-definite at five random configurations on the 3-link
planar fixture. These are necessary conditions for the forward-
dynamics solve $M\ddot q = \mathrm{rhs}$ to be well-posed.

## Configuration

| Parameter | Value |
|---|---|
| URDF | `tests/control/fixtures/three_link_planar.urdf` |
| Configurations | 5 samples from $\mathcal{U}[-\pi, \pi]^3$ (seed 20260430) |
| JAX precision | x64 enabled at module load |

## Procedure

For each random $q$:
1. Compute $M = \texttt{mass\_matrix}(\mathrm{tree}, q)$ as a NumPy array.
2. Verify $\|M - M^{\!\top}\|_F < 10^{-12}$.
3. Verify $\lambda_{\min}(M) > 0$ via `numpy.linalg.eigvalsh`.

## Result

**PASS**. Symmetric to within machine precision and PD across all 5
sampled configurations.

## Scope and Limitations

- Five random configs is a smoke-grade sample. Adding configs is cheap
  but tests are already comprehensive across the kinematics-package
  suite.
- Does not verify scale (numerical magnitude vs. analytical reference);
  scale is exercised indirectly by MIME-VER-122 / 123 / 124 where a
  wrong $M$ would produce wrong dynamics.

## Reproducibility

- Seed: NumPy `default_rng(20260430)`.
- Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest
  tests/verification/test_robot_arm.py::test_ver121_mass_matrix_symmetric_pd -x -q`.
