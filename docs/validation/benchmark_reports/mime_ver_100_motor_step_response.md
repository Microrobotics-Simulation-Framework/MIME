# MIME-VER-100 — MotorNode Step Response (Torque Mode, No Load)

**Date**: 2026-04-30
**Node under test**: `mime.nodes.actuation.motor.MotorNode`
**Algorithm ID**: `MIME-NODE-100`
**Benchmark type**: Analytical (Mode 2 independent)
**Test file**: `tests/verification/test_motor.py::test_ver100_torque_step_response`
**Acceptance**: relative RMS error of $\omega(t)$ over a 1 s trajectory < 5 %

---

## Goal

Verify that `MotorNode` in **torque-mode** with no electrical model and
no external load reproduces the analytical first-order velocity step
response of a damped rotor:

$$
J \dot{\omega} = \tau - b \omega
\quad \Longrightarrow \quad
\omega(t) = \frac{\tau}{b} \left( 1 - e^{-bt/J} \right).
$$

This is the simplest possible regression on the mechanical integration
path. It exercises the semi-implicit-Euler step, the boundary-input
plumbing for `commanded_torque`, and the rotor-pose composition into
`rotor_pose_world` (the latter cross-checked by a separate test, not
this one).

## Configuration

| Parameter | Value |
|-----------|-------|
| `inertia_kg_m2`     ($J$) | $1 \times 10^{-4}$ kg·m² |
| `damping_n_m_s`     ($b$) | $5 \times 10^{-4}$ N·m·s |
| `kt_n_m_per_a`      ($k_t$) | not exercised in torque-mode |
| `r_ohm`, `l_henry`        | not exercised in torque-mode |
| `commanded_torque`  ($\tau$) | $0.01$ N·m (constant step) |
| `commanded_voltage`        | $0.0$ |
| `commanded_velocity`       | $0.0$ |
| `parent_pose_world`        | identity |
| `axis_in_parent_frame`     | $(0,0,1)$ |
| Timestep $\Delta t$        | $1 \times 10^{-4}$ s |
| Duration                  | $1.0$ s ($10^4$ steps) |
| JAX precision             | x64 enabled at module load |

## Analytical reference (numpy, double precision)

$$
\omega_{\text{ref}}(t_n) = \frac{\tau}{b}\left( 1 - e^{-b t_n / J} \right),
\qquad t_n = n \cdot \Delta t.
$$

Steady-state value: $\omega_\infty = \tau/b = 20 \text{ rad/s}$.
Time constant: $J/b = 0.2$ s.

## Procedure

1. Construct the node and obtain its initial state.
2. For $n \in [0, 10^4)$: call `node.update(state, {"commanded_torque":
   0.01}, dt)`, append `state["angular_velocity"]` to a trajectory
   array.
3. Compute `omega_ref` on the same time grid using the closed form
   above.
4. Compute relative RMS error
   $\sqrt{\overline{(\omega - \omega_{\text{ref}})^2}} \,/\, \omega_\infty$.

## Result

Status: **PASS**.

Relative RMS error is well under the 5 % acceptance, dominated by
$O(\Delta t)$ truncation of the semi-implicit-Euler integrator
(theoretical bound at this step size: ~0.05 %). Exact value is
recorded by the test; see latest CI run.

## Scope and Limitations

This benchmark validates **only**:

- The torque-mode integration path of `MotorNode`.
- The first-order analytical agreement of a rotor-with-damping ODE.

It does **not** validate:

- The voltage-mode RL armature integration (covered by a separate
  test in the same file, not a verification benchmark).
- The velocity-mode PI loop (separate test).
- Cogging torque or rotor imbalance (out of scope for v1 — both
  `MotorMeta.has_cogging` and `MotorMeta.has_imbalance_vibration`
  default to `False`).
- The interaction with `RobotArmNode` end-effector reaction
  (one-way coupling in v1; tracked as a known anomaly).

## Reproducibility

- Hardware: any platform with JAX ≥ 0.4 (CPU is sufficient).
- Software: MIME `0.1.0`; MADDENING pinned in `pyproject.toml`.
- RNG seed: not used (deterministic input).
- Run with `JAX_PLATFORMS=cpu .venv/bin/python -m pytest
  tests/verification/test_motor.py::test_ver100_torque_step_response
  -x -q` if local GPU is contended.
