---
bibliography: ../../bibliography.bib
---

# RobotArmNode

**Module**: `mime.nodes.actuation.robot_arm`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-102`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

URDF-driven articulated rigid-body manipulator. Kinematic and inertial
structure is parsed once at construction from a URDF file; joint-space
dynamics are integrated each step with semi-implicit Euler, using
mass matrix from CRBA, nonlinear bias from RNEA, and per-link external
wrenches mapped via Jacobian transpose. v1 is fully rigid (no
flexibility, no contact, no cogging).

## Governing Equations

$$
M(q)\,\ddot q + c(q,\dot q) + g(q) \;=\; \tau_{\mathrm{eff}} + \sum_i J_i^{\!\top}\,F^{\mathrm{ext}}_i
$$

with
$$
\tau_{\mathrm{eff}} = \tau_{\mathrm{cmd}} - f \cdot \dot q,
\qquad
J_i = J_{\mathrm{geom}}(q, i) \in \mathbb{R}^{6\times N}.
$$

Integration:

$$
\dot q_{n+1} = \dot q_n + \ddot q\,\Delta t,
\qquad
q_{n+1} = q_n + \dot q_{n+1}\,\Delta t,
\qquad
q_{n+1} \leftarrow \mathrm{clip}(q_{n+1};\,q_{\mathrm{lo}},q_{\mathrm{hi}}).
$$

## Discretization

Semi-implicit (symplectic) Euler with explicit forces. The mass matrix
$M(q)$ is built by a vectorised CRBA pass using ancestor masking; the
nonlinear bias $c(q,\dot q) + g(q)$ is computed by a single RNEA pass.
Each per-link 6-DOF external wrench is multiplied by the transpose of
that link's geometric Jacobian (linear-then-angular ordering). Joint
limits are enforced by hard clipping after the position update.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---|---|---|
| $M(q)$ — joint-space mass matrix | `mime.control.kinematics.crba.mass_matrix` | CRBA, ancestor mask |
| $c(q,\dot q)+g(q)$ — nonlinear bias | `mime.control.kinematics.rnea.nonlinear_bias` | RNEA single pass |
| $J_i$ — geometric Jacobian for link $i$ | `mime.control.kinematics.fk.frame_jacobian` | linear-angular ordering |
| $\sum_i J_i^{\!\top}F^{\mathrm{ext}}_i$ | `mime.nodes.actuation.robot_arm._wrench_to_joint_torques` | static-N unrolled loop |
| $\ddot q = M^{-1}(\tau_{\mathrm{eff}} + \tau_{\mathrm{ext}} - \mathrm{bias})$ | `mime.nodes.actuation.robot_arm.RobotArmNode.update` | `jnp.linalg.solve` |
| Semi-implicit Euler $q,\dot q$ update | `mime.nodes.actuation.robot_arm.RobotArmNode.update` | clip to limits |
| Forward kinematics — link / EE poses | `mime.control.kinematics.fk.link_world_poses` | URDF link-COM frames |
| Pose composition base ⊕ link ⊕ tool | `mime.nodes.actuation.robot_arm.RobotArmNode.compute_boundary_fluxes` | scalar-first quat |

## Assumptions and Simplifications

1. Rigid joints — no flexibility, backlash, or compliant stops.
2. Linear viscous joint friction $\tau_f = f\,\dot q$. Coulomb friction
   not modelled.
3. Fixed base — `base_pose_world` is held constant by the caller.
4. Gravity is treated as a uniform inertial field in world coordinates.
5. External wrenches are expressed in world coordinates and applied at
   each link's inertial origin (the URDF `<inertial><origin>`).
6. URDF fixed joints are merged at parse time — caller cannot re-introduce
   a fixed joint at runtime.

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|---|---|---|
| `num_joints` | 1 – 10 | Tested on 3-link planar fixture; algorithm scales further |
| Timestep | $10^{-4}$ – $10^{-2}$ s | Stable for revolute manipulators with reasonable inertias |

## Known Limitations and Failure Modes

1. No contact / collision detection: links can pass through each other
   or through the world without any reaction force.
2. Joint limit clipping is hard — $\dot q$ is *not* zeroed at the limit,
   so the arm can integrate against the wall and produce large reaction
   spikes if energy is supplied.
3. Single solve per step — no fixed-point iteration with downstream
   nodes beyond what `GraphManager`'s coupling-group machinery provides.
4. Cogging torque, rotor imbalance, and motor current draw belong to
   `MotorNode` (the magnet-rotor stage at the EE), *not* this node. The
   per-axis arm motors are abstracted into joint-space torques /
   friction.

## Stability Conditions

Empirical: dt < 1e-2 s for typical revolute manipulators; tighter at
high stiffness. The semi-implicit scheme is symplectic, so energy drift
on conservative systems is bounded over long horizons.

## State Variables

| Field | Shape | Units | Description |
|---|---|---|---|
| `joint_angles` | (N,) | rad | $q$ |
| `joint_velocities` | (N,) | rad/s | $\dot q$ |

## Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `urdf_path` | str | — (required) | — | URDF file path |
| `end_effector_link_name` | str | — (required) | — | URDF link name to expose as EE |
| `end_effector_offset_in_link` | tuple[7] | identity | m, quat | tool-tip offset in EE-link frame |
| `base_pose_world` | tuple[7] | identity | m, quat | world pose of root link |
| `joint_friction_n_m_s` | tuple[N] or None | URDF damping | N·m·s/rad | per-joint viscous friction |
| `gravity_world` | tuple[3] | (0,0,-9.80665) | m/s² | world-frame gravity |
| `joint_limit_override` | tuple[(N,2)] or None | URDF limits | rad | (lower, upper) per joint |

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|---|---|---|---|---|
| `commanded_joint_torques` | (N,) | zeros | additive | $\tau_{\mathrm{cmd}}$ |
| `external_wrenches_per_link` | (N, 6) | zeros | additive | per-link world-frame wrench `[F_lin; F_ang]` |

## Boundary Fluxes (outputs)

| Field | Shape | Units | Description |
|---|---|---|---|
| `end_effector_pose_world` | (7,) | m, quat | tool-tip pose, scalar-first quat |
| `link_poses_world` | (N, 7) | m, quat | URDF link-COM poses |
| `joint_angles` | (N,) | rad | $q$ |
| `joint_velocities` | (N,) | rad/s | $\dot q$ |
| `joint_torques_actual` | (N,) | N·m | $\tau_{\mathrm{cmd}} - f\dot q + J^\top F_{\mathrm{ext}}$ |

## MIME-Specific Sections

### Anatomical Operating Context

Not applicable — RobotArmNode is an external apparatus operating in
the lab frame, not in any anatomical compartment. Its base pose is
positioned outside the patient.

### Clinical Relevance

The arm carries the magnet-rotor + permanent-magnet stage at its
end-effector. Its kinematic precision and stiffness directly determine
the achievable accuracy of microrobot actuation: lateral drift of the
EE relative to the microrobot's vessel translates into the
misalignment-induced wobble / step-out reduction effects analysed in
the actuation-decomposition plan. This node is *not* in patient
contact and makes no clinical claims.

### Mode 2 Independent Verification

All verification is independent (no MADDENING upstream wraps a
multibody arm). Evidence: MIME-VER-120 / 121 / 122 / 123 / 124.

## References

- [@Featherstone2008] Featherstone, R. (2008). *Rigid Body Dynamics
  Algorithms*. Springer. — CRBA, RNEA, spatial-vector formulations.
- [@Sciavicco2000] Sciavicco, L. & Siciliano, B. (2000). *Modelling and
  Control of Robot Manipulators*. Springer. — Jacobian-transpose
  wrench mapping; PD with gravity compensation.
- [@Annin2024AR4] Annin, C. (2024). *AR4 Open-Source 6-DOF Robot Arm*. — URDF and STL meshes for the bundled default-arm experiment under `MICROROBOTICA/experiments/ar4_helical_drive/`.

## Verification Evidence

- `tests/verification/test_robot_arm.py`
- `docs/validation/benchmark_reports/mime_ver_120_robot_arm_fk.md`
- `docs/validation/benchmark_reports/mime_ver_121_robot_arm_mass_matrix.md`
- `docs/validation/benchmark_reports/mime_ver_122_robot_arm_free_fall.md`
- `docs/validation/benchmark_reports/mime_ver_123_robot_arm_gravity_hold.md`
- `docs/validation/benchmark_reports/mime_ver_124_robot_arm_pd_tracking.md`

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-04-30 | Initial implementation. URDF parsing via `mime.control.kinematics.urdf.parse_urdf`. CRBA + RNEA pure-JAX path. Five Mode-2 verification benchmarks. |
