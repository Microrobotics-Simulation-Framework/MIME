---
bibliography: ../../bibliography.bib
---

# Motor Node

**Module**: `mime.nodes.actuation.motor`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-100`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

Single-axis rotary DC-brushed motor that spins a body-fixed rotor (and any tool rigidly attached to it) about an axis defined in the parent frame. Three command modes — direct torque, terminal voltage, and angular-velocity setpoint — are dispatched from the boundary inputs in a JAX-traceable manner with precedence torque > voltage > velocity. Mechanical and electrical states are integrated with semi-implicit Euler. The composed tool-tip pose in world coordinates is published as a boundary flux for downstream nodes (typically a `PermanentMagnetNode` riding on the rotor).

## Governing Equations

Mechanical:
$$
J\,\dot{\omega} = \tau_\text{cmd} + \tau_\text{load} + \tau_\text{motor} - b\,\omega,
\qquad \dot{\theta} = \omega
$$

Electrical (voltage-mode only):
$$
L\,\frac{di}{dt} = V - R\,i - k_e\,\omega, \qquad \tau_\text{motor} = k_t\,i.
$$

Velocity-mode internal PI:
$$
\tau_\text{cmd} = K_p\,(\omega_\text{des} - \omega) + K_i\!\!\int_0^t(\omega_\text{des} - \omega)\,d\tau.
$$

Pose composition (world frame, $q = [w,x,y,z]$ Hamilton convention):
$$
P_\text{rotor}^\text{world} \;=\; P_\text{parent}^\text{world}\;\oplus\;R(\hat{e}_\text{axis},\theta)\;\oplus\;P_\text{tool}^\text{rotor}.
$$

In SI units $k_t \equiv k_e$ numerically; the constructor enforces the default $k_e = k_t$.

## Discretization

Semi-implicit (symplectic) Euler for the rotor:

$$
\omega_{n+1} = \frac{J\,\omega_n + \Delta t \,\tau_\text{drive}}{J + \Delta t\,b},
\qquad \theta_{n+1} = \theta_n + \omega_{n+1}\,\Delta t,
$$

where $\tau_\text{drive} = \tau_\text{cmd,eff} + \tau_\text{motor} + \tau_\text{load}$. The implicit treatment of $b\,\omega$ is unconditionally stable for any $b\geq 0,\,\Delta t>0$.

Semi-implicit Euler for the armature current (voltage-mode only):

$$
i_{n+1} = \frac{i_n + \frac{\Delta t}{L}\,(V - k_e\,\omega_n)}{1 + \Delta t\,R/L}.
$$

Velocity-mode PI integrator: backward Euler on the error,
$\,I_{n+1} = I_n + (\omega_\text{des} - \omega_n)\,\Delta t$, frozen outside velocity-mode to suppress windup.

Pose composition is closed-form (quaternion product) — no integration error.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $J\,\dot{\omega} = \tau_\text{net} - b\,\omega$ | `mime.nodes.actuation.motor.MotorNode.update` | Implicit on $b\,\omega$ |
| $L\,\dot{i} = V - R\,i - k_e\,\omega$ | `mime.nodes.actuation.motor.MotorNode.update` | Implicit on $R\,i$ |
| $\tau_\text{motor} = k_t\,i$ | `mime.nodes.actuation.motor.MotorNode.update` | Voltage-mode only |
| $\tau_{PI} = K_p e + K_i\!\int e\,dt$ | `mime.nodes.actuation.motor.MotorNode.update` | Backward Euler on $\int e$ |
| $\theta_{n+1} = \theta_n + \omega_{n+1} \Delta t$ | `mime.nodes.actuation.motor.MotorNode.update` | Semi-implicit Euler |
| Pose composition $P_p \oplus R(\hat{e},\theta) \oplus P_t$ | `mime.nodes.actuation.motor.compose_pose` | Quaternion product |
| Axis-angle to quaternion | `mime.nodes.actuation.motor._quat_from_axis_angle` | Used by `compose_pose` |
| Mode dispatch (torque > voltage > velocity) | `mime.nodes.actuation.motor.MotorNode.update` | `jnp.where` only — fully traceable |

## Assumptions and Simplifications

1. Single rotational degree of freedom (one-axis rotor).
2. Linear viscous bearing friction $\tau_f = b\,\omega$; Coulomb friction is neglected.
3. DC-brushed armature: $V = R\,i + L\,\dot{i} + k_e\,\omega$.
4. $k_t = k_e$ numerically in SI units; default constructor enforces this.
5. Rotor inertia is constant — no rotor-imbalance modulation.
6. Tool offset is rigid in the rotor frame.

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| inertia_kg_m2 | $10^{-7}$ – $10^{-2}$ | Lab-scale to small servo motors |
| kt_n_m_per_a | $10^{-3}$ – $1.0$ | Small DC brushed to NEMA-23 servos |
| commanded_velocity | 0 – 500 rad/s | Default PI tested up to ≈50 rad/s |

## Known Limitations and Failure Modes

1. **No cogging torque** — predicted low-speed smoothness is optimistic versus real brushed motors.
2. **No thermal model** — winding heating not tracked; sustained overload won't derate in simulation.
3. **Voltage-mode neglects PWM ripple** — current ripple from the drive is absent.
4. **Velocity-mode PI gains require tuning per motor** — defaults (`Kp=1e-3`, `Ki=1e-2`) are chosen for a small lab rotor (`J≈1e-4 kg·m²`, `b≈0.01`) and must be retuned for stiffer or heavier rotors.
5. **No back-iron saturation** — torque is linear in current at all currents; large-current commands will produce non-physical torques.

## Stability Conditions

Semi-implicit Euler with implicit damping is unconditionally stable for the mechanical subsystem. The armature integration is stable for $\Delta t \,R/L \geq 0$ (always satisfied). Velocity-mode PI gains must be chosen so the closed-loop pole pair lies inside the unit circle of the discrete-time map; the supplied defaults satisfy this for the validated regime above but should be retuned otherwise.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| angle | () | rad | Cumulative rotor angle (not wrapped) |
| angular_velocity | () | rad/s | Rotor angular velocity |
| current | () | A | Armature current (zero outside voltage-mode) |
| velocity_integral_error | () | rad | Internal PI integrator (not exposed) |

`state_fields()` and `observable_fields()` expose the first three only.

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| inertia_kg_m2 | float | — | kg·m² | Rotor inertia about spin axis |
| kt_n_m_per_a | float | — | N·m/A | Torque constant |
| ke_v_s_per_rad | float | = kt | V·s/rad | Back-EMF constant (SI: $k_e=k_t$) |
| r_ohm | float | — | Ω | Armature resistance |
| l_henry | float | — | H | Armature inductance |
| damping_n_m_s | float | — | N·m·s/rad | Bearing/airgap viscous damping |
| axis_in_parent_frame | tuple[3] | (0,0,1) | — | Rotor axis in parent frame |
| tool_offset_in_rotor_frame | tuple[7] | identity | — | Tool-tip pose `[x,y,z,qw,qx,qy,qz]` |
| velocity_kp | float | 1e-3 | N·m·s/rad | PI proportional gain |
| velocity_ki | float | 1e-2 | N·m/rad | PI integral gain |

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|-------|-------|---------|---------------|-------------|
| commanded_torque | () | 0.0 | additive | Direct rotor torque [N·m] (mode 1) |
| commanded_voltage | () | 0.0 | replacive | Armature terminal voltage [V] (mode 2) |
| commanded_velocity | () | 0.0 | replacive | Desired ω [rad/s] (mode 3) |
| parent_pose_world | (7,) | identity | replacive | Parent frame world pose `[x,y,z,qw,qx,qy,qz]` |
| load_torque | () | 0.0 | additive | External reaction (e.g. magnetic drag) [N·m] |

Mode precedence: **torque > voltage > velocity** (first non-zero wins; ties default to torque).

## Boundary Fluxes (outputs)

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| rotor_angle | () | rad | Cumulative rotor angle |
| rotor_angular_velocity | () | rad/s | Rotor angular velocity |
| rotor_pose_world | (7,) | m, — | Tool-tip world pose `[x,y,z,qw,qx,qy,qz]` |

## MIME-Specific Sections

### Anatomical Operating Context

External-apparatus node — operates in the laboratory environment, not in an anatomical compartment. No `AnatomicalRegimeMeta` is attached.

### Clinical Relevance

The motor sits between the actuation arm (which positions the rotor) and the rotor-mounted permanent magnet (which couples to a microrobot through the magnetic field). The motor model is what converts a controller's torque/voltage/velocity command into the physical magnet rotation that produces the rotating magnetic field driving an in-vivo helical microrobot.

### Mode 2 Independent Verification

- **MIME-VER-100**: torque-mode step response vs. analytical first-order spin-up.
- Voltage-mode steady-state cross-check: $\omega_\infty = V k_t / (k_t k_e + R b)$.
- Velocity-mode PI tracking: <1 % steady-state error with default gains on the reference motor.
- Pose composition: 90° z-rotation matches closed-form quaternion.
- JAX traceability: `jit`, `grad`, and `vmap` of `update`.
- GraphManager wiring: single-step run with external `commanded_torque` and `parent_pose_world`.

## References

- [@Krause2013] Krause, P. C., Wasynczuk, O., Sudhoff, S. D., & Pekarek, S. D. (2013). *Analysis of Electric Machinery and Drive Systems*. — Reference for DC brushed motor electromechanical model.

## Verification Evidence

- Test files: `tests/verification/test_motor.py`
- Benchmark report: `docs/validation/benchmark_reports/mime_ver_100_motor_step_response.md`

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-04-30 | Initial implementation — torque/voltage/velocity modes, semi-implicit Euler, pose composition |
