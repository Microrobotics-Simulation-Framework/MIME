---
bibliography: ../../bibliography.bib
---

# Rigid Body Node

**Module**: `mime.nodes.robot.rigid_body`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-003`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

6-DOF rigid body dynamics in the overdamped Stokes regime. At Re << 1, inertia is negligible: velocity is instantaneously proportional to applied force. Position and quaternion orientation are integrated via explicit Euler.

## Governing Equations

Overdamped force balance (low Re):
$$
\sum \mathbf{F} \approx \mathbf{0} \implies \mathbf{V} = \mathbf{R}_T^{-1} \mathbf{F}_{\text{ext}}
$$
$$
\sum \mathbf{T} \approx \mathbf{0} \implies \boldsymbol{\omega} = \mathbf{R}_R^{-1} \mathbf{T}_{\text{ext}}
$$

Prolate ellipsoid resistance (Oberbeck-Stechert):
$$
f_x = 6a\pi\eta C_1 V_x, \quad f_y = 6a\pi\eta C_2 V_y, \quad M = 8ab^2\pi\eta C_3 \omega
$$

Coefficients as functions of eccentricity $e = \sqrt{1 - b^2/a^2}$:
$$
C_1 = \frac{8e^3/3}{\left[-2e + (1+e^2)\ln\left(\frac{1+e}{1-e}\right)\right]}
$$

Quaternion integration:
$$
\mathbf{q}_{n+1} = \text{normalize}\left(\Delta\mathbf{q}(\boldsymbol{\omega}, \Delta t) \cdot \mathbf{q}_n\right)
$$

## Discretization

Explicit Euler for position ($\mathbf{x} += \mathbf{V} \Delta t$). Exact quaternion rotation for orientation (rotation quaternion from angular velocity, then Hamilton product).

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\mathbf{V} = \mathbf{F}/R_T$ | `mime.nodes.robot.rigid_body.RigidBodyNode.update` | Element-wise divide by resistance diagonal |
| $C_1, C_2, C_3$ | `mime.nodes.robot.rigid_body.oberbeck_stechert_coefficients` | `jnp.where` guard for $e \to 0$ |
| $\Delta\mathbf{q}$ | `mime.core.quaternion.quat_from_angular_velocity` | Exact rotation quaternion |
| $\mathbf{q}$ normalisation | `mime.core.quaternion.quat_normalize` | `q / jnp.linalg.norm(q)` |

## Assumptions and Simplifications

1. Stokes regime: $Re \ll 1$, inertia negligible
2. Rigid body — no deformation
3. Prolate ellipsoid shape for analytical drag coefficients
4. Resistance tensor diagonal in body frame (no translation-rotation coupling in current implementation — coupling via R_12 deferred to IB-LBM)

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| Re | 0–0.1 | Stokes regime |
| semi_major_axis | 1–1000 um | Microrobot size range |

## Known Limitations and Failure Modes

1. Analytical drag only valid for $Re < 0.1$
2. No near-wall corrections (SurfaceContactNode needed)
3. No translation-rotation coupling (R_12 block) in analytical mode
4. Quaternion integration first-order (sufficient for overdamped dynamics)

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| position | (3,) | m | Centre of mass position |
| orientation | (4,) | - | Unit quaternion [w,x,y,z] |
| velocity | (3,) | m/s | Translational velocity |
| angular_velocity | (3,) | rad/s | Angular velocity |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| semi_major_axis_m | float | 100e-6 | m | Semi-major axis |
| semi_minor_axis_m | float | 100e-6 | m | Semi-minor axis (sphere if = a) |
| fluid_viscosity_pa_s | float | 8.5e-4 | Pa.s | CSF viscosity |
| use_analytical_drag | bool | True | - | Use Oberbeck-Stechert (vs. IB-LBM) |

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|-------|-------|---------|---------------|-------------|
| magnetic_force | (3,) | zeros | additive | From MagneticResponseNode |
| magnetic_torque | (3,) | zeros | additive | From MagneticResponseNode |
| drag_force | (3,) | zeros | additive | From CSFFlowNode (IB-LBM mode) |
| drag_torque | (3,) | zeros | additive | From CSFFlowNode (IB-LBM mode) |
| external_force | (3,) | zeros | additive | Gravity, contact, etc. |
| external_torque | (3,) | zeros | additive | Additional torques |

## MIME-Specific Sections

### Anatomical Operating Context

| Compartment | Flow Regime | Re Range | Viscosity Range |
|-------------|------------|----------|----------------|
| CSF | stagnant/pulsatile | 0–0.1 | 0.7–1.0 mPa.s |

### Clinical Relevance

The most fundamental node. Every microrobot simulation requires tracking position and orientation. All other physics (fluid coupling, drug delivery, sensing) reference the robot's pose.

## References

- [@Lighthill1976] Lighthill, J. (1976). *Flagellar Hydrodynamics*. — Resistance tensor theory for slender bodies.
- [@Rodenborn2013] Rodenborn, B. et al. (2013). *Propulsion of microorganisms by a helical flagellum*. — Experimental validation data.

## Verification Evidence

- MIME-VER-001: Stokes translational drag (< 5% error vs. analytical)
- MIME-VER-004: Ellipsoid drag anisotropy
- MIME-VER-005: Steady-state velocity consistency
- Unit tests: `tests/nodes/test_rigid_body.py` (16 tests)

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-03-20 | Initial implementation — analytical Oberbeck-Stechert drag |
