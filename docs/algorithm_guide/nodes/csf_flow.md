---
bibliography: ../../bibliography.bib
---

# CSF Flow Node

**Module**: `mime.nodes.environment.csf_flow`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-004`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

Analytical Stokes drag on a spherical body in quiescent or pulsatile CSF. Computes translational and rotational drag forces without resolving the flow field. This is the analytical fallback for benchmarks B0 and B2; it will be augmented/replaced by IB-LBM for full fluid-structure coupling.

## Governing Equations

Stokes drag (quiescent fluid):
$$
\mathbf{F}_{\text{drag}} = -6\pi\mu a (\mathbf{V} - \mathbf{u}^{\infty})
$$

Rotational drag:
$$
\mathbf{T}_{\text{drag}} = -8\pi\mu a^3 \boldsymbol{\omega}
$$

Pulsatile background flow (simplified centreline sinusoidal):
$$
\mathbf{u}^{\infty}(t) = v_{\text{peak}} \sin(\omega_c t) \hat{\mathbf{z}}
$$

## Discretization

Analytical — closed-form evaluation each timestep.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\mathbf{F} = -6\pi\mu a(\mathbf{V} - \mathbf{u})$ | `mime.nodes.environment.csf_flow.CSFFlowNode.update` | `jnp` arithmetic |
| $\mathbf{T} = -8\pi\mu a^3 \omega$ | `mime.nodes.environment.csf_flow.CSFFlowNode.update` | `jnp` arithmetic |
| $\mathbf{u}^{\infty}(t)$ | `mime.nodes.environment.csf_flow.CSFFlowNode.update` | `jnp.sin(omega_c * t)` |

## Assumptions and Simplifications

1. Spherical body for drag computation
2. Stokes regime ($Re \ll 1$)
3. Newtonian fluid (CSF at physiological protein levels)
4. Robot small compared to channel ($a \ll R$)
5. Pulsatile mode: centreline velocity only (no radial profile)

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| Re | 0–0.1 | Stokes regime |
| robot_radius | 10–500 um | Microrobot size range |

## Known Limitations and Failure Modes

1. No resolved flow field — point-force drag only
2. No fluid-structure interaction (one-way coupling)
3. No Faxen correction (negligible for $a = 100\mu$m at cardiac frequency)
4. No Basset history force (3.3% correction at cardiac frequency)

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| drag_force | (3,) | N | Drag force on robot |
| drag_torque | (3,) | N.m | Rotational drag |
| background_velocity | (3,) | m/s | CSF flow at robot position |
| sim_time | () | s | Accumulated time |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| fluid_viscosity_pa_s | float | 8.5e-4 | Pa.s | CSF dynamic viscosity |
| fluid_density_kg_m3 | float | 1002 | kg/m^3 | CSF density |
| robot_radius_m | float | 100e-6 | m | Effective hydrodynamic radius |
| pulsatile | bool | False | - | Enable pulsatile background flow |
| cardiac_freq_hz | float | 1.1 | Hz | Cardiac pulsation frequency |
| peak_velocity_m_s | float | 0.04 | m/s | Peak centreline velocity |

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|-------|-------|---------|---------------|-------------|
| position | (3,) | zeros | replacive | Robot position |
| velocity | (3,) | zeros | replacive | Robot velocity |
| angular_velocity | (3,) | zeros | replacive | Robot angular velocity |

## Boundary Fluxes (outputs)

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| drag_force | (3,) | N | To RigidBodyNode (additive) |
| drag_torque | (3,) | N.m | To RigidBodyNode (additive) |

## MIME-Specific Sections

### Anatomical Operating Context

| Compartment | Flow Regime | Re Range | Viscosity Range |
|-------------|------------|----------|----------------|
| CSF (aqueduct) | pulsatile | 0–0.1 | 0.7–1.0 mPa.s |

### Clinical Relevance

CSF flow is pulsatile (cardiac + respiratory). Drag on the microrobot determines navigation accuracy and energy budget. Pulsatile flow creates time-varying forces that affect closed-loop control performance.

## References

- [@Purcell1977] Purcell, E.M. (1977). *Life at Low Reynolds Number*. — Foundational low-Re hydrodynamics.

## Verification Evidence

- MIME-VER-001: Stokes translational drag (< 5% error)
- MIME-VER-002: Stokes rotational drag
- MIME-VER-003: Drag linearity
- Unit tests: `tests/nodes/test_csf_flow.py` (11 tests)

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-03-20 | Initial implementation — analytical Stokes drag |
