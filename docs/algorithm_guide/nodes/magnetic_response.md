---
bibliography: ../../bibliography.bib
---

# Magnetic Response Node

**Module**: `mime.nodes.robot.magnetic_response`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-002`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

Computes magnetic torque and force on a soft-magnetic microrobot body from an externally applied field. Implements the induced magnetization model with anisotropic susceptibility tensor and saturation clipping.

## Governing Equations

Induced magnetization in the body frame:
$$
\mathbf{m} = \frac{1}{\mu_0} \boldsymbol{\chi}_a \mathbf{B}_{\text{body}}
$$

Susceptibility tensor (diagonal in body frame):
$$
\boldsymbol{\chi}_a = \text{diag}\left(\frac{1}{n_{\text{axi}}}, \frac{1}{n_{\text{rad}}}, \frac{1}{n_{\text{rad}}}\right)
$$

Magnetic torque (lab frame):
$$
\mathbf{T}_{\text{mag}} = v (\mathbf{m}_{\text{lab}} \times \mathbf{B}_{\text{lab}})
$$

Magnetic force (lab frame):
$$
\mathbf{F}_{\text{mag}} = v (\mathbf{m}_{\text{lab}} \cdot \nabla)\mathbf{B}
$$

Saturation clipping: $|\mathbf{m}| \leq m_{\text{sat}}$.

## Discretization

Analytical — no discretisation. Single-evaluation algebraic computation.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\mathbf{m} = \chi_a \mathbf{B} / \mu_0$ | `MagneticResponseNode.update` | Element-wise multiply with `chi_diag` |
| Body frame rotation | `MagneticResponseNode.update` | `rotate_vector_inverse(q, B_lab)` |
| $\mathbf{T} = v(\mathbf{m} \times \mathbf{B})$ | `MagneticResponseNode.update` | `jnp.cross(m_lab, B_lab)` |
| $\mathbf{F} = v(\mathbf{m} \cdot \nabla)\mathbf{B}$ | `MagneticResponseNode.update` | `grad_B @ m_lab` |
| Saturation clip | `MagneticResponseNode.update` | `jnp.where(m_mag > m_sat, ...)` |

## Assumptions and Simplifications

1. Linear magnetization below saturation ($\mathbf{m} < \mathbf{m}_{\text{sat}}$)
2. No hysteresis or remnant magnetization (ideal soft-magnet)
3. Susceptibility tensor diagonal in body frame
4. Demagnetization factors satisfy $n_{\text{axi}} + 2n_{\text{rad}} = 1$

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| field_strength | 0–0.1 T | Below saturation for Co80Ni20 |

## Known Limitations and Failure Modes

1. Linear approximation fails above saturation magnetization
2. No hysteresis — cannot capture field-history effects
3. Assumes ellipsoidal body shape for demagnetization factors

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| magnetization | (3,) | A/m | Current magnetization (lab frame) |
| magnetic_torque | (3,) | N.m | Magnetic torque (lab frame) |
| magnetic_force | (3,) | N | Magnetic force (lab frame) |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| volume_m3 | float | 1e-15 | m^3 | Volume of magnetic material |
| n_axi | float | 0.2 | - | Axial demagnetization factor |
| n_rad | float | 0.4 | - | Radial demagnetization factor |
| m_sat | float | 0 | A/m | Saturation magnetization (0=disabled) |

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|-------|-------|---------|---------------|-------------|
| field_vector | (3,) | zeros | replacive | B field from ExternalMagneticFieldNode |
| field_gradient | (3,3) | zeros | replacive | dB/dx from ExternalMagneticFieldNode |
| orientation | (4,) | [1,0,0,0] | replacive | Quaternion from RigidBodyNode |

## Boundary Fluxes (outputs)

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| magnetic_torque | (3,) | N.m | To RigidBodyNode (additive) |
| magnetic_force | (3,) | N | To RigidBodyNode (additive) |

## MIME-Specific Sections

### Biocompatibility Context

Default material: Co80Ni20. Saturation magnetization: 1.19 x 10^6 A/m. Biocompatibility NOT assessed — manufacturer must perform ISO 10993 evaluation.

### Clinical Relevance

Converts the external actuation field into the forces and torques that drive microrobot motion. The step-out phenomenon (when viscous drag exceeds maximum magnetic torque) is fundamentally a magnetic response effect.

## References

- [@Abbott2009] Abbott, J.J. et al. (2009). *How Should Microrobots Swim?* — Magnetic actuation analysis.

## Verification Evidence

- Integration test: `tests/nodes/test_integration.py` (coupled chain)
- Unit tests: `tests/nodes/test_magnetic_response.py` (14 tests)

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-03-20 | Initial implementation — soft-magnet model |
