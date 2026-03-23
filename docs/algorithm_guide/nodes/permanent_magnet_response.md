---
bibliography: ../../bibliography.bib
---

# Permanent Magnet Response Node

**Module**: `mime.nodes.robot.permanent_magnet_response`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-008`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

Computes magnetic torque and force on a permanent-magnet microrobot body from an externally applied field. The magnetic moment is fixed in the body frame (rigid permanent magnet), rotated to the lab frame via the robot quaternion. No susceptibility tensor, no saturation, no volume multiplier.

## Governing Equations

Fixed moment in body frame:
$$
\mathbf{m}_{\text{body}} = n \cdot m_{\text{single}} \cdot \hat{\mathbf{a}}
$$

where $\hat{\mathbf{a}}$ is the normalised moment axis and $n$ is the number of magnets.

Rotate to lab frame:
$$
\mathbf{m}_{\text{lab}} = R(\mathbf{q}) \, \mathbf{m}_{\text{body}}
$$

Magnetic torque (lab frame):
$$
\mathbf{T} = \mathbf{m}_{\text{lab}} \times \mathbf{B}
$$

Magnetic force (lab frame):
$$
\mathbf{F} = (\nabla \mathbf{B}) \, \mathbf{m}_{\text{lab}}
$$

## Discretization

Analytical — no discretisation. Single-evaluation algebraic computation.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\mathbf{m}_{\text{body}} = n \cdot m_{\text{single}} \cdot \hat{\mathbf{a}}$ | `mime.nodes.robot.permanent_magnet_response.PermanentMagnetResponseNode.update` | Fixed moment from constructor params |
| $\mathbf{m}_{\text{lab}} = R(\mathbf{q}) \mathbf{m}_{\text{body}}$ | `mime.nodes.robot.permanent_magnet_response.PermanentMagnetResponseNode.update` | `rotate_vector(q, m_body)` |
| $\mathbf{T} = \mathbf{m}_{\text{lab}} \times \mathbf{B}$ | `mime.nodes.robot.permanent_magnet_response.PermanentMagnetResponseNode.update` | `jnp.cross(m_lab, B_lab)` |
| $\mathbf{F} = (\nabla \mathbf{B}) \mathbf{m}_{\text{lab}}$ | `mime.nodes.robot.permanent_magnet_response.PermanentMagnetResponseNode.update` | `grad_B @ m_lab` |

## Assumptions and Simplifications

1. Rigid permanent moment (no demagnetization)
2. Moment axis fixed in body frame
3. No temperature dependence of moment

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| field_strength | 0--0.1 T | NdFeB permanent magnet in mT-range field |

## Known Limitations and Failure Modes

1. No demagnetization effects
2. No hysteresis
3. No temperature-dependent moment

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| magnetization | (3,) | A*m^2 | Current magnetic moment (lab frame) |
| magnetic_torque | (3,) | N.m | Magnetic torque (lab frame) |
| magnetic_force | (3,) | N | Magnetic force (lab frame) |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| n_magnets | int | 1 | - | Number of permanent magnets |
| m_single | float | 1.07e-3 | A*m^2 | Magnetic moment per magnet |
| moment_axis | tuple | (0,1,0) | - | Moment direction in body frame |

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

Default material: NdFeB N45. Biocompatibility NOT assessed — manufacturer must perform ISO 10993 evaluation. Typically encapsulated in biocompatible resin or parylene coating.

### Clinical Relevance

Computes the torque and force on an untethered magnetic robot (UMR) containing permanent magnets driven by an external rotating field. The step-out phenomenon occurs when the field rotation frequency exceeds the maximum torque capacity of the permanent magnets.

## References

- [@deBoer2025] de Boer, M.C.J. et al. (2025). *Wireless mechanical and hybrid thrombus fragmentation of ex vivo endovascular thrombosis model in the iliac artery.*

## Verification Evidence

- Unit tests: `tests/nodes/test_permanent_magnet_response.py`

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-03-22 | Initial implementation — permanent magnet model |
