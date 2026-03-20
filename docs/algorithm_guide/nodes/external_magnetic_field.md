---
bibliography: ../../bibliography.bib
---

# External Magnetic Field Node

**Module**: `mime.nodes.actuation.external_magnetic_field`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-001`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

Generates a rotating uniform magnetic field B(t) = B_0 * [cos(omega*t), sin(omega*t), 0], modelling a Helmholtz coil pair driven in quadrature or a distant rotating permanent magnet.

## Governing Equations

$$
\mathbf{B}(t) = B_0 \begin{pmatrix} \cos(2\pi f t) \\ \sin(2\pi f t) \\ 0 \end{pmatrix}
$$

where $B_0$ is the field magnitude [T] and $f$ is the rotation frequency [Hz].

For a coil array, the general form is $\mathbf{B}(\mathbf{p}) = \mathcal{B}(\mathbf{p}) \mathbf{I}$ (Appendix C), but the current implementation uses the uniform-field approximation (valid near workspace centre).

## Discretization

Analytical — no discretisation. The field is evaluated exactly at each timestep.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $B_0 \cos(\omega t)$ | `mime.nodes.actuation.external_magnetic_field.ExternalMagneticFieldNode.update` | `jnp.cos(omega * t)` |
| $B_0 \sin(\omega t)$ | `mime.nodes.actuation.external_magnetic_field.ExternalMagneticFieldNode.update` | `jnp.sin(omega * t)` |
| mT to T conversion | `mime.nodes.actuation.external_magnetic_field.ExternalMagneticFieldNode.update` | `strength_mt * 1e-3` |

## Assumptions and Simplifications

1. Uniform field over the workspace (valid near Helmholtz coil centre)
2. No eddy currents or shielding from biological tissue
3. Coil inductance delay negligible (quasi-static field)
4. Field rotation in xy-plane only

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| frequency_hz | 0–200 | Typical microrobot actuation range |
| field_strength_mt | 0–100 | Below saturation for most soft-magnetic materials |

## Known Limitations and Failure Modes

1. Uniform field approximation invalid far from workspace centre
2. No spatial gradient in uniform mode (gradient = 0) — only torque actuation, no gradient force
3. 2D rotation only — no out-of-plane field components

## Stability Conditions

Unconditionally stable — analytical evaluation with no numerical integration.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| field_vector | (3,) | T | Current B field |
| field_gradient | (3,3) | T/m | Spatial gradient (zero in uniform mode) |
| sim_time | () | s | Accumulated simulation time |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| frequency_hz | float | 10.0 | Hz | Commandable rotation frequency |
| field_strength_mt | float | 10.0 | mT | Commandable field magnitude |

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|-------|-------|---------|---------------|-------------|
| frequency_hz | () | 10.0 | replacive | Rotation frequency from ControlPolicy |
| field_strength_mt | () | 10.0 | replacive | Field magnitude from ControlPolicy |

## Boundary Fluxes (outputs)

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| field_vector | (3,) | T | B field for MagneticResponseNode |
| field_gradient | (3,3) | T/m | dB/dx for MagneticResponseNode |

## MIME-Specific Sections

### Clinical Relevance

The external magnetic field is the primary actuation mechanism for helical microrobots navigating CSF. Field frequency and strength are the main control inputs for the ControlPolicy.

## References

- [@Abbott2009] Abbott, J.J. et al. (2009). *How Should Microrobots Swim?* — Analysis of magnetic actuation strategies for microrobots.

## Verification Evidence

- MIME-VER-005: Full chain force-velocity consistency
- MIME-VER-006: 1000-step chain stability
- Unit tests: `tests/nodes/test_external_magnetic_field.py` (15 tests)

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-03-20 | Initial implementation — uniform rotating field |
