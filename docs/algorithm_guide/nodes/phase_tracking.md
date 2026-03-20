---
bibliography: ../../bibliography.bib
---

# Phase Tracking Node

**Module**: `mime.nodes.robot.phase_tracking`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-005`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

Observational node that computes the phase error between the robot's magnetic moment direction and the external rotating field. Detects step-out when phase error exceeds pi/2.

## Governing Equations

Phase error:
$$
\theta_{\text{err}} = \arccos(\hat{\mathbf{m}} \cdot \hat{\mathbf{B}})
$$

where $\hat{\mathbf{m}}$ is the robot's body e1 axis rotated to the lab frame and $\hat{\mathbf{B}} = \mathbf{B}/|\mathbf{B}|$.

Step-out detection:
$$
\text{stepped\_out} = (\theta_{\text{err}} > \pi/2)
$$

## Discretization

Analytical — instantaneous angle computation. No time integration.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\hat{\mathbf{m}} = R(\mathbf{q}) \hat{\mathbf{e}}_1$ | `PhaseTrackingNode.update` | `rotate_vector(q, [1,0,0])` |
| $\cos\theta = \hat{\mathbf{m}} \cdot \hat{\mathbf{B}}$ | `PhaseTrackingNode.update` | `jnp.dot(m_hat, B_hat)` |
| $\theta = \arccos(\cos\theta)$ | `PhaseTrackingNode.update` | `jnp.arccos(jnp.clip(...))` |

## Assumptions and Simplifications

1. Robot magnetic moment aligned with body e1 axis
2. Step-out threshold is pi/2 (maximum torque angle)
3. No phase error ODE — instantaneous detection only

## Known Limitations and Failure Modes

1. No model of the asynchronous wobbling regime ($\omega > \omega_c$)
2. `arccos` has poor gradient signal near 0 and pi — use `cos_phase_error` for optimization
3. Zero field produces undefined phase error (clamped to safe value)

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| phase_error | () | rad | Angle between m and B [0, pi] |
| cos_phase_error | () | - | cos(phase_error) — better for gradients |
| stepped_out | () | bool | True if phase_error > pi/2 |

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|-------|-------|---------|---------------|-------------|
| orientation | (4,) | [1,0,0,0] | replacive | Quaternion from RigidBodyNode |
| field_vector | (3,) | zeros | replacive | B field from ExternalMagneticFieldNode |

## MIME-Specific Sections

### Clinical Relevance

Step-out detection is critical for microrobot navigation safety. Loss of synchronisation means loss of directional control. The StepOutDetector feedback policy uses this node's output to implement automatic frequency recovery.

## References

- [@Abbott2009] Abbott, J.J. et al. (2009). *How Should Microrobots Swim?* — Step-out analysis and swimming strategy.

## Verification Evidence

- Integration test: `tests/nodes/test_integration.py` (phase error tracking in coupled chain)
- Unit tests: `tests/nodes/test_phase_tracking.py` (11 tests)

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-03-20 | Initial implementation — phase error and step-out detection |
