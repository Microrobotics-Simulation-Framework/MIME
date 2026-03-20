---
bibliography: ../../bibliography.bib
---

# Flexible Body Node

**Module**: `mime.nodes.robot.flexible_body`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-006`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

Euler-Bernoulli beam dynamics for flexible flagellar microrobots. Models transverse bending waves along a filament under the balance of elastic restoring force and viscous drag.

## Governing Equations

$$
K\frac{\partial^4 y}{\partial x^4} = -\xi_{\perp} \frac{\partial y}{\partial t} + f_{\text{fluid}}(x,t)
$$

where $K = EI$ is the bending stiffness and $\xi_{\perp}$ is the perpendicular RFT drag coefficient (analytical fallback; replaced by IB-LBM forces in Phase 2+).

4th-order FD stencil:
$$
\frac{\partial^4 y}{\partial x^4} \approx \frac{y_{n+2} - 4y_{n+1} + 6y_n - 4y_{n-1} + y_{n-2}}{\Delta x^4}
$$

## Discretization

4th-order central finite differences for $\partial^4 y/\partial x^4$. Implicit Euler for time integration (required due to stiffness of the 4th-order spatial operator).

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $K \partial^4 y / \partial x^4$ | `build_beam_stiffness_matrix` | Stencil [1,-4,6,-4,1] / dx^4 |
| Implicit Euler | `FlexibleBodyNode.update` | `jnp.linalg.solve(A, rhs)` |
| Clamped BC at x=0 | `FlexibleBodyNode.update` | Row substitution in A matrix |

## Assumptions and Simplifications

1. Small-amplitude transverse deflections (linearised beam)
2. Inertia negligible (overdamped, low Re)
3. Uniform material properties along filament
4. Clamped-free boundary conditions

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| Sperm number $S_p$ | 0.5–3.0 | Optimal propulsion near $S_p \approx 2.1$ |

## Known Limitations and Failure Modes

1. Small-deformation only — fails for large curvatures ($S_p \gg 2.1$)
2. 1D transverse deflection — no torsion or 3D shape
3. RFT drag coefficient is a scalar approximation

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| deflection | (N,) | m | Transverse displacement y(x) |
| velocity | (N,) | m/s | Transverse velocity dy/dt |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| n_nodes | int | 20 | - | Discretisation nodes |
| length_m | float | 100e-6 | m | Filament length |
| bending_stiffness_nm2 | float | 4e-21 | N.m^2 | EI |
| drag_coeff_perp | float | auto | Pa.s | RFT $\xi_{\perp}$ |

## References

- [@Lighthill1976] Lighthill, J. (1976). *Flagellar Hydrodynamics*. — Slender body theory and RFT for flagellar filaments.

## Verification Evidence

- Unit tests: `tests/nodes/test_flexible_body.py` (7 tests)

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-03-20 | Initial implementation — small-deformation Euler-Bernoulli beam |
