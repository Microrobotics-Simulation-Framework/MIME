---
bibliography: ../../bibliography.bib
---

# Surface Contact Node

**Module**: `mime.nodes.robot.surface_contact`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-007`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

Near-wall drag corrections and soft penalty contact forces for a sphere near a plane wall. Implements Brenner (1961) perpendicular and Goldman-Cox-Brenner (1967) parallel wall corrections using the leading-order expansion form.

## Governing Equations

Brenner perpendicular correction (leading order):
$$
F_{\perp,\text{corrected}} = 6\pi\mu a V_\perp \left(1 + \frac{9a}{8h}\right)
$$

Goldman-Cox-Brenner parallel correction (leading order):
$$
F_{\parallel,\text{corrected}} = 6\pi\mu a V_\parallel \left(1 + \frac{9a}{16h}\right)
$$

Penalty contact force:
$$
\mathbf{F}_{\text{contact}} = k \cdot \max(-\text{gap}, 0) \cdot \hat{\mathbf{n}}_{\text{wall}}
$$

## Discretization

Analytical — closed-form evaluation. No spatial discretisation.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| Brenner $1 + 9a/(8h)$ | `brenner_correction_perpendicular` | Clamped $h \geq 1.5a$ |
| Goldman $1 + 9a/(16h)$ | `brenner_correction_parallel` | Clamped $h \geq 1.5a$ |
| Penalty force | `penalty_contact_force` | Linear spring, `jnp.maximum(-gap, 0)` |

## Assumptions and Simplifications

1. Sphere near an infinite plane wall
2. Stokes regime ($Re \ll 1$)
3. $h/a > 1.5$ (correction series valid; clamped below this)
4. Flat wall — no curvature corrections

## Known Limitations and Failure Modes

1. First-order truncation of infinite Brenner series
2. Penalty contact force is numerical regularisation, not physical
3. No adhesion (van der Waals, electrostatic)
4. Single wall only

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| wall_correction_perp | () | - | Perpendicular drag multiplier |
| wall_correction_par | () | - | Parallel drag multiplier |
| contact_force | (3,) | N | Penalty contact force |
| gap_distance | () | - | h/a ratio |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| robot_radius_m | float | 100e-6 | m | Sphere radius |
| wall_position | float | 0.0 | m | Wall location |
| wall_normal_axis | int | 2 | - | Wall normal axis (0=x, 1=y, 2=z) |
| contact_stiffness | float | 1e-6 | N/m | Penalty spring stiffness |

## MIME-Specific Sections

### Differentiability

**DIFFERENTIABILITY-LIMITED**: The penalty contact force has a kink at gap=0. `jax.grad` through contact events produces unreliable gradients. Use in forward simulation only, or apply a smoothed softplus approximation for differentiability.

### Clinical Relevance

Microrobots in CSF channels must navigate without wall adhesion. Contact with vessel walls is inevitable in narrow channels (aqueduct diameter ~1.5mm). Wall drag corrections are needed for physically meaningful confined navigation simulation.

## References

- [@Purcell1977] Purcell, E.M. (1977). *Life at Low Reynolds Number*. — Low-Re hydrodynamics fundamentals.

## Verification Evidence

- Unit tests: `tests/nodes/test_surface_contact.py` (13 tests)

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-03-20 | Initial implementation — leading-order Brenner/Goldman corrections + penalty contact |
