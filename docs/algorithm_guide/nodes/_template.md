---
bibliography: ../../bibliography.bib
---

# [Node Name]

**Module**: `mime.nodes.[subpackage].[module]`
**Stability**: [experimental | provisional | stable | deprecated]
**Algorithm ID**: `MIME-NODE-[XXX]`
**Version**: [semantic version]
**Verification Mode**: [Mode 1 (Wrapping) | Mode 2 (Independent)]
**Upstream Node**: [MADDENING node class, if Mode 1]
**MADDENING Version Pin**: [exact version, if Mode 1]

## Summary

[1-2 sentence description of what this node simulates.]

## Governing Equations

[Full mathematical formulation. Use LaTeX math blocks.]

$$
...
$$

## Discretization

[How the continuous equations are discretized. Explicit/implicit,
finite difference/finite element/lattice Boltzmann, order of accuracy.]

## Implementation Mapping

[Trace every term in the governing equations and discretization to the
specific Python/JAX function that implements it. Mandatory for
IEC 62304 Class C detailed design traceability (Clause 5.4).]

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|

## Assumptions and Simplifications

1. [e.g., "Incompressible flow (Mach number << 1)"]

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|

## Known Limitations and Failure Modes

[Feeds into IEC 62304 SOUP anomaly assessment.]

1. [e.g., "CFL > 1 causes numerical instability"]

## Stability Conditions

[Analytical or empirical stability bounds for the numerical scheme.]

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|-------|-------|---------|---------------|-------------|

## Boundary Fluxes (outputs)

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|

## MIME-Specific Sections

### Anatomical Operating Context

| Compartment | Flow Regime | Re Range | pH Range | Temp Range | Viscosity Range |
|-------------|------------|----------|----------|------------|----------------|

### Biocompatibility Context (robot_body nodes only)

[Materials, ISO 10993 classification, biocompatibility hazard hints.]

### Clinical Relevance

[Brief description of why this physics matters for the clinical application.
Not a clinical claim — context for understanding the model's role.]

### Mode 1 Scope Statement (if applicable)

[Cites upstream version, upstream verification IDs, documents what MIME adds.]

### Mode 2 Independent Verification (if applicable)

[Lists all independent verification evidence.]

## References

- [@ExampleKey] Author (Year). *Title*. — Relevance note.

## Verification Evidence

- Test files: `tests/verification/test_*.py`

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | YYYY-MM-DD | Initial implementation |
