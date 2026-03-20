# MIME SOUP Package Document

**Version**: 0.1.0
**Date**: 2026-03-20

## 1. Software Identification

| Field | Value |
|-------|-------|
| Name | MIME |
| Full Name | MIcrorobotics Multiphysics Engine |
| Version | 0.1.0 |
| Release Date | 2026-03-20 |
| Licence | LGPL-3.0-or-later |
| Source Repository | https://github.com/Microrobotics-Simulation-Framework/MIME |
| Python Version | >=3.10 |
| Primary Dependencies | MADDENING (local), JAX >=0.4, jaxlib >=0.4, NumPy >=1.24 |
| Build System | hatchling |

## 2. Functional Description

### Core Capabilities

- MimeNode ABC extending MADDENING's SimulationNode with domain metadata
- Domain metadata: anatomical regimes, biocompatibility, actuation, sensing, therapeutic
- GeometrySource protocol for parametric and mesh geometries
- USDViewport protocol for swappable rendering backends
- Control layer: ControlPolicy, ControlPrimitive, ControlSequence, PolicyRunner
- Uncertainty layer: sensing and actuation uncertainty injection
- Asset schema: MimeAssetSchema with compliance gate and benchmark results
- Benchmark suite: B0–B5 validation benchmarks

### Capabilities NOT Provided

- MIME does not interpret simulation results
- MIME does not provide clinical recommendations
- MIME does not validate that simulation parameters match any real physical system
- MIME does not enforce safety limits on user-provided parameters
- BiocompatibilityMeta is a technical descriptor, NOT a biocompatibility assessment

## 3. Known Anomalies

See `docs/validation/known_anomalies.yaml` for the complete registry.

*No anomalies registered at v0.1.0.*

## 4. Verification Evidence

*To be completed when B0–B5 benchmarks are implemented (Phase 1+).*

## 5. IEC 62304 Lifecycle Activities

See `DOCUMENTATION_ARCHITECTURE.md` Section 10 for the full lifecycle mapping.

## 6. Configuration Management

- Version control: Git (GitHub)
- Release tags: semantic versioning (`vX.Y.Z`)
- CI: GitHub Actions (test matrix + compliance job)

## 7. Anomaly Management Policy

See `DOCUMENTATION_ARCHITECTURE.md` Section 8.5 for the three-phase lifecycle and three-tier release gate model.

## 8. Dependencies (SOUP of SOUP)

### 8.1 MADDENING

| Field | Value |
|-------|-------|
| Name | MADDENING |
| Version | 0.1.0 (local install) |
| Licence | LGPL-3.0-or-later |
| SOUP package | See MADDENING `docs/validation/soup_package.md` |
| Known anomalies | See MADDENING `docs/validation/known_anomalies.yaml` |

#### MADDENING Anomalies Relevant to MIME

| MADDENING Anomaly | Affects MIME? | MIME Impact | MIME Mitigation |
|---|---|---|---|
| MADD-ANO-001 (LBM GPU segfault) | Potentially | CSFFlowNode if wrapping LBMPipeNode | CI tests on CPU; GPU users warned |
| MADD-ANO-002 (HeatNode CFL not enforced) | No | MIME does not use HeatNode | N/A |

#### MADDENING Version Update Policy

MIME pins to a specific MADDENING version. When MADDENING publishes a new version:

1. Review MADDENING's CHANGELOG.md — specifically Known Anomalies, Security, Verification sections
2. Review new/changed entries in MADDENING's known_anomalies.yaml
3. Run MIME's full test suite against the candidate version
4. Update the version pin
5. Update this section: version, anomaly table
6. Document the update in MIME's CHANGELOG.md

### 8.2 Other Dependencies

| Dependency | Version | License | Purpose |
|------------|---------|---------|---------|
| jax | >=0.4 | Apache-2.0 | Automatic differentiation, JIT compilation |
| jaxlib | >=0.4 | Apache-2.0 | XLA backend |
| numpy | >=1.24 | BSD-3-Clause | Array operations |

These dependencies are themselves SOUP when MIME is used in a regulated product. See MADDENING's SOUP package for upstream dependency credibility assessment.
