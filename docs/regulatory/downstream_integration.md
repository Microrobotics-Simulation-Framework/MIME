# Downstream Integration — Dependency Chain and Responsibilities

## Four-Layer Dependency Chain

```
Layer 1: MADDENING          (open source, LGPL, general-purpose framework)
    |
Layer 2: MIME                (open source, LGPL, microrobotics physics engine)
    |
Layer 3: MICROBOTICA         (open source, AGPL, research simulator + registry)
    |
Layer 4: [Commercial Product] (regulated, CE-marked, built by commercial entity)
```

| Layer | Status | Regulatory Obligation |
|-------|--------|----------------------|
| MADDENING | Open-source research tool | None (provides SOUP documentation voluntarily) |
| MIME | Open-source research tool | None (provides SOUP documentation voluntarily) |
| MICROBOTICA | Open-source research tool | None (provides SOUP documentation voluntarily) |
| Commercial Product | Regulated medical device | Full EU MDR manufacturer obligations |

## Per-Layer Responsibility Details

### Layer 1: MADDENING

**Role**: General-purpose JAX-based multiphysics simulation framework.

**Provides**: Core simulation infrastructure (graph management, JIT compilation, multi-rate scheduling, adaptive timestepping), physics node library, compliance infrastructure (`NodeMeta`, anomaly registry, `@verification_benchmark`, `@stability`).

**Does NOT provide**: Clinical claims, domain-specific physics for microrobotics, risk management file (ISO 14971).

### Layer 2: MIME (this project)

**Role**: Domain-specific physics engine for microrobot simulation, built on MADDENING.

**Provides**: MimeNode ABC, domain metadata (anatomical regimes, biocompatibility, actuation, sensing, therapeutic), GeometrySource protocol, control layer, uncertainty layer, asset schema, B0–B5 benchmark suite, own SOUP documentation and anomaly registry (MIME- prefix).

**Inherits from MADDENING**: Compliance schema types via `maddening.compliance`, `@verification_benchmark` and `@stability` decorators, `HealthCheckNode` base class.

### Layer 3: MICROBOTICA

**Role**: Simultaneously a full robotics simulator (C++17/Qt 6, USD scene authoring, 3D viewport) AND a community registry/leaderboard platform.

**Provides**: Pre-configured simulation scenarios, interactive visualisation, benchmark execution infrastructure, asset registry with leaderboard, own SOUP documentation (MBOT- prefix).

### Layer 4: Commercial Product

**This is the ONLY layer subject to EU MDR.** The commercial entity is the EU MDR manufacturer (Article 2(30)) and is solely responsible for all regulatory obligations.

## Commercial Responsibility Statement

The Layer 4 commercial entity is solely responsible for:

- Establishing and maintaining an **ISO 13485 QMS**
- Performing the **conformity assessment** procedure
- Engaging and satisfying the **Notified Body**
- Building and maintaining the **clinical evaluation report**
- Operating **post-market surveillance**
- Bearing **manufacturer liability**
- Performing **SOUP assessment** of all open-source layers per IEC 62304
- Maintaining **risk management file** (ISO 14971)
