# MIME Documentation Architecture Plan

## Executive Summary

**MIME** (MIcrorobotics Multiphysics Engine) is a Python/JAX domain-specific physics engine for microrobot simulation. It occupies Layer 2 of a four-layer open-source stack:

```
Layer 1 — MADDENING  (Python, JAX, LGPL-3.0)
    JAX-based differentiable multiphysics graph framework.
    Has its own DOCUMENTATION_ARCHITECTURE.md.

Layer 2 — MIME  (Python, LGPL-3.0)  ← THIS PROJECT
    MIcrorobotics Multiphysics Engine. Built on MADDENING.
    Node classes, asset schema, control, uncertainty, benchmarks.

Layer 3 — MICROBOTICA  (C++17 / Qt 6.6, AGPL-3.0)
    Open-source research simulator + community registry/leaderboard.
    USD-based scene authoring, 3D viewport, embedded Python scripting.

Layer 4 — Commercial Product  (future, by a spin-out or licensee)
    CE-marked SaMD built on top of the open-source stack.
    Bears all EU MDR obligations.
```

MIME serves three audiences simultaneously: researchers building microrobotics simulations with physics-informed models; developers implementing new `MimeNode` subclasses for additional physics; and downstream commercial manufacturers who need to cite MIME's verification record in regulatory submissions — without MIME itself making any clinical claims.

**Regulatory context**: The developer is EU-based (Netherlands). **EU MDR (EU 2017/745)** is the primary regulatory framework. A commercial product built on top of the stack would face EU MDR Class III classification for the most demanding plausible use case: intraoperative digital twin or RL policy validation for a device operating in the central nervous system, where erroneous output could cause irreversible neurological harm or death. The documentation architecture must be designed to support this most demanding plausible classification so that less demanding use cases are automatically covered.

Under **IEC 62304**, when any downstream commercial manufacturer incorporates MIME into a regulated product, MIME is classified as SOUP (Software of Unknown Provenance). For a Class III device, MIME will almost certainly be classified as **IEC 62304 Class C SOUP**, carrying the most stringent lifecycle documentation requirements. MADDENING, on which MIME depends, is MIME's own SOUP dependency (SOUP-of-SOUP).

**Relationship to MADDENING's documentation architecture**: This document inherits structural patterns, regulatory boundary language, compliance schema conventions, and tooling choices from MADDENING's `DOCUMENTATION_ARCHITECTURE.md`. Where a section is directly inherited, this is stated explicitly. Where a section is adapted for the microrobotics physics domain, the adaptation rationale is documented. Where a section is new (no MADDENING equivalent), the design rationale is provided from first principles. This document is self-contained — it can be read without MADDENING's document, though references are provided for deeper context.

This plan is informed by: **ASME V&V 40**, **FDA computational modeling guidance (2016, 2023)**, **IEC 62304:2006+AMD1:2015**, **EU MDR (EU 2017/745)**, **MDCG 2019-11**, **MDCG 2019-16**, **IMDRF SaMD N10/N12/N23**, **EN ISO 13485:2016**, and **ISO 14971:2019**.

---

## Table of Contents

1. [Documentation Structure](#1-documentation-structure)
2. [Regulatory Boundary Language](#2-regulatory-boundary-language)
3. [Algorithm and Node Documentation Standards](#3-algorithm-and-node-documentation-standards)
4. [V&V Documentation Hooks](#4-vv-documentation-hooks)
5. [Versioning and API Stability](#5-versioning-and-api-stability)
6. [Contributor and Extensibility Documentation](#6-contributor-and-extensibility-documentation)
7. [README and Repository-Level Documentation](#7-readme-and-repository-level-documentation)
8. [Code-Embedded Documentation and Structural Hooks](#8-code-embedded-documentation-and-structural-hooks)
   - 8.1 [NodeMeta and MimeNodeMeta Schemas](#81-nodemeta-and-mimenodameta-schemas)
   - 8.2 [Provenance and Reproducibility](#82-provenance-and-reproducibility)
   - 8.3 [Verification Test Registration](#83-verification-test-registration)
   - 8.4 [Deprecation and Stability Machinery](#84-deprecation-and-stability-machinery)
   - 8.5 [Anomaly Management System](#85-anomaly-management-system)
   - 8.6 [ISO 14971 Risk Management Hooks](#86-iso-14971-risk-management-hooks)
9. [IEC 62304 SOUP Compliance Package](#9-iec-62304-soup-compliance-package)
10. [IEC 62304 Software Lifecycle Documentation Mapping](#10-iec-62304-software-lifecycle-documentation-mapping)
11. [EU MDR Annex I and Annex II Alignment](#11-eu-mdr-annex-i-and-annex-ii-alignment)
12. [Configuration Management](#12-configuration-management)
13. [QMS Compatibility](#13-qms-compatibility)
14. [Registry Documentation](#14-registry-documentation)

**Appendices:**

- [A: Inheritance from MADDENING Summary Table](#appendix-a-inheritance-from-maddening-summary-table)
- [B: MIME New Node Checklist](#appendix-b-mime-new-node-checklist)

---

## 1. Documentation Structure

### Inheritance Statement

This section adapts MADDENING Section 1 for a Python-based microrobotics physics engine. The three-tier documentation model (API reference, algorithm guides, executable examples) is preserved. The tooling (Sphinx, MyST-Parser, sphinx-autodoc) is identical since both MADDENING and MIME are Python projects.

### MIME Documentation Structure

MIME adopts the same three-tier documentation model as MADDENING, adapted for the microrobotics domain:

```
MIME/
├── README.md                          # Project overview + disclaimers
├── ARCHITECTURE_PLAN.md               # Architecture decisions
├── MIME_NODE_TAXONOMY.md              # Node taxonomy (scientific rationale)
├── DOCUMENTATION_ARCHITECTURE.md      # This document
├── CHANGELOG.md                       # Structured changelog
├── CITATION.cff                       # Academic citation metadata
├── CONTRIBUTING.md                    # Contributor guide + MIME-NODE- prefix
├── SECURITY.md                        # Security reporting
├── LICENSE                            # LGPL-3.0-or-later
├── pyproject.toml                     # Build config (hatchling, src layout)
│
├── src/mime/                          # Python package (src layout)
│   ├── core/
│   │   ├── metadata.py               # Domain meta dataclasses (no JAX dep)
│   │   ├── node.py                   # MimeNode ABC
│   │   ├── geometry.py               # GeometrySource protocol
│   │   └── viewport.py               # USDViewport protocol
│   ├── nodes/{actuation,robot,environment,sensing,therapeutic}/
│   ├── control/
│   ├── uncertainty/
│   ├── schema/
│   └── benchmarks/
│
├── docs/
│   ├── conf.py                        # Sphinx configuration (MyST-Parser, bibtex)
│   ├── index.rst                      # Documentation root
│   │
│   ├── user_guide/
│   │   ├── installation.md
│   │   ├── quickstart.md              # First magnetic robot simulation
│   │   ├── concepts.md                # MimeNode, roles, control, uncertainty
│   │   └── tutorials/
│   │       ├── helical_robot.md       # Phase 1 tutorial: helical in CSF
│   │       └── drug_delivery.md       # Phase 2 tutorial: targeted delivery
│   │
│   ├── developer_guide/
│   │   ├── node_authoring.md          # How to write a MimeNode
│   │   ├── documentation_standards.md # MIME documentation requirements
│   │   └── testing_standards.md       # MIME test requirements
│   │
│   ├── algorithm_guide/               # Mathematical documentation (Tier 2)
│   │   ├── index.md                   # Algorithm documentation overview
│   │   ├── nodes/                     # One document per MimeNode
│   │   │   ├── _template.md           # MIME node algorithm guide template
│   │   │   ├── external_magnetic_field.md
│   │   │   ├── magnetic_response.md
│   │   │   ├── rigid_body.md
│   │   │   ├── csf_flow.md
│   │   │   ├── phase_tracking.md
│   │   │   └── ...
│   │   └── control/
│   │       ├── policy_runner.md       # PolicyRunner algorithm
│   │       └── uncertainty_models.md  # UncertaintyModel implementations
│   │
│   ├── validation/                    # V&V documentation
│   │   ├── index.md                   # V&V philosophy and scope
│   │   ├── known_anomalies.yaml       # MIME-ANO-* entries
│   │   ├── soup_package.md            # MIME SOUP document
│   │   ├── benchmark_reports/         # B0–B5 verification reports
│   │   │   ├── b0_experimental.md
│   │   │   ├── b1_step_out.md
│   │   │   ├── b2_stokes_drag.md
│   │   │   └── ...
│   │   └── cou_template.md            # Context-of-use template for microrobotics
│   │
│   ├── regulatory/                    # Regulatory context documentation
│   │   ├── intended_use.md            # MIME platform positioning statement
│   │   ├── downstream_integration.md  # MIME→MICROBOTICA→Commercial chain
│   │   └── iec62304_mapping.md        # MIME-specific lifecycle mapping
│   │
│   ├── api_reference/                 # Tier 1: Auto-generated API docs
│   │   └── (sphinx-autodoc output)
│   │
│   └── bibliography.bib              # MIME-specific academic references
│
├── tests/
│   ├── verification/                  # MIME-VER-* benchmarks
│   │   ├── test_experimental.py       # B0
│   │   ├── test_step_out.py           # B1
│   │   ├── test_stokes_drag.py        # B2
│   │   └── ...
│   └── ...
│
└── scripts/
    ├── check_anomalies.py             # Delegates to maddening.compliance
    ├── check_citations.py             # Validates MIME bibliography
    └── check_impl_mapping.py          # Validates implementation mappings
```

**Rationale**: The structure mirrors MADDENING's three-tier model (API reference + algorithm guide + examples) with two key additions: (1) `docs/algorithm_guide/control/` for the control and uncertainty layer (no MADDENING equivalent — MADDENING has no control abstraction); (2) benchmark reports in `docs/validation/benchmark_reports/` for the B0–B5 suite (MADDENING's benchmarks are per-node; MIME's are multi-node system-level scenarios).

**Tooling**: Sphinx with MyST-Parser (Markdown with LaTeX math support via `$...$` and `$$...$$`), `sphinxcontrib-bibtex` for centralised bibliography, sphinx-autodoc for API reference, PyData Sphinx Theme (consistent with MADDENING). Intersphinx cross-referencing links MIME's docs to MADDENING's API docs (e.g., linking to `GraphManager`, `NodeMeta`).

---

## 2. Regulatory Boundary Language

### Inheritance Statement

This section is **directly inherited** from MADDENING Section 2 with surface substitutions (project name, scope, layer position). The "platform, not product" framing, layered responsibility model, and commercial boundary language follow the same structure. The key difference is that MIME sits at Layer 2 — the physics engine that defines the domain-specific content that downstream tools consume.

### `docs/regulatory/intended_use.md`

**Platform Positioning Statement**:

> MIME (MIcrorobotics Multiphysics Engine) is a domain-specific physics engine for microrobot simulation, built on the MADDENING framework. It is open-source research software distributed under LGPL-3.0-or-later.
>
> **MIME is not a medical device** as defined by EU MDR (EU 2017/745) Article 2(1). It does not have a medical purpose, does not make clinical predictions, and does not provide diagnostic, therapeutic, or monitoring functionality. MIME is a computational tool for simulating microrobot physics — analogous to a domain-specific extension of a finite element library. Under the qualification criteria of MDCG 2019-11, software without a medical purpose is not a medical device.
>
> MIME provides microrobotics-specific physics models (magnetic actuation, rigid body dynamics in viscous flow, drug release kinetics) and structured metadata (anatomical operating regimes, biocompatibility descriptors, benchmark results) that downstream tools may use. When incorporated into a regulated medical device, MIME is classified as SOUP (Software of Unknown Provenance) under IEC 62304. MADDENING, on which MIME depends, is MIME's own SOUP dependency (SOUP-of-SOUP).
>
> The downstream commercial manufacturer is solely responsible for assessing MIME's (and MADDENING's) suitability for their specific context of use and for performing all required regulatory activities.

**Cybersecurity Boundary Statement** (MDCG 2019-16):

> MIME is a Python physics library that assumes trusted inputs. It performs no input sanitisation, authentication, authorisation, or network security functions. All simulation parameters, graph topologies, external inputs, and boundary conditions are assumed to be provided by a trusted caller.
>
> When MIME is incorporated into a regulated product, the commercial integration layer is solely responsible for: validating and sanitising all simulation parameters before they reach MIME/MADDENING; ensuring graph topologies are well-formed and represent physically meaningful configurations; and preventing injection of malicious parameters through any user-facing interface.

**LGPL Replaceability Statement**:

> MIME is licensed under LGPL-3.0-or-later. The LGPL "replaceability" obligation is satisfied via Python's standard module system: the end user can replace the `mime` package by installing a modified version into the same Python environment. No special linking, build steps, or binary compatibility mechanisms are required.

**Layered Responsibility Model**:

| Responsibility | Owner | Applicable Standard | Evidence |
|---|---|---|---|
| Physics algorithm correctness | MIME/MADDENING projects | IEC 62304 Clause 5.6 (via SOUP assessment) | Test suite, B0–B5 benchmarks, analytical comparisons |
| SOUP assessment (MIME) | Downstream manufacturer | IEC 62304 Clause 5.3.3, 5.3.4 | Using MIME's SOUP package |
| SOUP assessment (MADDENING) | Downstream manufacturer | IEC 62304 Clause 5.3.3, 5.3.4 | Using MADDENING's SOUP package (SOUP-of-SOUP) |
| Known anomaly evaluation | Downstream manufacturer | IEC 62304 Clause 7.1.3 | Using MIME's known anomalies registry |
| Calculation verification for specific COU | Downstream user | ASME V&V 40 | Mesh convergence, solver settings, error estimation |
| Validation for specific COU | Device manufacturer | EU MDR Annex I Section 17.2 | Experimental comparisons for their COU |
| Risk management | Device manufacturer | ISO 14971, EU MDR Annex I Chapter I | Risk management file |
| Clinical evidence | Device manufacturer | EU MDR Annex XIV | Clinical evaluation report; PMCF plan (Class III) |
| Regulatory submission | Device manufacturer | EU MDR Article 52+ | CE marking, Notified Body review |

### `docs/regulatory/downstream_integration.md`

Documents the specific four-layer dependency chain:

```
Layer 1: MADDENING          (open source, LGPL, general-purpose framework)
    |
Layer 2: MIME                (open source, LGPL, microrobotics physics engine)
    |
Layer 3: MICROBOTICA         (open source, AGPL, research simulator + registry)
    |
Layer 4: [Commercial Product] (regulated, CE-marked, built by commercial entity)
```

This document must address:
- MIME's specific scope (microrobotics physics) distinct from MADDENING's (general multiphysics)
- Which MADDENING components MIME depends on (listed in SOUP package Section 8)
- The dual role of MICROBOTICA: full robotics simulator (USD scenes, 3D visualisation, hardware-in-the-loop) AND community registry/leaderboard (asset publication, benchmark ranking)
- Why the benchmark suite requires MICROBOTICA: benchmarks are reproducibly executed in the simulator and results are published to the registry
- Commercial entity's responsibilities (ISO 13485 QMS, Notified Body, post-market surveillance, manufacturer liability)

### The Commercial Boundary

**MADDENING, MIME, and MICROBOTICA are open-source research tools. None of them will seek CE marking. None of them are medical devices. None of them carry manufacturer liability under EU MDR.**

The regulated clinical product is built by a downstream commercial entity on top of these open-source tools. That entity takes on the QMS, the Notified Body relationship, post-market surveillance obligations, and EU MDR manufacturer liability. This mirrors how ITK, VTK, and 3D Slicer relate to commercial products built on them.

---

## 3. Algorithm and Node Documentation Standards

### Inheritance Statement

This section **adapts** MADDENING Section 3 for MIME's domain. MADDENING documents physics nodes with governing equations, discretisation, and analytical benchmarks using the `NodeMeta` schema. MIME extends this with a second metadata layer (`MimeNodeMeta`) carrying domain-specific information: anatomical operating regimes, biocompatibility descriptors, actuation parameters, and sensing characteristics. The algorithm guide template is extended correspondingly.

### Per-Node Documentation Requirements

Every `MimeNode` subclass must document to the same standard as a MADDENING node, plus additional domain-specific requirements.

#### Required Artifacts

| Artifact | Description |
|----------|-------------|
| `meta` ClassVar | `NodeMeta` with `algorithm_id` (MIME-NODE-*), `stability`, `description`, `governing_equations`, `discretization`, `assumptions`, `limitations`, `validated_regimes`, `references`, `hazard_hints`, `implementation_map` |
| `mime_meta` ClassVar | `MimeNodeMeta` with role, anatomical regimes, domain-specific metadata |
| Algorithm guide | Document in `docs/algorithm_guide/nodes/` following the MIME template |
| NumPy-style docstring | Parameters, state fields, boundary inputs |
| Unit tests | Normal operation, edge cases, JAX-traceability (jit, grad, vmap) |
| Verification benchmark | At least one `@verification_benchmark` (MIME-VER-*) |
| Integration test | Test within a `GraphManager` with edges |
| Bibliography entries | All cited references in `docs/bibliography.bib` |
| Known anomalies | Limitations in `docs/validation/known_anomalies.yaml` |

#### Domain Metadata (`MimeNodeMeta`)

- **Node role**: which physical subsystem (external_apparatus, robot_body, environment, sensing, therapeutic)
- **Anatomical regimes**: physiological operating contexts with quantitative bounds (compartment, Re range, pH range, temperature range, viscosity range)
- **Role-specific metadata**:
  - `robot_body` nodes: `BiocompatibilityMeta` (materials, ISO 10993 level, biocompatibility hazard hints)
  - `external_apparatus` nodes: `ActuationMeta` (principle, commandable fields, force/torque specs)
  - `sensing` nodes: `SensingMeta` (modality, resolution, SNR, imaging artefact hints)
  - `therapeutic` nodes: `TherapeuticMeta` (payload, release kinetics, target anatomy)
- **`GeometrySource` dependency documented**: spatial nodes (`environment` and some `sensing` roles) that consume a `GeometrySource` must document: (a) which `GeometrySource` subtypes they accept (parametric only, or mesh also), (b) the minimum geometry description required (e.g., minimum mesh resolution, required coordinate frame), (c) which benchmarks require which geometry types. This ensures that B0/B4-T1 (parametric cylinder) and B4-T2/T3 (Neurobotika mesh) geometry requirements are traceable to specific node documentation.

### MIME Algorithm Guide Template

The MIME algorithm guide template (`docs/algorithm_guide/nodes/_template.md`) extends MADDENING's template with domain-specific sections. Every physics node must have a corresponding document following this template:

```markdown
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
[1-2 sentence description]

## Governing Equations
[Full LaTeX formulation using $$...$$ blocks]

## Discretization
[Numerical method, order of accuracy]

## Implementation Mapping
[IEC 62304 Clause 5.4 traceability table — every equation term to code.
Mandatory for all MIME physics nodes.]

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|

## Assumptions and Simplifications
[Numbered list of every physical and mathematical assumption]

## Validated Physical Regimes
| Parameter | Verified Range | Notes |
|-----------|---------------|-------|

## Known Limitations and Failure Modes
[Feeds into IEC 62304 SOUP anomaly assessment]

## Stability Conditions
[Numerical stability bounds]

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
[Which anatomical compartments and flow regimes this node is validated for.
Maps to `AnatomicalRegimeMeta` entries in `mime_meta`.]

| Compartment | Flow Regime | Re Range | pH Range | Temp Range | Viscosity Range |
|-------------|------------|----------|----------|------------|----------------|

### Biocompatibility Context (robot_body nodes only)
[Materials, ISO 10993 classification, biocompatibility hazard hints.
Maps to `BiocompatibilityMeta` in `mime_meta`.]

### Clinical Relevance
[Brief description of why this physics matters for the clinical application.
Not a clinical claim — context for understanding the model's role.]

### Mode 1 Scope Statement (if applicable)
[For nodes that wrap unmodified MADDENING nodes. Cites upstream version,
upstream verification IDs, and documents what MIME adds.]

### Mode 2 Independent Verification (if applicable)
[For nodes with new or modified physics. Lists all independent verification
evidence. No upstream evidence may be cited.]

## References
[@Key] citations with inline descriptions

## Verification Evidence
[Links to benchmark reports and test files]

## Changelog
| Version | Date | Change |
|---------|------|--------|
```

### Implementation Mapping

The `implementation_map` field in `NodeMeta` and the corresponding Markdown table in the algorithm guide are **mandatory** for all MIME physics nodes (not just recommended). This is because MIME nodes will be assessed as Class C SOUP for the most demanding downstream use cases, and IEC 62304 Clause 5.4 requires detailed design traceability at Class C.

MIME's `scripts/check_impl_mapping.py` validates that every function name in the Implementation Mapping table resolves to an existing callable in the codebase. This runs on every push and fails if any function name is stale.

### Bibliography and Citation System

MIME maintains its own `docs/bibliography.bib` containing domain-specific references:
- Microrobot dynamics papers (Purcell, Lighthill, Nelson, Martel, Fischer groups)
- CSF dynamics (Linninger, Kurtcuoglu, Sweetman groups)
- Drug delivery modelling (Higuchi, Korsmeyer, Saltzman groups)
- Magnetic actuation (Abbott, Nelson, Martel groups)
- Medical imaging physics (MRI, ultrasound)

Citations use Pandoc-style `[@Key]` syntax. CI validates all citations via `scripts/check_citations.py`. Each algorithm guide includes YAML frontmatter with `bibliography: ../../bibliography.bib` so that Pandoc can locate the `.bib` file automatically. Each References section includes human-readable inline descriptions alongside `[@Key]` markers so documents are readable without Pandoc rendering.

If a MIME algorithm guide cites a reference that also appears in MADDENING's bibliography, the entry is **copied into MIME's `.bib` file**. This avoids cross-project file dependencies and ensures MIME's documentation build is self-contained.

---

## 4. V&V Documentation Hooks

### Inheritance Statement

This section **adapts** MADDENING Section 4 for MIME's domain. MADDENING's V&V scope is general-purpose physics correctness (analytical benchmarks, convergence studies for heat, LBM, etc.). MIME's V&V scope is microrobotics-specific: step-out frequency validation, Stokes drag, drug release kinetics, closed-loop navigation, and experimental validation against published data. The V&V boundary principle — "we verify our layer, not upstream layers" — is preserved identically.

### MIME V&V Scope

**What MIME's V&V provides**:
- **Physics algorithm correctness**: B0–B5 benchmarks verify that MIME's physics nodes correctly solve the intended mathematical equations within documented validated regimes
- **Experimental validation**: B0 compares simulation output against published experimental data (Rodenborn et al. 2013)
- **Multi-node system verification**: B1, B4, B5 test the coupled system (magnetic field + magnetic response + rigid body + flow + control) rather than individual nodes in isolation
- **Robustness verification**: B4 and B5 test closed-loop performance under sensing and actuation uncertainty via ensemble evaluation

**What MIME's V&V does NOT provide**:
- **Clinical validation**: MIME does not demonstrate that any simulation output matches patient physiology. Clinical validation is the commercial manufacturer's responsibility.
- **Calculation verification for a specific COU**: MIME cannot verify that a specific simulation setup (mesh, parameters, boundary conditions) is adequate for a specific clinical question. That is the downstream user's responsibility per ASME V&V 40.
- **Framework-level verification**: MADDENING's `GraphManager`, `EdgeSpec`, coupling groups, multi-rate timestepping, and JIT compilation are verified by MADDENING's own test suite. MIME does not re-verify these.

### B0–B5 Benchmark Documentation

Each benchmark gets its own verification report in `docs/validation/benchmark_reports/`. The report follows this structure:

```markdown
# B[N]: [Benchmark Name] — Verification Report

**Benchmark ID**: MIME-VER-[XXX]
**Status**: [PASS | FAIL | NOT YET IMPLEMENTED]
**Last Run**: [date, MIME version, MADDENING version, MICROBOTICA version]

## Problem Description
[Physical setup, governing equations, analytical/reference solution]

## Acceptance Criterion
[Quantitative pass/fail criterion from ARCHITECTURE_PLAN.md §9]

## Test Configuration
[Node graph, parameters, initial conditions, control sequence]

## Results
[Measured metric, comparison to criterion, plots/tables]

## Scope and Limitations
[What this benchmark demonstrates and what it does not. Computational
verification, not clinical validation.]

## Reproducibility
[Hardware, software versions, RNG seed. Sufficient for exact reproduction.]
```

### Benchmark Results in MimeAssetSchema

Benchmark results in `MimeAssetSchema.benchmark_results` serve as V&V evidence. The traceability chain is:

```
MimeAssetSchema.benchmark_results[i]
  → BenchmarkResult.benchmark_id (e.g., "B1")
    → MIME-VER-* verification benchmark
      → docs/validation/benchmark_reports/b1_step_out.md
```

When a benchmark result is published to the MICROBOTICA registry, it includes all version information (MIME, MADDENING, MICROBOTICA, hardware) needed to reproduce the result. This traceability chain allows a manufacturer to:
1. Find an asset on the registry with benchmark results
2. Trace each result to the verification report
3. Understand exactly what was tested, under what conditions, with what acceptance criteria
4. Determine whether the verification is adequate for their context of use

### Mode 1 / Mode 2 Verification Evidence

MIME follows MADDENING's two-mode verification inheritance model (MADDENING Section 16):

**Mode 1 (Wrapping)**: For MIME nodes that wrap unmodified MADDENING nodes (e.g., CSFFlowNode wrapping LBMPipeNode), MIME may cite MADDENING's verification evidence with a Scope Statement. Additionally:

- A regression test must assert numerical agreement between the MIME wrapper and the MADDENING node for identical inputs. The tolerance must be explicitly documented using a three-tier tolerance classification:
  - **Tier 1 (bitwise / ULP)**: for purely algebraic wrappers with no additional computation
  - **Tier 2 (relative tolerance, rtol ~ 1e-5 to 1e-6)**: for nodes that introduce minor floating-point reordering
  - **Tier 3 (physics-derived tolerance)**: for LBM and other methods where floating-point non-determinism arises from JIT operation reordering or wrapper call-path divergence — not from stochasticity. The tolerance is derived from the known convergence properties of the numerical scheme and must be documented with a citation and a physical interpretation. A comment in the test file must explain: (a) why exact floating-point equality is unachievable (citing JIT reordering and/or wrapper path divergence as applicable), (b) the convergence order of the specific scheme, (c) the grid resolution used in the test, and (d) the physical meaning of the tolerance (e.g., "tolerance corresponds to < 0.1% error in drag force prediction at Re = 0.01, which is within the scheme's O(dx^2) truncation error at this resolution").

  LBM-based nodes should use Tier 3. LBM is deterministic given the same initial conditions and execution order, but two compounding sources of floating-point non-determinism prevent Tier 1/2 agreement: (1) `jax.jit` may reorder floating-point operations across calls, producing results that differ by more than machine epsilon even for identical inputs; (2) the MIME wrapper may invoke the MADDENING LBM node through a slightly different call path (boundary input resolution, state dict packing/unpacking) that introduces additional floating-point reordering relative to calling the MADDENING node directly.

- MIME must provide supplementary verification in CSF-specific parameter regimes

**Mode 2 (Independent)**: For nodes with new or modified physics, MIME provides full independent verification. No MADDENING evidence may be cited.

---

## 5. Versioning and API Stability

### Inheritance Statement

**Taken verbatim in structure** from MADDENING Section 5, with surface substitutions only.

### Semantic Versioning

MIME follows strict semantic versioning (SemVer 2.0):

- **MAJOR** (X.0.0): Breaking API changes, removed features, changed behaviour
- **MINOR** (0.X.0): New features, new nodes, new architectures (backward-compatible)
- **PATCH** (0.0.X): Bug fixes, documentation updates, performance improvements

Pre-release versions: `X.Y.Z-alpha.N`, `X.Y.Z-beta.N`, `X.Y.Z-rc.N`

### `CHANGELOG.md`

Following [Keep a Changelog](https://keepachangelog.com/) with regulatory-specific sections:

```markdown
# Changelog

All notable changes to MIME are documented in this file.

## [Unreleased]

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Verification
- [Changes to B0–B5 status, new verification benchmarks]
### Security
- [Required by MDCG 2019-16 cybersecurity guidance]
### Known Anomalies
- [Changes to MIME-ANO-* entries]
```

The **Verification**, **Security**, and **Known Anomalies** sections directly support EU regulatory workflows (IEC 62304, MDCG 2019-16).

### Commit Message Convention

| Prefix | Meaning |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring (no behaviour change) |
| `docs:` | Documentation only |
| `test:` | Test additions or changes |
| `perf:` | Performance improvement |
| `verify:` | Verification/validation evidence |
| `break:` | Breaking change |
| `deprecate:` | Deprecation notice |
| `security:` | Security-relevant change |

### API Stability Levels

Each public API surface carries a stability level (see Section 8.4 for code-level machinery):

| Level | Meaning | SemVer Guarantee |
|-------|---------|-----------------|
| **stable** | Covered by SemVer; breaking changes only in major versions | Full |
| **provisional** | API may change in minor versions with deprecation warnings | One minor version notice |
| **experimental** | API may change without notice | None |
| **deprecated** | Scheduled for removal; use alternative | Removed in next major |

### Version Tracking

Three version numbers are relevant to MIME:

| Version | What it tracks | Where it appears |
|---------|---------------|-----------------|
| MIME version | MIME package version (SemVer) | `pyproject.toml`, CHANGELOG, SOUP package |
| MIME schema version | `MimeAssetSchema` format version | `mime_schema_version` field in every asset |
| MADDENING version pin | Which MADDENING version MIME is tested against | `pyproject.toml`, SOUP package Section 8 |

Schema version changes follow their own SemVer:
- **MAJOR**: breaking changes to `MimeAssetSchema` (removed fields, changed semantics)
- **MINOR**: new optional fields, new enum values
- **PATCH**: documentation, description changes only

Each MIME node has an `algorithm_version` in its `NodeMeta`. This tracks changes to the numerical implementation:
- **MAJOR**: changed governing equations, different physical model
- **MINOR**: additional terms, extended parameter support, new boundary conditions
- **PATCH**: bug fixes, numerical improvements that don't change the mathematical model

---

## 6. Contributor and Extensibility Documentation

### Inheritance Statement

This section **adapts** MADDENING Section 7 for MIME. The contributor checklist structure is preserved; the checklist items extend MADDENING's with MIME-specific requirements.

### `CONTRIBUTING.md` (Repository Root)

Top-level contributor guide covering: development environment setup, branching model, commit convention, code style (NumPy-style docstrings, math-heavy code exception for mathematical variable names), and links to detailed guides. Documents the `MIME-NODE-`, `MIME-ANO-`, `MIME-VER-` ID prefix convention.

### `docs/developer_guide/node_authoring.md`

Comprehensive guide for writing a new `MimeNode`:

1. **The contract**: pure functions, JAX-traceability, parameters vs. state, `requires_halo` property
2. **Directory structure**: where the file lives, naming conventions
3. **Required metadata**: `NodeMeta` + `MimeNodeMeta` (Section 8.1)
4. **Required documentation**: docstring format + algorithm guide document (Section 3)
5. **Required tests**: unit tests + at least one verification benchmark (Section 8.3)
6. **Boundary inputs**: `BoundaryInputSpec` with `coupling_type` for additive inputs
7. **GeometrySource dependency**: spatial nodes must declare which `GeometrySource` subtypes they accept
8. **Checklist**: see Appendix B

### `docs/developer_guide/testing_standards.md`

Test requirements at three levels:

1. **Unit tests** (mandatory): test each public method, verify JAX-traceability (jit, grad, vmap)
2. **Integration tests** (mandatory for nodes): test the node within a `GraphManager` with edges
3. **Verification benchmarks** (mandatory for physics nodes): comparison to analytical solution, reference simulation, or published experimental data, registered via `@verification_benchmark` from `maddening.compliance`

---

## 7. README and Repository-Level Documentation

### Inheritance Statement

**Taken verbatim in structure** from MADDENING Section 8, adapted for MIME.

### Root-Level Governance Files

| File | Purpose |
|------|---------|
| `CITATION.cff` | Machine-readable citation metadata + configuration management artifact |
| `CONTRIBUTING.md` | Quick-start for contributors + ID prefix convention |
| `LICENSE` | LGPL-3.0-or-later |

---

## 8. Code-Embedded Documentation and Structural Hooks

This section defines concrete code-level mechanisms that make compliance, auditability, and documentation structural properties of the framework. It adapts MADDENING Section 9 for MIME's domain.

### 8.1 NodeMeta and MimeNodeMeta Schemas

#### Inheritance Statement

MIME uses MADDENING's `NodeMeta` schema (imported from `maddening.compliance`) unchanged and adds a second metadata layer (`MimeNodeMeta`) for domain-specific information. Both are pure Python dataclasses with no JAX dependency.

#### Dual-Metadata Design

Every `MimeNode` carries two metadata ClassVars:

- `meta: ClassVar[Optional[NodeMeta]]` — the standard MADDENING metadata consumed by the MADDENING harvester (`collect_node_metadata()`), compliance tooling, and Sphinx documentation build
- `mime_meta: ClassVar[Optional[MimeNodeMeta]]` — MIME-specific metadata consumed by MIME's own tooling, `MimeAssetSchema`, and the MICROBOTICA registry

The `meta` field is set directly (not aliased) because MADDENING's harvester reads `cls.meta`.

#### Scope Distinction: `validated_regimes` vs. `hazard_hints`

Inherited from MADDENING Section 9.1. Two `NodeMeta` fields carry risk-relevant information with non-overlapping scopes:

| Field | Nature | Scope | Consumer |
|---|---|---|---|
| `validated_regimes` | Quantitative, parameter-bound | Defines the envelope within which the node has been verified. Each entry specifies a named parameter, a numeric min/max range, and the evidence source. | Runtime validators, algorithm guide "Validated Physical Regimes" table |
| `hazard_hints` | Qualitative, non-parameter-bound | Describes technical conditions that cannot be reduced to a single parameter range — algorithmic limitations, modelling assumptions, platform-specific issues. | ISO 14971 hazard identification input, SOUP package "Known Limitations" section |

If a risk can be expressed as "parameter X must be within [a, b]," it belongs in `validated_regimes`. If it cannot, it belongs in `hazard_hints`. A given risk appears in exactly one of the two fields, never both.

#### Automatic Harvesting

```python
from maddening.compliance import collect_node_metadata, collect_hazard_hints

# Collects NodeMeta from all SimulationNode subclasses (MADDENING + MIME)
all_meta = collect_node_metadata()
all_hints = collect_hazard_hints()

# MIME-specific domain hints require a separate collector:
def collect_mime_hazard_hints() -> dict[str, dict]:
    """Collect domain-specific hazard hints from all MimeNode subclasses."""
    from mime.core.node import MimeNode
    result = {}
    for cls in MimeNode.__subclasses__():
        mm = getattr(cls, "mime_meta", None)
        if mm is None:
            continue
        hints = {}
        if mm.biocompatibility and mm.biocompatibility.biocompatibility_hazard_hints:
            hints["biocompatibility"] = list(mm.biocompatibility.biocompatibility_hazard_hints)
        if mm.sensing and mm.sensing.imaging_artifact_hints:
            hints["imaging"] = list(mm.sensing.imaging_artifact_hints)
        if hints:
            result[cls.__name__] = hints
    return result
```

### 8.2 Provenance and Reproducibility

Inherited from MADDENING Section 9.2. MIME uses MADDENING's `SimulationProvenance` dataclass (which captures software versions, graph topology, initial state hash, RNG state, and final state hash) unchanged. `PolicyRunner` should capture additional provenance fields: policy class name, policy constructor parameters, uncertainty model configuration, and B0–B5 benchmark IDs if running a benchmark.

### 8.3 Verification Test Registration

Inherited from MADDENING Section 9.3. MIME uses MADDENING's `@verification_benchmark` decorator imported from `maddening.compliance`. Verification benchmarks are registered with `MIME-VER-` prefixed IDs and run as part of CI. Structured output (JSON) is automatically aggregated into `docs/validation/`.

### 8.4 Deprecation and Stability Machinery

Inherited from MADDENING Section 9.5. MIME uses MADDENING's `@stability` decorator imported from `maddening.compliance`. All MIME public APIs are annotated with stability levels. New nodes start as `EXPERIMENTAL` and promote to `STABLE` when: at least one verification benchmark passes, algorithm guide is complete, and API has been stable for at least one minor release.

### 8.5 Anomaly Management System

#### Inheritance Statement

This section is **taken verbatim in structure** from MADDENING Section 9.7. MIME uses the same three-phase lifecycle, three-tier release gate, and YAML anomaly registry schema. The `validate_anomaly_registry()` function from `maddening.compliance` validates MIME's registry with prefix enforcement (`--prefix MIME-ANO-`).

#### Anomaly Registry

MIME maintains its own `docs/validation/known_anomalies.yaml` with `MIME-ANO-*` entries. The schema is identical to MADDENING's:

```yaml
schema_version: "1.0"
mime_version: "0.1.0"
generated_date: "2026-XX-XX"

anomalies:
  - anomaly_id: "MIME-ANO-001"
    title: "..."
    description: "..."
    affected_components: ["..."]
    affected_versions: "0.1.0 – current"
    severity: "major"
    safety_relevance: "context_dependent"
    safety_relevance_rationale: "..."
    workaround: "..."
    resolution_status: "open"
```

#### Three-Phase Lifecycle

1. **Discovery** (GitHub Issue): anomaly reported via structured issue template
2. **Formalisation** (YAML entry): developer manually creates the registry entry — this is a deliberate regulatory sign-off act, never automated
3. **CI verification**: schema validation, cross-reference checks, release gate enforcement

#### Three-Tier Release Gate

- **Tier 1** — `safety-relevant` label: hard gate, no grace period
- **Tier 2** — `anomaly:critical` or `anomaly:major`: no grace period
- **Tier 3** — `anomaly:minor`: two-cycle grace period (CI warning after one cycle, blocking after two)

#### Cross-Referencing

MIME anomalies may trace to MADDENING anomalies. The convention is to reference the upstream anomaly ID in the `description` or `safety_relevance_rationale`:

```yaml
- anomaly_id: "MIME-ANO-003"
  title: "CSF flow solver inherits LBM GPU segfault from MADDENING"
  safety_relevance_rationale: >
    Inherits from MADD-ANO-001. See MADDENING known_anomalies.yaml
    v0.3.2 for the upstream analysis.
```

### 8.6 ISO 14971 Risk Management Hooks

#### Inheritance Statement

This section **adapts** MADDENING Section 9.8 for MIME's domain. The technical-vs-clinical boundary principle is inherited unchanged. MIME adds a second layer of domain-specific hazard hints that complement MADDENING's general technical hints.

#### Two-Layer Hazard Hint Architecture

MIME nodes carry hazard hints at two levels:

**MADDENING layer** (`NodeMeta.hazard_hints`): Technical conditions related to the numerical implementation — numerical instability, unvalidated parameter regimes, algorithmic limitations.

**MIME layer** (`MimeNodeMeta` domain-specific hints): Domain-specific conditions related to the microrobotics application:

- `biocompatibility_hazard_hints` (in `BiocompatibilityMeta`): material-related concerns, e.g., "nickel content exceeds ISO 10993-5 cytotoxicity threshold for extended implantation"
- `imaging_artifact_hints` (in `SensingMeta`): imaging-related concerns, e.g., "susceptibility artefact from NdFeB robot obscures surrounding tissue within 5mm radius"
- Anatomical regime warnings: e.g., "model validated only for lateral ventricle CSF; behaviour in subarachnoid space uncharacterised"

#### Hazard Register Document

For a downstream commercial manufacturer, the hazard register connects MIME's hazard hints to the clinical context:

```
MIME hazard_hint (technical)
  → Manufacturer's hazard identification (ISO 14971 Clause 5.4)
    → Risk estimation (probability × severity for specific COU)
      → Risk evaluation (acceptable / unacceptable)
        → Risk control measure (if unacceptable)
```

MIME provides the first step only — the technical hint. The manufacturer provides the clinical context, risk estimation, and risk controls. This boundary is non-negotiable: a physics library cannot make clinical risk judgements because it does not know the clinical context.

---

## 9. IEC 62304 SOUP Compliance Package

### Inheritance Statement

This section **adapts** MADDENING Section 10 for MIME. The SOUP package template structure is identical; the content is MIME-specific.

### MIME as SOUP

When used in a regulated product, MIME is IEC 62304 SOUP. MADDENING is MIME's own SOUP dependency (SOUP-of-SOUP). The MIME SOUP package (`docs/validation/soup_package.md`) provides:

1. **Software identification** (Clause 5.3.3): MIME name, version, licence, SHA-256 hashes, Python version, primary dependencies (MADDENING version pin)
2. **Functional description** (Clause 5.3.4 support): list of all node types with descriptions, control layer, uncertainty layer, asset schema, benchmark suite, capabilities NOT provided
3. **Known anomalies** (Clauses 7.1.2, 7.1.3): MIME's own anomaly registry (`MIME-ANO-*` entries), with a summary table and a separate table for anomalies requiring Class C assessment
4. **Verification evidence** (Clauses 5.5/5.6 support): B0–B5 benchmark results, per-node verification, unit verification traceability table
5. **IEC 62304 lifecycle activities performed**: references to Section 10
6. **Configuration management**: Git, tagged releases, SBOM
7. **Anomaly management policy**: references to Section 8.5
8. **Dependencies (SOUP-of-SOUP)**: MADDENING at pinned version with SOUP package link, known anomalies link, SHA-256 hash, and a table of which MADDENING anomalies affect MIME

### MIME-Specific SOUP Concerns

- **Domain-specific limitations**: MIME models specific physics (CSF flow, magnetic actuation, drug delivery) with specific assumptions. "CSF viscosity assumed constant at 0.7 mPa·s" is a MIME-specific limitation with direct clinical relevance.
- **Anatomical regime validation**: `AnatomicalRegimeMeta` entries define physiological conditions under which each node has been verified. Operating outside these regimes means uncharacterised behaviour.
- **Biocompatibility metadata disclaimer**: `BiocompatibilityMeta` is a technical descriptor for search and comparison, **not** a biocompatibility assessment. The manufacturer must perform their own ISO 10993 evaluation.
- **Benchmark suite as V&V evidence**: B0–B5 results are computational verification, not clinical validation.

### Inheriting MADDENING's Compliance Infrastructure

MIME imports all compliance tooling from `maddening.compliance`:

| What | Import | MIME Usage |
|------|--------|-----------|
| Anomaly registry validator | `validate_anomaly_registry(path, prefix="MIME-ANO-")` | Validate MIME's registry in CI |
| NodeMeta harvester | `collect_node_metadata()` | Harvest metadata from all MimeNode subclasses |
| Hazard hints harvester | `collect_hazard_hints()` | Collect all MIME hazard hints for risk management |
| Verification benchmark decorator | `@verification_benchmark` | Register MIME benchmarks (MIME-VER-*) |
| Stability decorator | `@stability` | Mark MIME API stability levels |

**Compliance debt warning.** Inheriting MADDENING's compliance infrastructure is only as good as the content placed within it. `MIME-ANO-` entries must not remain as stubs indefinitely — each known limitation discovered during implementation must be formalised as an anomaly record before the affected node is marked `StabilityLevel.PROVISIONAL` or higher. The same applies to per-node `implementation_map` entries: a node with an empty or incomplete implementation map has not met its IEC 62304 Clause 5.4 traceability requirement regardless of what MADDENING's infrastructure provides. The inheritance model provides the *structure*; the engineer provides the *content*. Treat compliance documentation as structured engineering notes written for yourself, not as overhead written for a notified body.

---

## 10. IEC 62304 Software Lifecycle Documentation Mapping

### Inheritance Statement

This section **adapts** MADDENING Section 11 for MIME. The mapping structure is identical; the evidence artefacts reference MIME's own documentation.

### Scope

**MIME is not subject to IEC 62304.** It is an open-source research tool, not a medical device. This mapping is provided voluntarily to support downstream SOUP assessment.

### Mapping

| IEC 62304 Phase | Clause | What the Standard Requires | What MIME Provides |
|---|---|---|---|
| **Software development planning** | 5.1 | Development plan, standards, tools | `ARCHITECTURE_PLAN.md`, `CONTRIBUTING.md`, Git + GitHub CI |
| **Software requirements analysis** | 5.2 | Documented software requirements | `ARCHITECTURE_PLAN.md` Sections 1–9, test suite |
| **Software architectural design** | 5.3 | Architecture document, SOUP identification | `ARCHITECTURE_PLAN.md`, graph-based architecture, functional purity |
| **Software detailed design** | 5.4 | Detailed design for each software unit | Per-node algorithm documentation in `docs/algorithm_guide/`, `NodeMeta` + `MimeNodeMeta`, Implementation Mapping tables |
| **Software unit implementation** | 5.5 | Implement per detailed design | Source code in `mime/`, NumPy-style docstrings |
| **Software unit verification** | 5.6 | Verify each unit against design | `tests/` with verification benchmarks, B0–B5 suite |
| **Software integration testing** | 5.7 | Integration testing | Integration tests within `GraphManager`, multi-node benchmark scenarios |
| **Software system testing** | 5.8 | System-level testing | B4/B5 system-level benchmarks (full actuation chain + control + uncertainty) |
| **Software release** | 5.9 | Release documentation, known anomalies | `CHANGELOG.md`, tagged releases, `known_anomalies.yaml`, SOUP package |

---

## 11. EU MDR Annex I and Annex II Alignment

### Inheritance Statement

This section **references** MADDENING Section 12, which provides the full Annex I/II walkthrough. The same principles apply to MIME unchanged. Key points:

- **GSPR 17** (software requirements): MIME satisfies this through its algorithm documentation, V&V evidence, and known anomalies registry
- **Annex II** (technical documentation): MIME's SOUP package, algorithm guides, and verification reports provide specific, identified inputs to any downstream manufacturer's technical file
- MIME does not itself need to satisfy EU MDR — only the commercial Layer 4 product does

---

## 12. Configuration Management

### Inheritance Statement

**Taken verbatim in structure** from MADDENING Section 14.

### Configuration Artefacts

- Version control: Git (GitHub)
- Release tags: semantic versioning (`vX.Y.Z`)
- CI: GitHub Actions
- `CITATION.cff`: machine-readable citation metadata and version identification

### ID Namespacing

| Category | Prefix | Example |
|----------|--------|---------|
| Node algorithm IDs | `MIME-NODE-` | `MIME-NODE-001` |
| Anomaly IDs | `MIME-ANO-` | `MIME-ANO-001` |
| Verification benchmark IDs | `MIME-VER-` | `MIME-VER-001` |

### SOUP-of-SOUP Chain

```
[Commercial Product] assesses as SOUP:
  └── MICROBOTICA (MBOT-ANO-* anomalies)
      └── MIME (MIME-ANO-* anomalies)
          └── MADDENING (MADD-ANO-* anomalies)
              └── JAX, jaxlib, NumPy (transitive)
```

---

## 13. QMS Compatibility

### Inheritance Statement

**Taken verbatim** from MADDENING Section 15. The same model applies: MIME is an open-source research tool. The commercial entity building on the stack takes on the QMS (ISO 13485), the Notified Body relationship, post-market surveillance, and EU MDR manufacturer liability.

---

## 14. Registry Documentation

### What an Asset Author Must Provide

For a `MimeAssetSchema` submission to the MICROBOTICA registry, the asset author must provide — beyond the schema fields themselves:

#### Required Documentation

1. **Robot description**: human-readable description of the microrobot design (morphology, materials, actuation principle, intended application)
2. **Node graph specification**: the simulation graph as a USD file (see `asset_usd_path` in `MimeAssetSchema`) or a textual description of node composition, edge topology, and configuration parameters
3. **Parameter justification**: for key physical parameters, cite the source — measured, published, or assumed. Use `[@Key]` citations referencing `docs/bibliography.bib`.
4. **Benchmark execution environment**: hardware and software versions (captured in `BenchmarkResult` fields)
5. **Anatomical regime documentation**: for each `AnatomicalRegimeMeta` entry, cite evidence

#### Recommended Documentation

6. **Validation against experimental data**: comparisons between simulation predictions and experimental measurements
7. **Known limitations**: conditions where the simulation is inaccurate for this specific asset
8. **Control policy documentation**: how each compatible policy works with this asset

### Registry Quality Gate

| Tier | Requirements | Benefit |
|------|-------------|---------|
| **Listed** | `mime_compliant = True` | Appears in registry search |
| **Benchmarked** | At least B1 and B2 passed | Appears on leaderboard |
| **Validated** | All applicable B0–B5 passed + parameter justification | Featured asset |
| **Published** | Validated + associated publication (DOI) | Citable, peer-reviewed evidence |

These tiers are MICROBOTICA's concern (not MIME's), but MIME's schema and benchmark infrastructure enables them.

---

## Appendix A: Inheritance from MADDENING Summary Table

| MADDENING Section | MIME Treatment | Key Differences |
|---|---|---|
| 1. Documentation Structure | Adapted | Same Sphinx/MyST tooling; adds `algorithm_guide/control/` for PolicyRunner/UncertaintyModel |
| 2. Regulatory Boundary Language | Inherited with substitutions | MIME at Layer 2; scope is microrobotics physics, not general multiphysics |
| 3. Algorithm Documentation Standards | Extended | Adds `MimeNodeMeta`, anatomical regime tables, biocompatibility context, Mode 1/2 declarations |
| 4. V&V Documentation Hooks | Adapted | MIME's V&V is physics-domain-specific; B0–B5 multi-node system benchmarks (vs. per-node analytical) |
| 5. Versioning and API Stability | Inherited verbatim | Surface substitutions only |
| 7. Contributor Standards | Extended | Adds MIME-specific checklist items (MimeNodeMeta, GeometrySource, etc.) |
| 8. README | Inherited verbatim | Surface substitutions only |
| 9.1 NodeMeta Schema | Extended | Adds MimeNodeMeta second layer; preserves validated_regimes/hazard_hints scope distinction |
| 9.2 Provenance | Inherited | PolicyRunner adds control-specific provenance fields |
| 9.3 Verification Registration | Inherited | Uses `MIME-VER-` prefix |
| 9.5 Stability Machinery | Inherited | Uses `@stability` from `maddening.compliance` |
| 9.7 Anomaly Management | Inherited | Uses `MIME-ANO-` prefix; same three-phase lifecycle, three-tier gate |
| 9.8 ISO 14971 Hooks | Extended | Adds domain-specific hazard hints (biocompatibility, imaging artefacts) |
| 10. SOUP Package | Adapted | MIME-specific content; MADDENING listed as SOUP-of-SOUP dependency |
| 11. IEC 62304 Mapping | Adapted | References MIME's own artefacts |
| 12. EU MDR Alignment | Referenced | Same principles; not repeated |
| 15. QMS Compatibility | Inherited | Same model |
| 16. Downstream Inheritance | N/A | MIME *is* the downstream library that Section 16 describes |

---

## Appendix B: MIME New Node Checklist

### SimulationNode Contract (inherited from MADDENING)

- [ ] `SimulationNode` subclass with `initial_state()` and `update()`
- [ ] `update()` is JAX-traceable (jit, grad, vmap compatible)
- [ ] `boundary_input_spec()` overridden
- [ ] `compute_boundary_fluxes()` overridden if applicable
- [ ] `requires_halo` property implemented (default False for pointwise nodes)
- [ ] `@stability(StabilityLevel.EXPERIMENTAL)` applied
- [ ] `meta` ClassVar set with `NodeMeta` (MIME-NODE-* algorithm ID)
- [ ] NumPy-style docstring with Parameters, Boundary inputs
- [ ] Algorithm guide in `docs/algorithm_guide/nodes/` using MIME template
- [ ] Implementation Mapping traces every equation term to code
- [ ] `[@Key]` citations reference `docs/bibliography.bib`
- [ ] Assumptions and simplifications listed
- [ ] Validated physical regimes documented
- [ ] Known limitations and failure modes documented
- [ ] At least one `@verification_benchmark` (MIME-VER-*)
- [ ] Unit tests covering normal operation, edge cases, JAX-traceability
- [ ] Integration test within a `GraphManager`
- [ ] Known limitations in `docs/validation/known_anomalies.yaml` (MIME-ANO-*)
- [ ] All CI checks pass

### MIME-Specific Additions

- [ ] `mime_meta` ClassVar set with `MimeNodeMeta`
- [ ] Node role correctly assigned (external_apparatus, robot_body, environment, sensing, therapeutic)
- [ ] At least one `AnatomicalRegimeMeta` entry
- [ ] `BiocompatibilityMeta` present if role is `robot_body`
- [ ] `ActuationMeta` present if role is `external_apparatus`
- [ ] `SensingMeta` present if role is `sensing`
- [ ] `TherapeuticMeta` present if role is `therapeutic`
- [ ] `GeometrySource` dependency documented if spatial node
- [ ] `commandable_fields()` consistent with `boundary_input_spec()` keys
- [ ] `observable_fields()` consistent with `state_fields()`
- [ ] `validate_mime_consistency()` returns no errors
- [ ] Algorithm guide includes MIME-specific sections (Anatomical Operating Context, Clinical Relevance)
- [ ] Verification mode declared (Mode 1 or Mode 2)
- [ ] If Mode 1: Scope Statement with upstream version pin and regression test
- [ ] If Mode 2: full independent verification evidence
- [ ] Additive boundary inputs correctly marked (`coupling_type="additive"`)
- [ ] Differentiability status documented in node taxonomy
