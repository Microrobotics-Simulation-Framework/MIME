# IEC 62304 Software Lifecycle Documentation Mapping — MIME

## Scope

**MIME is not subject to IEC 62304.** It is an open-source research tool, not a medical device. This mapping is provided voluntarily to support downstream SOUP assessment.

## Mapping

| IEC 62304 Phase | Clause | What the Standard Requires | What MIME Provides | Gaps |
|---|---|---|---|---|
| **Software development planning** | 5.1 | Development plan, standards, tools | `ARCHITECTURE_PLAN.md`, `CONTRIBUTING.md`, Git + GitHub CI | No formal development plan in IEC 62304 format. Not required — MIME is not subject to IEC 62304. |
| **Software requirements analysis** | 5.2 | Documented software requirements | `ARCHITECTURE_PLAN.md` Sections 1–9, `MIME_NODE_TAXONOMY.md`, test suite | Requirements documented but not in formal specification with IDs and traceability. |
| **Software architectural design** | 5.3 | Architecture document, SOUP identification | `ARCHITECTURE_PLAN.md`, graph-based architecture, functional purity. MADDENING listed as SOUP dependency. | Architecture well-documented. |
| **Software detailed design** | 5.4 | Detailed design for each software unit | Per-node algorithm documentation in `docs/algorithm_guide/`, `NodeMeta` + `MimeNodeMeta`, Implementation Mapping tables | Coverage grows as algorithm guides are populated. |
| **Software unit implementation** | 5.5 | Implement per detailed design | Source code in `src/mime/`, NumPy-style docstrings | Formal coding standard in `CONTRIBUTING.md`. |
| **Software unit verification** | 5.6 | Verify each unit against design | `tests/` with B0–B5 verification benchmarks, registered via `@verification_benchmark` | Coverage grows as benchmarks are implemented. |
| **Software integration testing** | 5.7 | Integration testing | Integration tests within `GraphManager`, multi-node benchmark scenarios | Present via B1/B4/B5 system-level tests. |
| **Software system testing** | 5.8 | System-level testing | B4–B5 end-to-end benchmarks (full actuation chain + control + uncertainty) | Dependent on Neurobotika mesh. |
| **Software release** | 5.9 | Release documentation, known anomalies | `CHANGELOG.md`, tagged releases, `known_anomalies.yaml`, SOUP package | Well-covered once SOUP package is complete. |
