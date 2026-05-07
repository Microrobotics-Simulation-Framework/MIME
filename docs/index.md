# MIME

**Microrobotics Interaction Model Engine** — the physics layer of the
Microrobotics Simulation Framework. MIME extends
[MADDENING](https://microrobotica.org/maddening/)
with the specific node families needed to simulate magnetically actuated
microrobots in confined biological flows: rigid-body chains, magnetic
response, low-Reynolds hydrodynamics (Stokeslet, IBM-FVM with optional
GNN correction), and the actuation/sensing chain that bridges to the
[MICROROBOTICA](https://microrobotica.org/) IDE.

```{toctree}
:caption: User Guide
:maxdepth: 2
:glob:

user_guide/*
```

```{toctree}
:caption: Algorithm Guide
:maxdepth: 2
:glob:

algorithm_guide/defect_correction
algorithm_guide/nodes/*
```

```{toctree}
:caption: Architecture
:maxdepth: 2
:glob:

architecture/*
```

```{toctree}
:caption: Deliverables
:maxdepth: 2
:glob:

deliverables/*
```

```{toctree}
:caption: Validation
:maxdepth: 2
:glob:

validation/cou_template
validation/soup_package
validation/b0_dataset
validation/benchmark_reports/*
validation/umr_deboer2025/*
validation/development_history/*
```

```{toctree}
:caption: Infrastructure
:maxdepth: 2
:glob:

infrastructure/*
```

```{toctree}
:caption: Regulatory
:maxdepth: 2

regulatory/intended_use
regulatory/iec62304_mapping
regulatory/downstream_integration
```

```{toctree}
:hidden:
:caption: Other

jax_pallas_issues
```
