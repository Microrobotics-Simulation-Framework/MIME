# MIME

**MIcrorobotics Multiphysics Engine** — the physics layer of the
Microrobotics Simulation Framework. MIME extends
[MADDENING](https://microrobotica.org/maddening/)
with the specific node families needed to simulate magnetically actuated
microrobots in confined biological flows: rigid-body chains, magnetic
response, {term}`low-Reynolds <Low-Re>` hydrodynamics (Stokeslet,
{term}`IBM-FVM` with optional {term}`GNN correction`), and the
actuation/sensing chain that bridges to the
[MICROROBOTICA](https://microrobotica.org/) IDE.

::::{grid} 1 2 2 2
:gutter: 3
:margin: 4 4 0 0

:::{grid-item-card} Magnetic actuation & low-Re hydrodynamics
:class-card: msf-card

{term}`Dipole response`, {term}`Stokeslet` flow, IBM-FVM with optional
GNN closure, and a {term}`Lubrication correction` for near-wall confined
flow — all as composable {term}`Node`s.
:::

:::{grid-item-card} Validation suite (B0–B5)
:class-card: msf-card

Six standardised benchmarks with shipped reference data: digitised
de Jongh 2025 trajectories, dipole-field verification, robot-arm mass
matrix, free-fall, PD tracking, and confined-flow corrections.
:::

:::{grid-item-card} Regulatory metadata baked in
:class-card: msf-card

Every `MimeAssetSchema` carries ISO 10993 biocompatibility, {term}`SOUP`
classification, {term}`ISO 14971` hazard hints, and an
{term}`Anatomical regime` guard the loader enforces before simulation starts.
:::

:::{grid-item-card} Control primitives & policies
:class-card: msf-card

`ControlPolicy`, `ControlPrimitive`, `ControlSequence` and the
`PolicyRunner` orchestrate closed-loop strategies on top of the same
node graph — same autodiff, no second runtime.
:::

:::{grid-item-card} Composable environment effects
:class-card: msf-card

The [EffectModel contract](architecture/effect_model_contract.md) (v0.2
pilot) makes force/torque effects swappable and composable: the
`HydrodynamicModel` family (LBM / FVM / Stokeslet / DefectCorrection)
behind one builder + validation surface. Magnetic family in v0.3.
:::

::::

## How MIME is wired

MIME extends MADDENING with the **physics**, **control**, **uncertainty**,
and **regulatory metadata** specific to magnetically actuated microrobots
in confined biological flow. The autodiff and {term}`coupling <Coupling>`
all stay in MADDENING; everything below is what MIME adds on top.

```{mermaid-tips}
nodes:
  MAD:    MADDENING graph runtime. Provides SimulationNode, EdgeSpec, the topological scheduler, Gauss-Seidel coupling, JIT, vmap, autodiff. MIME treats it as a black box.
  Act:    Actuation nodes — MotorNode, PermanentMagnetNode, ExternalMagneticFieldNode. Turn commands (frequency, current) into field/force inputs.
  Robot:  Body nodes — RigidBodyNode, RobotArmNode. Carry mass/inertia/joint state and consume forces/torques from the rest of the graph.
  Env:    Environment / hydrodynamics nodes — regularised Stokeslet, IBM-FVM with optional GNN correction, lubrication near-wall correction.
  Sense:  Sensing nodes — magnetometer, barometer, range-of-motion checks. Inject realistic measurement uncertainty.
  Ther:   Therapeutic nodes — payload release kinetics, drug-elution models. Each carries its own pharmacology metadata.
  Enums:  Domain enums — AnatomicalCompartment, ActuationPrinciple, ReleaseKinetics, FlowRegime — keyed everywhere asset metadata is stored.
  Schema: MimeAssetSchema — the asset's filing cabinet. Holds the node graph, B0-B5 benchmark results, biocompatibility (ISO 10993), SOUP class, hazard hints (ISO 14971), and the anatomical regime envelope.
  Bench:  B0-B5 benchmark runners. Reproducible scoring pipelines for every asset; results are persisted into the asset's schema entry.
  Policy: ControlPolicy ABC — the place a closed-loop strategy lives. Operates on observations from sensing nodes.
  Prim:   ControlPrimitive ABC — atomic, named control actions (e.g. "advance helix 50 µm at 5 Hz"). Reused across policies for verification.
  Run:    PolicyRunner — orchestrates the MADDENING graph, the active ControlPolicy, and the UncertaintyModel for a single simulation episode.
  UQ:     UncertaintyModel ABC + concrete impls — input, epistemic, aleatoric. Differentiability flags travel alongside so callers know what they can backprop through.
  MR:     MICROROBOTICA IDE — loads a MimeAssetSchema, replays the recorded trajectory, and presents the audit metadata in its registry.
edges:
  MAD->Act:    All MIME nodes (actuation, robot, environment, sensing, therapeutic) subclass MADDENING's SimulationNode — same step contract.
  MAD->Robot:  Identical inheritance to actuation — RigidBody / RobotArm are pure MADDENING graph nodes.
  MAD->Env:    Stokeslet / IBM-FVM / lubrication are also MADDENING nodes. The Gauss-Seidel coupling in MADDENING handles the body↔fluid feedback.
  MAD->Sense:  Sensing nodes likewise subclass SimulationNode; their outputs become observations for the policy.
  MAD->Ther:   Therapeutic effect nodes are MADDENING nodes that take physical state and produce dosimetry / release timecourses.
  Enums->Schema: Domain enums make every asset entry comparable across the registry — same vocabulary, same units.
  Bench->Schema: Each benchmark run writes its score and provenance into the asset's schema entry.
  Act->Run:    The runner ticks the graph forward, observing actuation outputs each step.
  Robot->Run:  Body state feeds the policy and the uncertainty model.
  Env->Run:    Same — fluid forces become part of the runner's observation tuple.
  Sense->Run:  Sensing outputs go through the UncertaintyModel before reaching the policy.
  Policy->Run: The active policy receives observations and emits commands.
  Prim->Policy: Primitives are the policy's building blocks; reused across policies makes verification reusable too.
  UQ->Run:     Per-node and per-asset uncertainty models layer on top of the deterministic graph.
  Run->Schema: The runner finalises each episode by writing trajectory + scores back into the schema.
  Schema->MR:  The IDE indexes assets by their MimeAssetSchema entries and lets the user replay any trajectory in 3D.
```

```{mermaid}
flowchart LR
    MAD["MADDENING<br/><i>graph runtime &middot; coupling<br/>autodiff &middot; surrogates</i>"]:::ext

    subgraph DOM["domain nodes (SimulationNode subclasses)"]
      direction TB
      Act["actuation<br/><i>MotorNode &middot; PermanentMagnet<br/>ExternalMagneticField</i>"]
      Robot["robot<br/><i>RigidBody &middot; RobotArm</i>"]
      Env["environment<br/><i>Stokeslet &middot; IBM-FVM<br/>Lubrication &middot; GNN</i>"]
      Sense["sensing<br/><i>magnetometer<br/>range-of-motion</i>"]
      Ther["therapeutic<br/><i>release kinetics<br/>dosimetry</i>"]
    end

    subgraph META["regulatory metadata (MIME-only)"]
      direction TB
      Schema["MimeAssetSchema<br/><i>biocompat &middot; SOUP &middot; regime</i>"]
      Enums["domain enums<br/><i>compartment, regime,<br/>actuation, release</i>"]
      Bench["B0-B5 benchmarks"]
    end

    subgraph CTRL["control"]
      direction TB
      Policy["ControlPolicy ABC"]
      Prim["ControlPrimitive ABC"]
      Run["PolicyRunner"]
    end

    UQ["UncertaintyModel<br/><i>input &middot; epistemic &middot; aleatoric</i>"]

    MR["MICROROBOTICA<br/><i>loads asset, renders trajectory</i>"]:::ext

    MAD --> Act
    MAD --> Robot
    MAD --> Env
    MAD --> Sense
    MAD --> Ther

    Act --> Run
    Robot --> Run
    Env --> Run
    Sense --> Run
    Ther --> Run

    Policy --> Run
    Prim --> Policy
    UQ --> Run

    Enums --> Schema
    Bench --> Schema
    Run --> Schema

    Schema --> MR

    classDef ext stroke-dasharray:5 3
```

```{toctree}
:caption: User Guide
:maxdepth: 2
:glob:

user_guide/*
experiment_recordings
preempt_resume
glossary
```

```{toctree}
:caption: Release Notes
:maxdepth: 1

release_notes/index
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
