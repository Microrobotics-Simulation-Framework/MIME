# MIME Architecture Plan v0.6
*MIcrorobotics Multiphysics Engine — Layer 2 of the MICROBOTICA stack*

---

## 1. Position in the stack

```
MADDENING (Layer 1)   General HPC differentiable multiphysics. JAX/XLA.
      ^ extends
MIME (Layer 2)        Microrobotics domain layer. Node classes, asset schema,
                      control primitives, uncertainty models, benchmarks.
      ^ consumed by
MICROBOTICA (Layer 3) Full robotics simulator + community registry/leaderboard.
      ^ consumed by
[Commercial Product]  CE-marked medical device software (Layer 4).
```

**Analogy:** MADDENING is to MIME as PyTorch is to `transformers`: MADDENING provides the general compute graph and time-stepping infrastructure; MIME defines what it means to *simulate a microrobot* — the physics contracts, domain metadata, asset schema, and evaluation benchmarks.

**MICROBOTICA plays two simultaneous roles:**

1. **Full robotics simulator** (the Isaac Sim of medical microrobotics): USD-based scene authoring, real-time 3D visualisation, cloud streaming, hardware-in-the-loop interfaces. This is what makes MIME assets *runnable* in practice — a researcher interacts with MIME simulations through MICROBOTICA's desktop environment.

2. **Community registry and leaderboard platform** (the HuggingFace Hub analogue): researchers publish MIME-compliant robot assets, benchmark results are attached to assets via B1–B5, and the registry provides ranked comparison across assets on standardised tasks.

These two roles are inseparable. A MIME asset that is not runnable in MICROBOTICA cannot be benchmarked; an asset that is not benchmarked cannot appear on the leaderboard. MICROBOTICA is the environment that *executes* benchmarks reproducibly **and** the platform that *publishes and ranks* their results. This dual role is what makes the B1–B5 benchmark suite scientifically meaningful — benchmarks are not just unit tests but reproducible evaluations that can be compared across the community.

`MimeAssetSchema` is the model card that bridges both roles: it describes the robot asset for the simulator (node composition, control compatibility) and for the registry (benchmark results, provenance, regulatory metadata).

The closest single analogy is: MADDENING is PyTorch, MIME is `transformers`, and MICROBOTICA is HuggingFace Hub *plus* Isaac Sim combined. The PyTorch/`transformers`/Hub analogy holds for the data and registry side, but MICROBOTICA extends beyond HuggingFace because it is also a full C++17/Qt 6 desktop simulator with USD scene management, real-time 3D viewport, async physics integration, and embedded Python scripting — capabilities analogous to NVIDIA's Isaac Sim, not to a model registry.

Two key differentiators set this ecosystem apart from existing robotics platforms (Open X-Embodiment, LeRobot, Isaac Lab):

1. **Regulatory evidence chain in the asset format.** `MimeAssetSchema` carries the full scientific and regulatory evidence chain with the asset — not just model weights or trajectory data, but biocompatibility metadata, IEC 62304 SOUP classification, ISO 14971 hazard hints, anatomical regime validation, and benchmark results. This is what makes the registry meaningful for medical device development, not just academic benchmarking.

2. **Differentiable simulation programs, not static datasets.** Because MIME is built on MADDENING's differentiable physics graph, assets in the registry are not passive trajectory datasets — they are executable, differentiable simulation programs. This enables gradient-based controller training, sensitivity analysis, and uncertainty propagation in ways that a trajectory dataset (LeRobot's model) or a static FEM simulation (COMSOL) cannot support. Every node's `update()` function is JAX-traceable, making the simulation differentiable where the underlying physics permits. Nodes modelling well-behaved continuous physics (magnetic response, rigid body dynamics in the Stokes regime, diffusion, drug release kinetics) are fully differentiable. Nodes modelling discontinuous or stiff physics (contact mechanics with penalty forces, Rayleigh-Plesset bubble dynamics near collapse) are explicitly flagged as "differentiability-limited" in the node taxonomy — gradients through these nodes are unreliable and should not be used for training. The differentiable subgraph (excluding differentiability-limited nodes) supports gradient-based controller training, sensitivity analysis, and uncertainty propagation.

---

## 2. Module structure

```
src/mime/
├── core/
│   ├── metadata.py      # Domain meta dataclasses (no JAX dependency)
│   ├── node.py          # MimeNode ABC — extends MADDENING SimulationNode
│   ├── geometry.py      # GeometrySource protocol — consumed by spatial nodes
│   └── viewport.py      # USDViewport protocol — swappable rendering backend
├── nodes/
│   ├── actuation/       # ExternalMagneticFieldNode, MagneticResponseNode, ...
│   ├── robot/           # RigidBodyNode, FlexibleBodyNode, SurfaceContactNode, ...
│   ├── environment/     # CSFFlowNode, TissueDeformationNode, ...
│   ├── sensing/         # MRISignalNode, AcousticPressureNode, ...          [Phase 2]
│   └── therapeutic/     # DrugReleaseNode, ConcentrationDiffusionNode, ...  [Phase 2]
├── control/
│   ├── policy.py        # ControlPolicy ABC, ControlPrimitive ABC, ControlSequence
│   └── runner.py        # PolicyRunner — orchestrates Graph + Policy + Uncertainty
├── uncertainty/
│   └── base.py          # UncertaintyModel ABC + concrete implementations
├── schema/
│   └── asset.py         # MimeAssetSchema, BenchmarkResult
└── benchmarks/
    └── suite.py         # B1–B5 definitions
```

### Changes from v0.1

- **`nodes/robot/`** replaces `nodes/actuation/` for the robot body nodes. `actuation/` now contains only external apparatus nodes. This reflects the physical decomposition: actuation hardware is *outside* the body; robot body physics is the robot itself.
- **`nodes/` organisation by physical domain**, not by node role. Each subdirectory corresponds to a physical subsystem (external actuation hardware, robot body, fluid environment, imaging/sensing, therapeutic payload). This maps cleanly to the node taxonomy document and to how a researcher thinks about the physics.

---

## 3. Core metadata (`mime/core/metadata.py`)

Pure Python dataclasses, no JAX dependency. All importable without installing the simulation stack. This mirrors MADDENING's `maddening.core.compliance.metadata` pattern.

### Key enumerations

- `AnatomicalCompartment` — CSF, blood, interstitial, tissue, ...
- `FlowRegime` — pulsatile_csf, Poiseuille, stagnant, oscillatory, ...
- `ActuationPrinciple` — rotating_magnetic_field, gradient_magnetic_field, acoustic_streaming, ...
- `ImagingModality` — MRI, fluorescence, ultrasound, photoacoustic, ...
- `ReleaseKinetics` — pH_triggered, magnetic, enzymatic, passive_diffusion, ...
- `BiocompatibilityClass` — ISO 10993 level
- `RegulatoryClass` — EU MDR Class I–III, FDA Class I–III
- `SOUPClassification` — IEC 62304 Class A/B/C
- `NodeRole` — external_apparatus, robot_body, environment, sensing, therapeutic

### Meta dataclasses

**`AnatomicalRegimeMeta`** — physiological operating context. Extends (does not replace) MADDENING's `ValidatedRegime` with anatomically grounded bounds. A `MimeNode` carries both:
- `ValidatedRegime` entries in `NodeMeta` for parameter-bound numerical validity (e.g., "CFL < 0.5")
- `AnatomicalRegimeMeta` entries in `MimeNodeMeta` for physiological operating context (e.g., "CSF in lateral ventricles, Re 0.001–0.1")

Fields: compartment, anatomy string, flow regime, Re range, pH range, temperature range, viscosity range.

**`BiocompatibilityMeta`** — materials list, ISO 10993 level, test flags (cytotoxicity, haemocompatibility, genotoxicity, implantation), `biocompatibility_hazard_hints`. Technical descriptor only — not a safety claim.

**Display constraint for MICROBOTICA**: `BiocompatibilityMeta` fields must never be rendered with pass/fail iconography, colour coding, or compliance badge language that implies regulatory approval. The field is for search and comparison only. This constraint must be documented in MICROBOTICA's UI specification.

**`ActuationMeta`** — actuation principle, `is_onboard` flag (see §5), force/torque/field specs, step-out characterisation, `commandable_fields` (names of boundary inputs a ControlPolicy may set).

**`SensingMeta`** — modality, spatial/temporal resolution, SNR, `position_noise_std_mm`, `dropout_probability`, `imaging_artifact_hints`.

**`TherapeuticMeta`** — payload type/name, release kinetics, `target_anatomy`, `target_pathway` (e.g. `"prion_like_spread_inhibition"`), payload capacity, release half-life, therapeutic window ratio.

**`MimeNodeMeta`** — top-level container. Composes all of the above via composition (not inheritance). Fields:
- `role: NodeRole` — which physical subsystem this node belongs to
- `anatomical_regimes: tuple[AnatomicalRegimeMeta, ...]`
- `biocompatibility: Optional[BiocompatibilityMeta]`
- `actuation: Optional[ActuationMeta]`
- `sensing: Optional[SensingMeta]`
- `therapeutic: Optional[TherapeuticMeta]`
- Domain-specific hazard hints (e.g., `biocompatibility_hazard_hints`, `imaging_artifact_hints`)

---

## 4. MimeNode ABC (`mime/core/node.py`)

Extends `SimulationNode` with MIME domain concerns.

### Dual-metadata pattern and harvester compatibility

MADDENING's metadata harvester (`collect_node_metadata()` in `maddening.core.compliance.metadata`) discovers metadata by reading `cls.meta` from each `SimulationNode.__subclasses__()` entry. This means `MimeNode` **must** set `meta` as a ClassVar pointing to a `NodeMeta` instance for the harvester to work.

The design is:

```python
from maddening.core.node import SimulationNode
from maddening.core.compliance.metadata import NodeMeta

class MimeNode(SimulationNode):
    # MADDENING-level metadata — consumed by MADDENING's harvester,
    # compliance tooling, and Sphinx documentation build.
    meta: ClassVar[Optional[NodeMeta]] = None

    # MIME-level metadata — consumed by MIME's own tooling,
    # MimeAssetSchema, and MICROBOTICA registry.
    mime_meta: ClassVar[Optional[MimeNodeMeta]] = None
```

`meta` is **not** aliased from a `maddening_meta` attribute. It is simply the standard MADDENING `meta` ClassVar, set directly on each `MimeNode` subclass. The v0.1 plan's `maddening_meta` alias approach was incorrect — it would have broken the harvester because `collect_node_metadata()` looks for `cls.meta`, not `cls.maddening_meta`.

Compliance tooling can consume either layer independently:
- `collect_node_metadata()` returns `NodeMeta` for all `SimulationNode` subclasses (including `MimeNode` subclasses) — this is the MADDENING-level view
- A MIME-level harvester (`collect_mime_metadata()`) returns `MimeNodeMeta` for all `MimeNode` subclasses — this is the domain-level view

### Abstract interface

```python
class MimeNode(SimulationNode):
    meta: ClassVar[Optional[NodeMeta]] = None
    mime_meta: ClassVar[Optional[MimeNodeMeta]] = None

    @property
    def requires_halo(self) -> bool:
        """Whether this node's update() accesses spatial neighbors.

        Most MIME nodes are pointwise (rigid body, ODE-based actuation)
        and return False. Spatially-resolved nodes (CSF flow, diffusion)
        must return True if they use stencil operations.

        This is required by MADDENING's SimulationNode ABC.
        """
        return False  # Default for pointwise MIME nodes; override for spatial nodes

    def observable_fields(self) -> list[str]:
        """Which state fields are visible to a ControlPolicy (via UncertaintyModel).

        Default: all fields from initial_state(). Override to restrict
        visibility (e.g., a magnetic response node may expose orientation
        but not internal magnetisation state).
        """
        return self.state_fields()

    def commandable_fields(self) -> list[str]:
        """Which boundary inputs a ControlPolicy may set at runtime.

        Derived from ActuationMeta.commandable_fields if present.
        Must be a subset of boundary_input_spec() keys.
        """
        if self.mime_meta and self.mime_meta.actuation:
            return list(self.mime_meta.actuation.commandable_fields)
        return []

    def validate_mime_consistency(self) -> list[str]:
        """Check internal consistency of MIME metadata.

        Returns a list of error strings (empty = consistent).

        Checks:
        - commandable_fields subset of boundary_input_spec() keys
        - observable_fields subset of state_fields()
        - robot_body role requires biocompatibility metadata
        - sensing role requires sensing metadata
        - All MimeNode subclasses must have both meta and mime_meta set
        """
        ...
```

### Constructor contract (critical)

**Constructor params = static physical properties only.** Anything a ControlPolicy might command at runtime MUST be a boundary input declared in `boundary_input_spec()`, not a constructor param. This is enforced by `validate_mime_consistency()`.

### `requires_halo` property

MADDENING's `SimulationNode` ABC requires every subclass to implement `requires_halo`. Most MIME nodes are pointwise (rigid body dynamics, ODE-based actuation, drug release kinetics) and return `False`. Spatially-resolved nodes (CSF flow via LBM, concentration diffusion) must return `True` if their `update()` function uses stencil operations that require halo exchange when sharded across devices.

`MimeNode` provides a default of `False` to avoid boilerplate in the majority of MIME nodes. Spatial nodes override this.

### `derivatives()` and `implicit_residual()`

MADDENING's `SimulationNode` provides optional `derivatives()` and `implicit_residual()` methods for higher-order integration and implicit time-stepping. MIME nodes that model stiff ODEs (e.g., drug release kinetics with fast binding, acoustic bubble dynamics) should implement `implicit_residual()` to enable the graph manager's implicit Newton solver. MIME nodes with natural ODE form should implement `derivatives()` to enable RK4 and adaptive integration at the graph level.

These are not required by `MimeNode` but are recommended for physically motivated nodes where explicit Euler is insufficient.

### Node roles

| Role | Description |
|------|-------------|
| `external_apparatus` | Hardware outside the body: rotating magnet, Helmholtz coil, ultrasound transducer. `is_onboard=False`. Produces field outputs consumed by `robot_body` nodes via edges. |
| `robot_body` | The microrobot itself: rigid body kinematics, onboard magnet response, drug payload. Requires `BiocompatibilityMeta`. |
| `environment` | Physiological medium: CSF flow, tissue deformation, concentration diffusion. Bidirectionally coupled with `robot_body`. |
| `sensing` | Imaging/localisation physics: MRI signal model, acoustic pressure, fluorescence. Produces observables consumed by `UncertaintyModel`. |
| `therapeutic` | Drug delivery mechanics: release kinetics, pharmacokinetic transport. Coupled to `robot_body` (payload source) and `environment` (concentration sink). Requires `TherapeuticMeta`. |

---

## 5. Node class hierarchy

```
SimulationNode (MADDENING)
└── MimeNode (MIME base)
    ├── [external_apparatus]
    │   ├── ExternalMagneticFieldNode    # rotating magnet / Helmholtz coil
    │   ├── ExternalGradientFieldNode    # gradient steering            [Phase 2]
    │   ├── ExternalAcousticNode         # ultrasound transducer        [Phase 2]
    │   └── ExternalOpticalNode          # laser trap / photoacoustic   [Advanced]
    │
    ├── [robot_body]
    │   ├── RigidBodyNode                # 6-DOF kinematics             [Phase 1]
    │   ├── MagneticResponseNode         # onboard magnet -> torque/force
    │   ├── FlexibleBodyNode             # flagellar / compliant body   [Phase 2]
    │   ├── MicroBubbleNode              # acoustic bubble response     [Phase 2]
    │   ├── SurfaceContactNode           # wall effects, vessel contact [Phase 2]
    │   └── PhaseTrackingNode            # orientation phase observer   [Phase 1]
    │
    ├── [environment]
    │   ├── CSFFlowNode                  # pulsatile Stokes/LBM         [Phase 1]
    │   ├── TissueDeformationNode        # contact mechanics            [Phase 2]
    │   └── ConcentrationDiffusionNode   # drug spreading in CSF        [Phase 2]
    │
    ├── [sensing]
    │   ├── MRISignalNode                # k-space / contrast model     [Phase 2]
    │   ├── AcousticPressureNode         # ultrasound field             [Phase 2]
    │   └── FluorescenceNode             # optical signal model         [Phase 2]
    │
    └── [therapeutic]
        ├── DrugReleaseNode              # payload concentration + kinetics [Phase 2]
        └── PharmacokineticsNode         # uptake, clearance, target conc  [Phase 2]
```

### Standard edge topology (magnetic actuation)

```
ExternalMagneticFieldNode --[field_vector]----------------------> MagneticResponseNode
MagneticResponseNode      --[magnetic_torque, magnetic_force]--> RigidBodyNode
RigidBodyNode             --[position, velocity]---------------> CSFFlowNode
CSFFlowNode               --[drag_force, drag_torque]----------> RigidBodyNode
RigidBodyNode             --[orientation]----------------------> PhaseTrackingNode
```

Note: `drag_force` on `RigidBodyNode` should use `coupling_type="additive"` in `boundary_input_spec()` since multiple sources (CSF drag, contact forces, gravity) may contribute to the total force. This uses MADDENING's `BoundaryInputSpec(coupling_type="additive")` pattern.

### Why external and onboard are separate nodes

- Simulate the external apparatus independently (field calibration, sweep characterisation)
- Model multiple robots under the same field without duplication
- Inject actuation uncertainty at the correct physical boundary (the field source, not the robot)
- `ExternalMagneticFieldNode` is what the ControlPolicy commands; `MagneticResponseNode` just responds to physics
- Enables different timesteps: external field changes slowly (ms), robot dynamics evolves fast (us)

### PhaseTrackingNode

`PhaseTrackingNode` is an observational node that reads orientation from `MagneticResponseNode` (via an edge) and computes the phase error between the robot's magnetic moment direction and the external field rotation. It is needed for:
- B1 (step-out frequency detection): phase error exceeding pi/2 indicates step-out
- B5 (step-out recovery): detecting recovery requires tracking phase convergence
- Feedback control: `StepOutDetector` policy reads phase error to decide when to drop frequency

It does not model physics — it is a signal processing node. `requires_halo = False`, no boundary fluxes.

---

## 6. Control layer (`mime/control/`)

### Design principles

1. Policies are **stateless classes**. All carry state lives in `policy_state: dict`, passed in and returned each call. JAX-friendly.
2. Constructor params = configuration (gains, target node names). Nothing mutable on `self`.
3. A policy **never touches GraphManager**. It only sees `(t, observed_state, policy_state)` and returns `(external_inputs, new_policy_state)`.
4. Policies operate on `observed_state` (post-uncertainty), never on true simulation state.

### `ControlPolicy` ABC

```python
class ControlPolicy(ABC):
    def __call__(self, t, observed_state, policy_state) -> tuple[dict, dict]:
        """Compute external inputs given observed state.

        Parameters
        ----------
        t : float
            Current simulation time.
        observed_state : dict
            State as seen through the UncertaintyModel (may be noisy/incomplete).
        policy_state : dict
            Mutable policy state (carried across calls).

        Returns
        -------
        external_inputs : dict
            Mapping of {node_name: {field_name: value}} for ExternalInputSpec targets.
        new_policy_state : dict
            Updated policy state for next call.
        """
        ...

    def initial_policy_state(self) -> dict: ...
    def __or__(self, other) -> SequentialPolicy: ...   # policy_a | policy_b
```

### `ControlPrimitive` ABC (scripting atom)

```python
@dataclass
class ControlPrimitive(ABC):
    duration: float
    target_node: str
    def external_inputs(self, t_local, dt, observed_state) -> dict: ...
    def on_start(self, observed_state): ...   # hook
    def on_end(self, observed_state): ...     # hook
```

### Standard primitive library

| Primitive | Description |
|-----------|-------------|
| `RotateField(frequency, field_strength, duration)` | Fixed-frequency rotation |
| `SweepFrequency(f_start, f_end, duration)` | Linear frequency ramp — step-out characterisation |
| `HoldField(frequency, duration)` | Hold at fixed frequency |
| `RampField(frequency, target_strength, duration)` | Ramp field strength from zero — gentle startup |
| `StepDown(frequency, duration)` | Drop frequency instantly — step-out recovery |

### `ControlSequence`

Composes primitives into a time-ordered script with phase tracking. Supports `loop=True`. User-facing scripting surface:

```python
sequence = ControlSequence([
    RampField(frequency=10.0, duration=1.0),
    SweepFrequency(f_start=10.0, f_end=50.0, duration=5.0),
    HoldField(frequency=30.0, duration=2.0),
])
```

### Feedback policies (subclass `ControlPolicy` directly)

Power users subclass `ControlPolicy` for feedback control (e.g. `StepOutDetector` reads `phase_error` from `PhaseTrackingNode` and drops frequency on step-out detection). Control logic stays outside GraphManager — no MADDENING modifications required.

### State estimation

The current design injects observation noise via `UncertaintyModel.observe()`. For advanced closed-loop control, a state estimator (e.g., Extended Kalman Filter for robot pose) may sit between the uncertainty model and the policy:

```
true_state -> UncertaintyModel.observe() -> noisy_observation -> StateEstimator -> estimated_state -> ControlPolicy
```

The `StateEstimator` is an optional component, not part of the core control layer in Phase 0–1. It will be needed when imaging physics nodes (Phase 2) produce realistic measurement data rather than noisy ground truth. Design decision deferred to Phase 2 when the sensing pipeline exists.

### Differentiable control (future)

For differentiable controller training, `PolicyRunner` will eventually need a mode that compiles the entire policy+graph loop into a single `jax.lax.scan` call. This requires:
- `ControlPolicy.__call__` to be JAX-traceable (pure function, `jnp.where` for conditionals)
- `UncertaintyModel` channels to be JAX-traceable
- The loop structure to use `lax.scan` with carry = (graph_state, policy_state, rng_key)

This is deferred until a concrete training use case exists. The architecture is compatible — the key design decisions (stateless policies, dict-based state, functional purity) are already in place.

---

## 7. Uncertainty layer (`mime/uncertainty/`)

### Architecture

The `UncertaintyModel` sits at the boundary between true simulation state and the controller. It has two channels:

- `observe(true_state, t, rng_key) -> observed_state` — sensing uncertainty
- `actuate(commanded_inputs, t, rng_key) -> applied_inputs` — actuation uncertainty

### `PolicyRunner` step loop

```
1. observed_state  = uncertainty.observe(true_state)    # sensing noise
2. commanded_ext   = policy(t, observed_state)          # controller decides
3. applied_ext     = uncertainty.actuate(commanded_ext) # actuation noise
4. true_state      = graph.step(applied_ext)            # physics advances
```

The controller is genuinely flying partially blind. The true state evolves from `applied_ext`. The controller only ever sees `observed_state`.

### Provided implementations

**`IdentityUncertainty`** — perfect sensing and actuation. Baseline.

**`ActuationUncertainty`** — motor encoder jitter (`frequency_noise_std`), field inhomogeneity (`field_strength_noise_std`), robotic arm positioning error (`pointing_error_std`), slow thermal drift (`frequency_drift_rate`).

**`LocalisationUncertainty`** — Gaussian position noise (`position_noise_std_mm`), velocity noise, tracking dropouts (`dropout_probability`). Injects `tracking_confidence` field into observed state.

**`ModelUncertainty`** — fractional noise on observed state fields to mimic model parameter error (patient-to-patient variability, fabrication tolerances). For full model UQ use MADDENING's `GraphManager.run_sweep()` ensemble via `jax.vmap`.

**`ComposedUncertainty`** — stacks multiple models. Sugar: `model_a + model_b`.

### Uncertainty sources taxonomy

| Source | Model | Layer |
|--------|-------|-------|
| Motor encoder jitter | `ActuationUncertainty` | Control |
| Field inhomogeneity | `ActuationUncertainty` | Control |
| MRI resolution | `LocalisationUncertainty` | Control |
| Tracking dropout | `LocalisationUncertainty` | Control |
| CSF viscosity variability | `ModelUncertainty` / ensemble | Simulation |
| Robot fabrication tolerances | `ModelUncertainty` / ensemble | Simulation |
| Patient anatomy variability | Ensemble over mesh variants | Simulation |
| Thermal drift in apparatus | `ActuationUncertainty` | Control |

---

## 8. Asset schema (`mime/schema/asset.py`)

`MimeAssetSchema` is the registry model card. It serves dual roles mirroring MICROBOTICA's dual nature:

1. **Simulator artifact**: describes everything MICROBOTICA needs to instantiate and run the robot — node composition, graph topology, control compatibility, physical parameters.
2. **Registry artifact**: describes everything needed for community comparison — benchmark results, provenance, regulatory metadata, validated operating regimes.

Stored, indexed, and served by the MICROBOTICA registry. The `mime_compliant` property and `compliance_report()` method form the quality gate for registry publication.

### Key fields

- **Identity**: `asset_id`, `asset_version`, `mime_schema_version`, `maddening_version_pin`, `robot_morphology`, `characteristic_length_um`
- **Functional composition**: `onboard_node_classes`, `external_apparatus_node_classes`, `environment_node_classes`, `sensing_node_classes`, `therapeutic_node_classes`, `compatible_control_policies`, `asset_usd_path`
- **Domain metadata**: `biocompatibility`, `actuation`, `sensing`, `therapeutic`
- **Validated context**: `anatomical_regimes`
- **Regulatory**: `regulatory_class`, `soup_classification`
- **Benchmarks**: `benchmark_results` (B1–B5) — each result includes the MICROBOTICA version used to execute the benchmark, ensuring reproducibility
- **Provenance**: `authors`, `orcid_ids`, `zenodo_doi`, `license`
- **Verification mode**: per-node `verification_mode` (Mode 1 Wrapping / Mode 2 Independent) following MADDENING's Section 16 convention

### Compliance gate (`mime_compliant = True` requires)

- `asset_id` present
- `robot_morphology` present
- `characteristic_length_um > 0`
- At least one `onboard_node_classes` entry
- At least one `anatomical_regimes` entry
- `biocompatibility` not None — note: the presence of `BiocompatibilityMeta` in the compliance gate confirms that biocompatibility *has been described*, not that the material *has been assessed as safe*. MICROBOTICA must not present this gate as a safety certification.
- `maddening_version_pin` present

### Registry query examples (enabled by schema structure)

- "All helical robots validated in CSF"
- "All assets compatible with `ExternalMagneticFieldNode`"
- "All therapeutic assets targeting `lateral_ventricles`"
- "All assets that passed B1 and B4"
- "All assets ranked by B4 success rate" (leaderboard query)
- "All assets benchmarked on MICROBOTICA >= v0.3" (reproducibility filter)

### Benchmark result attachment

Each `BenchmarkResult` in `benchmark_results` carries:
- `benchmark_id` (B1–B5)
- `passed: bool`
- `metric_value: float` (the actual measured value, e.g., step-out frequency error %)
- `metric_threshold: float` (the pass criterion)
- `n_ensemble: int` (for B4/B5 ensemble benchmarks)
- `microbotica_version: str` (the MICROBOTICA version used to execute)
- `maddening_version: str`
- `mime_version: str`
- `execution_timestamp: str` (ISO 8601)
- `hardware_description: str` (GPU model, etc.)

This allows the registry to track whether benchmark results were produced under comparable conditions and to flag results that may need re-running after version updates.

### USD as the canonical asset format

`MimeAssetSchema` as a Python dataclass is the in-memory representation consumed by compliance tooling, benchmarks, and the registry API. On disk, the canonical format is USD (Universal Scene Description). This is the right choice for three reasons:

1. **MICROBOTICA alignment**: MICROBOTICA already uses USD as its scene format with a three-layer composition stack. A MIME asset is a USD file that MICROBOTICA can open directly without a conversion step.
2. **Self-describing portability**: USD's schema extension mechanism (custom `UsdSchemaBase` subclasses) allows `MimeNodeMeta`, `BiocompatibilityMeta`, `ActuationMeta`, and other MIME metadata to be encoded as structured USD attributes on prim types. The asset is self-describing — any USD-aware tool can inspect it without MIME installed.
3. **Simulation graph topology**: the node graph (which nodes, which edges, which parameters) is encoded as USD prims and relationships, not referenced by an opaque URI. The asset is a complete, inspectable simulation program, not a pointer to one.

The relationship between the two representations is:
- **On disk**: `.usd` file containing the simulation graph as USD prims, MIME metadata as USD attributes on those prims, and geometry references as USD mesh prims or external USD references
- **In memory**: `MimeAssetSchema` Python dataclass, populated by deserialising the USD file
- **Interface**: a `MimeAssetSchema.from_usd(path)` classmethod and a `MimeAssetSchema.to_usd(path)` method form the serialisation layer

The `MimeAssetSchema.asset_usd_path: str` field points to the root USD file for this asset. This replaces the previous `node_graph_uri` field — the node graph is now embedded in the USD file itself rather than referenced by an opaque URI.

USD schema authoring for MIME node types is a Phase 4 deliverable (MICROBOTICA integration phase). For Phase 0–3, the Python dataclass remains the primary interface and JSON is an acceptable intermediate serialisation. The migration path to USD is: Phase 0–3 (Python dataclass + JSON) → Phase 4 (Python dataclass + USD, with the dataclass as the authoritative in-memory form).

---

## 9. Benchmarks (`mime/benchmarks/suite.py`)

Five validation benchmarks. An asset that passes all five has demonstrated physical correctness across the core challenge set. Results attach directly to `MimeAssetSchema.benchmark_results` and are published to the MICROBOTICA registry leaderboard.

The benchmark suite is scientifically meaningful because MICROBOTICA provides:
- **Reproducible execution**: benchmarks run in a standardised simulator environment with controlled versions
- **Published results**: results are attached to assets in the registry where the community can inspect and compare them
- **Ranked comparison**: the leaderboard ranks assets on standardised tasks, enabling meaningful comparison across research groups

| ID | Name | Pass criterion | Phase |
|----|------|---------------|-------|
| B0 | Experimental validation | Simulated helical robot trajectory in a straight cylindrical channel matches a published experimental dataset (position RMSE < 15% of channel diameter, velocity RMSE < 20% of mean velocity) | Phase 1 |
| B1 | Step-out frequency detection | Simulated step-out frequency within ±5% of a regularised Stokeslet reference solution computed for the specific robot geometry, validated against Rodenborn et al. (2013, PNAS) experimental data | Phase 1 |
| B2 | Stokes drag in CSF | Drag force < 5% relative error vs. Stokes law F=6piηrv at Re < 0.1 | Phase 1 |
| B3 | Drug release kinetics | L2 error < 10% vs. analytical diffusion equation solution | Phase 2 |
| B4-T1 | Closed-loop navigation — simple geometry | >= 80% of N=32 ensemble runs reach within 2mm of target in a straight cylindrical channel (D=2mm, L=50mm) | Phase 3 |
| B4-T2 | Closed-loop navigation — realistic anatomy | >= 80% of N=32 ensemble runs reach within 2mm of target in a Neurobotika-derived ventricular mesh | Phase 3 |
| B4-T3 | Closed-loop navigation — pathological anatomy | >= 70% of N=32 ensemble runs reach within 2mm of target in a pathological anatomy variant (stenosed aqueduct, hydrocephalus) | Advanced |
| B5 | Step-out recovery under actuation uncertainty | Recovery within 5s for >= 90% of N=32 ensemble runs | Phase 3 |

### Dependencies

- B0 depends on: `ExternalMagneticFieldNode`, `MagneticResponseNode`, `RigidBodyNode`, `CSFFlowNode`, and a published experimental dataset with sufficient parameter documentation (robot dimensions, magnetic moment, channel geometry, fluid properties). Dataset selection is a Phase 0 deliverable — see §11.
- B1 depends on: `ExternalMagneticFieldNode`, `MagneticResponseNode`, `RigidBodyNode`, `CSFFlowNode`, `PhaseTrackingNode`
- B1 also requires: a regularised Stokeslet reference solution (Cortez, Fauci & Medovikov 2005; Rodenborn et al. 2013) computed for the robot geometry used in the benchmark. The regularised Stokeslet method is the high-fidelity reference — it agrees with experimental data within a few percent for the parameter ranges relevant to microrobotics, whereas RFT fails qualitatively and quantitatively for geometries with helix radius comparable to wavelength. RFT may be used as a coarse sanity check but is not the acceptance criterion. The Rodenborn et al. simulation codes are publicly available on MATLAB File Exchange ("Helical Swimming Simulator") and serve as the reference implementation.
- B2 depends on: `RigidBodyNode`, `CSFFlowNode`

**Recommended discretisation for `CSFFlowNode`**: Immersed Boundary Lattice-Boltzmann Method (IB-LBM). For bench-top validation (B0, B1, B2) involving simple parametric geometries, the regularised Stokeslet method is the appropriate high-fidelity reference for computing the robot's resistance tensor. For full fluid-structure coupling in confined geometries (B4-T1 through B4-T3) — a moving helical robot inside a CSF channel with pulsatile flow and near-wall interactions — IB-LBM is the recommended approach. IB-LBM handles complex moving boundaries by spreading boundary forces onto a fixed Eulerian fluid grid via delta function interpolation, and has been validated for microswimmer dynamics in channel flow. The two methods are complementary: regularised Stokeslets for resistance tensor characterisation, IB-LBM for full navigation simulation.
- B3 depends on: `DrugReleaseNode`, `ConcentrationDiffusionNode`
- B4-T1 depends on: `PolicyRunner`, `LocalisationUncertainty`, parametric cylindrical channel geometry, full magnetic actuation chain
- B4-T2 depends on: B4-T1 passing, Neurobotika-derived ventricular mesh
- B4-T3 depends on: B4-T2 passing, pathological anatomy mesh variants (stenosed aqueduct, hydrocephalus geometry)
- B5 depends on: `PhaseTrackingNode`, `ActuationUncertainty`, `StepOutDetector` feedback policy

### B4 tiered difficulty and the leaderboard

The three-tier structure of B4 serves a specific purpose in the MICROBOTICA registry leaderboard: it allows assets to demonstrate progressive navigational capability without requiring the full Neurobotika mesh or pathological anatomy variants upfront. An asset that passes B4-T1 has demonstrated functional closed-loop navigation; B4-T2 adds clinical realism; B4-T3 adds robustness to pathological anatomy that is directly relevant to the patient populations where microrobotic intervention is most needed (hydrocephalus, aqueductal stenosis, normal pressure hydrocephalus).

The pass rate threshold is deliberately lower for B4-T3 (70% vs. 80%) because pathological anatomy is genuinely harder — the constricted aqueduct in stenosis represents the most demanding navigation scenario in the benchmark suite. Lowering the threshold acknowledges the difficulty while still requiring meaningful robustness.

Registry leaderboard display: assets are ranked separately on B4-T1, B4-T2, and B4-T3. An asset that passes all three is badged "Clinically Robust Navigation." An asset that passes B4-T1 and B4-T2 but not B4-T3 is badged "Anatomically Validated Navigation." This tiering is MICROBOTICA's concern (not MIME's), but MIME's schema must support it — `BenchmarkResult.benchmark_id` should accept `"B4-T1"`, `"B4-T2"`, `"B4-T3"` as valid IDs in addition to `"B4"` for backward compatibility.

### Benchmark gaps and future extensions

The current B1–B5 suite covers the core mechanics. Notable gaps that may warrant future benchmarks:

- **Magnetic field accuracy (candidate B0)**: comparing computed external field to analytical solutions (e.g., Biot-Savart for a Helmholtz coil). Currently implicit in B1 but could be separated for independent validation of the field model.
- **Multi-physics coupling stability**: verifying that bidirectional fluid-structure coupling (robot drag <-> CSF flow) conserves energy/momentum. Important for long-duration simulations.
- **Imaging physics**: verifying MRI signal formation against analytical phantoms. Deferred to Phase 2+ when sensing nodes exist.
- **Acoustic actuation**: separate benchmark suite for non-magnetic actuation modalities. Deferred to Phase 2+.

**B0 dataset sourcing (Phase 0 action item).** Identifying and confirming the B0 experimental dataset is a Phase 0 deliverable — it must be done before Phase 1 implementation begins because the dataset constrains what `RigidBodyNode` and `CSFFlowNode` must be able to represent. B0 as a *passing* benchmark (i.e., running the simulation and comparing against the dataset) is a Phase 1 deliverable, not Phase 0. The dataset must provide: (a) helical robot physical parameters (body dimensions, magnetic moment, material), (b) channel geometry (diameter, length, wall material), (c) fluid properties (viscosity, density), (d) actuation parameters (field frequency, field strength), and (e) trajectory data (position and/or velocity vs. time). The primary candidate is **Rodenborn et al. (2013, PNAS)** — this paper provides experimental measurements of thrust, torque, and drag for helical swimmers across the parameter ranges relevant to microrobotics, alongside published simulation codes (regularised Stokeslet and slender body theory). It is also the paper that definitively establishes why RFT is insufficient as a B1 reference. Additional candidates: Fischer group (Max Planck Physical Intelligence), Nelson/Zhang group (ETH/UZH), Abbott group (University of Utah). The chosen paper should be confirmed and cited in `docs/bibliography.bib` before Phase 1 implementation begins.

---

## 10. Relationship to MADDENING

### API mapping

| MADDENING provides | MIME adds |
|--------------------|-----------|
| `SimulationNode` ABC (`initial_state`, `update`, `boundary_input_spec`, `compute_boundary_fluxes`, `requires_halo`, `state_fields`, `derivatives`, `implicit_residual`) | `MimeNode` ABC with roles, observable/commandable contracts, default `requires_halo=False` |
| `NodeMeta` (algorithm identity, IEC 62304, `hazard_hints`, `implementation_map`) | `MimeNodeMeta` (anatomy, biocompatibility, actuation, sensing, therapeutic) |
| `ValidatedRegime` (parameter bounds) | `AnatomicalRegimeMeta` (physiologically grounded bounds — extends, does not replace) |
| `GraphManager` + `ExternalInputSpec` | `PolicyRunner` (policy + uncertainty orchestration on top) |
| `EdgeSpec` (source/target/field/transform/additive) | Standard edge topologies for magnetic actuation, drug delivery, etc. |
| `BoundaryInputSpec` (shape, dtype, default, coupling_type) | MIME nodes declare domain-specific boundary inputs using this spec |
| `run_sweep()` vmap ensemble | `ModelUncertainty` + B4/B5 ensemble benchmarks |
| General `hazard_hints` in `NodeMeta` | Domain-specific `biocompatibility_hazard_hints`, `imaging_artifact_hints` in `MimeNodeMeta` |
| `@stability` decorator and `StabilityLevel` enum | Applied to all MIME public APIs with `MIME-` namespaced IDs |
| `@verification_benchmark` decorator | Applied to all MIME benchmarks with `MIME-VER-` namespaced IDs |
| `collect_node_metadata()` harvester | Works automatically on `MimeNode` subclasses (they set `meta` ClassVar) |
| `collect_hazard_hints()` harvester | Returns both MADDENING and MIME hazard hints when both are imported |
| `HealthCheckNode` base class (configured via `checks` dict: `{"field": {"finite": True, "min": ..., "max": ...}}`; receives monitored state via edges; writes pass/fail to own state) | MIME instantiates with domain-specific checks (CSF density bounds, velocity magnitude limits, concentration positivity) and connects via edges to monitored physics nodes |
| `maddening.compliance` namespace | MIME imports all compliance types from here; creates `MIME-` namespaced instances |

### What MIME does NOT duplicate from MADDENING

- `GraphManager`: MIME uses it directly, never subclasses or wraps it
- `EdgeSpec`: MIME creates edges using the standard MADDENING API
- Scheduling, multi-rate timestepping, coupling groups: all MADDENING infrastructure used as-is
- Compliance schemas (`NodeMeta`, `AnomalyRecord`, etc.): imported from `maddening.compliance`
- JIT compilation, `jax.vmap` sweeps, adaptive timestepping: all MADDENING

Dependency direction is strictly one-way: MIME imports MADDENING. MADDENING never imports MIME.

---

## 11. Phase roadmap

| Phase | Deliverables | Unblocked by |
|-------|-------------|--------------|
| 0 (now) | `core/metadata.py`, `MimeNode`, `ExternalMagneticFieldNode`, `MagneticResponseNode`, `ControlPolicy`/`ControlSequence`, `PolicyRunner`, `UncertaintyModel` suite, `MimeAssetSchema`, B0–B5 stubs, `GeometrySource` interface, `DOCUMENTATION_ARCHITECTURE.md`, `MIME_NODE_TAXONOMY.md` | Nothing |
| 1 | `RigidBodyNode`, `CSFFlowNode`, `PhaseTrackingNode`, B1 + B2 implemented | Phase 0 |
| 2 | `DrugReleaseNode`, `ConcentrationDiffusionNode`, `MRISignalNode`, `FlexibleBodyNode`, `SurfaceContactNode`, B3 implemented | Phase 1 |
| 3 | `StepOutDetector` feedback policy, B4 + B5 implemented (requires Neurobotika mesh), state estimator design | Phase 2 + Neurobotika |
| 4 | MICROBOTICA integration: USD scene format, MICROBOTICA desktop client, registry API, benchmark execution pipeline, leaderboard | Phase 3 + MICROBOTICA Phase 0 |

### Phase 0 details

Phase 0 establishes the foundation that all subsequent phases build on:
- **Metadata**: All enumerations, meta dataclasses, `MimeNodeMeta`
- **MimeNode ABC**: With `observable_fields`, `commandable_fields`, `validate_mime_consistency`, default `requires_halo=False`
- **Actuation nodes**: `ExternalMagneticFieldNode` (Helmholtz coil model), `MagneticResponseNode` (permanent magnet torque/force)
- **Control**: `ControlPolicy`, `ControlPrimitive`, `ControlSequence`, standard primitive library, `PolicyRunner`
- **Uncertainty**: `IdentityUncertainty`, `ActuationUncertainty`, `LocalisationUncertainty`, `ModelUncertainty`, `ComposedUncertainty`
- **Schema**: `MimeAssetSchema`, `BenchmarkResult`, compliance gate
- **GeometrySource interface**: An abstract base class or protocol in `mime/core/geometry.py` defining the contract that fluid environment nodes (`CSFFlowNode`, `ConcentrationDiffusionNode`, `TissueDeformationNode`) depend on for spatial domain definition. Must support at minimum: (a) parametric geometries (cylinder with given diameter and length — needed for B0 and B4-T1), (b) a versioned USD mesh reference (needed for B4-T2 and beyond, fulfilled by Neurobotika). The interface must be defined before Phase 1 node implementation begins, as `CSFFlowNode` and `ConcentrationDiffusionNode` constructors depend on it. Mesh loading and management is MICROBOTICA's responsibility; `GeometrySource` is the contract MIME nodes consume. Parametric subtypes serialise to USD as typed prims; mesh subtypes reference external `.usd` files by URI with a provenance payload (Neurobotika version, segmentation parameters, source MRI metadata).
- **Benchmarks**: B1–B5 as runnable stubs (test infrastructure, not yet passing); B0 experimental dataset identified, confirmed, and cited in `docs/bibliography.bib` (the stub exists but passing B0 is a Phase 1 deliverable)
- **Documentation**: `DOCUMENTATION_ARCHITECTURE.md`, `MIME_NODE_TAXONOMY.md`, `docs/regulatory/intended_use.md`

---

## 12. Open questions — resolved

### Q1: `ControlPolicy` inside `lax.scan` (from v0.1)

**Status: Deferred with clear path.**

For differentiable controller training, the entire `PolicyRunner` loop (observe → policy → actuate → step) needs to compile into a single `jax.lax.scan` call. The architecture already supports this:
- Policies are stateless (all state in `policy_state` dict)
- `UncertaintyModel` channels are pure functions with explicit RNG keys
- `GraphManager.step()` is already JIT-compiled

The missing piece is a `PolicyRunner.run_scan(n_steps)` method that wraps the loop in `lax.scan`. This requires all components (policy, uncertainty, graph step) to be JAX-traceable — no Python-level conditionals. Design and implement when a concrete differentiable control training use case arises (Phase 3+).

### Q2: Multi-robot (from v0.1)

**Status: Deferred to Phase 3 with design sketch.**

Swarm assets need multiple `RigidBodyNode` + `MagneticResponseNode` instances sharing a single `ExternalMagneticFieldNode`. The graph architecture already supports this (one source, multiple target edges). The challenge is:
- N robot instances under the same field: use MADDENING's `GraphManager` with N copies of the robot subgraph
- Robot-robot interactions (hydrodynamic, magnetic dipole-dipole): additional coupling edges
- `MimeAssetSchema` needs a `swarm_count` field and rules for when benchmark results are valid for swarm configurations

Design alongside Phase 3 when feedback control is implemented.

### Q3: Acoustic actuation topology (from v0.1)

**Status: Resolved.**

`ExternalAcousticNode` produces an acoustic pressure field (not a contact force). `MicroBubbleNode` reads the pressure field and computes the resulting streaming force and oscillation dynamics. The edge topology is:

```
ExternalAcousticNode --[pressure_field]--> MicroBubbleNode
MicroBubbleNode      --[streaming_force]--> RigidBodyNode
```

This differs from magnetic actuation where the response is a torque/force from field interaction. For acoustic, the key coupling is pressure → bubble dynamics → streaming force. The bubble node needs `implicit_residual()` because Rayleigh-Plesset dynamics is stiff.

### Q4: PhaseTrackingNode (from v0.1)

**Status: Resolved — included in Phase 1.**

`PhaseTrackingNode` is an observational node (not a physics node). It reads orientation from `MagneticResponseNode` and field rotation from `ExternalMagneticFieldNode` via edges, and computes `phase_error` = angle between magnetic moment and external field. It is essential for B1 (step-out detection) and B5 (step-out recovery), and useful for the `StepOutDetector` feedback policy.

See §5 for its position in the class hierarchy and edge topology.

### Q5: `ControlPolicy` in `MimeAssetSchema` (from v0.1)

**Status: Resolved — class name strings for now, serialisable spec later.**

Phase 0–2: `compatible_control_policies` stores class name strings (e.g., `"mime.control.ControlSequence"`). This is sufficient for registry queries ("which assets support feedback control?").

Phase 3+: when the registry needs to replay control experiments, introduce a `ControlPolicySpec` dataclass that captures the policy class name, constructor parameters, and initial policy state. This is a JSON-serialisable artifact that can reproduce the exact policy configuration. The `MimeAssetSchema` field becomes `compatible_control_policies: list[str | ControlPolicySpec]` with backward compatibility.

---

## 13. Compliance infrastructure

### Inherited from MADDENING

MIME imports all compliance infrastructure from `maddening.compliance`:

```python
from maddening.compliance import (
    NodeMeta, ValidatedRegime, Reference,
    StabilityLevel, UQReadiness,
    verification_benchmark, stability,
    collect_node_metadata, collect_hazard_hints,
    validate_anomaly_registry,
)
```

### MIME-specific compliance artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Known anomalies registry | `docs/validation/known_anomalies.yaml` | MIME-specific anomalies with `MIME-ANO-` prefix |
| SOUP package | `docs/validation/soup_package.md` | MIME's own SOUP documentation, lists MADDENING as dependency |
| Verification benchmarks | `tests/verification/` | Registered with `@verification_benchmark` using `MIME-VER-` prefix |
| Algorithm guides | `docs/algorithm_guide/nodes/` | One per MIME node, following MADDENING's template |
| Bibliography | `docs/bibliography.bib` | MIME-specific references (CSF dynamics, microrobotics) |
| Intended use | `docs/regulatory/intended_use.md` | MIME-specific platform positioning statement |

### ID namespacing

| Category | Prefix | Example |
|----------|--------|---------|
| Node algorithm IDs | `MIME-NODE-` | `MIME-NODE-001` |
| Anomaly IDs | `MIME-ANO-` | `MIME-ANO-001` |
| Verification benchmark IDs | `MIME-VER-` | `MIME-VER-001` |

### SOUP-of-SOUP chain

```
[Commercial Product] assesses as SOUP:
  └── MICROBOTICA (MBOT-ANO-* anomalies)
      └── MIME (MIME-ANO-* anomalies)
          └── MADDENING (MADD-ANO-* anomalies)
              └── JAX, jaxlib, NumPy (transitive)
```

MIME's SOUP package Section 8 lists MADDENING at a pinned version with links to MADDENING's SOUP package and known anomalies registry. Cross-references between anomaly registries use the namespace convention (e.g., "inherits from MADD-ANO-001").

---

## 14. Ecosystem Case Studies: Inspiration and Differentiation

This section analyses projects that attempted similar ecosystem plays in adjacent fields, drawing concrete design lessons for MIME and MICROBOTICA.

### LeRobot (HuggingFace)

**What they built:** A PyTorch-native library for robot learning with a standardised dataset format (`LeRobotDataset`: Parquet episodes + MP4 video), pretrained policies (ACT, Diffusion Policy, TDMPC), and HuggingFace Hub integration. Hardware-agnostic control interface with plugin system for physical robots (SO-100, Koch, Aloha).

**What worked:**
- `LeRobotDataset` design is excellent: episode-level abstraction, rich metadata for indexing, streaming without full download, delta-timestamp indexing for sub-episode access.
- Hub integration creates a flywheel: publishing a dataset is trivial, discovery is natural, community grows organically.
- Dataset editing tools (`lerobot-edit-dataset` for merging, splitting, removing episodes) lower the friction for data curation.
- Hardware abstraction is clean: `Robot` protocol with `teleop_step()` / `send_action()` / `capture_observation()` decouples policy code from hardware.

**What didn't work / gaps:**
- Format stores *trajectory data* (what the robot did), not simulation programs (what the robot *is*). A LeRobot dataset is passive — you can train on it but not re-simulate under different physics parameters.
- No regulatory traceability, no validated regime documentation, no hazard metadata. Entirely unsuitable for regulated device development.
- No reproducibility beyond "same dataset, same training hyperparameters" — no physics simulation to re-run.

**Lessons for MIME/MICROBOTICA:**
- **Take**: the dataset editing tools concept. MICROBOTICA should provide analogous tools for MIME assets: merging benchmark results from different runs, versioning node graphs, diffing asset schemas between versions.
- **Take**: the Hub integration model (publish from CLI, discover via web, download with one line of code). The MICROBOTICA registry API should be this frictionless.
- **Avoid**: the passive-data-only model. `MimeAssetSchema` is explicitly designed as a superset — everything LeRobot tracks (provenance, hardware description, policy metadata) plus the medical-grade evidence chain (biocompatibility, SOUP classification, hazard hints, anatomical regime validation, benchmark results) plus executable physics (the asset *is* a runnable simulation, not a recording of one).

### Open X-Embodiment (Google DeepMind + 21 institutions)

**What they built:** A cross-embodiment robot learning dataset aggregating 1M+ trajectories from 22 robot types, with a coarsely-aligned action/observation space (7-DOF end-effector actions, canonical camera views). Demonstrated that training on mixed-embodiment data improves transfer to new robots (RT-2-X).

**What worked:**
- The institutional collaboration model: recruited 21 labs directly rather than waiting for community adoption. Built a critical mass of data before launch.
- Demonstrated that cross-embodiment transfer is possible at all — a non-obvious scientific result.
- Simple, pragmatic format: RLDS (Reinforcement Learning Datasets) on TensorFlow Datasets, with per-dataset metadata.

**What didn't work / gaps:**
- Coarse action space alignment loses significant information. A 7-DOF end-effector action space cannot represent the physics of a magnetic helical microrobot (which has frequency, field strength, and field direction as its control inputs). The heterogeneity problem is even worse for microrobotics than for macro-robots.
- No reproducibility infrastructure — there is no way to re-run a trajectory in a simulator and verify it. The data is a snapshot, not a reproducible experiment.
- No ongoing community contribution mechanism after initial dataset aggregation.

**Lessons for MIME/MICROBOTICA:**
- **Take**: the institutional recruitment strategy. MICROBOTICA should recruit specific microrobotics labs (ETH/UZH Nelson group, Polytechnique Montreal Martel group, Max Planck Fischer group, etc.) to publish their robot designs as MIME assets, creating a critical mass before relying on organic community growth.
- **Take**: the lesson that heterogeneity is the hard problem. MIME solves it differently from Open X-Embodiment — not by projecting to a common action space (which destroys physics), but by defining a common *physics interface* (node roles, boundary input specs, commandable fields). A magnetic helical robot and an acoustic bubble robot have completely different action spaces but both express their physics through the same `MimeNode` → `boundary_input_spec()` → `ExternalInputSpec` pipeline. The comparison is: node composition varies, interface contracts are uniform.
- **Avoid**: the static dataset model with no reproducibility. MICROBOTICA's benchmark suite directly addresses this — every benchmark result is attached to version-pinned software (MIME, MADDENING, MICROBOTICA) and can be re-executed in the canonical MICROBOTICA environment.

### Isaac Lab / Isaac Sim (NVIDIA)

**What they built:** GPU-accelerated robot learning environment built on Isaac Sim (Omniverse/PhysX), with a structured task/environment API, articulation and deformable body physics, domain randomisation, and sim-to-real transfer support. Growing library of robot assets (URDF/USD import).

**What worked:**
- Environment API design is clean: `ManagerBasedEnv` composes scene, observations, actions, rewards, and domain randomisation into a structured pipeline. Tasks are modular and reusable.
- GPU parallelism is a genuine differentiator: thousands of environments running simultaneously on a single GPU, enabling RL training that would take weeks on CPU.
- USD as the scene format: enables interoperability with the broader Omniverse ecosystem (CAD tools, rendering, collaboration).
- Domain randomisation infrastructure is well-designed: randomise physics parameters (mass, friction, stiffness) across parallel environments with per-parameter distributions.

**What didn't work / gaps:**
- Physics is rigid body + contact dominated (PhysX). No viscous fluid dynamics, no magnetic field interactions, no drug diffusion. The physics stack is almost entirely irrelevant to microrobotics.
- No medical/regulatory awareness: no biocompatibility metadata, no SOUP classification, no hazard traceability. Asset format (URDF/USD) carries geometry and kinematics but no physics validation evidence.
- Tight NVIDIA dependency: requires Omniverse runtime, CUDA, specific GPU hardware. Not portable.

**Lessons for MIME/MICROBOTICA:**
- **Take**: the scene composition architecture. MICROBOTICA already uses USD as its scene format with a three-layer composition stack (base layer, override layer, results layer). This is architecturally aligned with Isaac Sim's approach.
- **Take**: the domain randomisation concept. MIME's `ModelUncertainty` + MADDENING's `GraphManager.run_sweep()` via `jax.vmap` provides the equivalent — randomise physics parameters across ensemble members. MICROBOTICA should expose this through a user-facing randomisation panel similar to Isaac Lab's `configclass` randomisation specs.
- **Take**: the task/environment API structure as inspiration for how MICROBOTICA wraps MIME simulations for RL training. Isaac Lab's `ManagerBasedEnv` separates observations, actions, and rewards — MIME's `PolicyRunner` + `UncertaintyModel` plays the same structural role.
- **Position against**: `MimeAssetSchema` is what Isaac Lab's asset format would look like if it had been designed for regulated medical applications. Same USD scene foundation, but with the full regulatory evidence chain (biocompatibility, SOUP, hazards, anatomical regimes, benchmark results) that Isaac Lab's format lacks.

### Therapeutics Data Commons (TDC)

**What they built:** A HuggingFace-inspired platform for drug discovery ML — standardised datasets, benchmarks, and model evaluation for ADMET prediction, molecular generation, protein-ligand binding, and clinical trial outcome prediction. ~80 datasets across 22 learning tasks.

**What worked:**
- Benchmark design is thoughtful: tasks are organised by therapeutic development stage (target discovery → lead optimisation → clinical trials), with tiered difficulty (scaffold split vs. time split for testing generalisation).
- Data standardisation is clean: unified API (`from tdc import Oracle; oracle = Oracle(name='Lipophilicity')`) with consistent access patterns across all datasets.
- Leaderboard creates healthy competition and surfaces state-of-the-art methods.

**What didn't work / gaps:**
- Purely data-centric: datasets + model evaluation, no simulation. Good benchmark performance does not guarantee good experimental performance — the gap between in-silico prediction and wet-lab validation is large and unaddressed.
- No regulatory traceability for the models or predictions.
- Limited to molecular scale — no device-level simulation or systems-level pharmacokinetics.

**Lessons for MIME/MICROBOTICA:**
- **Take**: the tiered benchmark difficulty concept. MIME's B4 (closed-loop navigation) should have difficulty tiers based on anatomy complexity: Tier 1 (simple cylindrical channel), Tier 2 (realistic ventricle geometry from Neurobotika mesh), Tier 3 (pathological anatomy — hydrocephalus, stenosed aqueduct). This makes the leaderboard more informative and allows researchers to demonstrate progressive capability.
- **Take**: the therapeutic development stage organisation. MIME's benchmarks already implicitly follow this: B1–B2 (basic physics validation) → B3 (drug delivery capability) → B4–B5 (clinical-scenario readiness). Making this staging explicit in the registry would help users understand what benchmark passage means clinically.
- **Design principle from TDC's weakness**: MIME's benchmarks are physics-first. B1 validates against resistive force theory, B2 against Stokes law — these are grounded in physical law, not statistical correlation. TDC's benchmarks measure model fit to data; MIME's benchmarks measure physical fidelity. This is a stated design principle: benchmark performance must imply physical correctness, not just statistical performance on a held-out set.

### Synthesis Table

| Aspect | LeRobot | Open X-Embodiment | Isaac Lab | TDC | MIME/MICROBOTICA |
|--------|---------|-------------------|-----------|-----|------------------|
| **Asset format** | Trajectory data (Parquet + MP4) | Trajectory dataset (RLDS) | URDF/USD geometry + kinematics | Molecular datasets (SMILES, PDB) | Executable differentiable simulation program (`MimeAssetSchema`) |
| **Core language** | Python/PyTorch | Python/TF | Python/Isaac Sim (C++/PhysX backend) | Python | Python/JAX (MIME) + C++17/Qt 6 (MICROBOTICA) |
| **Heterogeneity handling** | Common action space per robot family | Coarse 7-DOF alignment (information loss) | Task API + articulation abstraction | Dataset splits by task type | Physics interface (node roles + boundary input specs + commandable fields) |
| **Regulatory traceability** | None | None | None | None | IEC 62304 SOUP + ISO 14971 hazard hints + biocompatibility metadata built into asset schema |
| **Reproducibility** | Dataset download + training seed | Dataset download only | Sim re-run (PhysX determinism within same GPU) | Dataset splits (train/test) | Benchmarks executed in canonical MICROBOTICA environment with full version pinning |
| **Differentiability** | Training only (PyTorch autograd on policy, not environment) | No (dataset is static) | Partial (Isaac Gym supports differentiable simulation for some primitives) | No (datasets are static) | Full end-to-end (JAX through MADDENING graph: physics + control + uncertainty) |
| **Physics fidelity** | None (data-driven) | None (data-driven) | Rigid body + contact (PhysX) | Molecular (ADMET, binding) | Viscous fluid + magnetic fields + drug transport + tissue interaction (microrobotics-specific multiphysics) |
| **Community model** | HuggingFace Hub + Discord | Institutional recruitment (21 labs) | NVIDIA ecosystem + GTC | Academic (paper + PyPI) | Registry + leaderboard + open-source library + institutional recruitment |
| **Medical applicability** | None | None | None (macro-robotics) | Drug discovery only (molecular scale) | Designed for EU MDR Class III pathway (CSF microrobotics → commercial device) |
| **Benchmark grounding** | Task success rate on held-out episodes | Cross-embodiment transfer metrics | RL reward + sim-to-real transfer | Statistical fit to experimental data | Physics-first: analytical solutions (Stokes law, resistive force theory) + ensemble robustness |

---

## 15. USD Viewport and Rendering Architecture

### Overview

MIME assets are USD files. Visualising them — whether during local development, in MICROBOTICA's desktop environment, or streamed via WebRTC from a cloud GPU — requires a rendering pipeline that reads a live USD stage and produces pixels. This section specifies the `USDViewport` abstraction that decouples MIME's simulation code from the rendering backend, and the three concrete implementations that serve the three operational contexts.

### The `USDViewport` protocol (`mime/core/viewport.py`)

The protocol is minimal — simulation code only calls two methods:

```python
class USDViewport(Protocol):
    def render(self, stage: Usd.Stage, camera: str = "/Camera") -> np.ndarray:
        """Render the current stage state. Returns an HxWx3 uint8 RGB array."""
        ...

    def close(self) -> None:
        """Release GPU/window resources."""
        ...
```

The `stage` is a live `pxr.Usd.Stage` object — the same stage that the simulation writes to each timestep. The viewport reads it directly; no serialisation/deserialisation step. This is the key requirement that rules out PyVista's standard USD workflow (which reads files, not live stages) and rules out pygfx/wgpu (which has no USD stage concept).

### Three implementations

#### 1. `PyVistaViewport` — local development

**When to use**: local development, CI benchmark visualisation, asset inspection on a laptop without GPU cloud infrastructure.

**How it works**: reads the USD stage, converts to VTK geometry via `pxr.UsdGeom` traversal, renders via PyVista's offscreen renderer. Does not require EGL or a display server — PyVista's offscreen mode uses OSMesa or a virtual framebuffer.

**Limitations**: not a native USD renderer — geometry conversion loses USD-specific features (subdivision surfaces, instancing, USD materials). Acceptable for development and debugging; not suitable for publication-quality visualisation or MICROBOTICA integration.

**Dependencies**: `pyvista`, `vtk`. Both available via pip, no special GPU drivers needed.

#### 2. `HydraStormViewport` — production headless (cloud streaming)

**When to use**: headless GPU rendering for WebRTC streaming via Selkies; the MICROBOTICA cloud module; any context requiring native USD rendering without a display server.

**How it works**: uses `pxr.UsdImagingGL.Engine` (Hydra Storm renderer) with EGL headless context. Renders the live USD stage directly via Hydra — no geometry conversion, full USD material and lighting support. The rendered framebuffer feeds the WebRTC transport layer (Selkies) already validated in MADDENING's cloud module.

**Requirements**: `usd-core` compiled with OpenGL + EGL headless support (`PXR_ENABLE_GL_SUPPORT=ON`, EGL libraries). Requires a GPU with EGL support (any NVIDIA/AMD GPU with appropriate drivers; also works on CPU via Mesa llvmpipe for testing, with reduced performance). No X11 or display server required — EGL renders directly on the GPU framebuffer.

**Why this is the right path**: Hydra Storm is USD's native GPU renderer. It reads the stage directly, handles USD composition (layer overrides, variants, references) correctly, and is the renderer used by NVIDIA Omniverse and USD-compatible tools. The alternative paths (PyVista + Xvfb, pygfx/wgpu) either require X11 or do not read USD stages natively.

**Current status**: the WebRTC transport layer (Selkies) is validated in MADDENING. The missing piece is the `UsdImagingGL.Engine` + EGL pipeline that feeds it. This is the primary deliverable of the `HydraStormViewport` implementation.

#### 3. `MICROBOTICAViewport` — MICROBOTICA desktop integration

**When to use**: when running inside MICROBOTICA's full desktop environment. MICROBOTICA owns the USD stage, the Qt 6 viewport, and the Hydra rendering pipeline. MIME should not duplicate this.

**How it works**: a thin stub that delegates `render()` calls to MICROBOTICA's `RenderSession` interface (C++/Qt 6 side). MIME never owns the window or the GPU context in this mode — it just writes to the shared USD stage and MICROBOTICA's viewport reads it.

**Dependencies**: none (it's a stub until MICROBOTICA Phase 0). The protocol ensures MIME simulation code runs unchanged when MICROBOTICA swaps in its own renderer.

### How the three contexts map to implementations

| Context | Implementation | USD stage ownership | Rendering |
|---------|---------------|---------------------|-----------|
| Local development (MIME standalone) | `PyVistaViewport` | MIME owns the stage | PyVista offscreen |
| Cloud streaming (MICROBOTICA cloud) | `HydraStormViewport` | MIME owns the stage | `UsdImagingGL.Engine` + EGL → Selkies WebRTC |
| MICROBOTICA desktop | `MICROBOTICAViewport` (stub) | MICROBOTICA owns the stage | MICROBOTICA Qt/Hydra viewport |

### Relationship to MADDENING's cloud module

MADDENING's cloud module (MADDENING cloud) handles the WebRTC transport layer via Selkies and ZMQ. The `HydraStormViewport` sits upstream of this: it produces the pixel framebuffer that the WebRTC transport layer encodes and streams. The interface between the two is a raw pixel buffer (numpy array or shared memory), exactly as specified in MADDENING's `PREVIEW`/`STANDARD`/`CAPTURE` latency presets. No changes to MADDENING's transport layer are required — only the USD→pixels step is new.

### Phase assignment

| Phase | Deliverable |
|-------|-------------|
| Phase 0 | `USDViewport` protocol defined in `mime/core/viewport.py`; `PyVistaViewport` implemented and tested with parametric geometry |
| Phase 1 | `PyVistaViewport` extended to handle simulation graph USD output (robot body prims, field visualisation) |
| Phase 2 | `HydraStormViewport` implemented: `UsdImagingGL.Engine` + EGL pipeline, feeding existing Selkies WebRTC transport |
| Phase 4 | `MICROBOTICAViewport` stub replaced with real MICROBOTICA `RenderSession` delegation |

### Open question: USD stage ownership during simulation

When running in MIME standalone mode, the simulation graph (`GraphManager`) owns the physics state, and a separate USD stage is updated each timestep from that state. The question of how to efficiently write simulation state to a USD stage each timestep — without serialising to disk — is a design decision deferred to Phase 1 when the first spatial nodes (CSFFlowNode) produce geometry that needs visualising. The two candidate approaches are: (a) in-memory stage with `Usd.Stage.CreateInMemory()`, written to each timestep and read by the viewport; (b) stage as the authoritative state, with MIME nodes writing directly to USD attributes rather than to a separate JAX state dict. Option (a) is simpler and keeps MADDENING's state management unchanged. Option (b) is more architecturally coherent with USD-as-canonical-format but requires more invasive changes to how node state is stored.
