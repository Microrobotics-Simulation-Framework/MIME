# Changelog

All notable changes to MIME are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- The 0.2 line, for the 0.3 rewrite on the `v0.3` branch: the package, its tests, experiments, examples, job and
  cloud-task files, docker files, debug folders, verification and analysis scripts, the root plan documents, and the
  docs. All of it remains at the tag `pre-0.3`. Kept: the licence files, `CITATION.cff`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CLAUDE.md`, `.claude/`, the anomaly registry (`docs/validation/known_anomalies.yaml`; anomaly IDs
  are permanent records), `docs/bibliography.bib`, and the compliance scripts.

### Changed
- The anomaly registry (`docs/validation/known_anomalies.yaml`), now one continuous record across the 0.2 line and
  0.3, in schema 2.0: MADDENING's field names where it has them (`status` is `resolution_status`, `closed` is
  `resolved`, `affected_nodes` is `affected_components`, the node ids merged into the existing list where an entry
  had both), `affected_versions` as a PEP 440 range on every entry, and `resolution_version` and `verification` on
  resolved ones. Resolved for 0.3: MIME-ANO-001 and 002 (removed with their code), 003, 107, 110 and 111; 108 stays
  resolved. Open: 100 to 106 and 109, each with its plan. Corrected: 002's component, 100's workaround, 101's margin.
  The header no longer cites IEC 62304 or any standard.
- `scripts/check_anomalies.py` validates that schema: the status values, no unknown field, a `resolution_version` and a
  `verification` list on a resolved entry, and that each `verification` id names an existing test. It has a `main()`,
  and `tests/test_anomaly_registry.py` runs the real registry and controls against it.

### Added
- The body side of the first experiment (dev1, phase 4b):
  - `mime.core`: `MimeNode` (every constructor argument in `params`, the step written as `advance` and run compiled
    once per node), the quaternion helpers with an exponential whose value and derivatives are finite at a zero
    rotation, the `Shape` protocol and `Cylinder`, and the confinements' velocity rule `remove_outward`;
  - `mime.scene.Tube`, a registered geometry kind (axis, inner radius, wall, length, centre) with typed paths
    `tube.axis`, `tube.inner_surface` and `tube.lumen`; `Structure(geometry=...)`, `Medium(fills=...)`; settings
    marked `Direction` refer to a thing's direction (`rotation_axis=tube.axis`);
  - values declared in the experiment, `exp.declare("umr.mass", kg(...), reason=...)`, read through `mime.materials`,
    the one place that reads a thing's values, and labelled "declared in the experiment" in the report and the saved
    form;
  - `dynamics.RigidInertial` (`rigid_inertial` 0.1.0): implicit resistance, `(M/dt + R) V' = (M/dt) V + F`, with no
    explicit setting, and confinement by projection in its own update;
  - `interactions.Contact` (`contact` 0.1.0, rung `projection`, alias `Contact.Projection`; `penalty` reserved),
    `interactions.Gravity` (`gravity` 0.1.0, weight minus buoyancy and its couple), `interactions.Magnetic.Torque`
    (`magnetic.torque` 0.1.0, `tau = m x B`);
  - `fields.UniformRotatingField` (`uniform_rotating_field` 0.1.0), its time by step count, and `mime.drives` with the
    constant drive (`drive.constant` 0.1.0); an apparatus's `commands`, a quantity being short for the constant drive;
  - `observations.SwimSpeed` (`swim_speed` 0.1.0);
  - MIME's vocabulary for a body and the magnetic field, and `Capability.UNIFORM`; the registry's drive and geometry
    kinds, `Value` reads, `Confinement`, `Command`, reserved rungs, and entries judging their own settings;
  - lag accounting: every edge classified as reading this step, the previous step or the current iterate, from
    MIME's own plan and canonical order, shown in the build report and the graph snapshot, cross-checked against
    MADDENING's compiled schedule and `validate()`'s loop lines; an unplanned feedback loop refused
    (`lag.unplanned`); `latency_in_steps` for a controller's realised latency;
  - the magnetic, gravity and contact families as conformance suites (`mime.testing.MAGNETIC`, `GRAVITY`, `CONTACT`),
    the harness reading declared values, turning composites in the frame case and probing a confining law through
    its body's node; the family walk (every registered interaction passes its family's suite, or CI fails);
  - tests: the physics oracles with their controls (`tests/oracles/`), golden outputs per registered name and
    version, node cases and round-trip cases for every new entry, and node-level differentials against the 0.2 port
    (`tests/differential/`).
- The infrastructure for the physics (dev1, phase 4a):
  - test durations on every CI lane: each lane's tests run on pytest-xdist workers and write JUnit XML, and a
    `Test durations` step writes the bands, the slowest tests and files and the wall and CPU totals to the run page
    and holds MADDENING's budget (a warning over 5 s, a failure over 20 s once re-run alone, an allowlist with
    reasons in `tests/duration_allowlist.txt`). `scripts/report_test_durations.py` is a copy of MADDENING's at
    v0.4.2 with OpenMaterica's CPU columns, every change listed in its header; `tests/conftest.py` records each
    test's own and its children's CPU time. The `plant_slow_tests` input of a manual run is the budget's control;
  - units in annotations: an entry's dimensioned setting declares its dimension once, `Annotated[Quantity,
    Length]`, with markers in `mime.units`; registration reads them (`mime.registry.dimensioned_settings`), phase 1
    refuses a setting of another dimension (`setting.units`) naming the entry, the setting and both dimensions, and
    an entry reads its settings in SI from its context (`ctx.scalar`, `ctx.vector`);
  - typed node state: `mime.core.NodeFields`, `typed_view` and `node_fields`, by which a node types its state and
    inputs as closed TypedDicts of jaxtyping arrays inside MADDENING's own signatures; the convention is in
    `nodes/README.md`, and the toy nodes and the lab fixture's node follow it;
  - the family suites' harness in `mime.testing`: `Family`, `register_family`, `conformance` and `assert_conforms`
    with the case kinds sign, frame, units, material reach, oracle, gradients, swap, equal and opposite, and nodes,
    the sampled ones as invariants of MADDENING's `verify_node`; the node battery (`node_failures`,
    `assert_nodes_verified`, `NodeCase`, `OrderLadder`: `verify_node` on every node an entry builds, and
    `verify_node_order` on each order a node declares); the toy family (`mime.testing.toys.TOY_FAMILY`) as the
    control; and the node walk, which requires a node case for every registered entry;
  - `BuiltExperiment.plans`, phase 1's result, for inspection.
- The saved form on the toys (dev1, phase 3):
  - `mime.saved`: the document (`experiment.mime.yaml`: registered names and versions, settings written differently
    from their defaults, bindings by name, initial states, the run settings and world frame, and empty slots for
    materials, the scene and the controller), read with PyYAML's safe loader and hashed over canonical JSON; the
    lock (`mime.lock`: the environment, every entry with its distribution and version, derived seeds, hashes, the
    resolved record and the derived graph); the reference outputs (`reference.npz`); both files as closed TypedDicts,
    with a run-time check driven by the same annotations;
  - `Experiment.save`, `Experiment.load`, `Experiment.from_document`, `Experiment.document`,
    `BuiltExperiment.save` and `BuiltExperiment.verify`: a load finds entries through installed `mime.entries` entry
    points only, refuses a missing entry or another version with the line to install, rebuilds, and checks the input
    hash exactly, the environment by report, and the derived graph hash for hash within one environment and within
    tolerance across environments;
  - `mime verify <folder>` (`mime.cli`), and the `mime` command;
  - the derived hash: MADDENING's description of the built graph plus every static array's contents, observations
    apart from the physics;
  - units labels on every edge from the vocabulary, `expected_units` on every toy input, and a post-build check
    refusing a missing or mismatched label or an input with no units;
  - MADDENING's `validate()` and `compile()` warnings in the build report with their source, a units warning as a
    refusal;
  - the run's structure in the build report ("one compiled scan" for every experiment so far); a probe that traces
    the compiled step once at build and refuses a host callback inside it, naming the node;
  - `mime.testing`: `round_trip_failures`, `refused_round_trip_failures`, `installed_entries`, `covered`,
    `load_in_fresh_process` and `param_reach_failures` (every trainable leaf reaches the compiled step without a
    recompilation);
  - tests: the registry walk in the default lane (an entry or kind of thing without a round-trip case fails CI), with
    its controls; the document, load, schema and compiled-step tests; a packaging test that `py.typed` ships in the
    wheel; the typing gate's own tests;
  - typing: `pyrightconfig.json` (Python 3.12, strict over `src/mime`, `reportDeprecated`,
    `reportImplicitOverride` and unnecessary ignores as errors), `scripts/typing_check.py`, and a blocking
    `typecheck` CI job that refuses an untrustworthy run, plants a type error as its control, and type-checks a
    consumer snippet against the built wheel, with and without `py.typed`; `@override` on every overriding method.
- The core of 0.3 (dev1, phase 2), tested on toy entries with no physics:
  - `mime.units`: SI quantities and their constructors (`mm`, `ms`, `Hz`, `mT`, ...), checked by dimension in phase 1
    and turned into plain SI floats;
  - `mime.vocabulary`: interface entries with kind, units, frame, shape, sign, dtype and typical magnitude, versioned
    per namespace; the six target kinds; the closed list of capability flags; reserved names, refused as "reserved,
    not built";
  - `mime.things`: `Structure`, `Medium`, `Body` and `Apparatus` as declarations; `State`; handles, and typed paths
    to an apparatus's fields;
  - `mime.registry`: `register_interaction`, `_observation`, `_dynamics`, `_field_provider` and `_controller`,
    keyed by (name, version); entries as frozen dataclasses of their settings; plans as data; support tables as
    predicates over capability records, with no default rung and no default binding;
  - `mime.build`: phase 1, which resolves handles, typed paths and strings (with "did you mean"), checks reads,
    targets, rungs, rates and registrations, merges solve-together requests into coupling groups, derives ids and
    seeds, and refuses with every reason at once, each with a class and an action line; phase 2, which adds nodes in
    the canonical order (providers, interactions by id, media, bodies, apparatus, then observations) and edges by
    delivering id through restricted builders that set `additive` from the target's kind, and every coupling group
    with `solver="ift"`, `atol=0` and an explicit tolerance and cap; the post-build checks; the build report; the
    world frame as a stand-in until the scene lands;
  - `mime.experiment`: `Experiment` (`add_*`, `attach`, `observe`, `plan`, `build`; `dt` as the base step with no
    default, `rate=` a whole multiple of it) and `BuiltExperiment` (report, graph text, `run`);
  - `mime.gradients.GradientStatus`, which every entry declares;
  - `mime.testing`: the graph-invariant helpers (graph text, order invariance) and `mime.testing.toys`;
  - tests: the canonical order against a hand-assembled reference over every order of the calls, with its controls;
    the order-invariance property in Hypothesis, with its control; a mutation test per post-build check; golden
    graph text; pins of the MADDENING 0.4.x behaviours the build works around;
  - `lab_fixture/`, a package outside `src/mime` that registers an interaction from outside MIME; every CI lane
    builds and installs it.
- The 0.3 package skeleton: the subpackages of the 0.3 layout, each empty with a line stating its role; `py.typed`;
  `mime.testing` on `maddening.testing`, needing the new `verify` extra.
- Packaging for 0.3: version `0.3.0.dev0` (single-sourced from `mime.__version__`), Python 3.12 to 3.14,
  `maddening>=0.4.1,<0.5`, `jax>=0.10,<0.13` (MADDENING 0.4.1's own range), and `openmaterica>=0.1.0.dev3` and
  `usd-core>=26.5` as required dependencies; the `verify` and `dev` extras; a committed `uv.lock`.
- CI on uv with four lanes, each checking from the installed metadata that it installed what it claims: the locked
  PyPI releases on Python 3.12 to 3.14 (blocking); OpenMaterica's `main`; MADDENING's `main`; the newest usd-core,
  weekly. A manual run's `plant_failure` input fails a canary test in every lane.

### Changed
- Development environments use Python 3.14: `.python-version` steers `uv sync` and `uv venv`, which otherwise take the
  newest Python installed, and so failed on a machine with Python 3.15, for which usd-core ships no wheels. The
  package still supports Python 3.12 to 3.14, and CI still runs each of them.
- The canonical order gathers each coupling group's members at the place of its last member, so a law added between
  a group's members runs before the group: MADDENING 0.4.x runs a group as one block at its first member's place,
  and the full toy's body read the spring's load a step late. The full toy's node order and history change.
- The build report lists the values read and every edge's lag, and the graph snapshot each edge's lag; the toys'
  golden snapshots are regenerated.
- A delivery into an inequality-constraint target is an entry defect: a confinement is planned as a `Confinement`.
- MIME is licensed under the Apache License, Version 2.0, from the 0.3 line on, as MADDENING is from 0.4.2: `LICENSE`
  holds the Apache-2.0 text, `NOTICE` the copyright and the note that the 0.2 line (the tag `pre-0.3` and earlier)
  was distributed under the GNU Lesser General Public License, version 3 or later; `pyproject.toml` declares
  `license = "Apache-2.0"` and `license-files` (PEP 639) with no licence classifier; `CITATION.cff`, the README and the
  lab fixture say Apache-2.0; `COPYING.GPL-3` is removed. `CONTRIBUTING.md` states the licensing of contributions
  (inbound equals outbound) and a DCO sign-off on each commit, documented and not yet enforced.
- MIME requires MADDENING 0.4.2 (`maddening>=0.4.2,<0.5`).
- The toy drag is version 0.2.0: its quadratic rung takes the velocity's norm (with a derivative of zero at rest), so
  turning the world turns its load; 0.1.0 took each component's magnitude. The toys' settings are quantities with
  declared dimensions, at the SI values they held as plain numbers.
- `tests/build/test_canonical_order.py` runs 100 steps, which the hand-built control needs to tell the summation
  orders apart with the toy drag at 0.2.0.
- The build's host-callback probe traces each node alone through MADDENING's public node interface (its state, the
  inputs `resolve_boundary_inputs` gives it, `update` and `compute_boundary_fluxes`) instead of the whole step
  through MADDENING's private step function. Nothing private is read, the probe no longer needs the compiled step,
  and a callback in a node's flux hook is now refused too.
- Development tooling moved from the `dev` extra to a PEP 735 `dev` dependency group (pytest, pytest-xdist,
  Hypothesis, `pyright==1.1.414`, hatchling); the `verify` extra stays for labs. New direct dependencies:
  `typing_extensions>=4.16.0`, `jaxtyping>=0.3.11` and `pyyaml>=6.0`.
- The toy nodes name their initial conditions `initial_*`, which MADDENING takes as initial conditions rather than
  step parameters; the toy field holds a static phase table.
- A built graph's canonical text shows each edge's units labels.
- `RunSettings` holds the script's values as written (`Any`), since phase 1 is what checks them.
- `scripts/check_anomalies.py` checks the registry with its own check (top-level fields, unique prefixed IDs,
  required fields), the one the 0.2 line's CI ran: MADDENING 0.4's validator does not accept the registry's 0.2
  schema.

### Known Anomalies
- MIME-ANO-110: the 0.2 permanent-magnet response computes its moment in float32 even under x64 (found by the
  node-level differential; 0.3's torque does not).
- MIME-ANO-111: the 0.2 vessel constraint removes only `(R/r)^2` of the outward velocity of a body it projects back
  (found by the node-level differential; 0.3's projection removes all of it).

### Fixed
- `.gitignore` anchors `build/` and `dist/` at the root; unanchored, they ignored `src/mime/build/`.
- pytest collects `tests/build/`: its default `norecursedirs` skips any folder named `build`, so `pyproject.toml`
  lists the default folders without it, and `tests/test_collection.py` fails if a test file is ever left out.

## [0.2.0] - 2026-06-11

Brings MIME into full alignment with **MADDENING v0.2.1** and lands the two
pieces of new architecture the v1.0 stack roadmap puts on the critical path:
**multi-GPU sharding of the IB-LBM fluid node** and the **EffectModel contract
pilot**. Adopts the v0.2 node-API surfaces (`halo_width()`, the `static_data`
channel, compile-time edge validation) and folds in the TF32 / float32
numerical-precision fixes. Narrative summary in
[`docs/release_notes/v0.2.md`](docs/release_notes/v0.2.md).

### Added
- **IBLBM multi-GPU sharding** — `IBLBMFluidNode.update_padded` now has a
  working multi-device path (previously `NotImplementedError`): halo-aware
  streaming (`d3q19.stream_padded`), per-slab pipe + UMR missing-link masks
  (`bounce_back.compute_missing_mask_sharded`), the UMR body rebuilt on each
  slab's global coordinates (an `origin` offset threaded through the geometry
  helpers), and psum'd drag force/torque. Bit-identical to the single-device
  step under jit on a 4-device mesh (`tests/verification/test_lbm_sharded_contract.py`).
  Requires `maddening>=0.2.1`.
- **EffectModel contract pilot** (`mime.effects`, per ADR-2026-EFFECT-MODEL) —
  the Protocol surface (`EffectModel` / `SourcedEffectModel[S]` /
  `EffectHandle` / `CouplingSpec` / polymorphic `Regime`), a `@register_effect`
  registry, an `Experiment` with the six-pass `build()` validation contract
  (each pass a typed error) and load-time MIME-version compatibility checks,
  and the `HydrodynamicModel` family (LBM / FVM / Stokeslet / DefectCorrection
  backends adapting the existing fluid nodes over the shared drag contract).
  The MagneticModel family, `SourceInputProvider`, and cross-effect coupling
  implementation are deferred to v0.3.

### Changed
- Dependency floor raised to `maddening>=0.2.1,<0.3` (was `>=0.2.0`); CI now
  pins the MADDENING checkout to the `v0.2.1` tag. Unlocks the IBLBM sharded
  `static_data` / `domain_integral_fields` path.

### Fixed
- **`jax_enable_x64` test-isolation leak.** Seven verification modules enabled
  x64 at *import* time, which pytest runs during collection — so x64 leaked on
  for the whole session, made results order-dependent (a hazard under
  pytest-xdist), and produced spurious `float64 → float32` cast FutureWarnings
  in float32 nodes. Replaced with a `@pytest.mark.x64` marker + an autouse
  conftest fixture that scopes x64 per-test with deterministic teardown;
  `tests/test_x64_isolation.py` guards against regressions.
- **`oberbeck_stechert_coefficients` float32 cancellation.** The prolate-
  spheroid drag-coefficient denominators subtract a leading ±2e that cancels
  against `2·atanh(e)`, losing all float32 precision near e=0 (C₁=1.037 instead
  of 1.000 at e=0.01 — only masked when a test ran under the leaked x64).
  Reformulated cancellation-free via `g = atanh(e) − e` (series for small e);
  algebraically identical, validated to ≤2.6e-7 vs a float64 reference across
  e ∈ [0.001, 0.99].
- **AR4 safety-clamp test** updated for the controller perf refactor that fused
  `_update_target_from_body` into the jitted step law; it now drives the target
  filter to convergence through the public `compute()` path.

### Test infrastructure
- `@pytest.mark.x64` marker + an autouse conftest fixture scope
  `jax_enable_x64` to opting-in tests with deterministic teardown (replaces
  module-level `jax.config.update` calls that leaked into the whole session).

### Performance
- **`ar4_helical_drive` declarative actor poses.** The recorder now sources
  each arm-link and rotor/magnet pose directly from its producing node's output
  field (`pose_from` → `arm.link_poses_world[i]`, `motor.rotor_pose_world`)
  instead of re-deriving poses from per-actor `position`/`orientation` state.
  Significantly faster experiment iteration at identical physical output.

## [0.1.0] - 2026-05-11

First tagged release. Establishes the MIME package, the `MimeNode` ABC over
MADDENING's `SimulationNode`, the actuation-apparatus node decomposition
(Motor + PermanentMagnet + RobotArm), and the initial verification suite.

### Added
- Project scaffold: pyproject.toml, package structure, core modules
- `MimeNode` ABC extending MADDENING's `SimulationNode`
- `MimeNodeMeta` and all domain metadata dataclasses
- `GeometrySource` protocol with `CylinderGeometry` and `MeshGeometry`
- `USDViewport` protocol
- Known anomalies registry (empty — no anomalies yet)
- Bibliography with initial microrobotics references
- DOCUMENTATION_ARCHITECTURE.md (self-contained)
- ARCHITECTURE_PLAN.md v0.6
- MIME_NODE_TAXONOMY.md
- GitHub issue template for anomaly reporting
- Claude skills: commit-and-push, new-node, new-anomaly, new-verification-test
- **Actuation decomposition** — three new external-apparatus nodes
  introduced as a Motor + PermanentMagnet + RobotArm chain that
  replaces `ExternalMagneticFieldNode` whenever spatial field
  variation, finite-magnet near-field, or apparatus-mediated effects
  matter. The legacy node stays as a first-class peer for
  uniform-Helmholtz cases.
  - `MotorNode` (`MIME-NODE-100`) — single-axis rotary motor (DC
    brushed) with torque / voltage / velocity command modes.
  - `PermanentMagnetNode` (`MIME-NODE-101`) — bar-magnet field
    producer with three field models (`point_dipole`, `current_loop`,
    `coulombian_poles`). Drop-in for the existing
    `PermanentMagnetResponseNode` consumer.
  - `RobotArmNode` (`MIME-NODE-102`) — URDF-driven articulated
    rigid-body manipulator. FK / CRBA / RNEA / forward dynamics, all
    pure JAX, jit/grad/vmap traceable.
- New domain metadata: `MotorMeta`, `ArticulatedArmMeta`, plus
  `ActuationPrinciple.MOTOR_ROTOR` and `ActuationPrinciple.ARTICULATED_ARM`.
- `mime.control.kinematics` package (URDF parser, SE(3) helpers,
  6D spatial algebra, FK / CRBA / RNEA) — patterns mirrored from
  `tmp/frax`, no frax import.
- `mime.experiments.dejongh_new_chain.build_graph` — dejongh graph
  driven by Motor + PermanentMagnetNode (with body↔magnet coupling
  group per dejongh deliverable A.2).
- `MICROROBOTICA/experiments/_template_helical_drive/` — generic
  template for any URDF arm.
- `MICROROBOTICA/experiments/ar4_helical_drive/` — AR4 (Annin
  Robotics) demo bundled as the default-arm working example.
- `scripts/urdf_to_usd_scene.py` — author-time helper that derives
  a `world.usda` link hierarchy from a URDF.
- `scripts/run_ar4_helical_drive.py` — standalone runner for the AR4
  experiment that loads `physics/`, builds the graph, and writes
  per-frame JSON-lines matching the MICROROBOTICA `ResultFrame`
  schema. Useful for iterating on the simulation without launching
  the Qt viewer; runs on GPU when available.
- `tests/conftest.py` — sets the GPU env vars and JAX persistent
  compile cache before any test module imports JAX. Three knobs
  matter on this rig:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
    `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` so cuSolver's handle
    creation (called by `RobotArmNode.update` via `jnp.linalg.solve`)
    succeeds without JAX preallocating the whole device.
  - `XLA_FLAGS=--xla_gpu_autotune_level=0` skips XLA's per-shape
    autotune; for our tiny matrices the autotune cost dominates the
    runtime cost.
  - `jax_compilation_cache_dir` (default
    `~/.cache/jax_compilation_cache`) — persistent on-disk cache
    for compiled XLA executables. First run still pays the full
    compile, subsequent runs load the artefact from disk.

### Changed
- `mime.core.metadata.MimeNodeMeta` now carries optional `motor` and
  `articulated_arm` fields (additive — no breaking change).
- `MimeNode.validate_mime_consistency` now cross-checks
  `MotorMeta.commandable_fields` and `ArticulatedArmMeta.commandable_fields`
  against `ActuationMeta.commandable_fields` when both metas are set.
- `MotorNode.update` now also stores `rotor_pose_world` in state so
  downstream edges read it as a state field (matches the convention
  used by `ExternalMagneticFieldNode` for `field_vector`).

### Deprecated
- *None.* `ExternalMagneticFieldNode` (`MIME-NODE-001`) **stays as a
  first-class peer** — it remains the right choice for uniform-
  Helmholtz workspace simulations.

### Performance
- **Standalone runner XLA-compile speedup**:
  - Before: 678 ms/step steady-state, 55 s pre-warm (cold cache).
  - After: 76 ms/step steady-state with `--no-coupling-group`,
    95 ms/step with the coupling group; 14 s pre-warm (warm cache);
    40 s pre-warm (cold cache, autotune-off).
  - ~9× per-step speedup, ~4× pre-warm speedup. Wins came from
    (a) persistent JAX compile cache, (b) skipping XLA gemm/solver
    autotune, (c) pre-jitting the per-frame `link_world_poses`
    closure in `run_ar4_helical_drive.py` so the sample callback
    no longer retraces, (d) `dejongh_new_chain.build_graph` now
    accepts `use_coupling_group=False` for visualisation work
    (staggered back-edges; one-step phase lag, ~10° at 60 Hz).
  - Remaining per-step cost is structural — Gauss-Seidel inner
    loop + per-node cuSolver dispatch — and addressed in a future
    MADDENING-level optimisation.
- New bibliography entries: Featherstone2008, Sciavicco2000,
  Furlani2001, Annin2024AR4, Jackson1998, Krause2013, deJongh2025.

### Test infrastructure
- `pyproject.toml` `[tool.pytest.ini_options]` now defaults to
  `addopts = "-m 'not slow'"` so CI runs the fast suite by default.
  The four expensive tests
  (`test_torque_mode_step_response_analytical`,
  `test_voltage_mode_steady_state`, `test_velocity_mode_pi_tracking`,
  `test_ver131_dejongh_reproduction_short_window`) are tagged with
  `@pytest.mark.slow`. Run the full suite locally with
  `pytest -m 'slow or not slow'`.

### Verification
- `MIME-VER-100` — MotorNode torque-mode step response vs analytical
  first-order solution.
- `MIME-VER-110` / `MIME-VER-111` — PermanentMagnetNode dipole field
  and gradient vs analytical at far-field standoffs.
- `MIME-VER-112` — Earth-field superposition.
- `MIME-VER-120` — RobotArmNode forward kinematics on a 3-link planar
  fixture vs closed-form analytical reference.
- `MIME-VER-121` — Mass-matrix symmetry & PD across random configs.
- `MIME-VER-122` — RNEA / CRBA round-trip consistency at zero velocity.
- `MIME-VER-123` — Gravity-compensated static hold (1 s, < 1e-3 rad
  drift).
- `MIME-VER-124` — Inverse-dynamics + PD trajectory tracking on a
  0.5 Hz sinusoid (RMS error < 0.5°).
- `MIME-VER-130` — Motor + PermanentMagnetNode reproduce the legacy
  `ExternalMagneticFieldNode` rotating field at the UMR location in
  the far-field aligned limit.
- `MIME-VER-131` — Dejongh helical-swim short-window reproduction.
  Both chains produce a finite, same-sign axial velocity.
  Quantitative magnitude disagreement is by design — the new chain
  restores the previously-zero `(∇B)·m` gradient-force term flagged
  in dejongh deliverable Appendix A.1.
- `MIME-VER-132` — Misalignment-induced field tilt at the UMR grows
  monotonically with the magnet's lateral offset over a 0–6 mm sweep.

### Known Anomalies
- `MIME-ANO-100` — `point_dipole` model not faithful at r < 5·R_magnet.
- `MIME-ANO-101` — RobotArmNode v1 has fully rigid joints / links.
- `MIME-ANO-102` — MotorNode v1 is an ideal torque source.
- `MIME-ANO-103` — Motor reaction torque on arm is zero (one-way
  coupling); reframed as a future SENSING plan concern.
- `MIME-ANO-104` — v1 experiment controllers read microrobot position
  from physics truth (sensor pipeline deferred to SENSING plan).

[Unreleased]: https://github.com/Microrobotics-Simulation-Framework/MIME/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Microrobotics-Simulation-Framework/MIME/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Microrobotics-Simulation-Framework/MIME/releases/tag/v0.1.0
