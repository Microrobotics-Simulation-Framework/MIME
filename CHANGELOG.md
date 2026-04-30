# Changelog

All notable changes to MIME are documented in this file.

## [Unreleased]

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
  Furlani2001, Annin2024AR4, Jackson1998, Krause2013, deJongh2024.

### Test infrastructure
- `pyproject.toml` `[tool.pytest.ini_options]` now defaults to
  `addopts = "-m 'not slow'"` so CI runs the fast suite by default.
  The four expensive tests
  (`test_torque_mode_step_response_analytical`,
  `test_voltage_mode_steady_state`, `test_velocity_mode_pi_tracking`,
  `test_ver131_dejongh_reproduction_short_window`) are tagged with
  `@pytest.mark.slow`. Run the full suite locally with
  `pytest -m 'slow or not slow'`.

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

### Removed

### Fixed

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

### Security

### Known Anomalies
- `MIME-ANO-100` — `point_dipole` model not faithful at r < 5·R_magnet.
- `MIME-ANO-101` — RobotArmNode v1 has fully rigid joints / links.
- `MIME-ANO-102` — MotorNode v1 is an ideal torque source.
- `MIME-ANO-103` — Motor reaction torque on arm is zero (one-way
  coupling); reframed as a future SENSING plan concern.
- `MIME-ANO-104` — v1 experiment controllers read microrobot position
  from physics truth (sensor pipeline deferred to SENSING plan).
