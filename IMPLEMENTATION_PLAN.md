# MIME Implementation Plan

## Phase 0 — Remaining Foundation (no physics textbook needed)

### 0A. Uncertainty Layer (`src/mime/uncertainty/`)
**Depends on**: Control layer (done)
**Blocks**: Any closed-loop testing, B4/B5 benchmarks

- `UncertaintyModel` ABC (proper base class with composability)
- `IdentityUncertainty` (perfect sensing/actuation baseline)
- `ActuationUncertainty` — frequency jitter, field inhomogeneity, pointing error, thermal drift
- `LocalisationUncertainty` — Gaussian position noise, velocity noise, tracking dropouts, `tracking_confidence` field
- `ModelUncertainty` — fractional noise on state fields (patient variability, fabrication tolerances)
- `ComposedUncertainty` — stacks multiple models, `model_a + model_b` sugar

### 0B. Asset Schema (`src/mime/schema/`)
**Depends on**: Metadata (done), GeometrySource (done)
**Blocks**: Registry integration, benchmark result attachment

- `MimeAssetSchema` dataclass with all fields from ARCHITECTURE_PLAN.md §8
- `BenchmarkResult` dataclass
- `mime_compliant` property (compliance gate)
- `compliance_report()` method
- JSON serialisation (Phase 0–3 intermediate format before USD)
- `from_json()` / `to_json()` round-trip

### 0C. Benchmark Stubs (`src/mime/benchmarks/`)
**Depends on**: Asset schema
**Blocks**: Nothing directly — but establishes the test infrastructure for B0–B5

- `BenchmarkSuite` class that discovers and runs registered benchmarks
- B0–B5 as stub functions that raise `NotImplementedError` with clear messages about what's needed
- Registration mechanism compatible with MADDENING's `@verification_benchmark`

### 0D. B0 Experimental Dataset Selection
**Depends on**: Nothing (literature research)
**Blocks**: Phase 1 node design (constrains what RigidBodyNode must represent)

- Confirm Rodenborn et al. (2013) as primary dataset
- Document robot parameters, channel geometry, fluid properties
- This is a research task, not a code task

---

## Phase 1 — Core Physics Nodes (textbook required)

### 1A. External Magnetic Field Node
**Depends on**: MimeNode (done), Phase 0A complete
**Blocks**: 1C, all magnetic actuation scenarios

- Helmholtz coil (uniform field) model
- Rotating permanent magnet (dipole) model
- `boundary_input_spec`: frequency_hz, field_strength_mt, field_direction
- These are the `commandable_fields` that ControlPolicy targets

### 1B. Magnetic Response Node
**Depends on**: 1A
**Blocks**: 1C, B1

- Permanent magnet response: T = m x B, F = grad(m . B)
- Reads field_vector from 1A via edge, reads orientation from 1C
- Outputs magnetic_torque, magnetic_force to 1C

### 1C. Rigid Body Node (6-DOF, overdamped Stokes regime)
**Depends on**: 1A, 1B
**Blocks**: B0, B1, B2, everything else

- Overdamped dynamics: velocity = R_T^{-1} * F_total, omega = R_R^{-1} * T_total
- Quaternion orientation representation
- Resistance tensor (sphere/prolate ellipsoid analytical, then RFT for helices)
- Additive boundary inputs for forces/torques from multiple sources
- Wall correction factors (Brenner) for confinement

### 1D. CSF Flow Node
**Depends on**: GeometrySource (done), 1C
**Blocks**: B0, B2, B4

- Start with analytical Stokes drag (no resolved flow field) — sufficient for B0, B2
- Pulsatile component via Womersley analytical profiles in cylindrical geometry
- Later: IB-LBM for resolved flow
- Bidirectional coupling: receives robot position/velocity, returns drag force/torque

### 1E. Phase Tracking Node
**Depends on**: 1A, 1B
**Blocks**: B1, B5

- Observer node (not physics)
- Reads orientation from 1B, field rotation from 1A
- Computes phase_error = angle between magnetic moment and external field
- Step-out detection: phase_error > pi/2

### 1F. Benchmarks B0, B1, B2
**Depends on**: 1A–1E all complete

- **B0**: trajectory comparison against Rodenborn et al. experimental data
- **B1**: step-out frequency vs. regularised Stokeslet reference
- **B2**: drag force vs. Stokes law at Re < 0.1

---

## Phase 2 — Drug Delivery + Realistic Environment

### 2A. Drug Release Node
**Depends on**: 1C

- First-order kinetics, Higuchi model, Korsmeyer-Peppas
- Trigger mechanisms (passive, pH, magnetic, acoustic)

### 2B. Concentration Diffusion Node
**Depends on**: 1D, 2A, GeometrySource

- Advection-diffusion equation
- Potentially wraps MADDENING's HeatNode (Mode 1)

### 2C. MRI Signal Formation Node
**Depends on**: 1B, 1C

- Susceptibility artefact model
- Feeds into UncertaintyModel for realistic position noise

### 2D. Flexible Body Node
**Depends on**: 1C

- Cosserat rod / discrete elastic rods for flagellar robots

### 2E. Surface Contact Node
**Depends on**: 1C, GeometrySource

- Penalty-based contact, adhesion models
- Near-wall hydrodynamic corrections

### 2F. Benchmark B3
**Depends on**: 2A, 2B

### 2G. Additional nodes (parallel)
- Non-Newtonian Rheology, Acoustic nodes, etc.

---

## Phase 3 — Closed-Loop + Robustness

### 3A. StepOutDetector Feedback Policy
**Depends on**: 1E, control layer (done)

### 3B. State Estimator (EKF for robot pose)
**Depends on**: 2C, uncertainty layer

### 3C. Benchmarks B4 (T1, T2, T3), B5
**Depends on**: 3A, 3B, Neurobotika mesh (external)

---

## Phase 4 — MICROBOTICA Integration

### 4A. USD Asset Serialisation
**Depends on**: Asset schema, stable node APIs

### 4B. MICROBOTICA Desktop Integration
**Depends on**: 4A, MICROBOTICA Phase 0

### 4C. Registry API + Leaderboard
**Depends on**: 4B
