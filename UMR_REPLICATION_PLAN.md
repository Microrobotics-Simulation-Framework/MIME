# UMR Replication Plan — de Boer et al. (2025)

## Goal

Replicate and surpass Figure 12 from de Boer et al. (2025, Applied Physics Reviews) using MIME's differentiable multiphysics stack. The paper uses a scalar Euler force-balance ODE in uniform flow. The goal is to show that vessel confinement — absent from their model — shifts the step-out curves non-trivially, using the exact UMR geometry they report.

**Paper**: https://pubs.aip.org/aip/apr/article/12/1/011416/3336723

---

## Active Design Decisions

This section collects open decisions that affect the plan's architecture. Each is marked **[ACTIVE DESIGN DECISION]** inline where it first appears.

| ID | Decision | Recommended resolution | Status |
|----|----------|----------------------|--------|
| ADD-1 | BGK drag coefficients for discontinuous helix | Fit to 128 Hz / 0.4 m/s baseline point | **Confirmed**: paper does NOT tabulate drag coefficients (Eq. 1 is a scaling relation, not a closed-form model). Parameter extraction complete: `docs/validation/umr_deboer2025/deboer2025_params.md` |
| ADD-2 | LBM step time at 192³ — precomputed vs. real-time | Precomputed sweep confirmed | **Resolved**: 0.040s/step at 192³ on H100 SXM. Not real-time at any useful resolution. See `pre_t26_gate.md`. |
| ADD-3 | Extensibility: configuration vs. subclass vs. lambda | New subclass for permanent magnet (separate algorithm_id for IEC 62304 traceability); new subclass for novel drag | Resolved: option (b) for both |

---

## Tier 1 — Faithful Replication with Differentiability

### 1.1 What we're replicating

Figure 12: 6 curves of swimming speed vs. actuation frequency for:
- 2 UMR diameters (values from paper)
- 3 NdFeB magnet volumes (1×, 2×, 3× 1mm³)
- Step-out frequencies: 128, 144, 181, 204, 222, 250 Hz

The ODE model: scalar force balance at zero Reynolds number.
- Magnetic torque from permanent NdFeB magnet in rotating field (3 mT)
- BGK drag model for helical propulsion
- Discontinuous helical fin geometry
- UMR body diameter: 2.84 mm

### 1.2 Implementation steps

| Step | What | Depends on | Deliverable |
|------|------|-----------|-------------|
| T1.1 | Extract all parameters from de Boer paper | Paper access | `docs/validation/umr_deboer2025/deboer2025_params.md` **DONE** |
| T1.2 | Implement scalar ODE force balance | MagneticResponseNode, RigidBodyNode | `src/mime/nodes/robot/umr_ode.py` **DONE** |
| T1.3 | Reproduce 6 speed-vs-frequency curves | T1.2 | `examples/deboer_replication.py` **DONE** — d2.8 configs match paper within ±1 Hz on f_step and ±2% on U_peak. d2.1 deferred (MIME-VER-012). |
| T1.4 | Add JAX autodiff: ∂v/∂(magnet_vol), ∂f_step/∂(diameter) | T1.3 | Gradient computation + plot **DONE** — `jax.grad` through `jax.lax.scan` ODE integration; drag sensitivity bands (±10%/±20% C_rot) for T2 confinement comparison. |
| T1.5 | vmap over (diameter, magnet_vol) parameter space | T1.4 | Continuous Pareto surface **DONE** — nested `jax.vmap` over (n_mag, freq) and (freq, drag_factor); 2D speed contour with Couette confinement mapping. |

### 1.3 Accuracy target

The ODE replication must match the paper's curves to within **the line width of Figure 12** — approximately ±2% on peak velocity and ±1 Hz on step-out frequency. This is a code verification exercise: we're solving the same equations, so any discrepancy is a bug.

### 1.4 What exists today

- `ExternalMagneticFieldNode` — rotating uniform field (done)
- `MagneticResponseNode` — soft-magnet torque/force (done, but de Boer uses permanent NdFeB, needs a permanent magnet mode)
- `RigidBodyNode` — overdamped Stokes dynamics with Oberbeck-Stechert drag (done)
- `PhaseTrackingNode` — step-out detection (done)
- `ControlSequence` with `SweepFrequency` — frequency sweep primitive (done)
- `jax.grad` and `jax.vmap` — available via MADDENING/JAX (done)

### 1.5 What needs to be built

**Permanent magnet response**:

> **[ACTIVE DESIGN DECISION — ADD-3: Extensibility architecture]**
>
> `MagneticResponseNode` currently models soft-magnetic (induced) response. De Boer's UMR uses permanent NdBFe Grade N45 magnets with fixed moment m = 1.07 × 10⁻³ A·m² per magnet (1–3 magnets per UMR), oriented perpendicular to the long axis.
>
> **Options evaluated**:
> - **(a) Configuration parameter on existing node**: add `permanent_moment: Optional[jnp.ndarray]` to `MagneticResponseNode.__init__`. When set, bypass the susceptibility calculation and use T = m × B directly. Pro: minimal code change, single NodeMeta. Con: the governing equations are fundamentally different (no susceptibility tensor) — mixing them under one algorithm_id obscures the traceability.
> - **(b) New subclass `PermanentMagnetResponseNode`**: separate class with its own `MIME-NODE-*` ID, `NodeMeta`, and algorithm guide. Pro: clean IEC 62304 traceability (different equations = different algorithm). Con: code duplication for the frame rotation and force computation.
> - **(c) `LambdaPhysicsNode` accepting user-defined torque function**: Pro: maximum flexibility. Con: every lambda needs its own NodeMeta for compliance — defeats the purpose; also harder to audit.
>
> **Recommended**: **(b) New subclass `PermanentMagnetResponseNode`** with its own `MIME-NODE-*` algorithm ID.
>
> Option (a) was initially recommended for its simplicity, but it creates an **ambiguous audit record**: a soft-magnet simulation and a permanent-magnet simulation would both produce `algorithm_id = "MIME-NODE-002"` in the IEC 62304 traceability tables and anomaly registry. If an anomaly is filed against "MIME-NODE-002", a regulator cannot determine which physics was active. The `governing_equations` field documents both paths in prose, but machine-readable compliance tooling (which matches on `algorithm_id`) cannot distinguish them.
>
> Option (b) resolves this: `MagneticResponseNode` (MIME-NODE-002) = soft-magnet susceptibility tensor; `PermanentMagnetResponseNode` (MIME-NODE-008) = fixed moment T = m × B. The shared frame rotation and force computation logic is factored into a common base class or utility function to avoid code duplication. Each node has its own algorithm guide, its own `governing_equations`, and its own anomaly namespace.
>
> For the **drag model**, option (b) is recommended: the discontinuous helix drag is novel physics with different governing equations from Oberbeck-Stechert, warranting a new `HelicalFinDragNode` (or equivalent) with its own `MIME-NODE-*` ID.

**BGK drag model for helical fin**:

> **[ACTIVE DESIGN DECISION — ADD-1: BGK drag coefficient source]**
>
> **Confirmed from paper**: The paper does NOT tabulate drag coefficients. Eq. 1 (§VI.E p.15) gives a scaling relation `U ∝ R_cyl · ω · ε²_cyl · f(De, β)` but the function f(De, β) is not specified. The simulation uses "Newton's second law with Euler's method" (§VI.E p.16) — the drag model is embedded in their code but not published. The OpenFOAM CFD in Fig. 4(d) shows drag torque for continuous vs. discontinuous helices but gives absolute values, not non-dimensionalised coefficients.
>
> The drag model must be reconstructed. Options:
> - **(a) Fit to the 128 Hz / 0.4 m/s baseline point**: use the one known (f_step, v_max) pair with the smallest magnet volume as a single-parameter calibration of the effective drag coefficient. The remaining 5 curves serve as independent validation. **Recommended** — this is the most defensible approach: minimum free parameters, maximum independent validation data.
> - **(b) Resistive force theory approximation**: compute drag from the helix geometry using the RFT coefficients (xi_parallel, xi_perpendicular) and the helix parametrisation. This introduces model-form error (RFT is approximate for finite helix radius/wavelength ratio) but requires no fitting.
> - **(c) CFD reference**: run a separate high-fidelity simulation to compute the drag coefficient. Accurate but circular (we'd be validating our LBM against our own CFD).
>
> **Parameter uncertainty handling**: whichever approach is used, the effective drag coefficient should be treated as a parameter with uncertainty bounds. The Tier 1 Pareto surface (T1.5) should include a sensitivity band showing how the curves shift with ±10% drag coefficient variation. **Compliance note**: the fitted drag coefficient and its uncertainty bounds must be registered in the compliance infrastructure (node hyperparameter, SOUP input record, or versioned artifact) before T2.6 runs.
>
> **Scientific opportunity**: The fact that f(De, β) is unspecified in the paper means Figure 12 is technically non-reproducible from the publication alone — additional assumptions are required. This is not a criticism of the paper (their simulation code produces the curves; the publication simply doesn't include the drag model in closed form). It is, however, a concrete opportunity for MIME: the differentiable stack can identify the drag parameters that best reproduce the published curves via gradient-based fitting (`jax.grad` of the L2 error between simulated and digitised Figure 12 data, with respect to the effective drag coefficients). The fitted parameters are a tangible scientific output — they represent the implicit drag model that the paper's simulation embodies but does not publish. These fitted parameters can then be extended to the confined-flow regime in Tier 2, producing confinement-shifted curves that are directly comparable to the unconfined originals. The fitted drag model and the confinement shift data are concrete deliverables that could be offered to Khalil's group as part of an outreach email, demonstrating the value of the differentiable simulation approach for their ongoing UMR research programme.

**Step-out frequency extraction**: automate the detection of f_step from a frequency sweep — currently `PhaseTrackingNode` detects step-out, but we need a sweep runner that returns f_step as a scalar (for grad).

---

## Tier 2 — Physics Upgrade via IB-LBM

### 2.1 The scientific question

Does vessel confinement shift the step-out curves? De Boer's model uses uniform-flow drag (infinite domain). The UMR operates in the iliac artery (inner diameter 4.7–9.4 mm) with UMR diameter 2.84 mm — confinement ratio R_umr/R_vessel = 0.15–0.30. At these ratios, wall effects increase the effective drag by 5–30% depending on the exact geometry and rotation rate. This should measurably shift the step-out frequency downward.

### 2.2 Working backwards from the goal: accuracy requirements

**The measurement we need**: step-out frequency as a function of confinement ratio, accurate enough to resolve the shift.

**Expected shift magnitude**: At confinement ratio 0.3, the analytical Couette correction to rotational drag is:

    T_confined / T_infinite = R_vessel² / (R_vessel² - R_umr²)

For R_umr/R_vessel = 0.3: correction = 1 / (1 - 0.09) = 1.10 → **10% drag increase**.

Since step-out frequency is inversely proportional to drag (f_step ~ T_mag / drag), a 10% drag increase means ~9% reduction in f_step. For a 200 Hz step-out, that's **~18 Hz shift**.

**To resolve this shift, we need:**
- Torque accuracy: **< 5%** relative error at the operating confinement ratio. A 10% torque error would mask a 10% confinement effect.
- This sets the accuracy budget for each component below.

### 2.3 Accuracy budget by component

| Component | Error source | Budget | Current status | Gap |
|-----------|-------------|--------|---------------|-----|
| **Outer vessel wall** (cylindrical BC) | Domain geometry | < 1% | Pipe wall BB implemented; Couette combined error ~2% at 64x64 with simple BB | **Bouzidi IBB needed for <1%** |
| **Inner UMR wall** (bounce-back) | Geometric staircasing | < 3% | Simple halfway BB validated (Couette ~2% combined at 64x64, ~0.4% at 32x32 R1=6/R2=14). Smooth cylinder only — staircased helix will be worse. | **Bouzidi IBB needed for complex geometry** |
| **Viscosity mapping** | tau → nu conversion | < 0.1% | Implemented, analytical | OK |
| **Steady-state convergence** | Insufficient LBM steps | < 0.5% | Need convergence criterion | Add residual monitor |
| **Finite grid resolution** | Discretisation error | < 2% | At 256³: R/dx ≈ 15 for vessel | May need 512³ |
| **Total (RSS)** | — | **< 5%** | — | — |

### 2.4 Implementation steps — status

| Step | What | Status | Key result |
|------|------|--------|-----------|
| T2.1 | Pipe wall BB + Couette validation | **DONE** | 2.0% error at 64×64 simple BB. MIME-VER-008 passes. `tests/verification/test_ladd_cylinder.py` |
| T2.2 | Bouzidi IBB for cylindrical walls | **DONE** | 0.36% Couette error at 64×64. O(dx²) confirmed. `bounce_back.py:apply_bouzidi_bounce_back` |
| T2.3 | Convergence monitoring | **DONE** | `convergence.py:run_to_convergence` (velocity residual). Rotating UMR uses torque-period convergence (2% rel_change between periods, τ_floor=1e-8) in `run_confinement_sweep.py`. |
| T2.4 | UMR geometry on lattice | **DONE** | `create_umr_mask`, `umr_sdf`, `create_umr_mask_sdf`, `compute_q_values_sdf` (16-iter bisection). Fin geometry corrected (MIME-ANO-003 closed). Helix pitch 8.0mm assumed (MIME-ANO-002 open). |
| T2.5 | Per-step rotating mask | **DONE** | `rotating_body.py:rotating_body_step` with two-pass BB (pipe static, UMR rotating). Mach guard: Ma_tip < 0.1 at fin tips. |
| T2.6 | Confinement sweep | **DONE** | 9/9 runs converged on H100 SXM at 192³. See results below. |
| T2.7 | ODE-LBM coupling | **PENDING** | Apply T2.6 drag multipliers to ODE to produce confined f_step predictions. ~50 lines of script. |

### 2.5 T2.6 Production sweep results (2026-03-23)

**Hardware**: H100 SXM on RunPod (Iceland), 192³, tau=0.8, Ma=0.05, simple BB with two-pass architecture.

| Ratio | Mean torque (lu) | Drag multiplier | Steps to converge | Step time |
|-------|-----------------|----------------|-------------------|-----------|
| 0.15 (ref) | 89.75 | 1.000 | 24,801 | 0.041 s |
| 0.22 | 101.11 | 1.127 | 17,601 | 0.039 s |
| 0.30 | 107.15 | 1.194 | 15,001 | 0.040 s |
| 0.35 (held-out) | 115.20 | 1.284 | 13,801 | 0.040 s |
| 0.40 | 125.14 | 1.394 | 13,401 | 0.040 s |

**Confinement effect**: +12.7% drag at ratio 0.22, +19.4% at 0.30, +28.4% at 0.35, +39.4% at 0.40. Larger than the analytical Couette prediction (10% at 0.30) — the discontinuous fin geometry amplifies confinement effects.

**Orientation repeatability** (ratio 0.30): torque at 0°=107.15, 40°=107.13, 80°=107.17 — variance ±0.015%.

**Track B** (voxelised vs SDF mask at 128³): 0.000% drag difference. MIME-ANO-003 closed.

**Training data**: `data/umr_training_v1.h5` — 9 samples across 5 ratios. Schema correct, data reconstructed from logs.

### 2.6 Resolution decision (resolved)

**192³** selected. Fin circumferential arc = 4.1 lu (well-resolved). Step time: 0.040 s/step on H100 SXM.

| Resolution | Fin arc (lu) | Step time (H100 SXM) | Step time (A100 SXM) |
|-----------|-------------|---------------------|---------------------|
| 64³ | 1.4 | ~0.005 s | ~0.009 s |
| 128³ | 2.6 | ~0.012 s | ~0.021 s |
| **192³** | **4.1** | **0.040 s** | **0.059 s** |

### 2.7 Remaining gap: ODE-LBM coupling (T2.7)

The T2.6 drag multipliers must be applied to the ODE to produce confined step-out frequency predictions. This completes the scientific deliverable.

**Coupling approach**:
1. Scale C_rot by drag multiplier f(ratio) from T2.6
2. Scale C_trans by Haberman-Sayre analytical correction: `1 / (1 - (R_umr/R_vessel)²)`
3. Scale C_prop by geometric mean of C_rot and C_trans multipliers (propulsion involves both)
4. Re-run `sweep_frequency()` at each confinement ratio
5. Compare confined f_step predictions against unconfined baseline

**Infrastructure**: `umr_ode.py` already supports arbitrary C_rot/C_prop/C_trans via the params dict. No new code needed — just a script.

**Limitation**: Scaling C_prop by geometric mean is an approximation. The actual propulsive coupling in confined flow depends on the detailed near-body flow structure, which the LBM captures but the ODE scaling does not. This is documented as a known approximation, not a bug.

---

## Tier 3 — Interactive Cloud Demo (aligned with RENDERING_PLAN.md)

Tier 3 delivers two demos with shared USD scene infrastructure. Both are MICROBOTICA use cases — `.usda` scenes openable in the desktop simulator and streamable via Selkies.

### 3.1 Prerequisites from RENDERING_PLAN.md

| Rendering Plan Step | Status | Required for |
|---|---|---|
| Step 1: StageBridge | **DONE** | All T3 steps |
| Step 2: PyVistaViewport | **DONE** | Local development |
| Step 3: Demo script | **DONE** | Template for T3.A |
| Step 4: HydraStormViewport | **PENDING** | T3.B, T3.D |
| Step 5: Docker image (usd-gl) | **PENDING** | Cloud deployment |
| Step 6: WebRTC/Selkies wiring | **PENDING** | T3.B, T3.D |

### 3.2 Implementation steps (merged with rendering plan)

#### T2.7: ODE-LBM coupling (Tier 2 completion — blocks all T3)

Apply T2.6 drag multipliers to ODE. Produce confined f_step predictions in Hz. See §2.7.

#### T3.A: UMR USD scene infrastructure

Extend `StageBridge` for the UMR confinement demo:
1. UMR body as `UsdGeom.Xform` with `xformOp:orient` updated each step (rotation)
2. Cylindrical vessel as static `UsdGeom.Cylinder` (loaded once)
3. **LBM velocity cross-section**: y-z slice through UMR centre as `UsdGeom.Mesh` (flat NxN quad mesh) with per-vertex `displayColor` primvar (velocity magnitude → colour). Updated each frame. This addresses RENDERING_PLAN.md Open Question #1 for the LBM case.
4. Scene structure: `/World/Robot` (UMR), `/World/Vessel` (pipe), `/World/FlowField` (cross-section mesh), `/World/Camera`
5. `.usda` export for MICROBOTICA desktop viewer

**Depends on**: StageBridge (done), T2.7 (for precomputed data)

#### T3.B: Demo 1 — Quantitative parameter panel (RENDERING_PLAN.md Steps 4–6)

Selkies-streamed interactive UI. User adjusts vessel diameter, magnet count, field frequency.

**Architecture** (resolves ADD-2):
- Quantitative display updates from **precomputed ODE+LBM results** (T2.7 output). No live LBM computation — measured step time (0.040s at 192³) confirms real-time is not feasible.
- USD scene rendered by HydraStormViewport (RENDERING_PLAN.md Step 4)
- Parameter panel: **client-side HTML/JS** that sends updates to server via ZMQ (as specified in RENDERING_PLAN.md). Server interpolates precomputed curves and updates the scene.
- WebRTC stream via Selkies (RENDERING_PLAN.md Step 6)
- Docker image: `ghcr.io/mime:usd-gl` (RENDERING_PLAN.md Step 5)

**Depends on**: HydraStormViewport, Selkies wiring, T2.7, T3.A

#### T3.C: FSI coupling

Dynamic UMR rotation rate. At each LBM step:
1. Compute magnetic torque: T_mag = n·m·B·sin(θ) where θ = field_angle - umr_angle
2. Compute viscous drag from momentum exchange (existing `compute_momentum_exchange_torque`)
3. Integrate: I_eff × dΩ/dt = T_mag - T_drag (Euler, dt = dt_lbm)
4. Update UMR rotation angle: φ += Ω·dt
5. Track phase lag: θ = ∫(ω_field - Ω)dt

Step-out emerges naturally when T_drag > T_mag_max — Ω drops, θ grows unboundedly.

**Resolution**: 64³ for interactive demo (~0.005s/step on H100, ~2 fps with 200 steps/frame). Visual quality sufficient to show flow structure change at step-out.

**Euler stability**: dt_lbm/τ_rot ≈ 1/6000 — not stiff. Euler is sufficient.

**Depends on**: T2.5 (rotating body), `PermanentMagnetResponseNode` (done)

#### T3.D: Demo 2 — Emergent step-out visualisation

Live Selkies-streamed FSI simulation at 64³. User dials field frequency past step-out threshold, observes UMR lose synchrony in real time. Flow field cross-section shows transition from steady rotation to chaotic pulsing. Demo 1 (precomputed) predictions overlaid as reference.

**Visual distinguishability**: The transition from synchronous rotation (smooth, steady flow pattern) to tumbling (oscillating, unsteady flow) is visually dramatic — even non-experts recognise the flow "breaking." Annotation ("Step-out: UMR has lost synchrony") enhances but isn't required.

**Frame rate**: ~2 fps at 64³ on H100 SXM (200 LBM steps per rendered frame × 0.005s/step = 1.0s/frame). Acceptable for demonstration — the physics is the payload, not the frame rate. UI displays "0.5× real time."

**Depends on**: T3.C (FSI), HydraStormViewport, Selkies wiring, T3.A

#### T3.E: Integration

Combined interface: Demo 1 predictions displayed alongside live Demo 2 physics. Parameter panel drives both simultaneously. Single MICROBOTICA scene with precomputed overlay + live simulation viewports.

**Depends on**: T3.B, T3.D

#### T3.F: USDC recording

Replayable `.usdc` file for paper reproducibility. Time-sampled xformOps + velocity cross-section mesh. Openable in usdview, MICROBOTICA, Omniverse.

**Depends on**: T3.A, at least one demo (T3.B or T3.D)

### 3.3 Merged dependency graph

```
T2.7 (ODE-LBM coupling) ──────────────────────────────────┐
                                                            │
Rendering Plan Steps 4-6:                                   │
  HydraStormViewport ← RENDERING_PLAN.md Step 4            │
  Docker usd-gl image ← Step 5                             │
  Selkies WebRTC ← Step 6                                  │
                                                            │
T3.A (UMR USD scene) ← StageBridge (done) ─────────────────┤
                                                            │
T3.B (Parameter panel demo) ← HydraStorm, Selkies, T2.7 ──┤
T3.C (FSI coupling) ← T2.5 (done), PermanentMagnet (done)  │
T3.D (Step-out demo) ← T3.C, HydraStorm, Selkies, T3.A ───┤
T3.E (Integration) ← T3.B, T3.D                            │
T3.F (USDC recording) ← T3.A ──────────────────────────────┘
```

### 3.4 Compute requirements (ADD-2 resolved)

**Measured**: 0.040s/step at 192³ on H100 SXM. Not real-time at any useful resolution.

| Resolution | H100 step time | Steps/frame (200) | Frame time | FPS |
|-----------|---------------|-------------------|------------|-----|
| 192³ | 0.040s | 200 | 8.0s | 0.1 |
| 128³ | 0.012s | 200 | 2.4s | 0.4 |
| **64³** | **0.005s** | **200** | **1.0s** | **~1-2** |
| 32³ | ~0.001s | 200 | 0.2s | ~5 |

**Decision**: Demo 1 (T3.B) uses precomputed results — no live LBM. Demo 2 (T3.D) uses 64³ live FSI at ~2 fps. 32³ achieves ~5 fps but UMR is only ~3 lattice nodes across — too coarse for meaningful flow visualisation.

---

## Out of Scope — Known Limitations

The following physics are explicitly excluded from the current plan. They are natural extensions but not required for the core scientific question (does confinement shift step-out curves?).

### Viscoelastic blood rheology

De Boer et al. characterise blood as a viscoelastic fluid. The Deborah number for blood flow at the relevant shear rates is De ≈ 2 (estimated from published blood rheology data at ~100 s⁻¹ shear rate). **Tier 2 uses a Newtonian LBM** (BGK collision operator with constant viscosity). This is a known simplification.

The effect of viscoelasticity on microrobot drag is an active research question. Elastic stresses can either increase or decrease effective drag depending on the Deborah number and geometry. For De ≈ 2, the effect is likely O(10%) — comparable to the confinement effect we're trying to measure.

**Implication**: The confinement shift we measure in Tier 2 is the Newtonian confinement shift. The actual shift in viscoelastic blood may differ. This should be stated as a limitation in any paper using these results.

**Future extension**: Implement a viscoelastic LBM (e.g., Oldroyd-B via the regularised collision operator) and repeat the confinement sweep. This would be a separate study.

### Wear and surface degradation (Reye–Archard–Khrushchov)

The plan does not model mechanical wear between the UMR and the vessel wall. In practice, contact between a rotating metallic/polymeric robot and the vessel intima could cause tissue damage or robot degradation. The Reye–Archard–Khrushchov wear model relates wear volume to normal force, sliding distance, and material hardness.

**Why excluded**: Wear is a contact mechanics problem that requires modelling the tissue constitutive response and the robot surface properties. The current LBM + bounce-back framework models the fluid mechanics of confinement, not the solid mechanics of contact. Adding wear would require coupling to a tissue deformation model (Phase 2 in IMPLEMENTATION_PLAN.md) and is a separate research direction.

### Non-Newtonian shear-thinning

Blood is also shear-thinning (viscosity decreases with shear rate). At the shear rates generated by a rotating UMR (~100–1000 s⁻¹), blood viscosity drops from ~3.5 mPa·s (low shear) to ~3.0 mPa·s (high shear). This 15% variation is smaller than the confinement effect but not negligible. A Carreau-Yasuda model could be added to the LBM collision operator as a future extension.

---

## Dependency Graph

```
Tier 1: ALL DONE
  T1.1 → T1.2 → T1.3 → T1.4 → T1.5   ✓

Tier 2: T2.1–T2.6 DONE, T2.7 PENDING
  T2.1 → T2.2 → T2.6 (sweep)           ✓
  T2.3 (convergence)                     ✓
  T2.4 → T2.5 (geometry, rotating)      ✓
  T2.7 (ODE-LBM coupling) ← T2.6        ○

Tier 3: PENDING
  T2.7 ─────────────────────────────────────────────┐
  HydraStormViewport ← RENDERING_PLAN.md Step 4      │
  Docker usd-gl ← RENDERING_PLAN.md Step 5           │
  Selkies ← RENDERING_PLAN.md Step 6                 │
  T3.A (USD scene) ← StageBridge (done)              │
  T3.B (param panel) ← HydraStorm, Selkies, T2.7 ───┤
  T3.C (FSI) ← T2.5 (done)                           │
  T3.D (step-out demo) ← T3.C, HydraStorm, T3.A ────┤
  T3.E (integration) ← T3.B, T3.D                    │
  T3.F (USDC recording) ← T3.A ──────────────────────┘
```

## Timeline Estimate

| Phase | Steps | Status | Effort remaining |
|-------|-------|--------|-----------------|
| Tier 1 | T1.1–T1.5 | **DONE** | — |
| Tier 2 infrastructure | T2.1–T2.5 | **DONE** | — |
| Tier 2 science | T2.6 | **DONE** | — |
| Tier 2 coupling | T2.7 | **PENDING** | 1 session (~50 lines script) |
| Tier 3 rendering infra | HydraStorm + Docker + Selkies | **PENDING** | 2–3 sessions (RENDERING_PLAN.md Steps 4–6) |
| Tier 3 UMR scene | T3.A | **PENDING** | 1 session (StageBridge extensions) |
| Tier 3 param demo | T3.B | **PENDING** | 1 session (after rendering infra) |
| Tier 3 FSI | T3.C | **PENDING** | 1 session |
| Tier 3 step-out demo | T3.D | **PENDING** | 1 session (after T3.C + rendering infra) |
| Tier 3 integration | T3.E + T3.F | **PENDING** | 1 session |

**Critical path**: T2.7 → T3.A → HydraStorm → T3.B (quantitative demo). The FSI demo (T3.C → T3.D) runs in parallel with rendering infrastructure.

**Next action**: T2.7 (ODE-LBM coupling) — ~50 lines, unblocks all of Tier 3.
