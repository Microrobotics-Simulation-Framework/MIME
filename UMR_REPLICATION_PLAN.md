# UMR Replication Plan — de Boer et al. (2025)

## Goal

Replicate and surpass Figure 12 from de Boer et al. (2025, Applied Physics Reviews) using MIME's differentiable multiphysics stack. The paper uses a scalar Euler force-balance ODE in uniform flow. The goal is to show that vessel confinement — absent from their model — shifts the step-out curves non-trivially, using the exact UMR geometry they report.

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
| T1.1 | Extract all parameters from de Boer paper | Paper access | `docs/validation/deboer2025_params.md` |
| T1.2 | Implement scalar ODE force balance | MagneticResponseNode, RigidBodyNode | `src/mime/nodes/robot/umr_ode.py` |
| T1.3 | Reproduce 6 speed-vs-frequency curves | T1.2 | `examples/deboer_replication.py` |
| T1.4 | Add JAX autodiff: ∂v/∂(magnet_vol), ∂f_step/∂(diameter) | T1.3 | Gradient computation + plot |
| T1.5 | vmap over (diameter, magnet_vol) parameter space | T1.4 | Continuous Pareto surface |

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

- **Permanent magnet response**: `MagneticResponseNode` currently models soft-magnetic (induced) response. De Boer's UMR uses a permanent NdFeB magnet with fixed moment m. Add a `permanent_moment` parameter that bypasses the susceptibility calculation: T = m × B directly.
- **BGK drag model for helical fin**: the paper uses a specific drag model for the discontinuous helix. Extract the exact drag coefficients from the paper and implement as a resistance tensor option in `RigidBodyNode`.
- **Step-out frequency extraction**: automate the detection of f_step from a frequency sweep — currently `PhaseTrackingNode` detects step-out, but we need a sweep runner that returns f_step as a scalar (for grad).

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
| **Outer vessel wall** (cylindrical BC) | Domain geometry | < 1% | Not implemented (using periodic square domain) | **Must implement pipe wall BB** |
| **Inner UMR wall** (bounce-back) | Geometric staircasing | < 3% | Simple halfway BB (O(dx) wall position error) | **Bouzidi IBB needed for <3%** |
| **Viscosity mapping** | tau → nu conversion | < 0.1% | Implemented, analytical | OK |
| **Steady-state convergence** | Insufficient LBM steps | < 0.5% | Need convergence criterion | Add residual monitor |
| **Finite grid resolution** | Discretisation error | < 2% | At 256³: R/dx ≈ 15 for vessel | May need 512³ |
| **Total (RSS)** | — | **< 5%** | — | — |

### 2.4 Implementation steps (dependency order)

#### Step T2.1: Pipe wall bounce-back with Couette validation

**Goal**: Stationary cylindrical outer wall + rotating inner cylinder, validate torque against the exact Couette solution:

    T_Couette = 4πμΩR₁²R₂² / (R₂² - R₁²)

**What to build**:
- `create_pipe_walls()` already exists in `d3q19.py` — use it as the outer wall mask
- Combine with inner cylinder mask from `create_cylinder_mask_3d()`
- Both walls use the existing `apply_bounce_back` with `compute_missing_mask`
- Inner wall rotates (wall_velocity from `compute_cylinder_wall_velocity`), outer wall stationary

**Validation**: Couette torque within **3%** of analytical at R₁/R₂ = 0.3 and 0.5 (the relevant confinement ratios for the iliac artery).

**Resolution**: 64×64×3 for development, 128×128×3 for CI validation.

**Accuracy note**: the Couette analytical solution is exact for concentric cylinders. The LBM error comes only from wall position staircasing (O(dx)) and the BGK approximation. This is the cleanest possible test of our bounce-back accuracy.

#### Step T2.2: Bouzidi interpolated bounce-back for cylindrical walls

**Goal**: Reduce wall position error from O(dx) to O(dx²) for curved walls.

**What to build**:
- For each boundary link (fluid node + direction pointing into solid), compute the fractional distance q to the actual cylinder surface.
- For a cylinder, this is a closed-form quadratic: ray from fluid node along lattice direction e_q intersects the cylinder `(y-cy)² + (z-cz)² = R²` — solve the quadratic for the parameter t, then q = t / |e_q|.
- Implement the Bouzidi formula:
  - q < 0.5: f_opp(x_f) = 2q · f_q(x_f) + (1-2q) · f_q(x_ff)
  - q ≥ 0.5: f_opp(x_f) = f_q(x_f)/(2q) + (1-1/(2q)) · f_opp_pre(x_f)
  where x_ff is the next-nearest fluid node.

**Validation**: Couette torque within **1%** at the same confinement ratios. The improvement from ~3% (simple BB) to ~1% (Bouzidi) validates the implementation.

**Accuracy note**: Bouzidi requires `q` values for each boundary link. For two concentric cylinders, all q values are computed analytically. For the UMR helix, q computation requires numerical root-finding (deferred to T2.4).

#### Step T2.3: Convergence monitoring

**Goal**: Ensure the LBM reaches steady state before extracting torque.

**What to build**:
- After each N steps, compute the L2 norm of the velocity change: `||u_new - u_old||₂ / ||u_new||₂`
- Stop when the residual drops below a threshold (e.g., 1e-6)
- Log the number of steps to convergence for each parameter point

**Why this matters**: At 256³, each LBM step is expensive. Over-running wastes compute; under-running gives wrong torque. The residual monitor ensures we converge to within the accuracy budget.

#### Step T2.4: UMR geometry — discontinuous helix solid mask

**Goal**: Represent the actual de Boer UMR body on the lattice.

**What to build**:
- Extract the exact UMR geometry from the paper: body diameter 2.84mm, fin shape (discontinuous helix), NdFeB magnet placement
- Implement as a signed-distance function that can be evaluated on any lattice
- The `create_helix_mask()` function already handles continuous helices; extend it for the discontinuous fin described in the paper
- Compute `q` values for Bouzidi IBB on the UMR surface via numerical root-finding (Newton's method, as discussed previously — not analytical)

**Resolution requirement**: at dx = 2.84mm / 15 nodes ≈ 0.19mm, we get ~15 nodes across the UMR diameter. This is the minimum for resolving the helical fin structure. At 256³ with the vessel diameter as the domain size (~9.4mm), dx = 9.4/256 ≈ 0.037mm, giving ~77 nodes across the UMR — more than sufficient. But 256³ × 19 × 4 bytes = 1.3 GB, which fits on a single GPU.

#### Step T2.5: Per-step solid mask update for rotating UMR

**Goal**: The UMR rotates at the actuation frequency. The solid mask must be updated each LBM step.

**What to build**:
- Each step: rotate the UMR Lagrangian surface points by omega·dt
- Recompute the solid mask via signed-distance evaluation
- Recompute the missing_mask from the new solid mask
- Recompute q values for Bouzidi (if using IBB)

**Performance**: At 256³, the signed-distance evaluation is O(N_grid) per step. With the brute-force approach (distance from each grid point to the nearest helix centreline point), this is ~16M distance evaluations per step. On GPU, this parallelises perfectly and should take <1ms. The missing_mask recomputation is 19 × jnp.roll operations, also <1ms.

**Optimisation**: Precompute the signed-distance at the angular resolution of the lattice (1/R_umr_lattice radians), then interpolate. Not needed for 256³ but may be needed for 512³.

#### Step T2.6: Confinement sweep — the core scientific result

**Goal**: Produce the step-out frequency curves as a function of confinement ratio, showing the shift relative to the unconfined (Tier 1) prediction.

**What to build**:
- Fix UMR geometry and magnet volume
- Sweep vessel inner diameter from 4.7mm to 9.4mm (the reported iliac artery range) and ∞ (unconfined reference)
- At each vessel diameter, sweep actuation frequency through the step-out transition
- Extract f_step from the phase error signal at each confinement ratio
- Plot the confined curves alongside the Tier 1 (unconfined ODE) curves

**Expected result**: Step-out frequencies shift downward by 5–20 Hz as confinement increases (vessel diameter decreases). The shift is largest for the largest UMR (2.84mm) in the smallest vessel (4.7mm) where R_umr/R_vessel = 0.30.

**Compute budget**: Each (vessel_diameter, frequency) point requires running the IB-LBM to steady state (~5000–10000 steps at 256³). With ~10 vessel diameters × ~20 frequencies = 200 simulations. At ~1 second per simulation on A100 (estimated), the full sweep takes ~3 minutes. Tractable on a single GPU.

### 2.5 Resolution decision: 256³ vs. 512³

| Resolution | Memory | R_umr/dx | R_vessel/dx (4.7mm) | Expected BB error | Compute per step |
|-----------|--------|---------|---------------------|-------------------|-----------------|
| 128³ | 160 MB | ~19 | ~32 | ~5–10% | ~10ms CPU |
| 256³ | 1.3 GB | ~38 | ~63 | ~2–5% | ~100ms GPU |
| 512³ | 10 GB | ~77 | ~127 | ~1–2% | ~1s GPU |

**Decision**: Start at **256³** for development and the confinement sweep. Move to 512³ only if 256³ torque accuracy is insufficient (>5% error at the target confinement ratios). The 256³ resolution gives ~38 nodes across the UMR, which is well above the 10-node minimum for resolving the geometry.

---

## Tier 3 — Real-Time Cloud Demo

### 3.1 What we're building

A browser-accessible interactive demo where the user can:
- See the UMR rotating inside a cylindrical vessel cross-section at iliac artery dimensions
- Scrub through (diameter, magnet_volume) parameter space
- Watch the step-out frequency sweep update in real time
- See the confined IB-LBM prediction alongside the unconfined Euler ODE prediction

### 3.2 Rendering pipeline status

| Component | Status | What's needed |
|-----------|--------|--------------|
| `StageBridge` | Implemented | Extend for LBM velocity field visualisation |
| `PyVistaViewport` | Implemented | Works for local dev — not for cloud |
| `HydraStormViewport` | **Not implemented** | EGL + `UsdImagingGL.Engine` + framebuffer readback |
| `docker/Dockerfile.usd-gl` | **Not implemented** | OpenUSD with Python + GL from source |
| Selkies WebRTC transport | Exists in MADDENING | Wire HydraStorm framebuffer → Selkies encoder |
| SkyPilot job launcher | Exists in MADDENING | MIME-specific `JobConfig` with `ghcr.io/mime:usd-gl` image |
| Parameter panel UI | **Not implemented** | Browser-side sliders for (diameter, magnet_vol, frequency) |
| Real-time vs. sim-time display | **Not implemented** | Overlay on the WebRTC stream |

### 3.3 Implementation steps

#### Step T3.1: HydraStormViewport

Implement `src/mime/viz/hydra_viewport.py`:
- EGL surfaceless context (`EGL_PLATFORM=surfaceless`)
- `UsdImagingGL.Engine` initialisation
- Synchronous `glReadPixels` framebuffer readback (PBO deferred)
- Satisfies `USDViewport` protocol

#### Step T3.2: Docker base image

Create `docker/Dockerfile.usd-gl`:
- `FROM nvidia/cuda:12.2.0-devel-ubuntu22.04`
- Build OpenUSD from source with `--python` and GL support
- Install MADDENING + MIME + Selkies dependencies
- Publish as `ghcr.io/microrobotics-simulation-framework/mime:usd-gl`

#### Step T3.3: LBM velocity field visualisation

Extend `StageBridge` to write the 3D velocity field from the LBM solver to the USD stage:
- Cross-section colour map: velocity magnitude on a y-z plane through the UMR centre
- Streamlines or arrow glyphs showing flow around the UMR
- This is the visual payload for the WebRTC stream

#### Step T3.4: Wire to Selkies transport

Connect `HydraStormViewport.render()` output to MADDENING's Selkies `StreamingSession`:
- Framebuffer (numpy array) → Selkies encoder → WebRTC → browser
- Use `StreamConfig.from_preset(QualityPreset.STANDARD)` for 720p@30fps

#### Step T3.5: SkyPilot deployment

Create a MIME-specific SkyPilot job configuration:
- `container_image = "ghcr.io/microrobotics-simulation-framework/mime:usd-gl"`
- GPU: T4 or L4 (sufficient for 256³ LBM + Hydra rendering)
- Expose ports: 8000 (API), 8080 (WebRTC stream), 5555 (ZMQ state)

#### Step T3.6: Parameter panel

Build a browser-side control panel that sends parameter updates to the running simulation via ZMQ:
- Sliders: vessel diameter (4.7–9.4mm), magnet volume (1–3 mm³), actuation frequency (0–300 Hz)
- Display: real-time clock, simulation time, current step-out status
- Overlay: Tier 1 ODE curve (precomputed) alongside the live IB-LBM prediction

This is a web frontend (HTML + JS) that connects to the Selkies stream and the ZMQ parameter endpoint. It can be a simple single-page app served by the same container.

### 3.4 Compute requirements

The real-time demo needs the LBM solver to run faster than real time at the actuation frequency. At 200 Hz actuation, the UMR completes one rotation in 5 ms. If we run the LBM at dt_physical ≈ 1 μs (5000 LBM steps per rotation), we need to execute 5000 LBM steps in <33 ms (30 FPS rendering) to maintain real-time playback.

At 256³, each LBM step on an A100 takes ~0.5–2 ms (estimated from JAX benchmarks for similar array sizes). 5000 steps would take 2.5–10 seconds — **not real time**. Options:
- **Subsampled rendering**: render every 100th LBM step, showing 1/100 of real time but with real physics. The stream UI displays "50× slower than real time" explicitly.
- **Reduced resolution**: at 128³, steps are ~8× faster. 5000 steps in ~0.3–1.3 seconds. Still not real time but close.
- **Precomputed sweep**: run the full parameter sweep offline, store the results, and the demo interpolates between precomputed points. The live LBM runs for visualisation only, not for the curve.

**Recommendation**: Use the precomputed sweep approach. The confinement-shifted curves are computed offline (Tier 2 output). The live demo runs a single (diameter, frequency) point in the IB-LBM for visualisation, showing the flow field around the UMR in real time. The parameter panel updates which precomputed curve point is displayed, and the live LBM re-initialises at the new parameters.

---

## Dependency Graph

```
Tier 1:
  T1.1 (extract params) ──────────────────────────────────┐
  T1.2 (scalar ODE) ← T1.1                                │
  T1.3 (reproduce curves) ← T1.2                          │
  T1.4 (autodiff) ← T1.3                                  │
  T1.5 (vmap Pareto surface) ← T1.4                       │
                                                           │
Tier 2:                                                    │
  T2.1 (pipe wall BB + Couette) ← existing LBM            │
  T2.2 (Bouzidi IBB) ← T2.1                               │
  T2.3 (convergence monitor) ← T2.1                       │
  T2.4 (UMR geometry) ← T1.1                              │
  T2.5 (rotating mask update) ← T2.4                      │
  T2.6 (confinement sweep) ← T2.2, T2.3, T2.5, T1.3 ─────┘
                                                    │
Tier 3:                                             │
  T3.1 (HydraStormViewport) ← RENDERING_PLAN.md    │
  T3.2 (Docker image) ← T3.1                       │
  T3.3 (LBM viz) ← StageBridge, T2.6               │
  T3.4 (Selkies wire) ← T3.1, T3.2                 │
  T3.5 (SkyPilot deploy) ← T3.2, T3.4              │
  T3.6 (Parameter panel) ← T3.4, T2.6 ─────────────┘
```

## Timeline Estimate

| Phase | Steps | Estimated effort | Blocking dependencies |
|-------|-------|-----------------|----------------------|
| Tier 1 | T1.1–T1.5 | 2–3 sessions | Paper access only |
| Tier 2 infrastructure | T2.1–T2.3 | 2–3 sessions | None (starts now) |
| Tier 2 UMR-specific | T2.4–T2.5 | 1–2 sessions | T1.1 (paper params) |
| Tier 2 science | T2.6 | 1 session + GPU time | T2.1–T2.5, T1.3 |
| Tier 3 rendering | T3.1–T3.2 | 1–2 sessions | Need `usd-core` with GL |
| Tier 3 demo | T3.3–T3.6 | 2–3 sessions | T2.6, T3.1–T3.2 |

**Critical path**: T2.1 → T2.2 → T2.6 → T3.3 (pipe wall BB → Bouzidi → confinement sweep → visualisation). This is the path from where we are now to the core scientific result.

**Can proceed in parallel**:
- Tier 1 (ODE replication) — independent of LBM work
- T3.1–T3.2 (Docker + HydraStorm) — independent once rendering deps are available
