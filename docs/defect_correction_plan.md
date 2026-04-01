# Implementation Plan: Defect Correction Fluid Solver

## Architecture Recommendation: Monolithic Node

**Recommended:** `DefectCorrectionFluidNode` — a single node that owns both the BEM solver and the LBM far-field internally, with the same external interface as `StokesletFluidNode`.

**Why not CouplingGroup composition:**
- The inner iteration (BEM↔LBM) is a tightly coupled algorithmic pipeline, not a physics coupling between independent solvers. The polynomial fit, eval sphere sampling, and traction relaxation are cross-cutting operations that don't map cleanly to edge transforms.
- The LBM warm-start requires persistent state across iterations — not a CouplingGroup pattern.
- CouplingGroup's IQN-ILS operates on interface residuals between nodes. The defect correction's "residual" (traction change) is internal to the pipeline, not an inter-node quantity.
- A monolithic node hides the complexity from the experiment graph. The graph sees: body motion in, drag out — identical to `StokesletFluidNode`.

**What CouplingGroup IS used for:** The outer coupling between `DefectCorrectionFluidNode` and `RigidBodyNode` — the same RB↔fluid coupling as with `StokesletFluidNode`. No CouplingGroup for the inner BEM↔LBM iteration.

## Implementation Steps

### Step 1: `DefectCorrectionFluidNode` (~1 day)

**File:** `src/mime/nodes/environment/defect_correction/fluid_node.py`

```python
class DefectCorrectionFluidNode(SimulationNode):
    """Confined drag via defect correction: BEM body-only + IB-LBM wall correction.

    External interface identical to StokesletFluidNode:
        Inputs:  body_velocity, body_angular_velocity, body_orientation
        Outputs: drag_force, drag_torque

    Internally owns:
        - BEM body-only system (LU-factorized)
        - LBM far-field (vessel wall + open BCs, no body boundary)
        - IB spreading/interpolation stencils
        - Eval sphere stencils for polynomial extrapolation
        - Iteration state (traction, LBM distributions)
    """

    def __init__(self, name, timestep, mu, body_mesh,
                 vessel_radius, vessel_length, dx,
                 n_lbm_spinup=500, n_lbm_warmstart=200,
                 max_defect_iter=10, alpha=0.3, tol=0.01,
                 open_bc_axis=2, epsilon=None,
                 eval_radii_factors=(1.25, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0)):
        ...
        # Init-time setup:
        # 1. BEM: assemble body-only system, LU factorize
        # 2. LBM: create grid, pipe wall mask, open BCs
        # 3. IB: precompute spreading stencil (body surface → LBM grid)
        # 4. Eval: precompute interpolation stencils at each eval sphere
        # 5. BEM eval: precompute Stokeslet operator for eval spheres

    def update(self, state, boundary_inputs, dt):
        """Compute confined drag for current body motion.

        For each of the 6 unit motions (if computing full R matrix)
        or for the current body motion (if computing drag directly):

        1. BEM solve → traction f₀
        2. Iterate: spread → LBM → eval → polynomial → BEM re-solve
        3. Extract drag from converged traction
        """
```

**Key design choices:**

- `update()` computes drag for the current body motion, not the full 6×6 R matrix. The R matrix is computed by calling `update()` 6 times with unit motions — or by a separate `compute_resistance_matrix()` method.
- LBM state persists in `state["f"]` across timesteps. Between physical timesteps, the LBM is warm-started — only the force field changes.
- The iteration count adapts: rotation columns get 1 iteration (wall effect < 5%), translation columns get up to `max_defect_iter`.

### Step 2: Reuse Existing Components (~0.5 day)

**Reused unchanged:**
- `immersed_boundary.py` — Peskin delta, `precompute_ib_stencil`, `spread_forces`, `interpolate_velocity`
- `d3q19.py` — `lbm_step_split` with Guo forcing, `init_equilibrium`, `equilibrium`
- `bounce_back.py` — `compute_missing_mask`, `apply_bounce_back`
- `bem.py` — `assemble_system_matrix`, `compute_force_torque`
- `flow_field.py` — `evaluate_velocity_field`
- `surface_mesh.py` — `sphere_surface_mesh`, `sdf_surface_mesh`

**Modified:**
- `far_field_node.py` — the rewritten IB-mode `LBMFarFieldNode` stays as-is for standalone use, but `DefectCorrectionFluidNode` does NOT wrap it. Instead, it directly calls `lbm_step_split` and `apply_bounce_back` — the LBM is internal, not a separate node.

**New:**
- `src/mime/nodes/environment/defect_correction/__init__.py`
- `src/mime/nodes/environment/defect_correction/fluid_node.py` — the main node
- `src/mime/nodes/environment/defect_correction/wall_correction.py` — multi-radius eval + polynomial fit

### Step 3: Wall Correction Module (~0.5 day)

**File:** `src/mime/nodes/environment/defect_correction/wall_correction.py`

```python
def compute_wall_correction(
    u_lbm,              # (nx, ny, nz, 3) LBM velocity field
    traction,           # (N_body, 3) current BEM traction
    body_pts, body_wts, # BEM surface mesh
    eval_stencils,      # precomputed for each eval sphere
    eval_pts_phys,      # physical coords for BEM eval
    epsilon, mu,
) -> jnp.ndarray:
    """Multi-radius polynomial extrapolation of wall correction Δu.

    Returns Δu at the body surface (3,) — uniform correction.
    For off-centre bodies, returns (N_body, 3) via linear fit.
    """
```

This is ~40 lines — the core of the method extracted into a reusable function.

### Step 4: Experiment Graph Update (~0.5 day)

**File:** `experiments/umr_confinement/physics/setup.py`

The graph simplifies to 4 nodes (from the planned 5):
```
ext_field → magnet → rigid_body ↔ defect_correction_fluid
```

No separate LBM node in the graph. The LBM is internal to `DefectCorrectionFluidNode`.

Edge wiring:
```python
# Rigid body → fluid
EdgeSpec("rigid_body", "defect_fluid", "angular_velocity", "body_angular_velocity")
EdgeSpec("rigid_body", "defect_fluid", "velocity", "body_velocity")
EdgeSpec("rigid_body", "defect_fluid", "orientation", "body_orientation")

# Fluid → rigid body
EdgeSpec("defect_fluid", "rigid_body", "drag_force", "drag_force", additive=True)
EdgeSpec("defect_fluid", "rigid_body", "drag_torque", "drag_torque", additive=True)
```

Identical to the `StokesletFluidNode` wiring — drop-in replacement.

### Step 5: VER-029 Sweep Script (~0.5 day)

**File:** `scripts/run_defect_correction_sweep.py`

```python
for kappa in [0.15, 0.22, 0.30, 0.35, 0.40]:
    # Compute dx from body radius (not vessel)
    dx = body_radius / min_nodes_per_body_radius  # e.g., 6 lu

    # Domain size from vessel
    vessel_R = fin_R / kappa
    N = ceil(2.5 * vessel_R / dx)
    N = ((N + 7) // 8) * 8  # GPU alignment

    # Iteration parameters adapt to κ
    alpha = min(1.0, 1.0 / (1.0 + W_estimate(kappa)))
    max_iter = 1 if is_rotation else max(5, int(15 * kappa))

    # Create node and compute 6×6 R matrix
    node = DefectCorrectionFluidNode(...)
    R = node.compute_resistance_matrix()
```

### Step 6: Validation (~1 day)

**VER-030 revised:** Sphere-in-cylinder at λ=0.3
- Rotation: < 5% error vs NN-BEM (already at 1.2%)
- Translation: < 5% error vs NN-BEM (already at 1.4% with α=0.3, 10 iter)
- Full 6×6 R matrix comparison

**VER-029:** UMR at 5 κ values
- Compare drag multipliers (normalized to κ=0.15) against T2.6b
- Pass: < 10% at each κ

### Step 7: Flow Visualization (~0.5 day)

The walled LBM velocity field from the converged iteration IS the 3D flow. Composite with BEM near-field:

```python
# Inside eval sphere: BEM Stokeslet velocity (analytical, high-res)
# Outside eval sphere: LBM velocity (from grid)
# Blending zone: weighted average over eval sphere shell
```

This provides the Tier A/B/C visualization from the original plan:
- **Tier A:** Midplane velocity magnitude (2D heatmap from LBM midplane slice)
- **Tier B:** Streamlines (3D from composited velocity field)
- **Tier C:** Pressure field (from LBM density, p = ρc_s²)

## Timeline

| Step | Days | Dependencies |
|------|------|-------------|
| 1. DefectCorrectionFluidNode | 1.0 | — |
| 2. Reuse + integration | 0.5 | Step 1 |
| 3. Wall correction module | 0.5 | Step 1 |
| 4. Experiment graph | 0.5 | Steps 1-3 |
| 5. VER-029 sweep script | 0.5 | Steps 1-4 |
| 6. Validation | 1.0 | Steps 1-5 |
| 7. Flow visualization | 0.5 | Step 6 |
| **Total** | **4.5 days** | |

## Recommendation for VER-029 and Demo

**VER-029 (cylinder sweep):** Use **both** NN-BEM direct and defect correction. NN-BEM is the primary reference (validated at <2%, fast). Defect correction is validated against NN-BEM at the same κ values. The sweep report includes both methods — cross-validation.

**Outreach demo:** Use **defect correction** as the primary fluid solver. It provides:
1. Confined drag (for step-out prediction) — from the iterated traction
2. 3D flow visualization — from the LBM velocity field (free byproduct)
3. Geometry-agnostic wall handling — ready for clinical anatomy

The NN-BEM direct solver is the fallback for fast drag-only computation (e.g., parameter sweeps where visualization isn't needed).
