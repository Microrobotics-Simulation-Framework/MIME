# IBLBMFluidNode Specification

*Created 2026-03-24. Prerequisite for T3.C (FSI coupling) and T3.D (step-out demo).*

## 1. Purpose

`IBLBMFluidNode` wraps the existing LBM code (`d3q19.py`, `bounce_back.py`, `rotating_body.py`) as a proper MADDENING `SimulationNode`. This replaces the manual wiring in `run_confinement_sweep.py` with a node-graph approach where the LBM fluid solver is coupled to `RigidBodyNode` via edges.

## 2. MADDENING Architecture Compatibility

### Assessment (2026-03-24)

**No MADDENING extensions are required.** The existing `SimulationNode` interface supports all IBLBMFluidNode requirements:

| Requirement | Interface feature | Status |
|---|---|---|
| 1.3 GB f-array state | `initial_state()` returns dict of JAX arrays. `GraphManager` passes state through JIT — XLA manages buffer reuse, no Python copies. | Supported |
| Variable solid mask each step | Mask is a state array updated in `update()`. No fixed-geometry assumption in the interface. | Supported |
| Quaternion boundary input | `BoundaryInputSpec(shape=(4,))` — arbitrary shapes supported. | Supported |
| Circular dependency (LBM ↔ RigidBody) | `CouplingGroup` with Gauss-Seidel, or back-edges with one-step lag. One-step lag is standard in IB-LBM. | Supported |
| Spatial stencil (streaming) | `requires_halo = True` — interface property designed for exactly this. | Supported |
| Non-scannable geometry | `GraphManager.run()` uses Python loop, calling `_compiled_step` once per step. No requirement for `jax.lax.scan` compatibility. | Supported |
| Force/torque output | `compute_boundary_fluxes()` returns `dict` with arbitrary-shaped arrays. | Supported |

### Why run_scan() is not required

The LBM geometry generation (`create_umr_mask`) uses JAX operations that ARE traceable, but the Python for-loop over 19 directions in `compute_q_values_sdf_sparse` gets unrolled during tracing, making compilation expensive. `GraphManager.run()` with its Python loop is the correct execution path — each step is JIT-compiled, but the loop is in Python. This is identical to the current `run_confinement_sweep.py` pattern and is performant (measured 0.04s/step LBM overhead at 192^3 on H100).

## 3. Node Interface

```python
@stability(StabilityLevel.EXPERIMENTAL)
class IBLBMFluidNode(MimeNode):

    meta = NodeMeta(
        algorithm_id="MIME-NODE-010",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="3D IB-LBM fluid solver with Bouzidi IBB for confined microrobot flows",
        governing_equations="BGK-LBM D3Q19, Bouzidi interpolated bounce-back, momentum exchange",
        discretization="D3Q19 lattice Boltzmann with BGK collision operator",
        ...
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ENVIRONMENT,
        anatomical_regimes=(...),  # iliac artery, CSF, etc.
    )
```

### 3.1 Constructor parameters

| Parameter | Type | Description |
|---|---|---|
| `nx, ny, nz` | int | Lattice dimensions |
| `tau` | float | BGK relaxation time |
| `vessel_radius_lu` | float | Pipe wall radius in lattice units |
| `body_geometry_params` | dict | UMR geometry kwargs for `create_umr_mask` |
| `use_bouzidi` | bool | Enable Bouzidi IBB for body surface |
| `dx_physical` | float | Physical lattice spacing [m] for unit conversion |

### 3.2 State

```python
def initial_state(self) -> dict:
    return {
        "f": init_equilibrium(nx, ny, nz),          # (nx, ny, nz, 19) float32
        "solid_mask": initial_solid_mask,             # (nx, ny, nz) bool
        "body_angle": jnp.array(0.0),                # current rotation angle
        "drag_force": jnp.zeros(3),                   # (3,) float32
        "drag_torque": jnp.zeros(3),                  # (3,) float32
    }
```

### 3.3 Boundary inputs

```python
def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
    return {
        "body_angular_velocity": BoundaryInputSpec(
            shape=(3,), description="Body angular velocity [rad/s] from RigidBodyNode",
        ),
        "body_orientation": BoundaryInputSpec(
            shape=(4,), description="Body orientation quaternion from RigidBodyNode",
        ),
    }
```

### 3.4 Update

```python
def update(self, state, boundary_inputs, dt):
    omega_z = boundary_inputs["body_angular_velocity"][2]  # z-component
    new_angle = state["body_angle"] + omega_z * dt

    # Generate masks at new angle
    solid_mask = create_umr_mask(..., rotation_angle=new_angle)
    umr_missing = compute_missing_mask(solid_mask)
    pipe_missing = compute_missing_mask(pipe_wall)  # static, could be cached

    # LBM step
    f_pre, f_post, rho, u = lbm_step_split(state["f"], tau)

    # Two-pass BB
    f = apply_bounce_back(f_post, f_pre, pipe_missing, ...)  # pipe wall
    if use_bouzidi:
        q_values = compute_q_values_sdf_sparse(umr_missing, sdf_func)
        f = apply_bouzidi_bounce_back(f, f_pre, umr_missing, ..., q_values, ...)
    else:
        f = apply_bounce_back(f, f_pre, umr_missing, ...)

    # Momentum exchange
    force = compute_momentum_exchange_force(f_pre, f, umr_missing)
    torque = compute_momentum_exchange_torque(f_pre, f, umr_missing, center)

    return {
        "f": f,
        "solid_mask": solid_mask,
        "body_angle": new_angle,
        "drag_force": force,
        "drag_torque": torque,
    }
```

### 3.5 Boundary fluxes

```python
def compute_boundary_fluxes(self, state, boundary_inputs, dt):
    return {
        "drag_force": state["drag_force"],    # → RigidBodyNode.drag_force
        "drag_torque": state["drag_torque"],   # → RigidBodyNode.drag_torque
    }
```

### 3.6 Properties

```python
@property
def requires_halo(self) -> bool:
    return True  # LBM streaming accesses spatial neighbors
```

## 4. Bouzidi Q-Value Strategy

### 4.1 Why per-step recomputation is required

For a body rotating at ω = 0.001 rad/step with fin tip radius 29 lu (at 192³), the surface moves 0.029 lu per step. This exceeds the q-value precision needed for O(dx²) Bouzidi accuracy after a single step. Rotating or caching q-values introduces O(dx) errors that degrade Bouzidi to simple BB accuracy — defeating its purpose.

### 4.2 Sparse q-value computation

`compute_q_values_sdf_sparse` reduces the per-step cost by evaluating the SDF bisection only at boundary nodes (~112K at 192³) instead of the full domain (7.1M nodes).

Implementation uses `jnp.nonzero(mm_q, size=MAX_LINKS)` to gather boundary node indices within JAX's static-shape requirements. The boundary link count for a rigid body of fixed shape is approximately constant across rotation angles (~±10%), so a fixed `size=` parameter with 20% padding is safe.

**Expected performance**: ~0.1s per step at 192³ on H100 (vs ~6s for full-domain). Combined with 0.04s LBM step: ~0.14s total per step.

### 4.3 Bisection iteration count

16 iterations give ~10⁻⁵ precision. For Bouzidi with O(dx²) accuracy at 192³ (dx ≈ 0.05 mm), 8 iterations (~10⁻² lu ≈ 0.5 μm) are sufficient. Reducing to 8 iterations halves the sparse q-value cost to ~0.05s per step.

## 5. Coupling Architecture

```
ExternalMagneticFieldNode
    ↓ field_vector, field_gradient
PermanentMagnetResponseNode
    ↓ magnetic_torque
RigidBodyNode ←──────────────── IBLBMFluidNode
    ↓ angular_velocity, orientation    ↑ drag_force, drag_torque
    └──────────────────────────────────┘
```

Coupling mode: `CouplingGroup` with one-step lag (back-edge from IBLBMFluidNode → RigidBodyNode). This is standard IB-LBM practice — the drag from step t drives the orientation at step t+1.

For the confinement sweep (fixed omega, no FSI), the coupling simplifies: angular_velocity is constant, so there is no circular dependency. The IBLBMFluidNode runs standalone.

## 6. Effort Estimate

| Task | Effort |
|---|---|
| `IBLBMFluidNode` class (wrap existing functions) | 2 hours |
| `compute_q_values_sdf_sparse` | 1 hour |
| Wire boundary inputs/outputs + edges | 1 hour |
| Verification: torque matches standalone script | 1 hour |
| Replace `run_confinement_sweep.py` with node graph | 2 hours |
| **Total** | **7 hours** |

## 7. Dependencies

- No MADDENING extensions required
- Requires `compute_q_values_sdf_sparse` (implemented as part of T2.6b fix)
- Existing nodes: `RigidBodyNode`, `PermanentMagnetResponseNode`, `ExternalMagneticFieldNode`
