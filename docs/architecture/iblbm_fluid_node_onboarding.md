# IBLBMFluidNode — Onboarding Document

*Entry point for implementing the IBLBMFluidNode refactor. Start here.*

## 1. Purpose and Context

`IBLBMFluidNode` wraps the existing LBM fluid solver code (D3Q19 lattice Boltzmann with Bouzidi interpolated bounce-back) as a proper MADDENING `SimulationNode`, replacing the manual Python-loop wiring in `scripts/run_confinement_sweep.py` with a node-graph approach. Once this node exists, it can be wired to `RigidBodyNode` and `PermanentMagnetResponseNode` via `GraphManager` edges, enabling the FSI coupling (T3.C) and emergent step-out demo (T3.D) without any custom simulation loop code. The step-out effect — where the UMR loses synchrony with the rotating magnetic field — will emerge naturally from the node graph.

This work is tracked as T3.0 in `UMR_REPLICATION_PLAN.md`. The full interface specification is in `docs/architecture/iblbm_fluid_node_spec.md`. **No MADDENING extensions are required** — the existing `SimulationNode` interface supports all IBLBMFluidNode requirements, as assessed in a detailed architecture review (Concern 63, 2026-03-24). The LBM code, boundary condition infrastructure, and sparse q-value computation all already exist as tested utility functions — this refactor wraps them, it does not rewrite them.

## 2. Where to Find Everything

### MADDENING framework (sibling repo, editable install)

| File | Description | Relevance |
|------|-------------|-----------|
| `/home/nick/MSF/MADDENING/src/maddening/core/node.py` | `SimulationNode` ABC + `BoundaryInputSpec` dataclass | **The interface contract.** `initial_state()`, `update()`, `boundary_input_spec()`, `compute_boundary_fluxes()`, `requires_halo`, `state_fields()`, `derivatives()`, `implicit_residual()`, `compute_interface_correction()`. Read this entire file first. |
| `/home/nick/MSF/MADDENING/src/maddening/core/graph_manager.py` | `GraphManager` — orchestrates node execution | Key methods: `step()` (line ~1785), `run()` (line ~1809), `run_scan()` (line ~1842), `_build_step_fn()` (line ~1465), `compile()` (line ~1291). State is stored in `self._state: dict[str, dict]` and passed through JIT — XLA manages buffer reuse, no Python copies. |
| `/home/nick/MSF/MADDENING/src/maddening/core/coupling/group.py` | `CouplingGroup` — iterative coupling for circular dependencies | Gauss-Seidel iteration within a timestep. Used when LBM and RigidBody need bidirectional exchange within the same step (FSI mode). For fixed-omega confinement sweep, not needed — one-step lag via back-edges is sufficient. |
| `/home/nick/MSF/MADDENING/src/maddening/core/edge.py` | `EdgeSpec` — connects node outputs to node inputs | Defines `source_node`, `source_field`, `target_node`, `target_field`, `additive`, `transform`. Back-edges use previous-timestep values. |

### MIME node implementations (reference patterns)

| File | Description | Relevance |
|------|-------------|-----------|
| `src/mime/core/node.py` | `MimeNode(SimulationNode, ABC)` — MIME base class | Adds `mime_meta`, `observable_fields()`, `commandable_fields()`, `validate_mime_consistency()`. All MIME nodes inherit from this. |
| `src/mime/nodes/environment/csf_flow.py` | `CSFFlowNode` — analytical Stokes drag | **Closest analogue.** Same role (`ENVIRONMENT`), same output ports (`drag_force`, `drag_torque`), same boundary inputs (`position`, `velocity`, `angular_velocity`). IBLBMFluidNode should be a drop-in replacement for this node with a resolved flow field instead of analytical drag. |
| `src/mime/nodes/robot/rigid_body.py` | `RigidBodyNode` — 6-DOF overdamped dynamics | **The node IBLBMFluidNode exchanges data with.** Outputs `angular_velocity` and `orientation` (consumed by IBLBMFluidNode). Inputs `drag_force` and `drag_torque` (produced by IBLBMFluidNode). Has `use_analytical_drag` flag that switches between self-computed and externally-provided drag — set to `False` when coupled to IBLBMFluidNode. |
| `src/mime/nodes/robot/permanent_magnet_response.py` | `PermanentMagnetResponseNode` — magnetic torque computation | Produces `magnetic_torque` consumed by `RigidBodyNode`. Relevant for T3.C FSI coupling — the node graph is: `ExternalMagneticFieldNode` → `PermanentMagnetResponseNode` → `RigidBodyNode` ↔ `IBLBMFluidNode`. |
| `src/mime/nodes/actuation/external_magnetic_field.py` | `ExternalMagneticFieldNode` — rotating uniform field | Produces `field_vector` and `field_gradient`. Top of the node graph for the UMR simulation. |

### LBM utility modules (code being wrapped)

| File | Description | Relevance |
|------|-------------|-----------|
| `src/mime/nodes/environment/lbm/d3q19.py` | D3Q19 lattice constants, `lbm_step_split()`, `init_equilibrium()`, `compute_macroscopic()` | Core LBM operations. `lbm_step_split()` does collision + streaming and returns `f_pre, f_post, rho, u`. |
| `src/mime/nodes/environment/lbm/bounce_back.py` | `compute_missing_mask()`, `apply_bounce_back()`, `apply_bouzidi_bounce_back()`, `compute_q_values_sdf_sparse()`, `compute_momentum_exchange_force()`, `compute_momentum_exchange_torque()` | All boundary condition functions. `compute_q_values_sdf_sparse` is the sparse Bouzidi q-value implementation used for production. |
| `src/mime/nodes/environment/lbm/rotating_body.py` | `rotating_body_step()`, `run_rotating_body_simulation()` | Self-contained rotating body step function. IBLBMFluidNode's `update()` method will contain similar logic but receive body orientation as a boundary input instead of computing it internally. |
| `src/mime/nodes/environment/lbm/solver.py` | `IBLBMConfig`, `IBLBMState`, `step()`, `run()` | 2D IB-LBM solver (immersed boundary method). Different approach from the 3D bounce-back method used for the UMR. Reference for state management patterns but not directly used by IBLBMFluidNode. |
| `src/mime/nodes/environment/lbm/convergence.py` | `run_to_convergence()` | Convergence monitoring. Not needed inside the node — convergence is monitored externally by the sweep script or GraphManager callback. |
| `src/mime/nodes/robot/helix_geometry.py` | `create_umr_mask()`, `create_umr_mask_sdf()`, `umr_sdf()`, `compute_q_values_sdf()` | UMR geometry generation. `create_umr_mask()` generates the solid mask at a given rotation angle. `umr_sdf()` is the signed distance function for sparse q-value computation. |

### Scripts and existing wiring

| File | Description | Relevance |
|------|-------------|-----------|
| `scripts/run_confinement_sweep.py` | Manual LBM sweep loop (T2.6/T2.6b) | **The code IBLBMFluidNode replaces.** Lines 166-202 contain the per-step loop: mask generation → LBM step → two-pass BB → momentum exchange → convergence check. This loop becomes `IBLBMFluidNode.update()`. The sweep script must continue to work unchanged after the refactor — IBLBMFluidNode is additive. |

### Planning and specification documents

| File | Description | Relevance |
|------|-------------|-----------|
| `docs/architecture/iblbm_fluid_node_spec.md` | Full interface specification | Constructor parameters, state dict, boundary inputs, update() pseudocode, boundary fluxes, coupling architecture, effort estimates. **Read this after the SimulationNode interface.** |
| `UMR_REPLICATION_PLAN.md` | T3.0 (this work), T3.C (FSI coupling), T3.D (step-out demo) | Context for why this node exists and what it enables. |
| `RENDERING_PLAN.md` | T3.C and T3.D rendering requirements | The velocity cross-section visualization (§T3.A) reads from IBLBMFluidNode's velocity field output. |

## 3. Implementation Checklist

Complete in order. Each item is self-contained once its dependencies are met.

| # | Task | File(s) | Description | Depends on | Effort |
|---|------|---------|-------------|------------|--------|
| 1 | Create `IBLBMFluidNode` class skeleton | `src/mime/nodes/environment/lbm/fluid_node.py` (new) | Class with `meta`, `mime_meta`, `__init__`, `initial_state()`, `boundary_input_spec()`, `requires_halo`. No `update()` yet — just the shell with correct metadata, constructor parameters, and state shape. Follow the `CSFFlowNode` pattern exactly. | None | 30 min |
| 2 | Implement `update()` | `src/mime/nodes/environment/lbm/fluid_node.py` | Port the step loop from `run_confinement_sweep.py` lines 166-202 into `update()`. Receives `body_angular_velocity` as boundary input. Returns new state dict with `f`, `solid_mask`, `body_angle`, `drag_force`, `drag_torque`. Use `compute_q_values_sdf_sparse` when `use_bouzidi=True`. | #1 | 1 hour |
| 3 | Implement `compute_boundary_fluxes()` | `src/mime/nodes/environment/lbm/fluid_node.py` | Return `drag_force` and `drag_torque` from state. Identical pattern to `CSFFlowNode.compute_boundary_fluxes()`. | #2 | 15 min |
| 4 | Register in `__init__.py` | `src/mime/nodes/environment/lbm/__init__.py` | Add `IBLBMFluidNode` to the module's public API. | #3 | 5 min |
| 5 | Unit test: `update()` matches `rotating_body_step()` | `tests/verification/test_iblbm_fluid_node.py` (new) | Call `IBLBMFluidNode.update()` with a known initial state and fixed angular velocity. Call `rotating_body_step()` with identical inputs. Assert `f` arrays match within float32 tolerance. Assert drag force/torque match within 0.1%. | #3 | 1 hour |
| 6 | Integration test: `GraphManager` wiring | `tests/verification/test_iblbm_fluid_node.py` | Wire `IBLBMFluidNode` + `RigidBodyNode` + `ExternalMagneticFieldNode` + `PermanentMagnetResponseNode` via `GraphManager`. Run 100 steps. Confirm drag torque is non-zero and orientation changes. | #5 | 1 hour |
| 7 | Regression test: confinement sweep via node graph | `tests/verification/test_iblbm_fluid_node.py` | Run a short confinement simulation (64³, ratio 0.30, 200 steps) via `GraphManager.run()` with `IBLBMFluidNode`. Compare drag torque against the T2.6b Bouzidi result at the same resolution. Must match within 0.1%. | #6 | 1 hour |
| 8 | CSFFlowNode interchangeability test | `tests/verification/test_iblbm_fluid_node.py` | Build two `GraphManager` instances with identical structure except one uses `CSFFlowNode` and the other uses `IBLBMFluidNode`. Both must be runnable — no crashes, no missing edges. The drag values will differ (analytical vs resolved) but the graph structure must be compatible. | #6 | 30 min |
| 9 | Documentation: update `__init__.py` docstring | `src/mime/nodes/environment/lbm/__init__.py` | Add `IBLBMFluidNode` to the module docstring listing. | #4 | 5 min |

**Total: ~7 hours.**

## 4. Key Design Decisions Already Made

These were carefully reasoned in previous sessions. Do not revisit without strong justification.

### No MADDENING extensions required
The `SimulationNode` interface was assessed against six requirements (large state, variable geometry, quaternion inputs, circular dependencies, spatial stencil, JIT compilation) and found sufficient for all of them. XLA manages the 1.3 GB f-array via buffer reuse — no Python-level copies. The `requires_halo` property was designed for LBM. `CouplingGroup` handles bidirectional coupling. See the full assessment in `docs/architecture/iblbm_fluid_node_spec.md` §2.

### One-step lag for LBM ↔ RigidBody coupling
Standard IB-LBM practice. The drag torque from step t drives the angular velocity at step t+1. This is implemented via a back-edge in the `GraphManager` (detected automatically in `validate()`, line ~1280 of `graph_manager.py`). For the confinement sweep (fixed omega), there is no circular dependency at all — IBLBMFluidNode runs standalone.

### Per-step sparse q-value recomputation
The hybrid rotate+recompute approach was rejected (Concern 62 analysis, 2026-03-24). At ω = 0.001 rad/step and fin tip radius 29 lu, the surface moves 0.029 lu per step — enough to degrade Bouzidi from O(dx²) to O(dx) after a single step. Q-values must be recomputed every step. `compute_q_values_sdf_sparse` makes this feasible by evaluating only ~112K boundary nodes instead of 7.1M full domain (63× reduction).

### `run()` path, not `run_scan()`
`GraphManager.run()` uses a Python loop calling `_compiled_step` once per step. This is the correct execution path because the geometry generation (`create_umr_mask`) and sparse q-value computation (`compute_q_values_sdf_sparse`) use Python for-loops over 19 directions that JAX unrolls during tracing. `run_scan()` would require the full step to be scannable, which it is in principle (the geometry ops are JAX-traceable) but compilation time would be prohibitive. The Python-loop path is already performant: measured 0.04s/step LBM + ~0.1s/step sparse q-values at 192³ on H100.

### Solid mask in state dict
The UMR solid mask at 192³ is a boolean array of ~7 MB. Including it in the state dict is trivially practical — the f-array is 1.3 GB. The mask is regenerated every step in `update()` from the current `body_angle`. It is included in state for observability (rendering, debugging) and because it is part of the simulation state.

### `requires_halo = True`
LBM streaming accesses spatial neighbors (lattice velocities shift populations between adjacent nodes). This property must return `True` for IBLBMFluidNode. Currently, MADDENING's `ShardedNode` refuses to shard nodes with `requires_halo = True` — this is correct and means IBLBMFluidNode runs on a single device. Multi-GPU support would require halo exchange, which is out of scope.

## 5. Known Risks and Things to Watch Out For

### JIT compilation time
The first call to `_compiled_step` with IBLBMFluidNode will trigger JAX tracing. The 19-direction loops in `compute_missing_mask`, `compute_q_values_sdf_sparse`, and `apply_bouzidi_bounce_back` are Python for-loops that JAX unrolls into a flat XLA computation graph. At 192³, expect **30–60 seconds for the first step** (compilation), then ~0.14s per step thereafter. This is normal and matches the current `run_confinement_sweep.py` behavior.

### MAX_LINKS constant in `compute_q_values_sdf_sparse`
The `max_boundary_links_per_dir` parameter controls the fixed pad size for `jnp.nonzero`. It auto-computes as 1.5× the maximum boundary count across the 19 directions for the current mask. **If the UMR geometry changes** (different fin count, different body radius), the boundary link count changes. The 1.5× margin handles angle-to-angle variation for a fixed geometry, but a new geometry needs a fresh measurement. To measure: call `jnp.sum(missing_mask, axis=(1,2,3))` at several angles and take the maximum.

If the pad size is too small, `jnp.nonzero` silently truncates boundary links. The q-values for truncated links default to 0.5 (simple BB behavior). This degrades accuracy without any error message. **Always verify boundary link counts after geometry changes.**

### Padding mask correctness
In `compute_q_values_sdf_sparse`, after bisection, the `is_real` mask (based on `entry_idx < actual_count`) must be applied before scattering results back into the full q-values array. Padding entries undergo bisection on garbage coordinates and produce garbage q-values. These are masked to 0.5 before scatter. If this mask is removed or bypassed, the garbage values corrupt the Bouzidi computation silently — the simulation runs but produces wrong drag values. **Do not remove the `is_real` masking logic.**

### Circular dependency timing in FSI mode
For the confinement sweep (T2.6b), angular velocity is constant — no circular dependency. For FSI (T3.C), the LBM needs angular velocity from RigidBodyNode, and RigidBodyNode needs drag torque from LBM. The one-step lag is standard and acceptable for slowly varying flows (Stokes regime). However, near step-out (where the UMR transitions between synchronous and asynchronous rotation), the dynamics change rapidly. If the one-step lag causes visible artifacts at high field frequencies, switch to a `CouplingGroup` with Gauss-Seidel sub-stepping (`max_iterations=3`). Test by comparing step-out frequency with and without the coupling group — they should agree within 1%.

### Backward compatibility
`run_confinement_sweep.py` must continue to work unchanged after the refactor. IBLBMFluidNode is a new node class — it does not modify any existing code. The sweep script uses the utility functions directly (not through a node), so it is unaffected. Do not refactor the utility functions in a way that changes their signatures or behavior.

## 6. Verification Plan

### Unit test (checklist item #5)
```python
def test_iblbm_matches_rotating_body_step():
    """IBLBMFluidNode.update() produces identical results to rotating_body_step()."""
    # Setup: 32^3, tau=0.8, ratio=0.30, omega=0.003, use_bouzidi=False
    node = IBLBMFluidNode(name="lbm", timestep=1.0, nx=32, ny=32, nz=32, ...)
    state = node.initial_state()
    bi = {"body_angular_velocity": jnp.array([0, 0, 0.003])}
    new_state = node.update(state, bi, 1.0)

    # Compare against rotating_body_step
    f_ref, force_ref, torque_ref, _, _ = rotating_body_step(
        state["f"], tau=0.8, angular_velocity=0.003, dt_lbm=1.0,
        current_angle=0.0, geometry_params=..., center=..., use_bouzidi=False,
    )
    assert jnp.allclose(new_state["f"], f_ref, atol=1e-6)
    assert jnp.allclose(new_state["drag_torque"], torque_ref, atol=1e-4)
```

### Integration test (checklist item #6)
Wire all 4 nodes via `GraphManager`, add edges, run 100 steps. Assert:
- `drag_torque` is non-zero after step 10 (fluid has developed)
- `orientation` has changed from initial (body is rotating)
- No NaN or Inf in any state

### Regression test (checklist item #7)
Run 200-step simulation at 64³, ratio 0.30, with Bouzidi. Compare mean drag torque against the T2.6b sanity test value (21.3916 lu at 64³ with sparse Bouzidi, 8 bisection iterations). Must match within 0.1%.

### Interchangeability test (checklist item #8)
```python
def test_csf_and_iblbm_are_interchangeable():
    """Both CSFFlowNode and IBLBMFluidNode work as environment nodes."""
    for NodeClass in [CSFFlowNode, IBLBMFluidNode]:
        gm = GraphManager()
        gm.add_node(NodeClass(name="fluid", ...))
        gm.add_node(RigidBodyNode(name="body", ..., use_analytical_drag=False))
        gm.add_edge("fluid", "drag_force", "body", "drag_force")
        gm.add_edge("fluid", "drag_torque", "body", "drag_torque")
        gm.compile()  # Must not raise
        gm.step()     # Must not crash
```

## 7. How This Enables Tier 3

Once `IBLBMFluidNode` is a proper MADDENING node, the FSI demo (T3.C) requires zero custom simulation code. The node graph:

```
ExternalMagneticFieldNode → PermanentMagnetResponseNode → RigidBodyNode ↔ IBLBMFluidNode
```

produces the step-out effect naturally: as field frequency increases, the magnetic torque and viscous drag torque compete. Below step-out, the UMR locks to the field. Above step-out, the drag exceeds the magnetic torque and the UMR falls out of synchrony — angular velocity oscillates, flow field becomes unsteady. This transition is visible in the USD scene (T3.D) as a dramatic change from smooth rotation to chaotic tumbling.

The emergent step-out demo (T3.D in `UMR_REPLICATION_PLAN.md`) runs this graph at 64³ in real-time (~2 fps with HydraStorm rendering), streamed via Selkies. The user adjusts field frequency and watches the step-out transition happen live. The parameter panel (T3.B) overlays the precomputed predictions from T2.7. See `UMR_REPLICATION_PLAN.md` T3.C and T3.D, and `RENDERING_PLAN.md` §[UMR ADDITION] for rendering details.
