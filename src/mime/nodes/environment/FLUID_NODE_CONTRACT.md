# Shared fluid-node contract

MIME has four environment fluid nodes — `FVMFluidNode`, `IBLBMFluidNode`,
`StokesletFluidNode`, `DefectCorrectionFluidNode`. Each computes the
hydrodynamic load on an immersed rigid body by a different method (finite
volume + IBM; lattice Boltzmann; regularised-Stokeslet BEM; BEM/LBM defect
correction). This document defines the **interface they share** so one node can
be swapped for another in an experiment graph.

For the FVM node's method-specific details (lifting / homogenisation, the
`force_method` backends) see `fvm/FLUID_NODE_CONTRACT.md`.

## The interchangeable interface (single immersed body)

A fluid node coupled to one rigid body exposes:

### Boundary inputs — read from the coupled rigid-body node

| Input | Shape | Units | Meaning |
|---|---|---|---|
| `body_position` | `(dim,)` | m | body centroid |
| `body_velocity` | `(3,)` | m/s | body linear velocity |
| `body_angular_velocity` | `(3,)` | rad/s | body angular velocity |
| `body_orientation` | `(4,)` | — | body orientation quaternion `(w,x,y,z)` |

A node declares (via `boundary_input_spec()`) only the inputs its physics
needs; every per-body input it *does* declare uses these names.

### Output fields — read by downstream edges

| Field | Shape | Units | Meaning |
|---|---|---|---|
| `drag_force` | `(3,)` | N | hydrodynamic force on the body |
| `drag_torque` | `(3,)` | N·m | hydrodynamic torque on the body |

Forces/torques are **SI**, with one documented exception: `IBLBMFluidNode`
computes in lattice units, so a graph wiring its `drag_force`/`drag_torque`
carries the `lbm_to_si_force` / `lbm_to_si_torque` edge transforms (see
`lbm/fluid_node.py::make_iblbm_rigid_body_edges`). The field *names* are still
the contract names, so the graph stays structurally swappable — only the LBM
edges carry a transform.

> **Sign convention.** The contract *defines* `drag_force` / `drag_torque` as
> the hydrodynamic load **on the body** (opposing the body's motion).
> `FVMFluidNode` follows this. **`IBLBMFluidNode` and the standalone
> `StokesletFluidNode` instead report `+R·(motion)`** — the *reaction* (force
> on the fluid), the negative of the force on the body. Delivered as-is to a
> body that adds it, the reaction sign is **anti-dissipative** — verified: a
> de-Boer UMR diverges when fed IBLBM's `+R·ω` with the `omega_max` clamp
> removed (its bounded behaviour relies on that clamp).
>
> The **EffectModel `HydrodynamicModel` adapter normalizes every backend to
> the on-body convention** (`native_drag_sign`: IBLBM / Stokeslet flip, FVM
> unchanged), so a swapped backend always delivers dissipative drag —
> asserted by `tests/verification/test_effectmodel_stokes_drag_swap.py`
> (`test_backends_deliver_force_on_body`). Two follow-ups remain: (a)
> reconciling the sign **at the node level** (so non-adapter consumers like
> the hand-built de Boer graph don't depend on the `omega_max` clamp), and
> (b) confirming `DefectCorrectionFluidNode`'s sign on first real use (heavy
> LBM spin-up; not in the runnable concept-proof). The contract sign itself is
> now settled.

### Methods

* `__init__(self, name, timestep, ...)` — node-specific construction params.
* `initial_state(self) -> dict` — the JAX-array state pytree.
* `update(self, state, boundary_inputs, dt) -> dict` — one pure,
  JAX-traceable step.
* `boundary_input_spec(self) -> dict` — declares the boundary inputs above.
* `compute_boundary_fluxes(self, state) -> dict` — exposes `drag_force` /
  `drag_torque` (and any node extras) as readable output fields.

### Sharding extension points (v0.2)

* `halo_width(self) -> dict[int, int]` — empty `{}` = pointwise (shardable by
  MADDENING's pointwise sharder); non-empty = a stencil node, shardable via
  `ShardedStencilNode`. See `docs/architecture/node_api_migration.md`.
* `update_padded(self, state_padded, boundary_inputs, dt) -> dict` — optional;
  the halo-aware variant of `update()` that a `ShardedStencilNode` calls. Only
  stencil nodes that support multi-GPU sharding implement it.

## Per-node conformance

| | FVM | IBLBM | Stokeslet | DefectCorrection |
|---|---|---|---|---|
| base class | `MimeNode` | `MimeNode` | `SimulationNode` | `SimulationNode` |
| `drag_force` / `drag_torque` out | ✅ | ✅ | ✅ | ✅ |
| `body_*` inputs | ✅ | ✅⁴ | partial¹ | partial¹ |
| output units | SI | lattice² | SI | SI |
| `halo_width()` | `{0:1}` | `{0:1,1:1,2:1}` | `{}` | `{}` |
| `update_padded()` | — | ✅³ | — | — |

¹ A node declares only the inputs it needs — e.g. the BEM nodes take
`body_velocity` / `body_angular_velocity` / `body_orientation` but not
`body_position`.

⁴ As of MIME-NODE-010 v1.1.0 the LBM node declares **all four** `body_*`
inputs. Two caveats, both deliberate:

* `body_position`'s default is the body's **reference position** (the grid
  centre), not the origin — an unwired graph pins the body where the
  pre-v1.1.0 node froze it, which is what keeps `umr_confinement` unchanged.
  `make_iblbm_rigid_body_edges` therefore only wires `body_position` /
  `body_velocity` when passed `wire_translation=True`.
* `body_orientation` is *declared and plumbed to the geometry hook* but no
  shipped geometry reads it (the UMR mask uses the node's integrated z-angle;
  the sphere is symmetric).
² See the lattice-units note above.
³ Implemented for `IBLBMFluidNode` (fit-up §8 Step 5): the multi-device path
is bit-identical to single-device on a 4-device mesh. Sharded Bouzidi IBB is
deferred. See the IB-LBM node guide's *Multi-GPU sharding* section.

## Body geometry (FVM + IBLBM)

Both volumetric solvers rebuild the immersed body **inside `update()`** from
the pose inputs, via a factory, so the mask/SDF is differentiable in — and
traceable through — the pose:

| | FVM | IBLBM |
|---|---|---|
| factory | `BodyFactory = (inputs) -> IBMBody` | `LatticeBodyFactory = (pose, LatticeGrid) -> LatticeBody` |
| body carries | `sdf`, `u_body`, `ref_point` | `mask`, `sdf` (Bouzidi only), `ref_point`, `velocity`, `angular_velocity` |
| units | SI | lattice |
| sphere helper | `fvm.fluid_node.make_sphere_body_factory` | `lbm.body.make_sphere_body_factory` |

The dataclasses are **parallel, not shared**, because the solvers need
different primitives: FVM's diffuse-penalty IBM only samples `sdf(x)` at cell
centres (an SDF is the whole body), whereas LBM bounce-back is defined by which
*neighbouring lattice node* is solid and so needs a boolean occupancy mask —
and the validated UMR mask is an analytic region test that is **not** the
zero-set of `umr_sdf`, so an SDF-only body would silently change
`umr_confinement`. The sphere SDF itself *is* shared (`lbm.body` imports
`fvm.sdf.sphere_sdf`).

## Multi-body extension (FVM)

`FVMFluidNode` additionally supports **multiple** immersed bodies, constructed
via `dynamic_body_factories`. With more than one body it also exposes per-body
`force_<name>` / `torque_<name>` outputs and `<name>_position` / … inputs. The
single-body `drag_force` / `drag_torque` interface above is the interchangeable
subset; multi-body wiring is FVM-specific.

## Known cleanup items

* `StokesletFluidNode` and `DefectCorrectionFluidNode` extend `SimulationNode`
  directly rather than `MimeNode`, and carry no `NodeMeta` / `MimeNodeMeta`.
  This does not block graph wiring (both are `SimulationNode`s) but is a
  consistency gap — they should gain the MIME compliance metadata.
* The `IBLBMFluidNode` lattice-units output could be converted to SI inside the
  node so all four nodes' edges are transform-free; deferred, as it changes the
  `umr_confinement` experiment's edge wiring.
