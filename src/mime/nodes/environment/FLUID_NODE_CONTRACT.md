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
| `body_*` inputs | ✅ | partial¹ | partial¹ | partial¹ |
| output units | SI | lattice² | SI | SI |
| `halo_width()` | `{0:1}` | `{0:1,1:1,2:1}` | `{}` | `{}` |
| `update_padded()` | — | ✅³ | — | — |

¹ A node declares only the inputs it needs — e.g. the BEM nodes take
`body_velocity` / `body_angular_velocity` / `body_orientation` but not
`body_position`; the LBM node takes `body_angular_velocity` /
`body_orientation`.
² See the lattice-units note above.
³ Implemented for `IBLBMFluidNode` (fit-up §8 Step 5): the multi-device path
is bit-identical to single-device on a 4-device mesh. Sharded Bouzidi IBB is
deferred. See the IB-LBM node guide's *Multi-GPU sharding* section.

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
