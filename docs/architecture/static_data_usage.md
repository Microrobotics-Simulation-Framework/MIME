# The `static_data` / `StaticArray` channel

MIME environment nodes carry large, non-evolving arrays — unstructured FVM
meshes, BEM LU factors, lattice wall masks, MLP weights. Before MADDENING
v0.2 these were plain instance attributes closed over by `update()`, which
baked them into the compiled HLO as constants.
v0.2 added the `static_data` channel to hold them outside the JIT closure;
MIME adopted it across its environment nodes in the v0.2 fit-up (§3).

## The problem it solves

A node has two places to put a tensor:

* **`initial_state()`** — the array joins the JAX state pytree and is
  threaded through every `scan` / `fori_loop` / multi-rate step / gradient
  pass, even though it never changes. It is also checkpointed every save.
* **A plain `self._x` attribute** closed over by `update()` — JAX bakes it
  into the compiled HLO as a constant. Correct numerically, but the constant
  is re-embedded each time the step function is traced, and a 1 GB FVM mesh
  baked into HLO bloats compile time and memory.

`static_data` is the third option: a declared channel for arrays that
participate in `update()` but do not evolve. The GraphManager tracks it for
JIT-cache invalidation, keeps it out of the state pytree, and keeps it out
of checkpoints.

## The `StaticArray` wrapper

Array values in `static_data` must be wrapped in
`maddening.core.static_data.StaticArray` — a frozen dataclass that carries
the array alongside its sharding policy:

```python
StaticArray(value, replication="replicate", shard_axis=None)
```

* **`value`** — a NumPy or JAX array (anything with `.shape` and `.dtype`).
  Built once at construction; held by reference, not copied. Do not mutate
  after wrapping.
* **`replication`** — `"replicate"` (default): every device gets the full
  array. `"shard"`: each device gets a slice along `shard_axis`.
* **`shard_axis`** — required when `replication="shard"`, and must be `None`
  otherwise. The GraphManager slices `value` along this axis at sharding
  time.

`StaticArray` is strict by construction. `__post_init__` raises:

* `TypeError` if `value` is not array-like, or is itself a `list` / `tuple`
  / `dict` / `set`. **Nested structures are unsupported** — a pytree of
  arrays cannot be wrapped whole; unfold it into multiple top-level keys.
* `ValueError` if `replication` is unknown, if `shard_axis` is missing for a
  sharded array or set for a replicated one, or if `shard_axis` is out of
  range for the array's shape.

A bare array left in `static_data` is, for now, coerced to
`StaticArray(value=arr)` with a `FutureWarning`. **That coercion path is
removed in v0.3** — bare arrays will raise. Wrap them now.

Scalars, strings, and tuples do *not* need wrapping: they carry no sharding
decision and stay bare in the dict.

## The `static_data` property contract

`SimulationNode.static_data` is a property; the default returns `{}`.
Override it on nodes that need the channel.

* **Keys** are strings.
* **Values** are bare Python scalars / strings / tuples, or
  `StaticArray`-wrapped arrays.
* The returned dict must be **stable across calls** for a given node
  instance. Build it **once in `__init__`** and stash it on `self`; never
  reconstruct it per call. A property that rebuilds the dict — and the
  arrays inside it — every access defeats the channel.

MIME's FVM node is the reference adoption: `__init__` calls a
`_build_static_data()` helper once and stores the result, and the property
is a pure getter.

```python
# src/mime/nodes/environment/fvm/fluid_node.py

def __init__(self, ...):
    ...
    # Built once here; the property below just returns it.
    self._static_data = self._build_static_data()

def _build_static_data(self) -> dict:
    """Unfold the FVMMesh pytree into flat static_data keys.

    StaticArray rejects nested structures, so the mesh — a frozen
    pytree of arrays plus a tuple of BoundaryPatch — cannot be
    wrapped whole. Each leaf array becomes its own top-level key.
    """
    from maddening.core.static_data import StaticArray
    m = self._mesh
    sd = {
        "mesh_owner": StaticArray(m.owner),
        "mesh_neighbour": StaticArray(m.neighbour),
        "mesh_Sf": StaticArray(m.Sf),
        # ... mesh_n / mesh_area / mesh_d / mesh_w / mesh_V / mesh_x
        # Bare scalar / tuple metadata — hashed by repr().
        "N_cells": m.N_cells,
        "dim": m.dim,
        "cartesian_shape": m.cartesian_shape,
    }
    for p in m.patches:
        sd[f"patch_{p.name}_owner"] = StaticArray(p.owner)
        # ... per-patch Sf / n / area / d / face_x
    return sd

@property
def static_data(self) -> dict:
    return self._static_data
```

Every FVM array uses `replication="replicate"`: the face graph mixes
face-indexed (`owner`, `Sf`, …) and cell-indexed (`V`, `x`) arrays on
different axes, so no single `shard_axis` is valid — and the node's
`halo_width()` returns `{0: 1}`, which already blocks MADDENING's pointwise
sharder from sharding it.

## `static_data_hash()`

`SimulationNode.static_data_hash()` produces a stable hash over
`static_data`, used as part of the JIT cache key so a geometry change
triggers a recompile.

For each array value it hashes `(key, shape, dtype, replication,
shard_axis)` — **not the array contents**. `static_data` is expected to be
far larger than the state, so content hashing would be prohibitive; the
assumption is that an array's identity is captured by its shape, dtype, and
sharding policy. Bare scalars hash by `repr(value)`. An empty `static_data`
hashes to `0`.

Because sharding policy is in the key, a node that switches an array from
`"replicate"` to `"shard"` is correctly recognised as a recompile-worthy
change. Conversely, swapping in a *different* array of the same shape and
dtype will **not** invalidate the cache — if a node must change array
contents, it must also change shape/dtype or be reconstructed.

## Checkpoint caveat

**`static_data` is not checkpointed.** The `.npz` manifest carries no record
of it — not the arrays, not the `replication` / `shard_axis` metadata.

The consequence for node authors: the configuration needed to *rebuild* the
static arrays must live in `self.params`, which **is** checkpointed. On a
checkpoint/restore round-trip the node is reconstructed from its persisted
params, and `__init__` rebuilds `static_data` from them.

MIME's LBM node shows the pattern: the pipe-wall occupancy mask and its
D3Q19 missing-link mask are derived in `__init__` from `nx` / `ny` / `nz` /
`vessel_radius_lu`, all of which are passed up to `super().__init__()` as
params. A resumed node recomputes the masks from the persisted params — the
masks themselves never need to survive the checkpoint.

```python
# src/mime/nodes/environment/lbm/fluid_node.py

@property
def static_data(self) -> dict:
    """Non-evolving lattice masks closed over by update().

    Both masks are rebuilt from self.params (nx/ny/nz/
    vessel_radius_lu) in __init__; a checkpoint/restore round-trip
    reconstructs them from the persisted params, since static_data
    itself is not checkpointed.
    """
    from maddening.core.static_data import StaticArray
    return {
        "pipe_wall": StaticArray(self._pipe_wall),
        "pipe_missing": StaticArray(self._pipe_missing),
    }
```

## Per-step closures

A node's `update()` is JIT-compiled once — by the `GraphManager`, as a unit —
not afresh on every simulation step. A `make_*_step()` factory called
*inside* `update()` therefore runs a single time, during that one trace, and
the static arrays it closes over are captured once.

The §3 fit-up initially planned to memoise FVM's `make_piso_step` closure on
`self`, on the theory that rebuilding it per call would re-bake the mesh.
Investigation showed the concern does not apply: `make_piso_step` returns a
plain (un-`jit`'d) closure and is cheap, so `FVMFluidNode` deliberately keeps
the call inside `update()` — the factory runs once per trace, not per step.

The one pitfall to avoid: **never call `jax.jit` inside `update()`**. That
builds a fresh jitted function — and a fresh compilation — on every call,
which *does* defeat `static_data`. Leave the JIT boundary to the
`GraphManager`.

## MIME nodes that adopted the channel

| Node | What moved to `static_data` |
|---|---|
| `FVMFluidNode` | The `FVMMesh` pytree, unfolded into ~30 top-level keys: `mesh_*` interior face graph + cell geometry (`owner`, `neighbour`, `Sf`, `n`, `area`, `d`, `d_mag`, `w`, `V`, `x`, `V_owner`, `V_neighbour`), `patch_<name>_*` per boundary patch, plus bare scalar/tuple metadata (`N_cells`, `N_faces`, `dim`, `cartesian_shape`, …). |
| IBLBM `LBMFluidNode` | Pipe-wall occupancy mask (`pipe_wall`) and its D3Q19 missing-link mask (`pipe_missing`). Previously the pipe wall was threaded through state as `solid_mask`; now only the dynamic UMR portion stays in `update()`. |
| `Stokeslet` | BEM LU factors (`R`, `lu`, `piv`), the body and wall meshes, and the cylinder `WallTable` Green's-function tensor. |
| `MLPResistanceNode` | MLP weight tensors and the input/output normalization statistics. |
| `DefectCorrection` | LU factors and pipe masks. |

The GNN flux corrector is **not applicable** — it is a standalone pytree
utility with no host `SimulationNode`, so it has no `static_data` property
to populate.

---

*This page documents the §3 deliverable of the MIME v0.2 fit-up. Ground
truth: `maddening.core.static_data` (the `StaticArray` dataclass) and
`SimulationNode.static_data` / `static_data_hash()` in
`maddening.core.node`.*
