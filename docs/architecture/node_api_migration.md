# Node API migration: `requires_halo` → `halo_width()`

MADDENING v0.2 changed how a `SimulationNode` declares its ghost-cell
requirements. The v0.1 boolean property `requires_halo` was replaced by a
`halo_width()` method that returns a per-axis halo width. MIME completed this
migration in the v0.2 fit-up (§1); this page records the new contract and how
to migrate a custom node.

## What changed, and why

v0.1 nodes declared a single boolean:

```python
@property
def requires_halo(self) -> bool:
    return True          # "this node reads neighbouring cells"
```

A bare boolean is not enough for v0.2's sharding analysis. To decide whether a
spatially-resolved node can be split across devices — and how wide a halo each
device must exchange — MADDENING needs the halo width *per spatial axis*, not
just "yes / no". A node may need a one-cell halo on the streaming axes but none
on others; a boolean cannot express that.

v0.2 therefore replaces the property with a method:

```python
def halo_width(self) -> dict[int, int]:
    return {0: 1, 1: 1, 2: 1}
```

## The `halo_width()` contract

`halo_width()` returns `dict[int, int]` — a mapping from **spatial-axis index**
to **halo width in cells**:

* An entry `{axis: width}` means `update()` reads `width` cells beyond the
  shard boundary along that axis.
* An **empty dict `{}`** means the node needs no halo: it is pointwise and
  shardable. MADDENING's pointwise sharder will shard it freely.
* A **non-empty dict** marks the node as non-pointwise. MADDENING's pointwise
  sharder refuses to shard it; it runs replicated on a single device unless a
  stencil-aware sharder is available (v0.2 does not yet provide one).

`MimeNode` supplies the default `return {}` — most MIME nodes are pointwise
(rigid body, ODE-based actuation, BEM) and inherit it. Spatially-resolved nodes
override.

## What each MIME node returns

| Node | `halo_width()` returns | Reason |
|---|---|---|
| `IBLBMFluidNode` | `{0: 1, 1: 1, 2: 1}` | D3Q19 streaming reads one neighbour per spatial axis. |
| `FVMFluidNode` | `{0: 1}` | Cell-centred + face stencil needs one-cell connectivity; FVM state is `(N_cells, …)`, so the per-axis concept maps only loosely — a non-empty dict simply marks the node non-pointwise. |
| `StokesletFluidNode` | `{}` (inherited default) | Regularised-Stokeslet BEM; dense boundary-integral solve, no spatial stencil. |
| Pointwise nodes — `RigidBodyNode`, magnetic actuation, motor, `MLPResistanceNode`, … | `{}` (inherited default) | Pointwise / ODE-based; no ghost cells. |

A non-empty `halo_width()` is the single signal MADDENING uses to keep the LBM
and FVM nodes off the pointwise sharder — see
[`v0.2_optional_features.md`](v0.2_optional_features.md) on `validate_sharding()`.

## Migrating a custom node

A spatially-resolved node that subclasses a class still defining the old
property now emits a `FutureWarning`. To migrate, delete the property and add a
method.

**Before (v0.1):**

```python
class MyStencilNode(MimeNode):

    @property
    def requires_halo(self) -> bool:
        return True
```

**After (v0.2):**

```python
class MyStencilNode(MimeNode):

    def halo_width(self) -> dict[int, int]:
        """One-cell stencil on all three spatial axes."""
        return {0: 1, 1: 1, 2: 1}
```

A pointwise node needs no change at all: drop the old `requires_halo = False`
property entirely and inherit the `{}` default from `MimeNode`.

## Deprecation timeline

* **v0.2 (now)** — `halo_width()` is the contract. MADDENING still derives a
  `requires_halo` property from it for backward compatibility. Subclassing a
  node that *defines* the old `requires_halo` property emits a
  `FutureWarning`.
* **v0.3** — the `FutureWarning` escalates to a `MigrationError`; the derived
  compatibility property is removed. A node defining `requires_halo` will fail
  to import.

Migrate any remaining custom nodes before adopting MADDENING v0.3.
