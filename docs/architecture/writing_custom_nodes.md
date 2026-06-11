# Writing a custom MIME node

This is the general guide for adding a physics node to MIME. For the
LBM-specific walkthrough see `iblbm_fluid_node_onboarding.md`.

## The subclassing contract

Every MIME physics node subclasses `MimeNode` (`mime/core/node.py`), which
in turn extends MADDENING's `SimulationNode`. A node is a *descriptor*: it
carries metadata and parameters and exposes pure functions. **Nodes never
store mutable simulation state** — all state lives in the `GraphManager`.

`MimeNode` adds two domain concerns on top of `SimulationNode`:

* `meta` (a `NodeMeta`) — consumed by MADDENING's compliance harvester.
* `mime_meta` (a `MimeNodeMeta`) — consumed by MIME tooling and the
  MICROROBOTICA registry.

Both are required `ClassVar`s.

## Methods to implement

```python
from mime.core.node import MimeNode

class MyNode(MimeNode):
    meta = NodeMeta(...)
    mime_meta = MimeNodeMeta(...)

    def __init__(self, name, timestep, **kwargs):
        super().__init__(name, timestep, **kwargs)
        # stash parameters; build any static data once, here

    def initial_state(self) -> dict:
        # dict of JAX arrays — the node's evolving state
        ...

    def update(self, state, boundary_inputs, dt) -> dict:
        # pure, JAX-traceable function -> new state dict
        ...
```

* **`__init__`** — call `super().__init__(name, timestep, **kwargs)`.
  Validate and store parameters; this is where any static data is built.
* **`initial_state()`** — returns a dict of JAX arrays. These keys are
  the node's `state_fields()` and define its checkpointed state.
* **`update(state, boundary_inputs, dt)`** — the time step. Must be a
  **pure function** suitable for JAX tracing: no Python side-effects, and
  use `jnp.where` instead of `if` for value-dependent branching.
* **`boundary_input_spec()`** — declare each expected boundary input with
  a `BoundaryInputSpec` (shape, dtype, default, `expected_units`). This
  drives [edge validation](edge_validation.md); keep it complete.
* **`compute_boundary_fluxes()`** — override if the node exposes outputs
  (forces, fields) that are not part of its state.

A minimal worked example is
`mime/nodes/actuation/external_magnetic_field.py`.

## v0.2 considerations

* **`halo_width()`** — declare the per-axis ghost-cell width the node's
  `update()` needs. Pointwise nodes (rigid body, ODE actuation) inherit
  the empty default `{}`; stencil nodes (LBM, FVM, diffusion) override.
  See `node_api_migration.md`.
* **`static_data`** — the optional channel for large non-evolving arrays
  (meshes, wall masks, lookup tables) that `update()` closes over but
  that must stay out of the JAX state pytree. See `static_data_usage.md`.

## Compliance metadata

Set both `ClassVar`s:

* `meta` — a `NodeMeta` with `algorithm_id`, version, stability,
  governing equations, assumptions, limitations, validated regimes, and
  `implementation_map`.
* `mime_meta` — a `MimeNodeMeta` with the node's `NodeRole` and any
  role-specific block (`ActuationMeta`, `SensingMeta`,
  `BiocompatibilityMeta`, etc.).

`MimeNode.validate_mime_consistency()` cross-checks these: it requires
both to be set, requires the role-specific block its `NodeRole` demands,
and verifies that `commandable_fields` is a subset of
`boundary_input_spec()` keys and `observable_fields` a subset of
`state_fields()`. Run it once your node is wired up.

Decorate the class with `@stability(StabilityLevel.EXPERIMENTAL)` (or the
appropriate level) to register its maturity.

## Testing pattern

A new node carries three tiers of test:

* **Unit** — `initial_state()` shapes/dtypes, a single `update()` step,
  and `validate_mime_consistency()` returning no errors.
* **Integration** — the node wired into a small `GraphManager`, with
  `gm.validate()` clean of edge issues (see
  [edge validation](edge_validation.md)).
* **Verification** — physics correctness against an analytical or
  reference solution, under `tests/verification/`, following the node
  documentation template `docs/algorithm_guide/nodes/_template.md`.
