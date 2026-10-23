# `mime.core`

**What every node imports.**

## What goes here

- **`MimeNode`**, the base class of MIME's nodes, on MADDENING's `SimulationNode`.
- **Node metadata.**
- **Quaternion helpers**, with a norm that is safe at zero rotation. The 0.2 version produced a NaN there.
- **Typed node state**: `NodeFields`, `typed_view` and `node_fields`, the static views by which a node types its
  state and inputs inside MADDENING's signatures. The convention is set out in
  [`nodes/README.md`](../nodes/README.md#typed-node-state).

## Status

Since phase 4a: the typed node state helpers.

Since phase 4b:
- **`MimeNode`**: every constructor argument kept in `params`; `constants(params)` gives what a step reads, the
  graph's live leaves over the constructor's; a node writes its step as `advance(p, state, inputs, dt)`, and
  `update` runs it compiled once per node, so the verification battery's eager calls are one call each and the
  graph's trace inlines it; `node_input` declares an input with its units; `run_dtype` and `as_run_array`;
- **`quaternion`**: products, rotation matrices and `exponential`, whose value and derivatives are finite at a zero
  rotation (0.2 divided by `|omega|` and differentiated to NaN at rest);
- **shapes** (`Shape`, the protocol a geometry kind's surfaces are evaluated with: signed distance, outward normal,
  the axis where there is one), `Cylinder`, `register_shape` and `shape_named`; and the confinements' velocity rules
  (`VELOCITY_RULES`: `remove_outward`). Node metadata (MADDENING's `NodeMeta`) is not carried over from 0.2.

Phases and releases are explained in the [package manifest](../README.md#status).
