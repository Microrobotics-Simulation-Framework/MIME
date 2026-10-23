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

The typed node state helpers landed in dev1's phase 4a. The rest is ported in phase 4.

Phases and releases are explained in the [package manifest](../README.md#status).
