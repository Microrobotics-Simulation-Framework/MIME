# `mime.mappings`

**Transfers between discretisations**, and MIME's mapping and geometry kinds.

## What goes here

- **Gather and scatter.** A gather reads a field at a body's points, for example a flow velocity on a grid sampled at
  a robot's surface. A scatter delivers a body's force back onto the grid. The mapping kind is a saved setting.
- **MIME's mapping and geometry kinds**, with their records.
- **The workarounds for MADDENING 0.4.x** that concern mappings, each written to be removed.

## Boundaries

- The only importer of MADDENING's mapping registry and geometry-edge API.

## Status

Empty. It lands with the first interaction that transfers between discretisations.

Phases and releases are explained in the [package manifest](../README.md#status).
