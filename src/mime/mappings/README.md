# `mime.mappings`

**Transfers between discretisations**, and MIME's mapping and geometry kinds.

## What goes here

- **Gather and scatter.** A gather reads a field at a body's points, for example a flow velocity on a grid sampled at
  a robot's surface. A scatter delivers a body's force back onto the grid. The mapping kind is a saved setting.
- **MIME's mapping and geometry kinds**, with their records.
- **The workarounds for MADDENING 0.4.x** that concern mappings, each written to be removed.

## Boundaries

- The only importer of MADDENING's mapping registry and geometry-edge API.

## MADDENING 0.4's workarounds

The list of what MIME does because MADDENING 0.4.x cannot yet, each with what removes it. A field on a grid reaches the
graph as node state, so the list holds the grid's workarounds too, until they have their own module.

- **A column per MAC component.** A node's state is a flat mapping of arrays, so the flow's staggered velocity is three
  state fields, `velocity_x`, `velocity_y`, `velocity_z` (`mime.nodes.flow.fvm`), each on its own face lattice, and every
  operator takes them as a tuple; on an open axis the normal component has one face more than the others, which the
  separate fields already allow. Removed by a node state that holds a pytree.

## Status

Gather and scatter are empty: they land with the first interaction that transfers between discretisations.

Releases are explained in the [package manifest](../README.md#status).
