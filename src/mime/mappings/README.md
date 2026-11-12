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
  separate fields already allow. Removed by a node state that holds a pytree. The refinement patch's edges take one
  component each too (`_workarounds.columns`); removed by per-column kinds.
- **The patch's read points in the patch's frame** (`_workarounds.in_patch_frame`). An edge has one anchor, and the
  patch's read needs two geometries, the patch's origin and the body's points: the patch's node holds the points moved
  into its own frame, and the read's edge anchors there, so its map stays linear with an exact transpose. Removed by a
  mapping anchored at two nodes' geometries.
- **The patch's origin as float state holding whole cells** (`_workarounds.origin_geometry`): a geometry is a float
  array; the origin is rounded back where it is read.

## Status

The exchange law's kernel, `mime.gaussian_mac` (`_gaussian.py`): a gather or a scatter of one face set of the MAC grid
by a Gaussian of a structural width, its weights the grid kind's (`MACGrid.exchange_stencil`), on a geometry edge that
reads the partner's points; in three modes, the kernel's average, its Laplacian and its antisymmetric dipole (a couple's
reading, the fluid's rotation, and its delivery, a vector of three at each point); its first user is the point body
(`Drag.Stokes`). The refinement patch's two transfers, `mime.patch_boundary` (a coarse face set read at the patch's
given values, anchored at its origin) and `mime.patch_read` (the patch's field read at a body's points in its frame),
`_patch.py`: trilinear reads with their exact transposes, the functions the one-pass route calls inside its Krylov
loop (`mime.nodes.flow.fvm._patch_transfers`), registered for the lagged edge route.

Releases are explained in the [package manifest](../README.md#status).
