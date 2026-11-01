# `mime.viz`

**Recording and the visuals the stage draws.**

## What goes here

- **The recording layer** (`_recording.py`): a run's bound poses and field vectors as time samples over its run
  folder's own stage, `recording.usdc`, its sublayer the folder's `experiment/scene/scene.usda` by a path inside the
  folder; written by `run_to(..., stage_every=)` and by a live session's `record`. Its time code is the sample's
  index, its layer data names every sample's step, and it is written as the oldest crate usd-core writes for what it
  holds, which older OpenUSD releases read.
- **Meshes** (`_meshes.py`): an STL file read (binary or ASCII) and welded, decimated by vertex clustering to a
  triangle budget (`TriangleBudget`, a presentation setting with its basis), its distance to the original measured
  both ways, and written as a USD mesh layer.
- **The visuals in the asset cache** (`_visuals.py`): a visual set (`mime.scene.visual_sets`) fetched from its pinned
  upstream through the asset cache's resolver, each file held to its sha256, baked under `visuals/` in the cache with
  its record; the overlay, written in the cache, that references each baked mesh into the prim naming it and hides
  the stand-in bones; and a view of a stage or a recording with it, for any USD viewer. Nothing of it is written into
  a saved or a run folder.

## Boundaries

- With `scene`, the only importer of `pxr`.
- The desktop application, MICROROBOTICA, is a separate project. It adapts to MIME's interfaces, not the reverse.

## Status

Since dev1's phase 7n (design #197): the recording layer, the mesh importer and the visuals in the asset cache.
Viewports stay with the clients.

Phases and releases are explained in the [package manifest](../README.md#status).
