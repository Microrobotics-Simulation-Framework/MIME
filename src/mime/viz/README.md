# `mime.viz`

**Recording and the visuals the stage draws.**

## What goes here

- **The recording layer** (`_recording.py`): a run's bound poses and field vectors as time samples over its run
  folder's own stage, `recording.usdc`, its sublayer the folder's `experiment/scene/scene.usda` by a path inside the
  folder; written by `run_to(..., stage_every=)` and by a live session's `record`. Its time code is the sample's
  index, its layer data names every sample's step, and it is written as the oldest crate usd-core writes for what it
  holds, which older OpenUSD releases read. `timeCodesPerSecond` and `framesPerSecond` are the samples per simulated
  second, so **within one uninterrupted stretch of samples a time code over `timeCodesPerSecond` is the simulated
  seconds since the layer's first sample**, which is the state where recording began: the run's start for `run_to`, the
  step the `record` command took effect at for a live session (codes give seconds from the run's start only where
  recording began there). A session that turns recording off and on joins its stretches with consecutive codes, so across
  the gap the codes are not seconds and the layer data's steps name each sample's step. Both writers are tested at two
  spacings (`tests/viz/test_recording_time_base.py`).
- **The contract for every overlay.** A saved folder never holds results or presentation: overlays (recordings,
  comparisons, measured tracks, look-dev, previews) live in run or study folders or in the asset cache. Every layer MIME
  composes resolves inside its own folder, with no path out of it, so a folder resolves after it is moved or published.
  A preview baked from an asset is a derived file that carries the asset's licence tier (design #198) and is never
  pinned by its bytes in an experiment's lock (design #183). Root prims other than `/World` are reserved for these
  overlays (see [`mime.scene`](../scene/README.md)).
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
