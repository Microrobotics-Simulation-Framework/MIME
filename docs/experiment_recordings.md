# Experiment recordings — geometry budget

Experiment recordings (`experiments/<name>/output/recording.usdc`) are
self-contained USD files: `scripts/ar4_record_usdc.py` runs the experiment,
captures per-frame actor poses, and **flattens** the scene so the recording
embeds its mesh geometry directly (no external references). MICROROBOTICA
opens these in its "open recording" playback mode.

## The geometric-aliasing trap

Mesh assets imported from CAD/STL arrive at full manufacturing tessellation —
the AR4 arm, for example, was ~1,000,000 triangles. At that density, in a
normal viewport, **most triangles are smaller than a pixel**. Sub-pixel
triangles cannot be rasterised stably: which one "wins" a given pixel flips
from frame to frame as the camera moves. The result is a shimmer — a
"jitter" on the geometry during camera motion.

This is **geometric aliasing**: a sampling-rate problem (triangle detail
exceeds the pixel sampling rate), not a viewer bug. A faster GPU does not
help — it is render *quality*, not speed. It is the same class of problem
that level-of-detail (LOD) systems exist to solve.

Keep recording geometry viewer-friendly:

> **Rule of thumb:** ≲ 100k triangles total per recording, and ≲ ~20–30k per
> mesh. Past that, triangles begin to go sub-pixel and shimmer.

## How geometry is kept in budget

- **`experiments/ar4_helical_drive/scene/_bake_meshes.py`** bakes the AR4
  STLs into `assets/ar4_meshes.usdc` and **decimates each mesh** to
  ≤ `TARGET_FACES` triangles (VTK quadric decimation) as part of baking.
  `world.usda` references that asset and `ar4_record_usdc.py` flattens it
  into the recording — so recordings come out viewer-friendly by default.

- **`scripts/decimate_ar4_meshes.py`** decimates the meshes of an existing
  USD file in place — for a recording or asset that is already heavy (e.g.
  one produced before decimation was wired into the baker). `TARGET_FACES`
  at the top of either script is the quality knob.

## Gotcha: `.usdc` crate files do not shrink on `Save()`

`Usd.Stage.Save()` on a binary `.usdc` crate file appends new data and
leaves the superseded data as dead bytes — so a decimated file can stay
large on disk even though its live content is small. To compact it,
re-export the layer fresh:

```python
stage.GetRootLayer().Export("compacted.usdc")
```

`scripts/decimate_ar4_meshes.py` already accounts for this.
