# Recordings and the stage's visuals

A recording is a run's poses and field vectors as time samples over the stage of the experiment it ran, so a run can
be replayed in any USD viewer, scrubbed, and read back. It lives in the run's folder, beside its outputs, and is its
experiment's by construction.

## Writing one

```python
written = built.run_to("a0_run", s(2), stage_every=ms(16.5))   # a sample every 33 base steps
written.recording                                              # a0_run/recording.usdc
```

```sh
mime run a0_saved --duration 2 --out a0_run --stage-every 0.0165
```

A live session records between `session.record(True)` and `session.record(False)` (the `record` command), a sample
at each chunk's end ([live sessions](live.md)). `stage_every` is a whole number of the run's chunks: the run is
chunked for it, which changes nothing it computes, and keeps no step's state (no `history=True`).

## What it holds

- **`recording.usdc`**, in the run folder. Its one sublayer is the folder's own `experiment/scene/scene.usda`, by a
  path inside the folder, so the folder still resolves when it is moved or copied to another machine.
- **Overs on the bound prims** (`built.binding()`): a body's and each chain frame's `translate` and `orient`, in double
  precision, so a single-precision state's values are held exactly; a field vector's `mime:vector:value`, with its
  arrow placed at the body, turned along the vector and scaled to its length. A `translate` is in the stage's units:
  the state's metres divided by the stage's `metersPerUnit`, so that composed with it (as a viewer composes it) a
  prim is where the run put it, on a stage in millimetres as on one in metres.
- **Its time code is the sample's index.** `timeCodesPerSecond` and `framesPerSecond` are the samples per simulated
  second; `customLayerData` holds `mime:recording`, with the experiment's name and input hash, the base step, the
  spacing, and every sample's step.
- **Crate 0.8.0**, the oldest version usd-core writes for what it holds, which older OpenUSD releases (MICROROBOTICA's)
  read; exported fresh, with no dead bytes of an earlier save.

Its poses are the run's states at its samples' steps exactly, it opens with no finding of usd-core's validators
(`mime.scene.check_stage`), and a moved folder composes: each is a test.

## The stage it is over

The stage (`scene/scene.usda` in a saved folder, `experiment/scene/scene.usda` in a run folder) is written from the
experiment's declarations alone, the same bytes for the same declarations. For a live client and a recording it holds:

- **A prim for every frame of every chain segment**, below the segment's prim, with its frame's name as `mime:frame`
  (a name USD does not take as a prim's is written with each such character `_`). Each is placed in the world, its
  transform stack reset, at its pose at the start: an arm's links, a motor's base and rotor.
- **A prim for every field a law reads at a body** (a flux density), below its apparatus: an arrow, hidden until a
  recording or a client gives it a value.
- **What each frame draws** below it: its description's visuals (`visuals/`), a box, cylinder or sphere as its USD
  primitive, a mesh as a prim naming its visual set and the sha256 of its source; and the simple shapes that stand in
  for meshes the stage does not hold (`fallback/`): a bone from the frame's origin to each child's, and a link's
  collision primitives.
- **Drawn proxies within a triangle budget**: a geometry kind with no USD primitive (the screw) is drawn at fewer facets
  a turn where its drawing would hold more triangles than the budget (`mime.build._stage.PROXY_TRIANGLES`), which kept
  the screw's proxy from being nearly all of every run folder; a test holds script A's screw within its budget and
  near its full drawing, measured on every vertex.

The stage writer has a version, which the lock records (`stage.writer`, `mime.scene.STAGE_WRITER`). A folder saved by
the first writer (before every chain frame, visual and vector was drawn, and every proxy was drawn whole) is rebuilt
by the first writer when it loads, held to its lock byte for byte, and verifies as it did; saving it again writes the
current writer's stage.

## Visuals MIME does not ship

Some descriptions' meshes cannot be redistributed. The AR4's are Annin Robotics' design: their licence forbids
redistributing or hosting them in any form, and allows educational, research and personal non-commercial use. MIME
ships a **recipe** for them, the visual set `ar4_mk5`, and never their bytes:

```sh
mime visuals list              # each set, fetched or not, with its licence, tier and role
mime visuals fetch ar4_mk5     # prints the credit and the terms, then downloads and bakes
mime visuals view a0_run       # the layer that opens the run's recording with the meshes drawn
```

- **The recipe** names the upstream repository at a pinned commit and each file by the path a description writes it
  with, its sha256 and its git blob, with the set's licence (an SPDX `LicenseRef-`), its tier (no redistribution: the
  bytes are referenced, never uploaded) and its role (presentation: no result depends on it).
- **Fetching** downloads each file from that commit through the asset cache's one resolver, held to its sha256,
  decimates it to its share of the set's triangle budget (vertex clustering; every vertex moves at most a cell's
  diagonal), measures its distance to the original from every vertex of each to the other's surface, both ways (so
  that no denser sample of vertices finds more), and writes it as a USD layer in the asset cache, under `visuals/`,
  with a record of its source, parameters, checks and output, beside the most any vertex moved (the cell's
  diagonal), which bounds it from the original's side.
- **The stage refers to a mesh by its set and its source's sha256**, never by a path into the cache, and holds none of
  its bytes. When it is opened through `mime visuals view` or `mime.viz.open_stage`, or by a client from the overlay
  `hello` names, an overlay in the cache references each baked mesh into its prim and hides the bones. Without it, the
  links are drawn as bones, and `mime visuals view` and `hello` say which sets are missing and how to fetch them, with
  each set's notice (its credit and terms) whether it is drawn or not.
- **Nothing of it is ever written into a saved folder or a run folder**, so `mime pack` cannot carry it: a test packs
  A0's saved folder and a live session's run folder with a stand-in baked in the cache, and finds none of its bytes.

The triangle budget is a presentation setting with its basis (`mime.viz.TriangleBudget`): the earlier line's AR4
meshes at CAD density shimmered in the viewport, their triangles smaller than its pixels, and drew slowly on a
software renderer.
