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
  A preview baked from an asset is a derived file that carries the asset's licence tier and is never
  pinned by its bytes in an experiment's lock. Root prims other than `/World` are reserved for these
  overlays (see [`mime.scene`](../scene/README.md)).
- **Meshes** (`_meshes.py`): an STL file read (binary or ASCII) and welded, decimated by vertex clustering to a
  triangle budget (`TriangleBudget`, a presentation setting with its basis), its distance to the original measured
  both ways, and written as a USD mesh layer.
- **The visuals in the asset cache** (`_visuals.py`): a manifest of meshes (`mime.assets.Manifest`) named by
  `mime visuals fetch <target>` (a description's folder or file, with the manifest beside it; a manifest file; a saved
  or run folder, whose lock holds the manifests its stage draws; or a set an installed package ships, by name),
  fetched through the asset cache's resolver, each file held to its sha256, and baked under `visuals/` in the cache
  with its record: what it was made from and how, and its tier, the strictest of its source's (a baked mesh is a
  derived file, `mime.terms.derived_verdict`). Each set's index in the cache holds its credit, licence and notice, so
  the overlay needs no manifest; the overlay, written in the cache, references each baked mesh into the prim naming it
  and hides the stand-in bones, and says, for a set not baked, its terms (from the index, the folder's lock or an
  installed package) and what to fetch; a view of a stage or a recording with it is for any USD viewer. Nothing of it
  is written into a saved or a run folder; `mime assets audit` finds a baked mesh copied anywhere, by its hash, and
  `mime pack` and `mime push` refuse one copied into an experiment's folder.

- **A view that may be shared** (`_export.py`, `_web.py`, `_tiers.py`, `_primitives.py`, `_gltf/`): a stage, a recording
  or a folder's as one glTF binary (`gltf`), +Y up, its units scaled to a power of ten metres, a short recording
  slowed to about eight seconds and framed over its whole motion; and as one web page (`page`, `display` in a notebook,
  `mime visuals web`) that draws it with Google's `<model-viewer>` 4.3.1, inlined from the copy in `_viewer/` (with the
  licence of every part its bundle holds) or from jsDelivr held to its hash. The converter in `_gltf/` is derived from
  usd2gltf 0.3.5 under MIT (its `LICENSE` and `README.md` say what was taken and changed); MIME tessellates USD's
  spheres, cylinders, cubes and capsules (`_primitives.py`). **Such a view holds only the baked meshes whose tier lets
  them travel**, `open` and `conditional` (`SHAREABLE_TIERS`): the overlay it composes leaves the others
  out, drawing their frames' simple shapes and saying so, and a composed stage that reaches a baked mesh of another
  tier, or of none stated, is refused (`visuals.not-shareable`). It is never written into a saved or a run folder
  (`visuals.web-out`). A vector's arrow is drawn at most twice the largest extent of the body it is drawn at long
  (`ARROW_BODY_FACTOR`, read from that body's geometry in the stage) and at most a quarter of the scene's diagonal
  (`ARROW_CAP_FRACTION`, the scene without its arrows; an arrow at no body has this alone), in the glTF and the page alone
  and said in the page's notes; the stage and the recording hold its full length.

  The page's **basics** are each a switch, an argument of `display`, `page` and `web` and a flag of `mime visuals web`
  (`apparatus_shapes` is `gltf`'s, since it changes the GLB): a scale bar and an arrow legend (`_web.py`; on), focus and
  follow a body (off until a body is named; an unknown name is refused with the nearest, `visuals.unknown-body`), and an
  apparatus's motor and magnets drawn for the view alone (`_apparatus.py`; on). **The shapes are an overlay of the opened
  stage's session layer** under `/ApparatusView`, one prim to a frame following its world pose at every time code, sized
  from the saved experiment's declared solid or else a marked placeholder by a stated rule; they are never written into
  a stage. The page's arithmetic (the scale bar's round length, the followed position) is plain JavaScript in `_web.py`
  (`_LOGIC`), which `tests/viz/test_viewer_basics.py` runs in Node.

## Boundaries

- With `scene`, the only importer of `pxr`.
- The desktop application, MICROROBOTICA, is a separate project. It adapts to MIME's interfaces, not the reverse.

## Status

The recording layer, the mesh importer, the visuals in the asset cache, and the shareable view (a GLB and a web page).
Viewports stay with the clients.

Releases are explained in the [package manifest](../README.md#status).
