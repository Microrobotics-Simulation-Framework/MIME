- **Surfaces that deform and carry fields, recorded in USD and shown in MIME's page** (USD-SURFACES, from spike
  USD-DEFORM). An entry declares the surfaces its node moves, `Entry.surface_meshes(settings)` returning
  `mime.registry.SurfaceMesh` (fixed triangles, the rest in its thing's frame, the node field holding the points, `None`
  for a still surface, and its fields as `MeshField`s: name, unit, port, a range, a reduction), drawn below its thing's
  prim or the role it names (`on=`).
  - **The stage** holds each at rest, a `UsdGeom.Mesh` of float32 points with `mime:mesh:owner`, `mime:mesh:fields`,
    `mime:display` (`revolved`, `extruded`) and a vertex primvar `primvars:mime:<field>` per field. A surface over
    `SURFACE_TRIANGLES` (16,000) is decimated by vertex clustering at rest, each sample reduced over its clusters (a field's
    `mean`, or `max` for a damage-like field, so a front is not smeared); no drawn triangle folds under a crater up to ten
    widths deep (a test, with a mirrored map as its control). A surface whose rest's centre lies more than
    `SURFACE_ORIGIN_EXTENTS` (100) times its extent from its thing's origin is refused at build.
  - **The recording** time-samples its points and its fields' values (physics values, never colours) on the run's one
    time base, each the node's value at that step to the float32 bit, in float32 and float64 (each with a planted
    one-sample shift caught).
  - **The page** draws it with morph targets (POSITION and a four-component COLOR_0, viridis, one animation per field, at
    most 256 targets a surface), a field legend, a picker that keeps the time and the play state, and its own size. A page
    of about 15 MB holds about 3.5e5 vertex-frames of one moving field (`docs/recordings.md` has the rows).
  - **The stage binding is `mime.binding/2`**, with the kinds `points` and `field` (`space` `thing`, `vertices`,
    `surface_field`, `reduction`, `clusters` where decimated); its `pose` and `vector` entries are unchanged. Live
    frames leave a surface out unless `select=` names it. A ghosted comparison refuses members whose surfaces differ in
    vertex count (`compare.scene`).
  - A toy, `toy.indented_sheet`, declares one. Where no entry declares a surface, the stage, the recording, the run's
    state, the GLB and the page are the same bytes as before, in float32 and float64 (measured on three experiments, with
    the sheet added as the control that differs).
