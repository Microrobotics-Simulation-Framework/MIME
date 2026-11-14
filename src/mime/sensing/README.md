# `mime.sensing`

**The sensor core: a declared camera model, rendered in JAX.** A pure observer of the scene's state: nothing here acts
on the scene. No `pxr`: the camera's stage prim is `mime.scene`'s (`write_camera`, `read_camera`).

## What goes here

The chain, its stages kept apart (the detector and its rays; each subject's path length and coverage; the reduction;
the exposure; the response), so that a general sensor can be factored out of it when a second one needs the same
stages:

- **The detector.** `Pinhole` (pixels, focal lengths and principal point in pixels) with `BrownConrady` distortion,
  OpenCV's five coefficients in OpenCV's order and conventions (+y down the image, +z forward, a pixel's centre at
  integer coordinates). `ray_table` traces each pixel's ray from the intrinsics' leaves, so a derivative in any of
  them flows through every ray: the undistortion is a bracketed Newton solve of the radial map on its first monotone
  branch, then Newton steps for the tangential terms, with its derivative held; the point is judged after the last
  step, and one more step adds its derivative, not its value (the implicit function's derivative). A pixel the
  distortion cannot reach from its first branch, or whose point did not converge, is flagged (`beyond_distortion`). `project` is OpenCV's `projectPoints` on the same model.
- **The closed-form route** (`_forms.py`), on the sphere, the capsule, the capped cylinder and the straight tube's
  wall and lumen: each gives its chord along a ray (the grazing square root through a declared softplus, `GRAZING`, so
  values and derivatives stay finite at a tangent ray) and its **pieces**, edge functions negative where the ray meets
  the piece and vanishing linearly on its outline, with no pole in the image, composed by the form's own rule (a
  capped cylinder is the convex hull of its cap discs: the least of its caps and of its side within its span; each cap
  by its cone from the pinhole, bounded by its ball), whose zero set is the outline exactly. Everything is taken on the ray, never its whole line: a body behind the camera covers
  nothing, and one straddling the camera's plane only what its part in front shows.
- **The route is chosen by capability**: a kind with a closed form registered (`register_closed_form`) takes the closed-form route;
  any other is refused with its reason (`camera.no-closed-form`: the ellipsoid and the box have no run-time shape yet;
  the screw and the helix need the sampled route, not built yet). `route=` is an override only. In the in-loop camera a
  *structure* of a kind the stage draws (an ellipsoid, a box, a helix, a screw) stands still, and is imaged by its drawing
  as a static mesh (below); a body of one moves, and is refused.
- **Pixel integration** (`_coverage.py`), its own stage. **The box edge**: each piece's straight line within the pixel
  comes through `jvp` along the ray table's derivatives (at `g / |grad g|` px, normal `grad g / |grad g|`); the lines
  are composed by the form's rule on that one scale, the distance in pixels (raw values do not share a scale: a cap
  seen near edge-on changes thirteen times faster across the image than its side), and the coverage is the exact share
  of the pixel's square on the body's side of the composed line. **Corner sub-samples**: once two of a body's pieces
  cross a pixel, or a cap's outline does, every pixel of it is split into sub-pixels (`CORNER_SAMPLES` a side), each
  its own box edge. **Exact disc coverage**, a sphere's default (exact by default): the share of each pixel inside the
  ball's conic, exactly, the pixel's corners undistorted and its sides taken straight between them; always in a window
  (its silhouette's bounding box about its projected centre where none is declared, a declared size under a
  derivative), each axis clipped to the image on its own; for a ball reaching behind the camera (no ellipse), the box
  edge on the ray. The box edge is a sphere's declared opt-in (`coverage="box_edge"`), which the notes name.
- **Appearance belongs to materials.** A subject's declared `Appearance` (opaque, or attenuating per metre) wins;
  where none is declared, the stage's `displayColor` and `displayOpacity` for it (`display`, read with inheritance);
  with neither, opaque and mid-grey, and the note "appearance defaulted for: ...". Each subject's source is reported
  (`appearance_sources`); the in-loop camera reads the stage's display primvars for the parts of a body bound to a prim of
  a referenced asset, inherited as USD does (`mime.scene.Composition.display`), and its build report says which subjects
  took them ("appearance from the stage's display primvars for: ...") and which were defaulted. A part written
  Python-first has none on the stage (MIME writes its colour), so it is defaulted unless `absorbers` declares it; a
  declared appearance is never changed by the stage (the declared subjects' frames are bit for bit the same with or
  without a display on the stage). The backlit one-bin camera reads the stage's display opacity as a flat absorber (the share of
  the backlight it blocks where it covers a pixel); a display colour is a surface's reflected colour, which backlit
  transmission does not see. USD materials and textures plug into the same seam later.
- **The reduction**, backlit transmission in one bin over media on two levels: each subject multiplies the backlight by
  the uncovered share of the pixel plus the covered share times `tau`, what it transmits where it covers: nothing if
  opaque (coverage, never an infinite attenuation), the share it does not block if flat, `exp(-(mu - mu_parent) L)`
  along its chord if it attenuates. The
  parent's own term runs along its whole chord, so this is the displaced-medium rule, exact while a child lies inside
  its parent; a child leaving its parent beyond its tolerance is flagged (`outside_parent`). A tube's lumen with no
  subject in it displaces the wall as a clear lumen.
- **The exposure** (`Exposure(sub_frames=K)`): a global shutter averaging K sub-frames' poses.
- **The response** (`Response`): a Gaussian PSF, the electrons at full backlight, Gaussian shot and read noise drawn
  from the key it is given (the same key, the same frame: record the poses and render again), gain and offset,
  quantisation to `bits`; the expected counts before noise are the differentiable output. Every number is a
  declaration, never a measurement of a camera.
- **The static cache** (`static_cache`): the static subjects' transmission and flags and the ray table, once per
  camera, keyed by the sha256 of every input they were made from (every field of the camera, its pose, each static
  subject and its parent's appearance, read from the dataclasses' fields; the corner sub-samples; the renderer's
  constants). A cache whose key differs from the render's inputs now is refused (`camera.cache-stale`), and one given
  while those numbers are traced too (`camera.cache-traced`): it carries no derivative in the intrinsics or the
  structure.
- **A subject's window** (`window=`, its half-size in pixels): its pixels rendered in a rectangle about its projected
  centre, a static ray list of fixed size; `window_overflow` is raised whenever part of its silhouette (its enclosing
  ball's) that lies in the image lies outside the window.

**Flags** (`Capture.flags`, JAX values, never silent): `beyond_distortion`, `saturated`, and per moving subject
`outside_parent`, `small_silhouette` (a box-edged silhouette under `SMALL_SILHOUETTE_PX`, where the box edge's accuracy
falls), `window_overflow`, `behind` (its enclosing ball reaches the camera's plane; what is behind is not imaged),
`ghosting` (its sub-frames `GHOSTING_PX` or more apart), `thin_cap` (a cylinder's cap in view whose ellipse is
narrower than two sub-pixels, `THIN_CAP_SUBPIXELS`: the sub-samples do not resolve it, and a pixel there may miss by
about a sub-pixel's area).

- **Scenery** (`mime.scene.SceneryGprim`, `SceneryMesh`, from `Composition.scenery`): every `Gprim` (a mesh, an implicit
  shape, curves, points) of a referenced asset that no thing owns (below no prim a body or apparatus is bound to, not a kind's
  derived prim, `mime:derived_from`) is scenery. A `Mesh` is read, with its points in metres in the world, its triangles and
  its display colour and opacity; any other gprim is found and named by its type, not read. The camera **traces each mesh**
  (below) and **refuses any other gprim by name** (`camera.mesh-untraced`, naming `hide=("<prim path>",)`): scenery is never
  dropped silently. `hide=` takes a scenery prim's path beside a body's or structure's name, and the build report lists what
  was hidden. A time-sampled mesh is read at its earliest sample and flagged `animated`, and refused (`camera.mesh-moving`).
  A layer `mime.viz.write_mesh` writes with `decimation_distance=` records how far a decimation moved the mesh (the prim's
  `customData` key `mime:decimation_distance_m`, metres), and the build report gives it in pixels at this camera; a layer
  without it is reported as not recorded: unknown, not zero. `write_mesh` writes the key only where it is given, so a layer
  written without it has the bytes it had before.
  - **Classed scenery, conservatively** (`hide=` works on each): a prim with `visibility = invisible` or `purpose` `guide`
    or `proxy`, and a point instancer's prototype meshes.
  - **Display primvars are read as USD composes them** (`mime.scene.read_display`): the nearest constant primvar up the
    path, its indices resolved (`ComputeFlattened`), a time-sampled one at its earliest sample. A parent's primvar with
    vertex, per-face or per-corner interpolation does not inherit to its children, as USD defines it, so a child of such a
    parent has none. An empty constant array on a prim blocks inheritance from its parents: it reads as none. A primvar that
    is not constant on the prim that authors it is reduced to the mean of its values, **weighted by area on a mesh** (a
    face's area spread over its corners for a per-vertex or per-corner primvar), and the build report says so ("appearance:
    displayColor per-face (uniform) on <prim> reduced to its area-weighted mean for: <subjects>"); on a prim that is no mesh
    it is the plain mean, said the same way.

## Static meshes (`_mesh.py`)

A static mesh (a scenery `Mesh` of the stage, or a structure's drawing) is traced in the in-loop camera **once a run, in
the static structure**, by a bounding volume hierarchy, so a frame costs no more with a mesh in view.

- **The tree** is built on the host when the node is made: a complete binary tree in heap order, balanced splits along each
  group's longest centroid axis, leaves of a few triangles, its boxes fixed (the mesh stands). Its arrays are the node's
  parameter leaves (`mime.sensing._mesh.KEYS`), never constants of the program, so the step's program holds no copy of the
  mesh and compiles in the time it did; the triangles reach the node as arrays the derived graph records by their sha256.
  Rays are traced in tiles of a few dozen, each ray with its own stack, in one while loop a tile.
- **Every crossing, not the nearest** (one bin): each adds its `t` signed by its facing (leaving the solid `+t`, entering
  `-t`), so the sum is the chord inside a closed, consistently wound mesh, wherever the camera is, and each adds one to the
  crossing count. A sub-ray's factor from a mesh is `exp(-mu L)` where it attenuates (named in `absorbers`; it must be a
  closed solid, else `camera.mesh-open`), what its display opacity leaves, once for each crossing, where the stage gives it
  one, nothing where it is
  opaque and crossed (the default, noted), everything where missed. A pixel's factor is the mean over `MESH_SAMPLES` x
  `MESH_SAMPLES` sub-rays (the pixel's ray moved along the ray table's derivatives) of the product over the meshes, which
  multiplies into the standing subjects' transmission.
- **The crossing test is watertight** (Woop, Benthin and Wald's): the corners moved to the ray's origin and sheared so the
  ray runs along an axis, each edge's function computed from its two corners alone, so two triangles sharing an edge compute
  exactly opposite values and a ray through an edge is claimed by exactly one of them; each product is rounded on its own
  (an optimisation barrier keeps XLA from fusing a multiply into the difference). Moller-Trumbore's barycentric test is not
  watertight: at a shared edge a ray can cross both triangles or neither, and one crossing more or less moves the signed
  chord by its whole `t`, which a camera a millimetre away makes far larger than a wall's chord.
- **`MESH_SAMPLES` is a declared default, not a setting**, set by the mesh swap gate: past it the drawing's sagitta, not the
  sub-rays, sets a silhouette's error. It is in the node's spec (the derived graph) only where a mesh is traced, and in the
  build report. A saved experiment reproduces at its own MIME version; a MIME that changed the constant would build another
  derived graph, which `verify` and a load's derived-graph check report.
- **Once a run**: made when the build makes the node's parameters, again once a run where a derivative may be taken or the
  tree's detector numbers differ from the run's (`refreshed_held`), and so again at each study point that varies the
  detector's numbers. Measured through the built graph on GitHub's four-core runners for a closed tube-wall mesh of about a
  hundred thousand triangles filling the view: a few seconds at quarter VGA and about a quarter of a minute at VGA, the
  step's compile time unchanged and a frame's time unchanged (the outcomes of `tests/sensing/predictions_camera_mesh.json`).
- **Derivatives.** The rays and the tree are taken under `stop_gradient`: no derivative reaches the traversal's while loop
  (which has none in reverse mode), so a derivative in a pose, a gain or a limit passes with a mesh in view, and one in any
  of the detector's nine numbers, which would leave the mesh's part out, is refused by name in both modes and in a fit
  (`camera.mesh-derivative`, through the node hook `derivative_leaves_problem`, which `Study.gradient` and `Study.fit` ask
  before compiling): calibrate with the meshes hidden.
- **Refused, each naming its route:** a moving (time-sampled) mesh, `camera.mesh-moving` (hide it); a camera that may move
  with a mesh in view, `camera.mesh-mounted` (a camera on a moving frame keeps no static structure; fix it in the world, or
  hide the meshes); an open or inconsistently wound mesh named in `absorbers`, `camera.mesh-open`.
- **Drawn structures** take the stage writer's drawing (its facets, its triangle budget); the build report gives each
  drawing's sagitta (about the sagitta of a facet's diagonal on its widest ring) in pixels at this camera, or says it is
  exact (a box).
- **A body inside a closed attenuating mesh** at its start (a host point-in-mesh test) takes it as its medium: its own term
  runs at `mu - mu_mesh` along its chord (the displaced-medium rule), and the build report says `outside_parent` is not
  checked against it (the step holds no signed distance to the mesh).
- **The chords offline** (`mesh_chords`): each mesh's signed chord and crossings along each sub-ray of every pixel, the path
  lengths a transmission model such as an X-ray reduction reads; no derivative passes through the rays.
- **Checked** (`tests/sensing/`): the tree against brute force, every crossing, the chord, the nearest `t` and its triangle,
  on drawn closed meshes in both dtypes (`test_mesh_bvh.py`); a tube's and a sphere's drawings against their closed forms,
  within the sagitta's bracket on the same sub-rays and the design's line against the closed forms (`test_mesh_swap.py`); the
  refusals and a pose's derivative against central differences (`test_mesh_refusals.py`).

## What it does not do yet

The tracker and the controller that reads the in-loop camera, the sampled route and its crease measures, moving meshes
(a refit tree for offline renders and the display view) and the display view's lighting and colour, X-ray detectors: each
comes in a later piece of the camera lane. Its switch sites (the box edge's pieces, the chords'
intersections) go through `mime.core.switch` where a switch has a branch a derivative can take, under the owners
`camera.*` (`_switches.py`, declared on the `camera` entry, each a row of the switch harness); the sites that stay raw are
listed with their reasons in `docs/ratchets/raw_switches.txt`. A switch collector keeps what a loop's or a cond's body
traces beyond it, so while one is open `_box_coverage` takes the corner sub-samples unrolled and the choice as a select
(`switch.collector_open()`); no program a run compiles changes.

## The in-loop camera (`mime.observations.Camera`, node `_node.py`)

The renderer on the base step's clock, as an observation that sees the whole scene (the experiment's roster, never a
list from the user; `hide=` leaves some out and the build report says which). A frame is rendered every `frame_period`
base steps (the node's own count; `rate=` is refused) over the last `sub_frames` poses of every body, held between, and
`frame_step`, `age` and `new_frame` are the clock's. The detector's nine numbers are the node's leaves (bounded about the
build's: the disc windows are sized for them), so a study can fit them, and `expected` is the differentiable output.

- **The static structure.** Where the camera is fixed in the world the structure's image and the ray table (with the
  static flags) are one parameter of the node, `static_structure`, made when the node's parameters are made, beside every
  number it was made from (`static_structure_from`: the nine leaves, the scene's numbers, the camera's pose). So the
  build's parameter tree, a study's point's and an override's each hold it at their own leaves, and the save's resolution
  rerun is given it at its nudged ones (`made_again`). A frame step of MIME's runs reads it, made at the run's leaves,
  and never makes it: read, never carried, so the state holds no cache, a step between frames costs what holding the image costs and a frame step what drawing over
  it costs, at the build's leaves or any other. A forward run whose tree holds one made at other numbers than its leaves
  (a leaf written into the tree without it) makes it again once, on the host, before its steps (`refreshed_held`, a
  compare of the numbers); a run a derivative is taken through makes it once from the run's own leaves, before its steps
  (`run_constants`, for a derivative), by JAX's own rules, so the derivative is right from a continued state as from the
  first step, wherever the leaves are. A step given its run-constants (a differentiated run, a chunk that tallies the
  run's flags) holds no branch that makes it, and one that still meets a structure made at other numbers draws a frame
  that is not a number (no flag names the cause: the runs make it again first, so only a tree given past them reaches
  it). A step not given them (MADDENING's own scan, which MIME's recorded runs use, `graph.step`, MADDENING's
  verification, which injects a moved leaf and asks for the frame a node built with it draws) takes whatever tree it is
  given, so it chooses per frame: the tree's structure where its numbers are the step's, one made from them where any
  differs, a branch no run of MIME's takes, having made it again first; a derivative taken through such a step carries no
  part of the structure where they hold (MIME's derivatives go through its differentiated run). A camera mounted on a frame of a chain (`Mounted`) is taken to move: it
  holds none, and the structure is rendered in every frame, which the build report states. What it costs is measured
  through the built graph on one CPU with the commits interleaved (the outcomes of
  `tests/sensing/predictions_camera_runconst.json` and `tests/sensing/predictions_camera_runconst_fix.json`): the build
  renders it once; a forward run's chunks never, a CALLBACK controller's included; a forward run whose leaves differ
  from its tree's structure, once; a differentiated run once, at the build's leaves or off them alike.
- **Recording and re-rendering.** In-loop frames are not recorded (`record=` of `image` or `expected` is refused,
  `camera.recorded-frames`; `history=True`, the all-fields verification path, is not refused). The poses are, and a frame is
  reproduced by rendering them again with `render` and `frame_key(seed, frame_step)`. The re-render is a second compiled
  program, so it is not bit for bit: exact in float64 where it was measured exact, and in float32 within one count, the pixels
  whose value before rounding lies within the last bits of a rounding boundary rounding the other way; the line the gate holds
  it to is derived from that mechanism (`tests/sensing/test_in_loop_replay.py`).
- **Both modes; a batched study refused.** A derivative passes in forward and in reverse mode, through MIME's
  differentiated run (what `Study.gradient` compiles: reverse mode in checkpointed chunks, forward mode in one). Reverse
  mode keeps a `cond`'s residuals as large as its largest branch at every step, whichever branch runs, so two
  checkpoints hold its tape down: each corner sub-sample's (the static structure's make and each frame keep one
  sub-sample's residuals at a time) and each frame's render (a step keeps the render's inputs, and the backward pass draws
  the frame again). With both, the reverse program's temporary memory is a few hundred frames (a frame: rows times
  columns values of the run's dtype), the same at every detector size, dtype and number of steps measured; without them
  it is thousands, growing with the steps. That figure is a measurement, XLA's compiled temporary memory over a frame's
  bytes, not a count of the program's arrays, and this README does not say what it is made of: no measurement here
  separates the static structure's make from the rest. The render's checkpoint costs no measurable time, and neither changes a forward program: the step, a run and
  its forward-mode derivative compile to the same text, and every forward array and the reverse derivative are the same
  bit for bit. What reverse mode costs against forward mode, in memory and in time a step, is measured through the built
  graph on GitHub's runners (the outcomes of `tests/sensing/predictions_camera_reverse_merged.json`). A study's reverse
  derivative whose estimate (`CameraNode.reverse_estimate`: the larger of the derivative's compile and its run with its
  tape, times a margin) would not fit the memory a run can have is refused before it compiles (`camera.reverse-memory`),
  naming the offline renderer. The estimate is fitted to runs measured on GitHub's four-core runners, at the sizes the
  predictions file `predictions_camera_reverse_moving.json` lists (a small detector to a large one), both dtypes, up to six poses a frame; its per-pose term is the largest cost measured on any platform (a laptop's),
  not a fit to the runners. On those platforms and within those sizes it held every measured run; beyond them (another
  platform, more poses, other sizes) it is an extrapolation, not a bound. A batched (vmapped) study is refused (`camera.batched`): under
  `vmap` a `cond` is a `select`, so every step would render.

## Accuracy and cost, measured

Against exact oracles, through a pinhole; each line's number, its derivation and its measurement are in the test's
docstring (`tests/sensing/`), the predictions and outcomes in `predictions_camera_core.json` and
`predictions_fix_round.json`:

- **The box edge on a sphere**: coverage and its derivative per pixel within lines by silhouette radius
  (`test_box_edge.py`); **exact disc coverage** to the oracle's quadrature in float64, and in float32 to the rounding
  of the camera's rotation and of the disc's own arithmetic.
- **A capped cylinder**, per pixel against the hull of its rims' images, turned towards the line of sight, rolled and
  moved by a fraction of a pixel (`test_corners.py`): within a line set by the corner sub-samples (one sub-pixel's area),
  except about a cap whose ellipse is narrower than two sub-pixels, which `thin_cap` reports.
- **A capsule**, per pixel against the hull of its end balls' silhouettes (`test_capsule.py`): within the straight
  line's error on its outline's least curvature radius; its chord against a bisection of the ray's distance from its
  segment.
- **Not shown:** a translucent silhouette's per-pixel coverage (its aggregates' derivatives are, in float64,
  `test_camera_gradients.py`), exact disc coverage through a distorted lens, and the capsule's and the cylinder's
  per-pixel derivatives.
- **Frame times** on GitHub's runners are in the pull request that added this package: a runner's time, not a bound.

## Status

Built in dev2 as the camera lane's first piece. Releases are explained in the
[package manifest](../README.md#status).
