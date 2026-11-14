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
  the screw and the helix need the sampled route, not built yet). `route=` is an override only.
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
  (`appearance_sources`). The backlit one-bin camera reads the stage's display opacity as a flat absorber (the share of
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

## What it does not do yet

The tracker and the controller that reads the in-loop camera, the stage reader, the sampled route and its crease measures, meshes and the display view's lighting and colour, X-ray
detectors: each comes in a later piece of the camera lane. Its switch sites (the box edge's pieces, the chords'
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
  columns values of the run's dtype), the same at every detector size, dtype and number of steps measured, nearly all of
  it the static structure's make, kept for its derivative in the leaves; without them it is thousands, growing with the
  steps. The render's checkpoint costs no measurable time, and neither changes a forward program: the step, a run and
  its forward-mode derivative compile to the same text, and every forward array and the reverse derivative are the same
  bit for bit. What reverse mode costs against forward mode, in memory and in time a step, is measured through the built
  graph on GitHub's runners (the outcomes of `tests/sensing/predictions_camera_reverse_merged.json`). A study's reverse
  derivative whose estimate (`CameraNode.reverse_estimate`: the larger of the derivative's compile and its run with its
  tape, fitted to those runs, times a margin) would not fit the memory a run can have is refused before it compiles
  (`camera.reverse-memory`), naming the offline renderer. A batched (vmapped) study is refused (`camera.batched`): under
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
