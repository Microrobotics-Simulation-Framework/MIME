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
intersections) are raw `jnp` operations, written for a node that renders inside its own `lax.cond`.

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
  rerun is given it at its nudged ones (`made_again`). A frame step reads it and never makes it: read, never carried, so
  the state holds no cache, a step between frames costs what holding the image costs and a frame step what drawing over
  it costs, at the build's leaves or any other. A forward run whose tree holds one made at other numbers than its leaves
  (a leaf written into the tree without it) makes it again once, on the host, before its steps (`refreshed_held`, a
  compare of the numbers); a run a derivative is taken through makes it once from the run's own leaves, before its steps
  (`run_constants`, for a derivative), by JAX's own rules, so the derivative is right from a continued state as from the
  first step, wherever the leaves are. A step outside MIME's runs (MADDENING's own scan, `graph.step`) that meets one made
  at other numbers draws a frame that is not a number, never one at other leaves; no flag names the cause, since MIME's
  flags and refusals are read by its own runs alone, which make it again first. A camera mounted on a frame of a chain (`Mounted`) is taken to move: it
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
- **Forward mode only.** Reverse mode and a batched (vmapped) study are refused, naming the offline renderer
  (`camera.reverse`, `camera.batched`): under `vmap` a `cond` is a `select`, and reverse mode would keep every frame step's render.

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
