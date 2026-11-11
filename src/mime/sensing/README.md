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
  branch, then Newton steps for the tangential terms, with its derivative held, then one step with it flowing (the
  implicit function's derivative). A pixel the distortion cannot reach from its first branch is flagged
  (`beyond_distortion`). `project` is OpenCV's `projectPoints` on the same model.
- **The closed-form route** (`_forms.py`), on the sphere, the capsule, the capped cylinder and the straight tube's
  wall and lumen: each gives its chord along a ray (the grazing square root through a declared softplus, `GRAZING`, so
  values and derivatives stay finite at a tangent ray) and an **edge function with a distance**, negative where the
  ray meets the body and vanishing linearly on its silhouette. A capped cylinder is the convex hull of its cap discs:
  its edge function is the least of its caps' and of its side's beyond its span, whose zero set is the hull's outline
  exactly.
- **The route is chosen by capability**: a kind with a closed form registered (`register_closed_form`) takes the closed-form route;
  any other is refused with its reason (`camera.no-closed-form`: the ellipsoid and the box have no run-time shape yet;
  the screw and the helix need the sampled route, not built yet). `route=` is an override only.
- **Pixel integration** (`_coverage.py`), its own stage. **The box edge**: an edge function's line within the pixel
  comes through `jvp` along the ray table's derivatives (at `g / |grad g|` px, normal `grad g / |grad g|`), and the
  coverage is the exact share of the pixel's square on the body's side of it. **Corner sub-samples**: where two of a
  body's pieces cross a pixel (a cylinder seen side-on), the pixel is split into sub-pixels (`CORNER_SAMPLES` a side),
  each its own box edge. **Exact disc coverage**, a sphere's default (exact by default): the share of each pixel
  inside the ball's conic, exactly, the pixel's corners undistorted and its sides taken straight between them; always in
  a window about the silhouette (sized from its silhouette at its pose where none is declared, a declared size under a
  derivative), its overflow flagged. The box edge is a sphere's declared opt-in (`coverage="box_edge"`), which the notes name.
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
- **The static cache** (`static_cache`): the static subjects' transmission and the ray table, once per camera, keyed by the
  sha256 of every number they were made from. A cache whose key differs from the camera's and the static subjects'
  numbers now is refused (`camera.cache-stale`), and one given while those numbers are traced too
  (`camera.cache-traced`): it carries no derivative in the intrinsics or the structure.
- **A subject's window** (`window=`, its half-size in pixels): its pixels rendered in a square about its projected
  centre, a static ray list of fixed size; `window_overflow` is raised when its projected reach leaves the window.

**Flags** (`Capture.flags`, JAX values, never silent): `beyond_distortion`, `saturated`, and per moving subject
`outside_parent`, `small_silhouette` (a box-edged silhouette under `SMALL_SILHOUETTE_PX`, where the box edge's accuracy
falls), `window_overflow`, `behind`, `ghosting` (its sub-frames a pixel or more apart), `thin_cap` (a cylinder's cap seen within
a pixel of edge-on, its outline resolved to about a seventh of a pixel there).

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

- **The static cache.** Where the camera is fixed in the world the structure's image is made once at the build and used only
  while the leaves equal the numbers it was made from (compared at the step; a derivative in them takes the uncached path).
  A camera mounted on a frame of a chain (`Mounted`) is taken to move: it has no cache, and the structure is rendered in
  every frame, which the build report states. A stale image is never read.
- **Recording and re-rendering.** In-loop frames are not recorded (`record=` of `image` or `expected` is refused,
  `camera.recorded-frames`; `history=True`, the all-fields verification path, is not refused). The poses are, and a frame is
  reproduced by rendering them again with `render` and `frame_key(seed, frame_step)`. The re-render is a second compiled
  program, so it is not bit for bit: exact in float64 where it was measured exact, and in float32 within one count, the pixels
  whose value before rounding lies within the last bits of a rounding boundary rounding the other way; the line the gate holds
  it to is derived from that mechanism (`tests/sensing/test_in_loop_replay.py`).
- **Forward mode only.** Reverse mode and a batched (vmapped) study are refused, naming the offline renderer
  (`camera.reverse`, `camera.batched`): under `vmap` a `cond` is a `select`, and reverse mode would keep every frame step's render.

## Accuracy and cost, measured

- **The box edge**, against an exact oracle (each pixel's exact area, the boundary integral of its derivative), on a
  sphere through a pinhole: within the lines its gate states by silhouette radius (`tests/sensing/test_box_edge.py`,
  whose predictions are `tests/sensing/predictions_camera_core.json`); exact disc coverage to the oracle's quadrature,
  and in float32 to the rounding of the camera's pose.
- **Frame times** on GitHub's runners are in the pull request that added this package: a runner's time, not a bound.

## Status

Built in dev2 as the camera lane's first piece. Releases are explained in the
[package manifest](../README.md#status).
