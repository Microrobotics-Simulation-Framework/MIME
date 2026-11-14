---
describes: "0.1.0"
---
# A declared camera model in the loop

**What it measures.** What a camera of a declared model would see of the experiment, as an image: `Camera(detector=...,
mount=World(...), frame_period=ms(...))`. It sees the whole scene: every body (its outermost parts, at the pose its
dynamics publishes) and every structure, with nothing listed by the user; `hide=(...)` leaves some out, and the build
report names what is hidden. It is a pure observer of the scene's state, backlit with one bin: an opaque subject is its
shadow, a subject named in `absorbers` attenuates by Beer-Lambert, and every other subject takes the stated default
(opaque, with the build note "appearance defaulted for:"). Its numbers are declared, never a measured camera's, unless
`Pinhole.from_calibration(file)` read them from a calibration file, which labels them fitted with the file's name and
sha256.

**Required settings.** `detector`, `mount` and `frame_period` have no default, and omitting one is refused: the detector is
the declared camera model (MIME assumes none), the mount is where the camera is, and the period is the experimenter's
choice of clock.

**Its clock.** It runs on the base step and renders a frame every `frame_period`, a whole number of base steps (any
other period is refused, naming the nearest two; `rate=` is refused). The exposure is the `sub_frames` base steps ending
at the frame step. Between frames the image is held. Its outputs are `image` (counts), `expected` (the counts before
noise and quantisation), `frame_step` (the step the held image was made at), `age` (steps since) and `new_frame`; the
build report states the latency range, and a controller called between chunks reads the state after the previous chunk,
a chunk later.

**Mounts.** `World(position, orientation)` fixes the camera in the world, and `Mounted(frame="<apparatus>.<segment>.<frame>")`
fixes it to a frame of an apparatus's chain. Where the camera and the structure share the world's frame, the structure's image
is rendered once, when the build makes the node's parameters, and held as a parameter beside the numbers it was made from (the
detector's nine numbers, the scene's numbers, the camera's pose): a study's point and an override hold it at their own numbers.
A frame reads it and never renders it: it is read, never carried in the state, so a step between frames costs what holding the
image costs. A run whose leaves differ from the numbers its structure was made at (a leaf written into the tree without it)
renders it again once, before its steps, so no stale image is read; a run's chunks, a CALLBACK controller's included, render it
never; and a derivative's run renders it once, from its own leaves (measured on one CPU,
`tests/sensing/predictions_camera_runconst_fix.json`). A step called on its own, outside a run of MIME's (MADDENING's own
scan or `graph.step`), takes whatever parameters it is given, so it renders the structure again in any frame whose numbers
differ from the ones it was made at. Where the camera may move it is rendered in every frame, and the build
report says which holds. One camera to
an experiment: a second is refused (`id.duplicate`), and a second view is a second experiment.

**Recording and re-rendering.** In-loop frames are not recorded: `record=` of `image` or `expected` is refused
(`camera.recorded-frames`); `history=True` keeps every field of every node at every step, a camera's two frames too (megabytes a
step at a useful size: gigabytes over a long run), and is the verification and debugging path, not for a long run. The poses are recorded, and a frame is reproduced by rendering them again with
`mime.sensing.render` and the frame's noise key, `mime.sensing.frame_key(seed, frame_step)`. The re-render is two compiled
programs, not one, so it is not bit for bit: it is exact where it was measured exact (float64), and in float32 it agrees
to within one count, the few pixels whose value before rounding lies within the last bits of a rounding boundary rounding
the other way (the line the gate holds it to is derived from that mechanism in `tests/sensing/test_in_loop_replay.py`).
The replay is exact only while a frame's `window_overflow` flag is zero: a body that comes closer than its disc window was
sized for (from its initial pose, with the focal length raised by a quarter) is drawn clipped to the window, the flag is raised without
a warning, and the frame no longer equals the offline render, whose window is derived per frame.

**When not to use it.**
- For a microscope's or a real camera's optics: the model is a pinhole with OpenCV's five distortion coefficients, backlit,
  and nothing it says is a statement about a physical camera that was not calibrated against it.
- For a body or structure without a closed form (the screw, the helix, the box, the ellipsoid): it is refused with its
  reason until the sampled route lands; `hide` it, or image it offline.
- For a batched study: it is refused (`camera.batched`), naming the offline renderer, which is differentiable and vmaps:
  under `vmap` the frame's `cond` is a `select`, so every step would render.
- For reverse mode past the memory a run can have: it is refused before it compiles (`camera.reverse-memory`), naming the
  offline renderer; forward mode along a few axes keeps no tape.
- For frames at every step in a long run: the images are not recorded; record the poses.

**Its derivative** passes in forward and in reverse mode, checked through MIME's differentiated run (what `Study.gradient`
compiles) in both dtypes. Forward mode is checked against central differences (a ball's settling and the principal point
across a silhouette, the focal length in the tube wall's interior), with a planted `stop_gradient` on the edge measure as the
control. Reverse mode is checked against forward mode, against central differences (float32 against float64 where they cannot
discriminate) and a held step's derivative against its frame's, with a planted `stop_gradient` on the edge measure, on the
structure the run makes and on the held frame as the controls. Both are checked from the run's start and from a continued
state (the structure the tree holds, or a leaf written over it), and reverse mode over a CALLBACK controller's chunks too: a run
a derivative is taken through makes the structure once from the run's detector numbers, before its steps, and its derivative
follows by JAX's own rules, never from the state. Reverse mode's tape is a few hundred frames whatever the run's steps (each
corner sub-sample and each frame's render are checkpointed; what it costs is in `src/mime/sensing/README.md`), and a study
whose estimate would not fit the memory a run can have is refused before it compiles (`camera.reverse-memory`); the estimate
is fitted to runs on GitHub's four-core runners (the sizes in `predictions_camera_reverse_moving.json`, both dtypes, up to six poses a
frame) and is an extrapolation beyond those platforms and sizes. `expected`
carries the derivative in the poses and the detector's nine numbers (which are its leaves, so a study can fit them); `image`
and the counters carry none.

**Its gradient status** is `piecewise_jump`, the one its declared switches give (`mime.gradients.status_problem` enforces that
a declared status is the derived one). The renderer's switches are declared as owners `camera.*` (the pixel integration's box
edge and its pieces' composition, the exact disc's chain, the corner choice, the closed forms' intervals and caps), each with
its order in `expected` and a row of the switch harness (`tests/saved/test_switch_harness.py`) that places the renderer's own
function a quarter of a step from the switch, on one pixel or one ray, forces each branch and measures the difference of the
function's value. That bounds each switch's order at that point, from above: **a jump** where two pieces of a body's outline
meet inside a pixel (`camera.pieces.least` and `.greatest`: the nearer piece's line, with its own normal), where the pinhole
crosses a cap's plane (`camera.cap_side`), and at the exact disc's series threshold (`camera.disc.series`, where the series is a term short of the
closed form), and a kink or smoother at the others; the corner choice (`camera.corner`) measures continuous at its threshold,
where the lines that trigger it graze the pixel's corner. The harness does not show that the derivative across a kink is the
branch taken's; the derivative tests do, away from the switches. What stays raw is listed with its reason in
`docs/ratchets/raw_switches.txt`: the undistortion's solve (iterations on held values, whose derivative is the implicit
function's), the validity radius's construction, floors that engage only where the quantity they keep off zero is zero,
roundings, counts and flags, and the closed form of a capsule, which no switch-scan configuration images yet (its raw sites
are not scanned or declared). **The undistortion's derivative** is checked by a standing test
(`tests/sensing/test_undistortion.py`): the ray's point's derivative in `k1, k2, p1, p2, k3` and in the pixel, forward and
reverse, in both dtypes, against the implicit function's (written afresh in float64) and, in float64, central differences, at
half way and a hundredth short of the way to the validity radius, on five lenses, with three planted wrong derivatives (held, five per cent
off, the tangential coefficients dropped) failing every case; it clears the seven `stop_gradient`s the entry's status was held
at `unverified` by. A pixel beyond the validity radius is flagged and no claim of the model, and the derivative there is not
checked. The entry's gradient record states where, why and what is not established.

**Versions.** `camera` 0.1.0 is its first.
