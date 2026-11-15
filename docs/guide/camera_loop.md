# A closed loop through a camera

`examples/camera_loop.py` steers a magnetic sphere by what a camera sees: each frame's soft centroid, a PID on the
error, and the exact allocation of the coils' currents, all inside the compiled step. A derivative of the run then
passes from its objective back through the coils, the body, the camera and the tracker to the controller's gains and
its model of the camera. This page says how the loop is declared, what each check holds it to, and what it does not
show.

## The case

Camera spike R2's case, every number declared in the example: a millimetre sphere with a fixed magnetic moment in
Navion_2's field, the force alone, no gravity and no medium. A camera above it looks down, its pixel a quarter of a
millimetre at the sphere, which spans a few pixels, and it makes a frame every controller tick. The reference is a step
of a few millimetres, then a circle about the stepped point.

<!-- mime-include: examples/camera_loop.py#case -->

## The loop

There is one controller, in GRAPH mode (`mode="graph"`), ticking with the camera's frames, a base step late. It reads the camera
through a measurement port, `imaging.frames`, whose source is `camera:.expected` (an observation's output is written
`<observation id>.<output>`, and the camera's id is `camera:`). That is the frame's counts before the sensor's noise and
quantisation, the camera's differentiable output. A port's noise on the camera's `image` is refused, because its sensor
model already adds noise; on `expected` a port may add its own, and the build report then says the sensor model is
bypassed. The example adds none. The controller writes the coils' currents through a command port that holds them
within Navion_2's declared limit:

<!-- mime-include: examples/camera_loop.py#experiment -->

Its law composes two plain functions of the example, `tracked` and `pid`, with what its context gives it:
`ctx.believed("camera")`, the controller's model of the camera (a `CameraModel`, whose `project` is the camera's own
projection with the controller's numbers), and `ctx.allocator("coils", ...)`, the exact allocation of the currents
through the controller's model of the coils, within the port's bounds:

<!-- mime-include: examples/camera_loop.py#controller -->

- **The tracker.** It takes the soft centroid of the frame's darkness and back-projects it onto a declared plane, the
  sphere's starting height, through the believed camera. It inverts the belief's own `project` by Newton steps from the
  pinhole's guess, so a believed lens distortion is undone by the projection that applies it. A `believed=` camera that
  differs from the declared one is the miscalibration a study can vary: its numbers are leaves of the controller's node.
- **The PID** asks the sphere for a force, `m (kp e + ki int e - kd dp/dt)`, taking its derivative on the measurement.
  The allocation turns that force into currents at the measured point.

## The flags, and where to read them

A controller has no flags of its own that a run raises, so the example keeps its two flags in the controller's state.
A run's result reads them there, and the example prints them:

<!-- mime-include: examples/camera_loop.py#run -->

- **`state_lost`** counts the ticks the target was lost, and **`state_first_lost`** gives the first such tick. A target
  is lost where its shadow holds less than half a ball's area on the plane, or where the shadow reaches the frame's
  border, which pulls its centroid inwards. A lost tick holds the last point and freezes the integral, and the law
  writes no current. The first tick is always lost: it reads the camera before the camera's first frame.
- **`state_off_plane`** counts the ticks where the shadow's area put the depth more than `plane_tolerance` (a setting of
  the controller) off the plane. **`state_depth_miss`** is the largest such depth change.

`run(..., history=True).history["controller"]` holds each of these at every step.

## What the camera cannot see

Depth. The tracker assumes the sphere is on its plane. A sphere `h` above the plane and `x` off the optical axis is
seen where the ray through its centre meets the plane, `x h / (Z - h)` further out, `Z` being the camera's height over
the plane (`test_camera_loop_tracker.py` checks it). Nothing holds this sphere on the plane. On a tick where a coil sits
at its bound, the allocation trades the z row, so the sphere drifts in z and the camera does not measure it. On R2's
first draw the drift is millimetres within a fifth of a second, in R2's run and in this example's. The off-plane flag
says when this happens. The flag does not undo it.

## The derivative

The status is **`piecewise_jump`**:
- the allocation's active set changes where a coil reaches its bound, which makes a jump in the derivative, never in
  the currents;
- the lost guard switches the law off;
- the camera's exact pixel coverage makes the objective only once differentiable in a parameter, its derivative
  Hölder-continuous of order one half, because a disc's segment past a pixel's side grows as its depth to the power
  three halves.

So a central-difference check through the loop measures its convergence order, and never assumes the smooth case's. The example
differentiates the final miss in two gains by `Study.gradient` in reverse mode:

<!-- mime-include: examples/camera_loop.py#gradient -->

The tests differentiate R2's objective (the mean over the run of the squared miss) in `kp`, `ki`, `kd` and the believed
focal length. **The coils' limit is not among them.** It is a command port's bound, which the allocation reads as a
constant of the build, so no derivative is taken in it. Spike R2's `dJ/dlimit` came from a law that carried its own
limit.

## What the checks bound, and what they do not

`tests/examples/test_camera_loop.py` holds the loop to R2's checks on R2's draws:
- **the derivative**: reverse mode against forward mode, float32 against float64, and both against central differences
  whose order is measured;
- **the allocation**: SciPy's BVLS at every tick;
- **the two modes**: CALLBACK and GRAPH run one law. Since the allocation became one law in both, this shows only that
  the modes agree, and BVLS is the allocation's independent reference;
- **the lightly damped loop**: its lost target is held and flagged, never a NaN;
- **the limit**: as the latencies go to nothing and the frame rate to the step rate, the loop meets the true-state run
  within a floor computed from the tracker's own error and the loop's linear response.

`test_camera_loop_tracker.py` holds the tracker:
- its point against the believed back-projection of the silhouette's centroid, within the soft centroid's own error;
- its derivative against central differences;
- its guard and its out-of-plane bias.

Each check has a planted control that fails it on every run. The rates are in the pull request that added the example.

These are **acceptance tests of the loop's arithmetic**. They bound the derivative's error, the allocation's error and
the tracker's error from above. They do not say the loop tracks well: on a draw where a coil saturates, the camera arm
tracks worse than the true-state arm by R2's measure, because it cannot see the sphere leave its plane.

## Why it is an example, not a block

A block is data. A controller in a block's fragment names an entry, and that entry's code must come from a
distribution its `[requires]` names ([blocks](blocks.md)). `examples/` ships in no distribution, so the loop's
controller cannot be a block's yet. The route is an examples plugin, a small package of example entries that a block
can require.
