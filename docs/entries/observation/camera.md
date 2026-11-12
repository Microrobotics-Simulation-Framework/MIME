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
fixes it to a frame of an apparatus's chain. The structure's image is rendered once, at the first step, where the camera and
the structure share the world's frame, and held in the node's state with the numbers it was made from (the detector's nine
numbers, the scene's numbers, the camera's pose); it is rendered again at any step where one of them differs, as when a run
is continued with another detector number, so no stale image is read. Where the camera may move it is rendered in every
frame, and the build report says which holds. One camera to an experiment: a second is refused (`id.duplicate`), and a
second view is a second experiment.

**Recording and re-rendering.** In-loop frames are not recorded: `record=` of `image` or `expected` is refused
(`camera.recorded-frames`); `history=True` keeps every field of every node at every step, a camera's frames and its cache too (the cache is ten arrays of
the image's size, so megabytes a step at a useful size: gigabytes over a long run), and is the verification and debugging path, not for a long run. The poses are recorded, and a frame is reproduced by rendering them again with
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
- For a batched study or reverse mode: both are refused, naming the offline renderer, which is differentiable and vmaps.
- For frames at every step in a long run: the images are not recorded; record the poses.

**Its derivative** is forward mode only (reverse mode is refused by name, `camera.reverse`), checked through the built graph
against central differences in both dtypes (a ball's settling and the principal point across a silhouette, and the focal
length in the tube wall's interior, with a planted `stop_gradient` on the edge measure as the control) and still declared
unverified: the renderer's switches are listed raw and not declared, so no exact or piecewise status follows from them, and
the status has no value that says forward verified and reverse refused: `expected` carries the derivative in the poses and the detector's nine numbers (which are its leaves, so a study can
fit them), `image` and the counters carry none. The entry's gradient record states where, why and the route.

**Versions.** `camera` 0.1.0 is its first.
