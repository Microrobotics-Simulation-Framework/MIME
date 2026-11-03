---
describes: "0.1.0"
---
# The swim speed of a body

**What it measures.** The body's velocity along a direction, each step, and its mean over every step that ends after a
start time: `SwimSpeed(umr, along=tube.axis)`. It reads the body's twist, adds no edge into the physics, and keeps what it
measures in its own node's state, so a run need keep no history to report it. Its outputs, by name, are `speed` (each
step, a series when the run records it) and, at the end of the run, `mean_speed`, `distance` (the speed's integral over
the counted steps) and `observed` (the time they cover). The setting `after` starts the window; by default it starts
with the run.

A step counts whole once its end lies after `after`. The mean is the distance over the time observed, and is zero until
some time is observed.

**When not to use it.**
- For the speed of a body relative to a moving fluid: it reads the body's velocity in the world, along a fixed
  direction, and nothing about the fluid.
- For a speed along a direction that turns with the body (its own axis as it turns): the direction is a fixed vector in
  the world, as `tube.axis` is.
- For a claim: it is a number the run produced, and what it bounds or estimates is for the experiment to state, from the
  comparison it makes.
- Where the start time falls between two steps and the result must be smooth in it: see the derivative below.

**Its derivative** is piecewise: the window's edge is a switch, and as `after` moves the distance and the time
observed move in jumps of a step. The entry's gradient record declares the switch and the route to its derivative (count
the step in which the start time falls by the part of it that lies after the start, with the speed interpolated between
successive steps), and that the mean has none where observing begins.

**Versions.** `swim_speed` 0.1.0 is its first.
