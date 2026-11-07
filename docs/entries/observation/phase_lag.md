---
describes: "0.1.0"
---
# The phase lag of a body behind a turning field

**What it measures.** The angle from a field to the body's moment, each projected onto the plane the field turns in,
positive where the field leads, each step; the body's rate of turning about the field's axis (`spin`, in turns per
second); and their means over every step that ends after a start time: `PhaseLag(umr, field="field.magnetic",
about=tube.axis, after=...)`. It reads the body's pose and twist and the field the magnetic torque reads, adds no
edge into the physics, and keeps what it measures in its own node's state. Its outputs, by name, are `lag` and `spin`
(each step, a series when the run records it) and, at the end of the run, `mean_lag`, `mean_spin`, `lagged` and
`turned` (their integrals over the counted steps, the latter in turns) and `observed`.

**Net of the declared lag.** The magnetic torque reads the body's pose at the start of the step and the field at its end,
as the build report's lags list them, so the moment it turns is the one before the step, a step's turn of the body
behind the one an observation reads after it. The lag is that angle, the one the torque acts on: the moment after the
step turned back about the axis by the body's spin over the step. In a steady lock below step-out it meets the torque
balance `m B sin(lag) = R omega`, with `R` the body's rotational resistance and `omega` the field's angular rate; the
angle between the field and the moment as they both stand after the step is smaller by `omega dt`, more than any
change of the drag moves script A's lag.

A body is locked where its mean spin over the window is its field's frequency: `mime.study.step_out` reads the step-out
of a study run from rest that way, and `mime.study.ramp_crossings` a ramp's, from the spin recorded every step.

**When not to use it.**
- Where the body's laws read its pose inside a coupling group with it: they read the current iterate there, not the
  pose at the start of the step, and the correction is a step too many.
- For a body that tilts out of the plane as it turns: the turn back is its spin about the axis alone.
- Where the field is read anywhere but where a law reads it: an observation reads a uniform field where a law reads its
  flux density (the build report names the law), and refuses one no law reads so (it would need its own instance of
  the provider).
- For a claim: it is a number the run produced; what it bounds or estimates is for the experiment to state.

**Its derivative** is piecewise: the window's edge is a switch, as the swim speed's is, declared in its gradient record
with the route to its derivative, and the means have none where observing begins. The angle reads zero where the moment
or the field has no part in the plane.

**Versions.** `phase_lag` 0.1.0 is its first.
