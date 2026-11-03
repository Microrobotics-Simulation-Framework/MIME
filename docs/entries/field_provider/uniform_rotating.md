---
describes: "0.1.0"
---
# A uniform field turning about an axis

**What it models.** A magnetic field that is the same at every point and turns right-handed about a fixed axis at a
commanded frequency, with a commanded amplitude: an idealisation of the field a rotating-magnet or coil rig makes over
a body small against it. `UniformRotating(rotation_axis=..., initial_direction=...)` as an apparatus's field, driven through its
two commands, `frequency` and `amplitude`. The field points along the initial direction, less its part along the axis,
at the start of the run. A reader's slot takes no points, since the field does not depend on them, and the gradient it
publishes is zero everywhere. Its time is the step count times the timestep, never a running sum of timesteps.

It publishes the flux density and the gradient, so it feeds `Magnetic.Torque` (which reads the flux density) and
`Magnetic.Force` (which reads the gradient, zero here, and so delivers nothing beside it).

**When not to use it.**
- Where the field varies across the body or with the body's place: a uniform field has no gradient, no pull and no
  near field. A source with a position, such as the point dipole, is the model for that.
- For a source that the body reacts on: this field has no source mounted anywhere, so it takes no reaction. A body's
  moment turns in it and the field is not turned back.
- For a frequency that changes during the run, taken as a sweep: the phase is the present frequency times the time since
  the start, not the integral of the frequency, so a change of frequency changes the phase at once, and the field's
  direction jumps with it. A frequency held constant, or changed where a jump of phase is acceptable, is what it is for.
- For a field's delivery by coils or magnets with a bandwidth, a lag, a saturation or a nonuniformity: it is the ideal
  field, and the commanded amplitude is the amplitude delivered. Nothing here bounds how far a rig's field is from that.

**Its derivative** is `exact`, as the generated section states.

**Versions.** `uniform_rotating` 0.1.0 is its first.
