---
describes: ["0.1.0", "0.2.0"]
---
# A uniform field turning about an axis

**What it models.** A magnetic field that is the same at every point and turns right-handed about a fixed axis at a
commanded frequency, with a commanded amplitude: an idealisation of the field a rotating-magnet or coil rig makes over
a body small against it. `UniformRotating(rotation_axis=..., initial_direction=...)` as an apparatus's field, driven through its
two commands, `frequency` and `amplitude`. The field points along the initial direction, less its part along the axis,
at the start of the run. A reader's slot takes no points, since the field does not depend on them, and the gradient it
publishes is zero everywhere. Its time is the step count times the timestep, never a running sum of timesteps.

Its phase is the integral of the commanded frequency, in turns, each step's frequency held over the step, so the
frequency may change during a run, by a ramp (`drives.Ramp`) or a controller, and the field turns at the rate commanded
without jumping its angle. It holds the phase in turns as a pair of floating numbers, to which each step adds the frequency times the timestep as
an exact product by an error-free sum, so that the phase's error does not grow with the turns the run has made: in
float32 the angle stays within float32's spacing of an angle in a turn of the exact integral of the frequencies written,
at the end of a long run as at its start, where the product of a frequency and a time carries a rounding of the phase's own size. For
a frequency that never changes the field is not version 0.1.0's to the last bit: 0.1.0 carries that rounding.

It publishes the flux density and the gradient, so it feeds `Magnetic.Torque` (which reads the flux density) and
`Magnetic.Force` (which reads the gradient, zero here, and so delivers nothing beside it).

**When not to use it.**
- Where the field varies across the body or with the body's place: a uniform field has no gradient, no pull and no
  near field. A source with a position, such as the point dipole, is the model for that.
- For a source that the body reacts on: this field has no source mounted anywhere, so it takes no reaction. A body's
  moment turns in it and the field is not turned back.
- For a frequency that changes faster than a step resolves: each step's frequency is held over the step, so a
  frequency that changes within a step is its value for that step.
- For a field's delivery by coils or magnets with a bandwidth, a lag, a saturation or a nonuniformity: it is the ideal
  field, and the commanded amplitude is the amplitude delivered. Nothing here bounds how far a rig's field is from that.

**Its derivative** is `exact`, as the generated section states.

**Versions.** `uniform_rotating` 0.2.0 integrates its frequency. 0.1.0, the first, made the phase as the present
frequency times the time since the start, so a frequency that changed in a run turned the field at the wrong rate (a
ramp at twice the commanded rate) and jumped its angle where it stepped (MIME-ANO-139). For a constant frequency the two
compute the same field to the kept version's rounding of the phase, not to the last bit. 0.1.0 stays registered (`mime.fields.KEPT`) so that a folder saved with it loads
and reproduces, and refuses, when the experiment is built, any drive of its frequency but a constant and a controller
that writes it, naming 0.2.0; a script that writes `UniformRotating(...)` gets 0.2.0.
