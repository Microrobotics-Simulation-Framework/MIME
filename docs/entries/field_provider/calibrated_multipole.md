---
describes: "0.1.0"
---
# An electromagnet array, as a calibrated multipole model

**What it models.** The magnetic field of an array of electromagnets (an electromagnetic navigation system) and its
gradient, at each reader's point, linear in the current each coil carries: each coil's field per unit current is the sum
of a few fitted point sources, the multipole electromagnet model of order one (Petruska et al.), read from a calibration
MIME packages. `fields={"magnetic": CalibratedMultipole(calibration=asset("navion_2"))}`, with `commands={"currents":
A((...))}` (one current per coil, by a drive or written by a controller). The packaged calibrations are ETH MSRL's published calibration of Navion_2 (`models/Navion_2_Calibration_24-02-2020.yaml`), of CardioMag (`mag_control/mpem/cal/C_Mag_Calibration_06-25-2015_no_offset.yaml`) and of OctoMag (`OctoMag_Calibration.yaml`, undated in its name), each evaluated by MIME and shipped byte for byte as published; each has its card under `docs/assets/`, with its provenance, its licence and its conditioning. MIME is not affiliated with or endorsed by the makers of these systems.

The command `currents` reaches the node as `delivered_currents`, the current each coil carries: the currents are ideal,
delivered as commanded, so a later stage between the command and the coils (a slew limit, a coil's circuit) sits there
with no script changed. The calibration's own frame is placed in the world by `at` and `orientation`, one exact
transform (`B = R b(R^T (x - t))`, `G = R G_c R^T`); `background` adds a uniform field in the world frame, zero unless
declared. The sources' positions, directions and coefficients, the pose and the background are the node's leaves, read
by its step, so a derivative passes through them and a later release's overrides can vary them without a rebuild (this release has
no override or study axis for them).

It gives, outside a run, the field and gradient per unit current at a point (`provider.actuation(point)`, an
`Actuation`, whose `matrix()` is the actuation matrix: the field's rows over the gradient's independent ones) and
currents for a wanted field and gradient within limits the user declares (`allocate(actuation, field=..., gradient=...,
limits=..., field_scale=..., gradient_scale=...)`: bounded least squares, by the exact allocation run on the host in
float64, the least current among several minimisers). Inside a controller's law, where a derivative is wanted, call
`mime.control.exact_allocation` on the rows `placed_unit_fields` gives at the body's point: `allocate()` is the same
law on the host, and neither takes more coils than `mime.control.EXACT_COMMANDS` (the iterative path is the route
beyond it); the allocation note says which path to use. No calibration ships limits: the limits `allocate()` takes are the user's declarations, and the guard's factor is MIME's declaration (a convention set from the measurement each card gives), neither a maker's specification. The validity lines cite the sources of the figures published for each system.

It publishes the flux density and the gradient at the reader's point, or at the point fixed in the body a read names
(the magnet's centroid), so it feeds `Magnetic.Torque` and `Magnetic.Force`. With the energy ledger attached the array
is outside the components (an apparatus with no chain receives nothing from them), and the magnetic pair's work is
supplied by it. The ledger books the wrench the laws deliver as given, so it cannot see a wrong field: a mis-scaled
flux density closes its account; the oracle is what checks the field.

**When not to use it.**
- As evidence of accuracy inside the calibration's `Workspace_Dimensions`, or of extrapolation outside it: that region
  is declared, never fitted. A reader outside it raises the flag `outside_region` ("outside the declared region"), which
  says only that.
- Near a fitted source, where the field is singular. The sources are basis functions, not hardware: a fit may place them
  far out, or just outside its own declared region (OctoMag's, 20 um outside its -x face). `near_source` is raised where
  a coil's gradient per unit current at a reader is more than 30 times its median over the declared region, which is a
  source's own term (each card gives the medians and what the rule flags: nothing in Navion_2's or CardioMag's region,
  a pocket about 1.2 mm around the source in OctoMag's, 19 of 60 000 drawn points). `source_clearance`, a length,
  replaces the rule with a distance from a fitted source, and zero removes the guard.
- In float32 for OctoMag: its sources cancel so deeply in its region that float32's rounding would cost more of the
  field than the declared guard allows, and more of its gradient; it is refused, and float64 is the route. Navion_2 and
  CardioMag build in float32. On a GPU set `JAX_DEFAULT_MATMUL_PRECISION=highest`.
- Where the direction of rotation matters against hardware: the field follows mag_manip's sign convention, which ETH's measured Navion_2 data disagree with by a global sign, and which side is flipped is not known (MIME-ANO-153). Gradient
  pull on a field-aligned body does not change with it; the torque on a fixed moment does.
- Above core saturation, for a calibration of a higher order, a lab's own calibration file, or a coil's circuit
  (inductance, back-EMF): none is in this release. Saturation is deferred because no shipped calibration has fitted its
  parameter and it makes the field nonlinear in current; the reader refuses a higher order, which no shipped
  calibration has.
- For CardioMag at zero current, or wherever a current-independent field matters: its file carries an `Offset` block of
  eight sources, about 1.5 mT at the median over its region (3 to 30 % of a 5 to 50 mT field), that neither mag_manip nor
  MIME evaluates (upstream loads the block only when `Coil_List` names it, and the file's does not). `background` is
  not it, the sources being position dependent. Navion_2's and OctoMag's files carry none.

**What its agreement with mag_manip says.** The node and its helper match ETH's `mag_manip` on every system, to a
tolerance derived per system from the field's cancellation at the compared points
(`tests/oracles/test_calibrated_multipole.py`). That is a software oracle: it pins sign and frame to mag_manip's
convention and cannot ground them against hardware.

**Its derivative** is `exact`, as the generated section states.

**Versions.** `calibrated_multipole` 0.1.0 is its first. The previous line's `CoilArrayFieldNode` (MIME-NODE-103) held
the same model with a sign setting, a background field, saturation and higher orders; this one keeps the background,
states the sign convention instead of a setting, and leaves the rest out, as above.
