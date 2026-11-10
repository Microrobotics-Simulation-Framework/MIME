# `mime.fields`

**Field providers.** A field is anything read at points: a magnetic field, a flow velocity, an acoustic pressure.

## What goes here

A provider makes its field available in one of three ways:
- **Evaluated**: a uniform rotating field, a point dipole, a calibrated coil array (MPEM), a standing acoustic wave.
- **Solved**: a flow over a medium. Its field is the solving dynamics' own state, read through a gather.
- **Tabulated**: a field map, for example from COMSOL.

Each provider builds one node, with one reader slot per thing that reads it.

## Boundaries

- Built by `build`, from nodes in `nodes/`.
- Every field is named where it is used (`field=field.magnetic`). Fields of a kind are never summed silently.

## Status

**`UniformRotating`** (`uniform_rotating` 0.2.0): a uniform magnetic field turning
right-handed about `rotation_axis` (`tube.axis`, by reference) from `initial_direction`, at its commands `frequency`
and `amplitude`, which drives or a controller write; its phase is the integral of its frequency, each step's held over
the step, so a frequency may change during a run (a ramp, `drives.Ramp`). It holds the phase in turns as a pair of
floating numbers, to which each step adds the frequency times the timestep as an exact product by an error-free sum,
so that the phase's error does not grow with the turns the run has made (in float32 it stays within the spacing of an
angle in a turn of the integral of the frequencies written, at the end of a long run as at its start), where the frequency times the
time carries a rounding of the phase's own size. For a constant frequency it is no longer 0.1.0's field to the last
bit: 0.1.0's product carries that rounding. Its step is the node's own timestep, never a sum of traced steps. It claims `Capability.UNIFORM`, so its
readers' slots take no points; its gradient is zero. Version 0.1.0, whose phase is the present frequency times the
time (MIME-ANO-139), is kept registered for saved folders (`KEPT`), and refuses any writer of its frequency but a
constant drive (`command.held-constant`). A provider declares the commands it reads as constant over a run
(`constant_commands`), and the build refuses any other writer of them. The point dipole of an arm-held magnet and the
calibrated coil array are described below; the standing wave in a2.

**`PointDipole`** (`point_dipole` 0.1.0): a magnet part of the apparatus as a point
dipole, its field and gradient (by the closed form) at each reader's point, the part placed by its mount, so it
follows the chain carrying it. A reader's slot whose read names a point fixed in the reader's body (a magnetic law's
`magnet_centre`) is evaluated there, the pose's position plus its orientation applied to the point
(`Capability.BODY_POINTS`). Its moment is the apparatus's `magnetic_moment`, declared as a magnitude along the
magnet's declared direction; its node holds the magnet's radius, against which its near-field limit is stated
(MIME-ANO-100). A provider may read its apparatus's parts (`parts`) and values (`values`).

**`CalibratedMultipole`** (`calibrated_multipole` 0.1.0): an electromagnet array whose coils are a calibrated multipole model of order one, from one of the three packaged calibrations, ETH MSRL's published calibrations of Navion_2, OctoMag and CardioMag, evaluated by MIME (`asset("navion_2")`, `"octomag"`, `"cardiomag"`), on
`CoilArray`, the contract every kind of coil array keeps (not registered; coils from their geometry join it later): a
field linear in the command `currents`, which reaches the node as `delivered_currents`, so a stage between a command and
the coils can sit there later; the array placed by `at` and `orientation`, one exact transform; a uniform `background`,
zero unless declared. Its sources, its pose and its background are its node's leaves. It raises `outside_region` (a
reader outside the calibration's declared region, which is not a fit domain) and `near_source` (a coil's gradient per
unit current above 30 times its median over the declared region, a fitted source's own term; `source_clearance`
replaces it with a distance), and refuses float32 where
the sources' cancellation puts float32's rounding above its declared guard (OctoMag). Outside a run,
`provider.actuation(point)` is the field and gradient per unit current there (`Actuation`, its `matrix()` the actuation
matrix), and
`allocate(actuation, field=, gradient=, limits=)` the bounded least-squares currents for a wanted field and gradient
within limits the user declares. Its sign convention is mag_manip's, which hardware has not grounded (MIME-ANO-153).

Releases are explained in the [package manifest](../README.md#status).
