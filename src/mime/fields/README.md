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
the step, so a frequency may change during a run (a ramp, `drives.Ramp`). It holds the phase by summation by parts, as
the frequency times the time less a correction kept as a fraction of a turn, so its rounding does not grow with the
run, and for a constant frequency it computes 0.1.0's field to the last bit; its time is its step count times its
timestep, never a sum (0.2's float32 sum
drifted by about 1e-4 in phase). It claims `Capability.UNIFORM`, so its
readers' slots take no points; its gradient is zero. Version 0.1.0, whose phase is the present frequency times the
time (MIME-ANO-139), is kept registered for saved folders (`KEPT`), and refuses any writer of its frequency but a
constant drive (`command.held-constant`). A provider declares the commands it reads as constant over a run
(`constant_commands`), and the build refuses any other writer of them. The point dipole of an arm-held magnet is described below; the calibrated coil array and the standing wave in
a2.

**`PointDipole`** (`point_dipole` 0.1.0): a magnet part of the apparatus as a point
dipole, its field and gradient (by the closed form) at each reader's point, the part placed by its mount, so it
follows the chain carrying it. A reader's slot whose read names a point fixed in the reader's body (a magnetic law's
`magnet_centre`) is evaluated there, the pose's position plus its orientation applied to the point
(`Capability.BODY_POINTS`). Its moment is the apparatus's `magnetic_moment`, declared as a magnitude along the
magnet's declared direction; its node holds the magnet's radius, against which its near-field limit is stated
(MIME-ANO-100). A provider may read its apparatus's parts (`parts`) and values (`values`).

Releases are explained in the [package manifest](../README.md#status).
