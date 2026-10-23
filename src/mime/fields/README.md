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

Since phase 4b, **`UniformRotatingField`** (`uniform_rotating_field` 0.1.0): a uniform magnetic field turning
right-handed about `rotation_axis` (`tube.axis`, by reference) from `initial_direction`, at its commands `frequency`
and `amplitude`, which drives write; its time is its step count times its timestep, never a sum (0.2's float32 sum
drifted by about 1e-4 in phase). It claims `Capability.UNIFORM`, so its readers' slots take no points; its gradient
is zero. The point dipole of an arm-held magnet comes in phase 7; the calibrated coil array and the standing wave in
a2.

Phases and releases are explained in the [package manifest](../README.md#status).
