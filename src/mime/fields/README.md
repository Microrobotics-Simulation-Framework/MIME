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

Empty. The uniform rotating field lands in dev1's phase 4, and the point dipole of an arm-held magnet in phase 7. The
calibrated coil array and the standing wave come in a2.

Phases and releases are explained in the [package manifest](../README.md#status).
