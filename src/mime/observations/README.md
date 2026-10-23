# `mime.observations`

**What an experiment measures.**

## What goes here

- `SwimSpeed`, and `PhaseLag`.
- Later: cameras, signals, and harm observations such as contact force and wall shear.

## The contract

- An observation adds no edge into the physics.
- It keeps its own provider instances, so observing never changes what is observed.
- Its results live in its own state, so a long run never has to store every field at every step.

## Status

Since phase 4b, **`SwimSpeed(body, along=tube.axis, after=...)`** (`swim_speed` 0.1.0): the body's velocity along a
direction each step, and its mean over the steps that end after `after`, kept in its own node's state. `PhaseLag`
lands in dev2.

Phases and releases are explained in the [package manifest](../README.md#status).
