# `mime.observations`

**What an experiment measures.**

## What goes here

- `SwimSpeed`, and `PhaseLag`.
- Later: cameras, signals, and harm observations such as contact force and wall shear.

## The contract

- An observation declares its outputs (`outputs`, each a `mime.registry.Output`): its node's state field, its
  dimension and shape, per step or at the end, and for one the graph reduces, the per-step output and the rule
  (`Reduction`), and the setting its window starts after. Its page and a run's manifest are generated from them,
  and a run's result reads each by name (`mime.results`).
- An observation adds no edge into the physics.
- It keeps its own provider instances, so observing never changes what is observed.
- Its results live in its own state, so a long run never has to store every field at every step.

## Status

Since phase 4b, **`SwimSpeed(body, along=tube.axis, after=...)`** (`swim_speed` 0.1.0): the body's velocity along a
direction each step, and its mean over the steps that end after `after`, kept in its own node's state. `PhaseLag`
lands in dev2.

Since phase 7k, `SwimSpeed` declares its outputs: `speed` each step, and at the end `mean_speed` (the mean of `speed`
over its window), `distance` (its integral) and `observed` (the window's length), each window the steps that end
after `after`.

Phases and releases are explained in the [package manifest](../README.md#status).
