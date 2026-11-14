# `mime.observations`

**What an experiment measures.**

## What goes here

- `SwimSpeed`, `EnergyLedger`, `PhaseLag` and `Camera`.
- Later: signals, and harm observations such as contact force and wall shear.

## The contract

- An observation declares its outputs (`outputs`, each a `mime.registry.Output`): its node's state field, its
  dimension and shape, per step or at the end, and for one the graph reduces, the per-step output and the rule
  (`Reduction`), and the setting its window starts after. Its page and a run's manifest are generated from them,
  and a run's result reads each by name (`mime.results`).
- An observation adds no edge into the physics.
- It keeps its own provider instances, so observing never changes what is observed.
- Its results live in its own state, so a long run never has to store every field at every step.

## Status

**`SwimSpeed(body, along=tube.axis, after=...)`** (`swim_speed` 0.1.0): the body's velocity along a
direction each step, and its mean over the steps that end after `after`, kept in its own node's state.

**`PhaseLag(body, field=..., about=tube.axis, after=...)`** (`phase_lag` 0.1.0): the angle from the field to the
body's moment in the plane of rotation each step, net of the lag the build declares for the laws bound to the body and the field (the build report's lags: the
magnetic torque reads the pose a step behind, unless a coupling group solves it with the body, and a build whose laws
disagree is refused, `observation.lag-ambiguous`), and the body's spin about the axis in turns per second, with their means over the steps that end
after `after`. An observation reads a uniform field where a law already reads the same interface of it (the flux
density where the torque reads it, never where the force reads the gradient), adding no slot to its provider (so the
provider's node is what it is without the observation); the build report names the law, among the lines the build
computed. A field no law reads so, or one that varies in space, waits for observations' own provider instances.

**`EnergyLedger()`** (`energy_ledger` 0.1.0): the energy ledger in one line, `exp.observe(EnergyLedger())`. A switch
the build reads (`Observation.meters_energy`): it binds no role, builds no node and declares no output; attached, the
build meters each body, and a run's result reads the bodies' records as its account (`result.energy`).

**`Camera(detector=..., mount=World(...), frame_period=...)`** (`camera` 0.1.0): a declared camera model in the loop. It
sees the whole scene (`Observation.sees_scene`: the build gives its plan the roster of every body and structure, and the
plan names the bodies it reads, `Read(of=)`), renders a frame every `frame_period` on the base step's clock and holds it
between, from a ring of the last `sub_frames` poses. Its outputs: `image` and `expected` (setting-shaped, not recordable:
`Output.shape_from_settings`, `Output.recordable`), `frame_step`, `age`, `new_frame`. The detector's nine numbers are its
node's leaves, so a study can fit them, in forward or reverse mode; a batched study through it is refused (`Observation.unbatched`). `mime.sensing` holds the
renderer, and its README the model and the replay contract.

`SwimSpeed` declares its outputs: `speed` each step, and at the end `mean_speed` (the mean of `speed`
over its window), `distance` (its integral) and `observed` (the window's length), each window the steps that end
after `after`.

Releases are explained in the [package manifest](../README.md#status).
