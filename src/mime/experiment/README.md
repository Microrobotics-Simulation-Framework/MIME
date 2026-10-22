# `mime.experiment`

**The experiment a script declares, and its built form.** This package is the top of the dependency direction.

## What goes here

- **`Experiment`**, the declaration. It holds:
  - things by name, interactions by id, observations, the controller;
  - run settings: `dt` as the base step, each entry's own `rate=` as a whole multiple of it, and precision;
  - the profile (`exploratory`, `research` or `strict`);
  - conditions, gravity, and values declared from outside a record, each with its reason.

  Its methods are `add_*`, `attach`, `observe`, `set_controller`, `declare`, `plan`, `build`, `save`, `load`,
  `status` and `export_fmu`.
- **`BuiltExperiment`**, the result of `build()`: the build report, `run(...)`, and `verify()`.

## Boundaries

- It imports `build` and is imported by user scripts, `study` and `cli`.
- Declaring touches no graph. Everything happens at `build()`.

## Status

Empty. The declaration API lands in dev1's phase 2. Saving and loading land in phase 3 (see `saved/`).

Phases and releases are explained in the [package manifest](../README.md#status).
