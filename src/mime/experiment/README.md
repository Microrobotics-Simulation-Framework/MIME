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

Landed in dev1's phase 2: `Experiment` with `add_*`, `attach`, `observe`, `plan` and `build`; `dt` (no default),
`up`, `gravity` and `seed`; `BuiltExperiment` with its report, its graph's text and `run`. Saving and loading land in
phase 3 (see `saved/`).

Since phase 4b: `declare(path, value, reason=...)`, a value of a thing stated in the experiment (the report and the
saved form label it "declared in the experiment"); `add_structure` returns a handle whose typed paths are its
geometry's regions, surfaces and directions; `BuiltExperiment.snapshot()` shows the step each edge reads. The
controller, the profile, conditions and precision arrive with what reads them.

Since phase 6: `Experiment(scene=Scene.new(up=..., units=...), gravity=...)`. Up is declared on the scene, once;
`up=` given to the experiment is refused. `build()` also writes the scene's stage (`BuiltExperiment.stage_text`),
and `save` writes it as `scene/scene.usda` with the asset layers it references; `from_document` and `load` read those
from beside the document.

Phases and releases are explained in the [package manifest](../README.md#status).
