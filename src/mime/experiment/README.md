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
- **`BuiltExperiment`**, the result of `build()`: the build report, `run(...)`, `run_to(folder, duration)` (the
  outputs, their manifest, a record and the run's own saved form, which `mime run` writes), and `verify()`. A run
  returns a `RunResult` (`mime.results`): each observation's outputs by name, with units, and the series
  `record=` names, one every `every=`.
- **`ObservationHandle`**, what `observe` returns: the observation's id, and each declared output by name
  (`swim.speed`), as `record=` takes it; a run's result is indexed by it.

## Boundaries

- It imports `build` and is imported by user scripts, `study` and `cli`.
- Declaring touches no graph. Everything happens at `build()`.

## Status

`Experiment` with `add_*`, `attach`, `observe`, `plan` and `build`; `dt` (no default),
`up`, `gravity` and `seed`; `BuiltExperiment` with its report, its graph's text and `run`. Saving and loading are in `saved/`.

`declare(path, value, reason=...)`, a value of a thing stated in the experiment (the report and the
saved form label it "declared in the experiment"); `add_structure` returns a handle whose typed paths are its
geometry's regions, surfaces and directions; `BuiltExperiment.snapshot()` shows the step each edge reads. The
controller, the profile, conditions and precision arrive with what reads them.

`Experiment(scene=Scene.new(up=..., units=...), gravity=...)`. Up is declared on the scene, once;
`up=` given to the experiment is refused. `build()` also writes the scene's stage (`BuiltExperiment.stage_text`),
and `save` writes it as `scene/scene.usda` with the asset layers it references; `from_document` and `load` read those
from beside the document.

Its invariants are held by Hypothesis stateful tests over random scripts of its calls
(`tests/saved/test_experiment_states.py`): a build succeeds or refuses with reasons, `load(save(x))` keeps the
derived hash, and a refused build leaves the experiment as it was. A loaded experiment checks every build against
its lock, so once changed it is refused (`load.input-hash`) and cannot be saved again: what a loaded experiment that
is edited should become is open.

`run(duration, record=[...], every=...)` and `run_to(..., record=, every=)` return and write a
`RunResult`, read by each observation's declared outputs, and `run_to` writes `outputs.json` beside `outputs.npz`.
`record=` and `every=` are a run's options, not the experiment's: the saved form is unchanged, and `mime run` takes
them as `--record` and `--every`. A run that records nothing runs as before. `run_layout(plans)` is the layout a
result is read by.

`Experiment(..., profile=)`, `exploratory` when not given (`mime.validity`); gravity standard by
convention when not given; `declare(path, value)` with no reason, a pending declaration; and `source(path, reason=...)`,
a setting's source apart from its value. The report's `pending()` lists what is pending with the lines to paste, and a
run's result carries the profile's stamp.

`built.live(...)`, a live session (`mime.live`); `built.binding()`, the stage binding,
which field of the state moves which prim; `built.topology()`, the built graph in MIME's names, written as
`topology.json` beside a run's outputs; and `run_to(..., hook=, chunk=, stage_every=)`, a saved run that streams and
writes the recording layer.

A declaration made a second time (`_redeclared.py`), as when a notebook's cell runs twice: the same as the first
(`mime._content.same_content`: its type, every setting with its units and reason, what it binds) is kept once, with a
warning, in a script and in a notebook; a changed one is added and refused at the build in a script, and in a
notebook or IPython (`mime._notebook.in_notebook`, the one test the run's display also asks; it is also true under
`ipython script.py` and `%run`) replaces the earlier one, with a warning. A structure, medium, body or apparatus
(the same name and kind) is replaced wherever other declarations name it: each holds the thing by its name, which the
build resolves, so what names it follows it, and the one warning names each law, observation, value, source,
controller and other thing that does. A name of another kind, or a declaration that holds the earlier thing's object
and not its name, is kept beside the new one and refused at the build (`thing.duplicate`, saying a cell has probably
run twice). A value declared of the replaced thing still applies and overrides what the new declaration brings, and
the note says so. Every replacement is also recorded on the experiment (`plan().replaced`, the build report), since a
warnings filter removes the warning, and the note is raised before the change, so a warning raised as an error leaves
the experiment as it was; the saved form is unchanged. A law or an observation that a source or the controller names is
kept beside the new one likewise. A permission has no key, so only the same one again is kept once; a scene's reference
of the same asset at the same prim again is kept once by the scene.

Releases are explained in the [package manifest](../README.md#status).
