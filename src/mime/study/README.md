# `mime.study`

**Many runs of one experiment**: sweeps, variants, fits and sensitivities.

## What exists

- **A sweep.** `Study(exp).sweep(points, duration=..., record=..., every=..., batch=..., drop_refused=False)`, its
  points from `grid({...})` (every combination), `along({...})` (taken together) or a list of rows. `study.plan()`
  says what will run; `study.run()` returns a `mime.results.StudyResult`.
- **Axes through handles** (`_axes.py`): an apparatus's commands (`field.commands.frequency`), a thing's value as
  `declare` takes it (`"umr.mass"`), a law's setting through the handle `attach` returns (`drag.lubrication.epsilon`),
  a body's initial state (`umr.initial.position`), and `DT` and `PRECISION`. Strings are the paths' dotted forms, with
  did-you-mean.
- **A point is the experiment with its values declared** (`_point.py`): each declaration an axis names is replaced by
  one holding the point's value, then the point is planned in full (validity lines, material reads, refusals, in every
  profile) and assembled with nothing compiled (`mime.build._construct.assemble`). A node's own check (the MRS rung's
  resistance at the start) is judged on the point's own node wherever its leaves or starting inputs differ from the
  group's first point.
- **Groups by the compiled program** (`mime.build._key`): the structure key is found from what each point assembled,
  never declared: MADDENING's description of the graph with every parameter leaf's value replaced by its layout (but a
  leaf a static array is derived from, which keeps its value), the static arrays' contents, an in-graph controller's
  traced step, and the precision. One build and one compile per group. A leaf that varies among a group's members and
  that the compiled step does not read is refused (`study.leaf-unread`).
- **The batched run** (`mime.build._maddening.batched_scan`): the group's step scanned and vmapped over the members'
  states (every field, `_meta` seeded from the built graph and carried per member through the chunks) and their varying
  leaves; a leaf every member shares enters once, unbatched. Flags are reduced per member inside the scan; the fields a
  recorded run keeps are kept per member. Chunked as a run is (`automatic_chunks`). Members of an experiment whose
  controller is called on the host run one after another on the group's one compiled step, each with its own law.
- **Run-level overrides** (`_override.py`): `built.run(d, override={...})`, `run_to(..., override=...)` and
  `mime run --override path=value`, on the same resolution; refused with their routes where another compiled step
  would be needed; recorded in `run.json` and replayed by `mime verify --run`.

## Boundaries

- It builds and runs experiments through `experiment` and `build`; it holds no code of the user's, so a study is data.
- It stays in MIME until a second library needs it.

## Status

Sweeps and overrides are built (dev2, S2b-1). The drives that ramp a command, the step-out study, saving a study and its
table, checkpoints and forks, variants, fits and sensitivities come later.

Releases are explained in the [package manifest](../README.md#status).
