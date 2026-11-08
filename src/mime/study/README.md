# `mime.study`

**Many runs of one experiment**: sweeps, variants, fits and sensitivities.

## What exists

- **A sweep.** `Study(exp).sweep(points, duration=..., record=..., every=..., stage_every=None, batch=..., drop_refused=False)`, its
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
  that the compiled step does not read is refused (`study.leaf-unread`), and so is an axis whose values change no leaf,
  no initial state and no key (`study.axis-inert`): a study sees a value only where a node holds it in a leaf or a
  declared static, so a node must hold every value it consumes there. What splits a group is what a node makes from a
  value when it is built and keeps in its program: a static it derives and declares, a count or a layout (the MRS
  rung's force points from its spacing, a hull's counts and groups from the contact's spacing or margin), the node set,
  `dt` and the precision. The MRS rung's settings and wall tables are leaves, and so is its held body-frame matrix, made
  at build with the leaves it was made at beside it: a sweep of its blob is one group. A geometry's number is an axis,
  named as `exp.source` names it (`"ball.parts.shell.radius"`, `"tube.inner_radius"`); a kind's setting that is not one
  number (an ellipsoid's semi-axes, a box's size, a direction, a count) is refused as one (`study.held-geometry`). Without
  a declared range a geometry sweep splits as its key says (the MRS rung re-picks its force points from each radius);
  inside one it is a declared parameter, its counts fixed by the range. A value held in a node's program as a constant
  of its own, not as a leaf, is a silent bake: every member would run the first point's. The two-builds check
  (`mime.testing.two_builds`) is how an entry shows it holds none.
- **Declared ranges** (`exp.declare_range(path, low, high, reason=...)`, saved with the experiment): on any axis but `dt`,
  the precision, a rung and a numerical setting. Every choice an entry makes from the value (a count, a spacing) is made
  over the range through `mime.registry.choose_over`, with a rule it declares (`Monotone`, judged at the corners, or
  `Sampled`, inside as well; neither is refused as the entry's defect), so that every value inside builds one program: a
  sweep across it is one group and one compile, and an override inside it runs as data. A run of the experiment alone
  uses the range's discretisation; the build report prices what the range moved against the experiment's own value's
  choice, and the build plans the experiment at every corner (at most `MAX_RANGES` ranges, six, the corners doubling
  with each). A point outside a
  range is built for its own value, that range taken out, a group of its own whose split reason names it; members inside
  a range that still split are an entry's defect, reported in the plan (exploratory, research) or refused (`strict`,
  `study.range-split`). Before its first sweep a ranged study checks, in float64 and cached by the experiment's content,
  that the range reaches the build only through the choices entries report (the independence check: the build over the
  range against the build without it, every reported choice pinned, at the base and at both corners) and the two-builds
  trace form at a small move inside each range (`mime.study._ranges.MOVE`); either refuses as the entry's defect. The first consumer is
  `Drag.MRS`: its force points (and its default spacing) chosen at the range's corner with the most, monotone in the
  kinds' numbers measured so, sampled at nine points a number elsewhere (the screw's radius).
- **The plan's memory estimate** (`Memory`): on the device, XLA's buffer assignment of one chunk of the batched step
  compiled at the batch's own size and not run, and `fixed + each` a member from the line through two batch sizes
  (four and eight, or a smaller batch and half of it), checked at a third; no figure a member where the third lies more
  than a fifth of a member's bytes off the line. On the host, each member's own leaves, state, series and report.
- **The batched run** (`mime.build._maddening.batched_scan`): the group's step scanned and vmapped over the members'
  states (every field, `_meta` seeded from the built graph and carried per member through the chunks) and their varying
  leaves; a leaf every member shares enters once, unbatched. Flags are reduced per member inside the scan; the fields a
  recorded run keeps are kept per member. Chunked as a run is (`automatic_chunks`). Members of an experiment whose
  controller is called on the host run one after another on the group's one compiled step, each with its own law.
- **Run-level overrides** (`_override.py`): `built.run(d, override={...})`, `run_to(..., override=...)` and
  `mime run --override path=value`, on the same resolution; the point judged as a study's is, the nodes' own checks
  of their start included; refused with their routes where another compiled step would be needed, or where the
  override changes nothing the run computes from (`override.no-effect`); recorded in `run.json` and replayed by
  `mime verify --run`.

- **The drives' protocols** (`_protocol.py`): `sweep(..., protocol=FromRest())`, the default, holds every command at
  its point's value from the start; `protocol=Ramp(field.commands.frequency, to=..., rate=..., hold=...)` declares on
  each point a `drives.Ramp` starting from the point's value of the command, so the point is still the experiment with
  its values declared. The readings name their quantity and never mix them: `step_out(result, lag, axis)` the bracket
  from rest over a swept frequency at each value of the other axes (each member's mean spin over its frequency against
  `threshold=0.99`), `ramp_crossings(result, lag, window=...)` where each ramp's member steps out (`ramp up`) or is caught (`ramp
  down`), from the spin recorded every step over sliding windows, against the frequency the ramp commanded. Each
  member's start is read first, over the last window of the hold: a ramp up from a body that is not locked there, and a
  ramp down from one that is, report that start (`Crossing.read` false), never the start frequency as a step-out or a
  catch.

- **A study in a folder** (`_folder_run.py`, `_record.py`, `_content.py`, `_spec.py`): `study.run_to(folder)` writes a
  study folder (see `mime.results`) group by group and batch by batch, into a new or empty folder or one that holds the
  same study (anything else is refused and left untouched; `study.json` is written first and marks the folder as the
  study's), and a run into a folder that holds the same study resumes: a member is reused only if it is `done` under the
  same content key (its point, its group's structure key, the base's input hash, the run, and the code: every file of
  the installed `mime` package and of a package its classes come from, with the versions of MADDENING, OpenMaterica,
  JAX, jaxlib, NumPy and SciPy), so any change to MIME's files runs every member again. The spec file `study.mime.yaml`
  is the study as data (safe-loaded, closed, versioned, no code; `Study.from_spec`), `mime study plan` and `mime study
  run` (`--out`, `--pilot`, `--shard i/k`, by structure group) work on it, and `mime modal --study` adds a `study`
  entrypoint.
  In the research profile each axis needs its own reason (`sweep(..., reasons={...})`, `study.axis-reason`).
- **From a member to an experiment** (`_bridge.py`): `result.member(i).as_experiment(name)`, `Experiment.based_on(<the
  base>, name=name)` with the member's values declared, each with the reason `from study <id>, member <i>`, and a ramp's
  member with its ramp declared as the study declared it; a member outside a declared range is made without that range,
  as it was built. A study run on a ramp runs in memory: `run_to` refuses it
  (`study.protocol-folder`), since the folder and the spec file record no protocol yet.

## Boundaries

- It builds and runs experiments through `experiment` and `build`; it holds no code of the user's, so a study is data.
- It stays in MIME until a second library needs it.

## Status

Sweeps, overrides, the drives that ramp a command with the step-out readings, the study folder, the spec file and
`mime study`, resume and the bridge to an experiment are built (dev2, the study layer's first three phases), and
declared ranges with their first consumer (lane S's alignment with F0, its second PR); a study folder does not yet
record a protocol. Plots and comparisons, checkpoints and forks, variants, fits and sensitivities
come later.

Releases are explained in the [package manifest](../README.md#status).
