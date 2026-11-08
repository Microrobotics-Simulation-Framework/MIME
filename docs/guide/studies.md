# Studies: many runs of one experiment

A study runs one experiment at many points. **A point is the experiment with the point's values declared in it**, so every
check the build makes of that experiment holds at that point: its validity lines, its material reads and its refusals, in
every profile. Batching is how the points run, never a meaning of its own. Its protocol says how each point's commands
move in time: held from rest, or ramped (the step-out curve, below).

## A sweep

`examples/studies.py` sweeps script A's field below the frequency where, from rest, its body stops following:

<!-- mime-include: examples/studies.py#sweep -->

Each axis is named by a handle's typed path, as the experiment declares it:

- an apparatus's command, `field.commands.frequency` (a namespace of its own, so that no field's name collides with a
  command's);
- a thing's value, as `declare` takes it: `"umr.mass"`, `"water.viscosity"`;
- a law's setting, through the handle `attach` returns (still the law's id, a `str`): `drag.lubrication.epsilon`;
- a body's initial state, `umr.initial.position`;
- the run's base step and precision, `DT` and `PRECISION` from `mime.study`.

`exp.handle("field")` gives the handle of what an experiment declares under a name, for one that came with none (loaded, or
made by `Experiment.based_on`). Every path is also accepted as its dotted string, which is how it saves, and a misspelt one
is refused with the names there are. `grid({...})` takes every combination of the axes' values, `along({...})` takes them
together, and a list of rows is a table of points.

## The plan, before anything runs

`study.plan()` plans every point in full, then assembles each one's graph with nothing compiled, to find the compiled
program it needs: the built graph with its parameter values removed, its static data, an in-graph controller's traced step,
and the precision. Points that need one program are a **group**: built once, compiled once, its members run together by
`jax.vmap` over their own parameter leaves and initial states. A leaf every member shares (a table, a network's weights)
enters the step once, never copied per member. The plan says how many groups and compiles there are, why each group split
from the first (a static a node derives from a value, a changed node set, `dt`, the precision), every refused point with
every reason, and the memory each group needs.

On the device, the memory of a batch is XLA's buffer assignment of one chunk of the batched step, compiled at the batch's
size and not run, so the plan measures it rather than extrapolating it. Beside it the plan gives `fixed + each` a
member, for choosing `batch=`: the line through two batch sizes where the assignment grows by one member's bytes a
member (four and eight members; a batch under eight at its own size and half of it), checked at a third. At one member
XLA assigns otherwise: on the MRS screw in a tube the step grows far less from one member to two than from each member
to the next once there are three, so a line through one and two members gave a small fraction of the bytes a member
costs. Where the third size lies more than a fifth of a member's bytes off the line, the plan gives no figure a member.
On the host, each member holds its own leaves and state, its series and its report. The plan pays those compiles. Its
print below leaves the estimate out (`study.plan().text(memory=False)`), since the bytes depend on the machine, the
jaxlib and the Python it runs on; `print(study.plan())` gives them. It printed:

<!-- mime-output: examples/studies.py -->
```text
4 points over field.commands.frequency; 1 group, 1 compile to run it (planning compiled the group's build, and its batched step 3 times for the memory estimate); nothing refused
  group 0: 4 points, batched in 1 batch of 4, 1 compile
10 Hz: swim speed 2.92 mm/s over the second half, 0.292 mm per turn
20 Hz: swim speed 5.86 mm/s over the second half, 0.293 mm per turn
30 Hz: swim speed 8.90 mm/s over the second half, 0.297 mm per turn
40 Hz: swim speed 12.01 mm/s over the second half, 0.300 mm per turn
OVERRIDDEN for this run: field.commands.frequency = 20.0 Hz
step-out from rest: last locked 43 Hz, first unlocked 43.5 Hz
41 Hz 1.000, 41.5 Hz 1.000, 42 Hz 1.000, 42.5 Hz 1.000, 43 Hz 1.000, 43.5 Hz 0.037, 44 Hz 0.034, 44.5 Hz 0.033, 45 Hz 0.032, 45.5 Hz 0.030, 46 Hz 0.029
```

A value the compiled step does not read (one a node consumes when it is built, without declaring it) is refused, since
sweeping it would change nothing, and so is an axis whose values change nothing a member runs from: no parameter leaf, no
initial state, nothing the compiled step is built from (`study.axis-inert`; for a run's override, `override.no-effect`).
A study sees a value only where a node holds it, so **a node must hold every value it consumes in a parameter leaf or a
declared static**: one kept in a Python attribute alone is invisible to the study's key, and its members would all run
the value the group was built with. What splits a group is what a node makes from a value when it is built: a static it
declares, a count or a layout (the MRS rung's force points from its spacing, a hull's counts from the contact's spacing
or margin). The MRS rung's settings and wall tables are leaves, and so is the matrix it holds, made at build with the
leaves it was made at beside it, so a sweep of its blob is one group. A geometry's number is an axis, named as
`exp.source` names it (`"ball.parts.shell.radius"`, `"tube.inner_radius"`); a kind's setting that is not one number (an
ellipsoid's semi-axes) is refused as one. Without a declared range, a sweep of a radius splits where the MRS rung's force
points change; with one, it does not (below). The two-builds check (`mime.testing.two_builds`) is how an entry shows that no value of it is held as a constant of the
program. Such an entry is a defect to report; its values run as experiments of their own, `Experiment.based_on(...)`,
each built. A point outside a validity limit is refused in every profile, and `study.run()` refuses
the study with every refused point at once; `sweep(..., drop_refused=True)` runs the rest and records the dropped points
on the result. With a controller called on the host between chunks, a group's members run one after another on its one
compiled step, each with its own law.

## Declared ranges

A value a study will move, and whose choices would otherwise split it, is given a range in the experiment:

<!-- mime-fragment: one declaration of an experiment, shown for its form -->
```python
exp.declare_range("ball.parts.shell.radius", mm(1.0), mm(1.15), reason="the design range of the ball's radius")
```

Every choice an entry makes from that value is then made over the range instead (the MRS rung's force points at the
range's corner with the most, the samples moved with each member's own radius), so a sweep across it is one group and
one compile, and an override inside it runs as data on the built step. The price is that every member, and a run of the
experiment alone, takes the range's discretisation: the build report says what the range moved and what it costs (the
force points the range chose, at which of its corners, against the count at the experiment's own value, and the dense
solve's size against the other's). The range is saved with the
experiment, so a member made an experiment of its own (`member(i).as_experiment(...)`), `mime verify --run` and the spec
file reproduce it with the range.

A point outside the range is built for its own value, a group of its own, and the plan says why; a run's override
outside it is refused (`override.outside-range`). A range is refused on `dt`, the precision, a rung or a numerical
setting (each is a variant), on a value that is not one quantity, where the experiment's own value lies outside it, and
where the experiment is refused at one of its corners: the build plans each corner, so each validity line is judged
over the range. At most six ranges are declared, since the corners double with each. Before its first sweep, a ranged study checks, once for
the experiment's content, that the range reaches the build only through the choices its entries report, and that a
value moved inside the range moves no constant of the compiled step; a ranged study's plan reports the checks and their
cost. Members inside a range that still split are an entry's defect: reported in the exploratory and research profiles,
refused in strict.

## The result

`study.run()` returns a `StudyResult` (`mime.results`), one member axis in the order of the points:

<!-- mime-include: examples/studies.py#read -->

- `result.coordinates[path]`, each axis's values over the members, with their units;
- `result[swim].mean_speed`, an output at the end over the members, and `result[swim].speed`, a recorded series stacked
  over them where they share its time axis (refused otherwise, with the route to `member(i)`);
- `result.member(i)`, one member as a `RunResult`, with everything a run's result reads;
- `result.groups`, the group each member ran in, and `result.flags`, the union of the members' flags, where a flag a
  member's build does not declare reads `not declared`, never "not raised".

`result.table()`, `to_csv`, `to_npz`, `to_mat`, `grid()` and `order()` read it further, and `study.run_to(folder)` writes it to a folder
that a stopped run resumes from: see [reading a study's result](../results.md).

## The step-out curve

A body follows a rotating field up to a frequency, its step-out, and above it slips and rocks. Where that frequency lies
depends on how the field gets there, so a study's protocol says it, and every reading names the quantity it is:

- **from rest** (`protocol=FromRest()`, the default): each point's field turns at its frequency from the start, and the
  body, starting at rest, is captured or not;
- **ramp up** (`protocol=Ramp(field.commands.frequency, to=..., rate=..., hold=...)`, rising): a body locked at the
  point's frequency follows the field as it speeds up, until it steps out;
- **ramp down** (the same, falling): a body that rocks at a high frequency is caught as the field slows.

`examples/studies.py` sweeps script A's field through step-out from rest, with `observations.PhaseLag` observing the
body's moment against the field and its spin about the field's axis:

<!-- mime-include: examples/studies.py#step_out -->

`step_out(result, lag, axis)` reads each member's mean spin over the window against its frequency: a member is locked
where that ratio is at least its `threshold`, by default `0.99`. Below the bracket the body turns with the field;
above it the body rocks, its mean spin a few hundredths of the field's. On the reproduction arm in float64 the
bracket is `43.25` and `43.5` Hz, and at nine tenths of the amplitude `41.0` and `41.25` Hz
(`tests/study/test_step_out.py`). A ramp is read by `ramp_crossings(result, lag)` from the spin recorded every step,
over sliding windows against the frequency the ramp commanded:

<!-- mime-fragment: a ramp down, minutes of script A, longer than a page's example; its figure is the test's -->
```python
result = Study(observed).sweep(
    grid({frequency: [Hz(100.0)]}), duration=s(284.0), record=["phase_lag:umr.spin"],
    protocol=Ramp(frequency, to=Hz(5.0), rate=Hz(0.25) / s(1.0), hold=s(3.0))).run()
(caught,) = ramp_crossings(result, "phase_lag:umr")
print(caught.text())   # member 0, ramp down: is caught at 31.73 Hz (276.087 s), by 0.2 s windows
```

Held at a high frequency from rest the body rocks; ramped down slowly it is caught below the from-rest step-out
(`tests/study/test_ramp_relock.py`), and a body locked at a lower frequency and ramped up stays locked far past it,
until a frequency far outside the drag's declared Reynolds number. The whole curve lies outside it, the from-rest
and ramp-down readings as well (the build warns of it before the run, and the post-run watch after it), which is about the
model's limits, not the body's. `ramp_crossings` reads the state a ramp starts from over the last window of its hold: a
ramp up from a body that is not locked there, or a ramp down from one that is, reports that start and no crossing, since
its start frequency would be the from-rest figure under another name (or no catch at all). A printed crossing does not
carry the drag's validity status: read it from the build report's warning and the watch. The
field must integrate its frequency for a ramp to mean what it says: `UniformRotating` 0.2.0 does, and 0.1.0, kept for
saved folders, refuses a ramp. These figures are acceptance tests of the sweep and the inertial body, not claims about
where a real body steps out.

A study run on a ramp runs in memory, `.run()`: a study folder does not record a protocol yet, so `run_to` refuses it
(`study.protocol-folder`) rather than write a folder that would read back as run from rest. A ramp's member made an
experiment, `result.member(i).as_experiment(name, base=...)`, declares its ramp from the member's value, as the study
did.

## One run's overrides

`built.run(duration, override={...})` runs one point alone, on the build's own compiled step: its overrides are declared,
its plan is judged again, and every output is stamped:

<!-- mime-include: examples/studies.py#override -->

On the command line:

<!-- mime-fragment: a command line on a saved folder of the reader's own, shown for its flags -->
```sh
mime run <folder> --out <run> --duration 1s --override field.commands.frequency=20Hz
```

The run folder's `run.json` records each override (its path, its value with its unit, its SI value and the leaves it writes),
`report.json` is the point's, and `experiment/` stays the base's saved form; `mime verify --run <run>` makes the run again
from them and compares. The point is judged as a study's is, the nodes' own checks of their start included, so a start
the build refuses is refused when the run is asked for. What would need another compiled step is refused with its route:
a value baked into a static or the structure is a rebuild (a study, or `Experiment.based_on`), a value outside a declared
range is a study's point of its own, and a numerical setting, `dt`, the precision or a rung is a variant.

## What is bit for bit, and what is not

- A batch of one member is bit for bit its own unbatched scan; chunks of two steps and more are bit for bit the unchunked
  batch; an override at the base's own value runs the base's own program, bit for bit.
- A member of a batch of two or more is **not** bit for bit its own run: it is another compiled program. Members of script
  A's locked sweep lie within tens of eps of each field's scale of their own runs (the tests hold them to the tolerance
  committed before they first ran); past step-out, where nearby trajectories separate, a study is read by its outputs (a
  mean spin, a bracket), not trajectory by trajectory.
- Every bit-for-bit claim holds at one usable CPU count and one OpenBLAS thread count.
