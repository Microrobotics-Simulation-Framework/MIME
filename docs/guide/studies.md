# Studies: many runs of one experiment

A study runs one experiment at many points. **A point is the experiment with the point's values declared in it**, so every
check the build makes of that experiment holds at that point: its validity lines, its material reads and its refusals, in
every profile. Batching is how the points run, never a meaning of its own. The step-out curve (sweeping the field through
step-out, from rest and under a ramp) comes with the drives that ramp the frequency.

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
```

A value the compiled step does not read (one a node consumes when it is built, without declaring it) is refused, since
sweeping it would change nothing, and so is an axis whose values change nothing a member runs from: no parameter leaf, no
initial state, nothing the compiled step is built from (`study.axis-inert`; for a run's override, `override.no-effect`).
A study sees a value only where a node holds it, so **a node must hold every value it consumes in a parameter leaf or a
declared static**: one kept in a Python attribute alone is invisible to the study's key, and its members would all run
the value the group was built with. Such an entry is a defect to report; its values run as experiments of their own,
`Experiment.based_on(...)`, each built. A point outside a validity limit is refused in every profile, and `study.run()` refuses
the study with every refused point at once; `sweep(..., drop_refused=True)` runs the rest and records the dropped points
on the result. With a controller called on the host between chunks, a group's members run one after another on its one
compiled step, each with its own law.

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
a value baked into a static or the structure is a rebuild (a study, or `Experiment.based_on`), and a numerical setting,
`dt`, the precision or a rung is a variant.

## What is bit for bit, and what is not

- A batch of one member is bit for bit its own unbatched scan; chunks of two steps and more are bit for bit the unchunked
  batch; an override at the base's own value runs the base's own program, bit for bit.
- A member of a batch of two or more is **not** bit for bit its own run: it is another compiled program. Members of script
  A's locked sweep lie within tens of eps of each field's scale of their own runs (the tests hold them to the tolerance
  committed before they first ran); past step-out, where nearby trajectories separate, a study is read by its outputs (a
  mean spin, a bracket), not trajectory by trajectory.
- Every bit-for-bit claim holds at one usable CPU count and one OpenBLAS thread count.
