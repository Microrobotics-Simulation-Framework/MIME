# Reading a run's results

A run's result is read by what the experiment declared, never by indexing a node's raw state. Each observation
declares its outputs: what each one is, its dimension and shape, and whether it has a value at every step or at the
end of a run. Every observation's entry page lists them.

## One result, from a run or from its folder

<!-- mime-fragment: a reader's experiment continued from a build, shown for the accessors it calls -->
```python
swim = experiment.observe(SwimSpeed(umr, along=tube.axis, after=s(2.5)))
built = experiment.build()

result = built.run(s(5.0), record=[swim.speed], every=ms(5))
result[swim].mean_speed                     # a Reading: its value, its window and the flags raised over it
result[swim].mean_speed.to(mm_per_s)        # a number, in mm/s
speed = result[swim].speed                  # a Series, kept because record= names it
speed.time, speed.values                    # array quantities, in s and in m/s
speed.between(s(2.5), s(5.0)).mean()        # a window by time, reduced by the rule the node declares

built.run_to("run/", s(5.0), record=[swim.speed])
RunResult.read("run/")                      # the same accessors, from the folder (or what modal volume get fetched)
```

`result["swim_speed:umr"]` is the same as `result[swim]`: an observation is indexed by its handle or by its id.

## Outputs at the end, and series

- **An output at the end** is reduced by the graph, in the observation's own node, so every run has it whatever it
  records. It is a `Reading`, over its window.
- **A per-step output** is a series only when the run records it: `record=[swim.speed]`, or `history=True`, which
  keeps every field at every step and so records every per-step output. Reading one the run did not record is
  refused, and the refusal names the `record=` that keeps it. `every=` spaces the samples kept, by time, a whole number
  of base steps counted from the graph's start; it does not reduce the cost of recording, which is the whole state's
  history per step whatever is kept.

## Windows by time

Every window follows one rule, the observations' own: a window from `start` to `end` holds the steps whose end lies
after `start` and not after `end`. A sample is the state at the end of its step, and its time is that step's count
from the graph's start times the base step. The two ends are decided differently on purpose:

- **The start is decided as a node decides its `after`, in the run's float type**: the step's end (its count times the
  base step) and the start are each rounded to that type and then compared. In float32 a start on the grid can
  therefore fall on either side of the step that ends there: at common base steps more than half of the on-grid starts
  take the tied step, which a float64 comparison would not. A window of your own, `speed.between(start, end)`, shares
  no node with the output it is read beside and is decided the same way, so it includes the step ending at `start` or
  leaves it out exactly as `SwimSpeed(..., after=start)` does in that run; in float64 the comparison is the exact
  one, and a start off the grid by more than the type's rounding is unambiguous.
- **The end is on the grid**: a time within a tiny relative tolerance of a step's end is that step's end (the
  tolerance `steps_in` reads durations with). A node has no end to match; the run's end, and any end you write, take
  the step that ends there.

An output the graph reduces states its window and its rule (a mean, or an integral), so the series over the same
window, reduced by the same rule, gives the same value, to the rounding of the node's own sums, on the grid as off it.
The rules are a closed set today: a mean and an integral (`Reduction`). An observation whose graph reduces by another
rule (a maximum, a root mean square) needs a MIME edit, which registers the rule with its series-side reduction beside
it; that is not done yet.

## Units

A result's quantities are SI arrays: `.to(unit)` reads one in any unit of its dimension (`mm_per_s`, `um_per_s`,
`ms`) and refuses a unit of another. The graph itself sees SI floats only.

## Flags

An entry declares the flags its nodes raise (the surrogate drag's `clamp_fired` and `tilt_fired` are two: the first for
a body clamped near the wall, the second for a body whose axis has left the tube's by more than the bore allows). Every reading and series
carries the flags raised over its window, and every export and printout shows them; a flag never blocks a read. A
run that records anything records every flag at every step. A run that records nothing keeps no step's state, and
reduces each flag over its steps as it runs, inside the compiled scans beside the state: how many steps raised it, and
the first and the last (the manifest's `whole_run`, with `recorded` saying "the whole run"). A flag raised at any step
is reported raised, with its count and its first step, whether or not it is clear at the last; one raised at none is
not shown. That reduction answers for the run's own steps only, so a window that is a part of them reads a flag
**unknown** (the manifest's `was_raised` is `null`, with `raised_at_last_step` beside it for what is known), never "not
raised": a flag that fired earlier and is clear at the last step looks the same at the last step. The same holds for a
result without the reduction (a run folder written before it, a live session's). Record the run to know a window.

A segment whose dynamics carry a tree raises one flag more where its description limits its joints' speeds:
`<segment>.beyond_speed_limit`, at a step where a joint's speed is above its limit. It is read after the run from the
joint velocities the segment holds, against each joint's limit (the manifest lists them, `null` for a joint with
none; a URDF's velocity limit of zero is none, as MoveIt reads it), and adds nothing to the graph; a dynamic arm does
not apply the limit, so the flag is the record that the run went where the real mechanism would not.

### The Reynolds-number watch

A model that leaves out the fluid's inertia is declared up to a Reynolds number, and its entry's page states the bound
and its basis (the drag's two rungs are Stokes flow, the limit of vanishing Reynolds number). After every run MIME reads
the number the body moved at and compares it with the bound. The result is a flag in the same sense as the above: in
`result.watches` and in `result.text()`, in `outputs.json` under `watches` and in the `.mat` file's `watches` struct, and
in the build report, whose post-run section lists what each run is compared with. It is never a refusal, and it is no
part of the graph: the derived hash is what it would be without it.

- **Two numbers are read**, both by the hull's diameter and the medium's density and viscosity: the translation's, from
  the body's speed, and the turning's, from the speed of the hull's surface at its radial reach (a turning screw's
  surface moves far faster than the screw swims). The larger is compared with the bound. A run that kept its history
  (`history=True`) is read at the peak over the run; one that did not is read at its last step alone and says so, as a
  flag does: it knows nothing of the steps before.
- **It says how much is left out where that is known.** For a sphere's translation, Oseen's first correction raises the
  drag by three sixteenths of the Reynolds number. That is the leading term of an expansion in the Reynolds number: good
  below a Reynolds number of order one, and an overestimate above it. For a turning sphere the correction is of second
  order, and for any other body none is known; the watch says which it is. The bound itself is a convention taken
  from a sedimenting sphere's correction (about a fiftieth of its drag at the bound): a force-free swimmer has no such
  correction and none is measured for any body, so the bound says how far a run is from Stokes flow, not how much its
  answer is off. The peak includes the start-up, which can lie above the steady value.
- **A watch that cannot compute says why.** The medium's density is read where the experiment declares it, or an entry
  has read it (`Gravity` does); otherwise the watch reads "not computed" and names the declaration to add. An entry of
  the hydrodynamic family that declares no bound is watched all the same, and the watch says it has none to compare
  with. Nothing passes silently.

### The containment watch

The contact `0.1.0` (kept registered for reproductions, `mime.interactions.KEPT`) holds a body's centre by its extent and
never reads its orientation, so a body with a hull that turns across the region it is held in (a screw longer than the
tube is wide) reaches a pose its hull cannot take there, and the run goes on with the body through the wall. Nothing in
the graph sees it: script A's isotropic effective inertia keeps its axis along the tube, and the same body given its own
inertia tensor turns across in a fraction of a second. After every run MIME reads the body's pose and the surface's signed
distance at the hull's points. The result is in `result.containment` and in `result.text()`, in `outputs.json` under
`containment` and in the `.mat` file's `containment` struct, and in the build report beside the Reynolds watch. It is
never a refusal and no part of the graph, in exploratory and research (strict is reserved and cannot be built), as the Reynolds watch is: a run that went through the wall is
a result its author may want to read, so the watch says where, and the run is not discarded.

- **What is read.** The deepest point of the hull past the surface (`depth`, positive past it), the step it was deepest at
  and the first step the hull was past the surface by more than rounding. The flag rises past the surface, not the
  contact's margin: a hull that comes within the margin without passing the wall is held as the contact leaves it. A run
  that kept the body's pose at every step (`history=True`, or `record=`, which keeps the pose for this) is read at every
  step; one that kept nothing is read at its last step alone and says so: it knows nothing of the steps before it.
- **What it bounds.** The depth is of the hull's samples, which fall short of the body by up to `shortfall`, which the
  text quotes (tens of micrometres for the screw of script A at the watch's spacing, several times what the hull contact's
  finer spacing leaves): the body may be that much deeper, so `shortfall` bounds how far a flagged depth can understate
  the true one (it is measured over finitely many directions and may miss the largest by a little), and the watch
  cannot say a hull is inside that is outside by more than that. A run that kept no pose says "inside at the last step; earlier steps were not kept, so
  nothing is known of them": a hull that crossed the wall and came back before the last step reads as inside. A body
  with no hull (no parts) is held as a sphere of its extent, which that contact holds exactly, and is not watched. A
  body with a hull that the watch cannot check (the surface's shape takes one point at a time, the hull's kind gives no
  samples, or the body's dynamics publish no pose) is listed in the build report as not watched and read as not
  computed, with the reason.
- **The fix is in the text.** `Contact.Projection(margin=...)` holds the hull, its orientation read.
- **Cost.** No node, no edge: the graph and its derived hash are what they would be without it. It is read when first
  asked for (`result.containment`, `result.text()`, the manifest, the exports and the end-of-run summary), so a run that
  never looks pays nothing: then it evaluates the surface at the convex hull's vertices of the hull's samples (at a
  coarser spacing than the hull contact's, a share of the hull's size) at every step it reads, in blocks of bounded size
  and one compiled function. The hull's points are made once per process for the same hull, not at the build.

## The energy ledger's account

With `exp.observe(EnergyLedger())`, `result.energy` is the run's account of energy from the build's start to the
run's end (`None` without the ledger, and for a run read back from a folder, which keeps no law's parameters):
`result.energy.text()` says, in plain words, what each body stores, what each dissipative law lost, what was supplied,
named by what supplies it, and each potential's change. What was supplied is taken from the dissipation side, what was
lost plus the change in what is stored; the work the supplied laws paired in the steps differs from it by the scheme's
own numerical terms, which `text(details=True)` shows apart, never as a loss, with each law's work on each body and
the checks: each body's step identity at its worst step against its rounding, the work that arrived unattributed
(exactly zero where every delivery is a law's), each potential's paired work against its energy's change from the run's
two ends, and the books' balance against its rounding, each receiver's storage's jumps between steps, each workless
exchange's work summed over its receivers (a rotor and the arm carrying it: the exchange's own discretisation error,
reported, not judged), and each coupling group's verdict with its residual. `result.energy.efficiency(useful)` is a
useful work as a share of what was supplied. The build report's `energy` lines say how the build metered the
experiment ([reading a build report](guide/build_report.md)).

**A coupling group's verdict.** `result.groups` gives each coupling group's verdict at the run's last step, by its
members, from MADDENING's own (`converged`, or the passes after which it stopped at its cap), in every profile, with or
without the ledger: energy cannot tell a capped group from a converged one.

## Exports

- `result.table()` is a mapping of columns, each name carrying its unit (`time [s]`, `swim_speed:umr.speed [m/s]`),
  and a `flags` column: `pandas.DataFrame(table)` and `polars.DataFrame(table)` take it as it is. MIME depends on
  neither.
- `result.to_csv(path)` writes that table, its header the names with their units.
- `result.to_npz(path)` writes every array with the manifest inside it, which NumPy reads alone.
- `result.to_mat(path)` writes the declared outputs and flags as a MATLAB v5 file through `scipy.io.savemat`: a struct
  per observation holding a struct per output (`unit`, `dimension`, `value`, or `time` and `values` for a recorded
  series), `flags` (a flag the run saw only at its last step is `unknown`, never `false`), the manifest as a JSON
  string, and `name_map` from MATLAB's names (`swim_speed_umr`) back to the ids, which are not MATLAB names. It does not
  hold the raw state (`final`, `history`: those are in the NPZ), and refuses names that collide or run past 63
  characters and any variable of 2 GiB or more, naming the NPZ. The tests read it back with `scipy.io.loadmat`; MATLAB
  itself has not been run on one.
- `run_to` and `mime run` write `outputs.json` beside `outputs.npz`: what every array is, each output's unit, window
  and flags, and each output at the end written out, so a run folder is readable without MIME.

## A study's result, and its folder

`Study(exp).sweep(...).run()` returns a `StudyResult`: one member axis, in the order of the points, with
`result.coordinates[path]` (each axis's values and units), `result[swim].mean_speed` (an output at the end over the
members), `result.member(i)` (a `RunResult`) and the union of the members' flags. Its exports put the member axis first:

- `result.table()` is long, one row per member and sample. The coordinate columns come first, each with its unit in the
  header (`field.commands.frequency [Hz]`), then `member` and `group`, then the member's own columns (`time [s]`, each
  series with its unit) and `flags`. Members sampled at other times (another `dt`, another `every=`) are in one table,
  each at its own samples, with NaN where a member lacks a column. `to_csv(path)` writes it.
- `result.to_npz(path)` holds the coordinates, `groups`, `points`, every member's arrays under `m<point>/` as a run's
  NPZ holds them, and the manifest.
- `result.to_mat(path)` is a MATLAB v5 file through `scipy.io.savemat`: `coordinates` (each axis's values with its
  `unit`), `groups`, `points`, a struct per observation holding a struct per output with its `unit` and `dimension` and
  its `value` over the members (an output at the end) or its `time` and `values` (members first where the members
  share a time axis, a cell array of the members' own where they do not), `flags` by member (`true`, `false`,
  `unknown` or `not declared`), `study`, `manifest` and `name_map`. It holds the observations every member has; one
  only some members have is left out (read those members alone). The tests read it back with `scipy.io.loadmat`;
  MATLAB itself has not been run on one.
- `result.grid()` reshapes a study that is every combination of its axes' values (`grid.reshape(result[swim].mean_speed)`
  is an array with one dimension per axis), and refuses a study that is not, naming the combinations that are not
  there. `result.manifest()` is the folder's `study.json`.
- `result.order(output, along=DT)` is the observed order of convergence of an output at the end over three levels of
  a step, and the value it extrapolates to. A study of exactly the three levels, its other axes constant. It is refused
  (`study.order`) where the levels are not in the asymptotic range: the two refinement ratios differ by more than a tenth
  of the smaller, the values cross their limit or do not move, the order found lies outside the tenth to ten, or the
  ratio of the successive differences is more than a tenth from the ratio a scheme of the formal order gives at those
  levels (`r^n` for a ratio `r`). The formal order is `expected=` where it is given, and otherwise the whole number
  nearest the order found: a scheme's formal order is whole unless it is stated, so a fractional one is given with
  `expected=`. A first-order scheme whose second-order term still dominates the levels reads as an order between the
  first and the second, and is refused (`mime.results._order` and its tests give the series). Three levels fit three
  unknowns, so the range is judged on the levels themselves, and a contamination that moves the ratio by less than the
  tenth is not seen: a fourth level is the way to see it.

`study.run_to(folder)` runs the study into a **study folder**, group by group and batch by batch, and returns what the
folder holds:

<!-- mime-fragment: a layout of files, shown and never run -->
```text
study.json                 the manifest, format mime-study-outputs/1
study.mime.yaml            the study's spec file
experiment/                the base experiment: a saved folder, byte for byte what save() wrote
groups/<k>/outputs.npz     every member of group k: its arrays under m<point>/, its manifest and report as JSON text,
                           and the content key it ran under
groups/<k>/report.json     the build report's summary of the group's first member
groups/<k>/experiment/     the group's own build, for a group whose compiled program differs from group 0's
```

The folder is readable without MIME (JSON, YAML and NPZ). `study.json` holds the base experiment's name and input hash,
the axes with their units and reasons, every point, the run, the groups with their structure keys, the count of
programs the plan compiles for each and the sha256 of each group's `outputs.npz`, each member's status, overrides and
content key, the points a `drop_refused` sweep dropped, the seed policy (every member runs the experiment's own seed; no
entry of this release draws noise) and the lineage. Its schema is closed: a key it does not know is refused
(`study.manifest`), and so is a newer format (`study.format`), a file or folder it does not hold (`study.stray-file`) and
a folder that is no study's (`study.not-a-study-folder`). The folder takes no saved-form step: `experiment/` and each
`groups/<k>/experiment/` are ordinary saved folders at the current form.

`StudyResult.read(folder)` reads it, and reads a study that was stopped: `result.missing` lists the points not there and
`result.text()` says how many. A group's arrays are tied to the manifest: each member marked `done` must have its
entries in its group's `outputs.npz` under the content key the manifest gives it, and the file must be the one whose
sha256 the manifest records. A group file from another study (copied in from another folder), one that lacks a done
member's entries, and one whose arrays were changed are each refused (`study.group-file`), naming the group, and the
members whose entries are not there are counted in the refusal. Members are read from their group's NPZ when first asked
for.

**Where a study is written.** `run_to(folder)` writes into a new or empty folder, or into one that holds the same study.
A folder that holds anything else (a saved experiment, a file of yours) and no `study.json` is refused
(`study.folder-not-empty`), naming what it holds, and nothing in it is touched; so is one that holds another study
(`study.other-study`). The run writes `study.json` first, every member `pending`, so the folder is marked as the study's
before anything else goes into it; then the spec file, then the base (under `experiment.tmp`, renamed into place), then
the groups. A run stopped at any point resumes in the same folder. The run's temporaries have fixed names (each file or
folder it writes, with `.tmp` added, where it writes it), and a resumed run removes those and nothing else; a file of
yours with such a name elsewhere, or inside `experiment/`, is never touched, and one at the top of the folder is refused as
stray.

**Resume.** A run into a folder that holds the same study (the same `id`: the base's input hash, the axes with their
reasons, every point, the run and the seed policy) stages each batch's arrays into its group's NPZ, records the staged
file's sha256 in `study.json` with the batch's members marked `done`, and renames the file into place, so a stopped run
leaves a folder that says what is done (a run stopped between the last two steps leaves the staged file, which reads and
which the next run renames). Run again, it reuses a member only if it is `done` under the same **content key** and its
entries in the group's NPZ carry that key. The key holds the member's point, its group's structure key, the base's input
hash (which covers its assets), the run, and the code: every file of the installed `mime` package by its sha256 (the whole
tree, not only what the member imports; bytecode caches aside) and every file of each other package a class of its
nodes, entries or law comes from (a lab's plugin, a script's own file), with the versions of MADDENING, OpenMaterica,
JAX, jaxlib, NumPy and SciPy and the precision. It is coarse on purpose: any change to MIME's files runs every member
again. What it cannot see is code with no file (a law defined in a notebook's cell): after changing one, run the study
into a new folder. Since the point and the run are part of the study's `id`, a folder never holds a member whose point or
run changed (another point, another duration, a batch size or a reason is another study, refused); within one folder it
is a change of code, of those versions or of a group's structure that runs a member again. A member whose key changed is
run again, never reused. A group file that is not the one `study.json` records is refused (`study.group-file`) and never
written over; moved out of the folder, its group's members run again. The batches are the plan's; a planned batch cut by
a member that is reused runs the rest of it as a smaller batch, whose last bits are not those of the larger one. Bits are
compared at one usable CPU count and one BLAS thread count, as everywhere.

**The spec file**, `study.mime.yaml`, is the study as data, written into the folder in its normal form and read by
`mime study` and `Study.from_spec`:

<!-- mime-fragment: a spec file, shown for its keys and never run -->
```yaml
format: mime-study/1
base: {folder: experiment, input_hash: 'sha256:...'}   # the saved experiment; a base with another hash is refused
combine: grid                                          # grid, or along (the i-th value of each axis together)
axes:
  field.commands.frequency:
    values: [10 Hz, 20 Hz, 30 Hz]                      # a number and a unit of mime.units, or a name
    reason: the frequencies below the from-rest step-out bracket   # one string per axis
  dt:
    values: [0.5 ms, 0.25 ms]
duration: 1 s
record: [swim_speed:umr.speed]
every: 1 ms
batch: 16
drop_refused: false
seed: {policy: experiment}
```

It is read with YAML's safe loader and holds no code; a key given twice, a key it does not name, another format and a
base of another input hash are refused (`study.spec`).

<!-- mime-fragment: commands on a spec file of the reader's own, shown for their flags -->
```sh
mime study plan study.mime.yaml
mime study run study.mime.yaml --out my_study --pilot
mime study run study.mime.yaml --out my_study_shard --shard 0/2
```

`mime study plan` prints `study.plan()` (and exits one where a point is refused). `mime study run` writes the folder and
resumes it; `--pilot` runs the first chunk of the first group it runs twice and prints what a member costs, once
compiled; `--shard i/k` runs only the structure groups whose number is `i` modulo `k`, the others left `pending` in that
folder. Sharding is by group, as `mime study plan` lists them: a study whose points all compile alike is one group and
cannot be sharded, and a `k` above the number of groups is refused (`study.shard`), since its extra shards would run
nothing. Shards run one after another into one folder add up to the whole study. Shards run at the same time go to a
folder each: in one folder each would rewrite the other's `study.json`. Merging such folders is not built.

**One reason per swept axis in the research profile.** A swept value is a declared value, and the research profile asks
each declared value for its basis, so a research study needs a reason for each of its axes, one string per axis and not
one per point: `sweep(..., reasons={field.commands.frequency: "..."})`, or `reason:` under the axis in a spec file. Without
one it is refused (`study.axis-reason`), naming the axis. The reason is in `study.json` (the axis, and each member's
overrides) and is the reason of the value each member declares. The exploratory profile declares each value with the
generic study reason where an axis has none.

**From a member to an experiment.** `result.member(i).as_experiment(name)` is `Experiment.based_on(<the study's base>,
name=name)` with the member's values declared in it, each value with the reason `from study <id>, member <i>` (and its
axis's own); the new experiment, not saved until you save it, is that member's run when run alone, and its saved folder
names the base it was made from. It needs a study folder (or `base=` naming the saved base). A precision the study
varied is a setting of the process and cannot be declared (`study.bridge-precision`). The comparison the tests hold it to is
the sweep's acceptance (a member against the same point run alone, within tens of eps of each field's scale while the
body is locked), not an accuracy claim about the model.

## A run that was stopped

A run is compiled chunks, between which Python gets control back, so Ctrl+C reaches it. On `mime run` the first press
stops the run at the end of its chunk and writes the folder as the steps it reached; `outputs.json` and `run.json` hold
`stopped`, with the reason, the steps and the time reached and asked for, `RunResult.read(folder).stopped` says the
same, and `result.text()` opens with a `STOPPED` line. The second press quits at once and writes nothing; `outputs.json`
is the last file of a folder, so a folder quit part-way through is not one `RunResult.read` accepts. In a script,
`built.run(...)` is chunked the same way: a `KeyboardInterrupt` arrives between chunks (build the experiment again after
one), and `built.run(..., stop=f)` or `built.run_to(..., stop=f)` ends the run at the first chunk after which `f(progress)`
returns a reason (a `mime.control.Progress`: the chunk's index, the steps taken, the time and the step the run ends at),
writing it as a stopped run. A stopped run is a faithful run of the steps it reached, which is why the manifest keeps
its version: a reader that does not know `stopped` reads those steps. A run with a controller or a hook is chunked as
before, by its control period or by `chunk=`.

## A run in a notebook

In a notebook (Jupyter, or Colab: rich's own test of one), `built.run(...)` and `built.run_to(...)` show the live
display `mime run` shows, drawn through the notebook's output: the first chunk compiling, then the bar with the steps,
the simulated time, the speed and an estimate of the time left, and, in `run_to`, a phase while the run folder is
written, whose saved form runs its reference steps on the run's own graph. `built.save(folder)`
shows its phase too. `progress=False` turns it off; outside a notebook nothing is shown, and `progress=True` asks for
it on a terminal as `mime run` draws it. The display is drawn from `stop`'s place after each chunk, a `stop` of your own
still asked after it: it changes neither the chunks nor anything the run computes. It is drawn through `ipywidgets`,
which Colab has; where it is missing, rich says to install it. To stop a run there, interrupt the kernel: nothing of the
run is kept, and the experiment is built again before it runs on.
