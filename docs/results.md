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
inertia tensor turns across within 0.35 s (measured: 45 degrees at 0.34 s). After every run MIME reads the body's pose and the surface's signed
distance at the hull's points. The result is in `result.containment` and in `result.text()`, in `outputs.json` under
`containment` and in the `.mat` file's `containment` struct, and in the build report beside the Reynolds watch. It is
never a refusal and no part of the graph, in every profile, as the Reynolds watch is: a run that went through the wall is
a result its author may want to read, so the watch says where, and the run is not discarded.

- **What is read.** The deepest point of the hull past the surface (`depth`, positive past it), the step it was deepest at
  and the first step the hull was past the surface by more than rounding. The flag rises past the surface, not the
  contact's margin: a hull that comes within the margin without passing the wall is held as the contact leaves it. A run
  that kept the body's pose at every step (`history=True`, or `record=`, which keeps the pose for this) is read at every
  step; one that kept nothing is read at its last step alone and says so: it knows nothing of the steps before it.
- **What it bounds.** The depth is of the hull's samples, which fall short of the body by up to a few micrometres at the
  spacing the hull contact would use (`shortfall`, in the text): the body may be that much deeper, and the watch cannot
  say a hull is inside that is outside by more than that. A body with no hull (no parts) is held as a sphere of its
  extent, which that contact holds exactly, and is not watched; a surface whose shape takes one point at a time is not
  watched either.
- **The fix is in the text.** `Contact.Projection(margin=...)` holds the hull, its orientation read.
- **Cost.** No node, no edge: the graph and its derived hash are what they would be without it. After the run it
  evaluates the surface at the convex hull's vertices of the hull's samples at every step it reads, in blocks, about
  2e7 numbers a block; the hull's points are made once, when first read, not at the build.

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
written, whose saved form builds, compiles and runs its reference steps (on Colab a minute or two). `built.save(folder)`
shows its phase too. `progress=False` turns it off; outside a notebook nothing is shown, and `progress=True` asks for
it on a terminal as `mime run` draws it. The display is drawn from `stop`'s place after each chunk, a `stop` of your own
still asked after it: it changes neither the chunks nor anything the run computes. It is drawn through `ipywidgets`,
which Colab has; where it is missing, rich says to install it. To stop a run there, interrupt the kernel: nothing of the
run is kept, and the experiment is built again before it runs on.
