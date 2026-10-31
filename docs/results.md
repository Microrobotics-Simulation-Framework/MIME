# Reading a run's results

A run's result is read by what the experiment declared, never by indexing a node's raw state. Each observation
declares its outputs: what each one is, its dimension and shape, and whether it has a value at every step or at the
end of a run. Every observation's entry page lists them.

## One result, from a run or from its folder

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

An entry declares the flags its nodes raise (the surrogate drag's `clamp_fired` is one). Every reading and series
carries the flags raised over its window, and every export and printout shows them; a flag never blocks a read. A
run that records anything records every flag at every step. A run that records nothing knows each flag at its last
step only: what it says of a flag is **unknown** before that step, never "not raised" (the manifest's `was_raised` is
`null`, with `raised_at_last_step` beside it for what is known), because a flag that fired earlier and is clear at the
last step looks the same at the last step. Record the run to know.

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
