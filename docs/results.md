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
  refused, and the refusal names the `record=` that keeps it. `every=` spaces the samples by time, a whole number of
  base steps, counted from the graph's start.

## Windows by time

Every window follows one rule, the observations' own: a window from `start` to `end` holds the steps whose end lies
after `start` and not after `end`. A sample is the state at the end of its step, and its time is that step's count
from the graph's start times the base step. A time written as a multiple of the base step is that step's end. An
output the graph reduces states its window and its rule (a mean, or an integral), so the series over the same window,
reduced by the same rule, gives the same value, to the rounding of the node's own sums.

## Units

A result's quantities are SI arrays: `.to(unit)` reads one in any unit of its dimension (`mm_per_s`, `um_per_s`,
`ms`) and refuses a unit of another. The graph itself sees SI floats only.

## Flags

An entry declares the flags its nodes raise (the surrogate drag's `clamp_fired` is one). Every reading and series
carries the flags raised over its window, and every export and printout shows them; a flag never blocks a read. A
run that records anything records every flag at every step; a run that records nothing knows each flag at its last
step only, and says so.

## Exports

- `result.table()` is a mapping of columns, each name carrying its unit (`time [s]`, `swim_speed:umr.speed [m/s]`),
  and a `flags` column: `pandas.DataFrame(table)` and `polars.DataFrame(table)` take it as it is. MIME depends on
  neither.
- `result.to_csv(path)` writes that table, its header the names with their units.
- `result.to_npz(path)` writes every array with the manifest inside it, which NumPy reads alone.
- `run_to` and `mime run` write `outputs.json` beside `outputs.npz`: what every array is, each output's unit, window
  and flags, and each output at the end written out, so a run folder is readable without MIME.
