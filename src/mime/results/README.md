# `mime.results`

**A run's result, read by what the experiment declared, with units, never by indexing raw state.**

## What goes here

- **`RunResult`**, one object from `BuiltExperiment.run(...)` and from `RunResult.read(folder)`, of a folder `run_to`
  or `mime run` wrote (on a local disk, or a Modal Volume's mount, or what `modal volume get` fetched), or of an NPZ
  `to_npz` wrote. The same accessors on each:
  - `result[swim]` (the handle `observe` returned) or `result["swim_speed:umr"]` (the observation's id), and then
    each declared output by its name: `result[swim].mean_speed`, `result[swim]["speed"]`;
  - an output at the end, which the graph reduces and every run has, is a **`Reading`**: a quantity with its window
    and the flags raised over it;
  - a per-step output is a **`Series`** when the run recorded it (`run(..., record=[swim.speed], every=...)`, or
    `history=True`, which records every one; `every=` spaces the samples kept and does not reduce the cost of recording,
    the whole state's history per step): its time axis, its values as an array quantity, `between(start,
    end)` for a window by time, and `mean()` and `integral()`, the rules a node declares for what it reduces. One the
    run did not record is refused (`NotRecorded`), naming the `record=` that keeps it.
- **One inclusion rule** for every window, the observations' own: a window from `start` to `end` holds the steps
  whose end lies after `start` and not after `end`. A step's end is its count from the graph's start times the base
  step. The start is decided as a node decides its `after`, in the run's float type (so in float32 a start on the
  grid may take the step that ends there or not, as that node's does, and so does a start of the user's own); the
  end, a time written as a multiple of the base step, is that step's end.
- **Flags** (`Flag`, declared by the entry whose node raises it, `raises` in the registry): on every reading, series
  and table row, in every export and printout, over the window it covers, and never blocking a read. A run that
  records anything records every flag at every step; one that records nothing knows each flag at its last step only,
  and says so: a flag not raised at the last step is *unknown* before it (`was_raised` is `null` in the manifest),
  never read as clean.
- **The exports**: `table()`, a mapping of columns whose names carry their units (`time [s]`,
  `swim_speed:umr.speed [m/s]`), with a flags column, which `pandas.DataFrame` and `polars.DataFrame` take as it is
  (MIME depends on neither); `to_csv(path)`, the table with its units in its header; `to_npz(path)`, every array with
  the manifest under `manifest`; `to_mat(path)`, the declared outputs and flags as a MATLAB v5 file through
  `scipy.io.savemat` (names mapped to MATLAB's and the map in the file; read back by `scipy.io.loadmat` in the tests,
  MATLAB itself not run); and `manifest()`, which `run_to` writes as `outputs.json` beside `outputs.npz`, so
  a run folder is readable without MIME.

## The contract

- The graph sees SI floats only: a result's quantities hold arrays (`Quantity.of`), read in a unit by `.to(unit)`,
  which refuses a unit of another dimension. An array quantity is a result and never a declaration.
- A run that records nothing keeps no step's state, and is split into compiled chunks, about 256 of one length, which
  change nothing it computes bit for bit (a script's Ctrl+C arrives between them). A run stopped by its `stop=` is a
  result of the steps it reached, with `stopped` (the reason and the steps asked for), written to `outputs.json` as
  `stopped` and read back from it. Recording keeps every step's state on the device in blocks, and only the recorded fields and the flags on the host.
- The layout a result is read by is plain data, placed on the graph's nodes when the experiment is built
  (`mime.experiment.run_layout`), and the manifest is written from it; a lab's observation declares its outputs the
  same way and gets the same accessors and pages.

## Status

Since dev1's phase 7k. Since phase 7n, a run folder also holds `topology.json` (the built graph in MIME's names) and,
where `stage_every=` asks, the recording layer (`mime.viz`), the stage's time samples over the folder's own stage.
Since phase 7n-b, it holds `report.json` beside `run.json`, the build report's summary (`REPORT_FORMAT`: validity
lines, flags and their meanings, each setting's role and default, the profile), which `RunResult.read` reads as
`result.report` (`None` for a folder written before).
Later: `.mat` through a `matlab` extra and `StudyResult`, a result with a leading sweep axis whose table is long
(dev2's lane S).

Phases and releases are explained in the [package manifest](../README.md#status).
