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
  records anything records every flag at every step; one that records nothing reduces each flag over its steps as it
  runs (`FlagWhole`: steps raised, first and last, carried inside the compiled scan beside the state, no history), so
  that a flag raised at any step is reported raised. A part of the run, or a result without the reduction, knows a flag
  at its last step only, and says so: a flag not raised there is *unknown* before it (`was_raised` is `null` in the
  manifest), never read as clean.
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
- A run that records nothing keeps no step's state, and is split into compiled chunks, about 64 of one length, which
  change nothing it computes bit for bit (a script's Ctrl+C arrives between them). A run stopped by its `stop=` is a
  result of the steps it reached, with `stopped` (the reason and the steps asked for), written to `outputs.json` as
  `stopped` and read back from it. Recording keeps every step's state on the device in blocks, and only the recorded fields and the flags on the host.
- The layout a result is read by is plain data, placed on the graph's nodes when the experiment is built
  (`mime.experiment.run_layout`), and the manifest is written from it; a lab's observation declares its outputs the
  same way and gets the same accessors and pages.

## Status

A run folder also holds `topology.json` (the built graph in MIME's names) and,
where `stage_every=` asks, the recording layer (`mime.viz`), the stage's time samples over the folder's own stage.
It holds `report.json` beside `run.json`, the build report's summary (`REPORT_FORMAT`: validity
lines, flags and their meanings, each setting's role and default, the profile), which `RunResult.read` reads as
`result.report` (`None` for a folder written before).
A result of a run in this process remembers the declarations its build came from (`mime._lineage`): `stale` and its
`repr` and `text()` say so once the experiment has had a replacement or a removal since, and a result read from a folder
never does. A pickled result (sent from a worker process) keeps no link to the experiment and says so, "staleness
unknown" (`stale_unknown`), rather than being stale or fresh by accident.
- **`StudyResult`** (`_study.py`), a study's result in memory: one member axis in the order of the points that ran,
  `coordinates[path]` (each axis's values over the members, an array quantity with its units), `groups` (the group each
  member ran in), `result[swim].mean_speed` (an output at the end over the members) and `result[swim].speed` (a series
  stacked over them where they share its time axis, refused otherwise with the route to `member(i)`), `member(i)` (a
  `RunResult`), `flags` (the union, a flag a member's build does not declare reading `not declared`, never "not
  raised"), and `dropped` (the points `drop_refused=True` left out, with their refusals). On that shape: `grid()` (a full
  product reshaped), the long `table()` (coordinate columns with their units first, then `member`, `group`, the member's
  columns and `flags`; members of two sample counts in one table), `to_csv`, `to_npz`, `to_mat` (the member axis first
  in every array, the units in the file, read back by `scipy.io.loadmat` in the tests, MATLAB not run), `manifest()`
  (the folder's `study.json`), `order(output, along=...)` (the observed order of convergence over three levels, and the
  extrapolated value; refused, `study.order`, where the levels are not in the asymptotic range), and `read(folder)`.
- **A study folder** (`_study_folder.py`, `_study_read.py`), what `Study.run_to` writes and `StudyResult.read` reads:
  `study.json` (format `mime-study-outputs`, versioned and closed: a key it does not know is refused), `study.mime.yaml` (the spec),
  `experiment/` (the base, a saved folder byte for byte), `groups/<k>/outputs.npz` (every member's arrays under
  `m<point>/`, its manifest and report as JSON text; entries sorted and carrying no clock, so the same arrays are the
  same bytes), `groups/<k>/report.json` and, for a group whose compiled program differs from the first group's,
  `groups/<k>/experiment/`. Layout checked (a stray file is refused); readable without MIME; written atomically file by
  file, each batch as it finishes, so a stopped study reads as what it holds (`missing`). No saved-form step. See
  [docs/results.md](../../../docs/results.md).
- **`RunResult.as_experiment(name)`** on a member of a study (`MemberOf` says which): the bridge to an experiment of its
  own, in `mime.study._bridge`.
- **An overridden run** (`run(..., override=...)`) carries its `overrides` (each one's path, value with its unit, SI
  value and the leaves it wrote), in its text ("OVERRIDDEN for this run: ..."), its manifest and back from its folder.

Releases are explained in the [package manifest](../README.md#status).
