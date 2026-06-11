# Preempt / resume for experiments

Long cloud runs get interrupted — a spot instance is reclaimed, a job is
cancelled. MIME experiments are made resilient to that with the MADDENING
v0.2 checkpoint machinery (versioned, hash-checked `CHECKPOINT_SCHEMA_VERSION`
manifests). There are two patterns, depending on the experiment shape.

## Parameter sweeps — `ResumableSweep`

A sweep runs many independent work items (parameter combos). Wrap them in
`mime.data.sweep_resume.ResumableSweep`: each finished item is written to a
JSON progress checkpoint — carrying a MADDENING integrity manifest — so a
relaunch skips the finished items and runs only what is left.

```python
import os
from mime.data.sweep_resume import ResumableSweep

sweep = ResumableSweep(
    items=my_param_combos,
    key=lambda combo: combo["label"],          # unique id per item
    checkpoint_path="data/my_experiment.h5",   # progress sits beside it
    resume_from=os.environ.get("SWEEP_RESUME_FROM"),
    snapshot_dir=os.environ.get("SWEEP_SNAPSHOT_DIR"),
)

for result in sweep.completed:     # items finished in a prior run —
    write_output(result)           #   replay them into the output

for combo in sweep.pending:        # only the unfinished items
    result = run_one(combo)
    write_output(result)
    sweep.record(combo, result)    # checkpointed immediately
```

* **`resume_from`** — a path / `file://` / `http(s)://` URL of a prior
  checkpoint. A missing source is treated as a fresh start, so a job spec
  can set it unconditionally. With it unset, a relaunch on the same machine
  still resumes from the local checkpoint beside `checkpoint_path`.
* **`snapshot_dir`** — durable storage (a mounted volume/bucket) the
  checkpoint is mirrored to after every item, so it survives an instance
  teardown.

Worked example: `scripts/run_confinement_sweep.py`. Job-spec wiring:
`jobs/production_h100.yaml`.

## Single-graph experiments — MADDENING checkpoint API

An experiment that runs one long-lived graph (rather than a sweep) uses
MADDENING's graph-state checkpoint API directly:

```python
from maddening.core.simulation.checkpoint import (
    save_state_with_manifest, download_and_load_state,
)

# snapshot the running graph
save_state_with_manifest(graph, "snapshot.npz", extra={"step": step})

# resume it on a fresh instance
download_and_load_state(graph, "file:///mnt/durable/snapshot.npz")
```

For a server-style run, `maddening.cloud.entrypoint.make_preempt_snapshot_hook`
builds an on-preemption snapshot callback and `resume_from_url` is the resume
entry point. Both patterns write the same versioned, hash-checked manifest,
so a checkpoint from either is integrity-verified on load.
