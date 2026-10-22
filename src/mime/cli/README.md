# `mime.cli`

**The `mime` command.** It is a thin layer over the Python API, and does nothing the API cannot.

## Commands

- `mime verify <folder>`: load a saved experiment, rebuild it, check it against its lock, and rerun its reference
  steps (`Experiment.load(folder)`, then `.build().verify()`). It prints the load report and the verdict, and exits
  0 when the outputs are reproduced, 1 when one moved, and 2 when the load or the build is refused.

Planned:
- `mime status <folder>`: the experiment's validity and evidence (a1).
- `mime new` and `mime pack`: scaffold a new experiment folder, or pack one for sharing (0.3.0).

## Why MIME has its own

MIME 0.3 runs on MADDENING 0.4.x, whose scaffolding tool arrives with MADDENING 0.5. So this is MIME's own layer for
now. It doubles as a testbed for MADDENING's design. MIME 0.4.0 moves onto MADDENING 0.5's tool.

## Status

`mime verify` landed in dev1's phase 3; the rest grows with the features it wraps.

Phases and releases are explained in the [package manifest](../README.md#status).
