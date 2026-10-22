# `mime.cli`

**The `mime` command.** It is a thin layer over the Python API, and does nothing the API cannot.

## Planned commands

- `mime verify <folder>`: load an experiment and rerun its checks (dev1).
- `mime status <folder>`: the experiment's validity and evidence (a1).
- `mime new` and `mime pack`: scaffold a new experiment folder, or pack one for sharing (0.3.0).

## Why MIME has its own

MIME 0.3 runs on MADDENING 0.4.x, whose scaffolding tool arrives with MADDENING 0.5. So this is MIME's own layer for
now. It doubles as a testbed for MADDENING's design. MIME 0.4.0 moves onto MADDENING 0.5's tool.

## Status

Empty. `mime verify` lands in dev1's phase 3, and the rest grows with the features it wraps.

Phases and releases are explained in the [package manifest](../README.md#status).
