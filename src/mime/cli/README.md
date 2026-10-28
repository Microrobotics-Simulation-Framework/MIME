# `mime.cli`

**The `mime` command.** It is a thin layer over the Python API, and does nothing the API cannot.

## Commands

- `mime verify <folder>`: load a saved experiment, rebuild it, check it against its lock, and rerun its reference
  steps (`Experiment.load(folder)`, then `.build().verify()`). It prints the load report and the verdict, and exits
  0 when the outputs are reproduced, 1 when one moved, and 2 when the load or the build is refused.
- `mime registry [--json]`: every registered entry, MIME's and each installed namespace's; with `--json`, the registry
  export the docs are generated from (see [`mime.docs`](../docs/README.md)). It runs on a base install, where a
  namespace that needs the `verify` extra is listed as not loaded.
- `mime docs check [--root <checkout>] [--base <revision>]`: every check of the documentation's sources over a
  checkout of MIME, the docs walk, the number rule and the citations among them, with the anomaly script; with `--base`, the
  README start list against that revision's too, which it may not grow.
  Exits 0 when every check passes, 1 when one fails, and 2 without the `verify` extra. The pages are built in
  fresh processes under eight hash seeds and must be byte-equal; `--skip-hash-seed-builds` leaves that stage out, for
  the tests of the other stages, and CI never passes it.
- `mime docs build [--root <checkout>] [--out <directory>]`: the documentation's pages, generated from the registry
  export and the hand-written sources, to `docs/_build/` of the checkout (untracked, never committed) or the
  directory named. Exits zero when the pages are written, one when a source no longer resolves or a page holds a path
  of the machine, with no page written, and two without the `verify` extra.
  `mime.docs` is imported inside the command, so every other command runs without it.

- `mime assets list`, `mime assets fetch <name-or-folder>... | --all` and `mime assets check`: the large files MIME references
  by sha256, in the cache (see [`mime.assets`](../assets/README.md)). `fetch` takes an experiment folder too, and stores exactly the assets its lock references; it exits zero, or two
  when it is refused;
  `check` re-hashes every cached file in full and exits one when a file is corrupt. The verbs are named in one place,
  `mime/assets/_commands.py`.

Planned:
- `mime status <folder>`: the experiment's validity and evidence (a1).
- `mime new`, `mime pack`, `mime push` and `mime pull`: scaffold a new experiment folder; pack one as a bundle;
  publish a bundle to Zenodo; fetch a bundle and its assets through the assets resolver (0.3.0).

## Why MIME has its own

MIME 0.3 runs on MADDENING 0.4.x, whose scaffolding tool arrives with MADDENING 0.5. So this is MIME's own layer for
now. It doubles as a testbed for MADDENING's design. MIME 0.4.0 moves onto MADDENING 0.5's tool.

## Status

`mime verify` landed in dev1's phase 3; `mime registry` and `mime docs check` in the first documentation phase;
`mime docs build` in the generators' phase; `mime assets` before phase 7h; the rest grows with the features it wraps.

Phases and releases are explained in the [package manifest](../README.md#status).
