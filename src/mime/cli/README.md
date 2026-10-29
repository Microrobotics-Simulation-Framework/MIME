# `mime.cli`

**The `mime` command.** It is a thin layer over the Python API, and does nothing the API cannot.

## Commands

- `mime verify <folder>`: load a saved experiment, rebuild it, check it against its lock, and rerun its reference
  steps (`Experiment.load(folder)`, then `.build().verify()`), in the precision the lock records (as `mime run`
  does, so that a float64 experiment is not reported moved by a float32 process). It prints the load report and the
  verdict, and exits 0 when the outputs are reproduced, 1 when one moved, and 2 when the load or the build is refused.
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
  `fetch` hashes each file in full, cached ones too; `check` re-hashes every cached file in full and exits one when a file is corrupt, marking it so that a build refuses it until a fetch repairs it. The verbs are named in one place,
  `mime/assets/_commands.py`.

- `mime run <folder> --duration <seconds> --out <dir> [--history]`: load and rebuild a saved experiment in the
  precision its lock records, run it, and write into `<dir>` its outputs (`outputs.npz`: every field after the last
  step, and with `--history` at every step), a record of the run (`run.json`: the steps, the time each stage took, the
  environment and the load report) and the run's own saved form (`experiment/`), whose lock records the environment the
  run was made in. It is thin over `BuiltExperiment.run_to`, which a script calls to write the same. Exits zero, or
  two when the load, the build or the duration is refused.
- `mime env check [<folder>] --variant cpu|cuda12 [--gpu <type>] [--base] [--lab <file>] [--mime-version <v>]
  [--xla-cache <dir>] [--record <file>]`: what a runner calls before a run (see [`mime.env`](../env/README.md)). It holds the installed
  distributions to the variant's pins and the lab's, MIME to `--mime-version` (the pinned wheel's; without it MIME is
  not held to a version), with `--base` Python and the OS to MIME's base image for the series, and JAX's devices to
  the GPU requested, refusing a CPU fallback and a module imported from outside this environment's site-packages;
  then it applies the precision the folder's lock records (a GPU run whose lock records a matmul precision other than
  the highest is refused), and XLA's cache under `<dir>`. Exits
  zero, or two with every difference named; `--record` writes what it found, or what it refused, as JSON. The
  precision holds in the process it runs in, so a runner calls it in the run's own process
  (`mime.cli.main(["env", "check", ...])`), as the Modal script does.
- `mime modal <folder> [--gpu <type>] [--deterministic] [--python <series>] [--out <dir>] [--checkout <dir>]
  [--lab-requirements <file>] [--no-xla-cache] [--timeout <s>] [--cpu <n>] [--memory <MiB>]`: write the script that
  runs the saved experiment on Modal, by default into `<folder>-modal/` beside it, with MIME's requirements file
  for the variant copied byte for byte (see [Running on Modal](../../../docs/running_on_modal.md)). It never imports
  `modal`. A loose GPU type is refused, and so is `--deterministic` without a GPU. Exits zero, or two when refused.

Planned:
- `mime status <folder>`: the experiment's validity and evidence (a1).
- `mime new`, `mime pack`, `mime push` and `mime pull`: scaffold a new experiment folder; pack one as a bundle;
  publish a bundle to Zenodo; fetch a bundle and its assets through the assets resolver (0.3.0).

## Why MIME has its own

MIME 0.3 runs on MADDENING 0.4.x, whose scaffolding tool arrives with MADDENING 0.5. So this is MIME's own layer for
now. It doubles as a testbed for MADDENING's design. MIME 0.4.0 moves onto MADDENING 0.5's tool.

## Status

`mime verify` landed in dev1's phase 3; `mime registry` and `mime docs check` in the first documentation phase;
`mime docs build` in the generators' phase; `mime assets` before phase 7h; `mime run`, `mime env check` and `mime modal`
with the Modal template; the rest grows with the features it wraps.

Phases and releases are explained in the [package manifest](../README.md#status).
