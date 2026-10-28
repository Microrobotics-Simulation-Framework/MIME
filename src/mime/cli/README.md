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
  `fetch` hashes each file in full, cached ones too; `check` re-hashes every cached file in full and exits one when a file is corrupt, marking it so that a build refuses it until a fetch repairs it. The verbs are named in one place,
  `mime/assets/_commands.py`.

- `mime pack <folder> [--out <file>]`: write the saved experiment in the folder as one **bundle** file
  (`<folder>.mimebundle` beside it by default); see [the saved form](../saved/README.md#the-bundle). It checks that the
  folder loads, refuses to carry a path of this machine or any token MIME holds, and prints the file and its sha256.
  Exits zero, or two when refused.
- `mime pull <doi | address | file> [--into <folder>] [--no-assets] [--yes]`: get a bundle, check its hashes, unpack it
  into a new folder, fetch its assets (unless `--no-assets`) and say what is still needed: the packages the lock names
  that are not installed, with the command to install them. It installs and runs nothing. A download of assets over a
  gigabyte is asked about first in a terminal; `--yes` goes ahead, and without a terminal it stops with the exact
  command that continues. Exits zero, or two when refused.
- `mime auth login | logout | status <service>`: the token `mime push` will use (see [`mime.auth`](../auth/README.md)).

Planned:
- `mime status <folder>`: the experiment's validity and evidence (a1).
- `mime new` and `mime push`: scaffold a new experiment folder; publish a bundle to Zenodo, uploading the assets it
  references that have no Zenodo location yet (0.3.0).

## What goes online

Nothing is fetched unless asked for, and a build never downloads.

| command | network |
|---|---|
| `verify`, `registry`, `docs`, `pack`, `auth`, `assets list`, `assets check` | never |
| `assets fetch` | to download an asset not already in the cache, on request |
| `pull` | to download a bundle given as a DOI or an address (a file on this machine needs none), and to fetch its assets unless `--no-assets` |

`MIME_OFFLINE` closes both: they then use only the cache and what was copied in by hand.

## Why MIME has its own

MIME 0.3 runs on MADDENING 0.4.x, whose scaffolding tool arrives with MADDENING 0.5. So this is MIME's own layer for
now. It doubles as a testbed for MADDENING's design. MIME 0.4.0 moves onto MADDENING 0.5's tool.

## Status

`mime verify` landed in dev1's phase 3; `mime registry` and `mime docs check` in the first documentation phase;
`mime docs build` in the generators' phase; `mime assets` before phase 7h; `mime pack`, `mime pull` and `mime auth` after it; the rest grows with the features it wraps.

Phases and releases are explained in the [package manifest](../README.md#status).
