# `mime.cli`

**The `mime` command.** It is a thin layer over the Python API, and does nothing the API cannot.

## Commands

- `mime verify <folder>`: load a saved experiment, rebuild it, check it against its lock, and rerun its reference
  steps (`Experiment.load(folder)`, then `.build().verify()`), in the precision the lock records (as `mime run`
  does, so that a float64 experiment is not reported moved by a float32 process). It prints the load report and the
  verdict, and exits 0 when the outputs are reproduced, 1 when one moved, and 2 when the load or the build is refused.
- `mime cache clear`: delete the XLA compilation cache that `mime run` and `mime serve` keep in the user's cache
  directory, and say how much it held (see [the compilation cache](../../../docs/compilation_cache.md)). `mime run` and
  `mime serve` use it by default, prune its oldest entries at their start once it passes its cap, and take
  `--no-xla-cache`; `mime verify` never uses it. A cache JAX was already given (`JAX_COMPILATION_CACHE_DIR`) is left
  alone, and one that cannot be written is said once and the command goes on without it.
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
  by sha256, in the cache (see [`mime.assets`](../assets/README.md)). `fetch` takes an experiment folder too, and stores exactly the assets its lock references,
  then places each file of the folder that travels as a reference at its path (copied from the cache and hashed,
  never over another file); it exits zero, or two
  when it is refused, and two while a file that travels as a reference is left unplaced (another file in its place),
  each with why (`asset.missing`);
  `fetch` hashes each file in full, cached ones too; `check` re-hashes every cached file in full and exits one when a file is corrupt, marking it so that a build refuses it until a fetch repairs it. The verbs are named in one place,
  `mime/assets/_commands.py`.

- `mime pack <folder> [--out <file>]`: write the saved experiment in the folder as one **bundle** file
  (`<folder>.mimebundle` beside it by default); see [the saved form](../saved/README.md#the-bundle). It checks that the
  folder loads, carries only the files a saved folder names (an `.env` or a `.git` directory is refused, not packed),
  refuses to carry a path of this machine or any token MIME holds, never carries a file its terms do not let be
  redistributed (it travels as a reference, whatever its size), puts the folder's NOTICE in the bundle, and prints
  the file and its sha256, the files that travel as references, and the experiment's licence or that it declares
  none; for a folder of a form before terms that holds layers or descriptions, one line more (a third party's file in
  it is carried as the author's own). `mime pack .` packs the folder it is run in. Exits zero, or two when refused.
- `mime card <folder> [--out <file>] [--notice]`: the experiment's **card** (Markdown: what it is, what it bounds and
  estimates, what reproduces, its validity lines, its licences and credit, its availability, how to reproduce and cite
  it), or with `--notice` its NOTICE (each file's licence, credit, verdict and declared permission), offline and
  byte for byte as `mime push` will publish them; printed, or written to `--out`. It loads the folder as `pack` does.
  Exits zero, or two when refused.
- `mime pull <doi | address | file> [--into <folder>] [--no-assets] [--yes] [--max-bundle-bytes <n>]`: get a bundle,
  check its hashes, unpack it into a new folder, fetch its assets (unless `--no-assets`) and say what is still needed:
  the packages the lock names that are not installed, with the command to install them. It installs and runs nothing.
  The destination is checked before anything is downloaded, and a download stops at the size a record states, or at
  half a gigabyte for a bundle with no stated size (`--max-bundle-bytes` raises it). A download of assets over a gigabyte is
  asked about first in a terminal; `--yes` goes ahead, and without a terminal it stops with the exact command that
  continues. Each asset download is held to the size its lock declares. A concept DOI resolves to its latest version,
  which pull names. `mime pull --from zenodo|zenodo-sandbox <record>` pulls a published record by its number, read at
  that service's records API with no token: a sandbox DOI is under DataCite's test prefix, which doi.org does not
  resolve, so MIME never asks doi.org for one (`pull.sandbox-doi` says the command). Exits zero, or two when refused.
  Pulling a stranger's bundle: see [the saved form](../saved/README.md#the-bundle).
- `mime push <folder> --to zenodo|zenodo-sandbox [--init | --status | --discard | --publish [--confirm <DOI>]] [--yes]
  [--full-plan] [--keep-site-edits | --overwrite-site-edits]`: the experiment as a record on Zenodo (see
  [`mime.push`](../push/README.md)). A push shows the plan, computed offline, short (what goes up, what travels as a
  reference, the title, the licences, creators and contributors, warnings on lines of their own that start `!`; the full plan, every file with
  its tier and verdict and every field sent, with `--full-plan`), then makes or updates the private draft without a
  question, reserves its DOI and uploads each file that differs, its md5 checked against the service's; a push with
  nothing to send says so in one line. It writes to and deletes only the draft it made. A field edited on the site
  since it last wrote it is shown against `sharing.yaml` and, in a terminal, asked about (kept, written into
  `sharing.yaml`, or written over); `--keep-site-edits` and `--overwrite-site-edits` answer without a terminal.
  `--publish` publishes the draft, the one step that cannot be undone: the confirmation in words, each declared
  permission asked on its own (`--yes` never answers one), then the record number typed, or `--confirm <DOI>` from a
  script, which publishes only what the draft holds (a terminal shows each change first). The draft is read again
  immediately before the request, and a draft that moved is not published. Afterwards the record is checked as
  published and, on production, the folder's lock names it, so `mime pack <folder>` makes the published bundle again. Once published, a push of a changed folder makes a new version
  under the same concept DOI, its unchanged files linked; of an unchanged one, says so in one line. `--init` writes
  `sharing.yaml`, the record's credit; `--status` reads the draft as the service holds it; `--discard` deletes it,
  after asking. Each ends with a link. `--to` is required. A push stopped at any request is resumed by running it
  again. Exits zero, two when refused, three when the service refuses a field MIME builds.
- `mime credentials set | status | remove <service>`: the token `mime push` uses, for `zenodo` and `zenodo-sandbox`
  (see [`mime.credentials`](../credentials/README.md)). The token is asked for, hidden, and is never an argument;
  one typed as an argument is refused without being printed.
- `mime run <folder> --duration <seconds> --out <dir> [--history] [--record <output>]... [--every <seconds>]`: load
  and rebuild a saved experiment in the precision its lock records, run it, and write into `<dir>` its outputs
  (`outputs.npz`: every field after the last step, with `--history` at every step, each series `--record` names as
  `<observation id>.<output>`, one sample every `--every`, and every flag at every step when anything is recorded),
  their manifest beside them (`outputs.json`: each observation's outputs with their units, windows and flags, each
  output at the end written out, so the folder is readable without MIME), a record of the run (`run.json`: the steps,
  the time each stage took, the environment and the load report) and the run's own saved form (`experiment/`), whose
  lock records the environment the run was made in. It prints the result's summary. It is thin over
  `BuiltExperiment.run_to`, which a script calls to write the same, and `mime.results.RunResult.read(<dir>)` reads the
  folder back. Exits zero, or two when the load, the build, the duration or what to record is refused.
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
  [--lab-requirements <file>] [--no-xla-cache] [--timeout <s>] [--max-containers <n>] [--cpu <n>] [--memory <MiB>]`: write the script that
  runs the saved experiment on Modal, by default into `<folder>-modal/` beside it, with MIME's requirements file
  for the variant copied byte for byte (see [Running on Modal](../../../docs/running_on_modal.md)). It never imports
  `modal`. A loose GPU type is refused, and so is `--deterministic` without a GPU. Exits zero, or two when refused.

Planned:
- `mime status <folder>`: the experiment's validity and evidence (a1).
- `mime new`: scaffold a new experiment folder (0.3.0).
- `mime push --metadata`, `mime cite`, `mime search`: a published record's metadata updated, its citation, and the
  community's index (push's P3).

## What goes online

Nothing is fetched unless asked for, and a build never downloads.

| command | network |
|---|---|
| `verify`, `registry`, `docs`, `pack`, `card`, `credentials`, `assets list`, `assets check`, `push --init` | never |
| `push` | to the service `--to` names and no other, with its token, after the plan is shown (`--publish` after its confirmation) |
| `assets fetch` | to download an asset not already in the cache, on request |
| `pull` | to download a bundle given as a DOI or an address (a file on this machine needs none), or with `--from` to the named service's records API, without a token; and to fetch its assets unless `--no-assets` |

`MIME_OFFLINE` closes `assets fetch` and `pull`, which then use only the cache and what was copied in by hand, and
`push`, which then sends nothing.

## Why MIME has its own

MIME 0.3 runs on MADDENING 0.4.x, whose scaffolding tool arrives with MADDENING 0.5. So this is MIME's own layer for
now. It doubles as a testbed for MADDENING's design. MIME 0.4.0 moves onto MADDENING 0.5's tool.

## Status

`mime verify` landed in dev1's phase 3; `mime registry` and `mime docs check` in the first documentation phase;
`mime docs build` in the generators' phase; `mime assets` before phase 7h; `mime run`, `mime env check` and `mime modal`
with the Modal template; `mime pack`, `mime pull` and `mime credentials` with the sharing piece; `mime card`, and the
terms `pack` reads, with push's first piece, terms and tiers; `mime push` (drafts) with its second, P2a; `--publish`,
versions and `pull --from` with P2b; the rest grows with the features it wraps.

Phases and releases are explained in the [package manifest](../README.md#status).
