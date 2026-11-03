# `mime.cli`

**The `mime` command.** It is a thin layer over the Python API, and does nothing the API cannot.

## Commands

- `mime verify <folder>`: load a saved experiment, rebuild it, check it against its lock, and rerun its reference
  steps (`Experiment.load(folder)`, then `.build().verify()`), in the precision the lock records (as `mime run`
  does, so that a float64 experiment is not reported moved by a float32 process). It prints the load report and the
  verdict, and exits 0 when the outputs are reproduced, 1 when one moved, and 2 when the load or the build is refused.
  Ctrl+C ends it (130), the second press at once.
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
  resolve, so MIME never asks doi.org for one (`pull.sandbox-doi` says the command). `--signed-in` sends the token `mime
  credentials` stored for that service, for a record or files shared with that account, and `mime pull '<share link>'`
  the key a record's share link carries (a link is given in quotes, which keep a shell from reading its `?`); either goes
  to the service's own host alone, as push's token does, and is never printed, logged or written (`pull.restricted`
  names the route that works). A link is read by its parts: one on a service's host in another scheme, port or host
  name, or with a user part, is refused without its key, and a `token` parameter is hidden in every address pull prints.
  `--link -` asks for the link with no echo and `--link env` reads it from `MIME_SHARE_LINK`, so that its key is in
  neither the shell's history nor the process's arguments, which other accounts on the machine can read (`ps`,
  `/proc/<pid>/cmdline`) unless `/proc` is mounted with `hidepid`. Exits zero, or two when refused.
  Pulling a stranger's bundle: see [the saved form](../saved/README.md#the-bundle).
- `mime push <folder> --to zenodo|zenodo-sandbox [--init | --setup | --status | --discard | --publish [--confirm
  <DOI>[@<fingerprint>]]] [--yes] [--full-plan] [--keep-site-edits | --overwrite-site-edits]`, and `mime push
  --defaults`: the experiment as a record on Zenodo (see
  [`mime.push`](../push/README.md)). A push shows the plan, computed offline, short (what goes up, what travels as a
  reference, the title, the licences, creators and contributors, warnings on lines of their own that start `!`; the full plan, every file with
  its tier and verdict and every field sent, with `--full-plan`), then makes or updates the private draft without a
  question, reserves its DOI and uploads each file that differs, its md5 checked against the service's; a push with
  nothing to send says so in one line. It writes to and deletes only the draft it made. A field edited on the site
  since it last wrote it is shown against `sharing.yaml` and, in a terminal, asked about (kept, written into
  `sharing.yaml`, or written over); `--keep-site-edits` and `--overwrite-site-edits` answer without a terminal.
  `--publish` publishes the draft, the one step that cannot be undone: the confirmation in words, each declared
  permission asked on its own (`--yes` never answers one), then the record number typed, or `--confirm <DOI>` from a
  script, which publishes only what the draft holds (a terminal shows each change first); once a push has changed the
  draft after the push that reserved its DOI, or where that push came before the fingerprint, a script confirms with the
  fingerprint the last push printed beside it, `--confirm <DOI>@<fingerprint>`, and the DOI alone is refused
  (`push.confirm-stale`). The draft is read again immediately before the request, and a draft that moved, or no longer
  holds what the confirmation's fingerprint was printed for, is not published. Afterwards the record is checked as
  published and, on production, the folder's lock names it, so `mime pack <folder>` makes the published bundle again. Once published, a push of a changed folder makes a new version
  under the same concept DOI, its unchanged files linked; of an unchanged one, says so in one line. `--init` writes
  `sharing.yaml`, the record's credit, filled from the person's sharing defaults; `--setup` makes or edits it a field
  at a time in a terminal (Enter skips a field, `use <value>? [Y/n]` keeps one it holds, or a default where the file is
  new, an invalid value is asked again; an existing file is edited in its own line endings, through its link where it
  is one, and a read-only one is refused; with both outputs redirected to files its questions go to the terminal),
  then asks whether to save the person's own values as their defaults, which live in MIME's config directory
  and never in a folder (`mime push --defaults` shows them and where); `--setup` and `--defaults` take no flag of a
  push or a publish; `--status` reads the draft as the service holds
  it; `--discard` deletes it, after asking. Each ends with a link. `--to` is required, but for `--setup` and
  `--defaults`. A push stopped at any request is resumed by running it again. Exits zero, two when refused, three when
  the service refuses a field MIME builds.
- `mime credentials set | status | remove <service>`: the token `mime push` uses, for `zenodo` and `zenodo-sandbox`
  (see [`mime.credentials`](../credentials/README.md)). The token is asked for, hidden, and is never an argument;
  one typed as an argument is refused without being printed.
- `mime run <folder> --duration <time> --out <dir> [--history] [--record <output>]... [--every <time>] [--full]`: load
  and rebuild a saved experiment in the precision its lock records, run it, and write into `<dir>` its outputs
  (`outputs.npz`: every field after the last step, with `--history` at every step, each series `--record` names as
  `<observation id>.<output>`, one sample every `--every`, and every flag at every step when anything is recorded),
  their manifest beside them (`outputs.json`: each observation's outputs with their units, windows and flags, each
  output at the end written out, so the folder is readable without MIME), a record of the run (`run.json`: the steps,
  the time each stage took, the environment and the load report) and the run's own saved form (`experiment/`), whose
  lock records the environment the run was made in. It prints the result's summary (on a terminal, led by what needs
  attention: see [On a terminal](#on-a-terminal)). A `<time>` is `10s`, `500ms`, `250us`, `2min` or `1h`; a bare number is seconds. It is thin over
  `BuiltExperiment.run_to`, which a script calls to write the same, and `mime.results.RunResult.read(<dir>)` reads the
  folder back. Exits zero, or two when the load, the build, the duration or what to record is refused.

  **Ctrl+C.** A run is compiled chunks (about 64 of one length, a divisor of the steps where one lies near, else two
  lengths; a controller's period or `--stage-every` sets them where there is one), which change nothing it computes
  bit for bit. The first press is answered at once with one line, and the run stops at the end of its chunk and writes
  its folder as the steps reached: `outputs.json` and `run.json` hold `stopped` (the reason, the steps and the time
  reached and asked for), `RunResult.read(<dir>).stopped` says it, and the command exits 130. The second press quits
  within a second wherever the process is, inside a compiled chunk or a compile, writing nothing: `outputs.json` is
  the last file of a folder, so a run quit while it is written is not a folder that reads. A press while the
  experiment loads or builds ends the command (130, nothing written), as it does in `mime verify`; one in the first
  second or two, while `mime` itself is imported, is Python's own (a traceback). How it works, and what it cannot
  do, is in [`mime/_interrupt.py`](../_interrupt.py).
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
- `mime completion bash|zsh|fish`: print a completion script for the shell to standard output, written from the same
  parser the command runs on, so every command, flag and choice is completed and none is listed by hand. The script is
  static: pressing Tab runs the shell's own code and no Python (importing MIME imports JAX, which takes seconds), so it
  answers at once. A flag whose value is a path (`--out`, `--into`, a saved folder) completes as files and folders, and
  one with choices (`--to`, `--from`, `--variant`, a credentials service) as its choices, after `--to` or `--to=`.
  Install it once, then open a new shell:
  - bash: `mime completion bash > ~/.local/share/bash-completion/completions/mime` (loaded by the bash-completion package);
  - zsh: `mime completion zsh > ~/.zfunc/_mime`, a file named `_mime` in a folder on `fpath`, which `.zshrc` sets before
    `compinit` runs (`fpath=(~/.zfunc $fpath)`);
  - fish: `mime completion fish > ~/.config/fish/completions/mime.fish`.

  Write it again after upgrading MIME, since the script is a copy of what the parser held when it was written. The
  generator is `mime/cli/_completion.py`; `tests/cli/test_completion.py` asks each script, in its shell, what it
  completes at every command and compares that with the parser. Exits zero.

Planned:
- `mime status <folder>`: the experiment's validity and evidence (a1).
- `mime new`: scaffold a new experiment folder (0.3.0).
- `mime push --metadata`, `mime cite`, `mime search`: a published record's metadata updated, its citation, and the
  community's index (push's P3).

## On a terminal

**When output is not going to a terminal it is byte for byte what it always was**: a pipe, a file, CI's log and Modal's
are read by programs, and so are a terminal asked not to be drawn on (`--plain` anywhere on the line, `NO_COLOR` set and
not empty, `TERM=dumb`). Each stream is judged alone: progress is on standard error and results on standard output, so
`mime run ... | tee log` still shows its progress and the pipe gets what it always got, a line the run's own code prints to
standard output included. Colour never carries what the words
do not. `rich` is imported by one module, `mime/cli/_display.py`, and only to draw (a test holds both; `mime --help` never
loads it).

- **`mime run`'s progress** (standard error): a spinner naming the phase while the experiment loads, builds and compiles;
  then the bar, the steps, the simulated time, the speed, **the time remaining**, the flags raised so far and "Ctrl+C
  to stop", redrawn at most about four times a second. The time remaining is an **estimate**: the steps left over the
  speed of the chunks since the last one that compiled finished (the first chunk, and in a run of two chunk lengths the
  second as well), shown after a chunk that did not compile. A flag is
  known at a chunk's last step, so it is "raised **by** step N", the end of the chunk it was first seen at; one raised and
  gone again inside a chunk is not seen. The last chunk is not announced, so the run folder's writing (which can take
  longer than a chunk) shows as "finishing the last chunk, then writing the run folder".
- **The end of a run** leads with what needs attention (a stopped run, flags raised, post-run watches above their range or that could not be checked (no range declared, an input missing),
  the validity lines the build computed, MADDENING's warnings counted, the profile's pending items counted, an
  environment unlike the lock's), says what ran and how the time went, shows the first readings, the files written and
  the commands for this folder (read it, verify it, serve it), and ends with where the rest is. `--full` adds the load
  report, the build report's summary and every reading; the files hold all of it always. **What it cannot say:** a
  validity line is free text with no "at its limit" status, so the lines listed are those the build *computed*, which it
  gives when a model is at or past what it covers; MIME has no structured record of a fallback, so none is reported as
  one.
- **A refusal is a block**: its code and whose it is to fix, what is wrong, `what to do`, `why`, `where`, and the docs
  address under it on one line to copy; the full traceback's file as before. `--traceback` is unchanged.
- **`mime push` and `mime pull`** say the same words on a terminal, set differently (the lines themselves arrive tagged
  with what they are, `mime/_said.py`, and a tag is invisible to a pipe, a file and a test). The plan is one block with
  what goes up and what travels as a reference in rows of their own, and each warning (a line that starts `!`, as on a
  pipe) below it, outside it. The **publish step** is a heavy frame, its sentence that the step cannot be undone
  reversed on production (the sandbox's, which says the same of production, is bold) and each declared permission a frame
  of its own, with its question below it set apart by a `?` and a blank line, answered at `[y/N]`; the record number is
  typed after a `>`. Anything but `y` or `yes` is no, and the end of input is no, exactly as without the display; `--yes`
  still answers nothing it did not, and `--confirm` still names the draft. A field edited on the site is **two values side
  by side**, the site's and `sharing.yaml`'s, the stretches that differ underlined, then the question. **Uploads and
  downloads** (`mime push`; `mime pull`, the bundle and each asset) are one line on standard error that stays when they
  are done, `x/y files uploaded` with the bytes of the total, and under it the current file's bar with its name, bytes,
  speed and the time remaining (an **estimate**, shown after the first second); a file whole stays as that bar until the next
  starts, the last stays at the end, and a file of no stated size is counted without a bar. While no bytes are moving
  the line says what is: `waiting between requests, then ...` (push spaces its requests), `Zenodo is committing ... and its md5 is
  checked`, `checking the hash of ...`, `unpacking the bundle`; it is drawn about four times a second, takes itself off
  the screen for a question, and a fault of its own turns it off without touching the command. The statement of a
  publish is drawn whole when its last line is said, before the request; a terminal that reports no width is drawn at
  the width `COLUMNS` names, or at a default; a frame that is too narrow for a hash or an address is left off, so the word is not folded. A
  question and its hint are on the terminal the person answers at. A command printed for pasting
  (`mime/_shell.py`) holds a folder's name as one word, quoted where a shell would read it as more.
- **`mime serve`'s live view** (standard error): a spinner while the experiment loads and builds, then a strip of a few
  lines (the state in a word, the step, the simulated time against the wall time, the speed, the frames a second, the
  clients, recording and the flags, and how to stop) with a table of values under it, and an event above it, in the
  scrollback, for each client that connects or leaves and each command with what came of it. It is drawn about four times a
  second whatever the frame rate, from a plain record of the session (`mime.live.ServeState`) the serve loop only stores into,
  so a later `mime watch` fills the same record from a server's status and frames and draws the same view. A peer's words are
  written out as plain text on one line, never markup and never a command to the terminal. Standard output stays the two lines
  a launcher reads, byte for byte, whatever standard error is. The text is `mime/cli/_serve_view.py`, which no terminal is
  needed to test; the drawing is `ServeDisplay` in `_display.py`. Everything serve writes to a terminal goes through `BoundedWriter` in
  `_bounded.py`, so a terminal nobody reads delays no stop: a write it does not take is dropped and counted.
- **`mime env check`'s refusals are grouped** where they share a code. An environment with many distributions nothing
  pins refuses once for each, and the same sentence, three lines each, buried the list: on a terminal the refusals that
  are about a subject (`env.unpinned`, `env.missing`, and `env.pin` for an installed distribution at another version) and
  share their code, what is wrong, what to do, why and where are one block, with the code once, a table with a row for
  every distribution (its version, and the pin and what pins it where there is one), and what to do once, `<name>` standing
  where the row's distribution goes. Nothing is dropped: the rows are as many as the refusals (a test counts them, and
  fails when one is left out). Refusals whose actions differ are never merged; a code is a block of its own for each
  action. Piped, in a file, with `--plain`, `NO_COLOR` or `TERM=dumb`, each refusal is its three lines as before, in
  the same order, and `--record` writes the same JSON.
- **A damaged compilation-cache entry**: JAX's warning is shown as it is, and once beside it a note that `mime cache
  clear` removes the cache.
- **Plain text is right for** bare `mime`, `mime pack`, `mime card`, `mime credentials`, `mime cache clear`, `mime modal` and
  `mime registry --json`, and they are not drawn on, so nobody boxes them later. `pack` prints three lines whose point is a
  path and a sha256 to copy; `card` is a document people redirect to a file; `credentials` asks for a token with a
  hidden prompt; `cache clear` is one line; `modal` prints the commands to paste; bare `mime` is a list of commands, done
  already; `registry --json` is read by a program. A box or a table around any of them adds width and takes away
  selectability, which a person copies from. A command is drawn when its output is a list a table reads better, or a wait that a line
  of progress shortens, and not otherwise.
- **`mime --help`** is grouped (Run, Share, Environment, Reference), each command has an example under its `--help`,
  a bare `mime` says what one can do here (more in a saved experiment's folder or a run's), and a mistyped command or
  flag is answered with the nearest (on every supported Python, a terminal or not).
- **Ctrl+C while `mime` imports** (about the first second) ends it quietly, 130 (`mime/_import_guard.py`; installed only
  when the process is the `mime` command, so a script that imports MIME keeps Python's handling).

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
