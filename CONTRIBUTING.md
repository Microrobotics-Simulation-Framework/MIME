# Contributing to MIME

## Development Setup

```bash
git clone git@github.com:Microrobotics-Simulation-Framework/MIME.git
cd MIME

# The locked environment, with the `verify` extra and the `dev` dependency group (pytest, Hypothesis, pyright,
# hatchling), which uv installs by default. uv takes Python 3.14 from `.python-version`; CI runs every supported
# Python (3.12 to 3.14) and type-checks against 3.12, the floor.
uv sync --locked --extra verify
uv pip install --no-deps ./lab_fixture
```

## Licensing of Contributions

MIME is licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). A contribution is
made under the same licence (inbound equals outbound, as section 5 of the licence states), with no separate
agreement to sign.

Each commit carries a [Developer Certificate of Origin](https://developercertificate.org/) sign-off, the line
`Signed-off-by: Your Name <you@example.org>` that `git commit -s` adds: by it you certify that you wrote the change,
or otherwise have the right to submit it under the project's licence. This is documented, not yet enforced: no CI
check reads the sign-off today.

## ID Prefix Convention

MIME uses the `MIME-` prefix for all compliance identifiers:

| Category | Prefix | Example |
|----------|--------|---------|
| Node algorithm IDs | `MIME-NODE-` | `MIME-NODE-001` |
| Anomaly IDs | `MIME-ANO-` | `MIME-ANO-001` |
| Verification benchmark IDs | `MIME-VER-` | `MIME-VER-001` |

Never use `MADD-` prefixed IDs for MIME artifacts. Cross-reference upstream MADDENING anomalies by their `MADD-ANO-*` ID in the `safety_relevance_rationale` field.

## Naming

A class is named for what it is, in a name that reads in its namespace (`mime.fields.PointDipole`,
`Drag.Surrogate`). Rules 2 and 6 are enforced by `tests/test_naming_rules.py`, and so is rule 5 for one name: the
method of regularised Stokeslets is the `MRS` rung, and its old generic name stays out of the repository, free for a
rung that uses boundary elements; the others are for review.

1. **Name a class for what it is,** not who published it or where it came from. Attribution goes in the docstring and
   the entry's `references`. A physical law or model universally known by a person's name (Stokeslet, Oseen, Faxén,
   Rotne-Prager) keeps it, because that name is the standard term.
2. **No word the namespace already carries.** `mime.fields.UniformRotating`, not `UniformRotatingField`; a family's
   rung is `Drag.Surrogate`, not `SurrogateDrag`. Implementation classes keep their type's suffix (`...Node`,
   `...State`), because there the suffix says which layer a class belongs to.
3. **One concept, one stem, across layers:** the scene body, the core shape, the node, its state, its registry name
   and its switch owners share one stem (`Screw`, `SCREW`, `screw`, `screw.seam`).
4. **No qualifier that is always true** in its context (`Robot` in a robotics library; `Solid` where every scene body
   is solid).
5. **A generic name only for the general concept.** A specific variant is a setting or a qualified name, and a
   generic name a later concept will need stays free (`Helix` for a helical filament, the geometry slender-body
   theory and many microswimmers use).
6. **A module never re-exports another layer's class** under a name it also defines; same-named classes in
   different layers are imported from their own module.
7. **Spelling:** classes `CapWords`; singletons and constants `UPPER_SNAKE`; registry names `lower_snake` equal to
   the class stem; switch owners `<registry name>.<switch>`.

## Commit Message Convention

| Prefix | Meaning |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring (no behaviour change) |
| `docs:` | Documentation only |
| `test:` | Test additions or changes |
| `perf:` | Performance improvement |
| `verify:` | Verification/validation evidence |
| `break:` | Breaking change |
| `deprecate:` | Deprecation notice |
| `security:` | Security-relevant change |

## Running Tests

```bash
JAX_PLATFORMS=cpu uv run python -m pytest
uv run python scripts/typing_check.py    # pyright, strict over src/mime, at zero errors
```

## CI: how many shards

CI runs each test lane (`default` on three Pythons, `maddening-main` and `openmaterica-main`) as several shard jobs,
split by test file (`tests/_sharding.py`), and the lane's own job, which branch protection requires, waits for them and
checks that together they ran every collected test, with none failed or skipped (`scripts/check_shards.py`). How many
shards is a setting, **six unless set**:

- the repository variable **`MIME_CI_SHARDS`** (Settings, Secrets and variables, Actions, Variables) sets it for every
  run; unset or empty means 6;
- a manual run's **`shards`** input (Actions, CI, Run workflow) sets it for that run and wins over the variable;
- it is a whole number from **1 to 12**. Anything else fails the `shard count` job by name, and the lanes with it, so
  no run starts on a count nobody chose.

The trade-off: every shard job costs about the same job-minutes whatever the count (its own setup is about 20 s), so
more shards shorten a run's wall time and put more jobs against the organisation's concurrent limit (60 on the Team
plan), and fewer shards do the opposite. A run is 5 lanes' shard jobs (`default` on three Pythons, `maddening-main`,
`openmaterica-main`) plus 8 other jobs: 38 jobs at the default 6 and 18 at 2. Lower the count when several runs queue
behind one another, raise it for a single run's speed. The compilation cache is kept per shard count, so changing the
count starts that count's caches cold.

## CI: borderline tests

A push to `v0.3` runs cold (an empty compilation cache) and a shard draws whichever runner class is free, so a test
near the 20 s budget fails the `Test durations` step on some runs and not others, with nothing wrong in it. A test is
listed in `tests/duration_allowlist.txt` as persistently borderline when, over the last cold push runs of the CI
workflow, it was over 20 s at least once, or over 17 s (85 % of the budget) in at least a quarter of them. Its entry says
its largest cold duration and the runner, and its typical warm duration. The list is a record of measurements: no test
is made faster, slow-marked, shrunk or derandomised to leave it.

## CI: stress runs

A property test, or the control that must fail every run, is shown stable by running it many times with fresh Hypothesis
seeds. CI does that in parallel (`.github/workflows/stress.yml`), so it does not hold a maintainer's machine for the
better part of an hour. It is not a required check.

**Ask for one** by adding the label **`stress`** to the pull request. The run starts when the label is added and again on
every push while the label stays (a newer push cancels the older run); another label starts nothing. By default it
runs every test file the pull request changes (`tests/**/test_*.py`). A line of the pull request's description adds to
that, or sets the options:

```
stress: tests/oracles/test_spinner.py tests/test_x.py::test_y[a] runs=30 python=3.14 seed=7 only
```

- a bare token is a test file or node id under `tests/` (no spaces in a node id), run in addition to the changed files;
  `only` leaves the changed files out;
- `runs=N` is 1 to 40 runs, **20** if absent; `python=3.12`, `3.13` or `3.14` is the Python, **3.12** if absent (3.14's
  shards have met flakes the others did not, so ask for it when that is the question);
- `seed=N` is the first run's Hypothesis seed; absent, a random one. Run *k* uses `N + k - 1`.

Editing the description does not start a run: remove the label and add it again. Anything the line cannot use, and a
selection that is empty, fails the `stress selection` job by name; a stress run that ran nothing is not a pass.

**Read it** on the run's page: the `stress summary` job lists, per test, how many of the runs it passed (tests that did
not pass every run first), then for each failure the run, its Hypothesis seed, the falsifying example and the
`@reproduce_failure` blob, and every run's seed. It fails if any run failed, or had no report, or passed no test. Each
run is a job (`stress run k of n`) set up as the default lane is, with the compilation caches the last push to `v0.3`
saved for the shards holding the selected files, restored and never saved (at most three shards' worth; the rest of the
files compile cold, so a first run can be slower). Each run's JUnit report is a run artifact, `stress-run-<k>`.

**Repeat a failing seed locally** on the same Python, with the tests the summary names:

```bash
JAX_PLATFORMS=cpu uv run python -m pytest tests/test_x.py::test_y --hypothesis-seed=1234567 -n 0
```

or paste the `@reproduce_failure(...)` line above the test for the exact example. A seed repeats the draws of a run, not
its timing; if it does not fail locally, run `seed=N` again on CI.

## Compliance Scripts

```bash
uv run --no-sync mime docs check            # the documentation's sources, the docs walk, the number rule, the citations, the pages built twice
uv run --no-sync mime docs build            # the pages, to docs/_build/ (untracked, never committed), for you to read
python scripts/check_anomalies.py
python scripts/check_confidential.py --key-file /path/to/confidential_scan.key --messages origin/v0.3..HEAD \
  --diffs origin/v0.3..HEAD
python scripts/check_confidential.py --key-file /path/to/confidential_scan.key --directory docs/_build   # the built pages
```

## Confidentiality

This repository is public. It never names the maintainer's private projects, and never carries a private path: an
absolute home directory, a session scratchpad, Claude Code's temporary directory, or the private planning directory
beside the repository, which only `CLAUDE.md` names. `scripts/check_confidential.py`, OpenMaterica's scan copied,
reads each tracked file's path and its content as the working tree holds it; with `--messages RANGE`, each commit's
message in the range; with `--diffs RANGE`, what each commit in the range adds against its first parent: the paths it
adds or changes, the lines it adds (with the line after each), and a binary content or a new file whole, so a name
added and removed again within a pull request is found; and with `--ref-name`, a branch's name. CI's `confidential`
job passes the pull request's or the push's range to both, and a pull request's branch name; a change into `main`
leaves out v0.3's commits (`--public origin/v0.3`), public already and scanned on their way into v0.3.

The names cannot be listed here, and plain hashes of them would not hide them, since hashing a word list recovers
short words. So `scripts/confidential_digests.txt` holds their HMAC-SHA256 under a secret key, 64 hexadecimal digits,
which CI reads from the repository secret `CONFIDENTIAL_SCAN_KEY`; the key is OpenMaterica's, so the digest file is
OpenMaterica's too. A key the digests were not made with is refused. A finding gives its file and line, and a term's
first letter and length, never the term.

**Without the key the scan fails**, so it cannot pass having checked nothing: on a push, on a pull request from one of
this repository's branches, and on your machine. The one exception is a pull request from a fork, which GitHub gives no
secrets: told apart by the event (its head repository is not this one), never by the key's absence alone, it checks the
paths, warns that the names were not checked, and passes when the paths are clean; the push after its merge scans with
the key.

A text file is read as UTF-8, a byte that is not read as a replacement, which keeps every ASCII word of a text in any
ASCII-compatible encoding. A binary file, or a text in UTF-16 or UTF-32, is read by its printable runs of four ASCII
characters or more, as bytes and as UTF-16 and UTF-32 of either byte order; its findings say `(binary)`, and their line
is the run's number. What the scan does not read: a run shorter than four characters; whatever a file holds
compressed, such as a PNG's compressed text chunks, a PDF's streams or an archive's members; an untracked file; a
commit outside the ranges it is given; and a line a commit only removes, which an earlier commit added (and the scan
read there, when it is in the range). Before committing a compressed file, strip its metadata, or look at it
decompressed.

- **Run it**, from the repository's root, with the key in a file or in `CONFIDENTIAL_SCAN_KEY`:
  ```sh
  python3 scripts/check_confidential.py --key-file /path/to/confidential_scan.key --messages origin/v0.3..HEAD \
    --diffs origin/v0.3..HEAD
  ```
  Without a key it still prints what it finds among the paths, and fails.
- **Add a term** (this needs the key): add it to the private list of terms, one a line, and write the digests again
  from the whole list. A term is found as a word, a pair of adjacent words or a hyphenated run, whatever the case and
  separators: `a-b` finds `A B`, `a_b` and `a-b`. The digest file is shared with OpenMaterica, so a term added is added
  there too.
  ```sh
  python3 scripts/check_confidential.py --key-file /path/to/confidential_scan.key --digest \
    < /path/to/confidential_terms.txt > scripts/confidential_digests.txt
  ```
- **A pre-push hook**, which would check what each push publishes before it does, is not here yet: CI's scan runs only
  once a push is public.

## MADDENING Version Update Policy

When MADDENING publishes a new version:

1. Review MADDENING's CHANGELOG.md — specifically Known Anomalies, Security, Verification sections
2. Review new/changed entries in MADDENING's known_anomalies.yaml
3. Run MIME's full test suite against the candidate version
4. Update the version pin in pyproject.toml
5. Update docs/validation/soup_package.md Section 8
6. Document the update in CHANGELOG.md
