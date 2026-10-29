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
