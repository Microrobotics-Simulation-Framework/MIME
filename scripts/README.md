# `scripts/`

**The compliance checks CI runs on every lane.** Each one exits non-zero on failure.

| Script | What it checks |
|---|---|
| `check_anomalies.py` | the known-anomalies registry: required fields, MADDENING's status values, unique permanent IDs, no unknown field, and that each `verification` id names a test that exists (see [`docs/validation/`](../docs/validation/)) |
| `check_confidential.py` | no tracked file's path or content, no commit's message or added lines and paths (`--diffs`), and no branch name names one of the maintainer's private projects or carries a private path; run by CI's `confidential` job, it fails without its key but on a pull request from a fork (OpenMaterica's scan, copied: [`CONTRIBUTING.md`](../CONTRIBUTING.md#confidentiality)); with `--directory DIR` it scans the files under DIR instead, which CI runs over the built documentation pages (untracked, so never in the tree) before it uploads them. `confidential_digests.txt` holds the names' keyed digests, never the names |
| `typing_check.py` | the typing gate, run by CI's `typecheck` job: pyright, at the release the dev group pins, strict over `src/mime`, at zero errors; exits 2 when the run cannot be trusted (pyright missing or another release, an unexpected exit code, no JSON, too few files, a core import unresolved) |
| `report_test_durations.py` | the time budget, run after each lane's tests on the JUnit XML they wrote: the bands, slowest tests and files and the wall and CPU totals on the run page; exits 1 on an unlisted test over 20 s (once re-run alone), 2 when there is nothing to judge. A copy of MADDENING's at v0.4.2 with OpenMaterica's CPU columns, its header listing every change, until MADDENING ships `python -m maddening.compliance report-test-durations`. The budget is in [`tests/README.md`](../tests/README.md) |
| `stress_select.py`, `stress_summary.py` | the stress workflow's two halves (`.github/workflows/stress.yml`, [`CONTRIBUTING.md`](../CONTRIBUTING.md#ci-stress-runs)): what to run, from the pull request's changed tests and a `stress:` line of its description (`runs=`, `python=`, `seed=`, `only`), and the caches to restore; then per test, how many runs passed, with the failures' seeds and Hypothesis blobs. They exit non-zero on a request they refuse and on a failed run; unlike the rows above, they run only when a pull request carries the label `stress` |
| `zenodo_sandbox_lane.py` | `mime push`'s drafts against the Zenodo sandbox, **run by hand, never by CI** (CI never holds an upload token): each folder pushed, read back and pushed again (nothing sent again), the same against the offline fake, and the two compared answer by answer; drafts kept for a person to look at unless `--discard`; with `--publish` (the sandbox only) each draft published, pulled by its number and verified, and the toy's next version published, saying whether a file linked into it is stored once; a log of the answers' shapes, searched for the token before it is written. `--dry-run` replaces the sandbox by a fake of its own name, address and records (see [`mime.push`](../src/mime/push/README.md#testing-it)) |

One script is read by the platforms workflow, not the lanes:

| Script | What it does |
|---|---|
| `platforms_smoke.py` | runs the test files `tests/platforms.txt` names (one `path  # reason` a line) with pytest, with any further arguments passed on; the platforms workflow's macOS and Windows smoke jobs call it, so the selection has one home that `tests/test_platforms_selection.py` checks |

Two scripts are a maintainer's tools, run by hand and never by CI:

| Script | What it does |
|---|---|
| `measure_test_memory.py` | each test's peak memory, from a pytest run it wraps or a file of runs' lines (`--merge`), written to `tests/memory_peaks.json` for `tests/test_memory_heavy.py`'s guard of the `memory_heavy` marker (the tests that run one at a time across xdist's workers); Linux only |
| `live_pace_benchmark.py` | whether A0 served live keeps up with real time on the machine it runs on: unpaced and paced at real time through `mime serve`, timed from the first frame after the compile, with the machine, as JSON. A measurement for `docs/live.md`, never a CI check (CI holds only what pacing guarantees on any machine: no frame ahead of real time) |
| `bump_bases.py` | writes `src/mime/data/env/bases.json` and `uv.json` (uv's image pinned by digest): for each Python series MIME supports, the base image pinned by digest, with its Python, Debian, glibc and own distributions read from the image's layers on Docker Hub; `--check` exits one when the tags have moved. Its output lands in a pull request, never in a build |

`check_anomalies.py` runs inside `mime docs check` (see
[`mime.docs`](../src/mime/docs/README.md)), which CI runs in its default lane: the documentation's sources, the docs
walk, the number rule, the citation check over the built pages and the pages built under eight hash seeds, then
this script, unchanged. The citation check is `mime.docs`' parser of Pandoc's citation grammar, which replaced the
script `check_citations.py`: that read one key of `[@a; @b]` and passed when no page existed.

Run them from the repository root, for example `uv run --no-sync mime docs check` or `python scripts/check_anomalies.py`,
and the typing gate with `uv run python scripts/typing_check.py`.
