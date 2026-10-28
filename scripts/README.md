# `scripts/`

**The compliance checks CI runs on every lane.** Each one exits non-zero on failure.

| Script | What it checks |
|---|---|
| `check_anomalies.py` | the known-anomalies registry: required fields, MADDENING's status values, unique permanent IDs, no unknown field, and that each `verification` id names a test that exists (see [`docs/validation/`](../docs/validation/)) |
| `check_confidential.py` | no tracked file's path or content, no commit's message or added lines and paths (`--diffs`), and no branch name names one of the maintainer's private projects or carries a private path; run by CI's `confidential` job, it fails without its key but on a pull request from a fork (OpenMaterica's scan, copied: [`CONTRIBUTING.md`](../CONTRIBUTING.md#confidentiality)); with `--directory DIR` it scans the files under DIR instead, which CI runs over the built documentation pages (untracked, so never in the tree) before it uploads them. `confidential_digests.txt` holds the names' keyed digests, never the names |
| `typing_check.py` | the typing gate, run by CI's `typecheck` job: pyright, at the release the dev group pins, strict over `src/mime`, at zero errors; exits 2 when the run cannot be trusted (pyright missing or another release, an unexpected exit code, no JSON, too few files, a core import unresolved) |
| `report_test_durations.py` | the time budget, run after each lane's tests on the JUnit XML they wrote: the bands, slowest tests and files and the wall and CPU totals on the run page; exits 1 on an unlisted test over 20 s (once re-run alone), 2 when there is nothing to judge. A copy of MADDENING's at v0.4.2 with OpenMaterica's CPU columns, its header listing every change, until MADDENING ships `python -m maddening.compliance report-test-durations`. The budget is in [`tests/README.md`](../tests/README.md) |

`check_anomalies.py` runs inside `mime docs check` (see
[`mime.docs`](../src/mime/docs/README.md)), which CI runs in its default lane: the documentation's sources, the docs
walk, the number rule, the citation check over the built pages and the pages built twice under two hash seeds, then
this script, unchanged. The citation check is `mime.docs`' parser of Pandoc's citation grammar, which replaced the
script `check_citations.py`: that read one key of `[@a; @b]` and passed when no page existed.

Run them from the repository root, for example `uv run --no-sync mime docs check` or `python scripts/check_anomalies.py`,
and the typing gate with `uv run python scripts/typing_check.py`.
