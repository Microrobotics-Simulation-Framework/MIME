# `scripts/`

**The compliance checks CI runs on every lane.** Each one exits non-zero on failure.

| Script | What it checks |
|---|---|
| `check_anomalies.py` | the known-anomalies registry: required fields, MADDENING's status values, unique permanent IDs, no unknown field, and that each `verification` id names a test that exists (see [`docs/validation/`](../docs/validation/)) |
| `check_citations.py` | every `[@Key]` citation under `docs/` resolves to an entry in `docs/bibliography.bib` |
| `check_impl_mapping.py` | every name in an algorithm guide's implementation-mapping table imports and resolves to a callable |
| `typing_check.py` | the typing gate, run by CI's `typecheck` job: pyright, at the release the dev group pins, strict over `src/mime`, at zero errors; exits 2 when the run cannot be trusted (pyright missing or another release, an unexpected exit code, no JSON, too few files, a core import unresolved) |
| `report_test_durations.py` | the time budget, run after each lane's tests on the JUnit XML they wrote: the bands, slowest tests and files and the wall and CPU totals on the run page; exits 1 on an unlisted test over 20 s (once re-run alone), 2 when there is nothing to judge. A copy of MADDENING's at v0.4.2 with OpenMaterica's CPU columns, its header listing every change, until MADDENING ships `python -m maddening.compliance report-test-durations`. The budget is in [`tests/README.md`](../tests/README.md) |

While the docs are being rebuilt, `check_citations.py` and `check_impl_mapping.py` pass because there is nothing yet
to check. They start to bite as docs return with their code.

Run them from the repository root, for example `python scripts/check_anomalies.py`, and the typing gate with
`uv run python scripts/typing_check.py`.
