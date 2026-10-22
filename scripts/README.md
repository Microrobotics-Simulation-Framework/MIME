# `scripts/`

**The compliance checks CI runs on every lane.** Each one exits non-zero on failure.

| Script | What it checks |
|---|---|
| `check_anomalies.py` | the known-anomalies registry: required fields, unique permanent IDs, required members (see [`docs/validation/`](../docs/validation/)) |
| `check_citations.py` | every `[@Key]` citation under `docs/` resolves to an entry in `docs/bibliography.bib` |
| `check_impl_mapping.py` | every name in an algorithm guide's implementation-mapping table imports and resolves to a callable |
| `typing_check.py` | the typing gate, run by CI's `typecheck` job: pyright, at the release the dev group pins, strict over `src/mime`, at zero errors; exits 2 when the run cannot be trusted (pyright missing or another release, an unexpected exit code, no JSON, too few files, a core import unresolved) |

While the docs are being rebuilt, `check_citations.py` and `check_impl_mapping.py` pass because there is nothing yet
to check. They start to bite as docs return with their code.

Run them from the repository root, for example `python scripts/check_anomalies.py`, and the typing gate with
`uv run python scripts/typing_check.py`.
