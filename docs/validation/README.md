# `docs/validation/`

**The known-anomalies registry.**

[`known_anomalies.yaml`](known_anomalies.yaml) records every known defect, limitation and surprising behaviour, each
under a permanent `MIME-ANO-` id.

## Rules

- **IDs are permanent.** An entry is never renumbered or deleted. A resolved anomaly stays, marked resolved.
- An entry states its safety relevance, with the rationale.
- A new anomaly also gets a line under "Known Anomalies" in `CHANGELOG.md`.

## Schema

One registry runs across the 0.2 line and 0.3: nothing is wiped, and each entry carries its status for the version in
`mime_version`. The field names are MADDENING's where it has them: `resolution_status` (`open`, `resolved`,
`partially_resolved`, `wont_fix`, `duplicate`), `affected_components`, `affected_versions` (a PEP 440 range), and for a
`resolved` entry `resolution_version` and `verification` (pytest node ids; empty where the code is gone and so is its
test). `planned_resolution`, `affected_verification_ids` and `claims` are MIME's own, because MADDENING's schema lacks
them. "Removed" is not a status: an entry whose code 0.3 does not carry says so in `planned_resolution` and
`resolution_version`.

**Evidence** resolves through pytest's collection, never through a file's text: a `verification` id, or a claim's
evidence, names a test pytest collects (`tests/<file>.py::<test>`, with its parameters or without them for every
parametrisation), or a MADDENING `@verification_benchmark` id. A test renamed with its old name left in a comment fails.
**Evidence is a test that ran and passed in a named lane**: where a lane's JUnit report is read, only a pass there
counts; where only the collection is (`mime docs check`), a test marked `skip`, `xfail`, `slow` (no CI lane runs one)
or `gpu` (CI has no CUDA) prints "not run in CI" and does not count, and a `skipif` counts, since CI's lanes allow no
skipped test. A claim whose evidence, and a `resolved` entry whose verification, are all tests CI does not run fail
the check. A test collected and passing may still test nothing (a body of `pass`), and nothing ties a cited test to the
claim that cites it: that is each test's own control's to show, not this check's.

**A claim** (`claims`, a list) is a number the record states, kept with its owner so that a test reads it rather than
types it again (`mime.docs.claim("MIME-ANO-114.max_error", <registry>)`). It holds `id`, `quantity`, `value` and
`unit`, its `basis` (`declared`, `computed`, `measured`, `fitted`), its `role` (`bound`, `estimate`, `acceptance`,
`limit`, `rule`), its `oracle_kind` (`analytic`, `manufactured`, `software oracle`, `implementation`, `measurement`),
its `evidence`, its `conditions` (a point or a region), for a bound its `domain`, `n` and `held_out` evidence, for a fit
its `dataset`, and `says`, one line of what it bounds and what it estimates. The basis decides the roles: a declared
claim is a limit or a rule and cites no test; an acceptance is measured, and claims no accuracy; a bound states its
domain and the points it rests on, and rests on held-out evidence (an id not among its own evidence), whatever its
basis, since a maximum over finitely many points is otherwise an estimate; a value is a finite number; an
implementation oracle (the code's own formula evaluated again) is not accuracy evidence, and is never a bound's.

**`affected_components`** names a registered component in the controlled form: `entry: <kind> <name>` with its
version and `, rung <rung>` where they matter and `; <note>` for the rest, or `node: <class>`. A `MIME-NODE-` id is a
permanent id. Any other spelling is a legacy one, allowed while it names no registered entry or node, and refused once
one of that name exists, so that the conversion is done when it can be. The check matches names: it does not read the
migration table (`docs/data/migration_0.2.yaml`), so a 0.2 name whose 0.3 counterpart is registered under another name
(`MLPResistanceNode`, now `drag` at its rung `surrogate`; `PermanentMagnetNode`, now `field_provider point_dipole`) stays
an allowed legacy spelling.

`scripts/check_anomalies.py` validates exactly that, with the evidence resolver of `mime.docs`: required fields,
MADDENING's status values, no unknown field, every evidence id collected, each claim's shape, and the controlled form.
Moving to MADDENING's validator waits for MADDENING to accept the fields of MIME's own. The code that 0.2-era entries
name is at the tag `pre-0.3`.

**The evidence counts** (`mime docs build` writes `reference/evidence_counts.md`) list every claim and every
specification statement with the outcome of its tests in the default lane's JUnit report: counts first, grouped by role,
with the line that a test's body is not checked. It is not a regulatory record or certificate. The pages are built and
never committed (`src/mime/docs/README.md`).
