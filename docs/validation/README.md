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
test). `planned_resolution` and `affected_verification_ids` are MIME's own, because MADDENING's schema lacks them.

`scripts/check_anomalies.py` validates exactly that, with pyyaml alone: required fields, MADDENING's status values,
no unknown field, and that every `verification` id names a test that exists. Moving to MADDENING's validator waits for
MADDENING to accept the two fields of MIME's own. The code that 0.2-era entries name is at the tag `pre-0.3`.
