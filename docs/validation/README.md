# `docs/validation/`

**The known-anomalies registry.**

[`known_anomalies.yaml`](known_anomalies.yaml) records every known defect, limitation and surprising behaviour, each
under a permanent `MIME-ANO-` id.

## Rules

- **IDs are permanent.** An entry is never renumbered or deleted. A resolved anomaly stays, marked resolved.
- An entry states its safety relevance, with the rationale.
- A new anomaly also gets a line under "Known Anomalies" in `CHANGELOG.md`.

## Status

The registry is still in the 0.2 schema, and the code its entries name is at the tag `pre-0.3`. MADDENING 0.4's
validator does not accept that schema. So `scripts/check_anomalies.py` checks what stays true of a permanent record:
the top-level fields, unique IDs, and the required members. Migrating to MADDENING 0.4's schema without losing
information is a decision still to be made.
