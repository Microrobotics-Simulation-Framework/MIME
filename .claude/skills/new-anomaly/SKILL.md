# New Anomaly Skill — MIME

This skill guides the agent through adding a new entry to `docs/validation/known_anomalies.yaml`. The anomaly registry is a Notified Body-facing compliance artifact.

## Trigger

When the user reports a bug, discovers a limitation, or invokes `/new-anomaly`.

---

## Step 0 — Check if this needs an anomaly entry

**Add an anomaly entry if:**
- A known limitation produces incorrect or missing output without raising an error
- A behaviour that could mislead a downstream system is being documented rather than fixed now
- A bug is being fixed — the anomaly entry records it was open and is now resolved

**Do NOT add for:**
- Bugs fixed in the same commit they are discovered
- Feature gaps not yet implemented (use ARCHITECTURE_PLAN.md Phase roadmap instead)

---

## Step 1 — Assign the next ID

Look at the last `anomaly_id` in `docs/validation/known_anomalies.yaml` and increment.
Format: `MIME-ANO-XXX` (zero-padded, sequential, never reused).

---

## Step 2 — Classify severity

| Severity | When to use |
|---|---|
| `"critical"` | Incorrect results, no workaround |
| `"major"` | Incorrect results, workaround exists |
| `"minor"` | Cosmetic or inconvenience, no correctness impact |
| `"enhancement"` | Not a defect — a missing feature |

---

## Step 3 — Assess safety relevance

| Value | When to use |
|---|---|
| `"safety_relevant"` | Inherently safety-critical regardless of context |
| `"not_safety_relevant"` | Provably cannot affect any safety-critical output path |
| `"context_dependent"` | Depends on downstream deployment — **use when in doubt** |

The `safety_relevance_rationale` must be understandable by someone unfamiliar with the codebase. If cross-referencing a MADDENING anomaly, cite it by ID: "Inherits from MADD-ANO-001."

---

## Step 4 — Write the YAML entry

Append to the `anomalies:` list:

```yaml
  - anomaly_id: "MIME-ANO-XXX"
    title: "Short, specific, searchable title"
    description: >
      Detailed description of what happens, under what conditions,
      and what the expected behaviour would be.
    affected_components: ["NodeClassName"]
    affected_versions: "0.1.0 – current"
    severity: "major"
    safety_relevance: "context_dependent"
    safety_relevance_rationale: >
      Explanation of why this assessment was made.
    workaround: "Specific actionable workaround, or 'None available'"
    resolution_status: "open"
    github_issue: null
    date_reported: "YYYY-MM-DD"
```

---

## Step 5 — Update CHANGELOG.md

Under `## [Unreleased]` → `### Known Anomalies`:

```markdown
- `MIME-ANO-XXX`: one-line summary
```

---

## Step 6 — Run compliance check

```bash
python scripts/check_anomalies.py
```

Must exit 0.

---

## Updating an existing anomaly (when a fix lands)

1. Change `resolution_status` to `"fixed"`
2. Add CHANGELOG entry under `### Fixed`
3. Run `python scripts/check_anomalies.py`

**Never delete an anomaly entry, even after it is fixed.**
