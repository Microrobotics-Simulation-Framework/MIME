#!/usr/bin/env python3
"""CI script — validate MIME's known anomalies registry.

The registry is in MADDENING's field names where MADDENING has them (``resolution_status``,
``affected_components``, ``affected_versions``, ``resolution_version``, ``verification``), plus the two fields
MADDENING lacks, ``planned_resolution`` and ``affected_verification_ids``. This script validates exactly that,
until MADDENING's ``validate_anomaly_registry`` accepts those two fields. It needs pyyaml and nothing else, as the
0.2 line's compliance job installed pyyaml and not MADDENING.

What it checks: the top-level fields; unique IDs with the MIME-ANO- prefix; each entry's required fields; that
``resolution_status`` takes only MADDENING's values; that an entry carries no field outside the known set; that a
``resolved`` entry has a ``resolution_version`` and a ``verification`` list (possibly empty); and that every
``verification`` entry names a test that exists, meaning its file exists and defines a function of that name.
It does not parse the ``affected_versions`` ranges.
"""

import re
import sys
from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parent.parent

# MADDENING's values for resolution_status.
RESOLUTION_STATUSES = ("open", "resolved", "partially_resolved", "wont_fix", "duplicate")

REQUIRED_FIELDS = (
    "anomaly_id", "title", "description",
    "severity", "safety_relevance", "safety_relevance_rationale",
    "resolution_status", "affected_versions",
)
# MADDENING's optional fields, then the two of MIME's own.
OPTIONAL_FIELDS = (
    "affected_components", "workaround", "resolution_version", "verification", "residual_risk", "github_issue",
    "planned_resolution", "affected_verification_ids",
)
KNOWN_FIELDS = frozenset(REQUIRED_FIELDS + OPTIONAL_FIELDS)

RENAMED_FIELDS = {"status": "resolution_status", "affected_nodes": "affected_components"}


def verification_errors(aid, ids, repository):
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return [f"{aid}: 'verification' must be a list of pytest node ids"]
    errors = []
    for node_id in ids:
        file_part, _, test_part = node_id.partition("::")
        function = re.sub(r"\[.*\]$", "", test_part.rsplit("::", 1)[-1])
        if not function:
            errors.append(f"{aid}: verification '{node_id}' names no test (expected 'file::test_name')")
            continue
        test_file = repository / file_part
        if not test_file.is_file():
            errors.append(f"{aid}: verification '{node_id}': file '{file_part}' does not exist")
        elif not re.search(rf"\bdef {re.escape(function)}\b", test_file.read_text()):
            errors.append(f"{aid}: verification '{node_id}': no 'def {function}' in '{file_part}'")
    return errors


def entry_errors(aid, entry, repository):
    errors = []
    for field in entry:
        if field in RENAMED_FIELDS:
            errors.append(f"{aid}: field '{field}' is now '{RENAMED_FIELDS[field]}'")
        elif field not in KNOWN_FIELDS:
            errors.append(f"{aid}: unknown field '{field}'")
    for field in REQUIRED_FIELDS:
        if not entry.get(field):
            errors.append(f"{aid}: missing required field '{field}'")
    status = entry.get("resolution_status")
    if status and status not in RESOLUTION_STATUSES:
        errors.append(f"{aid}: resolution_status '{status}' is not one of {', '.join(RESOLUTION_STATUSES)}")
    if status == "resolved":
        if not entry.get("resolution_version"):
            errors.append(f"{aid}: a resolved entry needs 'resolution_version'")
        if "verification" not in entry:
            errors.append(f"{aid}: a resolved entry needs a 'verification' list (empty if no test exists)")
    if "verification" in entry:
        errors += verification_errors(aid, entry["verification"], repository)
    return errors


def validate_anomaly_registry(path, *, prefix="", repository=REPOSITORY):
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return ["File must contain a YAML mapping"]
    errors = []
    for req in ("schema_version", "generated_date"):
        if req not in data:
            errors.append(f"Missing top-level field: {req}")
    ids_seen = set()
    for a in data.get("anomalies", []):
        aid = a.get("anomaly_id", "<missing>")
        if aid in ids_seen:
            errors.append(f"Duplicate anomaly_id: {aid}")
        ids_seen.add(aid)
        if prefix and not str(aid).startswith(prefix):
            errors.append(f"{aid}: does not match required prefix '{prefix}'")
        errors += entry_errors(aid, a, Path(repository))
    return errors


def main():
    path = "docs/validation/known_anomalies.yaml"
    errors = validate_anomaly_registry(path, prefix="MIME-ANO-")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"OK: anomaly registry at {path} is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
