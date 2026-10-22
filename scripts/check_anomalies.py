#!/usr/bin/env python3
"""CI script — validate MIME's known anomalies registry.

The registry is in MIME's 0.2 schema (``status``, ``affected_nodes``, ``affected_verification_ids``,
``planned_resolution``), which MADDENING 0.4's ``validate_anomaly_registry`` does not accept, and the code its entries
name is at the tag ``pre-0.3``. So this checks what stays true of the permanent record: the top-level fields, unique
IDs with the MIME-ANO- prefix, and each entry's required fields. It is the check the 0.2 line's CI ran, whose
compliance job installed pyyaml and not MADDENING.
"""

import sys

import yaml


def validate_anomaly_registry(path, *, prefix=""):
    with open(path) as f:
        data = yaml.safe_load(f)
    errors = []
    if not isinstance(data, dict):
        return ["File must contain a YAML mapping"]
    for req in ("schema_version", "generated_date"):
        if req not in data:
            errors.append(f"Missing top-level field: {req}")
    ids_seen = set()
    for a in data.get("anomalies", []):
        aid = a.get("anomaly_id", "<missing>")
        if aid in ids_seen:
            errors.append(f"Duplicate anomaly_id: {aid}")
        ids_seen.add(aid)
        if prefix and not aid.startswith(prefix):
            errors.append(f"{aid}: does not match required prefix '{prefix}'")
        for field in ("anomaly_id", "title", "description",
                      "severity", "safety_relevance",
                      "safety_relevance_rationale"):
            if not a.get(field):
                errors.append(f"{aid}: missing required field '{field}'")
    return errors


path = "docs/validation/known_anomalies.yaml"
errors = validate_anomaly_registry(path, prefix="MIME-ANO-")
if errors:
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
print(f"OK: anomaly registry at {path} is valid")
