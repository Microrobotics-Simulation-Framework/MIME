"""The dtype audit's report: the wide operations in each configuration's float32 step, with 64-bit types enabled.

    python scripts/precision_audit.py [--json OUT.json] [--only SUBSTRING ...]

Every configuration of ``tests/saved/switch_configurations.py`` (script A's arms and every round-trip case whose build is
not refused, which together build every registered entry's nodes, and the test-only rigs beside them) and A0
(``tests/script_a0.py``) is built at ``precision="float32"`` and its compiled step traced (``mime.testing._dtype_audit``).
Report-only (precision's P1): it prints and records, and exits 0 whatever it finds, and 1 only where it itself cannot
run (no configuration at all). A configuration whose build or trace fails is listed as an error with its message and is
not counted."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tests" / "saved"), str(ROOT / "tests"), str(ROOT / "examples")]


def audits(only: list[str]) -> list[dict[str, Any]]:
    from switch_configurations import configurations

    from mime.build import BuildRefused
    from mime.build._maddening import _held
    from mime.testing._dtype_audit import audit_graph

    entries = dict(configurations())
    from script_a0 import a0

    entries["a0/tutorial_rig"] = a0
    found: list[dict[str, Any]] = []
    for name, make in entries.items():
        if only and not any(part in name for part in only):
            continue
        try:
            experiment = make()
            experiment._settings = dataclasses.replace(experiment._settings, precision="float32")  # noqa: SLF001
            built = experiment.build()
            audit = audit_graph(name, built.graph, built.held_inputs())
            record = audit.record()
            record["coupling_groups"] = len(_held(built.graph)._coupling_groups)  # noqa: SLF001
            record["text"] = audit.text()
        except BuildRefused as refused:
            record = {"name": name, "error": f"refused here: {', '.join(refused.codes())}"}
        except Exception as error:  # report-only: one configuration's failure is a row, not the run's
            record = {"name": name, "error": f"{type(error).__name__}: {str(error)[:300]}"}
        print(record.get("text") or f"{name}: ERROR {record['error']}", flush=True)
        found.append(record)
    return found


def summary(records: list[dict[str, Any]]) -> str:
    ok = [r for r in records if "error" not in r]
    errors = [r for r in records if "error" in r]
    dirty = [r for r in ok if r["wide"] or r["drifted"]]
    by_line: Counter[str] = Counter()
    for r in ok:
        for origin in r["origins"]:
            by_line[f"{origin['primitive']} {origin['dtypes']} {origin['line']}"] += origin["count"]
    lines = [f"configurations audited: {len(ok)}; errors: {len(errors)}; with wide operations or drifted state: {len(dirty)}",
             f"wide operations in all: {sum(r['wide'] for r in ok)} of {sum(r['equations'] for r in ok)} equations; "
             f"drifted fields: {sum(len(r['drifted']) for r in ok)}",
             "per configuration (wide / equations / drifted):"]
    lines += [f"  {r['name']}: {r['wide']} / {r['equations']} / {len(r['drifted'])}" for r in ok]
    lines += [f"  {r['name']}: ERROR {r['error']}" for r in errors]
    lines.append("origins of the wide operations, by source line (most first):")
    lines += [f"  {n:5d} {key}" for key, n in by_line.most_common(40)]
    return "\n".join(lines)


def main() -> int:
    warnings.simplefilter("ignore")  # the builds' warnings are the configurations', not the audit's
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--only", nargs="*", default=[])
    arguments = parser.parse_args()
    records = audits(arguments.only)
    report = summary(records)
    print("\n== SUMMARY ==\n" + report)
    if arguments.json:
        arguments.json.write_text(json.dumps({"records": records, "summary": report}, indent=1))
    return 0 if any("error" not in r for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
