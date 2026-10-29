#!/usr/bin/env python
"""A lane's shards ran the lane: run by a lane's own job after its shard jobs (``.github/workflows/ci.yml``).

Each shard job uploads a directory holding ``shard.json`` (what it collected and what it kept, written by
``tests/_sharding.py``), ``outcome.txt`` (its job's status) and its JUnit report. The lane passes only if there are
exactly ``--shards`` of them, one per shard number, each passed, all collected the same tests, their selections
partition that collection (no test run by no shard, none by two), and **the JUnit reports themselves agree with the
records**: every collected test has an outcome in exactly one shard's report, a shard's report holds the tests it
selected and no others, none failed, no shard's tests were all skipped, and the lane skipped no more than
``--max-skipped``. A record is what a shard says it did; the report is what ran. A shard that never uploaded (it failed
to install, was cancelled or never started) is missing, and fails the lane. Standard library only (it reads the
reports with ``scripts/report_test_durations.py``), so the lane's job needs no install to run it.

    python3 scripts/check_shards.py --shards 4 --max-skipped 0 results/*/

Exits 0 when the lane holds, 1 naming each problem otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


def _durations_script():
    """``scripts/report_test_durations.py``, which rebuilds a test's node id from its JUnit ``<testcase>``."""
    path = Path(__file__).resolve().with_name("report_test_durations.py")
    spec = importlib.util.spec_from_file_location("_check_shards_report_test_durations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _outcomes(directory: Path, number: int) -> tuple[dict[str, str] | None, list[str]]:
    """``{node id: outcome}`` from the one JUnit report in a shard's directory, and what is wrong with the report
    (``None`` when it cannot be read at all). The outcome is ``passed``, ``failure``, ``error`` or ``skipped``."""
    reports = sorted(directory.glob("test-results-shard*.xml"))
    if len(reports) != 1:
        return None, [f"shard {number}: {len(reports)} JUnit reports in {directory.name}; there is one"]
    durations = _durations_script()
    try:
        cases = durations.read_report(reports[0])
    except durations.ReportError as error:
        return None, [f"shard {number}: its JUnit report cannot be read ({error})"]
    seen = Counter(case.nodeid for case in cases)
    found = [f"shard {number}: {nodeid} has {count} outcomes in its JUnit report" for nodeid, count in
             sorted(seen.items()) if count > 1][:20]
    return {case.nodeid: case.outcome for case in cases}, found


def problems(directories: list[Path], shards: int, max_skipped: int) -> list[str]:
    found: list[str] = []
    records: dict[int, dict[str, object]] = {}
    reports: dict[int, dict[str, str]] = {}
    for directory in directories:
        try:
            record = json.loads((directory / "shard.json").read_text(encoding="utf-8"))
            outcome = (directory / "outcome.txt").read_text(encoding="utf-8").strip()
        except (OSError, ValueError) as error:
            found.append(f"{directory}: no readable shard record and outcome ({error})")
            continue
        number = record.get("shard")
        if record.get("of") != shards or not isinstance(number, int) or not 1 <= number <= shards:
            found.append(f"{directory}: shard {number!r} of {record.get('of')!r}; this lane runs {shards}")
            continue
        if number in records:
            found.append(f"{directory}: a second record for shard {number}")
            continue
        records[number] = record
        if outcome != "success":
            found.append(f"shard {number} of {shards}: its job's status is {outcome!r}")
        report, report_problems = _outcomes(directory, number)
        found += report_problems
        if report is not None:
            reports[number] = report
    missing = sorted(set(range(1, shards + 1)) - set(records))
    if missing:
        found.append(f"no record from shard(s) {', '.join(map(str, missing))} of {shards}: the job failed before it "
                     f"uploaded one, or never ran")
    if not records:
        return found
    collections = {tuple(record["collected"]) for record in records.values()}  # type: ignore[arg-type]
    if len(collections) != 1:
        found.append(f"the shards collected {len(collections)} different sets of tests; one collection is split")
        return found
    (collected,) = collections
    if not collected:
        found.append("the lane collected no tests")
    selected: dict[str, list[int]] = {}
    for number, record in sorted(records.items()):
        for nodeid in record["selected"]:  # type: ignore[union-attr]
            selected.setdefault(nodeid, []).append(number)
    twice = sorted(nodeid for nodeid, numbers in selected.items() if len(numbers) > 1)
    stray = sorted(set(selected) - set(collected))
    unrun = sorted(set(collected) - set(selected)) if not missing else []
    found += [f"{nodeid}: kept by shards {selected[nodeid]}" for nodeid in twice[:20]]
    found += [f"{nodeid}: kept by a shard and not collected" for nodeid in stray[:20]]
    found += [f"{nodeid}: collected and kept by no shard" for nodeid in unrun[:20]]
    found += report_problems_of(records, reports, set(collected), missing, max_skipped)
    return found


def report_problems_of(records: dict[int, dict[str, object]], reports: dict[int, dict[str, str]],
                       collected: set[str], missing: list[int], max_skipped: int) -> list[str]:
    """What the JUnit reports say against the records and the collection."""
    found: list[str] = []
    reported: dict[str, list[int]] = {}
    for number, outcomes in sorted(reports.items()):
        kept = set(records[number]["selected"])  # type: ignore[arg-type]
        absent = sorted(kept - set(outcomes))
        extra = sorted(set(outcomes) - kept)
        found += [f"shard {number}: kept {nodeid} and its JUnit report has no outcome for it" for nodeid in absent[:20]]
        found += [f"shard {number}: its JUnit report holds {nodeid}, which it did not keep" for nodeid in extra[:20]]
        if not outcomes:
            found.append(f"shard {number}: its JUnit report holds no tests")
        elif all(outcome == "skipped" for outcome in outcomes.values()):
            found.append(f"shard {number}: all {len(outcomes)} of its tests were skipped")
        bad = sorted(nodeid for nodeid, outcome in outcomes.items() if outcome in ("failure", "error"))
        found += [f"shard {number}: {nodeid} reported {outcomes[nodeid]}" for nodeid in bad[:20]]
        for nodeid in outcomes:
            reported.setdefault(nodeid, []).append(number)
    if not missing and len(reports) == len(records):
        found += [f"{nodeid}: collected and reported by no shard's JUnit" for nodeid in sorted(collected - set(reported))[:20]]
    found += [f"{nodeid}: reported by shards {numbers}" for nodeid, numbers in sorted(reported.items())
              if len(numbers) > 1][:20]
    found += [f"{nodeid}: reported by a shard and not collected" for nodeid in sorted(set(reported) - collected)[:20]]
    skipped = sum(outcome == "skipped" for outcomes in reports.values() for outcome in outcomes.values())
    if skipped > max_skipped:
        found.append(f"{skipped} tests were skipped; this lane allows {max_skipped}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directories", nargs="*", type=Path, help="one directory per shard job's upload")
    parser.add_argument("--shards", type=int, required=True, help="how many shards the lane runs")
    parser.add_argument("--max-skipped", type=int, required=True,
                        help="how many tests the lane may skip: what the unsplit lane skipped")
    args = parser.parse_args(argv)
    found = problems(args.directories, args.shards, args.max_skipped)
    for problem in found:
        print(f"::error title=Shards::{problem}")
    if not found:
        kept = len(json.loads((args.directories[0] / "shard.json").read_text(encoding="utf-8"))["collected"])
        print(f"{args.shards} shards passed and together ran the {kept} tests collected")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
