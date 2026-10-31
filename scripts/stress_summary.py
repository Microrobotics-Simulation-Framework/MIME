#!/usr/bin/env python
"""The result of a stress run (``.github/workflows/stress.yml``): per test, in how many of the runs it passed; for each
failure the run, its Hypothesis seed and reproduction blob; the seeds. Standard library only.

It reads ``<directory>/*/junit.xml``, one report per run's artifact (``stress-run-<k>``), and the run numbers
1..``--runs``. A run is **failed** when its report is missing or unreadable, a test failed or errored in it, or no test
passed in it (a selection that only skipped proves nothing). The summary goes to ``$GITHUB_STEP_SUMMARY`` (else stdout);
the exit status is 1 if any run failed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

#: What the summary page keeps of a failure, and of the passing tests' table (GitHub caps a job summary at 1 MiB).
BLOB_LINES, MAX_FAILURES_SHOWN, MAX_PASSING_ROWS = 14, 3, 1500
BLOB = re.compile(r"@reproduce_failure\(.*\)")


@dataclass
class Report:
    """One run's outcomes: test -> ``passed``/``failed``/``skipped``, and the failing tests' text."""
    outcomes: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    texts: dict[str, str] = field(default_factory=lambda: dict[str, str]())


def read_report(path: Path) -> Report | None:
    """``None`` when the file is missing or is not a JUnit report."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    report = Report()
    for case in root.iter("testcase"):
        test = f"{case.get('classname', '')}::{case.get('name', '')}"
        bad = case.find("failure") if case.find("failure") is not None else case.find("error")
        if bad is not None:
            report.outcomes[test] = "failed"
            report.texts[test] = f"{bad.get('message', '')}\n{bad.text or ''}"
        elif case.find("skipped") is not None:
            report.outcomes.setdefault(test, "skipped")
        else:
            report.outcomes.setdefault(test, "passed")
    return report


def run_failed(report: Report | None) -> str | None:
    """Why a run counts as failed, or ``None``."""
    if report is None:
        return "no readable report"
    failed = sum(outcome == "failed" for outcome in report.outcomes.values())
    if failed:
        return f"{failed} failed"
    if "passed" not in report.outcomes.values():
        return "no test passed"
    return None


def reproduction(text: str) -> str:
    """The falsifying example and the ``@reproduce_failure`` line of a failure's text, else its first lines."""
    lines = text.strip().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Falsifying example")), None)
    kept = lines[start:start + BLOB_LINES] if start is not None else lines[:BLOB_LINES]
    blob = BLOB.search(text)
    if blob and blob.group(0) not in kept:
        kept.append(blob.group(0))
    return "\n".join(kept)


def summarize(reports: dict[int, Report | None], seeds: dict[int, int]) -> tuple[str, bool]:
    """``(markdown, ok)`` for the runs' reports (run -> report) and their seeds."""
    runs = len(reports)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})
    failures: dict[str, list[tuple[int, str]]] = defaultdict(list)
    bad_runs: dict[int, str] = {}
    for run, report in sorted(reports.items()):
        why = run_failed(report)
        if why:
            bad_runs[run] = why
        for test, outcome in (report.outcomes if report else {}).items():
            counts[test][outcome] += 1
            if outcome == "failed":
                failures[test].append((run, reproduction(report.texts[test] if report else "")))
    ok = not bad_runs
    seed_list = ", ".join(f"{run}: {seeds[run]}" for run in sorted(reports))
    out = [f"## Stress: {runs - len(bad_runs)} of {runs} runs passed" + ("" if ok else " (FAILED)"), ""]
    if bad_runs:
        out += ["Failed runs (the Hypothesis seed repeats one locally, see CONTRIBUTING.md):", ""]
        out += [f"- run {run}, seed `{seeds[run]}`: {why}" for run, why in bad_runs.items()]
        out.append("")
    rows = sorted(counts.items(), key=lambda kv: (kv[1]["passed"] == runs, kv[0]))
    weak = [row for row in rows if row[1]["passed"] != runs]
    out += ["| test | passed | failed | skipped |", "|---|---|---|---|"]
    out += [f"| `{test}` | **{c['passed']}/{runs}** | {c['failed']} | {c['skipped']} |" for test, c in weak]
    solid = [row for row in rows if row[1]["passed"] == runs]
    out += [f"| `{test}` | {c['passed']}/{runs} | 0 | 0 |" for test, c in solid[:MAX_PASSING_ROWS]]
    if len(solid) > MAX_PASSING_ROWS:
        out.append(f"| ...{len(solid) - MAX_PASSING_ROWS} more tests that passed every run | | | |")
    for test, shown in failures.items():
        out += ["", f"### `{test}`: failed in {len(shown)} of {runs} runs", ""]
        for run, text in shown[:MAX_FAILURES_SHOWN]:
            out += [f"Run {run}, seed `{seeds[run]}`:", "```", text, "```"]
        if len(shown) > MAX_FAILURES_SHOWN:
            out.append(f"...and runs {', '.join(str(r) for r, _ in shown[MAX_FAILURES_SHOWN:])}.")
    out += ["", f"Seeds (run: seed): {seed_list}"]
    return "\n".join(out) + "\n", ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="holds one stress-run-<k>/junit.xml per run")
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True, help="the first run's seed; run k used seed + k - 1")
    args = parser.parse_args(argv)
    reports = {k: read_report(args.directory / f"stress-run-{k}" / "junit.xml") for k in range(1, args.runs + 1)}
    text, ok = summarize(reports, {k: args.seed + k - 1 for k in reports})
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as out:
            out.write(text)
    sys.stdout.write(text if not target else text.split("\n", 1)[0] + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
