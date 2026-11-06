#!/usr/bin/env python
"""A job that runs one test file alone ran it: none of its tests skipped, none failed, and at least one ran. Run by the
completion-tests jobs (``.github/workflows/ci.yml``) after pytest.

    python3 scripts/check_no_skips.py "$RUNNER_TEMP/report/test-results.xml"

The sharded jobs check this for the whole suite (``scripts/check_shards.py --max-skipped 0``); a job of its own for a
file has no shard record to check, so this reads the JUnit report. Standard library only. Exits 0 when the file's tests
all ran, 1 naming what did not."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _durations_script():
    """``scripts/report_test_durations.py``, which reads a JUnit report."""
    path = Path(__file__).resolve().with_name("report_test_durations.py")
    spec = importlib.util.spec_from_file_location("_check_no_skips_report_test_durations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def problems(report: Path) -> list[str]:
    durations = _durations_script()
    try:
        cases = durations.read_report(report)
    except durations.ReportError as error:
        return [f"the JUnit report cannot be read ({error})"]
    if not cases:
        return [f"{report.name} holds no tests; the file was not run"]
    return [f"{case.nodeid} reported {case.outcome}" for case in cases if case.outcome != "passed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path, help="the job's JUnit report")
    arguments = parser.parse_args(argv)
    found = problems(arguments.report)
    for problem in found:
        print(f"::error title=Completion tests::{problem}")
    if not found:
        print(f"{len(_durations_script().read_report(arguments.report))} tests ran, none skipped")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
