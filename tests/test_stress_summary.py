"""The result of a stress run (``scripts/stress_summary.py``): per test, how many runs passed; a failure's seed and
Hypothesis blob; a missing, empty or all-skipped run counts as failed. Each control plants a failure and expects the
summary to find it in every run it was planted in, not most."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_stress_summary", ROOT / "scripts" / "stress_summary.py")
assert spec is not None and spec.loader is not None
summary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summary  # a dataclass reads its module
spec.loader.exec_module(summary)

BLOB = "@reproduce_failure('6.170.5', b'AXicY2BgYAAAAAQAAQ==')"
FAILURE = (f"x = 3\n\nFalsifying example: test_b(\n    x=3,\n)\nYou can reproduce this example by temporarily "
           f"adding {BLOB} as a decorator on your test function\n")


def junit(cases: dict[str, str]) -> str:
    """A JUnit report (xunit1) of ``{test name: passed | failed | error | skipped}``."""
    body = ""
    for name, outcome in cases.items():
        inner = {"passed": "", "skipped": '<skipped message="s"/>', "failed": f'<failure message="boom">{FAILURE}</failure>',
                 "error": "<error message=\"collection\">trace</error>"}[outcome]
        body += f'<testcase classname="tests.test_a" name="{name}" time="0.1">{inner}</testcase>'
    return f'<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="{len(cases)}">{body}</testsuite></testsuites>'


def write_runs(directory: Path, runs: list[dict[str, str] | None]) -> None:
    """One part per run: ``stress-run-<k>-part-1``."""
    for k, cases in enumerate(runs, start=1):
        if cases is not None:
            (directory / f"stress-run-{k}-part-1").mkdir()
            (directory / f"stress-run-{k}-part-1" / "junit.xml").write_text(junit(cases), encoding="utf-8")


def read_page(tmp_path: Path, runs: int, parts: int, seed: int) -> tuple[int, str]:
    page = tmp_path / "page.md"
    import os
    os.environ["GITHUB_STEP_SUMMARY"] = str(page)
    try:
        status = summary.main([str(tmp_path), "--runs", str(runs), "--parts", str(parts), "--seed", str(seed)])
    finally:
        del os.environ["GITHUB_STEP_SUMMARY"]
    return status, page.read_text(encoding="utf-8")


def run_summary(tmp_path: Path, runs: list[dict[str, str] | None], seed: int = 100, capsys=None):
    write_runs(tmp_path, runs)
    return read_page(tmp_path, len(runs), 1, seed)


def write_parts(directory: Path, runs: list[list[dict[str, str] | None]]) -> None:
    """``runs[k - 1][p - 1]`` is the cases of run k's part p, or ``None`` for a job that left no report."""
    for k, parts in enumerate(runs, start=1):
        for p, cases in enumerate(parts, start=1):
            if cases is not None:
                (directory / f"stress-run-{k}-part-{p}").mkdir()
                (directory / f"stress-run-{k}-part-{p}" / "junit.xml").write_text(junit(cases), encoding="utf-8")


GOOD = {"test_a": "passed", "test_b": "passed"}


def test_every_run_passing_is_a_pass_with_per_test_counts(tmp_path):
    status, page = run_summary(tmp_path, [GOOD] * 4)
    assert status == 0
    assert "4 of 4 runs passed" in page and "| `tests.test_a::test_a` | 4/4 | 0 | 0 |" in page
    assert "Seeds (run: seed): 1: 100, 2: 101, 3: 102, 4: 103" in page


def test_one_failing_run_fails_the_summary_and_names_its_seed_and_blob(tmp_path):
    runs = [GOOD, GOOD, {"test_a": "passed", "test_b": "failed"}, GOOD]
    status, page = run_summary(tmp_path, runs)
    assert status == 1
    assert "3 of 4 runs passed (FAILED)" in page
    assert "run 3, seed `102`: 1 failed" in page
    assert "| `tests.test_a::test_b` | **3/4** | 1 | 0 |" in page
    assert BLOB in page and "Falsifying example: test_b(" in page


@pytest.mark.parametrize("planted", ["failed", "error"])
def test_control_a_failure_planted_in_every_run_is_found_in_every_run(tmp_path, planted):
    """The control that fails every run, not most: a planted failure in each of 8 runs fails each of them."""
    status, page = run_summary(tmp_path, [{"test_a": "passed", "test_b": planted}] * 8)
    assert status == 1
    assert "0 of 8 runs passed" in page and page.count("seed `") >= 8
    assert "| `tests.test_a::test_b` | **0/8** |" in page


def test_a_run_without_a_report_is_failed_not_missing(tmp_path):
    status, page = run_summary(tmp_path, [GOOD, None, GOOD])
    assert status == 1
    assert "run 2, seed `101`: no readable report" in page and "2 of 3 runs passed" in page


def test_an_unparseable_report_is_failed(tmp_path):
    write_runs(tmp_path, [GOOD])
    (tmp_path / "stress-run-1-part-1" / "junit.xml").write_text("<testsuites><oops", encoding="utf-8")
    assert summary.read_report(tmp_path / "stress-run-1-part-1" / "junit.xml") is None


def test_a_run_where_nothing_passed_is_failed_so_a_selection_that_only_skips_proves_nothing(tmp_path):
    status, page = run_summary(tmp_path, [{"test_a": "skipped"}, {}])
    assert status == 1 and page.count("no test passed") == 2


def test_a_skip_beside_a_pass_is_counted_but_does_not_fail(tmp_path):
    status, page = run_summary(tmp_path, [{"test_a": "passed", "test_s": "skipped"}] * 2)
    assert status == 0 and "| `tests.test_a::test_s` | **0/2** | 0 | 2 |" in page


def test_more_failing_runs_than_shown_are_listed_by_number(tmp_path):
    status, page = run_summary(tmp_path, [{"test_a": "failed"}] * 5)
    assert status == 1 and "failed in 5 of 5 runs" in page and "...and runs 4, 5." in page


A, B = {"test_a": "passed"}, {"test_b": "passed"}


def test_a_run_split_into_parts_counts_each_test_once_per_run_and_passes_when_every_part_passes(tmp_path):
    write_parts(tmp_path, [[A, B]] * 3)
    status, page = read_page(tmp_path, 3, 2, 10)
    assert status == 0 and "3 of 3 runs passed" in page
    assert "| `tests.test_a::test_a` | 3/3 | 0 | 0 |" in page and "| `tests.test_a::test_b` | 3/3 | 0 | 0 |" in page


def test_control_a_part_that_left_no_report_fails_its_run_naming_the_part_in_every_run(tmp_path):
    """The control that the parts are judged: the part with no report is planted in each of 4 runs, and each run fails, though
    the other part of the run passed (a summary that read only the first part would pass the runs where part 2 is gone)."""
    write_parts(tmp_path, [[A, None]] * 4)
    status, page = read_page(tmp_path, 4, 2, 10)
    assert status == 1 and "0 of 4 runs passed" in page
    assert page.count("part 2 of 2: no readable report") == 4 and "part 1 of 2" not in page
    assert "| `tests.test_a::test_b` | **0/4** | 0 | 0 |" not in page  # absent from every report, so not a row at all


def test_a_part_whose_tests_only_skipped_fails_its_run_though_the_other_part_passed(tmp_path):
    write_parts(tmp_path, [[A, {"test_b": "skipped"}]] * 2)
    status, page = read_page(tmp_path, 2, 2, 10)
    assert status == 1 and page.count("part 2 of 2: no test passed") == 2


def test_a_failure_in_the_second_part_names_the_run_seed_and_blob(tmp_path):
    write_parts(tmp_path, [[A, B], [A, {"test_b": "failed"}], [A, B]])
    status, page = read_page(tmp_path, 3, 2, 50)
    assert status == 1 and "2 of 3 runs passed (FAILED)" in page
    assert "run 2, seed `51`: part 2 of 2: 1 failed" in page and BLOB in page
    assert "| `tests.test_a::test_b` | **2/3** | 1 | 0 |" in page
