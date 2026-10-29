# A copy of MADDENING's tests/compliance/test_report_test_durations.py at its tag v0.4.2 (commit
# 8a6c9fa8066646ab57b55052a300e6badd85c658), Apache-2.0, beside the copy of the script it tests. Both are replaced
# once MADDENING ships `python -m maddening.compliance report-test-durations`.
#
# Changes to MADDENING's copy:
# 1. This header; the file sits in tests/, so REPO_ROOT is one level up, not two.
# 2. The five tests of tests/_jax_timing.py sit at the end of the file, importing it as `_jax_timing` (tests/ is not a
#    package), the registration probe run from tests/ and expecting the plugin's name "mime-jax-timing".
# 3. Left out, because they read MADDENING's sharded ci.yml (its `test` job, its `Test time budget` and
#    `Summarise the lane` steps, its per-shard cache modes) or its tests/usd/ job:
#    test_ci_runs_the_budget_on_the_default_lane,
#    test_no_shipped_allowlist_entry_names_a_test_that_needs_usd_core,
#    test_the_budget_step_can_fail_the_job,
#    test_the_budget_step_retries_an_outlier_before_failing,
#    test_the_budget_step_records_the_cache_mode_even_when_the_gate_fails,
#    test_a_lane_missing_a_shards_cache_mode_is_labelled_mixed,
#    test_a_lane_missing_a_shards_report_lists_no_allowlist_entry_as_removable,
#    test_the_lane_summary_labels_each_shard_from_its_hits,
#    and their helpers _budget_step, _lane, _render and _shim. MIME's own checks of its ci.yml take their place, at
#    the end of the file: every pytest job writes JUnit XML and runs the budget step after it, a step that can fail
#    the job and re-runs an outlier.
# 4. test_the_junit_times_the_budget_reads_include_setup_and_teardown's docstring no longer points at MADDENING's
#    tests/compliance/test_ci_sharding.py.
# 5. Added at the end: the CPU columns the script gains (MIME's change 2 to the script), and MIME's
#    tests/conftest.py recording a test's own and its children's CPU time.
# 6. test_every_shipped_allowlist_entry_is_a_node_id_pytest_collects collects without MIME_CI_PLANT_FAILURE, whose
#    planted run drops a node case and so a parameter id an allowlist entry names.
"""The CI test-time budget must fail on the slow test it exists to catch.

``scripts/report_test_durations.py`` reads pytest's JUnit XML and fails the
job on an unlisted test over the hard line.  Like every gate, it is only
evidence if it can fail: these tests plant a slow test and require a
non-zero exit, and plant an empty or unreadable report and require that it
does not pass either.

Everything runs in-process on hand-written XML -- no JAX, no subprocess --
so the file stays well inside the budget it tests.
"""

import importlib.util
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "report_test_durations.py"
ALLOWLIST = REPO_ROOT / "tests" / "duration_allowlist.txt"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_gate_report_test_durations", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(file, name, seconds, cls="", line=10, extra="", jax=None):
    """A ``<testcase>``; ``jax`` overrides the timing properties (``None``: none).

    A ``subprocesses`` key in ``jax`` is written only when given, as in a
    report from before the process count was recorded.
    """
    module = file[:-3].replace("/", ".")
    classname = f"{module}.{cls}" if cls else module
    if jax is not None:
        props = {"jax_trace_s": 0, "jax_lower_s": 0, "jax_compile_s": 0,
                 "jax_cache_read_s": 0, "jax_cache_hits": 0, "jax_cache_misses": 0, **jax}
        extra += "<properties>" + "".join(
            f'<property name="{k}" value="{v}"/>' for k, v in props.items()) + "</properties>"
    return (f'<testcase classname="{classname}" name="{name}" file="{file}" '
            f'line="{line}" time="{seconds}">{extra}</testcase>')


def _report(tmp_path, *cases, name="results.xml"):
    path = tmp_path / name
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
        + "".join(cases) + "</testsuite></testsuites>", encoding="utf-8")
    return path


def _run(gate, capsys, *argv):
    code = gate.main([*map(str, argv), "--markdown", str(argv[0]) + ".md"])
    return code, capsys.readouterr().out


def test_an_unlisted_test_over_the_hard_line_fails_the_job(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_slow", 25.0),
                     _case("tests/a/test_x.py", "test_fast", 0.2))
    code, out = _run(gate, capsys, report)
    assert code == 1
    assert "::error file=tests/a/test_x.py,line=11," in out
    assert "tests/a/test_x.py::test_slow took 25.0 s" in out


def test_the_same_test_on_the_allowlist_passes(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_slow", 25.0))
    allow = tmp_path / "allow.txt"
    allow.write_text("# header\n\ntests/a/test_x.py::test_slow # kept: the only end-to-end run\n")
    code, out = _run(gate, capsys, report, "--allowlist", allow)
    assert code == 0
    assert "::error" not in out
    assert "title=Test over" not in out     # neither the 20 s error nor the 5 s warning


@pytest.mark.parametrize("seconds, warned", [(6.0, False), (20.0, False), (25.0, True),
                                             (900.0, True)])
def test_an_allowlisted_test_over_the_hard_line_passes_but_is_not_silent(
        gate, tmp_path, capsys, seconds, warned):
    """The allowlist has no ceiling, so a kept test's regression must still show.

    A kept test that went from 6 s to 900 s used to pass with no
    annotation at all; its only trace was a row in the slowest-tests table.
    """
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_kept", seconds),
                     _case("tests/a/test_x.py", "test_other", 0.2))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/a/test_x.py::test_kept # kept: the only end-to-end check\n")
    code, out = _run(gate, capsys, report, "--allowlist", allow, "--cache-mode", "cold")
    md = Path(str(report) + ".md").read_text()
    assert code == 0 and "::error" not in out
    warnings = [ln for ln in out.splitlines() if ln.startswith("::warning")]
    if not warned:
        assert warnings == [] and "Allowlisted and over" not in md
        return
    (warning,) = warnings
    assert warning.startswith(
        "::warning file=tests/a/test_x.py,line=11,title=Allowlisted test over 20 s::"
        f"tests/a/test_x.py::test_kept took {gate._fmt(seconds)}"), warning
    assert "the allowlist has no ceiling" in warning
    assert "### Allowlisted and over 20 s" in md and "`tests/a/test_x.py::test_kept`" in md
    assert out.rstrip().endswith("0 unlisted over 20 s, 1 allowlisted over it."), out
    # The lane summary and the slow lane pass --fail-over 0: no hard line, no warning.
    Path(str(report) + ".md").unlink()
    code, out = _run(gate, capsys, report, "--allowlist", allow, "--fail-over", "0")
    assert code == 0 and "Allowlisted test over" not in out


def test_an_unlisted_test_over_the_policy_line_warns_without_failing(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_middling", 7.0, cls="TestA"))
    code, out = _run(gate, capsys, report)
    assert code == 0
    assert "::warning file=tests/a/test_x.py,line=11,title=Test over 5 s::" in out
    assert "tests/a/test_x.py::TestA::test_middling" in out


@pytest.mark.parametrize("seconds, code, warned", [(20.4, 1, False), (5.4, 0, True), (4.9, 0, False)])
def test_a_test_is_judged_on_its_fractional_seconds(gate, tmp_path, capsys, seconds, code, warned):
    """JUnit records milliseconds; truncated to whole seconds, an unlisted
    20.4 s test would pass the 20 s line and a 5.4 s one would escape the
    5 s warning."""
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_t", seconds))
    got, out = _run(gate, capsys, report)
    assert got == code, out
    assert ("title=Test over 5 s::" in out) is warned, out
    assert ("::error" in out) is (code == 1), out


def test_a_kept_test_over_the_policy_line_is_not_called_removable(gate, tmp_path, capsys):
    """An allowlisted test that passed at 8 s still needs its entry.

    Removable means it passed under the policy line (5 s) -- not under the
    hard line (20 s), which would offer every kept test between them for
    removal.  The 4 s entry beside it is listed, so the fixture can show
    the difference.
    """
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_kept", 8.0),
                     _case("tests/a/test_x.py", "test_now_fast", 4.0))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/a/test_x.py::test_kept # kept: x\n"
                     "tests/a/test_x.py::test_now_fast # kept: y\n")
    code, _ = _run(gate, capsys, report, "--allowlist", allow, "--cache-mode", "cold")
    assert code == 0
    removable = Path(str(report) + ".md").read_text().split("may be removable")[1]
    assert "`tests/a/test_x.py::test_now_fast`" in removable
    assert "test_kept" not in removable, removable


def test_fail_over_zero_reports_without_gating(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_slow", 900.0))
    code, out = _run(gate, capsys, report, "--fail-over", "0")
    assert code == 0
    assert "::error" not in out


def test_a_report_with_no_test_cases_does_not_pass(gate, tmp_path, capsys):
    code, out = _run(gate, capsys, _report(tmp_path))
    assert code == 2
    assert "no test cases" in out


def test_a_missing_report_does_not_pass(gate, tmp_path, capsys):
    code, out = _run(gate, capsys, tmp_path / "never_written.xml")
    assert code == 2
    assert "did pytest run?" in out


def test_an_unparsable_report_does_not_pass(gate, tmp_path, capsys):
    bad = tmp_path / "bad.xml"
    bad.write_text("<testsuites><testcase")
    code, out = _run(gate, capsys, bad)
    assert code == 2
    assert "not valid XML" in out


def test_an_xunit2_report_is_refused_rather_than_misread(gate, tmp_path, capsys):
    # xunit2 (pytest's default family) drops the file attribute, without
    # which neither the node id nor the annotation's location is known.
    xml = tmp_path / "x2.xml"
    xml.write_text('<testsuites><testsuite><testcase classname="tests.a.test_x" '
                   'name="test_slow" time="30"/></testsuite></testsuites>')
    code, out = _run(gate, capsys, xml)
    assert code == 2
    assert "junit_family=xunit1" in out


def test_an_allowlist_entry_without_a_reason_is_refused(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_slow", 25.0))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/a/test_x.py::test_slow\n")
    code, out = _run(gate, capsys, report, "--allowlist", allow)
    assert code == 2
    assert "has no reason" in out


def test_node_ids_are_rebuilt_exactly(gate, tmp_path):
    report = _report(
        tmp_path,
        _case("tests/core/test_x.py", "test_fn[a-b.c]", 1.0),
        _case("tests/core/test_x.py", "test_m", 1.0, cls="TestOuter.TestInner"),
        _case("tests/core/test_x.py", "test_s", 1.0, cls="TestA", extra="<skipped/>"),
    )
    got = {t.nodeid: t.outcome for t in gate.read_report(report)}
    assert got == {
        "tests/core/test_x.py::test_fn[a-b.c]": "passed",
        "tests/core/test_x.py::TestOuter::TestInner::test_m": "passed",
        "tests/core/test_x.py::TestA::test_s": "skipped",
    }


def test_two_lanes_are_judged_on_each_tests_slower_time(gate, tmp_path, capsys):
    fast = _report(tmp_path, _case("tests/a/test_x.py", "test_t", 2.0), name="a.xml")
    slow = _report(tmp_path, _case("tests/a/test_x.py", "test_t", 30.0), name="b.xml")
    assert _run(gate, capsys, fast, slow)[0] == 1
    assert _run(gate, capsys, slow, fast)[0] == 1


def test_the_summary_names_the_bands_and_the_removable_entries(gate, tmp_path, capsys):
    report = _report(tmp_path,
                     _case("tests/a/test_x.py", "test_fine", 0.5),
                     _case("tests/a/test_x.py", "test_watch", 3.0),
                     _case("tests/a/test_x.py", "test_now_fast", 0.4))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/a/test_x.py::test_now_fast # pending triage\n"
                     "tests/a/test_gone.py::test_renamed # pending triage\n")
    code, _ = _run(gate, capsys, report, "--allowlist", allow)
    md = Path(str(report) + ".md").read_text()
    assert code == 0
    assert re.search(r"\| up to 1 s: fine \| 2 \|", md)
    assert re.search(r"\| 1-5 s: watch, optimise \| 1 \|", md)
    assert "`tests/a/test_x.py::test_now_fast`" in md.split("may be removable")[1]
    assert "`tests/a/test_gone.py::test_renamed`" in md.split("may be removable")[1]


def test_an_allowlist_entry_exempts_exactly_its_node_id(gate, tmp_path, capsys):
    # Not a prefix: a base id does not exempt its parametrisations, and a
    # file does not exempt its tests.  One entry must not quietly cover
    # tests nobody decided to keep.
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_p[a]", 30.0),
                     _case("tests/a/test_x.py", "test_q", 30.0),
                     _case("tests/a/test_x.py", "test_kept[b]", 30.0))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/a/test_x.py::test_p # kept: only the base id is listed\n"
                     "tests/a/test_x.py # kept: only the file is listed\n"
                     "tests/a/test_x.py::test_kept[b] # kept: the exact id\n")
    code, out = _run(gate, capsys, report, "--allowlist", allow)
    assert code == 1
    assert "tests/a/test_x.py::test_p[a] took 30.0 s" in out
    assert "tests/a/test_x.py::test_q took 30.0 s" in out
    errors = [ln for ln in out.splitlines() if ln.startswith("::error")]
    assert not any("test_kept[b]" in ln for ln in errors), errors


def test_the_shipped_allowlist_names_real_tests_with_reasons(gate):
    # A stale entry exempts nothing, but a typo'd one would silently fail to
    # exempt the test it meant -- and a list nobody can check only grows.
    entries = gate.read_allowlist(ALLOWLIST)
    assert entries, "the allowlist parsed empty"
    for nodeid, reason in entries.items():
        # Every entry is a decision: the `pending triage` backlog the gate
        # started with is gone, and a new one would be a list nobody decided.
        assert reason.startswith("kept: "), (nodeid, reason)
        file, *_, name = nodeid.split("::")
        path = REPO_ROOT / file
        assert path.is_file(), f"{nodeid}: {file} does not exist"
        func = name.split("[", 1)[0]
        assert re.search(rf"def {re.escape(func)}\b", path.read_text(encoding="utf-8")), (
            f"{nodeid}: no `def {func}` in {file}")


def test_each_slow_test_is_diagnosed_by_where_its_time_went(gate):
    # The measured shapes of three real slow tests (2026-09-24 probe).
    loop = gate.TestTime("t.py::loop", "t.py", 1, 170.0, "passed",
                         {"jax_trace_s": 0.0, "jax_lower_s": 0.2, "jax_compile_s": 0.6,
                          "jax_cache_read_s": 0, "jax_cache_hits": 0, "jax_cache_misses": 0})
    retrace = gate.TestTime("t.py::retrace", "t.py", 1, 100.0, "passed",
                            {"jax_trace_s": 60.0, "jax_lower_s": 10.0, "jax_compile_s": 28.0,
                             "jax_cache_read_s": 0, "jax_cache_hits": 0, "jax_cache_misses": 0})
    compile_ = gate.TestTime("t.py::compile", "t.py", 1, 135.0, "passed",
                             {"jax_trace_s": 0.7, "jax_lower_s": 34.8, "jax_compile_s": 83.5,
                              "jax_cache_read_s": 0, "jax_cache_hits": 3, "jax_cache_misses": 12})
    assert gate.diagnose(loop) == "running 100%: un-jitted loop, heavy compute, or a subprocess"
    assert gate.diagnose(retrace).startswith("tracing/lowering 70%")
    assert gate.diagnose(compile_) == "compiling 62%: 12 cache misses"
    assert loop.uncacheable == pytest.approx(169.4)
    assert compile_.running == pytest.approx(135.0 - 83.5 - 35.5)
    # no timing recorded: no diagnosis, and nothing is assumed cacheable
    bare = gate.TestTime("t.py::bare", "t.py", 1, 9.0, "passed")
    assert gate.diagnose(bare) == "" and bare.uncacheable == 9.0


def test_a_test_slow_without_its_compile_is_listed_as_slow_even_when_warm(gate, tmp_path, capsys):
    report = _report(
        tmp_path,
        # 9 s, 1 s of it compiling: still 8 s with a perfect cache
        _case("tests/a/test_x.py", "test_loop", 9.0, jax={"jax_compile_s": 1.0}),
        # 9 s, 8 s compiling: a warm cache brings it to 1 s
        _case("tests/a/test_x.py", "test_compiles", 9.0, jax={"jax_compile_s": 8.0}),
    )
    code, out = _run(gate, capsys, report, "--cache-mode", "cold")
    md = Path(str(report) + ".md").read_text()
    warm = md.split("### Slow even with a warm cache (1)")[1].split("###")[0]
    assert "`tests/a/test_x.py::test_loop`" in warm
    assert "test_compiles" not in warm
    assert "**Compilation cache: cold**" in md
    assert ("running 9.0 s, tracing/lowering 0.0 s, XLA compile 9.0 s (cache reads excluded), "
            "cache reads 0.0 s") in md
    # "%" is escaped as %25 in a workflow command
    assert "(running 89%25: un-jitted loop, heavy compute, or a subprocess)" in out


def test_a_warm_run_never_calls_an_allowlist_entry_removable(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_cached", 0.4,
                                     jax={"jax_cache_hits": 5}))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/a/test_x.py::test_cached # pending triage\n")
    for mode, listed in (("warm", False), ("mixed", False), ("cold", True), ("off", True)):
        _run(gate, capsys, report, "--allowlist", allow, "--cache-mode", mode)
        md = Path(str(report) + ".md").read_text().split("## Test durations")[-1]
        assert ("may be removable" in md) is listed, mode
        assert ("not judged removable on a run that restored a cache" in md) is (not listed), mode


def test_the_junit_times_the_budget_reads_include_setup_and_teardown():
    """pytest's ``junit_duration_report`` must stay ``total`` (its default).

    ``"call"`` reports the test body alone, and a module fixture's compile
    -- charged to the setup of the first test that needs it -- would vanish
    from every time the budget judges.
    """
    import tomllib

    ini = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"]["pytest"]["ini_options"]
    assert ini.get("junit_duration_report", "total") == "total", ini["junit_duration_report"]
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "junit_duration_report" not in ci


def test_a_partial_lane_lists_no_allowlist_entry_as_removable(gate, tmp_path, capsys):
    # With one shard's report missing, an allowlisted test that is absent
    # may simply be on that shard.
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_fast", 0.2))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/b/test_y.py::test_elsewhere # pending triage\n")
    _run(gate, capsys, report, "--allowlist", allow)
    assert "may be removable" in Path(str(report) + ".md").read_text()
    Path(str(report) + ".md").unlink()
    _run(gate, capsys, report, "--allowlist", allow, "--no-removable")
    assert "may be removable" not in Path(str(report) + ".md").read_text()


def test_every_shipped_allowlist_entry_is_a_node_id_pytest_collects(gate):
    """An entry is matched exactly, so a typo in it exempts nothing, silently.

    The check above finds the file and the function; this one asks pytest
    for the real node ids, so a wrong parameter id or class name fails too.
    Slow-marked tests are collected as well: an entry for one is stale,
    which the lane summary reports, not a typo.
    """
    entries = gate.read_allowlist(ALLOWLIST)
    files = sorted({nodeid.split("::", 1)[0] for nodeid in entries})
    # Not this job's shard (which would deselect other shards' files), nor
    # this job's timing and release plugins, nor MIME's planted failure
    # (which drops a case, and so a parameter id an entry names).
    env = {k: v for k, v in os.environ.items()
           if k not in ("MADDENING_TEST_SHARD", "MADDENING_TEST_JAX_TIMING",
                         "MADDENING_TEST_RELEASE_BETWEEN_FILES", "PYTEST_ADDOPTS", "MIME_CI_PLANT_FAILURE")}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", "slow or not slow", *files],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    collected = {line.strip() for line in proc.stdout.splitlines() if "::" in line}
    missing = sorted(set(entries) - collected)
    assert not missing, f"allowlist entries pytest does not collect: {missing}"


def test_a_cache_read_is_counted_once_and_reported_on_its_own(gate, tmp_path, capsys):
    """JAX times the whole compile-or-read-the-cache call as the compile.

    ``backend_compile_duration`` wraps ``compile_or_get_cached()``, and the
    cache read (``cache_retrieval_time_sec``) is recorded inside it, so on
    a hit the "compile" *is* the read.  The shape below is a measured warm
    run of one program (jax 0.11.0, 2026-09-25): compile event 0.365 s of
    which 0.348 s was the read.
    """
    warm = {"jax_trace_s": 0.208, "jax_lower_s": 0.242, "jax_compile_s": 0.365,
            "jax_cache_read_s": 0.348, "jax_cache_hits": 3, "jax_cache_misses": 0}
    t = gate.TestTime("t.py::warm", "t.py", 1, 1.748, "passed", warm)
    assert t.compiling == pytest.approx(0.017)
    assert t.cache_read == pytest.approx(0.348)
    # Each part subtracted once: running is wall - compile event - build.
    assert t.running == pytest.approx(1.748 - 0.365 - 0.450)
    # A warm run is the run that pays the read, so it is not removable.
    assert t.uncacheable == pytest.approx(1.748 - 0.017)
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_warm", 6.0,
                                     jax={**warm, "jax_compile_s": 1.5, "jax_cache_read_s": 1.0}))
    _run(gate, capsys, report, "--cache-mode", "warm")
    md = Path(str(report) + ".md").read_text()
    assert "XLA compile 0.5 s (cache reads excluded), cache reads 1.0 s" in md
    assert "running 4.0 s" in md                   # 6.0 - 1.5 - 0.45


def test_a_test_whose_work_runs_in_a_subprocess_is_not_called_slow_even_when_warm(
        gate, tmp_path, capsys):
    """A child's compiles are invisible here, and it inherits the cache directory.

    Measured: a test whose JAX work runs in a child recorded 0 s of
    compile in the pytest process on both CI lanes, was listed as "slow
    even with a warm cache", and ran at 0.53x on the next warm run.
    """
    report = _report(
        tmp_path,
        # started a process and recorded no JAX of its own
        _case("tests/a/test_x.py", "test_child", 9.0, jax={"subprocesses": 1}),
        # started a process and compiled in-process too
        _case("tests/a/test_x.py", "test_both", 9.0,
              jax={"subprocesses": 2, "jax_compile_s": 1.0}),
        # a report from before the count: no JAX activity at all here
        _case("tests/a/test_x.py", "test_old_report", 9.0, jax={}),
        # plain Python, no process started: a cache cannot help it
        _case("tests/a/test_x.py", "test_sleeps", 9.0, jax={"subprocesses": 0}),
    )
    code, out = _run(gate, capsys, report, "--cache-mode", "cold")
    md = Path(str(report) + ".md").read_text()
    warm = md.split("### Slow even with a warm cache (1)")[1].split("###")[0]
    assert "test_sleeps" in warm
    assert not any(name in warm for name in ("test_child", "test_both", "test_old_report"))
    sub = md.split("### Work in a subprocess (not measured here) (3)")[1].split("###")[0]
    assert all(f"`tests/a/test_x.py::{name}`" in sub
               for name in ("test_child", "test_both", "test_old_report"))
    assert "| not recorded |" in sub
    assert "running 100%: work in a subprocess (not measured here)" in md
    assert "1 slow even with a warm cache; 3 with work in a subprocess" in out


def test_a_split_that_adds_up_to_more_than_the_wall_time_is_capped_and_flagged(gate):
    # Measured on jax 0.10.2 in CI: a 0.28 s test whose tracing events
    # summed to 0.33 s, printed as "tracing/lowering 118%".
    over = gate.TestTime("t.py::over", "t.py", 1, 0.28, "passed",
                         {"jax_trace_s": 0.30, "jax_lower_s": 0.03, "jax_compile_s": 0.0,
                          "jax_cache_read_s": 0, "jax_cache_hits": 0, "jax_cache_misses": 0})
    why = gate.diagnose(over)
    assert why.startswith("tracing/lowering 100%: ")
    assert "more than the wall time" in why
    assert not re.search(r"\b(10[1-9]|1[1-9][0-9]|[2-9][0-9]{2,})%", why)
    within = over._replace(seconds=1.0)
    assert "more than the wall time" not in gate.diagnose(within)


def test_a_report_holding_only_a_collection_error_does_not_pass(gate, tmp_path, capsys):
    """A file that fails to import stops pytest before any test runs.

    Every shard collects the whole suite, so one broken import leaves every
    shard's report with a single collection-error entry.  That is a run in
    which no test ran, which the gate must not pass -- and a lane summary
    built from it must not call allowlist entries removable.  The report
    here is real pytest output, from this interpreter's pytest.
    """
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_broken.py").write_text("import a_module_that_does_not_exist\n")
    (project / "tests" / "test_fine.py").write_text("def test_fine():\n    pass\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("MADDENING_TEST_SHARD", "MADDENING_TEST_JAX_TIMING",
                         "MADDENING_TEST_RELEASE_BETWEEN_FILES", "PYTEST_ADDOPTS")}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
                           "--junitxml=broken.xml", "-o", "junit_family=xunit1"],
                          cwd=project, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2, proc.stdout
    broken = project / "broken.xml"
    (case,) = gate.read_report(broken)
    assert case.collection_error and case.outcome == "error"
    code, out = _run(gate, capsys, broken)
    assert code == 2 and "collection error" in out
    # In a lane with another shard's real report: that report is judged,
    # the broken one left out, and nothing is called removable.
    fine = _report(tmp_path, _case("tests/a/test_x.py", "test_fast", 0.2), name="fine.xml")
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/b/test_y.py::test_elsewhere # pending triage\n")
    code, out = _run(gate, capsys, broken, fine, "--allowlist", allow, "--cache-mode", "cold")
    assert code == 0
    assert "::warning title=Test durations::left out" in out and "broken.xml" in out
    assert "may be removable" not in Path(str(broken) + ".md").read_text()


def _ci():
    return yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))


def _commands(script):
    joined = re.sub(r"\\\n", " ", script)
    return [ln.strip() for ln in joined.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _shell_words(command):
    """Words and shell operators, as a POSIX shell splits them."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


@pytest.mark.parametrize("tail", ["; true", ";true", " | tee gate.log", " || true", " && true",
                                  " &", " > /dev/null"])
def test_the_budget_step_check_sees_anything_chained_after_the_gate(tail):
    command = ("python scripts/report_test_durations.py test-results-shard{{matrix.shard}}.xml "
               "--allowlist tests/duration_allowlist.txt --markdown /dev/null")
    assert not [w for w in _shell_words(command) if set(w) <= set(";&|<>()")]
    assert [w for w in _shell_words(command + tail) if set(w) <= set(";&|<>()")], tail


def _again(gate, outcome="passed", seconds=1.2, ran=None):
    """A stand-in for ``run_pytest_again``: writes a re-run report holding every id it was given."""
    def run(nodeids, report, cache_dir, **_):
        if ran is not None:
            ran.append(list(nodeids))
        cases = []
        for nid in nodeids:
            file, _, name = nid.partition("::")
            extra = "<failure message='boom'/>" if outcome == "failed" else ""
            cases.append(_case(file, name, seconds, extra=extra))
        report.write_text('<testsuites><testsuite name="pytest">' + "".join(cases)
                          + "</testsuite></testsuites>", encoding="utf-8")
        return 0
    return run


@pytest.mark.parametrize("again_seconds, outcome, forgiven", [
    (1.2, "passed", True),           # the outlier of 2026-10-13: 23.9 s once, 1.1-1.6 s everywhere else
    (20.0, "passed", True),          # exactly on the line is not over it
    (20.1, "passed", False),         # over twice
    (1.2, "failed", False),          # a re-run that fails is not a pass
])
def test_a_test_over_the_hard_line_once_is_run_again_and_fails_only_if_it_is_over_twice(
        gate, tmp_path, capsys, monkeypatch, again_seconds, outcome, forgiven):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_outlier", 23.9),
                     _case("tests/a/test_x.py", "test_fast", 0.2))
    ran = []
    monkeypatch.setattr(gate, "run_pytest_again", _again(gate, outcome, again_seconds, ran))
    code, out = _run(gate, capsys, report, "--retry-outliers")
    assert ran == [["tests/a/test_x.py::test_outlier"]], ran    # only the test over the line, once
    if forgiven:
        assert code == 0 and "::error" not in out
        assert ("::warning title=Outlier forgiven::tests/a/test_x.py::test_outlier took 23.9 s here "
                f"and {gate._fmt(again_seconds)} on a re-run alone") in out, out
        assert "after a re-run (1 forgiven)" in out
    else:
        assert code == 1 and "::error file=tests/a/test_x.py,line=11," in out
        assert "Outlier forgiven" not in out


def test_a_test_whose_rerun_found_nothing_still_fails(gate, tmp_path, capsys, monkeypatch):
    # pytest could not start, or the test no longer exists under its id: no report, no pardon.
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_outlier", 23.9))
    monkeypatch.setattr(gate, "run_pytest_again", lambda nodeids, report, cache_dir, **_: 4)
    code, out = _run(gate, capsys, report, "--retry-outliers")
    assert code == 1 and "::error" in out
    monkeypatch.setattr(gate, "run_pytest_again",
                        lambda nodeids, report, cache_dir, **_: report.write_text("<testsuites/>") or 5)
    code, out = _run(gate, capsys, report, "--retry-outliers")
    assert code == 1


def test_without_the_flag_nothing_is_run_again(gate, tmp_path, capsys, monkeypatch):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_outlier", 23.9))
    monkeypatch.setattr(gate, "run_pytest_again", lambda *a, **k: pytest.fail("a re-run nobody asked for"))
    code, out = _run(gate, capsys, report)
    assert code == 1


def test_a_listed_test_and_a_fast_one_are_not_run_again(gate, tmp_path, capsys, monkeypatch):
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/a/test_x.py::test_kept # kept: the only end-to-end run\n")
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_kept", 90.0),
                     _case("tests/a/test_x.py", "test_fast", 0.2))
    monkeypatch.setattr(gate, "run_pytest_again", lambda *a, **k: pytest.fail("nothing here is an outlier"))
    code, out = _run(gate, capsys, report, "--allowlist", allow, "--retry-outliers")
    assert code == 0


def test_more_tests_over_the_line_than_the_retry_limit_are_a_regression_not_an_outlier(
        gate, tmp_path, capsys, monkeypatch):
    report = _report(tmp_path, *(_case("tests/a/test_x.py", f"test_{i}", 30.0) for i in range(6)))
    monkeypatch.setattr(gate, "run_pytest_again", lambda *a, **k: pytest.fail("six are not outliers"))
    code, out = _run(gate, capsys, report, "--retry-outliers")
    assert code == 1
    assert "6 tests are over 20 s, more than --retry-max 5: not outliers, none is re-run" in out
    # ... and five are.
    report = _report(tmp_path, *(_case("tests/a/test_x.py", f"test_{i}", 30.0) for i in range(5)),
                     name="five.xml")
    ran = []
    monkeypatch.setattr(gate, "run_pytest_again", _again(gate, ran=ran))
    code, out = _run(gate, capsys, report, "--retry-outliers")
    assert code == 0 and len(ran) == 1 and len(ran[0]) == 5


_FLAKY = """\
import os
import time
from pathlib import Path

HERE = Path(__file__).parent


def test_stalls_once():
    # The first run of this test stalls (a runner's burst); a run after it does not.
    seen = HERE / "seen"
    first = not seen.exists()
    seen.write_text("x")
    if first:
        time.sleep(1.6)


def test_always_slow():
    (HERE / "env").write_text(
        repr((os.environ.get("MADDENING_TEST_SHARD"), os.environ.get("JAX_COMPILATION_CACHE_DIR"))))
    time.sleep(1.6)
"""


def test_the_real_rerun_forgives_a_stall_that_does_not_repeat_and_fails_a_test_that_is_slow_twice(tmp_path):
    """No stand-ins: pytest writes the report, the gate runs a second pytest on the tests over the line.

    The scratch hard line is 1 s.  ``test_stalls_once`` takes 1.6 s the first
    time it runs and nothing after; ``test_always_slow`` takes 1.6 s every
    time.  The re-run must also name no shard and start with a compilation
    cache directory that is new, not the one the first run wrote.
    """
    work = tmp_path / "work"
    work.mkdir()
    (work / "test_flaky.py").write_text(_FLAKY)
    first_cache = tmp_path / "first-cache"
    first_cache.mkdir()
    env = {**os.environ, "MADDENING_TEST_SHARD": "9/9", "JAX_COMPILATION_CACHE_DIR": str(first_cache),
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    # This scratch directory has no root conftest, so the shard variable is inert in it.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--junitxml=results.xml",
         "-o", "junit_family=xunit1", "test_flaky.py"],
        cwd=work, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert eval((work / "env").read_text()) == ("9/9", str(first_cache))
    gate_proc = subprocess.run(
        [sys.executable, str(SCRIPT), "results.xml", "--markdown", "summary.md", "--watch-over", "0.1",
         "--slow-over", "0.5", "--fail-over", "1", "--retry-outliers"],
        cwd=work, env=env, capture_output=True, text=True, timeout=300)
    out = gate_proc.stdout
    assert gate_proc.returncode == 1, out + gate_proc.stderr
    errors = [ln for ln in out.splitlines() if ln.startswith("::error")]
    assert len(errors) == 1 and "test_flaky.py::test_always_slow took" in errors[0], errors
    assert "::warning title=Outlier forgiven::test_flaky.py::test_stalls_once took 1.6 s here and" in out, out
    # The re-run's environment, written by the test that was slow twice: no shard, a new cache directory.
    shard, cache = eval((work / "env").read_text())
    assert shard is None
    assert cache != str(first_cache) and "retry-outliers-" in cache, cache


def _lookups(hits, misses, n_tests=4):
    """``n_tests`` cases in one file sharing ``hits`` / ``misses``."""
    per = [(hits // n_tests + (i < hits % n_tests), misses // n_tests + (i < misses % n_tests))
           for i in range(n_tests)]
    return [_case("tests/a/test_x.py", f"test_{i}", 0.1,
                  jax={"jax_cache_hits": h, "jax_cache_misses": m}) for i, (h, m) in enumerate(per)]


def _shard(tmp_path, n, hits, misses, mode):
    report = _report(tmp_path, *_lookups(hits, misses), name=f"test-results-shard{n}.xml")
    if mode is not None:
        (tmp_path / f"cache-mode-shard{n}.txt").write_text(mode + "\n")
    return report


def test_a_restored_cache_that_most_lookups_missed_is_not_called_warm(gate, tmp_path, capsys):
    """The label follows the hits, not the restore.

    On CI about one PR shard in four restored a cache and then hit 17-20% of
    its lookups -- a cold run's rate, from programs it compiled itself --
    because JAX's key includes the host's CPU features and runners with one
    CPU model name can differ in them.  Those shards used to be labelled
    warm, with "these times are lower than a cold run's".
    """
    reports = [_shard(tmp_path, 1, 3856, 521, "warm"), _shard(tmp_path, 2, 625, 2916, "warm")]
    code, out = _run(gate, capsys, *reports, "--cache-mode", "warm")
    md = Path(str(reports[0]) + ".md").read_text()
    assert code == 0
    assert "**Compilation cache: mixed**" in md and "**Compilation cache: warm**" not in md
    assert "shard 1 warm, 3856 of 4377 lookups hit (88%)" in md
    assert "shard 2 restored but unused (cold), 625 of 3541 lookups hit (18%)" in md
    assert "compilation cache: shard 1 warm" in out            # the step log says it too


def test_a_lane_whose_restored_caches_all_went_unused_says_so(gate, tmp_path, capsys):
    reports = [_shard(tmp_path, n, 600, 2900, "warm") for n in (1, 2)]
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/b/test_y.py::test_elsewhere # kept: x\n")
    _run(gate, capsys, *reports, "--cache-mode", "warm", "--allowlist", allow)
    md = Path(str(reports[0]) + ".md").read_text()
    assert "**Compilation cache: restored but unused (cold)**" in md
    # A shard that restored a cache still lists nothing as removable: a low
    # hit rate can also be a change that touched most programs.
    assert "may be removable" not in md
    assert "not judged removable on a run that restored a cache" in md


@pytest.mark.parametrize("hits, label", [(51, "warm"), (50, "restored but unused (cold)"),
                                         (0, "restored but unused (cold)")])
def test_warm_needs_more_than_half_of_the_lookups_to_hit(gate, tmp_path, capsys, hits, label):
    report = _shard(tmp_path, 1, hits, 100 - hits, "warm")
    _run(gate, capsys, report)
    md = Path(str(report) + ".md").read_text()
    assert f"shard 1 {label}, {hits} of 100 lookups hit" in md


def test_a_warm_run_that_recorded_no_lookups_keeps_its_claim(gate, tmp_path, capsys):
    # Without the timing plugin there is nothing to judge the claim by.
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_t", 0.1),
                     name="test-results-shard1.xml")
    (tmp_path / "cache-mode-shard1.txt").write_text("warm\n")
    _run(gate, capsys, report)
    assert "**Compilation cache: warm**" in Path(str(report) + ".md").read_text()


def test_each_shards_mode_file_decides_its_label(gate, tmp_path, capsys):
    # --cache-mode is the workflow's summary of the same files; a file beside
    # a report is that shard's own record, and wins.
    cold = _shard(tmp_path, 1, 20, 80, "cold")
    warm = _shard(tmp_path, 2, 99, 1, "warm")
    _run(gate, capsys, cold, warm, "--cache-mode", "warm")
    md = Path(str(cold) + ".md").read_text()
    assert "shard 1 cold, 20 of 100" in md and "shard 2 warm, 99 of 100" in md
    assert "**Compilation cache: mixed**" in md


@pytest.mark.parametrize("content", ["", "warmish", "WARM"])
def test_an_unreadable_mode_file_is_unknown_not_guessed(gate, tmp_path, capsys, content):
    report = _shard(tmp_path, 1, 99, 1, None)
    (tmp_path / "cache-mode-shard1.txt").write_text(content)
    other = _shard(tmp_path, 2, 99, 1, "warm")
    _run(gate, capsys, report, other, "--cache-mode", "warm")
    md = Path(str(report) + ".md").read_text()
    assert "shard 1 unknown" in md and "**Compilation cache: mixed**" in md


def test_a_cold_run_is_not_said_to_pay_every_compile_in_full(gate, tmp_path, capsys):
    # A cold CI shard reads back 17-25% of its lookups from programs it
    # compiled earlier in the same run.
    report = _shard(tmp_path, 1, 20, 80, "cold")
    _run(gate, capsys, report)
    md = Path(str(report) + ".md").read_text()
    assert "**Compilation cache: cold**" in md
    assert "Every compile is paid in full" not in md
    assert "reads it back from the cache the run is writing" in md


def _off_report(tmp_path, files):
    cases = [_case("tests/a/test_first.py", "test_before", 0.1, jax={})]
    cases += [_case(f, f"test_{i}", 0.1, jax={"jax_cache_hits": 1, "jax_cache_misses": 2})
              for i, f in enumerate(files)]
    return _report(tmp_path, *cases, name="test-results-shard4.xml")


def test_an_off_run_whose_later_tests_hit_a_cache_is_flagged(gate, tmp_path, capsys):
    """The slow lane claims no cache; a test that leaves one on makes that false.

    ``tests/core/test_compile_cache.py`` switched a persistent cache on and
    never off, and 858 of one slow shard's 1503 tests then ran with it,
    under a summary that said "Compilation cache: off".
    """
    report = _off_report(tmp_path, ["tests/core/test_leaks.py", "tests/nodes/test_after.py",
                                    "tests/nodes/test_after.py"])
    _, out = _run(gate, capsys, report, "--cache-mode", "off", "--fail-over", "0")
    md = Path(str(report) + ".md").read_text()
    warning = [ln for ln in out.splitlines() if ln.startswith("::warning title=Compilation cache::")]
    assert len(warning) == 1, out
    assert "3 tests recorded persistent-cache lookups (hits 3, misses 6)" in warning[0]
    assert "from tests/core/test_leaks.py::test_0 on" in warning[0]
    assert "cache claimed off, but 3 tests recorded persistent-cache lookups" in md
    assert "left it on" in md


def test_an_off_run_with_lookups_in_one_file_is_noted_without_a_warning(gate, tmp_path, capsys):
    # A file whose tests configure a cache of their own records lookups; a
    # cache it left on would show in the files after it.
    report = _off_report(tmp_path, ["tests/core/test_own_cache.py"] * 2)
    _, out = _run(gate, capsys, report, "--cache-mode", "off", "--fail-over", "0")
    md = Path(str(report) + ".md").read_text()
    assert "::warning title=Compilation cache::" not in out
    assert "cache claimed off, but 2 tests recorded" in md and "All in one file" in md


def test_an_off_run_without_lookups_is_not_flagged(gate, tmp_path, capsys):
    report = _off_report(tmp_path, [])
    _, out = _run(gate, capsys, report, "--cache-mode", "off", "--fail-over", "0")
    assert "Compilation cache::" not in out
    assert "cache claimed off, but" not in Path(str(report) + ".md").read_text()


def test_a_skipped_or_failed_allowlisted_test_is_not_called_removable(gate, tmp_path, capsys):
    # Fast because it did not run, or stopped early: that says nothing about
    # what it costs when it passes.
    report = _report(tmp_path,
                     _case("tests/a/test_x.py", "test_passed_fast", 0.2),
                     _case("tests/a/test_x.py", "test_skipped", 0.0,
                           extra='<skipped type="pytest.skip" message="no tool"/>'),
                     _case("tests/a/test_x.py", "test_failed", 0.3,
                           extra='<failure message="assert 0">assert 0</failure>'),
                     _case("tests/a/test_x.py", "test_errored", 0.1,
                           extra='<error message="fixture failed">boom</error>'))
    allow = tmp_path / "allow.txt"
    allow.write_text("".join(f"tests/a/test_x.py::{n} # kept: x\n"
                             for n in ("test_passed_fast", "test_skipped", "test_failed",
                                       "test_errored")))
    _run(gate, capsys, report, "--allowlist", allow, "--cache-mode", "cold")
    removable = Path(str(report) + ".md").read_text().split("may be removable")[1]
    assert "`tests/a/test_x.py::test_passed_fast`" in removable
    for name in ("test_skipped", "test_failed", "test_errored"):
        assert name not in removable, name



# ---------------------------------------------------------------------------
# MIME's CI, in place of MADDENING's sharded workflow (change 3 above)
# ---------------------------------------------------------------------------

#: The gate as every lane's `Test durations` step runs it, the step's last command.
GATE = ["uv", "run", "--no-sync", "python", "scripts/report_test_durations.py", "$RUNNER_TEMP/durations.xml"]
JUNIT = "--junitxml=$RUNNER_TEMP/durations.xml"
AFTER_THE_TESTS = "${{ !cancelled() && steps.tests.outcome != 'skipped' }}"


def budget_problems(ci):
    """Every way a job of ``ci`` that runs pytest does not feed the budget its JUnit XML, or does not let the budget
    fail the job: an empty list when every one does."""
    problems = []
    lanes = 0
    for name, job in ci["jobs"].items():
        steps = job.get("steps", [])
        for index, step in enumerate(steps):
            words = _shell_words(" ".join(_commands(step.get("run", ""))))
            if "pytest" not in words:
                continue
            lanes += 1
            if step.get("id") != "tests":
                problems.append(f"{name}: the pytest step has no id `tests`, which the budget step's `if` reads")
            if JUNIT not in words:
                problems.append(f"{name}: pytest writes no JUnit XML to $RUNNER_TEMP/durations.xml")
            budget = steps[index + 1] if index + 1 < len(steps) else {}
            if budget.get("name") != "Test durations":
                problems.append(f"{name}: no `Test durations` step right after its tests")
                continue
            if budget.get("if") != AFTER_THE_TESTS:
                problems.append(f"{name}: the budget step runs {budget.get('if')!r}, not after a red test step too")
            for key in ("continue-on-error", "shell"):
                if key in budget:
                    problems.append(f"{name}: the budget step sets {key}, which can keep its exit code from the job")
            script = re.sub(r"\$\{\{.*?\}\}", "EXPR", budget.get("run", ""))
            for undo in ("--fail-over", "--markdown", "||", "set +e"):
                if undo in script:
                    problems.append(f"{name}: the budget step contains {undo!r}")
            last = _shell_words(_commands(budget.get("run", ""))[-1]) if _commands(budget.get("run", "")) else []
            operators = [w for w in last if w and set(w) <= set(";&|<>()")]
            if operators or last[:len(GATE)] != GATE:
                problems.append(f"{name}: the gate on the job's report is not the whole last command: {last}")
                continue
            allowlist = last[last.index("--allowlist") + 1] if "--allowlist" in last[:-1] else None
            if allowlist != "tests/duration_allowlist.txt":
                problems.append(f"{name}: the gate reads no tests/duration_allowlist.txt")
            if "--retry-outliers" not in last:
                problems.append(f"{name}: the gate fails on one runner's outlier without re-running it")
    if "continue-on-error" in ci["jobs"].get("default", {}):
        problems.append("default: the blocking lane has continue-on-error, so its budget cannot fail the run")
    if lanes < 4:
        problems.append(f"{lanes} jobs run pytest; the four lanes (default, openmaterica-main, maddening-main, "
                        f"usd-core-newest) each should")
    return problems


def test_every_lane_writes_junit_xml_and_its_budget_step_can_fail_the_job():
    assert budget_problems(_ci()) == []


def _lane_step(ci, job, name):
    return next(s for s in ci["jobs"][job]["steps"] if s.get("name") == name)


@pytest.mark.parametrize("mutation", [
    "no junit", "gate swallowed", "hard line off", "no retry", "no allowlist", "no budget step",
    "step continues on error", "only after green tests", "default lane continues on error",
])
def test_the_lane_check_fails_on_each_way_a_lane_can_stop_feeding_or_obeying_the_budget(mutation):
    ci = _ci()
    tests, budget = _lane_step(ci, "maddening-main", "Tests"), _lane_step(ci, "maddening-main", "Test durations")
    if mutation == "no junit":
        tests["run"] = tests["run"].replace(' --junitxml="$RUNNER_TEMP/durations.xml"', "")
    elif mutation == "gate swallowed":
        budget["run"] += " || true"
    elif mutation == "hard line off":
        budget["run"] += " --fail-over 0"
    elif mutation == "no retry":
        budget["run"] = budget["run"].replace(" --retry-outliers", "")
    elif mutation == "no allowlist":
        budget["run"] = budget["run"].replace(" --allowlist tests/duration_allowlist.txt", "")
    elif mutation == "no budget step":
        ci["jobs"]["maddening-main"]["steps"].remove(budget)
    elif mutation == "step continues on error":
        budget["continue-on-error"] = True
    elif mutation == "only after green tests":
        budget["if"] = "${{ success() }}"
    else:
        ci["jobs"]["default"]["continue-on-error"] = True
    problems = budget_problems(ci)
    assert problems, mutation
    assert all(p.startswith(("maddening-main:", "default:")) for p in problems), problems


def test_pytest_writes_the_junit_family_that_records_file_and_line():
    import tomllib

    ini = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    assert ini["junit_family"] == "xunit1"


# ---------------------------------------------------------------------------
# CPU time: the script's columns (MIME's change 2) and tests/conftest.py
# ---------------------------------------------------------------------------


def _cpu(own=None, children=None):
    props = [(k, v) for k, v in (("cpu_seconds", own), ("children_cpu_seconds", children)) if v is not None]
    return "<properties>" + "".join(f'<property name="{k}" value="{v}"/>' for k, v in props) + "</properties>"


def test_the_cpu_times_are_read_and_summed_beside_the_wall_time(gate, tmp_path, capsys):
    report = _report(tmp_path,
                     _case("tests/a/test_x.py", "test_burns", 9.0, extra=_cpu(8.5, 0.0)),
                     _case("tests/a/test_x.py", "test_waits", 2.0, extra=_cpu(0.1, 0.0)),
                     _case("tests/b/test_y.py", "test_spawns", 3.0, extra=_cpu(0.2, 2.6)))
    times = {t.nodeid: (t.cpu, t.children_cpu) for t in gate.read_report(report)}
    assert times["tests/a/test_x.py::test_burns"] == (8.5, 0.0)
    assert times["tests/b/test_y.py::test_spawns"] == (0.2, 2.6)
    code, _ = _run(gate, capsys, report)
    md = Path(str(report) + ".md").read_text()
    assert code == 0
    assert ("CPU time over the same spans: 8.8 s in the test processes (each test's own, its xdist worker's) and "
            "2.6 s in the processes they started and waited for.") in md
    assert "| band | tests | time | CPU | children's CPU | share |" in md
    assert "| over 5 s: mark slow, speed up, or keep with a reason | 1 | 9.0 s | 8.5 s | 0.0 s | 64% |" in md
    assert "| 3.0 s | 0.2 s | 2.6 s | `tests/b/test_y.py::test_spawns` |  |" in md
    assert "| 11.0 s | 8.6 s | 0.0 s | 2 | 5.50 s | `tests/a/test_x.py` |" in md
    assert "recorded no CPU time" not in md


def test_a_report_without_cpu_times_reads_as_maddenings(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_t", 0.2))
    _run(gate, capsys, report)
    md = Path(str(report) + ".md").read_text()
    assert "| band | tests | time | share |" in md and "CPU" not in md


def test_a_test_that_recorded_no_cpu_time_is_counted(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_t", 0.2, extra=_cpu(0.1, 0.0)),
                     _case("tests/a/test_x.py", "test_u", 0.2))
    _run(gate, capsys, report)
    assert "Tests that recorded no CPU time: 1." in Path(str(report) + ".md").read_text()


def test_an_unreadable_cpu_time_does_not_pass(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_t", 0.2, extra=_cpu("soon", 0.0)))
    code, out = _run(gate, capsys, report)
    assert code == 2 and "unreadable CPU time property" in out


_SPENDS = """\
import subprocess
import sys
import time


def _burn(seconds):
    end = time.process_time() + seconds
    while time.process_time() < end:
        pass


def test_burns():
    _burn(0.4)


def test_waits():
    time.sleep(0.4)


def test_spawns():
    subprocess.run([sys.executable, "-c", "import time\\nend = time.process_time() + 0.4\\n"
                    "while time.process_time() < end: pass"], check=True)
"""


def test_the_conftest_records_each_tests_own_and_its_childrens_cpu_time(gate, tmp_path):
    """MIME's ``tests/conftest.py``, in a scratch project, under a real pytest: a test that burns 0.4 s of CPU, one
    that sleeps 0.4 s (the control that CPU time is not wall time), and one whose child burns it."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "conftest.py").write_bytes((REPO_ROOT / "tests" / "conftest.py").read_bytes())
    (project / "test_spends.py").write_text(_SPENDS)
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml=r.xml",
                           "-o", "junit_family=xunit1", "test_spends.py"],
                          cwd=project, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    times = {t.nodeid.split("::")[-1]: t for t in gate.read_report(project / "r.xml")}
    burns, waits, spawns = times["test_burns"], times["test_waits"], times["test_spawns"]
    assert burns.cpu >= 0.35 and burns.children_cpu < 0.05
    assert waits.seconds >= 0.35 and waits.cpu < 0.1 and waits.children_cpu < 0.05
    assert spawns.children_cpu >= 0.35 and spawns.cpu < 0.2


# The JAX timing plugin, tests/_jax_timing.py (change 2 above).


def test_the_timing_plugin_attaches_its_properties_before_the_teardown_report():
    from types import SimpleNamespace
    import _jax_timing as jt

    timing = jt.JaxTiming()
    plugin = jt.Plugin(timing)
    item = SimpleNamespace(user_properties=[])
    plugin.pytest_runtest_setup(item)
    timing.on_duration("/jax/core/compile/backend_compile_duration", 2.5)
    timing.on_duration("/jax/core/compile/backend_compile_duration", 0.5)
    timing.on_duration("/jax/some/other_duration", 99.0)   # ignored
    timing.on_event("/jax/compilation_cache/cache_hits")
    for when in ("setup", "call"):
        plugin.pytest_runtest_makereport(item, SimpleNamespace(when=when))
    assert item.user_properties == []
    plugin.pytest_runtest_makereport(item, SimpleNamespace(when="teardown"))
    props = dict(item.user_properties)
    assert set(props) == set(jt.PROPERTIES)
    assert props["jax_compile_s"] == 3.0 and props["jax_cache_hits"] == 1
    assert props["jax_trace_s"] == 0
    # the next test starts from zero
    plugin.pytest_runtest_setup(item)
    assert timing.properties() == [(p, 0) for p in jt.PROPERTIES]


def test_jax_still_emits_the_events_the_timing_plugin_listens_for():
    # If JAX renames an event, the plugin would report zero for it and every
    # slow test would read as "running".  Check the names against the
    # installed JAX: the compile-pipeline durations by listening to a real
    # compile, the cache events by their presence in JAX's compiler module
    # (they only fire with a cache configured).
    import inspect
    import jax
    import jax.numpy as jnp
    from jax import monitoring
    from jax._src import compiler, compilation_cache
    import _jax_timing as jt

    seen, listening = set(), [True]

    def listener(event, secs, **_):
        if listening[0]:
            seen.add(event)

    # Switched off rather than unregistered: the unregister API differs
    # across the JAX versions CI runs, and a listener that ignores every
    # event costs nothing.
    monitoring.register_event_duration_secs_listener(listener)
    try:
        # A constant no other test uses, so this is a fresh trace and compile.
        jax.jit(lambda x: x * 1.2345678901 + 0.987654321)(jnp.ones(3)).block_until_ready()
    finally:
        listening[0] = False
    pipeline = [e for e in jt.DURATION_EVENTS if e.startswith("/jax/core/compile/")]
    assert pipeline and set(pipeline) <= seen, set(pipeline) - seen
    source = inspect.getsource(compiler) + inspect.getsource(compilation_cache)
    cache_events = [e for e in [*jt.DURATION_EVENTS, *jt.COUNT_EVENTS]
                    if e.startswith("/jax/compilation_cache/")]
    assert len(cache_events) == 3
    for event in cache_events:
        assert event in source, f"JAX no longer emits {event!r}"


def test_timing_plugin_counts_the_processes_a_test_starts():
    from types import SimpleNamespace
    import _jax_timing as jt

    timing = jt.JaxTiming()
    plugin = jt.Plugin(timing)
    item = SimpleNamespace(user_properties=[])
    plugin.pytest_runtest_setup(item)
    timing.on_audit("subprocess.Popen", ("python", ["python", "-c", ""], None, None))
    timing.on_audit("os.fork", ())
    timing.on_audit("open", ("f", "r", 0))            # not a process
    plugin.pytest_runtest_makereport(item, SimpleNamespace(when="teardown"))
    assert dict(item.user_properties)["subprocesses"] == 2


def test_registering_the_timing_plugin_starts_counting_processes():
    # What conftest calls when MIME_TEST_JAX_TIMING=1.  In a child:
    # neither the audit hook nor JAX's listeners can be removed once added.
    # JAX's listener API is stubbed (its events are checked against the real
    # JAX above), which keeps the child to a fraction of a second.
    probe = (
        "import subprocess, sys\n"
        "from types import ModuleType, SimpleNamespace\n"
        "jax = ModuleType('jax')\n"
        "jax.monitoring = SimpleNamespace(register_event_duration_secs_listener=lambda f: None,\n"
        "                                 register_event_listener=lambda f: None)\n"
        "sys.modules['jax'] = jax\n"
        "import _jax_timing as jt\n"
        "names = []\n"
        "config = SimpleNamespace(pluginmanager=SimpleNamespace(\n"
        "    register=lambda plugin, name: names.append(name)))\n"
        "timing = jt.register(config)\n"
        "subprocess.run([sys.executable, '-c', ''], check=True)\n"
        "print(timing.totals['subprocesses'], names)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONSAFEPATH"}
    proc = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT / "tests", env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.split(None, 1) == ["1", "['mime-jax-timing']\n"], proc.stdout


def test_python_still_raises_the_audit_events_the_process_count_listens_for():
    # If Python renamed one, the count would read zero and a subprocess
    # test would be called uncacheable again.  Checked in a child, since an
    # audit hook cannot be removed once added.
    import _jax_timing as jt

    probe = (
        "import os, subprocess, sys\n"
        "seen = set()\n"
        "sys.addaudithook(lambda e, a: seen.add(e))\n"
        "subprocess.run([sys.executable, '-c', ''], check=True)\n"
        "os.system('true')\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os._exit(0)\n"
        "os.waitpid(pid, 0)\n"
        "os.waitpid(os.posix_spawn('/bin/true', ['true'], dict(os.environ)), 0)\n"
        "print(' '.join(sorted(e for e in seen if e.startswith(('os.', 'subprocess.')))))\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    seen = set(proc.stdout.split())
    # (os.spawn* forks and execs on POSIX, raising os.fork; its own event is Windows-only.)
    for event in ("subprocess.Popen", "os.system", "os.fork", "os.posix_spawn"):
        assert event in seen, f"Python no longer raises {event!r}"
        assert event in jt.SPAWN_EVENTS
