"""The duration rule of 2026-11-09 (CONTRIBUTING.md, "CI: the duration gate"), branch by branch, and the weekly triage.

An unlisted test over 30 s fails a push whatever the diff, and fails a pull request only if it is the change's own (its
file is in the diff, or its node id is new on the base); on a pull request any other unlisted test fails only over a flat
90 s. A listed test has a ceiling from ``tests/duration_ceilings.json``, which fails a push and warns a pull request
unless the pull request owns the test. Every failure is re-run alone first.

Each branch is a test with a control beside it that must fail the other way: a rule that passes everything, or fails
everything, would pass half of every pair. Everything runs in-process on hand-written XML, except the one test that asks
git and pytest what a base collects, and the lane job's timings artifact, made by running its own steps
(``scripts/latest_attempt.py`` and the gather step's script in bash) over hand-written shard artifacts.
"""

import datetime
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_rule_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load("report_test_durations")


@pytest.fixture(scope="module")
def triage():
    return _load("duration_triage")


def _case(file, name, seconds):
    module = file[:-3].replace("/", ".")
    return (f'<testcase classname="{module}" name="{name}" file="{file}" line="10" time="{seconds}"></testcase>')


def _report(tmp_path, *cases, name="results.xml"):
    path = tmp_path / name
    path.write_text('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
                    + "".join(cases) + "</testsuite></testsuites>", encoding="utf-8")
    return path


def _ceilings(tmp_path, table):
    path = tmp_path / "ceilings.json"
    path.write_text(json.dumps({"ceilings": table}), encoding="utf-8")
    return path


def _run(gate, capsys, *argv):
    argv = [str(a) for a in argv]
    code = gate.main([*argv, "--markdown", argv[0] + ".md"])
    return code, capsys.readouterr().out


def _own(gate, monkeypatch, changed, base=None):
    """The scope of a change whose diff names ``changed`` and whose base collects ``base`` (file -> ids)."""
    base = base or {}
    monkeypatch.setattr(gate, "scope_against", lambda rev: gate.Scope(set(changed), lambda f: base.get(f, set())))


FILE = "tests/a/test_x.py"
NODE = f"{FILE}::test_slow"
PUSH, PULL = ["--event", "push"], ["--event", "pull_request"]


# -- an unlisted test, own or not ------------------------------------------------------------------------------------

@pytest.mark.parametrize("seconds, code", [(30.0, 0), (30.1, 1)])
def test_a_test_in_a_file_the_diff_touches_fails_over_the_30_s_line_and_not_at_it(
        gate, tmp_path, capsys, monkeypatch, seconds, code):
    report = _report(tmp_path, _case(FILE, "test_slow", seconds))
    _own(gate, monkeypatch, [FILE], {FILE: {NODE}})            # the id is old: only the file makes it the change's own
    got, out = _run(gate, capsys, report, "--own-against", "", *PULL)
    assert got == code, out
    assert ("title=Test over 30 s::" in out) is bool(code)


def test_control_the_same_slow_test_in_an_untouched_file_with_an_old_id_only_warns_on_a_pull_request(
        gate, tmp_path, capsys, monkeypatch):
    report = _report(tmp_path, _case(FILE, "test_slow", 35.0))
    _own(gate, monkeypatch, ["tests/other/test_y.py"], {FILE: {NODE}})
    code, out = _run(gate, capsys, report, "--own-against", "", *PULL)
    assert code == 0 and "::error" not in out
    assert "title=Test over 30 s, not this change's::" in out
    assert "not this change's, under the flat line -- warns" in Path(str(report) + ".md").read_text()
    # ... and without --own-against every test is the change's own, which fails the same report.
    assert _run(gate, capsys, report, *PULL)[0] == 1


@pytest.mark.parametrize("event", [["--event", "push"], ["--event", "workflow_dispatch"], ["--event", "schedule"]])
def test_on_a_push_or_a_manual_run_an_unlisted_test_over_30_s_fails_whatever_the_diff(
        gate, tmp_path, capsys, monkeypatch, event):
    """A slow test that a change caused without touching it (an un-jitted loop in ``src``): the push is the merged change."""
    report = _report(tmp_path, _case(FILE, "test_slow", 35.0))
    monkeypatch.setattr(gate, "scope_against", lambda rev: pytest.fail("a push reads no diff"))
    code, out = _run(gate, capsys, report, "--own-against", "HEAD^1", *event)
    assert code == 1 and "title=Test over 30 s::" in out
    # The control: the same report, the same flag and an untouched file on a pull request only warns ...
    _own(gate, monkeypatch, ["src/mime/x.py"], {FILE: {NODE}})
    code, out = _run(gate, capsys, report, "--own-against", "HEAD^1", *PULL)
    assert code == 0 and "::error" not in out
    # ... and a push still passes a test at the line, so the push rule is 30 s and not zero.
    ok = _report(tmp_path, _case(FILE, "test_slow", 30.0), name="ok.xml")
    monkeypatch.setattr(gate, "scope_against", lambda rev: pytest.fail("a push reads no diff"))
    assert _run(gate, capsys, ok, "--own-against", "HEAD^1", *event)[0] == 0


def test_a_push_forgives_a_test_over_30_s_that_a_re_run_alone_clears(gate, tmp_path, capsys, monkeypatch):
    report = _report(tmp_path, _case(FILE, "test_slow", 35.0))
    for again, code in ((12.0, 0), (31.0, 1)):
        monkeypatch.setattr(gate, "run_pytest_again", lambda nodeids, rep, cache, s=again, **_: rep.write_text(
            '<testsuites><testsuite name="pytest">' + _case(FILE, "test_slow", s) + "</testsuite></testsuites>") or 0)
        assert _run(gate, capsys, report, "--retry-outliers", *PUSH)[0] == code


def test_a_new_node_id_in_a_file_the_diff_does_not_touch_is_the_changes_own(gate, tmp_path, capsys, monkeypatch):
    """The case a registry entry adds: the test file is never opened, the id is new (#186's `rigid_inertial` case)."""
    new = f"{FILE}::test_slow[dynamics rigid_inertial 0.4.0]"
    report = _report(tmp_path, _case(FILE, "test_slow[dynamics rigid_inertial 0.4.0]", 34.0))
    _own(gate, monkeypatch, ["src/mime/nodes/body/_rigid_inertial.py"], {FILE: {f"{FILE}::test_slow[dynamics toy.body 0.1.0]"}})
    code, out = _run(gate, capsys, report, "--own-against", "", *PULL)
    assert code == 1 and new in out
    # The control: the base collects that id too, so it is not new, and the same time only warns.
    _own(gate, monkeypatch, ["src/mime/nodes/body/_rigid_inertial.py"], {FILE: {new}})
    code, out = _run(gate, capsys, report, "--own-against", "", *PULL)
    assert code == 0 and "::error" not in out


@pytest.mark.parametrize("seconds, code", [(90.0, 0), (90.1, 1)])
def test_a_test_that_is_not_the_changes_fails_a_pull_request_only_over_the_flat_90_s(
        gate, tmp_path, capsys, monkeypatch, seconds, code):
    report = _report(tmp_path, _case(FILE, "test_slow", seconds))
    _own(gate, monkeypatch, ["tests/other/test_y.py"], {FILE: {NODE}})
    got, out = _run(gate, capsys, report, "--own-against", "", *PULL)
    assert got == code, out
    assert ("title=Test over 90 s::" in out) is bool(code)


def test_a_base_that_cannot_be_read_makes_every_test_the_changes_own(gate, tmp_path, capsys, monkeypatch):
    report = _report(tmp_path, _case(FILE, "test_slow", 35.0))
    monkeypatch.setattr(gate, "scope_against", lambda rev: gate.Scope(None))
    code, out = _run(gate, capsys, report, "--own-against", "", *PULL)
    assert code == 1 and "could not be read, so every test counts as this change's own" in out
    # A diff that was read but whose base could not be collected is strict for the test it cannot place, not for all.
    monkeypatch.setattr(gate, "scope_against", lambda rev: gate.Scope({"x.py"}, lambda f: None))
    assert _run(gate, capsys, report, "--own-against", "", *PULL)[0] == 1
    monkeypatch.setattr(gate, "scope_against", lambda rev: gate.Scope({"x.py"}, lambda f: {NODE}))
    assert _run(gate, capsys, report, "--own-against", "", *PULL)[0] == 0


def test_the_base_is_collected_only_for_a_test_that_is_over_the_line(gate, tmp_path, capsys, monkeypatch):
    report = _report(tmp_path, _case(FILE, "test_fast", 3.0), _case(FILE, "test_middling", 29.0))
    monkeypatch.setattr(gate, "scope_against", lambda rev: gate.Scope(
        {"x.py"}, lambda f: pytest.fail("a test under the line has no need of the base")))
    assert _run(gate, capsys, report, "--own-against", "", *PULL)[0] == 0


def test_every_failure_is_run_again_alone_against_the_line_that_applied_to_it(gate, tmp_path, capsys, monkeypatch):
    mine, other = f"{FILE}::test_mine", f"tests/b/test_z.py::test_other"
    report = _report(tmp_path, _case(FILE, "test_mine", 35.0), _case("tests/b/test_z.py", "test_other", 95.0))
    _own(gate, monkeypatch, [FILE], {"tests/b/test_z.py": {other}})
    ran = []

    def again(seconds):
        def run(nodeids, report, cache_dir, **_):
            ran.append(list(nodeids))
            cases = [_case(n.split("::")[0], n.split("::")[1], seconds[n]) for n in nodeids]
            report.write_text('<testsuites><testsuite name="pytest">' + "".join(cases) + "</testsuite></testsuites>",
                              encoding="utf-8")
            return 0
        return run

    # Mine is forgiven by a 29 s re-run (under 30) and the other by an 89 s one (under 90).
    monkeypatch.setattr(gate, "run_pytest_again", again({mine: 29.0, other: 89.0}))
    code, out = _run(gate, capsys, report, "--own-against", "", "--retry-outliers", *PULL)
    assert code == 0 and sorted(ran[0]) == sorted([mine, other]), (code, out)
    # The control: 31 s is over mine's line of 30, though under the other's flat 90, so mine alone stays red.
    monkeypatch.setattr(gate, "run_pytest_again", again({mine: 31.0, other: 89.0}))
    code, out = _run(gate, capsys, report, "--own-against", "", "--retry-outliers", *PULL)
    assert code == 1 and f"{mine} took" in out and "::error" in out
    assert out.count("::error") == 1 and other not in "".join(ln for ln in out.splitlines() if ln.startswith("::error"))


# -- a listed test and its ceiling -----------------------------------------------------------------------------------

LISTED = f"{FILE}::test_kept"


def _listed(tmp_path):
    allow = tmp_path / "allow.txt"
    allow.write_text(f"{LISTED} # kept: the only end-to-end run\n")
    return allow


@pytest.mark.parametrize("seconds, code", [(120.0, 0), (120.1, 1), (60.0, 0)])
def test_a_listed_test_over_its_ceiling_fails_a_push(gate, tmp_path, capsys, seconds, code):
    ceiling = 120.0 if seconds >= 120 else 60.0
    report = _report(tmp_path, _case(FILE, "test_kept", seconds))
    got, out = _run(gate, capsys, report, "--allowlist", _listed(tmp_path), "--ceilings",
                    _ceilings(tmp_path, {LISTED: ceiling}), *PUSH)
    assert got == code, out
    assert ("title=Allowlisted test over its ceiling::" in out) is bool(code) or (not code and "warning" not in out)


def test_the_same_listed_test_on_a_pull_request_warns_unless_the_change_owns_it(gate, tmp_path, capsys, monkeypatch):
    report = _report(tmp_path, _case(FILE, "test_kept", 130.0))
    args = ("--allowlist", _listed(tmp_path), "--ceilings", _ceilings(tmp_path, {LISTED: 120.0}), "--own-against", "", *PULL)
    _own(gate, monkeypatch, ["tests/other/test_y.py"], {FILE: {LISTED}})
    code, out = _run(gate, capsys, report, *args)
    assert code == 0 and "::error" not in out
    assert "::warning file=tests/a/test_x.py,line=11,title=Allowlisted test over its ceiling::" in out
    # The control: the pull request touches that file, so the ceiling fails it.
    _own(gate, monkeypatch, [FILE], {FILE: {LISTED}})
    code, out = _run(gate, capsys, report, *args)
    assert code == 1 and "::error file=tests/a/test_x.py,line=11,title=Allowlisted test over its ceiling::" in out
    # ... as does a new id in an untouched file.
    _own(gate, monkeypatch, ["tests/other/test_y.py"], {FILE: set()})
    assert _run(gate, capsys, report, *args)[0] == 1


def test_a_listed_test_without_a_ceiling_or_without_the_file_is_never_failed_for_one(gate, tmp_path, capsys):
    report = _report(tmp_path, _case(FILE, "test_kept", 900.0))
    allow = _listed(tmp_path)
    assert _run(gate, capsys, report, "--allowlist", allow, *PUSH)[0] == 0                      # no file given
    assert _run(gate, capsys, report, "--allowlist", allow, "--ceilings", _ceilings(tmp_path, {"other::t": 60}),
                *PUSH)[0] == 0                                                                  # not in the file
    assert _run(gate, capsys, report, "--allowlist", allow, "--ceilings", _ceilings(tmp_path, {LISTED: 60}),
                *PUSH)[0] == 1                                                                  # the control


def test_a_ceiling_for_an_unlisted_test_is_not_a_ceiling(gate, tmp_path, capsys):
    report = _report(tmp_path, _case(FILE, "test_kept", 50.0))
    allow = tmp_path / "allow.txt"
    allow.write_text("# nothing listed\n")
    code, out = _run(gate, capsys, report, "--allowlist", allow, "--ceilings", _ceilings(tmp_path, {LISTED: 40}), *PUSH)
    assert code == 1 and "title=Test over 30 s::" in out and "ceiling" not in out


def test_a_listed_test_over_its_ceiling_is_run_again_alone_and_forgiven_under_it(gate, tmp_path, capsys, monkeypatch):
    report = _report(tmp_path, _case(FILE, "test_kept", 130.0))
    ceilings = _ceilings(tmp_path, {LISTED: 120.0})
    for again, code in ((100.0, 0), (121.0, 1)):
        monkeypatch.setattr(gate, "run_pytest_again", lambda nodeids, rep, cache, s=again, **_: rep.write_text(
            '<testsuites><testsuite name="pytest">' + _case(FILE, "test_kept", s) + "</testsuite></testsuites>") or 0)
        got, out = _run(gate, capsys, report, "--allowlist", _listed(tmp_path), "--ceilings", ceilings, "--retry-outliers", *PUSH)
        assert got == code, out
        assert ("Outlier forgiven" in out) is (code == 0)


def test_warn_only_turns_a_ceiling_into_a_warning_too(gate, tmp_path, capsys):
    report = _report(tmp_path, _case(FILE, "test_kept", 130.0))
    args = ("--allowlist", _listed(tmp_path), "--ceilings", _ceilings(tmp_path, {LISTED: 120.0}), *PUSH)
    assert _run(gate, capsys, report, *args, "--warn-only")[0] == 0
    assert _run(gate, capsys, report, *args)[0] == 1


def test_an_unreadable_ceilings_file_does_not_pass(gate, tmp_path, capsys):
    report = _report(tmp_path, _case(FILE, "test_fast", 1.0))
    broken = tmp_path / "ceilings.json"
    broken.write_text("{not json")
    assert _run(gate, capsys, report, "--ceilings", broken)[0] == 2
    assert _run(gate, capsys, report, "--ceilings", tmp_path / "missing.json")[0] == 2


# -- what the base collects, asked of git and pytest ------------------------------------------------------------------

def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args], cwd=cwd, check=True,
                   capture_output=True)


def test_scope_against_a_real_base_reads_the_diff_and_the_ids_the_base_collects(gate, tmp_path, monkeypatch):
    """A registry-style parametrisation: the head adds an id to `test_p.py` by changing `registry.py`, never `test_p.py`."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "registry.py").write_text("NAMES = ['a', 'b']\n")
    (tmp_path / "tests" / "test_p.py").write_text(
        "import pytest\nfrom registry import NAMES\n\n@pytest.mark.parametrize('n', NAMES)\ndef test_p(n):\n    pass\n")
    (tmp_path / "tests" / "test_q.py").write_text("def test_q():\n    pass\n")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "."), _git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "src" / "registry.py").write_text("NAMES = ['a', 'b', 'c']\n")
    (tmp_path / "tests" / "test_q.py").write_text("def test_q():\n    assert True\n")
    _git(tmp_path, "commit", "-q", "-am", "head")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "src"))
    scope = gate.scope_against("")                       # no revision given: the head's first parent
    try:
        assert scope.changed == {"src/registry.py", "tests/test_q.py"}
        T = gate.TestTime
        old = T("tests/test_p.py::test_p[a]", "tests/test_p.py", 1, 1.0, "passed")
        new = T("tests/test_p.py::test_p[c]", "tests/test_p.py", 1, 1.0, "passed")
        touched = T("tests/test_q.py::test_q", "tests/test_q.py", 1, 1.0, "passed")
        assert not scope.is_own(old)                     # in the base's collection, in a file nobody touched
        assert scope.is_own(new)                         # the control: the base does not collect it
        assert scope.is_own(touched)                     # its file is in the diff
    finally:
        scope.close()
    assert gate.scope_against("HEAD^1").changed == scope.changed
    # Strict, not lenient: a zero revision (a new branch's `before`) or one the clone cannot get makes every test the
    # change's own, where the diff of the last commit alone would have let the earlier commits' tests go.
    assert gate.scope_against("0" * 40).changed is None
    assert gate.scope_against("1234567890abcdef" * 2 + "12345678").changed is None


# -- CI runs the rule everywhere it runs the gate ---------------------------------------------------------------------

def _ci():
    return yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))


def _gate_steps(ci):
    for name, job in ci["jobs"].items():
        for step in job.get("steps", []):
            run = re.sub(r"\\\n", " ", step.get("run", ""))
            if "report_test_durations.py" in run and "--allowlist" in run and "--fail-over" not in run:
                yield name, job, step, run


def _problems(ci):
    problems, seen = [], 0
    for name, job, step, run in _gate_steps(ci):
        seen += 1
        if "--own-against HEAD^1" not in run:
            problems.append(f"{name}: the gate does not say what the change's own tests are")
        if "--ceilings tests/duration_ceilings.json" not in run:
            problems.append(f"{name}: the gate reads no ceilings")
        checkout = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout"))
        if (checkout.get("with") or {}).get("fetch-depth") != 2:
            problems.append(f"{name}: the checkout is not two deep, so the head's first parent is not there to diff")
    if seen < 7:
        problems.append(f"{seen} gate steps found; the three shard lanes, their three completion jobs and usd-core-newest have one each")
    return problems


def test_every_gate_step_of_ci_knows_the_changes_own_tests_and_the_ceilings():
    assert _problems(_ci()) == []


@pytest.mark.parametrize("mutation", ["no own-against", "no ceilings", "shallow checkout", "wrong base", "step lost"])
def test_control_the_check_fails_on_each_way_a_gate_step_can_stop_applying_the_rule(mutation):
    ci = _ci()
    name, job, step, _ = next(iter(_gate_steps(ci)))
    if mutation == "no own-against":
        step["run"] = step["run"].replace(" --own-against HEAD^1", "")
    elif mutation == "no ceilings":
        step["run"] = step["run"].replace(" --ceilings tests/duration_ceilings.json", "")
    elif mutation == "shallow checkout":
        next(s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout")).pop("with", None)
    elif mutation == "wrong base":
        step["run"] = step["run"].replace("--own-against HEAD^1", "--own-against HEAD")
    else:
        job["steps"].remove(step)
    assert _problems(ci), mutation


def test_the_planted_slow_test_stays_over_the_flat_line_so_that_every_lane_fails_on_it(gate):
    """``plant_slow_tests`` plants a test that is no change's own, so only the flat 90 s fails it (and its re-run alone)."""
    sleeps = [float(m) for m in re.findall(r"time\.sleep\(([\d.]+)\)", (REPO_ROOT / "tests" / "test_canary.py").read_text())]
    assert len(sleeps) == 2 and max(sleeps) > gate.OTHER_FAIL_OVER, sleeps
    assert 5.0 < min(sleeps) <= gate.OTHER_FAIL_OVER                      # and the other one only warns


def test_the_shipped_allowlist_holds_no_entry_whose_reason_the_triage_left_to_be_written():
    assert "to be written" not in (REPO_ROOT / "tests" / "duration_allowlist.txt").read_text(encoding="utf-8")


def test_the_shipped_ceilings_name_listed_tests_and_keep_the_floor(gate):
    table = json.loads((REPO_ROOT / "tests" / "duration_ceilings.json").read_text(encoding="utf-8"))
    assert table["written_by"].startswith("scripts/duration_triage.py") and table["source"] and table["generated"]
    allow = gate.read_allowlist(REPO_ROOT / "tests" / "duration_allowlist.txt")
    ceilings = gate.read_ceilings(REPO_ROOT / "tests" / "duration_ceilings.json")
    assert ceilings, "the ceilings file is empty"
    assert not [n for n in ceilings if n not in allow], "a ceiling for a test that is not listed"
    assert not [n for n, c in ceilings.items() if c < 60], "a ceiling under the 60 s floor"


# -- the weekly triage ------------------------------------------------------------------------------------------------

def _obs(triage, run_seconds):
    return [triage.Observation(1000 + i, "3.12", s) for i, s in enumerate(run_seconds)]


def test_the_triage_adds_a_test_over_the_line_once_or_near_it_in_a_quarter_of_the_reports(triage):
    observed = {
        "t::once": _obs(triage, [3, 3, 3, 3, 31]),                  # over 30 s once
        "t::near": _obs(triage, [3, 26, 3, 3]),                     # over 25.5 s in one of four
        "t::rare_near": _obs(triage, [3, 26, 3, 3, 3]),             # in one of five: not a quarter
        "t::fast": _obs(triage, [3, 4, 3, 4]),
        "t::listed": _obs(triage, [40, 41, 42]),                    # already listed: not added again
    }
    proposal = triage.triage(observed, {"t::listed": "kept: x"}, {})
    assert sorted(proposal.additions) == ["t::near", "t::once"]
    # The control: with every test listed nothing is added, and with none listed the same two are.
    assert triage.triage(observed, dict.fromkeys(observed, "kept: x"), {}).additions == {}


def test_the_triage_removes_a_listed_test_that_was_under_10_s_in_every_report_and_seen_enough(triage):
    observed = {"t::quick": _obs(triage, [2, 3, 9.9]), "t::once_slow": _obs(triage, [2, 3, 10.0]),
                "t::seen_twice": _obs(triage, [2, 3])}
    allow = dict.fromkeys(observed, "kept: x")
    proposal = triage.triage(observed, allow, {})
    assert sorted(proposal.removals) == ["t::quick"]
    assert "t::quick" not in proposal.ceilings
    assert triage.triage(observed, {}, {}).removals == {}            # an unlisted test is never "removed"


def test_the_triage_sets_a_ceiling_of_the_larger_of_3_medians_and_1_2_p95_never_under_60_s_and_keeps_an_unseen_tests(triage):
    observed = {"t::big": _obs(triage, [100, 110, 90, 120]), "t::small": _obs(triage, [15, 16, 17]),
                "t::twice": _obs(triage, [200, 200]),
                "t::spread": _obs(triage, [20] * 17 + [100, 200, 300])}      # median 20, p95 200
    allow = dict.fromkeys([*observed, "t::unseen"], "kept: x")
    proposal = triage.triage(observed, allow, {"t::unseen": 77.0, "t::twice": 55.0})
    assert proposal.ceilings["t::big"] == 315.0                      # 3 x median 105 (1.2 x p95 120 = 144)
    assert proposal.ceilings["t::small"] == 60.0                     # 3 x 16 = 48, under the floor
    assert proposal.ceilings["t::spread"] == 240.0                   # 1.2 x p95 200; 3 x median is only 60
    assert proposal.ceilings["t::twice"] == 55.0 and proposal.ceilings["t::unseen"] == 77.0
    assert triage.triage(observed, allow, {}).ceilings.get("t::unseen") is None


def test_a_ceiling_that_rose_over_1_5_times_is_listed_first_with_the_medians_and_never_folded_in(triage):
    """A sustained 5x regression must not turn into a bigger ceiling that the next body is silent about."""
    allow = dict.fromkeys(["t::slowed", "t::steady", "t::new", "t::eased"], "kept: x")
    observed = {"t::slowed": _obs(triage, [300] * 6), "t::steady": _obs(triage, [100] * 6),
                "t::new": _obs(triage, [100] * 6), "t::eased": _obs(triage, [50] * 6)}
    old = {"t::slowed": 230.0, "t::steady": 303.0, "t::eased": 600.0}
    proposal = triage.triage(observed, allow, old, old_medians={"t::slowed": 76.5, "t::steady": 101.0})
    assert proposal.raised == {"t::slowed": (230.0, 900.0, 76.5, 300.0)}          # not steady (1.0x), new (no old) or eased
    text = triage.body(proposal, today=datetime.date(2026, 11, 16), source="s", ceilings_only=False)
    assert text.index("Possible regressions") < text.index("Allowlist additions")
    assert "| `t::slowed` | 230 | 900 | 76.5 | 300.0 |" in text and "t::steady" not in text.split("### Allowlist additions")[0]
    # The control: a ceiling that stays within 1.5x is not listed, and a first triage has no old file to compare with.
    assert triage.triage(observed, allow, {"t::slowed": 700.0}).raised == {}
    assert triage.triage(observed, allow, {}).raised == {}


def test_the_ceilings_file_keeps_the_medians_the_next_triage_compares_against(triage, gate, tmp_path):
    observed = {"t::a": _obs(triage, [100, 101, 102])}
    proposal = triage.triage(observed, {"t::a": "kept: x"}, {})
    path = tmp_path / "c.json"
    path.write_text(triage.ceilings_json(proposal, today=datetime.date(2026, 11, 9), source="3 runs", medians={"t::a": 101.0}))
    assert triage.read_old(path) == ({"t::a": 303.0}, {"t::a": 101.0})
    assert triage.read_old(tmp_path / "absent.json") == ({}, {})
    assert gate.read_ceilings(path) == {"t::a": 303.0}                # the gate reads the ceilings and ignores the medians


def test_a_listed_test_whose_median_moved_within_the_window_is_named(triage):
    allow = dict.fromkeys(["t::moved", "t::flat", "t::few"], "kept: x")
    observed = {"t::moved": _obs(triage, [20] * 9 + [90] * 3), "t::flat": _obs(triage, [20, 40] * 6),
                "t::few": _obs(triage, [20, 90, 90])}
    assert list(triage.triage(observed, allow, {}).moved) == ["t::moved"]


def test_the_triage_never_proposes_the_planted_canary_or_a_test_pytest_does_not_collect(triage):
    canary = "tests/test_canary.py::test_a_planted_test_over_the_hard_line"
    observed = {canary: _obs(triage, [95, 95, 95]), "t::real": _obs(triage, [3, 3, 31]),
                "t::renamed": _obs(triage, [3, 3, 31])}
    assert sorted(triage.triage(observed, {}, {}).additions) == ["t::real", "t::renamed"]      # the canary is gone anyway
    proposal = triage.triage(observed, {}, {}, collected={"t::real", canary})
    assert sorted(proposal.additions) == ["t::real"] and proposal.uncollected == ("t::renamed",)
    assert canary not in proposal.additions and not any("canary" in n for n in proposal.ceilings)
    text = triage.body(proposal, today=datetime.date(2026, 11, 16), source="s", ceilings_only=False)
    assert "`t::renamed`" in text and "not collected" in text


def test_the_triage_names_the_listed_tests_a_ceiling_would_have_failed_and_the_slow_ones_the_line_passes(triage):
    observed = {"t::drifting": _obs(triage, [100] * 28 + [150, 500]),   # median 100, ceiling 300, the largest report over it
                "t::steady": _obs(triage, [100] * 30),
                "t::slowish": _obs(triage, [22, 23, 24, 22])}
    proposal = triage.triage(observed, {"t::drifting": "kept: x", "t::steady": "kept: x"}, {})
    assert list(proposal.over_ceiling) == ["t::drifting"]
    assert list(proposal.flagged) == ["t::slowish"] and "t::slowish" not in proposal.additions


def test_the_allowlist_edit_removes_and_appends_and_leaves_the_rest_alone(triage, gate, tmp_path):
    observed = {"a.py::new": _obs(triage, [31, 5, 5]), "a.py::quick": _obs(triage, [1, 1, 1])}
    text = ("# header\n#\na.py::keep # kept: still slow\na.py::quick # kept: was slow\n")
    proposal = triage.triage(observed, {"a.py::keep": "kept: still slow", "a.py::quick": "kept: was slow"}, {})
    edited = triage.apply_allowlist(text, proposal)
    lines = edited.splitlines()
    assert lines[:3] == ["# header", "#", "a.py::keep # kept: still slow"] and not any("a.py::quick" in ln for ln in lines)
    assert lines[3].startswith("a.py::new # kept: borderline on cold push runs: its largest cold duration was 31.0 s")
    assert "WHY IT MUST RUN ON EVERY PUSH: to be written" in lines[3]
    path = tmp_path / "allow.txt"
    path.write_text(edited)
    assert set(gate.read_allowlist(path)) == {"a.py::keep", "a.py::new"}        # the gate reads what the triage wrote


def test_the_ceilings_file_the_triage_writes_is_the_one_the_gate_reads(triage, gate, tmp_path):
    observed = {"t::a": _obs(triage, [100, 101, 102])}
    proposal = triage.triage(observed, {"t::a": "kept: x"}, {})
    path = tmp_path / "c.json"
    path.write_text(triage.ceilings_json(proposal, today=datetime.date(2026, 11, 9), source="3 runs"))
    assert gate.read_ceilings(path) == {"t::a": 303.0}
    assert json.loads(path.read_text())["source"] == "3 runs"


def _artifact(run_dir, python, shard, attempt, mode, cases):
    d = run_dir / f"durations-default-python-{python}-shard-{shard}-attempt-{attempt}"
    d.mkdir(parents=True)
    (d / f"cache-mode-shard{shard}.txt").write_text(mode + "\n")
    (d / f"test-results-shard{shard}.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">' + "".join(cases)
        + "</testsuite></testsuites>")


def test_a_downloaded_run_gives_its_cold_passing_tests_from_each_shards_newest_attempt(triage, gate, tmp_path):
    run = tmp_path / "555"
    _artifact(run, "3.12", 1, 1, "cold", [_case(FILE, "test_a", 40.0)])
    _artifact(run, "3.12", 1, 2, "cold", [_case(FILE, "test_a", 44.0)])          # the re-run's report replaces the first
    _artifact(run, "3.14", 2, 1, "cold", [_case(FILE, "test_b", 12.0)])
    _artifact(run, "3.13", 3, 1, "cold", [_case(FILE, "test_c", 12.0)])          # not a gating lane
    _artifact(run, "3.12", 4, 1, "warm", [_case(FILE, "test_d", 1.0)])           # a restored cache: not a cold time
    observed = triage.read_run(run, 555, gate)
    assert observed == {f"{FILE}::test_a": [triage.Observation(555, "3.12", 44.0)],
                        f"{FILE}::test_b": [triage.Observation(555, "3.14", 12.0)]}


#: The step of the default lane's job that gathers what the triage reads, and the shards one run's lanes hold: shard 1 of 3.12
#: re-run (its second attempt's report replaces the first), a warm shard, and each lane's shards with their own tests.
GATHER = "Gather the lane's timings for the weekly triage"
SHARDS = [("3.12", 1, 1, "cold", [("test_a", 40.0), ("test_b", 3.0)]), ("3.12", 1, 2, "cold", [("test_a", 44.0), ("test_b", 3.5)]),
          ("3.12", 2, 1, "cold", [("test_c", 31.0)]), ("3.12", 3, 1, "warm", [("test_d", 1.0)]),
          ("3.14", 1, 1, "cold", [("test_a", 41.0), ("test_b", 2.0)]), ("3.14", 2, 1, "cold", [("test_c", 29.5)]),
          ("3.14", 3, 1, "cold", [("test_d", 9.0)])]


def _shards_of_a_run(run: Path, lanes=("3.12", "3.14")) -> None:
    for python, shard, attempt, mode, cases in SHARDS:
        if python in lanes:
            _artifact(run, python, shard, attempt, mode, [_case(FILE, name, seconds) for name, seconds in cases])


def _timings_made_as_ci_makes_them(tmp_path: Path, run: Path, python: str, attempt: int = 1) -> Path:
    """The lane job's timings artifact for ``python``, made as ``ci.yml`` makes it: the lane's shard artifacts downloaded
    (each with its shard record and outcome), their newest attempts kept by ``scripts/latest_attempt.py``, then the gather
    step's own script run in bash."""
    runner = tmp_path / f"runner-{python}"
    for artifact in run.glob(f"durations-default-python-{python}-shard-*"):
        shutil.copytree(artifact, runner / "shards" / artifact.name)
        (runner / "shards" / artifact.name / "shard.json").write_text("{}")
        (runner / "shards" / artifact.name / "outcome.txt").write_text("success\n")
    subprocess.run([sys.executable, str(SCRIPTS / "latest_attempt.py"), str(runner / "shards")], check=True,
                   capture_output=True)
    (step,) = [s for s in _ci()["jobs"]["default"]["steps"] if s.get("name") == GATHER]
    subprocess.run(["bash", "-e", "-c", step["run"]], env={**os.environ, "RUNNER_TEMP": str(runner)}, check=True)
    merged = tmp_path / "merged" / run.name / f"timings-default-python-{python}-attempt-{attempt}"
    shutil.copytree(runner / "timings", merged)
    return merged


def test_a_runs_merged_timings_give_the_triage_what_its_shards_give(triage, gate, tmp_path):
    """The same tests, times and cache modes from the lane jobs' merged artifacts as from the run's shard artifacts, and
    each merged artifact holds every kept shard's report and cache mode and nothing else. ``MIME_PLANT_TIMINGS_MISMATCH``
    leaves one report out of a merged artifact: the control, rated by ``.github/probe/detection_rate.py``."""
    run = tmp_path / "shards" / "555"
    _shards_of_a_run(run)
    merged = [_timings_made_as_ci_makes_them(tmp_path, run, python) for python in ("3.12", "3.14")]
    assert sorted(p.name for p in merged[0].iterdir()) == [
        "cache-mode-shard1.txt", "cache-mode-shard2.txt", "cache-mode-shard3.txt",
        "test-results-shard1.xml", "test-results-shard2.xml", "test-results-shard3.xml"]
    if os.environ.get("MIME_PLANT_TIMINGS_MISMATCH"):
        (merged[1] / "test-results-shard2.xml").unlink()
    from_shards = triage.read_run(run, 555, gate)
    assert from_shards[f"{FILE}::test_a"] == [triage.Observation(555, "3.12", 44.0), triage.Observation(555, "3.14", 41.0)]
    assert from_shards[f"{FILE}::test_d"] == [triage.Observation(555, "3.14", 9.0)]      # 3.12's shard 3 is warm
    assert triage.read_run(merged[0].parent, 555, gate) == from_shards


def test_control_a_report_left_out_of_the_merged_timings_is_seen(triage, gate, tmp_path):
    run = tmp_path / "shards" / "555"
    _shards_of_a_run(run)
    merged = [_timings_made_as_ci_makes_them(tmp_path, run, python) for python in ("3.12", "3.14")]
    assert triage.read_run(merged[0].parent, 555, gate) == triage.read_run(run, 555, gate)
    (merged[1] / "test-results-shard2.xml").unlink()
    assert triage.read_run(merged[0].parent, 555, gate) != triage.read_run(run, 555, gate)


def test_a_lane_without_merged_timings_is_read_from_its_shards_and_the_newest_merged_attempt_wins(triage, gate, tmp_path):
    """A lane whose run has no merged artifact (a run from before the lane job uploaded it) is read from its shard
    artifacts; a lane whose job ran twice is read from its second attempt's artifact, and not from its shards again."""
    run = tmp_path / "shards" / "555"
    _shards_of_a_run(run)
    expected = triage.read_run(run, 555, gate)
    merged = _timings_made_as_ci_makes_them(tmp_path, run, "3.12", attempt=2)
    stale = merged.parent / "timings-default-python-3.12-attempt-1"
    stale.mkdir()
    (stale / "cache-mode-shard9.txt").write_text("cold\n")
    _report(stale, _case(FILE, "test_gone", 99.0), name="test-results-shard9.xml")
    _shards_of_a_run(merged.parent)
    assert triage.read_run(merged.parent, 555, gate) == expected


def test_a_runs_download_asks_for_the_merged_timings_first_and_the_shards_only_for_a_lane_without(triage, tmp_path,
                                                                                                     monkeypatch):
    calls = []

    def gh(argv, **_):
        patterns = [argv[i + 1] for i, a in enumerate(argv) if a == "--pattern"]
        calls.append(patterns)
        if patterns[0].startswith("timings-") and argv[3] == "555":                   # 3.14's lane job uploaded none
            (Path(argv[argv.index("--dir") + 1]) / "timings-default-python-3.12-attempt-1").mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(triage.subprocess, "run", gh)
    triage.download_run("o/r", 555, tmp_path / "555")
    assert calls == [["timings-default-python-3.12-attempt-*", "timings-default-python-3.14-attempt-*"],
                     ["durations-default-python-3.14-shard-*"]]
    calls.clear()
    triage.download_run("o/r", 556, tmp_path / "556")                                   # a run from before: no merged one
    assert calls == [["timings-default-python-3.12-attempt-*", "timings-default-python-3.14-attempt-*"],
                     ["durations-default-python-3.12-shard-*", "durations-default-python-3.14-shard-*"]]
    assert (tmp_path / "556").is_dir()


def test_main_writes_the_two_files_and_prints_the_body_and_a_dry_run_writes_nothing(triage, tmp_path, monkeypatch, capsys):
    reports = tmp_path / "reports"
    for i, seconds in enumerate((31.0, 20.0, 21.0)):
        _artifact(reports / str(900 + i), "3.12", 1, 1, "cold",
                  [_case("tests/a/test_x.py", "test_new", seconds), _case("tests/a/test_x.py", "test_kept", 80.0 + i),
                   _case("tests/a/test_x.py", "test_gone", 1.0)])
    allow, ceilings = tmp_path / "allow.txt", tmp_path / "c.json"
    allow.write_text("tests/a/test_x.py::test_kept # kept: a\ntests/a/test_x.py::test_gone # kept: b\n")
    monkeypatch.setattr(triage, "ALLOWLIST", allow)
    monkeypatch.setattr(triage, "CEILINGS", ceilings)
    monkeypatch.setattr(triage, "collect_ids", lambda files: {"tests/a/test_x.py::test_new"})
    assert triage.main(["--reports", str(reports), "--dry-run"]) == 0
    assert not ceilings.exists() and "test_gone" in allow.read_text()
    assert "test_new" in capsys.readouterr().out
    assert triage.main(["--reports", str(reports), "--body-file", str(tmp_path / "body.md")]) == 0
    assert "tests/a/test_x.py::test_new # kept: borderline" in allow.read_text() and "test_gone" not in allow.read_text()
    table = json.loads(ceilings.read_text())
    assert table["ceilings"]["tests/a/test_x.py::test_kept"] == 243.0       # 3 x median 81
    assert "runs 900 to 902" in table["source"]
    body = (tmp_path / "body.md").read_text()
    assert "### Allowlist additions (1)" in body and "### Allowlist removals (1)" in body
    # --ceilings-only leaves the allowlist as it is.
    allow.write_text("tests/a/test_x.py::test_kept # kept: a\ntests/a/test_x.py::test_gone # kept: b\n")
    assert triage.main(["--reports", str(reports), "--ceilings-only"]) == 0
    assert allow.read_text().count("\n") == 2 and "test_new" not in allow.read_text()


def test_what_main_writes_to_the_allowlist_holds_no_canary_and_no_test_pytest_does_not_collect(triage, tmp_path, monkeypatch):
    """Pinned for ``test_report_test_durations.py``'s two checks of the shipped allowlist (every entry a real, collected id)."""
    reports = tmp_path / "reports"
    stale, canary, real = "tests/a/test_x.py::test_renamed_since", "tests/test_canary.py::test_a_planted_test_over_the_hard_line", \
        "tests/a/test_x.py::test_real"
    for i in range(3):
        _artifact(reports / str(900 + i), "3.12", 1, 1, "cold", [
            _case("tests/a/test_x.py", "test_renamed_since", 40.0), _case("tests/a/test_x.py", "test_real", 41.0),
            _case("tests/test_canary.py", "test_a_planted_test_over_the_hard_line", 95.0)])
    allow, ceilings = tmp_path / "allow.txt", tmp_path / "c.json"
    allow.write_text("")
    monkeypatch.setattr(triage, "ALLOWLIST", allow)
    monkeypatch.setattr(triage, "CEILINGS", ceilings)
    monkeypatch.setattr(triage, "collect_ids", lambda files: {real})
    assert triage.main(["--reports", str(reports)]) == 0
    written = allow.read_text()
    assert real in written and stale not in written and "test_canary" not in written
    assert "test_canary" not in ceilings.read_text()


def test_collect_ids_asks_pytest_and_survives_a_file_the_checkout_no_longer_has(triage):
    ids = triage.collect_ids(["tests/test_canary.py", "tests/gone/test_deleted.py"])
    assert "tests/test_canary.py::test_a_planted_test_over_the_hard_line" in ids
    assert not [n for n in ids if "gone" in n]
    assert triage.collect_ids(["tests/gone/test_deleted.py"]) == set()


def test_the_triage_is_not_a_scheduled_job_and_runs_as_a_person():
    """A pull request opened with GITHUB_TOKEN starts no checks, so no workflow runs the triage."""
    for workflow in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        runs = [step.get("run", "") for job in document["jobs"].values() for step in job.get("steps", [])]
        assert not [r for r in runs if "duration_triage" in r], workflow.name
        assert "schedule" not in (document.get(True) or document.get("on") or {}) or workflow.name in (
            "ci.yml", "nightly.yml"), workflow.name
