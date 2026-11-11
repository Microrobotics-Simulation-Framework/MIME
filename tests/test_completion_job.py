"""CI runs the shell-completion tests (``tests/cli/test_completion.py``) in a job of their own and installs the shells
they run under (zsh and fish) there, on no shard: the install is an ``apt-get`` that took 8 min 19 s on one runner (run
42049802999, lost to its 20-minute limit), 9 min 51 s on another and 6 min on a third (run 42098189628), and beside a
shard's 11 to 14 minutes of tests no cut or retry made it fit. This file holds the job's wiring in each of the three
lanes the shards run in (the same Python, set-up and condition as the lane's shards, the shells installed, the file run
alone, a skip failing it, a timeout above the install script's worst case), the shards' share (no shard installs a
shell, each deselects the file, and the option really keeps the file out of a collection), the lane's job under its own
name needing the job and checking it, the check that reads the job's report, and the install script itself
(``scripts/install_shells.sh``) against fake ``sudo``, ``dpkg`` and ``apt-get``.

The completion tests fail, not skip, under CI when a shell is missing (``shell_path``), so a shard that kept the file
by mistake is red by name; the controls are fixed plants, each failing on every run."""

import copy
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
COMPLETION = "tests/cli/test_completion.py"
CHECK = ROOT / "scripts" / "check_no_skips.py"
#: Each lane: the job that makes its required check, its shard job and its completion tests job.
LANES = {
    "default": ("default-tests", "default-completion"),
    "openmaterica-main": ("openmaterica-main-tests", "openmaterica-main-completion"),
    "maddening-main": ("maddening-main-tests", "maddening-main-completion"),
}
#: The weekly lane that runs the whole suite in one job, and installs the shells itself.
UNSHARDED = "usd-core-newest"
#: What the job's limit must exceed the install script's worst case by, for the set-up before and the file after it.
MARGIN_SECONDS = 10 * 60
#: The conditions the steps that decide a lane's result run under: ``scripts/check_no_skips.py`` after the tests step, and
#: a lane's read of its completion job after anything red. Any other (``${{ false }}`` included) switches the check off.
LANE_RAN = "${{ !cancelled() && env.MIME_CI_RUNS == 'true' }}"
AFTER_TESTS = "${{ !cancelled() && steps.tests.outcome != 'skipped' }}"
#: The completion job's own ``if``, per lane: the lane's shards' (the default lane's every Python that runs; the others, where the count job says the lane runs).
JOB_IF = {"default": "github.event_name != 'schedule'",
          "openmaterica-main": "${{ github.event_name != 'schedule' && needs.shards.outputs.openmaterica == 'true' }}",
          "maddening-main": "${{ github.event_name != 'schedule' && needs.shards.outputs.maddening == 'true' }}"}
#: What the lane's own read of the completion job runs under: the default lane's job runs in every mode but refuses in a
#: reduced run (``MIME_CI_RUNS``); the other two have no job where their lane does not run.
LANE_CHECK_IF = {"default": LANE_RAN, "openmaterica-main": "${{ !cancelled() }}", "maddening-main": "${{ !cancelled() }}"}
NEUTERING = re.compile(r"\|\|\s*(?:true|:)(?![\w-])|\bset\s+\+e\b")
INSTALL_RUN = "bash scripts/install_shells.sh"
needs_posix = pytest.mark.skipif(sys.platform == "win32", reason="a CI Linux runner's script")


def runs(job: dict, command: str) -> list[dict]:
    return [step for step in job.get("steps") or [] if command in str(step.get("run", ""))]


def setup_python(job: dict) -> list[object]:
    return [(step.get("with") or {}).get("python-version") for step in job.get("steps") or []
            if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")]


def step_faults(where: str, step: dict, expected_if: str | None = None) -> list[str]:
    found = []
    if step.get("if") != expected_if:
        found.append(f"{where}: runs under if: {step.get('if')!r}, not {expected_if!r}, so a condition can switch it off")
    if step.get("continue-on-error"):
        found.append(f"{where}: continue-on-error hides its failure")
    if NEUTERING.search(str(step.get("run", ""))):
        found.append(f"{where}: its script holds an '|| true' or a 'set +e'")
    return found


def completion_job_problems(lane: str, jobs: dict, worst: int) -> list[str]:
    tests_id, completion_id = LANES[lane]
    job, shard_job = jobs.get(completion_id), jobs[tests_id]
    if not isinstance(job, dict):
        return [f"{lane}: there is no job {completion_id} that runs the completion tests"]
    found: list[str] = []
    installs = runs(job, "install_shells.sh")
    if len(installs) != 1 or installs[0]["run"].strip() != INSTALL_RUN or "if" in installs[0]:
        found.append(f"{completion_id}: does not install zsh and fish exactly once, by scripts/install_shells.sh, on every run")
    else:
        found += step_faults(f"{completion_id}: the install", installs[0])
    tests = [step for step in job.get("steps") or [] if step.get("id") == "tests"]
    if len(tests) != 1 or f"pytest {COMPLETION} " not in tests[0].get("run", "") + " ":
        found.append(f"{completion_id}: does not run {COMPLETION} in its tests step")
    else:
        found += step_faults(f"{completion_id}: the tests", tests[0])
        if installs and job["steps"].index(installs[0]) > job["steps"].index(tests[0]):
            found.append(f"{completion_id}: installs after the tests")
        if "--deselect" in tests[0]["run"] or "-k " in tests[0]["run"] or "--ignore" in tests[0]["run"]:
            found.append(f"{completion_id}: its tests step leaves tests of the file out")
    skips = runs(job, "scripts/check_no_skips.py")
    if len(skips) != 1 or "test-results.xml" not in skips[0]["run"]:
        found.append(f"{completion_id}: does not fail on a skipped test (scripts/check_no_skips.py over the tests' report)")
    else:
        found += step_faults(f"{completion_id}: the skip check", skips[0], AFTER_TESTS)
    limit = job.get("timeout-minutes")
    if not isinstance(limit, int) or limit * 60 < worst + MARGIN_SECONDS:
        found.append(f"{completion_id}: timeout-minutes {limit!r} is under the install script's worst case ({worst} s) "
                     f"and {MARGIN_SECONDS // 60} minutes for the rest")
    if job.get("if") != JOB_IF[lane]:
        found.append(f"{completion_id}: runs under if: {job.get('if')!r}, not {JOB_IF[lane]!r}, so a condition can switch it off")
    # the lane it stands for: what its shards ran in, so no coverage is lost
    if job.get("if") != shard_job.get("if") or job.get("continue-on-error") != shard_job.get("continue-on-error"):
        found.append(f"{completion_id}: runs under a condition other than the lane's shards' ({shard_job.get('if')!r})")
    if setup_python(job) != setup_python(shard_job):
        found.append(f"{completion_id}: sets up another Python than the lane's shards")
    matrix = lambda each: {k: v for k, v in ((each.get("strategy") or {}).get("matrix") or {}).items() if k != "shard"}
    if matrix(job) != matrix(shard_job):
        found.append(f"{completion_id}: does not run for each Python the lane's shards run for")
    sync = [step.get("run") for step in shard_job["steps"] if "uv sync" in str(step.get("run", ""))]
    if [step.get("run") for step in job["steps"] if "uv sync" in str(step.get("run", ""))] != sync:
        found.append(f"{completion_id}: installs the lane differently from its shards")
    if job.get("env", {}).get("MIME_CI_LANE") != shard_job.get("env", {}).get("MIME_CI_LANE"):
        found.append(f"{completion_id}: names another lane than its shards")
    return found


def shard_job_problems(lane: str, jobs: dict) -> list[str]:
    tests_id, _ = LANES[lane]
    steps = jobs[tests_id]["steps"]
    found = [f"{tests_id}: the step {step.get('name')!r} installs the shells; only the completion tests job does"
             for step in steps if re.search(r"install_shells\.sh|apt-get|\bapt\b", str(step.get("run", "")))]
    pytest_runs = [step for step in steps if "pytest" in str(step.get("run", ""))]
    if len(pytest_runs) != 1 or len(re.findall(rf"--deselect {COMPLETION}(?=\s)", pytest_runs[0]["run"])) != 1:
        found.append(f"{tests_id}: its pytest does not deselect {COMPLETION} (--deselect), so the shard that holds it "
                     f"would run it without the shells")
    return found


def summary_problems(lane: str, jobs: dict) -> list[str]:
    """The lane's job under its own name reads what the completion job ended as from the run's own list of jobs, as the
    default lane does for its shards, and not from ``needs.<job>.result``: a job with ``continue-on-error`` reads
    ``success`` there when it failed (the manual run's ``plant_completion_failure`` measured it, see the workflow)."""
    tests_id, completion_id = LANES[lane]
    summary = jobs[lane]
    if completion_id not in summary["needs"]:
        return [f"{lane}: the job that makes the required check does not wait on {completion_id}, so a red completion "
                f"job would not fail it"]
    if not isinstance(jobs.get(completion_id), dict):
        return []
    name = jobs[completion_id]["name"]
    steps = [step for step in summary["steps"] if "scripts/check_shard_jobs.py" in str(step.get("run", ""))
             and "completion tests (" in str(step.get("run", ""))]
    if len(steps) != 1:
        return [f"{lane}: {len(steps)} steps check the completion tests job's result; one does"]
    step = steps[0]
    run = str(step.get("run", ""))
    found = step_faults(f"{lane}: the step checking the completion tests job", step, LANE_CHECK_IF[lane])
    if not re.search(r"^\s*set -e?o pipefail$", run, re.MULTILINE) or "python3 scripts/run_jobs.py \"$JOBS_URL\"" not in run \
            or "scripts/check_shard_jobs.py" not in run or not run.rstrip().endswith("--shards 1"):
        found.append(f"{lane}: the step checking the completion tests job does not read the run's jobs by scripts/run_jobs.py "
                     f"under pipefail into scripts/check_shard_jobs.py for exactly one job")
    if f'--prefix "{name}"' not in run:
        found.append(f"{lane}: checks the completion tests job of another Python than its own, or of another lane's "
                     f"(its name is {name!r})")
    if "filter=latest" not in str((step.get("env") or {}).get("JOBS_URL", "")):
        found.append(f"{lane}: the step checking the completion tests job does not read the jobs at their latest attempts")
    if (summary.get("permissions") or {}).get("actions") != "read":
        found.append(f"{lane}: cannot read the run's jobs (permissions: actions: read)")
    return found


def workflow_problems(workflow: dict, worst: int) -> list[str]:
    jobs = workflow["jobs"]
    found: list[str] = []
    for lane in LANES:
        found += completion_job_problems(lane, jobs, worst) + shard_job_problems(lane, jobs) + summary_problems(lane, jobs)
    whole = [step for step in jobs[UNSHARDED]["steps"] if "apt-get install" in step.get("run", "")]
    if len(whole) != 1 or "if" in whole[0]:
        found.append(f"{UNSHARDED}: runs the whole suite and must install the shells unconditionally")
    return found


def worst_case_seconds(tmp_path: Path) -> int:
    """The install script's own printed worst case, from a run that waits for nothing."""
    done, _ = run_install(tmp_path, "exit 0\n", recorded=True)
    return int(re.search(r"worst case ([0-9]+) s", done.stdout).group(1))


@pytest.fixture
def workflow() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


@needs_posix
def test_every_lane_runs_the_completion_tests_in_a_job_of_its_own_and_no_shard_installs_a_shell(tmp_path, workflow):
    assert workflow_problems(workflow, worst_case_seconds(tmp_path)) == []


def _drop_step(job_id, find):
    def plant(workflow):
        job = workflow["jobs"][job_id]
        job["steps"] = [step for step in job["steps"] if not find(step)]
    return plant


def _alter(job_id, key, value):
    def plant(workflow):
        workflow["jobs"][job_id][key] = value
    return plant


def _alter_step(job_id, find, **changes):
    def plant(workflow):
        for step in workflow["jobs"][job_id]["steps"]:
            if find(step):
                step.update(changes)
    return plant


def _edit_run(job_id, find, old, new):
    def plant(workflow):
        for step in workflow["jobs"][job_id]["steps"]:
            if find(step):
                assert old in step["run"], old
                step["run"] = step["run"].replace(old, new)
    return plant


def _append_step(job_id, **step):
    def plant(workflow):
        workflow["jobs"][job_id]["steps"].insert(3, step)
    return plant


def _plants(lane: str):
    tests_id, completion_id = LANES[lane]
    install = lambda step: "install_shells.sh" in str(step.get("run", ""))
    pytest_step = lambda step: step.get("id") == "tests"
    skip_check = lambda step: "check_no_skips.py" in str(step.get("run", ""))
    summary_check = lambda step: "scripts/check_shard_jobs.py" in str(step.get("run", "")) \
        and f"completion tests (" in str(step.get("run", ""))
    plants = {
        "the-job-is-gone": (lambda w: w["jobs"].pop(completion_id), "there is no job"),
        "the-shells-are-not-installed": (_drop_step(completion_id, install), "does not install zsh and fish exactly once"),
        "the-install-is-conditional": (_alter_step(completion_id, install, **{"if": "false"}), "exactly once"),
        "the-install-may-fail": (_alter_step(completion_id, install, **{"continue-on-error": True}), "continue-on-error"),
        "another-file-is-run": (_edit_run(completion_id, pytest_step, COMPLETION, "tests/cli/test_other.py"),
                                "does not run tests/cli/test_completion.py"),
        "the-file-is-run-in-part": (_edit_run(completion_id, pytest_step, f"pytest {COMPLETION} ",
                                              f"pytest {COMPLETION} --deselect {COMPLETION}::test_x "),
                                    "leaves tests of the file out"),
        "a-skip-is-not-a-failure": (_drop_step(completion_id, skip_check), "does not fail on a skipped test"),
        "the-skip-check-reads-another-report": (_edit_run(completion_id, skip_check, "test-results.xml", "other.xml"),
                                                "does not fail on a skipped test"),
        "the-skip-check-is-swallowed": (_edit_run(completion_id, skip_check, "test-results.xml\"",
                                                  "test-results.xml\" || true"), "'|| true'"),
        "the-limit-is-the-shards": (_alter(completion_id, "timeout-minutes", 20), "is under the install script's worst case"),
        "the-limit-is-the-worst-case-alone": (_alter(completion_id, "timeout-minutes", 14),
                                              "is under the install script's worst case"),
        "another-condition": (_alter(completion_id, "if", "${{ github.event_name == 'push' }}"),
                              "under a condition other than the lane's shards'"),
        "another-python": (lambda w: _set_python(w, completion_id), "another Python than the lane's shards"),
        "a-shard-installs-the-shells": (_append_step(tests_id, name="Shells", run=INSTALL_RUN),
                                        "installs the shells; only the completion tests job does"),
        "a-shard-installs-them-by-apt": (_append_step(tests_id, name="Shells", run="sudo apt-get install -y zsh fish"),
                                         "installs the shells; only the completion tests job does"),
        "a-shard-keeps-the-file": (_edit_run(tests_id, lambda s: "pytest" in str(s.get("run", "")),
                                             f" --deselect {COMPLETION}", ""), "does not deselect"),
        "a-shard-deselects-another-file": (_edit_run(tests_id, lambda s: "pytest" in str(s.get("run", "")),
                                                     COMPLETION, "tests/cli/test_other.py"), "does not deselect"),
        "the-lane-does-not-wait-for-it": (lambda w: _unneed(w, lane, completion_id), "does not wait on"),
        "the-lane-does-not-check-it": (_drop_step(lane, summary_check), "steps check the completion tests job's result"),
        "the-lane-reads-the-needs-result": (_read_the_needs_result(lane, completion_id, summary_check),
                                            "steps check the completion tests job's result"),
        "the-lane-reads-another-jobs-name": (_edit_run(lane, summary_check, '--prefix "completion tests (',
                                                       '--prefix "completion tests (other '),
                                             "another Python than its own"),
        "the-lane-expects-no-job": (_edit_run(lane, summary_check, "--shards 1", "--shards 0"), "for exactly one job"),
        "the-lane-expects-two-jobs": (_edit_run(lane, summary_check, "--shards 1", "--shards 2"), "for exactly one job"),
        "the-lane-reads-no-run-jobs": (_edit_run(lane, summary_check, "python3 scripts/run_jobs.py", "gh api"),
                                       "does not read the run's jobs by scripts/run_jobs.py"),
        "the-lane-reads-without-pipefail": (_edit_run(lane, summary_check, "set -eo pipefail", "set -e"),
                                            "does not read the run's jobs by scripts/run_jobs.py"),
        "the-lane-reads-no-latest-attempts": (_alter_env(lane, summary_check, "JOBS_URL", "repos/o/r/actions/runs/1/jobs"),
                                              "latest attempts"),
        "the-lane-may-not-read-the-jobs": (_alter(lane, "permissions", {"contents": "read"}), "cannot read the run's jobs"),
        "the-lane-check-is-switched-off": (_alter_step(lane, summary_check, **{"if": "${{ false }}"}), "can switch it off"),
        "the-lane-check-is-switched-off-bare": (_alter_step(lane, summary_check, **{"if": "false"}), "can switch it off"),
        "the-lane-check-is-switched-off-by-a-conjunction": (
            _alter_step(lane, summary_check, **{"if": "${{ !cancelled() && false }}"}), "can switch it off"),
        "the-lane-check-runs-only-when-green": (_alter_step(lane, summary_check, **{"if": "${{ success() }}"}),
                                                "can switch it off"),
        "the-skip-check-is-switched-off": (_alter_step(completion_id, skip_check, **{"if": "${{ false }}"}), "can switch it off"),
        "the-skip-check-is-switched-off-bare": (_alter_step(completion_id, skip_check, **{"if": "false"}), "can switch it off"),
        "the-skip-check-runs-only-when-green": (_alter_step(completion_id, skip_check, **{"if": "${{ success() }}"}),
                                                "can switch it off"),
        "the-tests-are-switched-off": (_alter_step(completion_id, pytest_step, **{"if": "${{ false }}"}), "can switch it off"),
        "the-job-and-its-shards-are-switched-off": (lambda w: _switch_off_both(w, tests_id, completion_id),
                                                    "can switch it off"),
        "the-lane-s-check-is-swallowed": (_swallow(lane, summary_check), "'|| true'"),
    }
    if lane == "default":
        plants["the-matrix-lacks-a-python"] = (
            lambda w: w["jobs"][completion_id]["strategy"]["matrix"].update({"python-version": ["3.12", "3.13"]}),
            "does not run for each Python")
        plants["another-pythons-job-is-checked"] = (
            _edit_run(lane, summary_check, "(default, Python ${{ matrix.python-version }})", "(default, Python 3.12)"),
            "another Python than its own")
    return {f"{lane}:{name}": plant for name, plant in plants.items()}


def _set_python(workflow, job_id):
    for step in workflow["jobs"][job_id]["steps"]:
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
            step["with"]["python-version"] = "3.10"


def _alter_env(job_id, find, key, value):
    def plant(workflow):
        for step in workflow["jobs"][job_id]["steps"]:
            if find(step):
                step["env"][key] = value
    return plant


def _read_the_needs_result(lane, completion_id, find):
    """The workflow before the fix round: the step reads ``needs.<completion job>.result`` in place of the jobs' list."""
    def plant(workflow):
        for step in workflow["jobs"][lane]["steps"]:
            if find(step):
                step["env"] = {"COMPLETION": f"${{{{ needs.{completion_id}.result }}}}"}
                step["run"] = 'if [ "$COMPLETION" != success ]; then exit 1; fi\n'
    return plant


def _switch_off_both(workflow, tests_id, completion_id):
    workflow["jobs"][tests_id]["if"] = workflow["jobs"][completion_id]["if"] = "${{ false }}"


def _unneed(workflow, lane, completion_id):
    workflow["jobs"][lane]["needs"] = [each for each in workflow["jobs"][lane]["needs"] if each != completion_id]


def _swallow(lane, find):
    def plant(workflow):
        for step in workflow["jobs"][lane]["steps"]:
            if find(step):
                step["run"] += " || true\n"
    return plant


PLANTS = {key: value for lane in LANES for key, value in _plants(lane).items()}
NAMED = PLANTS


@needs_posix
@pytest.mark.parametrize("plant", NAMED, ids=list(NAMED))
def test_control_a_break_in_the_completion_job_s_wiring_is_named(tmp_path, workflow, plant):
    change, expected = NAMED[plant]
    worst = worst_case_seconds(tmp_path)
    assert workflow_problems(workflow, worst) == []
    broken = copy.deepcopy(workflow)
    change(broken)
    assert any(expected in problem for problem in workflow_problems(broken, worst)), (plant, workflow_problems(broken, worst))


def _jobs_reply(lane: str, completion: list[tuple[str, str | None]]) -> str:
    """The run's jobs as the endpoint gives the one page of them: the lane's shard jobs, all green, and ``completion``."""
    others = [(f"{lane} tests (shard {n} of 3)", "success") for n in (1, 2, 3)] + [("shard count", "success")]
    jobs = [{"name": name, "conclusion": conclusion} for name, conclusion in [*others, *completion]]
    return json.dumps({"total_count": len(jobs), "jobs": jobs}) + "\n"


def run_summary_check(tmp_path: Path, lane: str, reply: str, *, edit=lambda text: text) -> subprocess.CompletedProcess[str]:
    """The lane's step that checks the completion job, as the workflow has it, run for real against a ``gh`` that
    answers ``reply`` (the run's jobs, whatever GitHub gives for the job that ended as it ended)."""
    document = yaml.safe_load(CI.read_text(encoding="utf-8"))
    name = document["jobs"][LANES[lane][1]]["name"].replace("${{ matrix.python-version }}", "3.12")
    step = next(step for step in document["jobs"][lane]["steps"] if "scripts/check_shard_jobs.py" in str(step.get("run", ""))
                and "completion tests (" in str(step.get("run", "")))
    bin_ = tmp_path / "bin"
    bin_.mkdir(parents=True)
    (tmp_path / "reply").write_text(reply, encoding="utf-8")
    (bin_ / "gh").write_text(f'#!/bin/sh\ncat "{tmp_path}/reply"\n', encoding="utf-8")
    (bin_ / "gh").chmod(0o755)
    run = edit(step["run"].replace("${{ matrix.python-version }}", "3.12"))
    assert f'--prefix "{name}"' in run
    return subprocess.run(["bash", "-e", "-c", run], cwd=ROOT, capture_output=True, text=True, check=False, timeout=60,
                          env={"PATH": f"{bin_}:/usr/bin:/bin", "JOBS_URL": "repos/o/r/actions/runs/1/jobs?filter=latest",
                               "GH_TOKEN": "x"})


@needs_posix
@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("conclusion, status", [("success", 0), ("failure", 1), ("cancelled", 1), ("skipped", 1), (None, 1)],
                         ids=["success", "failure", "cancelled", "skipped", "not-finished"])
def test_the_lane_s_check_of_the_completion_job_follows_what_the_run_s_jobs_say_it_ended_as(tmp_path, lane, conclusion, status):
    """Run for real, each lane, with the job's conclusion as the jobs' list gives it: the job with ``continue-on-error``
    that failed is listed ``failure`` (the run 42111720966 shard jobs; the plant run of the fix round), and only
    ``success`` passes."""
    name = {"default": "completion tests (default, Python 3.12)"}.get(lane, f"completion tests ({lane})")
    done = run_summary_check(tmp_path, lane, _jobs_reply(lane, [(name, conclusion)]))
    assert done.returncode == status, (lane, conclusion, done)


@needs_posix
@pytest.mark.parametrize("lane", LANES)
def test_the_lane_s_check_fails_when_the_completion_job_is_absent_twice_listed_or_the_list_is_empty(tmp_path, lane):
    name = {"default": "completion tests (default, Python 3.12)"}.get(lane, f"completion tests ({lane})")
    for at, reply in enumerate([_jobs_reply(lane, []), _jobs_reply(lane, [(name, "success")] * 2), "",
                                _jobs_reply(lane, [(name.replace("completion", "other"), "success")])]):
        done = run_summary_check(tmp_path / str(at), lane, reply, edit=lambda text: text.replace(
            'run_jobs.py "$JOBS_URL"', 'run_jobs.py "$JOBS_URL" --pause 0'))  # an empty reply is read again: no waiting
        assert done.returncode == 1, (lane, at, done)


@needs_posix
@pytest.mark.parametrize("lane", LANES)
def test_control_the_lane_s_check_that_reads_nothing_passes_a_failed_job_so_the_test_above_can_fail(tmp_path, lane):
    """The plant of the two tests above: the same step with the checker replaced by ``cat`` passes a failed job and an
    absent one, where the real step fails them."""
    name = {"default": "completion tests (default, Python 3.12)"}.get(lane, f"completion tests ({lane})")
    plant = lambda text: text.replace("python3 scripts/check_shard_jobs.py", "cat #")
    done = run_summary_check(tmp_path / "failed", lane, _jobs_reply(lane, [(name, "failure")]), edit=plant)
    absent = run_summary_check(tmp_path / "absent", lane, _jobs_reply(lane, []), edit=plant)
    assert done.returncode == 0 and absent.returncode == 0, (done, absent)
    assert run_summary_check(tmp_path / "real", lane, _jobs_reply(lane, [(name, "failure")])).returncode == 1


def collected(tmp_path: Path, *options: str) -> list[str]:
    done = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *options, COMPLETION],
                          capture_output=True, text=True, cwd=ROOT, check=False, timeout=300,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "JAX_PLATFORMS": "cpu"})
    return [line for line in done.stdout.splitlines() if line.startswith(COMPLETION + "::")]


@pytest.mark.parametrize("lane", LANES)
def test_the_option_the_shards_pass_keeps_the_file_out_of_what_they_collect(tmp_path, workflow, lane):
    """The shards' own option, read from their Tests step and run against the file: nothing of it is collected (a
    deselected test is in no shard's record or report). The control: without it the same command collects the file."""
    steps = [step for step in workflow["jobs"][LANES[lane][0]]["steps"] if step.get("id") == "tests"]
    options = re.findall(r"--deselect \S+", steps[0]["run"])
    assert options == [f"--deselect {COMPLETION}"], options
    assert collected(tmp_path, *options[0].split()) == []
    assert len(collected(tmp_path)) > 20, "the control: the file is collected when nothing deselects it"


JUNIT = ('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="{n}">{cases}</testsuite></testsuites>')
CASE = '<testcase classname="tests.cli.test_completion" name="{name}" file="tests/cli/test_completion.py" time="0.1">{body}</testcase>'


def report(tmp_path: Path, *bodies: str) -> Path:
    cases = "".join(CASE.format(name=f"test_{at}", body=body) for at, body in enumerate(bodies))
    path = tmp_path / "test-results.xml"
    path.write_text(JUNIT.format(n=len(bodies), cases=cases), encoding="utf-8")
    return path


def check(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(CHECK), str(path)], capture_output=True, text=True, check=False)


def test_the_skip_check_passes_a_report_where_every_test_ran(tmp_path):
    done = check(report(tmp_path, "", "", ""))
    assert done.returncode == 0 and "3 tests ran, none skipped" in done.stdout, done


@pytest.mark.parametrize("body, said", [
    ('<skipped type="pytest.skip" message="zsh is not installed">x</skipped>', "reported skipped"),
    ('<failure message="x">x</failure>', "reported failure"),
    ('<error message="x">x</error>', "reported error"),
], ids=["a-skip", "a-failure", "an-error"])
def test_control_the_skip_check_fails_a_report_with_one_test_that_did_not_pass(tmp_path, body, said):
    """Planted in the middle of passing tests: one skip, failure or error among twenty-seven passes fails the job."""
    done = check(report(tmp_path, *[""] * 13, body, *[""] * 13))
    assert done.returncode == 1 and said in done.stdout and "test_13" in done.stdout, done


def test_control_the_skip_check_fails_a_report_with_no_tests_and_one_that_is_not_there(tmp_path):
    assert check(report(tmp_path)).returncode == 1
    done = check(tmp_path / "no-such-report.xml")
    assert done.returncode == 1 and "cannot be read" in done.stdout, done


WORD = re.compile(r"\b(?:zsh|fish)\b")


def files_asking_for_a_shell(root: Path) -> list[str]:
    """The test files under ``root`` that name zsh or fish."""
    return sorted(str(path.relative_to(root)) for path in root.rglob("*.py") if WORD.search(path.read_text(encoding="utf-8")))


def test_no_test_file_but_the_completion_tests_asks_for_zsh_or_fish():
    """The shells are installed for one file only; a second file using them would run on a shard, with no shell (and
    fail there by name). The control: the scan finds a file that asks."""
    asking = files_asking_for_a_shell(ROOT / "tests")
    assert set(asking) <= {"cli/test_completion.py", "test_completion_job.py"}, asking
    assert "cli/test_completion.py" in asking


def test_control_the_scan_finds_a_file_that_asks_for_a_shell(tmp_path):
    (tmp_path / "test_other.py").write_text('import shutil\nshutil.which("zsh")\n', encoding="utf-8")
    (tmp_path / "test_plain.py").write_text("selfish = 1\n", encoding="utf-8")
    assert files_asking_for_a_shell(tmp_path) == ["test_other.py"]


INSTALL = ROOT / "scripts" / "install_shells.sh"


def fake_tools(folder: Path, apt_body: str, *, recorded: bool = False) -> dict[str, str]:
    """An environment whose ``sudo``, ``dpkg`` and ``apt-get`` are scripts in ``folder``, so that the script is run for
    real and installs nothing. ``apt-get`` runs ``apt_body`` (``$COUNT_FILE`` holds the attempts so far, ``$attempt``
    the number of this one) and logs its arguments to ``$COUNT_FILE.apt``; ``sudo`` logs that it was used and runs its
    arguments. With ``recorded``, ``timeout`` and ``sleep`` are fakes that log their arguments to ``$COUNT_FILE.calls``
    and cut and wait for nothing (``timeout`` runs the command after its own options), so that the script's defaults
    can be read off a run that takes no time."""
    folder.mkdir()
    (folder / "sudo").write_text('#!/bin/sh\necho "sudo $1" >> "$COUNT_FILE.sudo"\nexec "$@"\n', encoding="utf-8")
    (folder / "dpkg").write_text('#!/bin/sh\necho "dpkg $*" >> "$COUNT_FILE.dpkg"\n', encoding="utf-8")
    (folder / "apt-get").write_text('#!/bin/sh\necho x >> "$COUNT_FILE"\necho "$*" >> "$COUNT_FILE.apt"\n'
                                    'attempt=$(wc -l < "$COUNT_FILE")\n' + apt_body, encoding="utf-8")
    if recorded:
        (folder / "timeout").write_text(
            '#!/bin/sh\necho "timeout $1 $2 $3" >> "$COUNT_FILE.calls"\nshift 2\nexec "$@"\n', encoding="utf-8")
        (folder / "sleep").write_text('#!/bin/sh\necho "sleep $1" >> "$COUNT_FILE.calls"\n', encoding="utf-8")
    for tool in folder.iterdir():
        tool.chmod(0o755)
    return {"PATH": f"{folder}:/usr/bin:/bin", "COUNT_FILE": str(folder / "count")}


def run_install(tmp_path: Path, apt_body: str, *, recorded: bool = False,
                **settings: str) -> tuple[subprocess.CompletedProcess[str], int]:
    environment = fake_tools(tmp_path / "bin", apt_body, recorded=recorded) | settings
    if not recorded:  # the recorded runs wait for nothing, so they keep the script's own pause
        environment.setdefault("SHELLS_PAUSE_SECONDS", "0")
    done = subprocess.run(["bash", str(INSTALL)], env=environment, capture_output=True, text=True, check=False, timeout=120)
    counted = Path(environment["COUNT_FILE"])
    return done, len(counted.read_text().splitlines()) if counted.exists() else 0


def lines(tmp_path: Path, suffix: str) -> list[str]:
    path = tmp_path / "bin" / f"count.{suffix}"
    return path.read_text().splitlines() if path.exists() else []


@needs_posix
def test_an_attempt_that_stalls_is_cut_and_made_again_and_the_second_installs(tmp_path):
    done, attempts = run_install(tmp_path, '[ "$attempt" -ge 2 ] && exit 0\nexec sleep 30\n', SHELLS_ATTEMPT_SECONDS="1")
    assert done.returncode == 0 and attempts == 2, done
    assert "attempt 1 of 3 failed or ran over 1 s" in done.stdout and "installed on attempt 2 of 3" in done.stdout


@needs_posix
def test_control_without_the_cut_a_stalled_attempt_is_waited_for(tmp_path):
    """The same stall with a limit longer than it: nothing is cut, so the one attempt is the only one."""
    done, attempts = run_install(tmp_path, "sleep 2\nexit 0\n", SHELLS_ATTEMPT_SECONDS="60")
    assert done.returncode == 0 and attempts == 1 and "attempt 1 of 3" in done.stdout


@needs_posix
def test_the_last_attempt_has_a_limit_of_its_own_so_an_install_slower_than_the_cut_still_passes(tmp_path):
    """The history's slow installs (121 to 320 s) all passed on one attempt: the cut ends the first attempts only."""
    done, attempts = run_install(tmp_path, '[ "$attempt" -ge 2 ] && { sleep 3; exit 0; }\nexec sleep 30\n',
                                 SHELLS_ATTEMPTS="2", SHELLS_ATTEMPT_SECONDS="1", SHELLS_FINAL_SECONDS="20")
    assert done.returncode == 0 and attempts == 2, done
    assert "installed on attempt 2 of 2" in done.stdout


@needs_posix
def test_control_the_last_attempt_is_cut_at_its_own_limit_and_the_step_then_fails(tmp_path):
    done, attempts = run_install(tmp_path, "exec sleep 30\n", SHELLS_ATTEMPTS="2", SHELLS_ATTEMPT_SECONDS="1",
                                 SHELLS_FINAL_SECONDS="2")
    assert done.returncode == 1 and attempts == 2, done
    assert "attempt 2 of 2 failed or ran over 2 s" in done.stdout


@needs_posix
def test_a_failing_install_is_tried_the_set_number_of_times_and_then_fails_the_step(tmp_path):
    done, attempts = run_install(tmp_path, "exit 100\n", SHELLS_ATTEMPTS="2", SHELLS_ATTEMPT_SECONDS="5")
    assert done.returncode == 1 and attempts == 2, done
    assert "were not installed in 2 attempts" in done.stdout
    assert len(lines(tmp_path, "dpkg")) == 1, "the half-done install is repaired between attempts, not after the last"


@needs_posix
def test_the_defaults_are_the_ones_the_step_documents_and_the_worst_case_is_inside_the_job(tmp_path):
    """Run with no ``SHELLS_*`` setting, against a ``timeout`` and a ``sleep`` that record their arguments: the first
    attempts are cut at 120 s, the last at 360 s, ``dpkg --configure -a`` at 60 s, a pause of 15 s follows each failed
    attempt that is not the last, apt gets what it needs (as root, unattended, without its recommendations, zsh and
    fish), and the worst case that the script prints is under the job's 20 minutes and under 15."""
    done, attempts = run_install(tmp_path, '[ "$attempt" -ge 3 ] && exit 0\nexit 100\n', recorded=True)
    assert done.returncode == 0 and attempts == 3, done
    assert lines(tmp_path, "calls") == [
        "timeout --kill-after=10 120 apt-get", "timeout --kill-after=10 60 dpkg", "sleep 15",
        "timeout --kill-after=10 120 apt-get", "timeout --kill-after=10 60 dpkg", "sleep 15",
        "timeout --kill-after=10 360 apt-get"]
    assert lines(tmp_path, "apt") == ["install -y --no-install-recommends zsh fish"] * 3
    assert lines(tmp_path, "dpkg") == ["dpkg --configure -a"] * 2
    assert lines(tmp_path, "sudo") == ["sudo timeout"] * 5, "every apt-get (3) and dpkg (2) runs as root, under the cut"
    worst = int(re.search(r"worst case ([0-9]+) s", done.stdout).group(1))
    assert worst == (120 + 10 + 60 + 10 + 15) * 2 + 360 + 10 and worst < 15 * 60, worst


@needs_posix
def test_an_apt_that_cannot_get_its_lock_is_not_asked_again_at_once(tmp_path):
    """apt exits 100 at once while another process holds its lock; the retries must be spaced, not three in a row."""
    done, attempts = run_install(tmp_path, "exit 100\n", recorded=True)
    assert done.returncode == 1 and attempts == 3, done
    assert [line for line in lines(tmp_path, "calls") if line.startswith("sleep")] == ["sleep 15", "sleep 15"]


@needs_posix
def test_a_dpkg_that_hangs_is_cut_and_does_not_hold_the_next_attempt(tmp_path):
    folder = tmp_path / "bin"
    environment = fake_tools(folder, '[ "$attempt" -ge 2 ] && exit 0\nexit 100\n')
    (folder / "dpkg").write_text("#!/bin/sh\nexec sleep 30\n", encoding="utf-8")
    environment |= {"SHELLS_DPKG_SECONDS": "1", "SHELLS_PAUSE_SECONDS": "0"}
    started = time.monotonic()
    done = subprocess.run(["bash", str(INSTALL)], env=environment, capture_output=True, text=True, check=False, timeout=120)
    assert done.returncode == 0 and "installed on attempt 2 of 3" in done.stdout, done
    assert time.monotonic() - started < 15
