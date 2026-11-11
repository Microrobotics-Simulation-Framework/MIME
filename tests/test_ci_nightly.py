"""The nightly workflow (``.github/workflows/nightly.yml``) dispatches the full matrix of ``ci.yml`` on each active
release branch. The wiring read from the file, and the job's script run in bash with ``gh`` stubbed: one dispatch per
branch whose newest commit is under 45 days old, the branch of a closed line left alone, a refused dispatch failing the
job after the others were tried, and nothing to dispatch on failing it too (a nightly run that ran nothing is no pass).
GitHub runs a schedule from the default branch's file only, so this file fires from ``main`` once it is there.

The job then waits for the runs it dispatched (``scripts/nightly_wait.py``, run here with ``gh`` faked) and fails itself,
naming the branch and the lane, when a lane job is missing or red, when the run concluded red, or when the run never
appeared: the two ``-main`` lanes' shard jobs are ``continue-on-error``, so without the wait nothing would show them red.
Controls: the same wait on a run whose lanes are all green passes, and a run that reads ``success`` over a red lane job
(the case ``continue-on-error`` makes) still fails."""

import datetime
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("mime_nightly_wait", ROOT / "scripts" / "nightly_wait.py")
assert _SPEC is not None and _SPEC.loader is not None
wait = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = wait
_SPEC.loader.exec_module(wait)
NIGHTLY = ROOT / ".github" / "workflows" / "nightly.yml"
# The stub applies the one jq filter the job asks for (the names that start with release/) itself.
STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$1" in
  api)
    case "$*" in
      *"/branches?per_page"*) printf '%s\\n' $STUB_BRANCHES | grep '^release/' ;;
      *"/branches/release%2Fv0.3.0"*) echo "$STUB_NEW" ;;
      *"/branches/release%2Fv0.2.0"*) [ -n "$STUB_UNREADABLE" ] && exit 1; echo "$STUB_OLD" ;;
      *) echo "unexpected: $*" >&2; exit 9 ;;
    esac ;;
  workflow) case "$*" in *"--ref $STUB_REFUSED "*) exit 1 ;; esac ;;
esac
"""


def load() -> dict:
    return yaml.safe_load(NIGHTLY.read_text(encoding="utf-8"))


def run(tmp_path: Path, branches: str, new: str, old: str, refused: str = "-", unreadable: str = ""):
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "gh").write_text(STUB, encoding="utf-8")
    (stubs / "gh").chmod(0o755)
    step = load()["jobs"]["dispatch"]["steps"][0]
    log = tmp_path / "log"
    log.touch()
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "STUB_LOG": str(log), "STUB_BRANCHES": branches, "STUB_NEW": new,
           "STUB_OLD": old, "STUB_REFUSED": refused, "STUB_UNREADABLE": unreadable, "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
           "GITHUB_OUTPUT": str(tmp_path / "output"), "GH_TOKEN": "t", "REPOSITORY": "an-owner/a-repository", "ACTIVE_DAYS": step["env"]["ACTIVE_DAYS"]}
    done = subprocess.run(["bash", "-e", "-c", step["run"]], capture_output=True, text=True, check=False, env=env)
    return done.returncode, [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("workflow")]


def outputs(tmp_path: Path) -> dict[str, str]:
    written = tmp_path / "output"
    lines = written.read_text(encoding="utf-8").splitlines() if written.exists() else []
    return dict(line.split("=", 1) for line in lines)


def ago(days: int) -> str:
    when = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_it_is_scheduled_nightly_runs_on_demand_and_may_only_start_workflow_runs():
    workflow = load()
    on = workflow.get("on", workflow.get(True))
    assert set(on) == {"schedule", "workflow_dispatch"} and len(on["schedule"]) == 1
    minute, hour, *rest = on["schedule"][0]["cron"].split()
    assert rest == ["*", "*", "*"] and minute.isdigit() and hour.isdigit(), "a fixed time, every day"
    assert workflow["permissions"] == {"actions": "write", "contents": "read"}
    assert "permissions" not in workflow["jobs"]["dispatch"]


def test_the_job_waits_for_the_runs_it_dispatched_after_a_failed_dispatch_too():
    steps = load()["jobs"]["dispatch"]["steps"]
    waiting = [step for step in steps if "scripts/nightly_wait.py" in str(step.get("run", ""))]
    assert len(waiting) == 1 and steps.index(waiting[0]) == len(steps) - 1
    step = waiting[0]
    assert "!cancelled()" in step["if"] and "steps.dispatch.outputs.branches" in step["if"]
    assert "continue-on-error" not in step, "a red lane must fail the job"
    assert steps[0]["id"] == "dispatch"
    assert "--since" in step["run"] and "$BRANCHES" in step["run"]
    assert any(str(each.get("uses", "")).startswith("actions/checkout@") for each in steps), "the script is checked out"
    assert load()["jobs"]["dispatch"]["timeout-minutes"] * 60 > 300 * 60, "the wait's limit is inside the job's"


def test_it_dispatches_ci_with_a_warm_cache_on_each_active_release_branch_and_leaves_a_closed_line_alone(tmp_path):
    status, dispatched = run(tmp_path, "main\nrelease/v0.2.0\nrelease/v0.3.0\nfeat/x", ago(1), ago(150))
    assert status == 0
    assert dispatched == ["workflow run ci.yml --repo an-owner/a-repository --ref release/v0.3.0 -f compilation_cache=warm"]
    written = outputs(tmp_path)
    assert written["branches"] == "release/v0.3.0" and written["started"].endswith("Z")


def test_it_dispatches_on_every_active_branch(tmp_path):
    status, dispatched = run(tmp_path, "release/v0.2.0\nrelease/v0.3.0", ago(2), ago(3))
    assert status == 0 and len(dispatched) == 2


def test_a_refused_dispatch_fails_the_job_after_the_others_were_tried(tmp_path):
    status, dispatched = run(tmp_path, "release/v0.2.0\nrelease/v0.3.0", ago(2), ago(3), refused="release/v0.2.0")
    assert status == 1 and len(dispatched) == 2
    assert outputs(tmp_path)["branches"] == "release/v0.3.0", "the wait still has the accepted dispatch"


def test_a_branch_whose_newest_commit_cannot_be_read_fails_the_job_after_the_others_were_tried(tmp_path):
    status, dispatched = run(tmp_path, "release/v0.2.0\nrelease/v0.3.0", ago(2), ago(3), unreadable="yes")
    assert status == 1 and len(dispatched) == 1 and outputs(tmp_path)["branches"] == "release/v0.3.0"


def test_with_no_active_branch_it_fails(tmp_path):
    status, dispatched = run(tmp_path, "main\nrelease/v0.2.0", ago(1), ago(150))
    assert status == 1 and dispatched == []


# -- the wait ---------------------------------------------------------------------------------------------------------

SINCE = "2026-11-09T02:16:00Z"
REPOSITORY = "an-owner/a-repository"
LANES = wait.release.LANE_JOBS


class FakeGh:
    """The calls the wait makes: a branch's dispatched runs, a run by id, and a run's jobs. ``rounds`` is how many reads
    of the run say ``in_progress`` before it completes."""

    def __init__(self, conclusion: str, jobs: list[dict], created: str = "2026-11-09T02:17:30Z", rounds: int = 2,
                 listed: bool = True) -> None:
        self.conclusion, self.jobs, self.created, self.rounds, self.listed, self.reads = conclusion, jobs, created, rounds, listed, 0

    def __call__(self, arguments: list[str]) -> str:
        url = arguments[1]
        if "/workflows/ci.yml/runs?" in url:
            assert "event=workflow_dispatch" in url
            runs = [{"id": 7, "created_at": self.created, "html_url": "https://example/runs/7"}] if self.listed else []
            return json.dumps({"workflow_runs": runs})
        if url.endswith("/actions/runs/7"):
            self.reads += 1
            done = self.reads > self.rounds
            return json.dumps({"status": "completed" if done else "in_progress",
                               "conclusion": self.conclusion if done else None})
        if "/actions/runs/7/jobs" in url:
            page = int(url.rsplit("&page=", 1)[1])
            return json.dumps({"total_count": len(self.jobs), "jobs": self.jobs[(page - 1) * 50:page * 50]})
        raise AssertionError(f"gh called with {arguments}")


def green(**changed: str) -> list[dict]:
    return [{"name": lane, "conclusion": changed.get(lane, "success")} for lane in LANES] + [
        {"name": "typecheck (pyright, Python 3.12)", "conclusion": "success"}]


def problems(gh: FakeGh, **keywords) -> list[str]:
    return wait.branch_problems(REPOSITORY, "release/v0.3.0", SINCE, gh, sleep=lambda seconds: None, **keywords)


def test_a_green_run_passes_after_waiting_for_it_to_complete():
    gh = FakeGh("success", green())
    assert problems(gh) == [] and gh.reads == 3


@pytest.mark.parametrize("lane", LANES)
def test_a_red_lane_fails_the_wait_naming_the_branch_and_the_lane_even_under_a_run_that_reads_success(lane):
    """The control: ``continue-on-error`` leaves a run ``success`` over a red job, and the wait reads the jobs."""
    found = problems(FakeGh("success", green(**{lane: "failure"})))
    assert len(found) == 1 and found[0].startswith(f"release/v0.3.0: lane {lane}: concluded failure")


def test_a_lane_job_that_is_missing_or_skipped_is_not_green():
    jobs = [job for job in green() if job["name"] != "maddening-main"]
    jobs = [{**job, "conclusion": "skipped"} if job["name"] == "openmaterica-main" else job for job in jobs]
    found = problems(FakeGh("success", jobs))
    assert any("maddening-main: the run has no such job" in line for line in found)
    assert any("openmaterica-main: concluded skipped" in line for line in found)


def test_a_red_run_over_green_lanes_still_fails_naming_the_red_job():
    jobs = green() + [{"name": "no private name or path", "conclusion": "failure"}]
    found = problems(FakeGh("failure", jobs))
    assert found == ["release/v0.3.0: job no private name or path concluded red (https://example/runs/7)"]


def test_a_run_that_never_appears_or_never_completes_fails():
    assert "no run of ci.yml was found" in problems(FakeGh("success", green(), listed=False))[0]
    assert "did not complete within" in problems(FakeGh("success", green(), rounds=10**6), minutes=2.0, poll=60.0)[0]


def test_a_run_created_before_the_dispatch_is_not_the_one_waited_for():
    assert "no run of ci.yml was found" in problems(FakeGh("success", green(), created="2026-11-08T02:17:30Z"))[0]
