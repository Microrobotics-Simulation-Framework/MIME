"""The nightly workflow (``.github/workflows/nightly.yml``) dispatches the full matrix of ``ci.yml`` on each active
release branch. The wiring read from the file, and the job's script run in bash with ``gh`` stubbed: one dispatch per
branch whose newest commit is under 45 days old, the branch of a closed line left alone, a refused dispatch failing the
job after the others were tried, and nothing to dispatch on failing it too (a nightly run that ran nothing is no pass).
GitHub runs a schedule from the default branch's file only, so this file fires from ``main`` once it is there."""

import datetime
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NIGHTLY = ROOT / ".github" / "workflows" / "nightly.yml"
# The stub applies the one jq filter the job asks for (the names that start with release/) itself.
STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$1" in
  api)
    case "$*" in
      *"/branches?per_page"*) printf '%s\\n' $STUB_BRANCHES | grep '^release/' ;;
      *"/branches/release%2Fv0.3.0"*) echo "$STUB_NEW" ;;
      *"/branches/release%2Fv0.2.0"*) echo "$STUB_OLD" ;;
      *) echo "unexpected: $*" >&2; exit 9 ;;
    esac ;;
  workflow) case "$*" in *"--ref $STUB_REFUSED "*) exit 1 ;; esac ;;
esac
"""


def load() -> dict:
    return yaml.safe_load(NIGHTLY.read_text(encoding="utf-8"))


def run(tmp_path: Path, branches: str, new: str, old: str, refused: str = "-"):
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "gh").write_text(STUB, encoding="utf-8")
    (stubs / "gh").chmod(0o755)
    step = load()["jobs"]["dispatch"]["steps"][0]
    log = tmp_path / "log"
    log.touch()
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "STUB_LOG": str(log), "STUB_BRANCHES": branches, "STUB_NEW": new,
           "STUB_OLD": old, "STUB_REFUSED": refused, "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
           "GH_TOKEN": "t", "REPOSITORY": "an-owner/a-repository", "ACTIVE_DAYS": step["env"]["ACTIVE_DAYS"]}
    done = subprocess.run(["bash", "-e", "-c", step["run"]], capture_output=True, text=True, check=False, env=env)
    return done.returncode, [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("workflow")]


def ago(days: int) -> str:
    when = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_it_is_scheduled_nightly_runs_on_demand_and_may_only_start_workflow_runs():
    workflow = load()
    on = workflow.get("on", workflow.get(True))
    assert set(on) == {"schedule", "workflow_dispatch"} and len(on["schedule"]) == 1
    minute, hour, *rest = on["schedule"][0]["cron"].split()
    assert rest == ["*", "*", "*"] and minute.isdigit() and hour.isdigit(), "a fixed time, every day"
    assert workflow["permissions"] == {"actions": "write"}
    assert "permissions" not in workflow["jobs"]["dispatch"]


def test_it_dispatches_ci_with_a_warm_cache_on_each_active_release_branch_and_leaves_a_closed_line_alone(tmp_path):
    status, dispatched = run(tmp_path, "main\nrelease/v0.2.0\nrelease/v0.3.0\nfeat/x", ago(1), ago(150))
    assert status == 0
    assert dispatched == ["workflow run ci.yml --repo an-owner/a-repository --ref release/v0.3.0 -f compilation_cache=warm"]


def test_it_dispatches_on_every_active_branch(tmp_path):
    status, dispatched = run(tmp_path, "release/v0.2.0\nrelease/v0.3.0", ago(2), ago(3))
    assert status == 0 and len(dispatched) == 2


def test_a_refused_dispatch_fails_the_job_after_the_others_were_tried(tmp_path):
    status, dispatched = run(tmp_path, "release/v0.2.0\nrelease/v0.3.0", ago(2), ago(3), refused="release/v0.2.0")
    assert status == 1 and len(dispatched) == 2


def test_with_no_active_branch_it_fails(tmp_path):
    status, dispatched = run(tmp_path, "main\nrelease/v0.2.0", ago(1), ago(150))
    assert status == 1 and dispatched == []
