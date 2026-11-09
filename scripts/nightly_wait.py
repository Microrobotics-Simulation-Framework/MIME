#!/usr/bin/env python
"""Wait for the runs of ``ci.yml`` that the nightly workflow dispatched, and fail naming each lane that is not green:
the second step of ``.github/workflows/nightly.yml``. Standard library and ``gh`` alone.

``gh workflow run`` answers before the run exists and names none. So for each branch this finds the newest
``workflow_dispatch`` run of ``ci.yml`` on it created at or after ``--since`` (the time the dispatching step began),
waits for it to complete, and reads its jobs at their latest attempts. The run's own conclusion is not trusted alone:
the two ``-main`` lanes and ``usd-core-newest`` are ``continue-on-error``, so a run can read ``success`` over a red lane
job (the lane jobs themselves are red on a dispatch, so the run is red too; this reads the jobs as well, from the same
list the release gate reads, ``scripts/release.py check-ci``). A branch fails when a lane job
(``release.LANE_JOBS``) is missing or not ``success``, when the run concluded other than ``success``, or when its run
never appeared or did not complete within ``--wait-minutes``. The failure names the branch, the lane and the run's
address, so GitHub's mail about a failed scheduled run says which lane broke.

    python3 scripts/nightly_wait.py --repository OWNER/NAME --since 2026-11-09T02:17:00Z release/v0.3.0 [BRANCH...]

Exits 0 when every branch's run is green, 1 otherwise."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release  # noqa: E402  (the lane names and the jobs' read, one place for the gate and the nightly)

Gh = Callable[[list[str]], str]
#: A job that ended badly and is not a lane's: named too, since the run is red for it.
BAD = ("failure", "timed_out", "cancelled", "action_required", "startup_failure")


def gh_with_retries(arguments: list[str], tries: int = 3, pause: float = 20.0,
                    sleep: Callable[[float], None] = time.sleep) -> str:
    """``gh`` with ``arguments``, asked again after a failed call (a server error in a long wait must not fail the night)."""
    for attempt in range(1, tries + 1):
        try:
            return release.run_gh(arguments)
        except (subprocess.CalledProcessError, FileNotFoundError):
            if attempt == tries:
                raise
            sleep(pause)
    raise AssertionError("unreachable")


def find_run(repository: str, branch: str, since: str, gh: Gh, sleep: Callable[[float], None], minutes: float) -> dict[str, Any] | None:
    """The newest dispatched run of ``ci.yml`` on ``branch`` created at or after ``since``, asked for every 15 s until
    ``minutes`` have passed (the run exists a few seconds after the dispatch is accepted)."""
    url = (f"repos/{repository}/actions/workflows/{release.WORKFLOW}/runs?branch={branch}&event=workflow_dispatch"
           f"&per_page=30")
    waited = 0.0
    while True:
        listed = json.loads(gh(["api", url]))
        runs = [run for run in listed.get("workflow_runs", []) if str(run.get("created_at", "")) >= since]
        if runs:
            return max(runs, key=lambda run: int(run["id"]))
        if waited >= minutes * 60:
            return None
        sleep(15.0)
        waited += 15.0


def wait_for(repository: str, run_id: int, gh: Gh, sleep: Callable[[float], None], minutes: float, poll: float) -> dict[str, Any] | None:
    """The run once it has completed, or ``None`` after ``minutes``."""
    waited = 0.0
    while True:
        run = json.loads(gh(["api", f"repos/{repository}/actions/runs/{run_id}"]))
        if run.get("status") == "completed":
            return run
        if waited >= minutes * 60:
            return None
        sleep(poll)
        waited += poll


def branch_problems(repository: str, branch: str, since: str, gh: Gh, sleep: Callable[[float], None] = time.sleep,
                    minutes: float = 300.0, poll: float = 60.0) -> list[str]:
    """Everything wrong with the run dispatched on ``branch``, each line naming the branch and what failed."""
    found = find_run(repository, branch, since, gh, sleep, 5.0)
    if found is None:
        return [f"{branch}: no run of {release.WORKFLOW} was found for the dispatch (created at or after {since})"]
    run = wait_for(repository, int(found["id"]), gh, sleep, minutes, poll)
    address = str(found.get("html_url", found["id"]))
    if run is None:
        return [f"{branch}: {address} did not complete within {minutes:g} minutes"]
    jobs = release.run_jobs(repository, int(found["id"]), gh)
    problems = [f"{branch}: lane {problem} ({address})" for problem in release.lane_problems(jobs)]
    others = [str(job["name"]) for job in jobs if job.get("conclusion") in BAD and str(job["name"]) not in release.LANE_JOBS
              and not str(job["name"]).startswith(("default tests (", "openmaterica-main tests (", "maddening-main tests (",
                                                   "completion tests ("))]
    problems += [f"{branch}: job {name} concluded red ({address})" for name in others]
    if run.get("conclusion") != "success" and not problems:
        problems.append(f"{branch}: the run concluded {run.get('conclusion')} ({address})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--since", required=True, help="UTC time, 2026-11-09T02:17:00Z form: the dispatching began")
    parser.add_argument("--wait-minutes", type=float, default=300.0, help="the longest wait for one run (default 300)")
    parser.add_argument("branches", nargs="+")
    arguments = parser.parse_args(argv)
    failed = False
    for branch in arguments.branches:
        try:
            problems = branch_problems(arguments.repository, branch, arguments.since, gh_with_retries,
                                       minutes=arguments.wait_minutes)
        except (subprocess.CalledProcessError, ValueError, KeyError) as error:
            problems = [f"{branch}: the run could not be read ({type(error).__name__}: {error})"]
        for problem in problems:
            print(f"::error title=Nightly CI::{problem}")
        if not problems:
            print(f"{branch}: all five lanes are green")
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
