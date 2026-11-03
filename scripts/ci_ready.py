#!/usr/bin/env python
"""Adding the label ``ready`` to a pull request runs CI's full matrix on its head: ``.github/workflows/ready.yml`` runs
this, which re-runs the head's newest run of each workflow named (``ci.yml``, ``platforms.yml``). Those workflows read
the label when a run starts, not from the event (their mode step), so the re-run takes the full matrix. Standard
library and ``gh`` only.

Why a re-run, and not a run of its own (CONTRIBUTING.md, "CI: the ``ready`` label"): a workflow that listened to
``labeled`` would start a run for every label, whose jobs are all skipped for a label other than ``ready``; that run,
the workflow's newest on the head, left the pull request blocked with every required check green. A manual run
(``workflow_dispatch``) is no way round it: GitHub evaluates for a pull request only the checks of runs a ``push``,
``pull_request`` or a few other events started, and a manual run is not among them. A re-run is the pull request's own
run, on the same commit, in the same check suite, its newest attempt counting.

For each workflow, the newest run a pull request event started on the head (the pull request's head commit as it is
now), and what is done with it:

- no run yet: nothing; the run the push starts reads the label itself;
- in flight, its mode step not started: nothing; it reads the label when it starts;
- in flight and full, or ended full and passed: nothing;
- in flight and reduced: cancelled, and re-run whole once it has ended;
- ended otherwise (reduced, failed, or cancelled before it took a mode): re-run whole.

Before a re-run the head is read again: a newer push's run reads the label itself, so an older head is left alone. A
run's mode is read from its jobs (``filter=latest``): the job that reads the labels (``--run WORKFLOW=JOB``) has a step
``The full matrix runs``, which runs only in the full matrix. The run is the pull request's, so a re-run stays in its
concurrency group and a newer push still cancels it.

    python3 scripts/ci_ready.py --repository OWNER/NAME --pull-request N --run "ci.yml=shard count" \\
        --run "platforms.yml=the run's mode"

Each line it prints also goes to the run page (``$GITHUB_STEP_SUMMARY``), so the ``Ready`` run says which runs it
re-ran and which it left, and why. Exits 0 when each workflow's run is full, will be, or was re-run; 1 when a request
fails (a pull request from a fork gets a read-only token: a maintainer re-runs its run, Re-run all jobs, which reads the
label too).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

MARKER = "The full matrix runs"
FULL, REDUCED, UNREAD, READING = "full", "reduced", "unread", "reading"
LEAVE, RERUN, CANCEL_AND_RERUN = "leave", "rerun", "cancel and rerun"

Api = Callable[..., Any]


def mode_of(jobs: list[dict[str, Any]], mode_job: str) -> str:
    """The mode a run took, read from its jobs at their latest attempt: ``full`` or ``reduced`` once its mode step has
    run, ``reading`` while it runs, ``unread`` before it starts (or when the run has no such job)."""
    job = next((each for each in jobs if each.get("name") == mode_job), None)
    steps = (job or {}).get("steps") or []
    marker = next((step for step in steps if step.get("name") == MARKER), None)
    if marker is not None and marker.get("status") == "completed":
        return {"success": FULL, "skipped": REDUCED}.get(str(marker.get("conclusion")), UNREAD)
    mode_step = next((step for step in steps if str(step.get("name", "")).startswith("The run's mode")), None)
    if mode_step is None or mode_step.get("status") == "queued" or not mode_step.get("started_at"):
        return UNREAD
    return READING


def action(run: dict[str, Any], mode: str) -> tuple[str, str]:
    """What to do with a workflow's newest run on the head, and why."""
    if run.get("status") != "completed":
        if mode == UNREAD:
            return LEAVE, "in flight, its mode not read yet: it reads the label when it starts"
        if mode == FULL:
            return LEAVE, "in flight, and full already"
        return CANCEL_AND_RERUN, "in flight in the reduced set"
    if mode == FULL and run.get("conclusion") == "success":
        return LEAVE, "ended full, and passed"
    why = {FULL: f"ended full, {run.get('conclusion')}", REDUCED: "ended in the reduced set"}.get(
        mode, f"ended {run.get('conclusion')} before it took a mode")
    return RERUN, why


def gh(path: str, method: str = "GET") -> Any:
    """``gh api`` on ``path``: its JSON, or None for an empty answer."""
    done = subprocess.run(["gh", "api", "--method", method, path], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"gh api --method {method} {path}: {done.stderr.strip() or done.stdout.strip()}")
    return json.loads(done.stdout) if done.stdout.strip() else None


def newest_run(api: Api, repository: str, workflow: str, head: str) -> dict[str, Any] | None:
    runs = (api(f"repos/{repository}/actions/workflows/{workflow}/runs?head_sha={head}&event=pull_request"
                f"&per_page=50") or {}).get("workflow_runs") or []
    return max(runs, key=lambda run: (str(run.get("created_at")), int(run.get("id", 0))), default=None)


def jobs_of(api: Api, repository: str, run_id: int) -> list[dict[str, Any]]:
    return (api(f"repos/{repository}/actions/runs/{run_id}/jobs?filter=latest&per_page=100") or {}).get("jobs") or []


def head_of(api: Api, repository: str, number: str) -> str:
    return str(api(f"repos/{repository}/pulls/{number}")["head"]["sha"])


def until(check: Callable[[], bool], seconds: float, sleep: Callable[[float], None], every: float = 5.0) -> bool:
    waited = 0.0
    while not check():
        if waited >= seconds:
            return False
        sleep(every)
        waited += every
    return True


def ready(api: Api, repository: str, number: str, runs: list[tuple[str, str]],
          sleep: Callable[[float], None] = time.sleep, say: Callable[[str], None] = print) -> int:
    """Re-run each workflow's newest run on the head that is not full already (the module's list); 0, or 1 when one
    could not be."""
    failed = 0
    head = head_of(api, repository, number)
    for workflow, mode_job in runs:
        run = newest_run(api, repository, workflow, head)
        if run is None:
            say(f"{workflow}: no run on the head {head[:12]} yet; the run its push starts reads the label")
            continue
        url = run.get("html_url", run["id"])
        mode = mode_of(jobs_of(api, repository, run["id"]), mode_job)
        if mode == READING and run.get("status") != "completed":
            # Its mode step is reading the labels now: wait for what it read.
            until(lambda: mode_of(jobs_of(api, repository, run["id"]), mode_job) != READING, 120, sleep)
            mode = mode_of(jobs_of(api, repository, run["id"]), mode_job)
            run = api(f"repos/{repository}/actions/runs/{run['id']}")
        what, why = action(run, mode)
        if what == LEAVE:
            say(f"{workflow}: {url}: {why}; left as it is")
            continue
        if what == CANCEL_AND_RERUN:
            api(f"repos/{repository}/actions/runs/{run['id']}/cancel", method="POST")
            ended = until(lambda: api(f"repos/{repository}/actions/runs/{run['id']}").get("status") == "completed",
                          300, sleep)
            if not ended:
                say(f"::error title=ready::{workflow}: {url} did not end within 5 minutes of its cancelling")
                failed = 1
                continue
        if head_of(api, repository, number) != head:
            say(f"{workflow}: the head moved past {head[:12]}; its push's run reads the label itself")
            continue
        api(f"repos/{repository}/actions/runs/{run['id']}/rerun", method="POST")
        say(f"{workflow}: {url}: {why}; re-run whole, and it reads the label `ready` as it starts")
    return failed


def say_and_record(line: str) -> None:
    """``line`` printed, and added to the run page's summary when there is one."""
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as page:
            page.write(f"- {line.removeprefix('::error title=ready::')}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--repository", required=True, help="OWNER/NAME")
    parser.add_argument("--pull-request", required=True, help="the pull request's number")
    parser.add_argument("--run", action="append", required=True, metavar="WORKFLOW=JOB",
                        help="a workflow's file and the name of its job that reads the labels (repeatable)")
    arguments = parser.parse_args(argv)
    runs = [tuple(each.split("=", 1)) for each in arguments.run]
    if any(len(each) != 2 or not all(each) for each in runs):
        parser.error("--run takes WORKFLOW=JOB")
    try:
        return ready(gh, arguments.repository, arguments.pull_request, [(w, j) for w, j in runs], say=say_and_record)
    except RuntimeError as error:
        say_and_record(f"::error title=ready::{error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
