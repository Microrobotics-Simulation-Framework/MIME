#!/usr/bin/env python
"""A lane's shard jobs, each at its latest attempt, all passed: run by the ``default`` lane's job after
``scripts/latest_attempt.py`` (``.github/workflows/ci.yml``).

``scripts/latest_attempt.py`` keeps the newest uploaded report of each shard. A shard job that was re-run and failed
before it uploaded one leaves its first attempt's report as the newest, which may have passed. The run's own list of
jobs at their latest attempts (``GET /repos/{repo}/actions/runs/{id}/jobs?filter=latest``, read by
``scripts/run_jobs.py``, which asks again after a server error) says what the shard jobs ended as, so this reads it,
one JSON object ``{"name", "conclusion"}`` per line on standard input, and checks that exactly ``--shards`` jobs have a
name beginning ``--prefix`` and that each concluded ``success``. The prefix names one Python (``default tests (Python 3.12, ``): the matrix job's own result is one for all three, and would turn a lane red
for another Python's shard.

    python3 scripts/run_jobs.py ".../jobs?filter=latest" \\
        | python3 scripts/check_shard_jobs.py --prefix "default tests (Python 3.12, " --shards 10

Exits 0 when they all passed, 1 naming each that did not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable


def problems(lines: Iterable[str], prefix: str, shards: int) -> list[str]:
    jobs: list[tuple[str, str | None]] = []
    for line in lines:
        if line.strip():
            try:
                job = json.loads(line)
                job["name"], job["conclusion"]
            except (ValueError, KeyError, TypeError):
                return [f"the jobs' list holds a line that is not a job ({line.strip()[:200]!r}); no shard was checked"]
            if str(job["name"]).startswith(prefix):
                jobs.append((job["name"], job["conclusion"]))
    found = [f"{name}: its latest attempt concluded {conclusion!r}" for name, conclusion in jobs
             if conclusion != "success"]
    if len(jobs) != shards:
        found.append(f"{len(jobs)} shard jobs begin {prefix!r}; this lane runs {shards}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prefix", required=True, help="what the lane's shard jobs' names begin with")
    parser.add_argument("--shards", type=int, required=True, help="how many shards the lane runs")
    arguments = parser.parse_args(argv)
    found = problems(sys.stdin, arguments.prefix, arguments.shards)
    for problem in found:
        print(f"::error title=Shards::{problem}")
    if not found:
        print(f"{arguments.shards} shard jobs, each at its latest attempt, passed")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
