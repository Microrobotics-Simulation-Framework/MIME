#!/usr/bin/env python
"""The run's own jobs at their latest attempts, one JSON object ``{"name", "conclusion"}`` per line on standard output:
the read the lane's job pipes into ``scripts/check_shard_jobs.py`` (``.github/workflows/ci.yml``).

    python3 scripts/run_jobs.py "repos/<owner>/<repo>/actions/runs/<id>/jobs?filter=latest&per_page=100" | \\
        python3 scripts/check_shard_jobs.py --prefix "default tests (Python 3.12, " --shards 10

``gh api`` answers a server error now and then (push run 42098189628: ``gh: Server Error (HTTP 502)``, which the checker
then read as JSON and died on, failing a required check whose shards had all passed). So a server error (HTTP 5xx) or a
reply that is not the lines asked for is asked again, ``--tries`` times in all with ``--pause`` seconds doubling between
them; an error that asking again cannot change (401, 403, 404) fails at once. A read that never succeeds exits 1 with
one line naming the failure, and prints nothing to standard output: it is never an empty list that a check could take
for a lane with no shards. Needs ``gh`` on the PATH and ``GH_TOKEN``."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable

SERVER_ERROR = re.compile(r"HTTP 5\d\d")
QUERY = ".jobs[] | {name, conclusion}"


def read(url: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["gh", "api", url, "--paginate", "--jq", QUERY], capture_output=True, text=True, check=False)


def lines_of(text: str) -> list[str] | None:
    """The lines of ``text`` if every one is a JSON object with a name and a conclusion, else ``None``."""
    lines = [line for line in text.splitlines() if line.strip()]
    try:
        parsed = [json.loads(line) for line in lines]
    except ValueError:
        return None
    return lines if all(isinstance(each, dict) and {"name", "conclusion"} <= each.keys() for each in parsed) else None


def fetch(url: str, tries: int, pause: float, sleep: Callable[[float], None] = time.sleep) -> tuple[list[str] | None, str]:
    """The jobs' lines, or ``None`` and what went wrong on the last try."""
    why = "no attempt was made"
    for attempt in range(1, tries + 1):
        done = read(url)
        found = lines_of(done.stdout) if done.returncode == 0 else None
        if found is not None:
            return found, ""
        text = " ".join((done.stderr + " " + done.stdout).split())[:300]
        why = (f"gh api exited {done.returncode}: {text}" if done.returncode != 0
               else f"gh api answered something other than the jobs' lines: {text}")
        retryable = done.returncode == 0 or bool(SERVER_ERROR.search(text))
        if not retryable or attempt == tries:
            return None, why + (f" (after {attempt} tries)" if attempt > 1 else "")
        sleep(pause * 2 ** (attempt - 1))
    return None, why


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("url", help="the run's jobs endpoint, without the host")
    parser.add_argument("--tries", type=int, default=3, help="reads before giving up (default 3)")
    parser.add_argument("--pause", type=float, default=10.0, help="seconds before the second read, doubling (default 10)")
    arguments = parser.parse_args(argv)
    if arguments.tries < 1:
        parser.error("--tries must be at least 1")
    lines, why = fetch(arguments.url, arguments.tries, arguments.pause)
    if lines is None:
        print(f"::error title=Run jobs::the run's jobs could not be read ({why}); no shard was checked")
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
