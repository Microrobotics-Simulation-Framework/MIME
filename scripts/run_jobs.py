#!/usr/bin/env python
"""The run's own jobs at their latest attempts, one JSON object ``{"name", "conclusion"}`` per line on standard output:
the read the lane's job pipes into ``scripts/check_shard_jobs.py`` (``.github/workflows/ci.yml``).

    python3 scripts/run_jobs.py "repos/<owner>/<repo>/actions/runs/<id>/jobs?filter=latest&per_page=100" | \\
        python3 scripts/check_shard_jobs.py --prefix "default tests (Python 3.12, " --shards 10

``gh api`` answers a server error now and then (push run 42098189628: ``gh: Server Error (HTTP 502)``, twice running
about 15 s apart, which the checker then read as JSON and died on, failing a required check whose shards had all
passed). So a read that can succeed when asked again is asked again, ``--tries`` times in all (5), with ``--pause``
seconds (10) doubling between them and never above ``--max-wait`` (60):
- a server error (HTTP 5xx), a request timeout (408) and a failure with no HTTP status at all (a network timeout, a
  refused or reset connection, a name that does not resolve, a ``gh`` call cut at ``--call-seconds``);
- a rate limit: 429, or 403 whose text says so. It waits what GitHub asks, from ``Retry-After`` or
  ``X-RateLimit-Reset`` (in the error's text, else from the free ``rate_limit`` call), at most ``--max-wait``;
- a reply that is not the jobs asked for, an empty one included (exit 0 and no job: a lane has jobs, so none is a
  failed read, never "no jobs to check").
Any other 4xx (401, 403 that is no rate limit, 404, 422) and a ``gh`` that is missing or has no token fail at once.
The worst case is the tries' calls (``--call-seconds`` each) and the waits between them: 5 x 30 s + (10 + 20 + 40 +
60) s = 280 s, and a rate limit's probe of the headers adds up to 30 s to each wait (400 s); the lane job's limit is
10 minutes (a cut job is red, never green). A read that never succeeds exits 1 with one line naming the failure, and prints nothing else to
standard output: it is never an empty list that a check could take for a lane with no shards. Needs ``gh`` on the PATH
and ``GH_TOKEN``."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable

HTTP_STATUS = re.compile(r"HTTP (\d{3})")
RATE_LIMITED = re.compile(r"rate limit|retry-after|x-ratelimit-remaining: 0|abuse", re.IGNORECASE)
RETRY_AFTER = re.compile(r"retry-after[:=]\s*(\d+)", re.IGNORECASE)
RESET_AT = re.compile(r"x-ratelimit-reset[:=]\s*(\d+)", re.IGNORECASE)
#: Failures with no HTTP status that asking again cannot change: no token, not logged in, a flag the ``gh`` lacks.
HOPELESS = re.compile(r"GH_TOKEN|gh auth login|not logged in|unknown (?:flag|command)|accepts \d+ arg", re.IGNORECASE)
QUERY = ".jobs[] | {name, conclusion}"


def gh(*arguments: str, seconds: float) -> subprocess.CompletedProcess[str]:
    """``gh`` with ``arguments``: a call cut at ``seconds`` or a ``gh`` that is not there is a failed process, not a traceback."""
    try:
        return subprocess.run(["gh", *arguments], capture_output=True, text=True, check=False, timeout=seconds)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(["gh", *arguments], 124, "", f"gh api timed out after {seconds:g} s (timeout)")
    except FileNotFoundError:
        return subprocess.CompletedProcess(["gh", *arguments], 127, "", "gh is not on the PATH (command not found); not logged in")


def read(url: str, seconds: float = 30.0) -> subprocess.CompletedProcess[str]:
    return gh("api", url, "--paginate", "--jq", QUERY, seconds=seconds)


def rate_limit_headers(seconds: float = 30.0) -> str:
    """The headers of the ``rate_limit`` call, which GitHub does not count against the limit."""
    done = gh("api", "rate_limit", "--include", seconds=seconds)
    return done.stdout if done.returncode == 0 else ""


def lines_of(text: str) -> list[str] | None:
    """The lines of ``text`` if there is at least one and every one is a JSON object with a name and a conclusion,
    else ``None`` (a reply with no job is not the jobs of a lane)."""
    lines = [line for line in text.splitlines() if line.strip()]
    try:
        parsed = [json.loads(line) for line in lines]
    except ValueError:
        return None
    ok = bool(lines) and all(isinstance(each, dict) and {"name", "conclusion"} <= each.keys() for each in parsed)
    return lines if ok else None


def classify(done: subprocess.CompletedProcess[str], text: str) -> str:
    """What to do about a read that did not give the jobs: ``"retry"``, ``"rate limit"`` (retry, after what GitHub asks) or ``"stop"``."""
    if done.returncode == 0:
        return "retry"  # exit 0, but not the jobs: an error text on standard output, an HTML page, nothing at all
    status = HTTP_STATUS.search(text)
    if status is None:
        return "stop" if HOPELESS.search(text) else "retry"  # a network failure: no status came back
    code = int(status.group(1))
    if code == 429 or (code == 403 and RATE_LIMITED.search(text)):
        return "rate limit"
    return "retry" if code >= 500 or code == 408 else "stop"


def wait_for(text: str, now: float, backoff: float, cap: float) -> float:
    """Seconds to wait before the next read: what a rate limit's ``Retry-After`` or ``X-RateLimit-Reset`` (an epoch
    second) in ``text`` asks, else the backoff; never above ``cap``."""
    asked, reset = RETRY_AFTER.search(text), RESET_AT.search(text)
    wanted = float(asked.group(1)) if asked else max(float(reset.group(1)) - now, 0.0) if reset else backoff
    return min(wanted, cap)


def fetch(url: str, tries: int, pause: float, sleep: Callable[[float], None] = time.sleep, *, cap: float = 60.0,
          call_seconds: float = 30.0, now: Callable[[], float] = time.time) -> tuple[list[str] | None, str]:
    """The jobs' lines, or ``None`` and what went wrong on the last try."""
    why = "no attempt was made"
    for attempt in range(1, tries + 1):
        done = read(url, call_seconds)
        found = lines_of(done.stdout) if done.returncode == 0 else None
        if found is not None:
            return found, ""
        text = " ".join((done.stderr + " " + done.stdout).split())[:300]
        why = (f"gh api exited {done.returncode}: {text}" if done.returncode != 0
               else f"gh api answered something other than the jobs' lines ({'nothing' if not text else text})")
        verdict = classify(done, text)
        if verdict == "stop" or attempt == tries:
            return None, why + (f" (after {attempt} tries)" if attempt > 1 else "")
        backoff = min(pause * 2 ** (attempt - 1), cap)
        if verdict == "rate limit":
            hint = text if RETRY_AFTER.search(text) or RESET_AT.search(text) else rate_limit_headers(call_seconds)
            sleep(wait_for(hint, now(), backoff, cap))
        else:
            sleep(backoff)
    return None, why


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("url", help="the run's jobs endpoint, without the host")
    parser.add_argument("--tries", type=int, default=5, help="reads before giving up (default 5)")
    parser.add_argument("--pause", type=float, default=10.0, help="seconds before the second read, doubling (default 10)")
    parser.add_argument("--max-wait", type=float, default=60.0, help="the longest wait between two reads, a rate limit's included (default 60)")
    parser.add_argument("--call-seconds", type=float, default=30.0, help="the longest one gh call may take (default 30)")
    arguments = parser.parse_args(argv)
    if arguments.tries < 1:
        parser.error("--tries must be at least 1")
    lines, why = fetch(arguments.url, arguments.tries, arguments.pause, cap=arguments.max_wait, call_seconds=arguments.call_seconds)
    if lines is None:
        print(f"::error title=Run jobs::the run's jobs could not be read ({why}); no shard was checked")
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
