"""The lane's read of the run's own jobs survives a server error and never turns a failed read into a pass.

On push run 42098189628 ``default (Python 3.14)`` failed with every shard passed: ``gh api`` answered ``gh: Server Error
(HTTP 502)`` and ``scripts/check_shard_jobs.py`` read that text as JSON and died on it. ``scripts/run_jobs.py`` now asks
again after a server error or an unreadable reply, fails at once on an error asking again cannot change, and, when no
read succeeds, exits 1 with one line naming the failure and prints no job. These tests run it against a stub ``gh`` that
answers a set sequence of replies. The controls are fixed plants, each failing on every run."""

import importlib.util
import json
import re
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_JOBS = ROOT / "scripts" / "run_jobs.py"
CHECK = ROOT / "scripts" / "check_shard_jobs.py"
CI = ROOT / ".github" / "workflows" / "ci.yml"
URL = "repos/o/r/actions/runs/1/jobs?filter=latest&per_page=100"
PREFIX = "default tests (Python 3.14, "
GOOD = "".join(json.dumps({"name": f"{PREFIX}shard {k} of 3)", "conclusion": "success"}) + "\n" for k in (1, 2, 3))
BAD_GATEWAY = (1, "", "gh: Server Error (HTTP 502)\n")
needs_posix = pytest.mark.skipif(sys.platform == "win32", reason="a CI Linux runner's script")


def stub_gh(folder: Path, replies: list[tuple[int, str, str]], rate_limit: str = "") -> dict[str, str]:
    """A ``gh`` that answers ``replies`` in turn (the last for every call after) and counts its calls, bar the call for
    ``rate_limit``, which answers ``rate_limit`` (headers) and is counted in ``rate_calls``."""
    folder.mkdir()
    (folder / "rate.out").write_text(rate_limit)
    for at, (code, out, err) in enumerate(replies, start=1):
        (folder / f"{at}.rc").write_text(str(code))
        (folder / f"{at}.out").write_text(out)
        (folder / f"{at}.err").write_text(err)
    gh = folder / "gh"
    gh.write_text(f'#!/bin/sh\ncase "$*" in *rate_limit*) echo x >> "{folder}/rate_calls"; cat "{folder}/rate.out"; exit 0;; esac\nn=$(( $(cat "{folder}/calls" 2>/dev/null || echo 0) + 1 ))\necho $n > "{folder}/calls"\n'
                  f'[ "$n" -gt {len(replies)} ] && n={len(replies)}\ncat "{folder}/$n.out"\ncat "{folder}/$n.err" >&2\n'
                  f'exit $(cat "{folder}/$n.rc")\n', encoding="utf-8")
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    return {"PATH": f"{folder}:/usr/bin:/bin"}


def calls(folder: Path) -> int:
    return int((folder / "calls").read_text())


def run_jobs(tmp_path: Path, replies, *arguments: str, rate_limit: str = "") -> subprocess.CompletedProcess[str]:
    environment = stub_gh(tmp_path / "gh", replies, rate_limit)
    return subprocess.run([sys.executable, str(RUN_JOBS), URL, "--pause", "0", *arguments], env=environment,
                          capture_output=True, text=True, check=False, timeout=60)


@needs_posix
def test_a_server_error_then_the_jobs_is_read_again_and_passes(tmp_path):
    done = run_jobs(tmp_path, [BAD_GATEWAY, (0, GOOD, "")])
    assert done.returncode == 0 and done.stdout == GOOD and calls(tmp_path / "gh") == 2, done


@needs_posix
def test_control_without_the_retry_the_same_server_error_fails_the_read(tmp_path):
    """The plant of the test above: one try is the code before this change. It must fail, every run."""
    done = run_jobs(tmp_path, [BAD_GATEWAY, (0, GOOD, "")], "--tries", "1")
    assert done.returncode == 1 and calls(tmp_path / "gh") == 1 and done.stdout.count("\n") == 1, done


@needs_posix
def test_a_reply_that_is_not_the_jobs_is_read_again(tmp_path):
    """Exit 0 with the error text on standard output: what the checker read as JSON."""
    done = run_jobs(tmp_path, [(0, "gh: Server Error (HTTP 502)\n", ""), (0, "<html>bad gateway</html>\n", ""), (0, GOOD, "")])
    assert done.returncode == 0 and done.stdout == GOOD and calls(tmp_path / "gh") == 3, done


@needs_posix
def test_a_server_error_that_never_clears_fails_naming_the_http_failure_and_prints_no_job(tmp_path):
    done = run_jobs(tmp_path, [BAD_GATEWAY])
    assert done.returncode == 1 and calls(tmp_path / "gh") == 5, done
    assert "HTTP 502" in done.stdout and "could not be read" in done.stdout and "after 5 tries" in done.stdout
    assert "Traceback" not in done.stdout + done.stderr
    assert [line for line in done.stdout.splitlines() if not line.startswith("::error")] == []


@needs_posix
def test_control_a_guard_that_let_a_failed_read_through_would_print_a_list_the_check_takes_for_no_shards(tmp_path):
    """The failed read's output goes into the check: it must fail the lane, not pass an empty list, under the step's
    ``pipefail`` or without it (the check alone sees no jobs and no shard)."""
    done = subprocess.run(f"{shlex.quote(sys.executable)} {shlex.quote(str(RUN_JOBS))} {shlex.quote(URL)} --pause 0 | "
                          f"{shlex.quote(sys.executable)} {shlex.quote(str(CHECK))} --prefix {shlex.quote(PREFIX)} --shards 3",
                          shell=True, executable="/bin/bash", env=stub_gh(tmp_path / "gh", [BAD_GATEWAY]),
                          capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode == 1 and "Traceback" not in done.stdout + done.stderr, done


@needs_posix
@pytest.mark.parametrize("reply", [(1, "", "gh: Not Found (HTTP 404)\n"), (1, "", "gh: Resource not accessible (HTTP 403)\n")],
                         ids=["not-found", "forbidden"])
def test_an_error_that_asking_again_cannot_change_fails_at_once(tmp_path, reply):
    done = run_jobs(tmp_path, [reply, (0, GOOD, "")])
    assert done.returncode == 1 and calls(tmp_path / "gh") == 1 and reply[2].strip()[4:30] in done.stdout, done


def _module():
    spec = importlib.util.spec_from_file_location("_run_jobs", RUN_JOBS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pause_before_each_new_read_doubles_from_the_one_given(monkeypatch):
    module = _module()
    answers = iter([subprocess.CompletedProcess([], 1, "", "HTTP 503")] * 3)
    monkeypatch.setattr(module, "read", lambda url, seconds=30.0: next(answers))
    waited: list[float] = []
    lines, why = module.fetch(URL, 3, 10, waited.append)
    assert lines is None and "HTTP 503" in why and waited == [10, 20], (why, waited)


def _fetch(monkeypatch, replies, *, tries=5, pause=10, **options):
    """``fetch`` against ``replies`` (the last repeating), with the waits recorded and not waited for."""
    module = _module()
    answers = iter(replies + [replies[-1]] * tries)
    monkeypatch.setattr(module, "read", lambda url, seconds=30.0: next(answers))
    headers = options.pop("headers", "")
    monkeypatch.setattr(module, "rate_limit_headers", lambda seconds=30.0: headers)
    waited: list[float] = []
    lines, why = module.fetch(URL, tries, pause, waited.append, **options)
    return lines, why, waited


def failed(code: int, text: str, *, out: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, out, text)


def test_the_defaults_are_five_tries_with_waits_of_ten_twenty_forty_and_sixty_seconds(monkeypatch):
    """Two server errors about 15 s apart were seen live; five tries wait 130 s in all, and a read of 30 s a try is cut:
    the worst case, 5 x 30 + 130 = 280 s, is the docstring's."""
    lines, why, waited = _fetch(monkeypatch, [failed(1, "gh: Server Error (HTTP 502)")])
    assert lines is None and waited == [10, 20, 40, 60], (why, waited)
    assert sum(waited) + 5 * 30 == 280


def test_two_server_errors_in_a_row_are_survived_with_room_to_spare(monkeypatch):
    """The 502, 502, then 200 the review saw 15 s apart: the third try reads."""
    good = subprocess.CompletedProcess([], 0, GOOD, "")
    lines, why, waited = _fetch(monkeypatch, [failed(1, "gh: Server Error (HTTP 502)")] * 2 + [good])
    assert lines == GOOD.splitlines() and waited == [10, 20], (why, waited)


@pytest.mark.parametrize("text", [
    "Get \"https://api.github.com/x\": net/http: TLS handshake timeout",
    "error connecting to api.github.com",
    "dial tcp: lookup api.github.com: no such host",
    "read tcp 10.0.0.1:443: connection reset by peer",
    "Get \"https://api.github.com/x\": context deadline exceeded (Client.Timeout exceeded while awaiting headers)",
    "gh api timed out after 30 s (timeout)",
    "gh: Request Timeout (HTTP 408)",
], ids=["tls-timeout", "cannot-connect", "no-such-host", "reset", "deadline", "cut-by-this-script", "http-408"])
def test_a_network_failure_with_no_http_status_is_read_again(monkeypatch, text):
    good = subprocess.CompletedProcess([], 0, GOOD, "")
    lines, why, waited = _fetch(monkeypatch, [failed(1, text), good])
    assert lines == GOOD.splitlines() and waited == [10], (text, why, waited)


@needs_posix
@pytest.mark.parametrize("reply", [
    (1, "", 'Get "https://api.github.com/x": net/http: TLS handshake timeout\n'),
    (1, "", "gh: Too Many Requests (HTTP 429)\n"),
    (1, "", "gh: API rate limit exceeded for user ID 1. (HTTP 403)\n"),
    (1, "", "gh: You have exceeded a secondary rate limit. (HTTP 403)\n"),
], ids=["network-timeout", "429", "403-rate-limit", "403-secondary"])
def test_a_timeout_a_429_and_a_rate_limit_403_are_read_again_through_the_script(tmp_path, reply):
    done = run_jobs(tmp_path, [reply, (0, GOOD, "")])
    assert done.returncode == 0 and done.stdout == GOOD and calls(tmp_path / "gh") == 2, done


def test_a_rate_limit_is_waited_for_as_long_as_it_asks_and_no_longer_than_the_cap(monkeypatch):
    good = subprocess.CompletedProcess([], 0, GOOD, "")
    text = "gh: API rate limit exceeded (HTTP 403) retry-after: 25"
    assert _fetch(monkeypatch, [failed(1, text), good])[2] == [25]
    assert _fetch(monkeypatch, [failed(1, text.replace("25", "3600")), good])[2] == [60], "capped"
    assert _fetch(monkeypatch, [failed(1, text.replace("25", "3600")), good], cap=30)[2] == [30]
    reset = "gh: API rate limit exceeded (HTTP 403) x-ratelimit-reset: 1000100"
    assert _fetch(monkeypatch, [failed(1, reset), good], now=lambda: 1000070.0)[2] == [30]
    assert _fetch(monkeypatch, [failed(1, reset), good], now=lambda: 1000200.0)[2] == [0], "a reset already past: no wait"


def test_a_rate_limit_whose_text_holds_no_wait_asks_the_free_rate_limit_call_for_the_reset(monkeypatch):
    good = subprocess.CompletedProcess([], 0, GOOD, "")
    lines, why, waited = _fetch(monkeypatch, [failed(1, "gh: API rate limit exceeded (HTTP 403)"), good],
                                headers="HTTP/2.0 200 OK\nX-Ratelimit-Reset: 1000045\n", now=lambda: 1000000.0)
    assert lines is not None and waited == [45], (why, waited)
    lines, why, waited = _fetch(monkeypatch, [failed(1, "gh: Too Many Requests (HTTP 429)"), good])
    assert lines is not None and waited == [10], "no header anywhere: the backoff"


def test_a_429_waits_what_it_says_in_place_of_the_backoff(monkeypatch):
    good = subprocess.CompletedProcess([], 0, GOOD, "")
    assert _fetch(monkeypatch, [failed(1, "gh: Too Many Requests (HTTP 429) Retry-After: 7"), good])[2] == [7]


@pytest.mark.parametrize("text", ["gh: Not Found (HTTP 404)", "gh: Bad credentials (HTTP 401)", "gh: Resource not accessible by integration (HTTP 403)",
                                  "gh: Validation Failed (HTTP 422)", "gh: Gone (HTTP 410)",
                                  "gh: To use GitHub CLI in a GitHub Actions workflow, set the GH_TOKEN environment variable.",
                                  "gh is not on the PATH (command not found); not logged in"],
                         ids=["404", "401", "403-no-rate-limit", "422", "410", "no-token", "no-gh"])
def test_any_other_client_error_and_a_gh_that_cannot_run_fail_at_once(monkeypatch, text):
    good = subprocess.CompletedProcess([], 0, GOOD, "")
    lines, why, waited = _fetch(monkeypatch, [failed(1, text), good])
    assert lines is None and waited == [] and why.startswith("gh api exited 1") and "after" not in why, (text, why, waited)


@needs_posix
@pytest.mark.parametrize("reply", [(0, "", ""), (0, "\n\n", ""), (0, "[]\n", ""), (0, '{"message": "Not Found"}\n', ""),
                                   (0, "[1, 2]\n", ""), (0, '{"name": "x"}\n', ""), (0, "1\n", "")],
                         ids=["empty", "blank-lines", "empty-list", "an-error-object", "a-list", "no-conclusion", "a-number"])
def test_a_reply_with_no_job_is_a_failed_read_with_a_message_and_prints_no_list(tmp_path, reply):
    """Exit 0 and nothing (or something that is no job) once held 'no jobs to check': it is asked again, and when it never
    changes the step fails, saying what came back."""
    done = run_jobs(tmp_path, [reply])
    assert done.returncode == 1 and calls(tmp_path / "gh") == 5, done
    assert "could not be read" in done.stdout and "something other than the jobs' lines" in done.stdout, done
    assert [line for line in done.stdout.splitlines() if not line.startswith("::error")] == []


@needs_posix
def test_an_empty_reply_that_clears_is_read_again_and_passes(tmp_path):
    done = run_jobs(tmp_path, [(0, "", ""), (0, GOOD, "")])
    assert done.returncode == 0 and done.stdout == GOOD and calls(tmp_path / "gh") == 2, done


@needs_posix
def test_the_defaults_of_the_script_are_the_ones_the_docstring_gives(tmp_path):
    """Run with no option against a ``gh`` that always answers 502, with ``time.sleep`` replaced by a recorder: five calls
    and the waits 10, 20, 40 and 60 s."""
    folder = tmp_path / "gh"
    environment = stub_gh(folder, [BAD_GATEWAY])
    code = ("import runpy, sys, time\n"
            f"sys.argv = ['run_jobs.py', {URL!r}]\n"
            "time.sleep = lambda seconds: print('sleep', seconds, file=sys.stderr)\n"
            f"runpy.run_path({str(RUN_JOBS)!r}, run_name='__main__')\n")
    done = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode == 1 and calls(folder) == 5, done
    assert re.findall(r"sleep ([0-9.]+)", done.stderr) == ["10.0", "20.0", "40.0", "60.0"], done.stderr


@needs_posix
def test_a_gh_call_that_hangs_is_cut_and_read_again(tmp_path):
    folder = tmp_path / "gh"
    folder.mkdir()
    (folder / "gh").write_text(f'#!/bin/sh\nn=$(( $(cat "{folder}/calls" 2>/dev/null || echo 0) + 1 ))\necho $n > "{folder}/calls"\n'
                               f'[ "$n" -ge 2 ] && {{ cat <<\'EOF\'\n{GOOD}EOF\nexit 0; }}\nexec sleep 30\n', encoding="utf-8")
    (folder / "gh").chmod(0o755)
    started = time.monotonic()
    done = subprocess.run([sys.executable, str(RUN_JOBS), URL, "--pause", "0", "--call-seconds", "1"],
                          env={"PATH": f"{folder}:/usr/bin:/bin"}, capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode == 0 and done.stdout == GOOD and calls(folder) == 2, done
    assert time.monotonic() - started < 15, "the hung call (30 s) was waited for, not cut at 1 s"


@needs_posix
def test_a_gh_that_is_not_there_fails_at_once_with_a_message_not_a_traceback(tmp_path):
    done = subprocess.run([sys.executable, str(RUN_JOBS), URL, "--pause", "0"], env={"PATH": str(tmp_path)},
                          capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode == 1 and "could not be read" in done.stdout and "Traceback" not in done.stdout + done.stderr, done
    assert "after" not in done.stdout


def test_the_check_names_a_line_that_is_not_a_job_and_does_not_pass_it():
    done = subprocess.run([sys.executable, str(CHECK), "--prefix", PREFIX, "--shards", "3"], input="gh: Server Error (HTTP 502)\n",
                          capture_output=True, text=True, check=False)
    assert done.returncode == 1 and "not a job" in done.stdout and "no shard was checked" in done.stdout, done
    assert "Traceback" not in done.stderr


def test_the_check_still_passes_the_jobs_it_is_given():
    done = subprocess.run([sys.executable, str(CHECK), "--prefix", PREFIX, "--shards", "3"], input=GOOD,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done


def wiring_problems(text: str) -> list[str]:
    """Every step of ci.yml that runs the shard-jobs check reads the jobs through ``scripts/run_jobs.py`` under
    ``pipefail``, and no step reads them with a bare ``gh api``."""
    found = []
    for job_id, job in yaml.safe_load(text)["jobs"].items():
        for step in job.get("steps") or []:
            run = str(step.get("run", ""))
            if "check_shard_jobs.py" in run:
                if "python3 scripts/run_jobs.py" not in run or not re.search(r"set -e?o pipefail", run):
                    found.append(f"{job_id}: the step {step.get('name')!r} does not read the jobs by scripts/run_jobs.py under pipefail")
            if "gh api" in run and "jobs?filter=latest" in str(step.get("env", "")):
                found.append(f"{job_id}: the step {step.get('name')!r} reads the jobs with a bare gh api")
    return found


def test_every_read_of_the_jobs_in_ci_goes_through_the_retrying_script():
    text = CI.read_text(encoding="utf-8")
    assert wiring_problems(text) == []
    assert text.count("python3 scripts/run_jobs.py") == 4, "the default lane's shards and completion job, and the other two lanes' completion job"


@pytest.mark.parametrize("old, new", [
    ('python3 scripts/run_jobs.py "$JOBS_URL"', 'gh api "$JOBS_URL" --paginate --jq \'.jobs[] | {name, conclusion}\''),
    ("          set -o pipefail\n", ""),
    ("          set -eo pipefail\n", ""),
], ids=["a-bare-gh-api", "no-pipefail", "no-pipefail-in-the-completion-step"])
def test_control_a_read_of_the_jobs_without_the_script_or_pipefail_is_named(old, new):
    text = CI.read_text(encoding="utf-8")
    assert old in text
    assert wiring_problems(text.replace(old, new)) != []
