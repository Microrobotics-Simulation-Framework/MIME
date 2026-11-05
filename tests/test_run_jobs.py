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


def stub_gh(folder: Path, replies: list[tuple[int, str, str]]) -> dict[str, str]:
    """A ``gh`` that answers ``replies`` in turn (the last for every call after) and counts its calls."""
    folder.mkdir()
    for at, (code, out, err) in enumerate(replies, start=1):
        (folder / f"{at}.rc").write_text(str(code))
        (folder / f"{at}.out").write_text(out)
        (folder / f"{at}.err").write_text(err)
    gh = folder / "gh"
    gh.write_text(f'#!/bin/sh\nn=$(( $(cat "{folder}/calls" 2>/dev/null || echo 0) + 1 ))\necho $n > "{folder}/calls"\n'
                  f'[ "$n" -gt {len(replies)} ] && n={len(replies)}\ncat "{folder}/$n.out"\ncat "{folder}/$n.err" >&2\n'
                  f'exit $(cat "{folder}/$n.rc")\n', encoding="utf-8")
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    return {"PATH": f"{folder}:/usr/bin:/bin"}


def calls(folder: Path) -> int:
    return int((folder / "calls").read_text())


def run_jobs(tmp_path: Path, replies, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = stub_gh(tmp_path / "gh", replies)
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
    assert done.returncode == 1 and calls(tmp_path / "gh") == 3, done
    assert "HTTP 502" in done.stdout and "could not be read" in done.stdout and "after 3 tries" in done.stdout
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
    monkeypatch.setattr(module, "read", lambda url: next(answers))
    waited: list[float] = []
    lines, why = module.fetch(URL, 3, 10, waited.append)
    assert lines is None and "HTTP 503" in why and waited == [10, 20], (why, waited)


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
    assert text.count("python3 scripts/run_jobs.py") == 2


@pytest.mark.parametrize("old, new", [
    ('python3 scripts/run_jobs.py "$JOBS_URL"', 'gh api "$JOBS_URL" --paginate --jq \'.jobs[] | {name, conclusion}\''),
    ("          set -o pipefail\n", ""),
    ("          set -eo pipefail\n", ""),
], ids=["a-bare-gh-api", "no-pipefail", "no-pipefail-in-the-completion-step"])
def test_control_a_read_of_the_jobs_without_the_script_or_pipefail_is_named(old, new):
    text = CI.read_text(encoding="utf-8")
    assert old in text
    assert wiring_problems(text.replace(old, new)) != []
