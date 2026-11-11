"""What adding the label ``ready`` does to a pull request's runs (``scripts/ci_ready.py``, run by
``.github/workflows/ready.yml``): the head's newest run of each workflow that reads the label is re-run whole unless it
is full already or will read the label itself, a reduced run in flight is cancelled first, and a head that moved is
left to its own push's run. The GitHub API is a fake that keeps the runs, their jobs and the pull request's head, and
records every request that changes something; ``tests/test_ci_ready_label.py`` reads the workflows' side."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_ci_ready", ROOT / "scripts" / "ci_ready.py")
assert spec is not None and spec.loader is not None
ci_ready = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_ready)

REPOSITORY = "an-owner/a-repository"
RUNS = [("ci.yml", "shard count"), ("platforms.yml", "the run's mode")]
HEAD = "a" * 40


def mode_job(name: str, mode: str) -> dict[str, Any]:
    """The job that reads the labels, as the jobs API lists it: ``unread`` (not started), ``reading``, or ended
    ``full`` or ``reduced``."""
    if mode == "unread":
        return {"name": name, "status": "queued", "steps": []}
    read = {"name": "The run's mode, from the pull request's labels as they are now", "started_at": "2026-11-03T12:00:00Z",
            "status": "in_progress" if mode == "reading" else "completed",
            "conclusion": None if mode == "reading" else "success"}
    marker = {"name": ci_ready.MARKER, "status": "queued" if mode == "reading" else "completed",
              "conclusion": {"full": "success", "reduced": "skipped"}.get(mode)}
    return {"name": name, "status": "in_progress" if mode == "reading" else "completed", "steps": [read, marker]}


class Api:
    """The GitHub API, faked: one run per workflow on the head, its mode, and what each POST does."""

    def __init__(self, runs: dict[str, dict[str, Any]], heads: list[str] | None = None, cancel_ends: bool = True,
                 settles_after: int = 0) -> None:
        self.runs = runs
        self.heads = heads or [HEAD]
        self.cancel_ends = cancel_ends
        self.settles_after = settles_after
        self.posts: list[str] = []
        self.reads = 0

    def __call__(self, path: str, method: str = "GET") -> Any:
        if method == "POST":
            self.posts.append(path)
            run = next(run for run in self.runs.values() if f"/runs/{run['id']}/" in path)
            if path.endswith("/cancel") and self.cancel_ends:
                run.update(status="completed", conclusion="cancelled")
            return None
        if path.endswith("/pulls/30"):
            head = self.heads[0] if len(self.heads) == 1 else self.heads.pop(0)
            return {"head": {"sha": head}}
        if "/actions/workflows/" in path:
            workflow = path.split("/actions/workflows/")[1].split("/")[0]
            run = self.runs.get(workflow)
            return {"workflow_runs": [run] if run and f"head_sha={run['head_sha']}" in path else []}
        if "/jobs?filter=latest&per_page=50&page=1" in path:
            run = next(run for run in self.runs.values() if f"/runs/{run['id']}/jobs" in path)
            self.reads += 1
            mode = run["mode"]
            if mode == "reading" and self.reads > self.settles_after:
                mode = run["settles_to"]
            return {"jobs": [mode_job(run["mode_job"], mode)]}
        run = next(run for run in self.runs.values() if path.endswith(f"/runs/{run['id']}"))
        return dict(run)


def run(workflow: str, status: str, conclusion: str | None, mode: str, **more: Any) -> dict[str, Any]:
    mode_job_name = dict(RUNS)[workflow]
    number = 100 if workflow == "ci.yml" else 200
    return {"id": number, "head_sha": HEAD, "status": status, "conclusion": conclusion, "mode": mode,
            "mode_job": mode_job_name, "html_url": f"https://example.invalid/runs/{number}",
            "created_at": "2026-11-03T12:00:00Z", **more}


def ready(api: Api) -> tuple[int, list[str]]:
    said: list[str] = []
    return ci_ready.ready(api, REPOSITORY, "30", RUNS, sleep=lambda _: None, say=said.append), said


@pytest.mark.parametrize("status, conclusion, mode, posts", [
    ("completed", "failure", "reduced", ["rerun"]),
    ("completed", "success", "reduced", ["rerun"]),
    ("completed", "success", "full", []),
    ("completed", "failure", "full", ["rerun"]),
    ("completed", "cancelled", "unread", ["rerun"]),
    ("in_progress", None, "unread", []),
    ("queued", None, "unread", []),
    ("in_progress", None, "full", []),
    ("in_progress", None, "reduced", ["cancel", "rerun"]),
])
def test_each_run_is_left_or_re_run_by_its_state_and_mode(status, conclusion, mode, posts):
    api = Api({"ci.yml": run("ci.yml", status, conclusion, mode)})
    status_code, said = ready(api)
    assert status_code == 0, said
    assert [path.rsplit("/", 1)[1] for path in api.posts] == posts, said
    assert all("/runs/100/" in path for path in api.posts)


def test_both_workflows_are_read_and_a_missing_run_is_left_to_its_push():
    api = Api({"ci.yml": run("ci.yml", "completed", "failure", "reduced")})
    status, said = ready(api)
    assert status == 0 and api.posts == [f"repos/{REPOSITORY}/actions/runs/100/rerun"]
    assert any(line.startswith("platforms.yml: no run on the head") for line in said), said
    api = Api({"ci.yml": run("ci.yml", "completed", "failure", "reduced"),
               "platforms.yml": run("platforms.yml", "completed", "success", "reduced")})
    assert ready(api)[0] == 0
    assert api.posts == [f"repos/{REPOSITORY}/actions/runs/100/rerun", f"repos/{REPOSITORY}/actions/runs/200/rerun"]


def test_a_run_reading_the_labels_is_waited_for():
    """Its mode step started before the label may have been added: what it read decides."""
    for settles_to, posts in (("reduced", ["cancel", "rerun"]), ("full", [])):
        api = Api({"ci.yml": run("ci.yml", "in_progress", None, "reading", settles_to=settles_to)}, settles_after=3)
        assert ready(api)[0] == 0
        assert [path.rsplit("/", 1)[1] for path in api.posts] == posts


def test_a_head_that_moved_is_left_to_its_own_run():
    api = Api({"ci.yml": run("ci.yml", "completed", "failure", "reduced")}, heads=[HEAD, "b" * 40])
    status, said = ready(api)
    assert status == 0 and api.posts == [], said
    assert any("the head moved" in line for line in said)


def test_a_cancel_that_does_not_end_fails_and_re_runs_nothing():
    api = Api({"ci.yml": run("ci.yml", "in_progress", None, "reduced")}, cancel_ends=False)
    status, said = ready(api)
    assert status == 1 and [path.rsplit("/", 1)[1] for path in api.posts] == ["cancel"], said


def test_the_newest_run_on_the_head_is_the_one_read():
    older = {"id": 7, "created_at": "2026-11-03T11:00:00Z"}
    newer = {"id": 9, "created_at": "2026-11-03T12:00:00Z"}
    same_time = {"id": 8, "created_at": "2026-11-03T12:00:00Z"}
    assert ci_ready.newest_run(lambda _: {"workflow_runs": [older, same_time, newer]}, REPOSITORY, "ci.yml", HEAD) == newer
    assert ci_ready.newest_run(lambda _: {"workflow_runs": []}, REPOSITORY, "ci.yml", HEAD) is None


@pytest.mark.parametrize("mode", ["unread", "reading", "full", "reduced"])
def test_the_mode_is_read_off_the_marker_step(mode):
    assert ci_ready.mode_of([{"name": "other"}, mode_job("shard count", mode)], "shard count") == mode
    assert ci_ready.mode_of([mode_job("another job", mode)], "shard count") == "unread"


def test_a_failed_request_fails_the_job(monkeypatch):
    def failing(path: str, method: str = "GET") -> Any:
        raise RuntimeError(f"gh api --method {method} {path}: HTTP 403: Resource not accessible by integration")
    monkeypatch.setattr(ci_ready, "gh", failing)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)  # CI's own run page is not the script's
    assert ci_ready.main(["--repository", REPOSITORY, "--pull-request", "30", "--run", "ci.yml=shard count"]) == 1
    with pytest.raises(SystemExit):
        ci_ready.main(["--repository", REPOSITORY, "--pull-request", "30", "--run", "ci.yml"])


def test_the_run_page_says_what_was_done(monkeypatch, tmp_path):
    """Each line goes to the step summary too, an error without its workflow command."""
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    api = Api({"ci.yml": run("ci.yml", "completed", "failure", "reduced")})
    monkeypatch.setattr(ci_ready, "gh", api)
    assert ci_ready.main(["--repository", REPOSITORY, "--pull-request", "30", "--run", "ci.yml=shard count",
                          "--run", "platforms.yml=the run's mode"]) == 0
    lines = summary.read_text(encoding="utf-8").splitlines()
    assert lines == ["- ci.yml: https://example.invalid/runs/100: ended in the reduced set; re-run whole, and it reads "
                     "the label `ready` as it starts",
                     "- platforms.yml: no run on the head aaaaaaaaaaaa yet; the run its push starts reads the label"]

    def failing(path: str, method: str = "GET") -> Any:
        raise RuntimeError("gh api --method GET x: HTTP 403")
    monkeypatch.setattr(ci_ready, "gh", failing)
    assert ci_ready.main(["--repository", REPOSITORY, "--pull-request", "30", "--run", "ci.yml=shard count"]) == 1
    assert summary.read_text(encoding="utf-8").splitlines()[-1] == "- gh api --method GET x: HTTP 403"


def test_a_run_of_more_than_one_page_of_jobs_is_read_whole_at_fifty_a_request():
    """``jobs_of`` pages through ``scripts/run_jobs.py``: a run of 94 jobs (run 42921435109) is two requests of 50, and
    the job that reads the labels, on the second page, is found."""
    every = [{"name": f"job {number}", "status": "completed", "conclusion": "success"} for number in range(93)]
    every.append(mode_job("shard count", "full"))
    asked: list[str] = []

    def api(path: str, method: str = "GET") -> Any:
        asked.append(path)
        page = int(path.rsplit("page=", 1)[1])
        return {"total_count": 94, "jobs": every[(page - 1) * 50:page * 50]}

    jobs = ci_ready.jobs_of(api, REPOSITORY, 7)
    assert len(jobs) == 94 and ci_ready.mode_of(jobs, "shard count") == ci_ready.FULL
    assert asked == [f"repos/{REPOSITORY}/actions/runs/7/jobs?filter=latest&per_page=50&page={page}" for page in (1, 2)]
