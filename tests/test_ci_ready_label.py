"""CI runs the full matrix only on a pull request's final head, the one that carries the label ``ready``, and a required
check never passes by being skipped (``.github/workflows/ci.yml``, ``platforms.yml`` and ``ready.yml``,
``scripts/ci_ready.py``, CONTRIBUTING.md, "CI: the ``ready`` label").

The workflows are parsed, and each run is played through them: which jobs start (a job's ``if`` evaluated as GitHub
does for names, string literals (compared without case), numbers (a string compared with one is read as a number),
``==``, ``!=``, ``&&``, ``||``, ``!``, ``contains``, ``fromJSON`` and the status functions; a job after a skipped one it
needs is skipped unless its own ``if`` names a status function; a matrix expanded with its ``exclude``), the run's mode
(the mode step's script run in ``bash -e``, ``gh`` stubbed to answer the labels the run reads), and what each started
summary job's steps do (the steps that run, the refusing step's script run in ``bash -e``, and a failed step's
``continue-on-error`` honoured, so the job's own result is what is judged). What this shows:

- **Full** (a push, a manual run, a pull request whose labels hold ``ready`` when the run starts): 63 jobs for a push, 64
  for a manual run (``usd-core-newest``), the required checks among them, none refusing. A full pull request runs the
  lanes no longer required (``default`` on Python 3.13, ``maddening-main``, ``openmaterica-main``) only when its diff
  touches the pins, CI or an adapter (``scripts/ci_boundary.py``, ``tests/test_ci_boundary.py``): 27 jobs when it
  touches none of them, 63 when it touches the pins or CI, 39 when it touches one upstream's adapter alone.
- **Reduced** (a pull request without ``ready``): 16 jobs, the typecheck, the confidentiality scan, the docs check and
  ``default`` on Python 3.12 with all its shards and its completion tests job. The one required job that did not run its
  lane (``default`` on 3.14) starts, runs nothing but the refusing step, and fails with its message. Every required name
  starts in both modes: no required job's ``if`` can skip it. The lanes no longer required have no job in a reduced run.
- **The mode is read when a run starts**, never from the event: a re-run (attempt 2) of a reduced run after ``ready``
  was added runs in full, a re-run without the label refuses again, and a re-run of the failed jobs alone keeps the
  first attempt's mode, and refuses.
- **No label starts a run of ci.yml or platforms.yml**: neither listens to ``labeled``. ``ready.yml`` does; its one job
  runs for ``ready`` alone (in any case), bears no required name, has a group of its own for any other label, and
  re-runs every workflow that reads the label (``scripts/ci_ready.py``, ``tests/test_ci_ready.py``).

Why the label run had to go: a run that a label other than ``ready`` started, every job skipped, was the newest of
ci.yml on the head of the pull request that introduced the label, and the pull request read BLOCKED with every
required check green; a full run of the same head after it read CLEAN, and another skipped run after that BLOCKED
again (the runs are listed in that pull request). The model plays the workflows; GitHub's own rule is read off those
runs.

The model is ``test_ci_wiring.py``'s: a function returns the workflows' problems, a test requires the real files to
have none, and a parametrised test plants one break in a copy and requires the named problem. The workflows and the
plants are fixed, so each plant is reported on every run (there is no detection rate to measure)."""

import itertools
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_ci_wiring import REQUIRED

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
MESSAGE = "the full matrix has not run on this head: add the `ready` label"
#: Where the full matrix and the reduced set stand, counted at the default ten shards.
SHARDS = 10
REFUSING = {"default (Python 3.14)"}
#: The lanes no longer required: they start on a push and a manual run, and on a full pull request by its diff.
MOVED = {"default (Python 3.13)", "maddening-main", "openmaterica-main"}
NO_LANES: frozenset[str] = frozenset()
ALL_LANES = frozenset(MOVED)
REFUSAL_IF = "${{ env.MIME_CI_RUNS != 'true' }}"
#: The workflows that read the label when a run starts, and the job that reads it (``ready.yml``'s ``--run``).
READERS = {"ci.yml": "shards", "platforms.yml": "mode"}
#: The step of the count job that says which lanes a run starts, and the files it outputs.
BOUNDARY_OUTPUTS = ("maddening", "openmaterica", "exclude", "lane_exclude")
MODE_FILTER = 'any(.labels[].name; ascii_downcase == "ready")'
MARKER = "The full matrix runs"
STATUS = re.compile(r"cancelled\(\)|always\(\)|failure\(\)")
FUNCTIONS = {"contains": "_contains", "fromJSON": "_json", "cancelled": "_false", "success": "_true",
             "always": "_true", "failure": "_false"}
TOKEN = re.compile(r"'(?:[^']|'')*'|\d+|[A-Za-z_][\w.*-]*|==|!=|&&|\|\||[()!,]|\s+")
#: ``gh`` for the mode step: it answers the one filter the step must ask with the labels the run reads, and fails on
#: anything else.
GH_STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_LOG"
[ -n "$STUB_FAIL" ] && { echo "gh: HTTP 502" >&2; exit 1; }
[ "$1" = api ] && [ "$3" = --jq ] && [ "$4" = 'any(.labels[].name; ascii_downcase == "ready")' ] || {
  echo "the stub answers one filter, not: $*" >&2; exit 3; }
case "$2" in repos/*/pulls/[0-9]*) ;; *) echo "not a pull request's path: $2" >&2; exit 4 ;; esac
printf '%s\\n' "$STUB_LABELS" | tr 'A-Z' 'a-z' | grep -qx ready && echo true || echo false
"""


#: ``git`` for the boundary step: it answers the one diff the step may ask, with the files the pull request changes.
GIT_STUB = """#!/bin/sh
printf 'git %s\\n' "$*" >> "$STUB_LOG"
[ "$1 $2 $3 $4 $5" = "diff --name-only --no-renames HEAD^1 HEAD" ] || { echo "the stub answers one diff, not: $*" >&2; exit 3; }
[ -z "$STUB_GIT_FAILS" ] || exit 128
printf '%s\\n' $STUB_CHANGED
"""


class Text(str):
    """A string literal: GitHub compares strings without case."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Number):
            return other == self
        return isinstance(other, str) and self.lower() == other.lower()

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = str.__hash__


class Number:
    """A number literal: GitHub reads the other side of a comparison as a number."""

    def __init__(self, value: str) -> None:
        self.value = float(value)

    def __eq__(self, other: object) -> bool:
        try:
            return float(str(other) or 0) == self.value
        except ValueError:
            return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Run:
    """A run: its event, the labels its payload lists, the labels its mode step reads when it starts, its attempt,
    the mode it must take and the jobs it must start (None for a re-run of failed jobs, which GitHub picks)."""

    event: str
    action: str = ""
    payload: tuple[str, ...] = ()
    read: tuple[str, ...] = ()
    attempt: int = 1
    mode: str = "full"
    jobs: int | None = 63
    platforms: int | None = 5
    changed: tuple[str, ...] = ()
    lanes: frozenset[str] = ALL_LANES


FULL, REDUCED = "full", "reduced"
SCENARIOS: dict[str, Run] = {
    "a push to a branch": Run("push"),
    "a manual run": Run("workflow_dispatch", jobs=64),
    "a pull request opened without the label": Run("pull_request", "opened", mode=REDUCED, jobs=16, platforms=1,
                                                   lanes=NO_LANES),
    "a push to a pull request without the label": Run("pull_request", "synchronize", mode=REDUCED, jobs=16,
                                                      platforms=1, lanes=NO_LANES),
    "a push to a pull request with another label": Run("pull_request", "synchronize", ("stress",), ("stress",),
                                                       mode=REDUCED, jobs=16, platforms=1, lanes=NO_LANES),
    "a push to a pull request carrying ready": Run("pull_request", "synchronize", ("ready",), ("stress", "ready"),
                                                   jobs=27, lanes=NO_LANES),
    "a pull request opened with ready, its payload without it": Run("pull_request", "opened", (), ("ready",), jobs=27,
                                                                    lanes=NO_LANES),
    "ready removed before the run read the labels": Run("pull_request", "synchronize", ("ready",), (), mode=REDUCED,
                                                        jobs=16, platforms=1, lanes=NO_LANES),
    "the label Ready, capitalised": Run("pull_request", "synchronize", ("Ready",), ("Ready",), jobs=27, lanes=NO_LANES),
    "a re-run after ready was added": Run("pull_request", "synchronize", (), ("ready",), attempt=2, jobs=27,
                                          lanes=NO_LANES),
    "a re-run without ready": Run("pull_request", "synchronize", (), (), attempt=2, mode=REDUCED, jobs=16, platforms=1,
                                  lanes=NO_LANES),
    # The count job passed, so it is not re-run: the run keeps the mode its first attempt read.
    "a re-run of the failed jobs after ready was added": Run("pull_request", "synchronize", (), (), attempt=2,
                                                             mode=REDUCED, jobs=None, platforms=None, lanes=NO_LANES),
    # The boundary: what a full pull request's diff touches decides the lanes no longer required.
    "a full pull request that changes only tests and a guide": Run(
        "pull_request", "synchronize", ("ready",), ("ready",), jobs=27, lanes=NO_LANES,
        changed=("tests/test_x.py", "docs/guide.md", "src/mime/nodes/body/_gravity.py")),
    "a full pull request that changes the lock": Run(
        "pull_request", "synchronize", ("ready",), ("ready",), jobs=63, changed=("uv.lock", "docs/guide.md")),
    "a full pull request that changes a workflow": Run(
        "pull_request", "synchronize", ("ready",), ("ready",), jobs=63, changed=(".github/workflows/ci.yml",)),
    "a full pull request that changes a script the workflows call": Run(
        "pull_request", "synchronize", ("ready",), ("ready",), jobs=63, changed=("scripts/check_shards.py",)),
    "a full pull request that changes MADDENING's adapter": Run(
        "pull_request", "synchronize", ("ready",), ("ready",), jobs=39, lanes=frozenset({"maddening-main"}),
        changed=("src/mime/build/_maddening.py",)),
    "a full pull request that changes OpenMaterica's adapter": Run(
        "pull_request", "synchronize", ("ready",), ("ready",), jobs=39, lanes=frozenset({"openmaterica-main"}),
        changed=("src/mime/materials/_read.py",)),
    "a pull request without the label that changes the lock": Run(
        "pull_request", "synchronize", mode=REDUCED, jobs=16, platforms=1, lanes=NO_LANES, changed=("uv.lock",)),
}


def load(directory: Path, name: str) -> dict[str, Any]:
    return yaml.safe_load((directory / name).read_text(encoding="utf-8"))


def triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    return workflow.get("on", workflow.get(True)) or {}


class Context:
    """A context of names, and the expression language over it."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def value(self, name: str) -> Any:
        return self.values.get(name, "")

    def raw(self, text: str) -> Any:
        """The value of an expression, bare or wrapped in ``${{ }}``."""
        inner = re.fullmatch(r"\s*\$\{\{(.*)\}\}\s*", text, re.S)
        code = []
        tokens = [token for token in TOKEN.findall((inner.group(1) if inner else text)) if not token.isspace()]
        for at, token in enumerate(tokens):
            following = tokens[at + 1] if at + 1 < len(tokens) else ""
            if token.startswith("'"):
                code.append(f"_s({token[1:-1].replace(chr(39) * 2, chr(39))!r})")
            elif token.isdigit():
                code.append(f"_n({token!r})")
            elif re.fullmatch(r"[A-Za-z_][\w.*-]*", token):
                code.append(FUNCTIONS[token] if following == "(" else f"_v({token!r})")
            else:
                code.append({"&&": " and ", "||": " or ", "!": " not "}.get(token, token))

        def contains(haystack: Any, needle: Any) -> bool:
            if isinstance(haystack, list):
                return any(str(each).lower() == str(needle).lower() for each in haystack)
            return str(needle).lower() in str(haystack).lower()

        scope = {"_v": self.value, "_s": Text, "_n": Number, "_contains": contains, "_json": json.loads,
                 "_true": lambda: True, "_false": lambda: False}
        return eval("".join(code), {"__builtins__": {}}, scope)  # noqa: S307 - a workflow file's own expression

    def text(self, template: str) -> str:
        """``template`` with each ``${{ }}`` replaced by its value as GitHub prints it."""
        def shown(match: re.Match[str]) -> str:
            value = self.raw(match.group(0))
            return str(value).lower() if isinstance(value, bool) else str(value)
        return re.sub(r"\$\{\{.*?\}\}", shown, template)

    def holds(self, condition: object) -> bool:
        return True if condition is None else bool(self.raw(str(condition)))

    def with_values(self, **more: Any) -> "Context":
        return Context({**self.values, **more})


def base_context(run: Run, workflow: str) -> Context:
    return Context({"github.event_name": run.event, "github.event.action": run.action,
                    "github.event.pull_request.labels.*.name": list(run.payload),
                    "github.event.pull_request.number": "30" if run.event == "pull_request" else "",
                    "github.run_id": "700", "github.run_attempt": str(run.attempt), "github.workflow": workflow,
                    "github.repository": "an-owner/a-repository", "github.token": "a-token",
                    "needs.shards.outputs.count": str(SHARDS),
                    "needs.shards.outputs.list": json.dumps(list(range(1, SHARDS + 1)))})


def run_script(script: str, env: dict[str, str], scratch: Path, labels: tuple[str, ...] = (),
               gh_fails: bool = False, changed: tuple[str, ...] = (),
               git_fails: bool = False) -> tuple[int, str, dict[str, str]]:
    """``script`` in ``bash -e`` with ``gh`` and ``git`` stubbed (``scripts/`` is the repository's own, so the scripts a
    step runs are run for real): its exit status, its output and what it wrote to ``$GITHUB_OUTPUT``."""
    scratch.mkdir(parents=True, exist_ok=True)
    stubs = scratch / "bin"
    stubs.mkdir(exist_ok=True)
    for command, text in (("gh", GH_STUB), ("git", GIT_STUB)):
        (stubs / command).write_text(text, encoding="utf-8")
        (stubs / command).chmod(0o755)
    if not (scratch / "scripts").exists():
        (scratch / "scripts").symlink_to(ROOT / "scripts")
    output = scratch / "output"
    output.write_text("", encoding="utf-8")
    done = subprocess.run([shutil.which("bash") or "bash", "-e", "-c", script], capture_output=True, text=True,
                          check=False, cwd=scratch,
                          env={"PATH": f"{stubs}:/usr/bin:/bin", "GITHUB_OUTPUT": str(output),
                               "STUB_LOG": str(scratch / "gh.log"), "STUB_LABELS": "\n".join(labels),
                               "STUB_FAIL": "1" if gh_fails else "", "STUB_CHANGED": "\n".join(changed),
                               "STUB_GIT_FAILS": "1" if git_fails else "", **env})
    written = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines() if "=" in line)
    return done.returncode, done.stdout + done.stderr, written


def mode_step(job: dict[str, Any]) -> dict[str, Any] | None:
    return next((step for step in job.get("steps") or [] if step.get("id") == "mode"), None)


def read_mode(workflow: dict[str, Any], name: str, run: Run, scratch: Path) -> tuple[str, list[str]]:
    """The ``full`` output the workflow's mode step writes for ``run`` (its script run with ``gh`` answering the labels
    the run reads), and the problems met doing so."""
    job = (workflow.get("jobs") or {}).get(READERS[name]) or {}
    step = mode_step(job)
    if step is None:
        return "", [f"{name}: the job {READERS[name]} has no step `mode` to read the run's mode"]
    context = base_context(run, name)
    env = {key: context.text(str(value)) for key, value in (step.get("env") or {}).items()}
    status, said, written = run_script(context.text(str(step.get("run", ""))), env, scratch, run.read)
    if status != 0:
        return "", [f"{name}: the mode step fails for {run} (exit {status}): {said.strip()}"]
    return written.get("full", ""), []


def boundary_step(job: dict[str, Any]) -> dict[str, Any] | None:
    return next((step for step in job.get("steps") or [] if step.get("id") == "boundary"), None)


def read_lanes(workflow: dict[str, Any], run: Run, full: str, scratch: Path) -> tuple[dict[str, str], list[str]]:
    """The outputs the count job's boundary step writes for ``run`` once its mode step wrote ``full`` (the step's script
    run for real, ``git`` answering the files the pull request changes), and the problems met doing so."""
    job = (workflow.get("jobs") or {}).get("shards") or {}
    step = boundary_step(job)
    if step is None:
        return {}, ["ci.yml: the job shards has no step `boundary` to say which lanes a run starts"]
    context = base_context(run, "CI").with_values(**{"steps.mode.outputs.full": full})
    env = {key: context.text(str(value)) for key, value in (step.get("env") or {}).items()}
    status, said, written = run_script(context.text(str(step.get("run", ""))), env, scratch, changed=run.changed)
    if status != 0:
        return {}, [f"ci.yml: the boundary step fails for {run} (exit {status}): {said.strip()}"]
    declared = {key: value.removeprefix("${{ steps.boundary.outputs.").removesuffix(" }}")
                for key, value in (job.get("outputs") or {}).items() if "steps.boundary" in str(value)}
    if set(declared) != set(BOUNDARY_OUTPUTS) or any(key != value for key, value in declared.items()):
        return {}, [f"ci.yml: the job shards outputs {sorted(declared)} from its boundary step, not {list(BOUNDARY_OUTPUTS)}"]
    return {key: written.get(key, "") for key in BOUNDARY_OUTPUTS}, []


def combos(job: dict[str, Any], context: Context) -> list[dict[str, Any]]:
    """The matrix combinations a job runs as (one empty combination without a matrix), ``exclude`` applied."""
    matrix = dict((job.get("strategy") or {}).get("matrix") or {})
    exclude = matrix.pop("exclude", [])
    exclude = context.raw(exclude) if isinstance(exclude, str) else exclude
    matrix.pop("include", None)
    axes = {axis: context.raw(values) if isinstance(values, str) else values for axis, values in matrix.items()}
    made = [dict(zip(axes, values, strict=True)) for values in itertools.product(*axes.values())]
    return [one for one in made if not any(all(one.get(k) == v for k, v in out.items()) for out in exclude)]


def starts(workflow: dict[str, Any], context: Context) -> dict[str, list[dict[str, Any]]]:
    """The jobs that start in a run, with their combinations: a job's ``if`` holds, and (when it names no status
    function) every job it needs started."""
    jobs: dict[str, Any] = workflow.get("jobs") or {}
    started: dict[str, list[dict[str, Any]]] = {}

    def runs(job_id: str) -> bool:
        if job_id in started:
            return bool(started[job_id])
        job = jobs[job_id]
        needs = job.get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        condition = job.get("if")
        status = condition is not None and STATUS.search(str(condition)) is not None
        upstream = all([runs(need) for need in needs])
        ok = context.holds(condition) and (status or upstream)
        started[job_id] = combos(job, context) if ok else []
        return ok

    for job_id in jobs:
        runs(job_id)
    return {job_id: made for job_id, made in started.items() if made}


def job_names(workflow: dict[str, Any], context: Context) -> list[str]:
    jobs: dict[str, Any] = workflow.get("jobs") or {}
    return [context.with_values(**{f"matrix.{k}": v for k, v in combo.items()}).text(str(jobs[job_id].get("name", job_id)))
            for job_id, made in starts(workflow, context).items() for combo in made]


def steps_run(job: dict[str, Any], context: Context) -> tuple[list[str], list[str], bool]:
    """The names of the steps of a started job that run, the output of those that fail, and whether the job fails. Only
    the refusing step is run (in ``bash -e``); every other step that runs is taken to pass, which is what a green lane
    does. A failed step with ``continue-on-error`` fails neither the job nor the steps after it."""
    env = {f"env.{k}": context.text(str(v)) for k, v in (job.get("env") or {}).items()}
    context = context.with_values(**env)
    ran, failed_with, failed = [], [], False
    for step in job.get("steps") or []:
        condition = step.get("if")
        status = condition is not None and STATUS.search(str(condition)) is not None
        runs = context.holds(condition) and (status or not failed)
        if not runs:
            continue
        ran.append(str(step.get("name", step.get("uses", ""))))
        if "has not run on this head" in str(step.get("run", "")):
            done = subprocess.run(["bash", "-e", "-c", str(step["run"])], capture_output=True, text=True, check=False)
            if done.returncode != 0:
                failed_with = [*failed_with, done.stdout + done.stderr]
                continues = step.get("continue-on-error", False)
                failed = failed or not (continues if isinstance(continues, bool) else context.holds(str(continues)))
    return ran, failed_with, failed


def ci_problems(directory: Path, scratch: Path) -> list[str]:
    """Everything wrong with how ci.yml runs its two modes, read by playing each run through it."""
    found: list[str] = []
    workflow = load(directory, "ci.yml")
    found += trigger_problems("ci.yml", workflow)
    jobs: dict[str, Any] = workflow.get("jobs") or {}
    for name, run in SCENARIOS.items():
        full, problems = read_mode(workflow, "ci.yml", run, scratch / "ci" / re.sub(r"\W+", "-", name))
        found += problems
        if run.jobs is None:  # the count job is not re-run: its first attempt's output stands
            full = "true" if run.mode == FULL else "false"
        lanes, problems = read_lanes(workflow, run, full, scratch / "lanes" / re.sub(r"\W+", "-", name))
        found += problems
        if not lanes:
            continue
        context = base_context(run, "CI").with_values(**{"needs.shards.outputs.full": full}, **{
            f"needs.shards.outputs.{key}": value for key, value in lanes.items()})
        names = job_names(workflow, context)
        if run.jobs is not None and len(names) != run.jobs:
            found.append(f"ci.yml: {name}: starts {len(names)} jobs; the {run.mode} set is {run.jobs}")
        missing = sorted(set(REQUIRED) - set(names))
        found += [f"ci.yml: {name}: the required check {each!r} does not start, and a skipped job passes"
                  for each in missing]
        started = sorted(MOVED & set(names))
        if set(started) != set(run.lanes):
            found.append(f"ci.yml: {name}: starts the lanes {started}; this run starts {sorted(run.lanes)}")
        found += lane_problems(name, run.mode, jobs, workflow, context)
        group = context.text(str((workflow.get("concurrency") or {}).get("group", "")))
        if run.event == "pull_request" and group != "CI-30":
            found.append(f"ci.yml: {name}: in the group {group!r}, not the pull request's, so a newer push cannot "
                         f"cancel it")
    found += refusal_problems(jobs)
    return found


def trigger_problems(name: str, workflow: dict[str, Any]) -> list[str]:
    kinds = (triggers(workflow).get("pull_request") or {}).get("types") or []
    return [f"{name}: the pull_request trigger has `{kind}`, so a label starts a run of it"
            for kind in ("labeled", "unlabeled") if kind in kinds]


def lane_problems(scenario: str, mode: str, jobs: dict[str, Any], workflow: dict[str, Any], context: Context) -> list[str]:
    """What each started summary job does in a run: refuse (fail, having run nothing else) exactly when its lane did not
    run."""
    found: list[str] = []
    started = starts(workflow, context)
    for job_id in ("default", "openmaterica-main", "maddening-main"):
        for combo in started.get(job_id, []):
            at = context.with_values(**{f"matrix.{k}": v for k, v in combo.items()})
            name = at.text(str(jobs[job_id].get("name", job_id)))
            ran, failures, job_failed = steps_run(jobs[job_id], at)
            should_refuse = mode == REDUCED and name in REFUSING
            if should_refuse and not (job_failed and len(failures) == 1 and MESSAGE in failures[0] and len(ran) == 1):
                found.append(f"ci.yml: {scenario}: {name!r} does not just refuse with {MESSAGE!r} (it ran {ran}, "
                             f"failed with {failures}, the job {'fails' if job_failed else 'passes'})")
            if not should_refuse and failures:
                found.append(f"ci.yml: {scenario}: {name!r} refuses, and its lane runs here")
            if not should_refuse and not any(each.startswith("Every shard passed") for each in ran):
                found.append(f"ci.yml: {scenario}: {name!r} does not check its shards, and its lane runs here")
    return found


def refusal_problems(jobs: dict[str, Any]) -> list[str]:
    """The refusing step of the one summary job that refuses, ``default``: the first step, its ``if`` the mode alone (no
    attempt number, so a re-run refuses too), with no ``continue-on-error`` and no shell of its own. The lanes no longer
    required have none: they have no job where they do not run."""
    found: list[str] = []
    for job_id in ("default",):
        steps = (jobs.get(job_id) or {}).get("steps") or [{}]
        first = steps[0]
        if "has not run on this head" not in str(first.get("run", "")):
            found.append(f"ci.yml: {job_id}: its first step is not the refusal")
            continue
        if first.get("if") != REFUSAL_IF:
            found.append(f"ci.yml: {job_id}: the refusal runs only if {first.get('if')!r}, not {REFUSAL_IF!r}")
        for key in ("continue-on-error", "shell"):
            if key in first:
                found.append(f"ci.yml: {job_id}: the refusal sets {key}, so its failure may not fail the job")
    return found


def mode_job_problems(name: str, workflow: dict[str, Any], scratch: Path) -> list[str]:
    """The job that reads the run's mode: it may read the pull request, outputs what its mode step wrote, and its marker
    step (``scripts/ci_ready.py`` reads it) runs in the full matrix alone; the step fails when the labels cannot be read
    or are answered with anything but true or false."""
    job_id = READERS[name]
    job = (workflow.get("jobs") or {}).get(job_id) or {}
    found: list[str] = []
    if (job.get("permissions") or {}).get("pull-requests") != "read":
        found.append(f"{name}: the job {job_id} may not read the pull request, so it cannot read the labels")
    if (job.get("outputs") or {}).get("full") != "${{ steps.mode.outputs.full }}":
        found.append(f"{name}: the job {job_id} does not output the mode its step wrote")
    step = mode_step(job)
    if step is None:
        return [*found, f"{name}: the job {job_id} has no step `mode`"]
    if not str(step.get("name", "")).startswith("The run's mode"):
        found.append(f"{name}: the mode step is named {step.get('name')!r}; scripts/ci_ready.py looks for \"The run's "
                     f"mode\"")
    markers = [each for each in job.get("steps") or [] if each.get("name") == MARKER]
    for mode, value in (("full", "true"), ("reduced", "false")):
        runs = len(markers) == 1 and Context({"steps.mode.outputs.full": value}).holds(markers[0].get("if"))
        if runs != (mode == "full"):
            found.append(f"{name}: the step {MARKER!r} {'does not run' if mode == 'full' else 'runs'} in the {mode} "
                         f"set, so scripts/ci_ready.py misreads the run's mode")
    context = base_context(SCENARIOS["a push to a pull request carrying ready"], name)
    env = {key: context.text(str(value)) for key, value in (step.get("env") or {}).items()}
    script = context.text(str(step.get("run", "")))
    status, _, written = run_script(script, env, scratch / "unreadable", ("ready",), gh_fails=True)
    if status == 0:
        found.append(f"{name}: the mode step passes when the labels cannot be read (it wrote {written})")
    return found


def platforms_problems(directory: Path, scratch: Path) -> list[str]:
    """platforms.yml: the mode job alone for a reduced pull request, the smoke jobs in full."""
    found: list[str] = []
    workflow = load(directory, "platforms.yml")
    found += trigger_problems("platforms.yml", workflow)
    for name, run in SCENARIOS.items():
        if run.platforms is None:
            continue
        full, problems = read_mode(workflow, "platforms.yml", run, scratch / "platforms" / re.sub(r"\W+", "-", name))
        found += problems
        context = base_context(run, "Platforms").with_values(**{"needs.mode.outputs.full": full})
        names = job_names(workflow, context)
        if len(names) != run.platforms:
            found.append(f"platforms.yml: {name}: starts {len(names)} jobs; the {run.mode} set is {run.platforms}")
    return found


def ready_problems(directory: Path) -> list[str]:
    """ready.yml: on `labeled` alone, its job runs for `ready` alone and bears no required name, any other label's run
    has a group of its own, and it re-runs every workflow that reads the label, naming the job that reads it."""
    found: list[str] = []
    workflow = load(directory, "ready.yml")
    on = triggers(workflow)
    kinds = (on.get("pull_request") or {}).get("types") or []
    if set(on) != {"pull_request"} or kinds != ["labeled"]:
        found.append(f"ready.yml: runs on {sorted(on)} {kinds}, not on a pull request's `labeled` alone")
    ci_branches = set((triggers(load(directory, "ci.yml")).get("pull_request") or {}).get("branches") or [])
    if not ci_branches <= set((on.get("pull_request") or {}).get("branches") or []):
        found.append("ready.yml: does not run for a pull request into every branch ci.yml runs for")
    groups = {}
    for label, number in (("ready", "700"), ("Ready", "701"), ("ready", "702"), ("stress", "703"), ("documentation", "704")):
        context = Context({"github.event_name": "pull_request", "github.event.action": "labeled",
                           "github.event.label.name": label, "github.event.pull_request.number": "30",
                           "github.run_id": number, "github.workflow": "Ready"})
        names = job_names(workflow, context)
        if (label.lower() == "ready") != bool(names):
            found.append(f"ready.yml: the label {label} starts {names}; the label ready alone starts a job")
        found += [f"ready.yml: the label {label}: the job {each!r} bears a required check's name" for each in names
                  if each in REQUIRED]
        groups[(label, number)] = context.text(str((workflow.get("concurrency") or {}).get("group", "")))
        found += [f"ready.yml: the job {job_id} bears a required check's name" for job_id, job in
                  (workflow.get("jobs") or {}).items() if context.text(str(job.get("name", job_id))) in REQUIRED]
    for other in (("stress", "703"), ("documentation", "704")):
        sharing = [key for key in groups if key != other and groups[key] == groups[other]]
        if sharing:
            found.append(f"ready.yml: the label {other[0]} shares a group with {sharing}, so it could cancel a re-run "
                         f"waiting there")
    jobs = [job for job in (workflow.get("jobs") or {}).values()
            if any("scripts/ci_ready.py" in str(step.get("run", "")) for step in job.get("steps") or [])]
    if len(jobs) != 1:
        return [*found, f"ready.yml: {len(jobs)} jobs run scripts/ci_ready.py; one does"]
    if (jobs[0].get("permissions") or {}).get("actions") != "write":
        found.append("ready.yml: the job may not re-run a workflow (permissions: actions: write)")
    named = dict(re.findall(r'--run "([\w.-]+)=([^"]+)"', " ".join(str(step.get("run", "")) for step in
                                                                jobs[0].get("steps") or [])))
    for name, job_id in READERS.items():
        job = (load(directory, name).get("jobs") or {}).get(job_id) or {}
        if name not in named:
            found.append(f"ready.yml: {name} reads the label when a run starts, and is not re-run")
        elif named[name] != job.get("name", job_id):
            found.append(f"ready.yml: --run {name}={named[name]}: the job that reads the label is "
                         f"{job.get('name', job_id)!r}")
    found += [f"ready.yml: re-runs {name}, which does not read the label" for name in named if name not in READERS]
    return found


def problems(directory: Path, scratch: Path) -> list[str]:
    found = ci_problems(directory, scratch) + platforms_problems(directory, scratch) + ready_problems(directory)
    for name in READERS:
        found += mode_job_problems(name, load(directory, name), scratch / "mode" / name)
    return found


def test_the_two_modes_start_the_jobs_they_name_and_no_required_check_is_skipped(tmp_path):
    assert problems(WORKFLOWS, tmp_path) == []


def test_a_copy_of_the_workflows_passes_too(tmp_path):
    """The plants' baseline: the copy they are made in passes before a plant."""
    shutil.copytree(WORKFLOWS, tmp_path / "workflows")
    assert problems(tmp_path / "workflows", tmp_path / "scratch") == []


def test_the_reduced_set_is_what_the_brief_names(tmp_path):
    """16 jobs, and the names are the five things that run plus the one refusal."""
    run = SCENARIOS["a push to a pull request without the label"]
    workflow = load(WORKFLOWS, "ci.yml")
    full, found = read_mode(workflow, "ci.yml", run, tmp_path)
    assert (full, found) == ("false", [])
    lanes, found = read_lanes(workflow, run, full, tmp_path / "lanes")
    assert found == []
    names = job_names(workflow, base_context(run, "CI").with_values(**{"needs.shards.outputs.full": full}, **{
        f"needs.shards.outputs.{key}": value for key, value in lanes.items()}))
    assert sorted(set(names) - {f"default tests (Python 3.12, shard {k} of {SHARDS})" for k in range(1, SHARDS + 1)}) == sorted(
        [*REFUSING, "default (Python 3.12)", "completion tests (default, Python 3.12)", "typecheck (pyright, Python 3.12)",
        "no private name or path", "shard count"])
    assert sum(name.startswith("default tests (Python 3.12") for name in names) == SHARDS and len(names) == 16


def test_the_mode_step_reads_the_pull_request_and_nothing_else_on_a_push(tmp_path):
    """The step asks GitHub for the pull request's labels on a pull request alone; on a push it asks nothing and the
    run is full. An answer other than true or false fails it."""
    workflow = load(WORKFLOWS, "ci.yml")
    step = mode_step(workflow["jobs"]["shards"])
    assert step is not None and MODE_FILTER in str(step["run"])
    for event, asked in (("pull_request", ["api repos/an-owner/a-repository/pulls/30 --jq " + MODE_FILTER]),
                         ("push", []), ("workflow_dispatch", [])):
        context = base_context(Run(event), "CI")
        env = {key: context.text(str(value)) for key, value in step["env"].items()}
        status, said, written = run_script(context.text(str(step["run"])), env, tmp_path / event, ("ready",))
        log = tmp_path / event / "gh.log"
        assert (status, written.get("full")) == (0, "true"), said
        assert (log.read_text(encoding="utf-8").splitlines() if log.exists() else []) == asked
    context = base_context(Run("pull_request"), "CI")
    env = {key: context.text(str(value)) for key, value in step["env"].items()}
    for answer in ("", "[]", "True", "null"):
        junk = str(step["run"]).replace(READ.strip(), f"full=$(echo '{answer}')")
        assert junk != step["run"]
        status, said, written = run_script(junk, env, tmp_path / f"junk-{answer or 'empty'}", ("ready",))
        assert status != 0 and "full" not in written, (answer, said)


def in_job(text: str, job_id: str, old: str, new: str) -> str:
    """``text`` with ``old`` replaced by ``new`` once, inside the job ``job_id``."""
    start = text.index(f"\n  {job_id}:\n") + 1
    following = re.search(r"\n  [\w-]+:\n", text[start + 1:])
    end = start + 1 + following.start() if following else len(text)
    block = text[start:end]
    assert block.count(old) == 1, f"{old!r} is in {job_id} {block.count(old)} times"
    return text[:start] + block.replace(old, new) + text[end:]


REFUSAL = ("      - name: Refuse a head the full matrix has not run on\n"
           "        if: ${{ env.MIME_CI_RUNS != 'true' }}\n"
           "        run: |\n"
           "          echo '::error::the full matrix has not run on this head: add the `ready` label'\n"
           "          exit 1\n")
REFUSAL_IF_LINE = "        if: ${{ env.MIME_CI_RUNS != 'true' }}\n"
EXCLUDE = re.compile(r"        # Python 3.13 runs on a push.*\n        # a reduced run.*\n        exclude: .*\n")
FULL = "needs.shards.outputs.full == 'true'"
FROM_THE_EVENT = "contains(github.event.pull_request.labels.*.name, 'ready')"
READ = "            full=$(gh api \"$PULL_REQUEST\" --jq 'any(.labels[].name; ascii_downcase == \"ready\")')\n"

#: Each plant: the file, what it does to the text, and the problem it must raise (a part of it).
PLANTS = {
    "a-required-job-skipped-in-a-reduced-run": (
        "ci.yml", lambda t: in_job(t, "default", "\n    if: ${{ !cancelled() && ", f"\n    if: ${{{{ !cancelled() && {FULL} && "),
        "a pull request opened without the label: the required check 'default (Python 3.12)' does not start"),
    "a-required-job-skipped-on-a-push": (
        "ci.yml", lambda t: in_job(t, "typecheck", "github.event_name != 'schedule'", "github.event_name == 'pull_request'"),
        "a push to a branch: the required check 'typecheck (pyright, Python 3.12)' does not start"),
    "python-3.14-leaves-a-reduced-run-instead-of-refusing": (
        "ci.yml", lambda t: in_job(t, "default", "fromJSON(needs.shards.outputs.lane_exclude)",
                                   "fromJSON(needs.shards.outputs.exclude)"),
        "a push to a pull request without the label: the required check 'default (Python 3.14)' does not start"),
    "the-refusal-removed": (
        "ci.yml", lambda t: in_job(t, "default", REFUSAL, ""),
        "a push to a pull request without the label: 'default (Python 3.14)' does not just refuse"),
    "the-refusal-exits-0": (
        "ci.yml", lambda t: in_job(t, "default", "          exit 1\n", "          exit 0\n"),
        "a push to a pull request without the label: 'default (Python 3.14)' does not just refuse"),
    "the-refusal-gives-no-reason": (
        "ci.yml", lambda t: in_job(t, "default", "the full matrix has not run on this head: add the `ready` label",
                                   "has not run on this head"),
        "a push to a pull request without the label: 'default (Python 3.14)' does not just refuse"),
    "the-refusal-runs-in-the-full-matrix-too": (
        "ci.yml", lambda t: in_job(t, "default", "env.MIME_CI_RUNS != 'true'", "env.MIME_CI_RUNS != 'x'"),
        "a push to a branch: 'default (Python 3.12)' refuses, and its lane runs here"),
    "the-lane-runs-beside-the-refusal": (
        "ci.yml", lambda t: in_job(t, "default", "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1\n",
                                   "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1\n        if: always()\n"),
        "a push to a pull request without the label: 'default (Python 3.14)' does not just refuse"),
    # The review's F3: a refusal that fails without failing its job, and a refusal a re-run skips.
    "the-refusal-continues-on-error": (
        "ci.yml", lambda t: in_job(t, "default", REFUSAL_IF_LINE, REFUSAL_IF_LINE + "        continue-on-error: true\n"),
        "a push to a pull request without the label: 'default (Python 3.14)' does not just refuse"),
    "the-refusal-skipped-on-a-second-attempt": (
        "ci.yml", lambda t: in_job(t, "default", REFUSAL_IF_LINE,
                                   "        if: ${{ env.MIME_CI_RUNS != 'true' && github.run_attempt != 2 }}\n"),
        "a re-run without ready: 'default (Python 3.14)' does not just refuse"),
    "the-refusal-on-the-first-attempt-only": (
        "ci.yml", lambda t: in_job(t, "default", REFUSAL_IF_LINE,
                                   "        if: ${{ env.MIME_CI_RUNS != 'true' && github.run_attempt == 1 }}\n"),
        "a re-run of the failed jobs after ready was added: 'default (Python 3.14)' does not just refuse"),
    "the-refusal-names-the-attempt": (
        "ci.yml", lambda t: in_job(t, "default", REFUSAL_IF_LINE,
                                   "        if: ${{ env.MIME_CI_RUNS != 'true' && github.run_attempt != 9 }}\n"),
        "ci.yml: default: the refusal runs only if"),
    "every-python-in-a-reduced-run": (
        "ci.yml", lambda t: EXCLUDE.sub("", t),
        "a push to a pull request without the label: starts 38 jobs; the reduced set is 16"),
    "a-lane-no-longer-required-in-a-reduced-run": (
        "ci.yml", lambda t: in_job(t, "openmaterica-main-tests", " && needs.shards.outputs.openmaterica == 'true'", ""),
        "a push to a pull request without the label: starts 26 jobs; the reduced set is 16"),
    "a-lane-no-longer-required-ignores-the-boundary": (
        "ci.yml", lambda t: in_job(t, "openmaterica-main-tests", "needs.shards.outputs.openmaterica == 'true'",
                                   FULL),
        "a push to a pull request carrying ready: starts 37 jobs; the full set is 27"),
    "python-3.13-ignores-the-boundary": (
        "ci.yml", lambda t: in_job(t, "default-tests", "fromJSON(needs.shards.outputs.exclude)",
                                   "fromJSON(needs.shards.outputs.full == 'true' && '[]' || "
                                   "'[{\"python-version\":\"3.13\"},{\"python-version\":\"3.14\"}]')"),
        "a push to a pull request carrying ready: starts 37 jobs; the full set is 27"),
    "the-boundary-step-removed": (
        "ci.yml", lambda t: in_job(t, "shards", "      - name: Which lanes this run starts\n        id: boundary\n",
                                   "      - name: Which lanes this run starts\n        id: elsewhere\n"),
        "ci.yml: the job shards has no step `boundary`"),
    "the-boundary-not-told-the-mode": (
        "ci.yml", lambda t: in_job(t, "shards", "          FULL: ${{ steps.mode.outputs.full }}\n",
                                   "          FULL: 'true'\n"),
        "a push to a pull request without the label: starts 27 jobs; the reduced set is 16"),
    "a-required-job-renamed": (
        "ci.yml", lambda t: re.sub(r"(  confidential:\n    name: ).*\n", r"\1confidentiality\n", t),
        "a push to a branch: the required check 'no private name or path' does not start"),
    # The mode read from the event: a re-run reads the payload of the run it repeats, so it would stay reduced.
    "the-matrix-reads-the-event": (
        "ci.yml", lambda t: in_job(t, "default-tests", "fromJSON(needs.shards.outputs.exclude)",
                                   f"fromJSON({FROM_THE_EVENT} && '[{{\"python-version\":\"3.13\"}}]' || "
                                   "'[{\"python-version\":\"3.13\"},{\"python-version\":\"3.14\"}]')"),
        "a re-run after ready was added: starts 17 jobs; the full set is 27"),
    "a-summary-reads-the-event": (
        "ci.yml", lambda t: in_job(t, "default", f"MIME_CI_RUNS: ${{{{ {FULL} || ", f"MIME_CI_RUNS: ${{{{ {FROM_THE_EVENT} || "),
        "a re-run after ready was added: 'default (Python 3.14)' refuses, and its lane runs here"),
    "the-mode-step-reads-every-event": (
        "ci.yml", lambda t: in_job(t, "shards", '          if [ "$EVENT" = pull_request ]; then\n' + READ + "          fi\n",
                                   READ.replace("            ", "          ", 1)),
        "ci.yml: the mode step fails for Run(event='push'"),
    "the-mode-step-swallows-an-unread-label": (
        "ci.yml", lambda t: in_job(t, "shards", READ, READ.replace("\")')\n", "\")' || echo false)\n")),
        "ci.yml: the mode step passes when the labels cannot be read"),
    "the-mode-read-with-case": (
        "ci.yml", lambda t: in_job(t, "shards", 'ascii_downcase == "ready"', '. == "ready"'),
        "ci.yml: the mode step fails for"),
    "the-mode-job-may-not-read-the-pull-request": (
        "ci.yml", lambda t: in_job(t, "shards", "      pull-requests: read\n", ""),
        "ci.yml: the job shards may not read the pull request"),
    "the-marker-runs-in-every-mode": (
        "ci.yml", lambda t: in_job(t, "shards", "        if: steps.mode.outputs.full == 'true'\n", ""),
        "ci.yml: the step 'The full matrix runs' runs in the reduced set"),
    "the-mode-not-output": (
        "ci.yml", lambda t: in_job(t, "shards", "      full: ${{ steps.mode.outputs.full }}\n", ""),
        "ci.yml: the job shards does not output the mode its step wrote"),
    "labeled-back-in-ci": (
        "ci.yml", lambda t: t.replace("types: [opened, synchronize, reopened]\n", "types: [opened, synchronize, reopened, labeled]\n"),
        "ci.yml: the pull_request trigger has `labeled`"),
    "unlabeled-in-ci": (
        "ci.yml", lambda t: t.replace("types: [opened, synchronize, reopened]\n", "types: [opened, synchronize, reopened, unlabeled]\n"),
        "ci.yml: the pull_request trigger has `unlabeled`"),
    "a-re-run-in-a-group-of-its-own": (
        "ci.yml", lambda t: t.replace("github.event_name == 'pull_request' && github.event.pull_request.number || github.run_id",
                                      "github.event_name == 'pull_request' && github.run_attempt == 1 && github.event.pull_request.number || github.run_id"),
        "ci.yml: a re-run after ready was added: in the group 'CI-700'"),
    "labeled-back-in-platforms": (
        "platforms.yml", lambda t: t.replace("types: [opened, synchronize, reopened]\n", "types: [opened, synchronize, reopened, labeled]\n"),
        "platforms.yml: the pull_request trigger has `labeled`"),
    "platforms-in-a-reduced-run": (
        "platforms.yml", lambda t: in_job(t, "smoke", "    if: needs.mode.outputs.full == 'true'\n", ""),
        "platforms.yml: a push to a pull request without the label: starts 5 jobs; the reduced set is 1"),
    "platforms-reads-the-event": (
        "platforms.yml", lambda t: in_job(t, "smoke", "    if: needs.mode.outputs.full == 'true'\n",
                                          f"    if: github.event_name != 'pull_request' || {FROM_THE_EVENT}\n"),
        "platforms.yml: a re-run after ready was added: starts 1 jobs; the full set is 5"),
    "platforms-marker-missing": (
        "platforms.yml", lambda t: in_job(t, "mode", "      - name: The full matrix runs\n", "      - name: Done\n"),
        "platforms.yml: the step 'The full matrix runs' does not run in the full set"),
    "ready-for-every-label": (
        "ready.yml", lambda t: in_job(t, "rerun", "    if: github.event.label.name == 'ready'\n", ""),
        "ready.yml: the label stress starts ['the full matrix on the head (the label ready)']"),
    "ready-bears-a-required-name": (
        "ready.yml", lambda t: in_job(t, "rerun", "    name: the full matrix on the head (the label ready)\n",
                                      "    name: typecheck (pyright, Python 3.12)\n"),
        "ready.yml: the label ready: the job 'typecheck (pyright, Python 3.12)' bears a required check's name"),
    "ready-skipped-name-is-required": (
        "ready.yml", lambda t: in_job(t, "rerun", "    name: the full matrix on the head (the label ready)\n",
                                      "    name: default (Python 3.14)\n"),
        "ready.yml: the job rerun bears a required check's name"),
    "ready-one-group-for-every-label": (
        "ready.yml", lambda t: t.replace("github.event.label.name == 'ready' && github.event.pull_request.number || github.run_id",
                                         "github.event.pull_request.number"),
        "ready.yml: the label stress shares a group with"),
    "ready-without-platforms": (
        "ready.yml", lambda t: t.replace(" --run \"platforms.yml=the run's mode\"", ""),
        "ready.yml: platforms.yml reads the label when a run starts, and is not re-run"),
    "ready-names-another-job": (
        "ready.yml", lambda t: t.replace("--run \"ci.yml=shard count\"", "--run \"ci.yml=typecheck (pyright, Python 3.12)\""),
        "ready.yml: --run ci.yml=typecheck (pyright, Python 3.12): the job that reads the label is 'shard count'"),
    "ready-may-not-re-run": (
        "ready.yml", lambda t: in_job(t, "rerun", "      actions: write\n", "      actions: read\n"),
        "ready.yml: the job may not re-run a workflow"),
    "ready-on-unlabeled-too": (
        "ready.yml", lambda t: t.replace("types: [labeled]", "types: [labeled, unlabeled]"),
        "ready.yml: runs on ['pull_request'] ['labeled', 'unlabeled']"),
}


@pytest.mark.parametrize("plant", PLANTS, ids=list(PLANTS))
def test_a_break_in_the_modes_fails_naming_it(tmp_path, plant):
    name, change, problem = PLANTS[plant]
    copy = tmp_path / "workflows"
    shutil.copytree(WORKFLOWS, copy)
    text = (copy / name).read_text(encoding="utf-8")
    planted = change(text)
    assert planted != text, f"the plant {plant} changed nothing"
    (copy / name).write_text(planted, encoding="utf-8")
    found = problems(copy, tmp_path / "scratch")
    assert any(problem in each for each in found), found
