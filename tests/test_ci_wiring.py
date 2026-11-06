"""CI's wiring of the checks it must run, read from ``.github/workflows/``:

- every job name the development branch's protection requires is made by one job, of a workflow that runs on a pull
  request into the branch (``release/v0.3.0``) and on a push to it, and that job is skipped by nothing but the
  schedule (a skipped job passes a required check); and no workflow names the branch's old name, ``v0.3``;
- a required job that waits on a lane's shard jobs runs after a red shard too (``!cancelled()``), and fails unless
  ``scripts/check_shards.py`` passes, with its exit status intact, over as many shards as the jobs it waits on run (the
  matrix and their ``--mime-shard``), and those jobs are skipped by nothing but the schedule;
- the shard jobs and the job that waits on them have a ``timeout-minutes``, and the check allows the lane no skipped
  test (``--max-skipped 0``: the suite skips none in CI);
- a pull request's newer push cancels its older run, and nothing else is ever cancelled by a newer run (the workflow's
  ``concurrency``), read by evaluating its two expressions for each event;
- the default lane runs ``mime docs check`` against the development branch, ``--base origin/release/v0.3.0``, and
  fails before the check where that branch cannot be read (it never falls back to another name), with its exit status
  intact;
- no step of any workflow runs ``mime docs check`` with a ``--skip-*`` flag unless ``SKIPPABLE`` names that job for that
  flag (``ci.yml``'s default lane and ``platforms.yml``'s smoke jobs both run it, and one check reads every workflow
  file): ``--skip-example-runs`` is the non-required smoke job's alone, and a job that may skip is no required check.
  The flag is only ever spelled in full, since ``mime docs`` accepts no abbreviation (``allow_abbrev=False``). And
  no step runs ``mime docs build`` with any ``--skip-*`` flag (the published pages are built with their evidence
  resolved). Each is read from the step's text and from its commands when it is run, so a flag built from a
  variable shows too;
- the ``confidential`` job runs ``scripts/check_confidential.py`` with the key's secret in its environment, and
  ``--messages``, ``--diffs`` and ``--ref-name``, with its exit status intact.

"Intact" is read by running the step's script in bash as GitHub runs it (``bash -e``), its commands stubbed: a stub that
fails must fail the step and one that passes must pass it, so an ``|| true``, an ``exit 0`` or a dropped argument
shows, whatever its form; and neither step may carry ``continue-on-error``, an ``if`` beyond ``!cancelled()``, its own
shell, an ``|| true`` or a ``set +e``. Each control plants one break in a copy of the workflows and runs the same check,
which must fail naming it."""

import itertools
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
ACTIONS = ROOT / ".github" / "actions"
#: The development branch, and the name it had until 2026-11-05, which no workflow may name.
BRANCH, OLD_BRANCH = "release/v0.3.0", "v0.3"
#: The old name as a branch: ``v0.3`` not followed by a digit or by a ``.`` and a word character, so that a version
#: (``v0.3.0``, ``v0.3.0.dev1``) is not the name but ``v0.3`` ending a sentence is.
OLD_NAME = re.compile(r"\bv0\.3(?!\d|\.\w)")
#: The development branch's required status checks, by the names its branch protection gives them.
REQUIRED = ("default (Python 3.12)", "default (Python 3.13)", "default (Python 3.14)", "maddening-main",
            "openmaterica-main", "typecheck (pyright, Python 3.12)", "no private name or path")
DOCS_CHECK, SCAN = "default (Python 3.12)", "no private name or path"
SCHEDULE_FILTER = EVENT_FILTER = "github.event_name != 'schedule'"
#: The ``if`` of a required job that waits on shard jobs: a red shard must not skip it, since a skipped job passes.
AFTER_ITS_SHARDS = f"${{{{ !cancelled() && {EVENT_FILTER} }}}}"
#: The run's mode, read by the count job from the pull request's labels when the run starts (tests/test_ci_ready_label.py).
FULL_MODE = "needs.shards.outputs.full == 'true'"
#: The ``if`` of the shard jobs of the lanes that run in the full matrix alone.
FULL_LANE_TESTS = f"${{{{ {SCHEDULE_FILTER} && {FULL_MODE} }}}}"
#: A step that runs after a red one but, in a reduced run, only for the lanes that ran.
LANE_RAN = "${{ !cancelled() && env.MIME_CI_RUNS == 'true' }}"
GROUP_LINE = ("  group: ${{ github.workflow }}-${{ github.event_name == 'pull_request' && "
              "github.event.pull_request.number || github.run_id }}\n")
SHARD_CHECK = "scripts/check_shards.py"
#: The jobs that run the shell-completion tests alone, one per lane: a lane's job waits on its own, which is no shard job
#: (tests/test_completion_job.py holds their wiring).
COMPLETION_JOBS = ("default-completion", "openmaterica-main-completion", "maddening-main-completion")
COUNT_JOB, COUNT_SCRIPT = "shards", "scripts/shard_matrix.py"
SHARD_COUNT = "${{ needs.shards.outputs.count }}"
SHARD_LIST = "${{ fromJSON(needs.shards.outputs.list) }}"
#: What the count job outputs in the run the wiring is read for, and what the lanes' check must then be told.
COUNTED = "6"
#: Each skip flag of ``mime docs check``: why CI runs the stage it skips, and the jobs (a workflow file and a job id)
#: that may pass it. A flag the parser has and this table lacks fails ``test_every_skip_flag_of_the_command_is_listed``.
SKIPPABLE: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "--skip-evidence": ("CI resolves the evidence the specification and the anomaly registry cite", ()),
    "--skip-hash-seed-builds": ("CI runs the second build under another hash seed", ()),
    "--skip-example-runs": ("CI runs the scripts the guides' output blocks name", (("platforms.yml", "smoke"),)),
}
STEP_IFS = (None, "${{ !cancelled() }}", LANE_RAN)
NEUTERING = re.compile(r"\|\|\s*(?:true|:)(?![\w-])|\bset\s+\+e\b")
SECRET = "the-secret-stands-in"
#: What a run's expressions are, for a pull request into the development branch and for a push to it.
PULL_REQUEST = {"github.event_name": "pull_request", "github.head_ref": "a-branch", "github.base_ref": BRANCH,
                "github.ref_name": "30/merge", "github.event.pull_request.base.sha": "BASE",
                "github.event.pull_request.head.sha": "HEAD", "github.sha": "MERGE", "matrix.python-version": "3.12",
                "secrets.CONFIDENTIAL_SCAN_KEY": SECRET,
                "needs.shards.outputs.count": COUNTED}
PUSH = {**PULL_REQUEST, "github.event_name": "push", "github.head_ref": "", "github.base_ref": "",
        "github.ref_name": BRANCH, "github.event.before": "BEFORE", "github.sha": "AFTER",
        "github.event.pull_request.base.sha": "", "github.event.pull_request.head.sha": ""}
#: A manual run on a pull request's branch: no pull request in its payload, its ref the branch.
MANUAL = {**PUSH, "github.event_name": "workflow_dispatch", "github.ref_name": "a-branch", "github.event.before": "",
          "github.sha": "TIP"}
STUB = """#!/bin/sh
{ printf '%s' "${0##*/}"; for argument in "$@"; do printf '\\t%s' "$argument"; done
  printf '\\tkey=%s\\n' "${CONFIDENTIAL_SCAN_KEY-}"; } >> "$STUB_LOG"
if [ "${0##*/}" = git ]; then
  for name in $STUB_MISSING; do
    case " $* " in *"/$name "*|*"/$name:"*) exit 1 ;; esac
  done
  exit 0
fi
exit "$STUB_EXIT"
"""


def workflows(directory: Path) -> list[tuple[str, dict[str, object]]]:
    return [(path.name, yaml.safe_load(path.read_text(encoding="utf-8")))
            for path in sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")])]


def names_of(job_id: str, job: dict[str, object]) -> list[str]:
    """The check names a job makes: its name, each ``${{ matrix.<axis> }}`` in it taken over the matrix's values."""
    name = str(job.get("name", job_id))
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    if not isinstance(matrix, dict):  # a matrix read from an expression (stress.yml's) names no required check
        matrix = {}
    axes = re.findall(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}", name)
    names = []
    for values in itertools.product(*(matrix.get(axis, []) for axis in axes)):
        made = name
        for axis, value in zip(axes, values, strict=True):
            made = re.sub(rf"\$\{{\{{\s*matrix\.{re.escape(axis)}\s*\}}\}}", str(value), made)
        names.append(made)
    return names


def runs_on_the_branch(workflow: dict[str, object]) -> bool:
    """Whether the workflow runs on a pull request into the development branch and on a push to it."""
    triggers = workflow.get("on", workflow.get(True)) or {}
    return all(BRANCH in ((triggers.get(event) or {}).get("branches") or []) for event in ("pull_request", "push"))


def old_name_problems(directory: Path, actions: Path = ACTIONS) -> list[str]:
    """Each line of a workflow in ``directory``, or of a composite action under ``actions``, that names the branch's old
    name: a trigger, a fallback, a comment."""
    files = [(path.name, path) for path in sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")])]
    files += [(f"{path.parent.name}/{path.name}", path)
              for path in sorted([*actions.rglob("*.yml"), *actions.rglob("*.yaml")])]
    return [f"{name}: names the old branch name {OLD_BRANCH}: {line.strip()}"
            for name, path in files
            for line in path.read_text(encoding="utf-8").splitlines()
            if OLD_NAME.search(line)]


def branch_filter_problems(directory: Path) -> list[str]:
    """Each trigger, in any workflow in ``directory``, that filters by branch names and leaves out the development
    branch: a workflow dropping it would silently stop running on the branch (the ``stress`` label would do nothing)."""
    found = []
    for name, workflow in workflows(directory):
        triggers = workflow.get("on", workflow.get(True))
        for event, configuration in (triggers.items() if isinstance(triggers, dict) else []):
            branches = configuration.get("branches") if isinstance(configuration, dict) else None  # schedule: a list
            if branches is not None and BRANCH not in branches:
                found.append(f"{name}: the {event} trigger filters by branch and does not list {BRANCH}")
    return found


def evaluate(text: str, context: dict[str, str]) -> str:
    """``text`` with each ``${{ a || b }}`` replaced as GitHub would, from ``context`` (unknown names are empty)."""
    def value(match: re.Match[str]) -> str:
        return next((context[name.strip()] for name in match.group(1).split("||") if context.get(name.strip())), "")
    return re.sub(r"\$\{\{(.*?)\}\}", value, text)


def run_step(step: dict[str, object], context: dict[str, str], exit_status: int, scratch: Path,
             missing: tuple[str, ...] = ()):
    """The step's script run in ``bash -e`` with ``uv``, ``python3`` and ``git`` stubbed, each stub recording its
    arguments and the key it saw, ``uv`` and ``python3`` exiting ``exit_status``, and ``git`` failing for any call that
    names one of the branches in ``missing``: the script's exit status, and each stubbed call as its command, its
    arguments and the key."""
    scratch.mkdir(parents=True)
    stubs = scratch / "bin"
    stubs.mkdir()
    for command in ("uv", "python3", "git"):
        (stubs / command).write_text(STUB, encoding="utf-8")
        (stubs / command).chmod(0o755)
    (scratch / "step.sh").write_text(evaluate(str(step["run"]), context), encoding="utf-8")
    env = {name: evaluate(str(value), context) for name, value in (step.get("env") or {}).items()}
    log = scratch / "calls.log"
    log.touch()
    done = subprocess.run([shutil.which("bash") or "bash", "-e", str(scratch / "step.sh")], capture_output=True,
                          text=True, check=False, cwd=scratch,
                          env={"PATH": f"{stubs}:/usr/bin:/bin", "RUNNER_TEMP": str(scratch),
                               "GITHUB_STEP_SUMMARY": str(scratch / "summary.md"), "STUB_LOG": str(log),
                               "STUB_EXIT": str(exit_status), "STUB_MISSING": " ".join(missing), **env})
    calls = [line.split("\t") for line in log.read_text(encoding="utf-8").splitlines()]
    return done.returncode, [(call[0], call[1:-1], call[-1].removeprefix("key=")) for call in calls]


def pairs(arguments: list[str]) -> set[tuple[str, str]]:
    return {(arguments[at], arguments[at + 1]) for at in range(len(arguments) - 1) if arguments[at].startswith("--")}


def step_problems(where: str, step: dict[str, object]) -> list[str]:
    found = []
    if step.get("if") not in STEP_IFS:
        found.append(f"{where}: runs only if {step.get('if')!r}; it runs on every lane's run, or under !cancelled()")
    if step.get("continue-on-error"):
        found.append(f"{where}: continue-on-error hides its failure from the job")
    if "shell" in step:
        found.append(f"{where}: sets its own shell; it runs in GitHub's bash -e")
    if NEUTERING.search(str(step.get("run", ""))):
        found.append(f"{where}: its script holds an '|| true' or a 'set +e'")
    return found


def exit_problems(where: str, step: dict[str, object], context: dict[str, str], scratch: Path) -> list[str]:
    failed, _ = run_step(step, context, 1, scratch / "failing")
    passed, _ = run_step(step, context, 0, scratch / "passing")
    found = [] if failed != 0 else [f"{where}: exits 0 when its command fails; it must exit with the command's status"]
    return found + ([] if passed == 0 else [f"{where}: exits {passed} when its command passes"])


def one_step(job_name: str, job: dict[str, object], command: str) -> tuple[list[str], dict[str, object] | None]:
    steps = [step for step in job.get("steps") or [] if command in str(step.get("run", ""))]
    if len(steps) != 1:
        return [f"{job_name}: {len(steps)} steps run {command}; one does"], None
    return [], steps[0]


def wiring_problems(directory: Path, scratch: Path, actions: Path = ACTIONS) -> list[str]:
    """Everything wrong with how the workflows in ``directory`` run the required checks (``scratch`` holds the step
    runs)."""
    found: list[str] = []
    made: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = {}
    for _, workflow in workflows(directory):
        for job_id, job in (workflow.get("jobs") or {}).items():
            for name in names_of(job_id, job):
                made.setdefault(name, []).append((workflow, job))
    for name in REQUIRED:
        jobs = made.get(name, [])
        if len(jobs) != 1:
            found.append(f"{name!r}: {len(jobs)} jobs make this required check; one does")
            continue
        workflow, job = jobs[0]
        if not runs_on_the_branch(workflow):
            found.append(f"{name!r}: its workflow does not run on both a pull request into the development branch and "
                         f"a push to it")
        if job.get("needs"):
            found += shard_problems(name, job, workflow, scratch / "shards" / re.sub(r"\W+", "-", name))
        elif job.get("if") not in (None, EVENT_FILTER):
            found.append(f"{name!r}: runs only if {job.get('if')!r}, and a skipped job passes a required check")
    found += [problem for name, workflow in workflows(directory) if name == "ci.yml"
              for problem in concurrency_problems(workflow)]
    if len(made.get(DOCS_CHECK, [])) == 1:
        found += docs_check_problems(made[DOCS_CHECK][0][1], scratch / "docs")
    found += old_name_problems(directory, actions)
    found += branch_filter_problems(directory)
    found += skip_flag_problems(directory, scratch / "skip-flags")
    if len(made.get(SCAN, [])) == 1:
        found += scan_problems(made[SCAN][0][1], scratch / "scan")
    return found


def count_job_problems(name: str, workflow: dict[str, object], scratch: Path) -> list[str]:
    """The job that says how many shards a lane runs as: it exists, is skipped by nothing but the schedule, runs the
    count script with the variable and the manual input, and outputs the count and the list the others read."""
    job = (workflow.get("jobs") or {}).get(COUNT_JOB)
    if not isinstance(job, dict):
        return [f"{name!r}: there is no {COUNT_JOB} job to say how many shards a lane runs as"]
    found = []
    if job.get("if") not in (None, EVENT_FILTER):
        found.append(f"{name!r}: the {COUNT_JOB} job runs only if {job.get('if')!r}")
    if set((job.get("outputs") or {})) != {"count", "list", "full"}:
        found.append(f"{name!r}: the {COUNT_JOB} job outputs {sorted(job.get('outputs') or {})}, not count, list and "
                     f"full (the run's mode)")
    problems, step = one_step(f"{name}: the {COUNT_JOB} job", job, COUNT_SCRIPT)
    found += problems
    if step is not None:
        where = f"{name}: the {COUNT_JOB} job's step {step.get('name')!r}"
        found += step_problems(where, step) + exit_problems(where, step, PULL_REQUEST, scratch)
        env = step.get("env") or {}
        if env.get("VARIABLE") != "${{ vars.MIME_CI_SHARDS }}" or env.get("INPUT") != "${{ inputs.shards }}":
            found.append(f"{where}: does not read the repository variable MIME_CI_SHARDS and the shards input")
    return found


def shard_problems(name: str, job: dict[str, object], workflow: dict[str, object], scratch: Path) -> list[str]:
    """What is wrong with a required job that stands for a lane's shard jobs (``needs``)."""
    found: list[str] = []
    if job.get("if") != AFTER_ITS_SHARDS:
        found.append(f"{name!r}: runs only if {job.get('if')!r}; it waits on shard jobs, so it must run after a red "
                     f"one too ({AFTER_ITS_SHARDS}), since a skipped job passes a required check")
    needed = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
    if COUNT_JOB not in needed:
        found.append(f"{name!r}: does not wait on the {COUNT_JOB} job, so its --shards is not the count the shards ran")
    found += count_job_problems(name, workflow, scratch / "count")
    for needed_id in needed:
        if needed_id == COUNT_JOB or needed_id in COMPLETION_JOBS:
            continue
        shard_job = (workflow.get("jobs") or {}).get(needed_id) or {}
        if shard_job.get("if") not in (None, SCHEDULE_FILTER, FULL_LANE_TESTS):
            found.append(f"{name!r}: the shard job {needed_id} runs only if {shard_job.get('if')!r}")
        if not isinstance(shard_job.get("timeout-minutes"), int):
            found.append(f"{name!r}: the shard job {needed_id} has no timeout-minutes; a hung shard holds the lane for "
                         f"six hours")
        if shard_job.get("needs") != COUNT_JOB:
            found.append(f"{name!r}: the shard job {needed_id} does not wait on the {COUNT_JOB} job")
        listed = ((shard_job.get("strategy") or {}).get("matrix") or {}).get("shard")
        if listed != SHARD_LIST:
            found.append(f"{name!r}: the shard job {needed_id}'s matrix is {listed!r}, not the {COUNT_JOB} job's list")
        given = {of for step in shard_job.get("steps") or []
                 for of in re.findall(r'--mime-shard "\$\{\{ matrix\.shard \}\}/(.*?)"', str(step.get("run", "")))}
        if given != {SHARD_COUNT}:
            found.append(f"{name!r}: the shard job {needed_id} splits its files into {sorted(given)}, not "
                         f"{SHARD_COUNT}")
    if not isinstance(job.get("timeout-minutes"), int):
        found.append(f"{name!r}: has no timeout-minutes")
    problems, step = one_step(name, job, SHARD_CHECK)
    found += problems
    if step is None:
        return found
    where = f"{name}: the step {step.get('name')!r}"
    found += step_problems(where, step) + exit_problems(where, step, PULL_REQUEST, scratch)
    _, calls = run_step(step, PULL_REQUEST, 0, scratch / "arguments")
    checked = {value for command, arguments, _ in calls if command == "python3" and SHARD_CHECK in arguments
               for option, value in pairs(arguments) if option == "--shards"}
    if checked != {COUNTED}:
        found.append(f"{where}: checks {sorted(checked)} shards; its shard jobs run {COUNTED}")
    ceilings = {value for command, arguments, _ in calls if command == "python3" and SHARD_CHECK in arguments
                for option, value in pairs(arguments) if option == "--max-skipped"}
    if ceilings != {"0"}:
        found.append(f"{where}: allows {sorted(ceilings)} skipped tests; the suite skips none in CI")
    return found


#: A run of each event, as its expressions see it: two runs of one pull request, of another, and of one push.
RUNS = {
    "pull request 30, run 1": {"github.event_name": "pull_request", "github.event.pull_request.number": "30",
                               "github.run_id": "501", "github.workflow": "CI", "github.ref": "refs/pull/30/merge"},
    "pull request 30, run 2": {"github.event_name": "pull_request", "github.event.pull_request.number": "30",
                               "github.run_id": "502", "github.workflow": "CI", "github.ref": "refs/pull/30/merge"},
    "pull request 31": {"github.event_name": "pull_request", "github.event.pull_request.number": "31",
                        "github.run_id": "503", "github.workflow": "CI", "github.ref": "refs/pull/31/merge"},
    "push, run 1": {"github.event_name": "push", "github.run_id": "504", "github.workflow": "CI",
                    "github.ref": f"refs/heads/{BRANCH}"},
    "push, run 2": {"github.event_name": "push", "github.run_id": "505", "github.workflow": "CI",
                    "github.ref": f"refs/heads/{BRANCH}"},
    "manual": {"github.event_name": "workflow_dispatch", "github.run_id": "506", "github.workflow": "CI",
               "github.ref": "refs/heads/a-branch"},
    "schedule": {"github.event_name": "schedule", "github.run_id": "507", "github.workflow": "CI",
                 "github.ref": "refs/heads/main"},
}
TOKEN = re.compile(r"'[^']*'|[A-Za-z_][\w.]*|==|!=|&&|\|\||\S")


def expression(text: str, context: dict[str, str]) -> str:
    """The workflow expressions in ``text`` (``${{ a == 'b' && c || d }}``) evaluated as GitHub does for names,
    string literals, ``==``, ``!=``, ``&&`` and ``||``, and the rest of ``text`` kept."""
    def value(match: re.Match[str]) -> str:
        code = []
        for token in TOKEN.findall(match.group(1)):
            code.append({"&&": " and ", "||": " or ", "!": " not "}.get(token) or (
                repr(context.get(token, "")) if re.fullmatch(r"[A-Za-z_][\w.]*", token) else token))
        result = eval("".join(code), {"__builtins__": {}}, {})  # noqa: S307 - a workflow file's own expression
        return str(result).lower() if isinstance(result, bool) else str(result)
    return re.sub(r"\$\{\{(.*?)\}\}", value, text)


def concurrency_problems(workflow: dict[str, object]) -> list[str]:
    """What is wrong with how a workflow's runs supersede each other: a pull request's runs cancel the older of the
    two, another pull request's and every other run's do not."""
    block = workflow.get("concurrency")
    if not isinstance(block, dict) or "group" not in block or "cancel-in-progress" not in block:
        return ["ci.yml: no concurrency group that cancels a pull request's superseded runs"]
    group = {run: expression(str(block["group"]), context) for run, context in RUNS.items()}
    cancel = {run: expression(str(block["cancel-in-progress"]), context).lower() for run, context in RUNS.items()}
    found = [f"ci.yml: {run} {'does not cancel' if run.startswith('pull') else 'cancels'} its group's older runs"
             for run in RUNS if (cancel[run] == "true") != run.startswith("pull")]
    if group["pull request 30, run 1"] != group["pull request 30, run 2"]:
        found.append("ci.yml: a pull request's two runs are in different groups, so the newer cannot cancel the older")
    if group["pull request 30, run 1"] == group["pull request 31"]:
        found.append("ci.yml: two pull requests share a group, so one cancels the other's run")
    others = [run for run in RUNS if not run.startswith("pull")]
    if len({group[run] for run in others}) != len(others) or {group[run] for run in others} & {
            group[run] for run in RUNS if run.startswith("pull")}:
        found.append("ci.yml: a push, manual or scheduled run shares a group with another run, which could cancel it")
    return found


def docs_check_problems(job: dict[str, object], scratch: Path) -> list[str]:
    found, step = one_step(DOCS_CHECK, job, "mime docs check")
    if step is None:
        return found
    where = f"{DOCS_CHECK}: the step {step.get('name')!r}"
    found += step_problems(where, step) + exit_problems(where, step, PULL_REQUEST, scratch)
    # The base is the development branch, ``origin/release/v0.3.0``. A step that cannot read it fails before the check,
    # which it never runs: the old name is no fallback (the stub reads every name but the new one, so a fallback to
    # the old would run the check).
    _, calls = run_step(step, PULL_REQUEST, 0, scratch / "arguments")
    if not any(" ".join(arguments).find("mime docs check") >= 0 and ("--base", f"origin/{BRANCH}") in pairs(arguments)
               for command, arguments, _ in calls if command == "uv"):
        found.append(f"{where}: does not run mime docs check --base origin/{BRANCH}")
    status, calls = run_step(step, PULL_REQUEST, 0, scratch / "arguments-unreadable", (BRANCH,))
    if status == 0 or any(command == "uv" for command, _, _ in calls):
        found.append(f"{where}: with the branch unreadable, runs the check or exits 0")
    return found


def docs_commands(run: str, calls: list[tuple[str, list[str], str]]) -> list[tuple[str, list[str]]]:
    """The ``mime docs`` commands a step runs, each as its subcommand and the arguments after it: the step's text
    (``uv run mime docs build ...``, ``mime docs check ...``) and the commands it makes when run with them stubbed, so
    that a flag built from a variable shows too."""
    found: list[tuple[str, list[str]]] = []
    for line in run.replace("\\\n", " ").splitlines():
        words = line.split()
        for at in range(len(words) - 2):
            if words[at] == "mime" and words[at + 1] == "docs":
                found.append((words[at + 2], words[at + 3:]))
    for command, arguments, _ in calls:
        words = [command, *arguments]
        for at in range(len(words) - 2):
            if words[at] == "mime" and words[at + 1] == "docs":
                found.append((words[at + 2], words[at + 3:]))
    return found


def skip_flag_problems(directory: Path, scratch: Path, required: tuple[str, ...] = REQUIRED) -> list[str]:
    """Every step of every workflow in ``directory`` that runs ``mime docs check`` with a ``--skip-*`` flag its job may
    not pass (``SKIPPABLE``), or ``mime docs build`` with any ``--skip-*`` flag, read from its text and from its
    commands' arguments when it is run with them stubbed (``docs_commands``): CI runs each stage a flag skips, wherever
    it runs the check, and builds the published pages with nothing skipped. A job that may skip a stage makes no
    required check."""
    found: list[str] = []
    for name, workflow in workflows(directory):
        for job_id, job in (workflow.get("jobs") or {}).items():
            for index, step in enumerate(job.get("steps") or []):
                run = str(step.get("run", ""))
                if "mime docs" not in run:
                    continue
                _, calls = run_step(step, PULL_REQUEST, 0, scratch / f"{name}-{job_id}-{index}")
                where = f"{name}: the job {job_id}'s step {step.get('name')!r}"
                commands = docs_commands(run, calls)
                for flag, (reason, allowed) in SKIPPABLE.items():
                    if not any(subcommand == "check" and flag in arguments for subcommand, arguments in commands):
                        continue
                    if (name, job_id) not in allowed:
                        found.append(f"{where}: passes {flag}; {reason}")
                    elif set(names_of(job_id, job)) & set(required):
                        found.append(f"{name}: the job {job_id} passes {flag} and makes a required check; {reason}")
                for subcommand, arguments in commands:
                    if subcommand == "build":
                        found += [f"{where}: passes {flag} to mime docs build; CI builds the published pages with "
                                  f"nothing skipped" for flag in sorted({a for a in arguments if a.startswith("--skip")})]
    return list(dict.fromkeys(found))


def scan_problems(job: dict[str, object], scratch: Path) -> list[str]:
    found, step = one_step(SCAN, job, "scripts/check_confidential.py")
    if step is None:
        return found
    where = f"{SCAN}: the step {step.get('name')!r}"
    found += step_problems(where, step) + exit_problems(where, step, PULL_REQUEST, scratch)
    if (step.get("env") or {}).get("CONFIDENTIAL_SCAN_KEY") != "${{ secrets.CONFIDENTIAL_SCAN_KEY }}":
        found.append(f"{where}: its environment does not set CONFIDENTIAL_SCAN_KEY from the secret")
    # A manual run scans what a pull request of its head into the development branch does (every commit the branch
    # does not hold) and the branch's name, so its later green can never stand for a weaker scan of the same head. What
    # the branch holds is read from ``origin/release/v0.3.0`` alone: with it unreadable the step still names it, and
    # never another (the stub reads every name but that one, so a fallback to the old name shows).
    into_main = {**PULL_REQUEST, "github.base_ref": "main"}
    manual = {("--messages", "HEAD"), ("--diffs", "HEAD"), ("--ref-name", "a-branch")}
    expected = {"pull request": (PULL_REQUEST, {("--messages", "BASE..HEAD"), ("--diffs", "BASE..HEAD"),
                                                ("--ref-name", "a-branch")}, ()),
                "push": (PUSH, {("--messages", "BEFORE..AFTER"), ("--diffs", "BEFORE..AFTER")}, ()),
                "manual run": (MANUAL, manual | {("--public", f"origin/{BRANCH}")}, ()),
                "manual run, the branch unreadable": (MANUAL, manual | {("--public", f"origin/{BRANCH}")},
                                                      (BRANCH,)),
                "pull request into main": (into_main, {("--messages", "BASE..HEAD"), ("--diffs", "BASE..HEAD"),
                                                       ("--ref-name", "a-branch"), ("--public", f"origin/{BRANCH}")},
                                           ()),
                "pull request into main, the branch unreadable": (
                    into_main, {("--messages", "BASE..HEAD"), ("--diffs", "BASE..HEAD"), ("--ref-name", "a-branch"),
                                ("--public", f"origin/{BRANCH}")}, (BRANCH,))}
    for event, (context, wanted, missing) in expected.items():
        _, calls = run_step(step, context, 0, scratch / event.replace(" ", "-").replace(",", ""), missing)
        scans = [(arguments, key) for command, arguments, key in calls
                 if command == "python3" and arguments[:1] == ["scripts/check_confidential.py"]]
        if len(scans) != 1:
            found.append(f"{where}: on a {event}, runs scripts/check_confidential.py {len(scans)} times; once")
            continue
        arguments, key = scans[0]
        given = pairs(arguments)
        found += [f"{where}: on a {event}, does not pass {option} {value}" for option, value in sorted(wanted - given)]
        found += [f"{where}: on a {event}, passes {option} {value}" for option, value in sorted(given - wanted)]
        if key != SECRET:
            found.append(f"{where}: on a {event}, the scan runs without the key's secret")
    return found


def test_ci_runs_the_required_checks_and_each_with_its_exit_status(tmp_path):
    assert wiring_problems(WORKFLOWS, tmp_path) == []


def test_a_copy_of_the_workflows_passes_too(tmp_path):
    """The controls' baseline: the copy the plants are made in passes before a plant."""
    shutil.copytree(WORKFLOWS, tmp_path / "workflows")
    assert wiring_problems(tmp_path / "workflows", tmp_path / "scratch") == []


DOCS_STEP = f"{DOCS_CHECK}: the step 'Docs check'"
#: What the every-workflow guard says of a docs step of ci.yml's default job that passes a skip flag.
HASH_SEEDS = ("ci.yml: the job default's step 'Docs check': passes --skip-hash-seed-builds; "
              "CI runs the second build under another hash seed")
EXAMPLE_RUNS = ("ci.yml: the job default's step 'Docs check': passes --skip-example-runs; "
                "CI runs the scripts the guides' output blocks name")
SCAN_STEP = f"{SCAN}: the step \"No private name or path in the tracked files, the commits or the branch's name\""
DOCS_BASE = 'mime docs check --base "$base"'
SMOKE_DOCS_LINE = '          uv run --no-sync mime docs check --base "$base" --skip-example-runs\n'
SCAN_LINE = '            ${public:+--public "$public"} ${name:+--ref-name "$name"}\n'
LANE_MATRIX = 'python-version: ["3.12", "3.13", "3.14"]\n    env:\n      MIME_CI_LANE: default\n      # Python 3.12'
SHARD_LINE = ('        if: ' + LANE_RAN + '\n        run: python3 scripts/check_shards.py --shards "${{ needs.shards.outputs.count }}" --max-skipped 0 "$RUNNER_TEMP"/shards/*/\n')

BUILD_RUN = "uv run --no-sync mime docs build --out"
BUILD_JUNIT = '--junit "$RUNNER_TEMP/junit" \\\n'
BUILD_SKIPS = ("ci.yml: the job default's step 'Build the documentation pages': passes --skip-evidence to mime docs "
               "build; CI builds the published pages with nothing skipped")
PRIVATE_STEP = "      - name: No private name or path in the built pages\n"
SECOND_BUILD = ("      - name: A second build\n        run: uv run --no-sync mime docs build --out \"$RUNNER_TEMP/again\" "
                "--skip-evidence\n")

NOT_ON_THE_BRANCH = ("'typecheck (pyright, Python 3.12)': its workflow does not run on both a pull request into the "
                     "development branch and a push to it")
#: Each plant: the text replaced in ci.yml, its replacement, and the problem it must raise.
PLANTS = {
    "the-docs-check-step-if-false": (
        f"        if: {LANE_RAN}\n        run: |\n          status=0\n",
        "        if: false\n        run: |\n          status=0\n",
        f"{DOCS_STEP}: runs only if False; it runs on every lane's run, or under !cancelled()"),
    "the-confidential-job-if-false": (
        f"    name: no private name or path\n    if: {EVENT_FILTER}\n",
        "    name: no private name or path\n    if: false\n",
        "'no private name or path': runs only if False, and a skipped job passes a required check"),
    "messages-dropped": ('--messages "$range" --diffs', "--diffs",
                         f"{SCAN_STEP}: on a pull request, does not pass --messages BASE..HEAD"),
    "ref-name-dropped": (SCAN_LINE, '            ${public:+--public "$public"}\n',
                         f"{SCAN_STEP}: on a pull request, does not pass --ref-name a-branch"),
    "the-key-env-dropped": ("          CONFIDENTIAL_SCAN_KEY: ${{ secrets.CONFIDENTIAL_SCAN_KEY }}\n"
                            "          EVENT: ${{ github.event_name }}\n", "          EVENT: ${{ github.event_name }}\n",
                            f"{SCAN_STEP}: on a pull request, the scan runs without the key's secret"),
    "or-true-after-the-scan": (SCAN_LINE, SCAN_LINE.replace("\n", " || true\n"),
                               f"{SCAN_STEP}: exits 0 when its command fails; it must exit with the command's status"),
    "the-docs-step-exits-0": ('          exit "$status"\n', "          exit 0\n",
                              f"{DOCS_STEP}: exits 0 when its command fails; it must exit with the command's status"),
    "diffs-dropped": ('--messages "$range" --diffs "$range"', '--messages "$range"',
                      f"{SCAN_STEP}: on a pull request, does not pass --diffs BASE..HEAD"),
    "the-base-dropped": (DOCS_BASE, "mime docs check",
                         f"{DOCS_STEP}: does not run mime docs check --base origin/{BRANCH}"),
    "the-docs-step-on-the-old-name": (
        "base=origin/release/v0.3.0\n", "base=origin/v0.3\n",
        f"{DOCS_STEP}: does not run mime docs check --base origin/{BRANCH}"),
    "the-docs-step-running-without-a-base": (
        '|| { echo "$base cannot be read" >&2; exit 1; }', '|| echo "$base cannot be read" >&2',
        f"{DOCS_STEP}: with the branch unreadable, runs the check or exits 0"),
    "the-docs-step-falling-back-to-the-old-name": (
        '|| { echo "$base cannot be read" >&2; exit 1; }', "|| base=origin/v0.3",
        f"{DOCS_STEP}: with the branch unreadable, runs the check or exits 0"),
    "the-scan-falling-back-to-the-old-name": (
        "          dev=origin/release/v0.3.0\n",
        '          dev=origin/release/v0.3.0\n          git rev-parse --verify --quiet "$dev" > /dev/null || dev=origin/v0.3\n',
        f"{SCAN_STEP}: on a manual run, the branch unreadable, does not pass --public origin/{BRANCH}"),
    "the-old-name-in-a-comment": (
        "      - name: Docs check\n        if: ${{ !cancelled() && env.MIME_CI_RUNS == 'true' }}\n",
        "      # held against origin/v0.3\n      - name: Docs check\n        if: ${{ !cancelled() && env.MIME_CI_RUNS == 'true' }}\n",
        "ci.yml: names the old branch name v0.3: # held against origin/v0.3"),
    "the-docs-step-skips-the-hash-seed-builds": (
        DOCS_BASE, DOCS_BASE + " --skip-hash-seed-builds",
        HASH_SEEDS),
    "the-docs-step-skips-the-hash-seed-builds-before-the-base": (
        DOCS_BASE, "mime docs check --skip-hash-seed-builds --base \"$base\"",
        HASH_SEEDS),
    "the-docs-step-skips-the-example-runs": (
        DOCS_BASE, DOCS_BASE + " --skip-example-runs",
        EXAMPLE_RUNS),
    "the-docs-step-skips-the-evidence": (
        DOCS_BASE, DOCS_BASE + " --skip-evidence",
        "ci.yml: the job default's step 'Docs check': passes --skip-evidence; CI resolves the evidence the "
        "specification and the anomaly registry cite"),
    "the-build-step-skips-the-evidence": (
        BUILD_JUNIT, BUILD_JUNIT.replace(' \\\n', ' --skip-evidence \\\n', 1), BUILD_SKIPS),
    "the-build-step-skips-another-check": (
        BUILD_JUNIT, BUILD_JUNIT.replace(' \\\n', ' --skip-citations \\\n', 1),
        BUILD_SKIPS.replace("--skip-evidence", "--skip-citations")),
    "the-build-step-skips-the-evidence-built-from-a-variable": (
        BUILD_RUN, "PART=evidence; " + BUILD_RUN.replace("docs build", "docs build --skip-$PART", 1), BUILD_SKIPS),
    "a-second-build-step-skips-the-evidence": (
        PRIVATE_STEP, SECOND_BUILD + PRIVATE_STEP, BUILD_SKIPS.replace("Build the documentation pages",
                                                                       "A second build")),
    "the-docs-step-continues-on-error": (
        f"        if: {LANE_RAN}\n        run: |\n          status=0\n",
        f"        if: {LANE_RAN}\n        continue-on-error: true\n        run: |\n          status=0\n",
        f"{DOCS_STEP}: continue-on-error hides its failure from the job"),
    "a-manual-run-scans-the-tree-alone": (
        '            *) range="HEAD"; public="$dev"; name="$REF_NAME" ;;\n', '            *) range="" ;;\n',
        f"{SCAN_STEP}: on a manual run, does not pass --messages HEAD"),
    "a-manual-run-scans-the-branch-s-commits-too": (
        '            *) range="HEAD"; public="$dev"; name="$REF_NAME" ;;\n',
        '            *) range="HEAD"; name="$REF_NAME" ;;\n',
        f"{SCAN_STEP}: on a manual run, does not pass --public origin/{BRANCH}"),
    "a-manual-run-without-the-branch-name": (
        '            *) range="HEAD"; public="$dev"; name="$REF_NAME" ;;\n',
        '            *) range="HEAD"; public="$dev" ;;\n',
        f"{SCAN_STEP}: on a manual run, does not pass --ref-name a-branch"),
    "set-plus-e-in-the-scan": ("          public=\"\"\n", "          set +e\n          public=\"\"\n",
                               f"{SCAN_STEP}: its script holds an '|| true' or a 'set +e'"),
    "the-scan-job-renamed": ("    name: no private name or path\n", "    name: confidentiality\n",
                             "'no private name or path': 0 jobs make this required check; one does"),
    "a-python-dropped-from-the-matrix": (
        LANE_MATRIX, LANE_MATRIX.replace('["3.12", "3.13", "3.14"]', '["3.12", "3.13"]'),
        "'default (Python 3.14)': 0 jobs make this required check; one does"),
    "the-lane-job-skipped-after-a-red-shard": (
        f"    needs: [shards, default-tests, default-completion]\n    if: {AFTER_ITS_SHARDS}\n",
        "    needs: [shards, default-tests, default-completion]\n    if: github.event_name != 'schedule'\n",
        "'default (Python 3.12)': runs only if \"github.event_name != 'schedule'\"; it waits on shard jobs, so it must "
        f"run after a red one too ({AFTER_ITS_SHARDS}), since a skipped job passes a required check"),
    "the-shard-check-swallowed": (
        SHARD_LINE, SHARD_LINE.replace("/*/\n", "/*/ || true\n"),
        "default (Python 3.12): the step 'Every shard passed, and together they ran the lane': exits 0 when its "
        "command fails; it must exit with the command's status"),
    "a-shard-dropped-from-a-matrix": (
        "        shard: ${{ fromJSON(needs.shards.outputs.list) }}\n    env:\n      MIME_CI_LANE: openmaterica-main\n",
        "        shard: [1, 2, 3, 4, 5]\n    env:\n      MIME_CI_LANE: openmaterica-main\n",
        "'openmaterica-main': the shard job openmaterica-main-tests's matrix is [1, 2, 3, 4, 5], not the shards job's "
        "list"),
    "a-lane-told-another-count-than-the-shards-ran": (
        'check_shards.py --shards "${{ needs.shards.outputs.count }}" --max-skipped 0 "$RUNNER_TEMP"/shards/*/\n      - name: Test durations\n'
        f'        if: {LANE_RAN}\n        env:\n          TITLE: "Test durations: openmaterica-main,',
        'check_shards.py --shards 4 --max-skipped 0 "$RUNNER_TEMP"/shards/*/\n      - name: Test durations\n'
        f'        if: {LANE_RAN}\n        env:\n          TITLE: "Test durations: openmaterica-main,',
        "openmaterica-main: the step 'Every shard passed, and together they ran the lane': checks ['4'] shards; its "
        "shard jobs run 6"),
    "a-shard-job-not-waiting-on-the-count": (
        "    name: openmaterica-main tests (shard ${{ matrix.shard }} of ${{ needs.shards.outputs.count }})\n    needs: shards\n",
        "    name: openmaterica-main tests (shard ${{ matrix.shard }} of ${{ needs.shards.outputs.count }})\n",
        "'openmaterica-main': the shard job openmaterica-main-tests does not wait on the shards job"),
    "a-lane-not-waiting-on-the-count": (
        "    needs: [shards, openmaterica-main-tests, openmaterica-main-completion]\n",
        "    needs: [openmaterica-main-tests, openmaterica-main-completion]\n",
        "'openmaterica-main': does not wait on the shards job, so its --shards is not the count the shards ran"),
    "the-count-job-not-reading-the-variable": (
        "          VARIABLE: ${{ vars.MIME_CI_SHARDS }}\n", "          VARIABLE: ''\n",
        "default (Python 3.12): the shards job's step 'The shard count': does not read the repository variable "
        "MIME_CI_SHARDS and the shards input"),
    "the-count-job-renamed": (
        "  shards:\n    name: shard count\n", "  counting:\n    name: shard count\n",
        "'default (Python 3.12)': there is no shards job to say how many shards a lane runs as"),
    "the-shard-check-removed": (
        "      - name: Every shard passed, and together they ran the lane\n"
        "        run: python3 scripts/check_shards.py --shards \"${{ needs.shards.outputs.count }}\" --max-skipped 0 \"$RUNNER_TEMP\"/shards/*/\n"
        "      - name: Test durations\n        if: " + LANE_RAN + "\n        env:\n"
        "          TITLE: \"Test durations: maddening-main, ${{ needs.shards.outputs.count }} shards\"\n",
        "      - name: Test durations\n        if: " + LANE_RAN + "\n        env:\n"
        "          TITLE: \"Test durations: maddening-main, ${{ needs.shards.outputs.count }} shards\"\n",
        "maddening-main: 0 steps run scripts/check_shards.py; one does"),
    "a-skip-ceiling-above-the-suites-none": (
        SHARD_LINE, SHARD_LINE.replace("--max-skipped 0", "--max-skipped 1"),
        "default (Python 3.12): the step 'Every shard passed, and together they ran the lane': allows ['1'] skipped "
        "tests; the suite skips none in CI"),
    "the-skip-ceiling-dropped": (
        SHARD_LINE, SHARD_LINE.replace(" --max-skipped 0", ""),
        "default (Python 3.12): the step 'Every shard passed, and together they ran the lane': allows [] skipped "
        "tests; the suite skips none in CI"),
    "a-shard-job-without-a-timeout": (
        "  default-tests:\n    name: default tests (Python ${{ matrix.python-version }}, shard ${{ matrix.shard }} of ${{ needs.shards.outputs.count }})\n"
        "    needs: shards\n    if: github.event_name != 'schedule'\n    runs-on: ubuntu-latest\n    timeout-minutes: 20\n",
        "  default-tests:\n    name: default tests (Python ${{ matrix.python-version }}, shard ${{ matrix.shard }} of ${{ needs.shards.outputs.count }})\n"
        "    needs: shards\n    if: github.event_name != 'schedule'\n    runs-on: ubuntu-latest\n",
        "'default (Python 3.12)': the shard job default-tests has no timeout-minutes; a hung shard holds the lane for "
        "six hours"),
    "every-run-cancels-its-group": (
        "  cancel-in-progress: ${{ github.event_name == 'pull_request' }}\n", "  cancel-in-progress: true\n",
        "ci.yml: push, run 1 cancels its group's older runs"),
    "no-run-cancels-its-group": (
        "  cancel-in-progress: ${{ github.event_name == 'pull_request' }}\n", "  cancel-in-progress: false\n",
        "ci.yml: pull request 30, run 1 does not cancel its group's older runs"),
    "one-group-for-every-run": (
        GROUP_LINE,
        "  group: ${{ github.workflow }}\n",
        "ci.yml: two pull requests share a group, so one cancels the other's run"),
    "pushes-to-a-branch-in-one-group": (
        GROUP_LINE,
        "  group: ${{ github.workflow }}-${{ github.ref }}\n",
        "ci.yml: a push, manual or scheduled run shares a group with another run, which could cancel it"),
    "no-pull-request-into-the-branch": (
        "  pull_request:\n    branches: [main, release/v0.3.0]\n", "  pull_request:\n    branches: [main]\n",
        NOT_ON_THE_BRANCH),
    "the-pull-request-trigger-on-the-old-name": (
        "  pull_request:\n    branches: [main, release/v0.3.0]\n", "  pull_request:\n    branches: [main, v0.3]\n",
        NOT_ON_THE_BRANCH),
    "no-push-to-the-branch": (
        "  push:\n    branches: [main, release/v0.3.0]\n", "  push:\n    branches: [main]\n", NOT_ON_THE_BRANCH),
    "the-old-name-back-in-the-pull-request-trigger": (
        "  pull_request:\n    branches: [main, release/v0.3.0]\n",
        "  pull_request:\n    branches: [main, release/v0.3.0, v0.3]\n",
        "ci.yml: names the old branch name v0.3: branches: [main, release/v0.3.0, v0.3]"),
    "the-old-name-back-in-the-push-trigger": (
        "  push:\n    branches: [main, release/v0.3.0]\n", "  push:\n    branches: [main, release/v0.3.0, v0.3]\n",
        "ci.yml: names the old branch name v0.3: branches: [main, release/v0.3.0, v0.3]"),
}


@pytest.mark.parametrize("plant", PLANTS, ids=list(PLANTS))
def test_a_break_in_the_wiring_fails_naming_it(tmp_path, plant):
    old, new, problem = PLANTS[plant]
    copy = tmp_path / "workflows"
    shutil.copytree(WORKFLOWS, copy)
    text = (copy / "ci.yml").read_text(encoding="utf-8")
    assert text.count(old) == 1, f"the plant's anchor {old!r} is not in ci.yml once"
    (copy / "ci.yml").write_text(text.replace(old, new), encoding="utf-8")
    assert problem in wiring_problems(copy, tmp_path / "scratch")


def _branch_lists() -> list[tuple[str, int, str]]:
    """Each ``branches: [...]`` line naming the development branch in a real workflow: its file, its line, its event."""
    found = []
    for path in sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]):
        event = ""
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if key := re.match(r"^  ([\w-]+):\s*$", line):
                event = key.group(1)
            if re.match(r"^\s+branches: \[.*\]$", line) and BRANCH in line:
                found.append((path.name, number, event))
    return found


def test_the_branch_filters_the_plants_below_are_made_in_cover_every_workflow_that_has_one():
    """Without this the plants below could pass over a workflow (stress.yml, ready.yml) that no test reads."""
    assert {name for name, _, _ in _branch_lists()} >= {"ci.yml", "platforms.yml", "ready.yml", "stress.yml"}
    assert branch_filter_problems(WORKFLOWS) == []


@pytest.mark.parametrize("plant", _branch_lists(), ids=[f"{n}:{e}:{k}" for n, k, e in _branch_lists()])
def test_every_workflow_filtering_by_branch_lists_the_development_branch_each_in_turn(tmp_path, plant):
    """One rule for every workflow: dropping the branch from the trigger of any one (a trigger the required-checks
    wiring does not read, stress.yml's, included) fails, naming it; text replacements, so every run."""
    name, number, event = plant
    copy = tmp_path / "workflows"
    shutil.copytree(WORKFLOWS, copy)
    lines = (copy / name).read_text(encoding="utf-8").splitlines(keepends=True)
    dropped = re.sub(rf",\s*{re.escape(BRANCH)}|{re.escape(BRANCH)},\s*|{re.escape(BRANCH)}", "", lines[number], count=1)
    assert dropped != lines[number]
    lines[number] = dropped
    (copy / name).write_text("".join(lines), encoding="utf-8")
    assert f"{name}: the {event} trigger filters by branch and does not list {BRANCH}" in branch_filter_problems(copy)
    assert f"{name}: the {event} trigger filters by branch and does not list {BRANCH}" in wiring_problems(
        copy, tmp_path / "scratch")


def test_the_old_name_in_a_composite_action_fails_naming_it(tmp_path):
    """The scan reads ``.github/actions`` too: a plant in a copy of the action, ``v0.3`` in a comment."""
    copy = tmp_path / "actions"
    shutil.copytree(ACTIONS, copy)
    action = next(copy.rglob("action.yml"))
    action.write_text("# held against origin/v0.3\n" + action.read_text(encoding="utf-8"), encoding="utf-8")
    assert old_name_problems(WORKFLOWS, ACTIONS) == []
    assert old_name_problems(WORKFLOWS, copy) == [
        f"{action.parent.name}/action.yml: names the old branch name v0.3: # held against origin/v0.3"]


@pytest.mark.parametrize("line", [
    "# a branch, v0.3.", "git fetch origin v0.3.", "origin/v0.3", "origin/v0.3 and more", "'v0.3'", "v0.3\n",
    "v0.3-helix-ref"])
def test_the_old_name_is_found_in_every_place_a_branch_is_named(line):
    assert OLD_NAME.search(line)


@pytest.mark.parametrize("line", ["v0.3.0", "release/v0.3.0", "v0.3.0.dev1", "tag v0.3.0.dev1.", "0.3.0", "v0.31", "xv0.3"])
def test_a_version_is_not_the_old_name(line):
    assert not OLD_NAME.search(line)


@pytest.mark.parametrize("plant", ["--skip-evidence", "--skip-$PART"])
def test_control_the_platforms_docs_check_skipping_the_evidence_fails(tmp_path, plant):
    """The one check reads every workflow: platforms.yml's docs check passing --skip-evidence fails, written out and
    built from a variable its step sets, which only running the step shows."""
    copy = tmp_path / "workflows"
    shutil.copytree(WORKFLOWS, copy)
    text = (copy / "platforms.yml").read_text(encoding="utf-8")
    old = SMOKE_DOCS_LINE
    assert text.count(old) == 1
    new = old.replace("\n", f" {plant}\n")
    if plant == "--skip-$PART":
        new = "          PART=evidence\n" + new
    (copy / "platforms.yml").write_text(text.replace(old, new), encoding="utf-8")
    assert wiring_problems(copy, tmp_path / "scratch") == [
        "platforms.yml: the job smoke's step 'Docs check': passes --skip-evidence; CI resolves the evidence the "
        "specification and the anomaly registry cite"]


def test_every_skip_flag_of_the_command_is_listed():
    """A skip flag the parser gains must be classified in ``SKIPPABLE``, with the jobs that may pass it, or the guard
    over the workflows would not know it."""
    from mime.cli import build_parser
    from mime.cli._guide import _subparsers_of  # pyright: ignore[reportPrivateUsage]

    docs = _subparsers_of(build_parser())
    assert docs is not None
    check = _subparsers_of(docs.choices["docs"])
    assert check is not None
    flags = {s for a in check.choices["check"]._actions for s in a.option_strings if s.startswith("--skip-")}  # pyright: ignore[reportPrivateUsage]
    assert flags == set(SKIPPABLE)


def test_mime_docs_takes_a_skip_flag_only_spelled_in_full():
    """The guard reads the full spelling, so the parser must refuse every shorter one: argparse takes ``--skip-ex`` for
    ``--skip-example-runs`` unless the parser says ``allow_abbrev=False``. The control is the full spelling."""
    from mime.cli import build_parser
    from mime.cli._guide import _subparsers_of  # pyright: ignore[reportPrivateUsage]

    parser = build_parser()
    docs = _subparsers_of(parser)
    assert docs is not None
    assert parser.parse_args(["docs", "check", "--skip-example-runs"]).skip_example_runs is True
    for flag in SKIPPABLE:
        for length in range(len("--s"), len(flag)):
            with pytest.raises(SystemExit):
                parser.parse_args(["docs", "check", flag[:length]])
    assert all(not child.allow_abbrev for child in _every_docs_parser(docs.choices["docs"]))


def _every_docs_parser(parser):
    from mime.cli._guide import _subparsers_of  # pyright: ignore[reportPrivateUsage]

    yield parser
    held = _subparsers_of(parser)
    for child in (held.choices.values() if held is not None else ()):
        yield from _every_docs_parser(child)


def _planted_workflows(tmp_path: Path, old: str, new: str, *, extra: str | None = None) -> Path:
    copy = tmp_path / "workflows"
    shutil.copytree(WORKFLOWS, copy)
    text = (copy / "platforms.yml").read_text(encoding="utf-8")
    assert text.count(old) == 1
    (copy / "platforms.yml").write_text(text.replace(old, new), encoding="utf-8")
    if extra is not None:
        (copy / "planted.yml").write_text(extra, encoding="utf-8")
    return copy


def test_control_the_smoke_job_may_skip_the_example_runs_and_the_real_workflows_pass(tmp_path):
    assert skip_flag_problems(WORKFLOWS, tmp_path / "scratch") == []


@pytest.mark.parametrize("flag, reason", [
    ("--skip-hash-seed-builds", "CI runs the second build under another hash seed"),
    ("--skip-evidence", "CI resolves the evidence the specification and the anomaly registry cite")])
def test_the_smoke_job_skipping_another_stage_fails_naming_the_flag(tmp_path, flag, reason):
    copy = _planted_workflows(tmp_path, SMOKE_DOCS_LINE, SMOKE_DOCS_LINE.replace("--skip-example-runs", flag))
    assert skip_flag_problems(copy, tmp_path / "scratch") == [
        f"platforms.yml: the job smoke's step 'Docs check': passes {flag}; {reason}"]


def test_another_job_skipping_the_example_runs_fails_in_any_workflow(tmp_path):
    other = ("on: [pull_request]\njobs:\n  elsewhere:\n    runs-on: ubuntu-latest\n    steps:\n"
             "      - name: Docs\n        run: uv run mime docs check --skip-example-runs\n")
    copy = _planted_workflows(tmp_path, SMOKE_DOCS_LINE, SMOKE_DOCS_LINE, extra=other)
    assert skip_flag_problems(copy, tmp_path / "scratch") == [
        "planted.yml: the job elsewhere's step 'Docs': passes --skip-example-runs; "
        "CI runs the scripts the guides' output blocks name"]


def test_a_job_named_smoke_in_another_workflow_may_not_skip_the_example_runs(tmp_path):
    other = ("on: [pull_request]\njobs:\n  smoke:\n    runs-on: ubuntu-latest\n    steps:\n"
             "      - name: Docs\n        run: uv run mime docs check --skip-example-runs\n")
    copy = _planted_workflows(tmp_path, SMOKE_DOCS_LINE, SMOKE_DOCS_LINE, extra=other)
    assert len(skip_flag_problems(copy, tmp_path / "scratch")) == 1, "a job id 'smoke' in another file is not the smoke job"


def test_a_job_that_may_skip_the_example_runs_and_makes_a_required_check_fails(tmp_path):
    required = (*REQUIRED, "smoke (macos-latest, Python 3.12)")
    assert skip_flag_problems(WORKFLOWS, tmp_path / "control", REQUIRED) == []
    assert skip_flag_problems(WORKFLOWS, tmp_path / "plant", required) == [
        "platforms.yml: the job smoke passes --skip-example-runs and makes a required check; "
        "CI runs the scripts the guides' output blocks name"]


def test_a_second_job_making_a_required_check_fails(tmp_path):
    """A job of another workflow taking a required check's name could pass it in the real job's place."""
    copy = tmp_path / "workflows"
    shutil.copytree(WORKFLOWS, copy)
    (copy / "planted.yml").write_text("on: [pull_request]\njobs:\n  shadow:\n    name: no private name or path\n"
                                      "    runs-on: ubuntu-latest\n    steps:\n      - run: 'true'\n", encoding="utf-8")
    assert "'no private name or path': 2 jobs make this required check; one does" in wiring_problems(
        copy, tmp_path / "scratch")
