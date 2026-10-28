"""CI's wiring of the checks it must run, read from ``.github/workflows/``:

- every job name v0.3's branch protection requires is made by one job, of a workflow that runs on a pull request into
  v0.3 and on a push to it, and that job is skipped by nothing but the schedule (a skipped job passes a required check);
- the default lane runs ``mime docs check --base origin/v0.3`` with its exit status intact;
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
#: v0.3's required status checks, by the names its branch protection gives them.
REQUIRED = ("default (Python 3.12)", "default (Python 3.13)", "default (Python 3.14)", "maddening-main",
            "openmaterica-main", "typecheck (pyright, Python 3.12)", "no private name or path")
DOCS_CHECK, SCAN = "default (Python 3.12)", "no private name or path"
EVENT_FILTER = "github.event_name != 'schedule'"
STEP_IFS = (None, "${{ !cancelled() }}")
NEUTERING = re.compile(r"\|\|\s*(?:true|:)(?![\w-])|\bset\s+\+e\b")
SECRET = "the-secret-stands-in"
#: What a run's expressions are, for a pull request into v0.3 and for a push to it.
PULL_REQUEST = {"github.event_name": "pull_request", "github.head_ref": "a-branch", "github.base_ref": "v0.3",
                "github.ref_name": "30/merge", "github.event.pull_request.base.sha": "BASE",
                "github.event.pull_request.head.sha": "HEAD", "github.sha": "MERGE", "matrix.python-version": "3.12",
                "secrets.CONFIDENTIAL_SCAN_KEY": SECRET}
PUSH = {**PULL_REQUEST, "github.event_name": "push", "github.head_ref": "", "github.base_ref": "",
        "github.ref_name": "v0.3", "github.event.before": "BEFORE", "github.sha": "AFTER",
        "github.event.pull_request.base.sha": "", "github.event.pull_request.head.sha": ""}
STUB = """#!/bin/sh
{ printf '%s' "${0##*/}"; for argument in "$@"; do printf '\\t%s' "$argument"; done
  printf '\\tkey=%s\\n' "${CONFIDENTIAL_SCAN_KEY-}"; } >> "$STUB_LOG"
[ "${0##*/}" = git ] && exit 0
exit "$STUB_EXIT"
"""


def workflows(directory: Path) -> list[tuple[str, dict[str, object]]]:
    return [(path.name, yaml.safe_load(path.read_text(encoding="utf-8")))
            for path in sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")])]


def names_of(job_id: str, job: dict[str, object]) -> list[str]:
    """The check names a job makes: its name, each ``${{ matrix.<axis> }}`` in it taken over the matrix's values."""
    name = str(job.get("name", job_id))
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    axes = re.findall(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}", name)
    names = []
    for values in itertools.product(*(matrix.get(axis, []) for axis in axes)):
        made = name
        for axis, value in zip(axes, values, strict=True):
            made = re.sub(rf"\$\{{\{{\s*matrix\.{re.escape(axis)}\s*\}}\}}", str(value), made)
        names.append(made)
    return names


def runs_on_v0_3(workflow: dict[str, object]) -> bool:
    triggers = workflow.get("on", workflow.get(True)) or {}
    return all("v0.3" in ((triggers.get(event) or {}).get("branches") or []) for event in ("pull_request", "push"))


def evaluate(text: str, context: dict[str, str]) -> str:
    """``text`` with each ``${{ a || b }}`` replaced as GitHub would, from ``context`` (unknown names are empty)."""
    def value(match: re.Match[str]) -> str:
        return next((context[name.strip()] for name in match.group(1).split("||") if context.get(name.strip())), "")
    return re.sub(r"\$\{\{(.*?)\}\}", value, text)


def run_step(step: dict[str, object], context: dict[str, str], exit_status: int, scratch: Path):
    """The step's script run in ``bash -e`` with ``uv``, ``python3`` and ``git`` stubbed, each stub recording its
    arguments and the key it saw, ``uv`` and ``python3`` exiting ``exit_status``: the script's exit status, and each
    stubbed call as its command, its arguments and the key."""
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
                               "STUB_EXIT": str(exit_status), **env})
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


def wiring_problems(directory: Path, scratch: Path) -> list[str]:
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
        if not runs_on_v0_3(workflow):
            found.append(f"{name!r}: its workflow does not run on both a pull request into v0.3 and a push to it")
        if job.get("if") not in (None, EVENT_FILTER):
            found.append(f"{name!r}: runs only if {job.get('if')!r}, and a skipped job passes a required check")
    if len(made.get(DOCS_CHECK, [])) == 1:
        found += docs_check_problems(made[DOCS_CHECK][0][1], scratch / "docs")
    if len(made.get(SCAN, [])) == 1:
        found += scan_problems(made[SCAN][0][1], scratch / "scan")
    return found


def docs_check_problems(job: dict[str, object], scratch: Path) -> list[str]:
    found, step = one_step(DOCS_CHECK, job, "mime docs check")
    if step is None:
        return found
    where = f"{DOCS_CHECK}: the step {step.get('name')!r}"
    found += step_problems(where, step) + exit_problems(where, step, PULL_REQUEST, scratch)
    _, calls = run_step(step, PULL_REQUEST, 0, scratch / "arguments")
    if not any(" ".join(arguments).find("mime docs check") >= 0 and ("--base", "origin/v0.3") in pairs(arguments)
               for command, arguments, _ in calls if command == "uv"):
        found.append(f"{where}: does not run mime docs check --base origin/v0.3")
    return found


def scan_problems(job: dict[str, object], scratch: Path) -> list[str]:
    found, step = one_step(SCAN, job, "scripts/check_confidential.py")
    if step is None:
        return found
    where = f"{SCAN}: the step {step.get('name')!r}"
    found += step_problems(where, step) + exit_problems(where, step, PULL_REQUEST, scratch)
    if (step.get("env") or {}).get("CONFIDENTIAL_SCAN_KEY") != "${{ secrets.CONFIDENTIAL_SCAN_KEY }}":
        found.append(f"{where}: its environment does not set CONFIDENTIAL_SCAN_KEY from the secret")
    expected = {"pull request": (PULL_REQUEST, {("--messages", "BASE..HEAD"), ("--diffs", "BASE..HEAD"),
                                                ("--ref-name", "a-branch")}),
                "push": (PUSH, {("--messages", "BEFORE..AFTER"), ("--diffs", "BEFORE..AFTER")})}
    for event, (context, wanted) in expected.items():
        _, calls = run_step(step, context, 0, scratch / event.replace(" ", "-"))
        scans = [(arguments, key) for command, arguments, key in calls
                 if command == "python3" and arguments[:1] == ["scripts/check_confidential.py"]]
        if len(scans) != 1:
            found.append(f"{where}: on a {event}, runs scripts/check_confidential.py {len(scans)} times; once")
            continue
        arguments, key = scans[0]
        given = pairs(arguments)
        found += [f"{where}: on a {event}, does not pass {option} {value}" for option, value in sorted(wanted - given)]
        found += [f"{where}: on a {event} into v0.3, passes {option} {value}"
                  for option, value in sorted(given - wanted)]
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
SCAN_STEP = f"{SCAN}: the step \"No private name or path in the tracked files, the commits or the branch's name\""
SCAN_LINE = '            ${public:+--public "$public"} ${HEAD_REF:+--ref-name "$HEAD_REF"}\n'

#: Each plant: the text replaced in ci.yml, its replacement, and the problem it must raise.
PLANTS = {
    "the-docs-check-step-if-false": (
        "        if: ${{ !cancelled() }}\n        run: |\n          status=0\n",
        "        if: false\n        run: |\n          status=0\n",
        f"{DOCS_STEP}: runs only if False; it runs on every lane's run, or under !cancelled()"),
    "the-confidential-job-if-false": (
        "    name: no private name or path\n    if: github.event_name != 'schedule'\n",
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
    "the-base-dropped": ("mime docs check --base origin/v0.3", "mime docs check",
                         f"{DOCS_STEP}: does not run mime docs check --base origin/v0.3"),
    "the-docs-step-continues-on-error": (
        "        if: ${{ !cancelled() }}\n        run: |\n          status=0\n",
        "        if: ${{ !cancelled() }}\n        continue-on-error: true\n        run: |\n          status=0\n",
        f"{DOCS_STEP}: continue-on-error hides its failure from the job"),
    "set-plus-e-in-the-scan": ("          public=\"\"\n", "          set +e\n          public=\"\"\n",
                               f"{SCAN_STEP}: its script holds an '|| true' or a 'set +e'"),
    "the-scan-job-renamed": ("    name: no private name or path\n", "    name: confidentiality\n",
                             "'no private name or path': 0 jobs make this required check; one does"),
    "a-python-dropped-from-the-matrix": (
        'python-version: ["3.12", "3.13", "3.14"]', 'python-version: ["3.12", "3.13"]',
        "'default (Python 3.14)': 0 jobs make this required check; one does"),
    "no-pull-request-into-v0.3": (
        "  pull_request:\n    branches: [main, v0.3]\n", "  pull_request:\n    branches: [main]\n",
        "'typecheck (pyright, Python 3.12)': its workflow does not run on both a pull request into v0.3 and a push "
        "to it"),
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


def test_a_second_job_making_a_required_check_fails(tmp_path):
    """A job of another workflow taking a required check's name could pass it in the real job's place."""
    copy = tmp_path / "workflows"
    shutil.copytree(WORKFLOWS, copy)
    (copy / "planted.yml").write_text("on: [pull_request]\njobs:\n  shadow:\n    name: no private name or path\n"
                                      "    runs-on: ubuntu-latest\n    steps:\n      - run: 'true'\n", encoding="utf-8")
    assert "'no private name or path': 2 jobs make this required check; one does" in wiring_problems(
        copy, tmp_path / "scratch")
