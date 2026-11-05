"""The platforms workflow runs the tests `tests/platforms.txt` names and nothing else, and every one of them exists.

A list read only by a workflow rots silently: a renamed file is a path pytest reports as an error on one platform and
nothing on another, or a deleted one drops out of the smoke unseen. So the names are checked here, in the default lane,
and the workflow is read for what it must do: run the list through `scripts/platforms_smoke.py`, on both platforms and
two Pythons, install as the default lane does, set no encoding the user's machine would not have, and bound each job."""

import importlib.util
import re
import shutil
from pathlib import Path

import pytest
import yaml
from test_ci_wiring import BRANCH, OLD_BRANCH, OLD_NAME, PULL_REQUEST, REQUIRED, run_step

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "platforms.yml"
spec = importlib.util.spec_from_file_location("_platforms_smoke", ROOT / "scripts" / "platforms_smoke.py")
assert spec is not None and spec.loader is not None
platforms_smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(platforms_smoke)


def problems(root: Path) -> list[str]:
    """Everything wrong with the list under ``root``."""
    found: list[str] = []
    listed = platforms_smoke.selection(root)
    if not listed:
        found.append("the list names no test file")
    for name in listed:
        path = root / name
        if not (name.startswith("tests/") and path.name.startswith("test_") and path.suffix == ".py"):
            found.append(f"{name}: is not a test file under tests/")
        elif not path.is_file():
            found.append(f"{name}: does not exist")
    found += [f"{name}: listed twice" for name in sorted({name for name in listed if listed.count(name) > 1})]
    lines = (root / platforms_smoke.SELECTION).read_text(encoding="utf-8").splitlines()
    found += [f"{line!r}: has no reason after '#'" for line in lines
              if line.strip() and not line.startswith("#") and "#" not in line]
    return found


def test_every_file_the_selection_names_exists_and_has_a_reason():
    assert problems(ROOT) == []


@pytest.mark.parametrize("planted, expected", [
    ("tests/platforms/test_gone.py  # a file that was renamed\n", "does not exist"),
    ("tests/platforms/test_smoke.py  # twice\n", "listed twice"),
    ("tests/platforms/helper.py  # not a test file\n", "is not a test file"),
    ("tests/platforms/test_smoke.py\n", "has no reason"),
])
def test_a_planted_fault_in_a_copy_of_the_list_is_reported(tmp_path, planted, expected):
    shutil.copytree(ROOT / "tests" / "platforms", tmp_path / "tests" / "platforms")
    shutil.copy(ROOT / "tests" / "platforms.txt", tmp_path / "tests" / "platforms.txt")
    with (tmp_path / "tests" / "platforms.txt").open("a", encoding="utf-8") as stream:
        stream.write(planted)
    assert any(expected in problem for problem in problems(tmp_path)), problems(tmp_path)


def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def branch_problems(text: str, scratch: Path) -> list[str]:
    """What is wrong with how a copy of the workflow (``text``) names the development branch: its triggers, the base of
    its docs check, and any mention of the old name ``v0.3`` (``scratch`` holds the step runs)."""
    found: list[str] = []
    triggers = yaml.safe_load(text)[True]  # YAML 1.1 reads the key `on` as a boolean
    for event in ("pull_request", "push"):
        if triggers[event]["branches"] != [BRANCH]:
            found.append(f"{event} runs on {triggers[event]['branches']}, not [{BRANCH!r}] alone")
    found += [f"names the old branch name {OLD_BRANCH}: {line.strip()}" for line in text.splitlines()
              if OLD_NAME.search(line)]
    step = next(step for step in yaml.safe_load(text)["jobs"]["smoke"]["steps"] if step.get("name") == "Docs check")
    _, calls = run_step(step, PULL_REQUEST, 0, scratch / "ran")
    checks = [arguments for command, arguments, _ in calls if command == "uv"]
    if not (len(checks) == 1 and ["--base", f"origin/{BRANCH}"] == checks[0][checks[0].index("--base"):][:2]):
        found.append(f"the docs check does not compare with origin/{BRANCH}")
    status, calls = run_step(step, PULL_REQUEST, 0, scratch / "unreadable", (BRANCH,))
    if status == 0 or [call for call in calls if call[0] == "uv"]:
        found.append("with the branch unreadable, the docs check runs or the step exits 0")
    return found


def test_the_workflow_runs_on_pull_requests_and_pushes_to_the_development_branch_and_is_no_required_check(tmp_path):
    """On ``release/v0.3.0`` alone, never the old name ``v0.3``; and the docs check compares with ``origin/release/v0.3.0``
    where the step is run as GitHub runs it (``bash -e``, git and uv stubbed), and fails without running the check where
    that branch cannot be read (the stub reads every other name, so a fallback to the old one shows)."""
    assert branch_problems(WORKFLOW.read_text(encoding="utf-8"), tmp_path) == []
    names = [job.get("name", "") for job in workflow()["jobs"].values()]
    assert not set(names) & set(REQUIRED)


BRANCH_PLANTS = {
    "the-pull-request-trigger-drops-the-branch": (
        "  pull_request:\n    branches: [release/v0.3.0]\n", "  pull_request:\n    branches: []\n",
        f"pull_request runs on [], not [{BRANCH!r}] alone"),
    "the-push-trigger-on-the-old-name": (
        "  push:\n    branches: [release/v0.3.0]\n", "  push:\n    branches: [v0.3]\n",
        f"push runs on ['v0.3'], not [{BRANCH!r}] alone"),
    "the-old-name-back-in-the-pull-request-trigger": (
        "  pull_request:\n    branches: [release/v0.3.0]\n", "  pull_request:\n    branches: [release/v0.3.0, v0.3]\n",
        f"pull_request runs on ['release/v0.3.0', 'v0.3'], not [{BRANCH!r}] alone"),
    "the-old-name-in-a-comment": (
        "      - name: Docs check\n", "      # held against origin/v0.3\n      - name: Docs check\n",
        "names the old branch name v0.3: # held against origin/v0.3"),
    "the-docs-check-on-the-old-name": (
        "base=origin/release/v0.3.0\n", "base=origin/v0.3\n", f"the docs check does not compare with origin/{BRANCH}"),
    "the-docs-step-running-without-a-base": (
        '|| { echo "$base cannot be read" >&2; exit 1; }', '|| echo "$base cannot be read" >&2',
        "with the branch unreadable, the docs check runs or the step exits 0"),
    "the-docs-step-falling-back-to-the-old-name": (
        '|| { echo "$base cannot be read" >&2; exit 1; }', "|| base=origin/v0.3",
        "with the branch unreadable, the docs check runs or the step exits 0"),
}


@pytest.mark.parametrize("plant", BRANCH_PLANTS, ids=list(BRANCH_PLANTS))
def test_a_break_in_how_the_workflow_names_the_branch_fails_naming_it(tmp_path, plant):
    """The control of the test above: each plant fails it, every run."""
    old, new, problem = BRANCH_PLANTS[plant]
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"the plant's anchor {old!r} is not in platforms.yml once"
    assert problem in branch_problems(text.replace(old, new), tmp_path)


def test_each_job_is_bounded_and_the_matrix_is_both_platforms_on_two_pythons():
    jobs = workflow()["jobs"]
    assert jobs and all(isinstance(job.get("timeout-minutes"), int) for job in jobs.values())
    matrix = jobs["smoke"]["strategy"]["matrix"]
    assert sorted(matrix["os"]) == ["macos-latest", "windows-latest"]
    assert matrix["python-version"] == ["3.12", "3.14"] and jobs["smoke"]["strategy"]["fail-fast"] is False


def test_the_workflow_sets_up_as_the_default_lane_does_and_sets_no_encoding():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "uv sync --locked --extra verify" in text and "uv pip install --no-deps ./lab_fixture" in text
    assert "scripts/platforms_smoke.py" in text
    # Only a comment may name them: a step that set either would meet a machine no user has.
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "PYTHONUTF8" not in code and "PYTHONIOENCODING" not in code
    default = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for pinned in re.findall(r"uses: (\S+@[0-9a-f]{40})", text):
        assert pinned in default, f"{pinned} is not an action version ci.yml pins"
