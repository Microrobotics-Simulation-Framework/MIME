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
from test_ci_wiring import REQUIRED

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


def test_the_workflow_runs_on_pull_requests_and_pushes_to_v0_3_and_is_no_required_check():
    triggers = workflow()[True]  # YAML 1.1 reads the key `on` as a boolean
    assert triggers["pull_request"]["branches"] == ["v0.3"] and triggers["push"]["branches"] == ["v0.3"]
    names = [job.get("name", "") for job in workflow()["jobs"].values()]
    assert not set(names) & set(REQUIRED)


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
