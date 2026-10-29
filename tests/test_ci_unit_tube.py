"""CI stores the tube wall kind's table before the tests that read it (``.github/actions/unit-tube``): every job that
runs the tests uses the action after the environment is installed and before the tests, the action keys its cache by
the sha256 of the reference MIME ships (read from that file, so a new table's key changes with it), points
``MIME_ASSETS_DIR`` at the cached directory and fetches with ``mime assets fetch`` (a hit too). The tests themselves
never download: a missing asset fails by name. Control: a copy of the workflow with the step dropped from one job, and
one of the action with a fixed key, each fail naming the break, every run."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from mime import assets

ROOT = Path(__file__).resolve().parents[1]
ACTION = "./.github/actions/unit-tube"
TEST_JOBS = ("default-tests", "openmaterica-main-tests", "maddening-main-tests", "usd-core-newest")
NAME = "unit_tube_R0.997"


def problems(root: Path) -> list[str]:
    found: list[str] = []
    workflow = yaml.safe_load((root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    for name in TEST_JOBS:
        steps = workflow["jobs"][name]["steps"]
        used = [i for i, step in enumerate(steps) if step.get("uses") == ACTION]
        tests = [i for i, step in enumerate(steps) if step.get("id") == "tests"]
        installed = [i for i, step in enumerate(steps) if "uv sync" in str(step.get("run", ""))]
        if len(used) != 1 or len(tests) != 1 or len(installed) != 1 or not installed[0] < used[0] < tests[0]:
            found.append(f"{name}: the unit tube action is not used once, after the install and before the tests")
    action = yaml.safe_load((root / ".github" / "actions" / "unit-tube" / "action.yml").read_text(encoding="utf-8"))
    steps = action["runs"]["steps"]
    caches = [step for step in steps if str(step.get("uses", "")).startswith("actions/cache@")]
    if len(caches) != 1 or "steps.asset.outputs.sha256" not in caches[0]["with"]["key"] \
            or f"{NAME}-" not in caches[0]["with"]["key"]:
        found.append("the action's cache is not keyed by the asset's sha256")
    fetches = [step for step in steps if f"mime assets fetch {NAME}" in str(step.get("run", ""))]
    if len(fetches) != 1 or "if" in fetches[0]:
        found.append("the action does not fetch the asset on a hit too")
    if "MIME_ASSETS_DIR" not in str(steps[0]["run"]):
        found.append("the action does not set MIME_ASSETS_DIR")
    return found


def test_every_test_job_stores_the_asset_before_its_tests():
    assert problems(ROOT) == []


def test_the_action_reads_the_hash_of_the_reference_the_package_ships():
    script = yaml.safe_load((ROOT / ".github" / "actions" / "unit-tube" / "action.yml").read_text(encoding="utf-8"))[
        "runs"]["steps"][0]["run"]
    done = subprocess.run(["bash", "-e", "-c", script], cwd=ROOT, capture_output=True, text=True, check=False,
                          env={"PATH": "/usr/bin:/bin:/usr/local/bin", "RUNNER_TEMP": "/tmp",
                               "GITHUB_OUTPUT": "/dev/stdout", "GITHUB_ENV": "/dev/null"})
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"sha256={assets.reference(NAME).sha256}"


@pytest.mark.parametrize("plant", ["a job without the step", "a fixed key"])
def test_control_a_break_fails_naming_it_every_run(tmp_path, plant):
    shutil.copytree(ROOT / ".github", tmp_path / ".github")
    workflow, action = tmp_path / ".github" / "workflows" / "ci.yml", tmp_path / ".github" / "actions" / "unit-tube" / "action.yml"
    if plant == "a job without the step":
        text = workflow.read_text(encoding="utf-8")
        marker = "      - name: The unit tube table, in the cache shared by every job\n        uses: ./.github/actions/unit-tube\n"
        assert marker in text
        workflow.write_text(text.replace(marker, "", 1), encoding="utf-8")
    else:
        text = action.read_text(encoding="utf-8")
        assert "${{ steps.asset.outputs.sha256 }}" in text
        action.write_text(text.replace("${{ steps.asset.outputs.sha256 }}", "fixed"), encoding="utf-8")
    for _ in range(20):
        assert problems(tmp_path), plant
