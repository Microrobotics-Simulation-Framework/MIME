"""CI installs the shells the completion tests run under (zsh and fish) on the one shard of a lane that holds
``tests/cli/test_completion.py``, not on all ten: the install is an ``apt-get`` that took 8 min 19 s on one runner
(run 42049802999, the default lane's shard 5, lost to its 20-minute limit) and over 100 s on 23 of 943 shard jobs of the last
twenty runs, with no test of the shard using either shell. The tests fail, not skip, under CI when a shell is missing, so a
wrong holder is red by name; this file keeps the holder right (``scripts/shard_of_file.py`` asks the plugin's own deal), the
workflow wired to it in every sharded lane, and no other test file asking for either shell.

The controls are fixed plants, each failing on every run."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import _sharding
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
COMPLETION = "tests/cli/test_completion.py"
#: The lanes that run as shards, and the lane that runs whole (the weekly one, which installs the shells itself).
SHARDED = ("default-tests", "openmaterica-main-tests", "maddening-main-tests")
UNSHARDED = "usd-core-newest"


def _script():
    spec = importlib.util.spec_from_file_location("_shard_of_file", ROOT / "scripts" / "shard_of_file.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shard_of_file = _script()


def kept_by(shard: int, of: int, nodeids: list[str]) -> list[str]:
    """What the real plugin keeps of ``nodeids`` as shard ``shard`` of ``of``."""
    items = [SimpleNamespace(nodeid=nodeid) for nodeid in nodeids]
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: None))
    _sharding.Plugin(shard, of, _sharding.load_weights(), None).pytest_collection_modifyitems(config, items)
    return [item.nodeid for item in items]


@pytest.mark.parametrize("of", [1, 2, 3, 6, 10, 12])
def test_the_shard_the_script_names_is_the_one_shard_the_plugin_keeps_the_completion_tests_in(of):
    nodeids = [f"{COMPLETION}::test_{name}[{shell}]" for name in ("script", "replies") for shell in ("bash", "zsh", "fish")]
    holder = shard_of_file.shard_of_file(COMPLETION, of)
    assert 1 <= holder <= of
    assert [shard for shard in range(1, of + 1) if kept_by(shard, of, nodeids)] == [holder]
    assert kept_by(holder, of, nodeids) == nodeids
    if of > 1:  # the control: the shard after it keeps none, so the check can tell a wrong holder
        assert kept_by(holder % of + 1, of, nodeids) == []


def test_the_script_prints_the_shard_and_refuses_a_file_that_is_not_there():
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "shard_of_file.py"), COMPLETION, "--of", "10"],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0 and done.stdout.strip() == str(shard_of_file.shard_of_file(COMPLETION, 10)), done
    missing = subprocess.run([sys.executable, str(ROOT / "scripts" / "shard_of_file.py"), "tests/cli/test_nothing.py",
                              "--of", "10"], capture_output=True, text=True, check=False)
    assert missing.returncode == 2 and "is not a file" in missing.stderr and missing.stdout == ""


def workflow_problems(text: str) -> list[str]:
    """What is wrong with how a lane installs the shells: a sharded lane asks the script for the completion file's
    shard and installs on that shard alone, and the lane that runs whole installs unconditionally."""
    jobs = yaml.safe_load(text)["jobs"]
    found: list[str] = []
    for name in SHARDED:
        steps = jobs[name]["steps"]
        asks = [step for step in steps if step.get("id") == "shells"]
        installs = [step for step in steps if "apt-get install" in step.get("run", "")]
        if len(asks) != 1 or f"scripts/shard_of_file.py {COMPLETION} --of" not in asks[0].get("run", ""):
            found.append(f"{name}: does not ask scripts/shard_of_file.py for the shard of {COMPLETION}")
        elif asks[0].get("env", {}).get("SHARDS") != "${{ needs.shards.outputs.count }}":
            found.append(f"{name}: asks for the shard of a count other than the lane's")
        if len(installs) != 1 or "zsh" not in installs[0]["run"] or "fish" not in installs[0]["run"]:
            found.append(f"{name}: does not install zsh and fish exactly once")
        elif installs[0].get("if") != "steps.shells.outputs.holder == matrix.shard":
            found.append(f"{name}: installs the shells on a shard other than the one that holds them, or on all")
        elif asks and steps.index(asks[0]) > steps.index(installs[0]):
            found.append(f"{name}: installs before asking")
    whole = [step for step in jobs[UNSHARDED]["steps"] if "apt-get install" in step.get("run", "")]
    if len(whole) != 1 or "if" in whole[0]:
        found.append(f"{UNSHARDED}: runs the whole suite and must install the shells unconditionally")
    return found


def test_every_sharded_lane_installs_the_shells_on_the_completion_tests_shard_alone():
    assert workflow_problems(CI.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("old, new, expected", [
    ("        if: steps.shells.outputs.holder == matrix.shard\n", "", "installs the shells on a shard other than"),
    ("scripts/shard_of_file.py tests/cli/test_completion.py", "scripts/shard_of_file.py tests/cli/test_other.py",
     "does not ask scripts/shard_of_file.py"),
    ("          SHARDS: ${{ needs.shards.outputs.count }}\n", "          SHARDS: '10'\n", "a count other than the lane's"),
], ids=["installs-on-all-shards", "asks-for-another-file", "asks-for-a-fixed-count"])
def test_control_a_break_in_the_wiring_is_named(old, new, expected):
    text = CI.read_text(encoding="utf-8")
    assert text.count(old) == len(SHARDED), old
    assert any(expected in problem for problem in workflow_problems(text.replace(old, new))), old


WORD = re.compile(r"\b(?:zsh|fish)\b")


def files_asking_for_a_shell(root: Path) -> list[str]:
    """The test files under ``root`` that name zsh or fish."""
    return sorted(str(path.relative_to(root)) for path in root.rglob("*.py") if WORD.search(path.read_text(encoding="utf-8")))


def test_no_test_file_but_the_completion_tests_asks_for_zsh_or_fish():
    """The install is on one shard because one file uses the shells; a second file using them would run, on another
    shard, with no shell (and fail there by name). The control: the scan finds a file that asks."""
    asking = files_asking_for_a_shell(ROOT / "tests")
    assert set(asking) <= {"cli/test_completion.py", "test_shells_on_one_shard.py"}, asking
    assert "cli/test_completion.py" in asking


def test_control_the_scan_finds_a_file_that_asks_for_a_shell(tmp_path):
    (tmp_path / "test_other.py").write_text('import shutil\nshutil.which("zsh")\n', encoding="utf-8")
    (tmp_path / "test_plain.py").write_text("selfish = 1\n", encoding="utf-8")
    assert files_asking_for_a_shell(tmp_path) == ["test_other.py"]
