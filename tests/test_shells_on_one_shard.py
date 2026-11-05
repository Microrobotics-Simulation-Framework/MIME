"""CI installs the shells the completion tests run under (zsh and fish) on the one shard of a lane that holds
``tests/cli/test_completion.py``, not on all ten: the install is an ``apt-get`` that took 8 min 19 s on one runner
(run 42049802999, the default lane's shard 5, lost to its 20-minute limit) and over 100 s on 23 of 943 shard jobs of the last
twenty runs, with no test of the shard using either shell. The tests fail, not skip, under CI when a shell is missing, so a
wrong holder is red by name; this file keeps the holder right (``scripts/shard_of_file.py`` asks the plugin's own deal), the
workflow wired to it in every sharded lane, and no other test file asking for either shell. The install itself
(``scripts/install_shells.sh``) cuts an attempt at a limit and makes it again, since the one shard that installs was
itself met by a 9 min 51 s install in this change's first run (run 42058520080, ``maddening-main`` shard 8).

The controls are fixed plants, each failing on every run."""

import importlib.util
import re
import subprocess
import sys
import time
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
#: The step that asks, word for word: the count it is given, the output it writes and the file it asks about.
ASK = ('echo "holder=$(uv run --no-sync python scripts/shard_of_file.py tests/cli/test_completion.py --of "$SHARDS")" '
       '>> "$GITHUB_OUTPUT"')


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
        installs = [step for step in steps if "install_shells.sh" in step.get("run", "")]
        if len(asks) != 1 or f"scripts/shard_of_file.py {COMPLETION} --of" not in asks[0].get("run", ""):
            found.append(f"{name}: does not ask scripts/shard_of_file.py for the shard of {COMPLETION}")
        elif asks[0].get("env", {}).get("SHARDS") != "${{ needs.shards.outputs.count }}":
            found.append(f"{name}: asks for the shard of a count other than the lane's")
        elif asks[0]["run"].strip() != ASK:
            found.append(f"{name}: the run line of the ask is not the one that writes the holder for the lane's count")
        if len(installs) != 1 or installs[0]["run"].strip() != "bash scripts/install_shells.sh":
            found.append(f"{name}: does not install zsh and fish exactly once, by scripts/install_shells.sh")
        elif "continue-on-error" in installs[0]:
            found.append(f"{name}: the install may fail without failing the job (continue-on-error)")
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
    ('--of "$SHARDS")" >> "$GITHUB_OUTPUT"', '--of 10)" >> "$GITHUB_OUTPUT"', "the run line of the ask"),
    ('echo "holder=$(uv run', 'echo "holdr=$(uv run', "the run line of the ask"),
    ("        if: steps.shells.outputs.holder == matrix.shard\n",
     "        if: steps.shells.outputs.holder == matrix.shard\n        continue-on-error: true\n", "continue-on-error"),
], ids=["installs-on-all-shards", "asks-for-another-file", "asks-for-a-fixed-count", "writes-the-run-line-with-a-fixed-count",
        "writes-the-holder-under-another-name", "lets-the-install-fail"])
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


INSTALL = ROOT / "scripts" / "install_shells.sh"
needs_posix = pytest.mark.skipif(sys.platform == "win32", reason="a CI Linux runner's script")


def fake_tools(folder: Path, apt_body: str, *, recorded: bool = False) -> dict[str, str]:
    """An environment whose ``sudo``, ``dpkg`` and ``apt-get`` are scripts in ``folder``, so that the script is run for
    real and installs nothing. ``apt-get`` runs ``apt_body`` (``$COUNT_FILE`` holds the attempts so far, ``$attempt``
    the number of this one) and logs its arguments to ``$COUNT_FILE.apt``; ``sudo`` logs that it was used and runs its
    arguments. With ``recorded``, ``timeout`` and ``sleep`` are fakes that log their arguments to ``$COUNT_FILE.calls``
    and cut and wait for nothing (``timeout`` runs the command after its own options), so that the script's defaults
    can be read off a run that takes no time."""
    folder.mkdir()
    (folder / "sudo").write_text('#!/bin/sh\necho "sudo $1" >> "$COUNT_FILE.sudo"\nexec "$@"\n', encoding="utf-8")
    (folder / "dpkg").write_text('#!/bin/sh\necho "dpkg $*" >> "$COUNT_FILE.dpkg"\n', encoding="utf-8")
    (folder / "apt-get").write_text('#!/bin/sh\necho x >> "$COUNT_FILE"\necho "$*" >> "$COUNT_FILE.apt"\n'
                                    'attempt=$(wc -l < "$COUNT_FILE")\n' + apt_body, encoding="utf-8")
    if recorded:
        (folder / "timeout").write_text(
            '#!/bin/sh\necho "timeout $1 $2 $3" >> "$COUNT_FILE.calls"\nshift 2\nexec "$@"\n', encoding="utf-8")
        (folder / "sleep").write_text('#!/bin/sh\necho "sleep $1" >> "$COUNT_FILE.calls"\n', encoding="utf-8")
    for tool in folder.iterdir():
        tool.chmod(0o755)
    return {"PATH": f"{folder}:/usr/bin:/bin", "COUNT_FILE": str(folder / "count")}


def run_install(tmp_path: Path, apt_body: str, *, recorded: bool = False,
                **settings: str) -> tuple[subprocess.CompletedProcess[str], int]:
    environment = fake_tools(tmp_path / "bin", apt_body, recorded=recorded) | settings
    if not recorded:  # the recorded runs wait for nothing, so they keep the script's own pause
        environment.setdefault("SHELLS_PAUSE_SECONDS", "0")
    done = subprocess.run(["bash", str(INSTALL)], env=environment, capture_output=True, text=True, check=False, timeout=120)
    counted = Path(environment["COUNT_FILE"])
    return done, len(counted.read_text().splitlines()) if counted.exists() else 0


def lines(tmp_path: Path, suffix: str) -> list[str]:
    path = tmp_path / "bin" / f"count.{suffix}"
    return path.read_text().splitlines() if path.exists() else []


@needs_posix
def test_an_attempt_that_stalls_is_cut_and_made_again_and_the_second_installs(tmp_path):
    done, attempts = run_install(tmp_path, '[ "$attempt" -ge 2 ] && exit 0\nexec sleep 30\n', SHELLS_ATTEMPT_SECONDS="1")
    assert done.returncode == 0 and attempts == 2, done
    assert "attempt 1 of 3 failed or ran over 1 s" in done.stdout and "installed on attempt 2 of 3" in done.stdout


@needs_posix
def test_control_without_the_cut_a_stalled_attempt_is_waited_for(tmp_path):
    """The same stall with a limit longer than it: nothing is cut, so the one attempt is the only one."""
    done, attempts = run_install(tmp_path, "sleep 2\nexit 0\n", SHELLS_ATTEMPT_SECONDS="60")
    assert done.returncode == 0 and attempts == 1 and "attempt 1 of 3" in done.stdout


@needs_posix
def test_the_last_attempt_has_a_limit_of_its_own_so_an_install_slower_than_the_cut_still_passes(tmp_path):
    """The history's slow installs (121 to 320 s) all passed on one attempt: the cut ends the first attempts only."""
    done, attempts = run_install(tmp_path, '[ "$attempt" -ge 2 ] && { sleep 3; exit 0; }\nexec sleep 30\n',
                                 SHELLS_ATTEMPTS="2", SHELLS_ATTEMPT_SECONDS="1", SHELLS_FINAL_SECONDS="20")
    assert done.returncode == 0 and attempts == 2, done
    assert "installed on attempt 2 of 2" in done.stdout


@needs_posix
def test_control_the_last_attempt_is_cut_at_its_own_limit_and_the_step_then_fails(tmp_path):
    done, attempts = run_install(tmp_path, "exec sleep 30\n", SHELLS_ATTEMPTS="2", SHELLS_ATTEMPT_SECONDS="1",
                                 SHELLS_FINAL_SECONDS="2")
    assert done.returncode == 1 and attempts == 2, done
    assert "attempt 2 of 2 failed or ran over 2 s" in done.stdout


@needs_posix
def test_a_failing_install_is_tried_the_set_number_of_times_and_then_fails_the_step(tmp_path):
    done, attempts = run_install(tmp_path, "exit 100\n", SHELLS_ATTEMPTS="2", SHELLS_ATTEMPT_SECONDS="5")
    assert done.returncode == 1 and attempts == 2, done
    assert "were not installed in 2 attempts" in done.stdout
    assert len(lines(tmp_path, "dpkg")) == 1, "the half-done install is repaired between attempts, not after the last"


@needs_posix
def test_the_defaults_are_the_ones_the_step_documents_and_the_worst_case_is_inside_the_job(tmp_path):
    """Run with no ``SHELLS_*`` setting, against a ``timeout`` and a ``sleep`` that record their arguments: the first
    attempts are cut at 120 s, the last at 360 s, ``dpkg --configure -a`` at 60 s, a pause of 15 s follows each failed
    attempt that is not the last, apt gets what it needs (as root, unattended, without its recommendations, zsh and
    fish), and the worst case that the script prints is under the job's 20 minutes and under 15."""
    done, attempts = run_install(tmp_path, '[ "$attempt" -ge 3 ] && exit 0\nexit 100\n', recorded=True)
    assert done.returncode == 0 and attempts == 3, done
    assert lines(tmp_path, "calls") == [
        "timeout --kill-after=10 120 apt-get", "timeout --kill-after=10 60 dpkg", "sleep 15",
        "timeout --kill-after=10 120 apt-get", "timeout --kill-after=10 60 dpkg", "sleep 15",
        "timeout --kill-after=10 360 apt-get"]
    assert lines(tmp_path, "apt") == ["install -y --no-install-recommends zsh fish"] * 3
    assert lines(tmp_path, "dpkg") == ["dpkg --configure -a"] * 2
    assert lines(tmp_path, "sudo") == ["sudo timeout"] * 5, "every apt-get (3) and dpkg (2) runs as root, under the cut"
    worst = int(re.search(r"worst case ([0-9]+) s", done.stdout).group(1))
    assert worst == (120 + 10 + 60 + 10 + 15) * 2 + 360 + 10 and worst < 15 * 60, worst


@needs_posix
def test_an_apt_that_cannot_get_its_lock_is_not_asked_again_at_once(tmp_path):
    """apt exits 100 at once while another process holds its lock; the retries must be spaced, not three in a row."""
    done, attempts = run_install(tmp_path, "exit 100\n", recorded=True)
    assert done.returncode == 1 and attempts == 3, done
    assert [line for line in lines(tmp_path, "calls") if line.startswith("sleep")] == ["sleep 15", "sleep 15"]


@needs_posix
def test_a_dpkg_that_hangs_is_cut_and_does_not_hold_the_next_attempt(tmp_path):
    folder = tmp_path / "bin"
    environment = fake_tools(folder, '[ "$attempt" -ge 2 ] && exit 0\nexit 100\n')
    (folder / "dpkg").write_text("#!/bin/sh\nexec sleep 30\n", encoding="utf-8")
    environment |= {"SHELLS_DPKG_SECONDS": "1", "SHELLS_PAUSE_SECONDS": "0"}
    started = time.monotonic()
    done = subprocess.run(["bash", str(INSTALL)], env=environment, capture_output=True, text=True, check=False, timeout=120)
    assert done.returncode == 0 and "installed on attempt 2 of 3" in done.stdout, done
    assert time.monotonic() - started < 15
