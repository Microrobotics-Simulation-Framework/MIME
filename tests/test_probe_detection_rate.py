"""``.github/probe/detection_rate.py`` measures how often a control catches its planted error, in fresh processes.

Read here with scripts and pytest nodes written for the purpose, each reading the plant's environment variable and the seed:
the classification of a run (caught, not caught, inconclusive: a crash, a timeout, a skipped or an erroring test is never
"not caught"), the plant removed from the clean arm even when the caller's environment holds it, one distinct seed per run
on both arms, and the exact lower bound. Controls: a control that misses one seed in four is measured at three in four with
its missed seeds named; a control that fires with nothing planted is flagged at its seeds; each makes ``every_run`` false and
``--strict`` exit 1, every run (the targets are deterministic in the seed)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("mime_probe_detection_rate", ROOT / ".github" / "probe" / "detection_rate.py")
assert _SPEC is not None and _SPEC.loader is not None
detection_rate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = detection_rate
_SPEC.loader.exec_module(detection_rate)

PLANT = "TEST_PLANT_ERROR"

#: The target: when the plant is on it "detects" (exit 1) at every seed but those named in MISS_EVERY's multiples; with the
#: plant off it fires (a false alarm) at the seeds named in FALSE_ALARM_AT; SEED_LOG gets the seed each run saw.
SCRIPT = """
import os, sys
seed = int(os.environ["DETECTION_SEED"])
with open(os.environ["SEED_LOG"], "a") as log:
    log.write(f"{seed} {os.environ.get('TEST_PLANT_ERROR', '-')}\\n")
mode = os.environ.get("MODE", "exact")
planted = "TEST_PLANT_ERROR" in os.environ
if mode == "crash":
    sys.exit(2)
if mode == "slow":
    import time; time.sleep(30)
if planted:
    miss_every = int(os.environ.get("MISS_EVERY", "0"))
    sys.exit(0 if miss_every and seed % miss_every == 0 else 1)
sys.exit(1 if str(seed) in os.environ.get("FALSE_ALARM_AT", "").split(",") else 0)
"""


@pytest.fixture
def target(tmp_path, monkeypatch):
    script = tmp_path / "target.py"
    script.write_text(SCRIPT, encoding="utf-8")
    monkeypatch.setenv("SEED_LOG", str(tmp_path / "seeds.log"))
    return script


def measure(tmp_path, *extra, script=None, tail=()):
    output = tmp_path / "report.json"
    flags = ["--node", extra[0]] if extra and "::" in extra[0] else ["--script", str(script)]
    rest = extra[1:] if extra and "::" in extra[0] else extra
    status = detection_rate.main([*flags, "--plant-var", PLANT, "--output", str(output), "--cwd", str(tmp_path), *rest,
                                  *(["--", *tail] if tail else [])])
    return status, json.loads(output.read_text(encoding="utf-8"))


def test_a_control_that_catches_every_run_is_measured_at_n_of_n_with_a_quiet_clean_arm(tmp_path, target):
    status, report = measure(tmp_path, "--runs", "8", "--strict", script=target)
    assert status == 0 and report["every_run"]
    assert report["planted"]["caught_of_runs"] == "8/8" and report["planted"]["missed_seeds"] == []
    assert report["clean"]["flagged"] == 0 and report["clean"]["runs"] == 8
    # N clean catches show a rate no higher than 0.05 ** (1 / N): eight of eight cannot show the one-in-eight miss away
    assert report["planted"]["lower_bound_95"] == pytest.approx(0.05 ** (1 / 8)) and report["planted"]["lower_bound_95"] < 0.7


def test_each_run_is_a_fresh_process_with_its_own_seed_and_the_plant_only_on_the_planted_arm(tmp_path, target):
    status, report = measure(tmp_path, "--runs", "5", "--seed-base", "100", script=target)
    assert status == 0
    lines = [line.split() for line in (tmp_path / "seeds.log").read_text(encoding="utf-8").splitlines()]
    planted, clean = lines[:5], lines[5:]
    assert [int(seed) for seed, _ in planted] == [100, 101, 102, 103, 104] == [int(seed) for seed, _ in clean]
    assert {value for _, value in planted} == {"1"} and {value for _, value in clean} == {"-"}


def test_the_plant_is_removed_from_the_clean_arm_even_when_the_callers_environment_holds_it(tmp_path, target, monkeypatch):
    monkeypatch.setenv(PLANT, "inherited")
    status, report = measure(tmp_path, "--runs", "3", "--plant-value", "2.5e-6", script=target)
    lines = [line.split() for line in (tmp_path / "seeds.log").read_text(encoding="utf-8").splitlines()]
    assert {value for _, value in lines[:3]} == {"2.5e-6"} and {value for _, value in lines[3:]} == {"-"}
    assert report["clean"]["flagged"] == 0


def test_control_a_control_that_misses_one_seed_in_four_is_measured_with_its_missed_seeds(tmp_path, target, monkeypatch):
    monkeypatch.setenv("MISS_EVERY", "4")
    status, report = measure(tmp_path, "--runs", "12", "--seed-base", "1", "--strict", script=target)
    assert status == 1 and not report["every_run"]
    assert report["planted"]["missed_seeds"] == [4, 8, 12] and report["planted"]["caught_of_runs"] == "9/12"
    assert report["planted"]["detection_rate"] == 0.75
    assert report["planted"]["lower_bound_95"] < 0.75  # the bound is below the measured rate, always
    assert len(report["planted"]["records"]) == 3 and "output_tail" in report["planted"]["records"][0]


def test_control_a_control_that_fires_with_nothing_planted_is_flagged_at_its_seeds(tmp_path, target, monkeypatch):
    monkeypatch.setenv("FALSE_ALARM_AT", "2,3")
    status, report = measure(tmp_path, "--runs", "6", "--clean-runs", "4", "--strict", script=target)
    assert status == 1 and report["planted"]["caught"] == 6 and not report["every_run"]
    assert report["clean"]["flagged"] == 2 and report["clean"]["flagged_seeds"] == [2, 3] and report["clean"]["runs"] == 4


def test_a_crash_and_a_timeout_are_inconclusive_and_count_against_caught_of_n(tmp_path, target, monkeypatch):
    monkeypatch.setenv("MODE", "crash")
    status, report = measure(tmp_path, "--runs", "3", "--strict", script=target)
    assert status == 1 and report["planted"]["caught"] == 0 and report["planted"]["inconclusive"] == 3
    assert report["planted"]["missed_seeds"] == [] and report["planted"]["inconclusive_seeds"] == [0, 1, 2]
    monkeypatch.setenv("MODE", "slow")
    status, report = measure(tmp_path, "--runs", "1", "--timeout", "1", script=target)
    assert report["planted"]["inconclusive"] == 1 and report["planted"]["detection_rate"] == 0.0


def test_a_script_gets_the_arguments_after_the_double_dash(tmp_path):
    script = tmp_path / "args.py"
    script.write_text("import sys\nsys.exit(1 if sys.argv[1:] == ['--size', '3'] else 0)\n", encoding="utf-8")
    status, report = measure(tmp_path, "--runs", "2", script=script, tail=["--size", "3"])
    assert report["planted"]["caught_of_runs"] == "2/2" and report["script_arguments"] == ["--size", "3"]


NODES = """
import os
import pytest

def test_detects():
    assert "TEST_PLANT_ERROR" not in os.environ  # fails, which is the oracle catching the plant, when it is planted

def test_skips_when_planted():
    if "TEST_PLANT_ERROR" in os.environ:
        pytest.skip("the plant needs a draw this machine lacks")

@pytest.fixture
def broken():
    raise RuntimeError("setup error")

def test_errors(broken):
    pass

def test_seeded():
    assert int(os.environ["DETECTION_SEED"]) % 2 == 0 or "TEST_PLANT_ERROR" not in os.environ
"""


@pytest.fixture
def nodes(tmp_path):
    (tmp_path / "test_nodes.py").write_text(NODES, encoding="utf-8")
    return "test_nodes.py"


def test_a_node_that_fails_when_planted_is_caught_and_passes_clean(tmp_path, nodes):
    status, report = measure(tmp_path, f"{nodes}::test_detects", "--runs", "2", "--strict")
    assert status == 0 and report["planted"]["caught_of_runs"] == "2/2" and report["clean"]["flagged"] == 0


@pytest.mark.parametrize("name", ["test_skips_when_planted", "test_errors", "test_does_not_exist"])
def test_control_a_skipped_node_an_erroring_one_and_a_missing_one_are_inconclusive_never_not_caught(tmp_path, nodes, name):
    status, report = measure(tmp_path, f"{nodes}::{name}", "--runs", "1", "--clean-runs", "1")
    planted = report["planted"]
    assert planted["caught"] == 0 and planted["missed_seeds"] == [] and planted["inconclusive"] == 1  # nothing ran: no miss


def test_the_seed_reaches_a_node_through_the_environment_and_a_miss_names_its_seed(tmp_path, nodes):
    status, report = measure(tmp_path, f"{nodes}::test_seeded", "--runs", "3", "--clean-runs", "1", "--seed-base", "10")
    assert report["planted"]["missed_seeds"] == [10, 12] == [
        seed for seed in range(10, 13) if seed % 2 == 0]
    assert report["planted"]["caught_of_runs"] == "1/3"


def test_classify_reads_pytest_summaries_and_the_scripts_exit_codes():
    classify = detection_rate.classify
    assert classify(1, "FAILED x\n1 failed, 2 passed in 0.1s\n", pytest=True) == "caught"
    assert classify(1, "1 error in 0.1s\n", pytest=True) == "inconclusive"
    assert classify(0, "3 passed, 1 skipped in 0.1s\n", pytest=True) == "not_caught"
    assert classify(0, "1 skipped in 0.1s\n", pytest=True) == "inconclusive"
    assert classify(5, "no tests ran in 0.1s\n", pytest=True) == "inconclusive"
    assert classify(None, "", pytest=True) == "inconclusive"
    assert classify(1, "", pytest=False) == "caught" and classify(0, "", pytest=False) == "not_caught"
    assert classify(2, "", pytest=False) == "inconclusive" and classify(-9, "", pytest=False) == "inconclusive"


def test_the_lower_bound_is_exact_at_the_ends_and_below_the_rate_between():
    bound = detection_rate.lower_bound_95
    assert bound(0, 10) == 0.0 and bound(10, 10) == pytest.approx(0.05 ** 0.1)
    assert bound(1, 1) == pytest.approx(0.05)
    assert 0.0 < bound(7, 10) < 0.7 and bound(70, 100) > bound(7, 10)
    # at 7 of 10, P(X >= 7 | p) = 0.05 at p = 0.3934 (the Clopper-Pearson value)
    assert bound(7, 10) == pytest.approx(0.3934, abs=2e-4)
