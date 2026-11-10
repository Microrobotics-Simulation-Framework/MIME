"""``.github/probe/bit_compare.py`` compares arrays between a base ref and this checkout, bit for bit, in one job.

Its parts are read here without the base's own ``uv sync`` (``prepare_base`` is replaced by a directory named ``base``): the
comparison of two NPZ files, the pinning of both runs to one CPU with one BLAS thread (the command writes what it saw, and
the two must agree), and the report. Controls: a one-ulp difference, a dtype change with equal values, a missing array and
a failing command each turn the report's ``all_equal`` false, named, every run."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("mime_probe_bit_compare", ROOT / ".github" / "probe" / "bit_compare.py")
assert _SPEC is not None and _SPEC.loader is not None
bit_compare = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bit_compare
_SPEC.loader.exec_module(bit_compare)

pytestmark = pytest.mark.skipif(shutil.which("taskset") is None or shutil.which("git") is None,
                                reason="needs taskset and git (CI's runners have both)")


def npz(path: Path, **arrays: np.ndarray) -> np.lib.npyio.NpzFile:
    np.savez(path, **arrays)
    return np.load(path, allow_pickle=False)


def test_equal_arrays_are_equal_and_each_difference_is_named(tmp_path):
    ulp = np.nextafter(np.float64(1.0), np.float64(2.0))
    base = npz(tmp_path / "b.npz", same=np.arange(4.0), ulp=np.array([1.0]), kind=np.array([1.0], np.float32),
               nan=np.array([np.nan]), gone=np.zeros(2), shape=np.zeros(3))
    head = npz(tmp_path / "h.npz", same=np.arange(4.0), ulp=np.array([ulp]), kind=np.array([1.0], np.float64),
               nan=np.array([np.nan]), new=np.zeros(2), shape=np.zeros((3, 1)))
    arrays = bit_compare.compare_arrays(base, head)
    assert arrays["same"]["equal"] and arrays["same"]["max_abs_difference"] == 0.0
    assert not arrays["ulp"]["equal"] and arrays["ulp"]["max_abs_difference"] == float(ulp - 1.0)
    assert arrays["kind"]["values_equal"] and not arrays["kind"]["equal"]  # the same value, another dtype
    assert not arrays["nan"]["equal"] and arrays["nan"]["equal_nan"]  # array_equal's NaN, and the NaN-matching reading
    assert arrays["gone"] == {"equal": False, "only_in": "base"} and arrays["new"] == {"equal": False, "only_in": "head"}
    assert not arrays["shape"]["equal"]


DUMP = """
import os, sys
from pathlib import Path
import numpy as np
out, tree, program = sys.argv[1:4]
is_base = Path(tree).name == "base"
if len(os.sched_getaffinity(0)) != 1 or os.environ["OPENBLAS_NUM_THREADS"] != "1" or "intra_op_parallelism_threads=1" not in os.environ["XLA_FLAGS"]:
    sys.exit(4)  # the command itself checks it was pinned to one CPU with one BLAS thread: a failed command is reported
x = np.full(3, 1.0)
if program == "ulp" and is_base:
    x = np.nextafter(x, 2.0)
arrays = {"x": x, "cpus": np.array([len(os.sched_getaffinity(0))]), "blas": np.array([int(os.environ["OPENBLAS_NUM_THREADS"])])}
if program == "dtype" and is_base:
    arrays["x"] = x.astype(np.float32)
if program == "missing" and is_base:
    arrays["extra"] = x
if program == "fails" and is_base:
    sys.exit(3)
np.savez(out, **arrays)
"""


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    head = tmp_path / "head"
    head.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], cwd=head, check=True)
    (head / "f").write_text("x", encoding="utf-8")
    subprocess.run([*git, "add", "f"], cwd=head, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "c"], cwd=head, check=True)
    (tmp_path / "dump.py").write_text(DUMP, encoding="utf-8")

    def base_tree(root, commit, where):  # a tree named `base` in place of the worktree and its own `uv sync`
        (where / "base").mkdir()
        return where / "base"
    monkeypatch.setattr(bit_compare, "prepare_base", base_tree)
    return head


def compare(checkout, tmp_path, programs, *extra):
    command = f"{sys.executable} {tmp_path / 'dump.py'} {{out}} {{tree}} {{program}}"
    output = tmp_path / "report.json"
    status = bit_compare.main(["--base", "HEAD", "--head", str(checkout), "--programs", *programs, "--command", command,
                               "--output", str(output), *extra])
    return status, json.loads(output.read_text(encoding="utf-8"))


def test_both_sides_run_on_one_cpu_with_one_blas_thread_and_equal_arrays_pass(tmp_path, checkout):
    status, report = compare(checkout, tmp_path, ["plain"], "--strict")
    program = report["programs"]["plain"]
    assert status == 0 and report["all_equal"] and program["all_equal"]
    # Each command exits 4 unless it saw one CPU, one BLAS thread and XLA's single-thread flag, so a pass here is both.
    assert report["base_commit"] == report["head_commit"] and len(report["cpus"]) >= 1
    assert program["arrays"]["cpus"]["equal"] and program["arrays"]["blas"]["equal"]


@pytest.mark.parametrize("program, named", [("ulp", "x"), ("dtype", "x"), ("missing", "extra")])
def test_control_a_planted_difference_is_reported_naming_the_array_and_fails_strict(tmp_path, checkout, program, named):
    status, report = compare(checkout, tmp_path, ["plain", program], "--strict")
    assert status == 1 and not report["all_equal"]
    assert report["programs"]["plain"]["all_equal"]
    assert report["programs"][program]["arrays"][named]["equal"] is False
    assert [name for name, entry in report["programs"][program]["arrays"].items() if not entry["equal"]] == [named]


def test_control_a_failing_command_is_reported_for_its_program_alone(tmp_path, checkout):
    status, report = compare(checkout, tmp_path, ["plain", "fails"], "--strict")
    assert status == 1 and report["programs"]["fails"]["status"] == "command failed"
    assert list(report["programs"]["fails"]["failures"]) == ["base"] and report["programs"]["plain"]["all_equal"]


def test_the_cpu_set_is_the_one_asked_for_on_both_sides(tmp_path, checkout):
    available = sorted(os.sched_getaffinity(0))
    status, report = compare(checkout, tmp_path, ["plain"], "--cpus", str(available[0]))
    assert status == 0 and report["cpus"] == str(available[0]) and report["all_equal"]
