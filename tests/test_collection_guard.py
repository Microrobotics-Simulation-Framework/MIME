"""The collection guard (``tests/_collection_guard.py``): a run that names the files of one directory, a file of the
directory above, then another file of the first, meets pytest's loss of that directory's conftest fixtures. That is how
``tests/saved/test_registry_walk.py`` ran in float32 and failed 89 tests beside the five files before it: the autouse
``x64`` fixture of ``tests/saved/conftest.py`` was never applied.

A small project in a temporary directory is run in a pytest of its own (the guard is loaded from ``tests/`` as a
plugin). The control is the unguarded run: it must lose the fixture, or the guard guards against nothing; when pytest is
upgraded to a version that no longer does, that assertion fails and the guard comes out.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
CONFTEST = '''import os

import pytest


@pytest.fixture(autouse=True, scope="module")
def precision():
    os.environ["MINI_PRECISION"] = "float64"
    yield
    del os.environ["MINI_PRECISION"]
'''
APPLIED = 'import os\n\n\ndef test_{name}():\n    assert os.environ.get("MINI_PRECISION") == "float64"\n'
PARAMETRIZED = ('import pytest\n\n\n@pytest.mark.parametrize("precision", ["float32", "float64"])\n'
                'def test_a_parameter_named_for_the_fixture_stands_in_for_it(precision):\n    assert precision\n')
HAZARD = ("below/test_first.py", "test_top.py", "below/test_second.py")
IN_ORDER = ("test_top.py", "below/test_first.py", "below/test_second.py")


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("guarded")
    (root / "below").mkdir()
    (root / "below" / "conftest.py").write_text(CONFTEST)
    (root / "below" / "test_first.py").write_text(APPLIED.format(name="first"))
    (root / "below" / "test_second.py").write_text(APPLIED.format(name="second"))
    (root / "below" / "test_parametrized.py").write_text(PARAMETRIZED)
    (root / "test_top.py").write_text("def test_top():\n    pass\n")
    return root


def pytest_in(project: Path, paths: tuple[str, ...], guarded: bool) -> subprocess.CompletedProcess[str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("PYTEST_", "MINI_"))}
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *(["-p", "_collection_guard"] if guarded else []),
               *paths]
    return subprocess.run(command, cwd=project, capture_output=True, text=True, check=False,
                          env={**environment, "PYTHONPATH": str(TESTS)})


def test_the_unguarded_run_of_the_hazard_order_loses_the_fixture(project):
    done = pytest_in(project, HAZARD, guarded=False)
    assert "FAILED below/test_second.py::test_second" in done.stdout, (
        "this pytest keeps a revisited directory's conftest fixtures, so tests/_collection_guard.py guards against "
        "nothing: delete it\n" + done.stdout)
    assert "1 failed, 2 passed" in done.stdout


def test_the_guard_refuses_the_hazard_order_by_name_and_runs_nothing(project):
    done = pytest_in(project, HAZARD, guarded=True)
    assert done.returncode == 4
    assert "the autouse fixture 'precision' of below/conftest.py is missing from 1 test(s)" in done.stderr + done.stdout
    assert "passed" not in done.stdout and "failed" not in done.stdout


@pytest.mark.parametrize("paths", [IN_ORDER, tuple(sorted(HAZARD)), ("below/test_parametrized.py", "test_top.py")])
def test_the_guard_lets_a_run_in_directory_order_and_a_parameter_named_for_the_fixture_through(project, paths):
    done = pytest_in(project, paths, guarded=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "passed" in done.stdout


def test_this_suites_own_run_is_guarded_the_hazard_order_over_tests_saved_is_refused_and_the_same_files_in_order_are_not():
    """The wiring (``tests/conftest.py``), on the order that failed 89 tests: the files are small, and only collected."""
    hazard = ("tests/saved/test_schema.py", "tests/test_collection.py", "tests/saved/test_load.py")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}

    def collect(paths: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *paths],
                              cwd=TESTS.parent, capture_output=True, text=True, check=False, env=environment)

    refused = collect(hazard)
    assert refused.returncode == 4, refused.stdout + refused.stderr
    assert "the autouse fixture 'x64' of tests/saved/conftest.py is missing from" in refused.stdout + refused.stderr
    assert collect(tuple(sorted(hazard))).returncode == 0
