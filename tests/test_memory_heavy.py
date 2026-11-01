"""The memory-heavy tests are marked, and the marked tests take turns across pytest-xdist's workers.

CI's runner has 16 GB and runs four workers. A test that peaks at several GB (``tests/memory_peaks.json`` lists what
was measured, with the date and the line) must not run beside another: the runner of a pull request was lost once,
with the shard's tests at 95 %, while a 5.9 GB test and its neighbours were running. ``@pytest.mark.memory_heavy``
makes a test hold the run's lock from its setup to its teardown (``tests/conftest.py``).

**What this guards, and what it does not.** It fails a test the file lists over the line that is not marked, a marked
test the file does not list, and a listed test that does not exist (a renamed one would otherwise leave its entry
guarding nothing). It cannot know a test nobody measured: a new test that peaks over the line enters the file when
someone measures it, and ``scripts/measure_test_memory.py`` does that for a whole run."""

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
PEAKS = json.loads((TESTS / "memory_peaks.json").read_text(encoding="utf-8"))
LINE_MIB = PEAKS["line_mib"]
MARKER = "memory_heavy"


def marks_heavy(node: ast.expr) -> bool:
    """Whether a decorator or a ``pytestmark`` value names ``pytest.mark.memory_heavy`` (a list or tuple of marks too)."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return any(marks_heavy(element) for element in node.elts)
    if isinstance(node, ast.Call):
        node = node.func
    return isinstance(node, ast.Attribute) and node.attr == MARKER


def functions(tree: ast.Module, path: str) -> dict[str, bool]:
    """``{test id without its parameters: whether it is marked}`` for the test functions of a module, a mark on its
    class or on the module counting for each of its tests."""
    module_marked = any(marks_heavy(statement.value) for statement in tree.body
                        if isinstance(statement, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == "pytestmark" for target in statement.targets))
    found: dict[str, bool] = {}

    def visit(body: list[ast.stmt], prefix: str, inherited: bool) -> None:
        for statement in body:
            if isinstance(statement, ast.ClassDef) and statement.name.startswith("Test"):
                visit(statement.body, f"{prefix}{statement.name}::",
                      inherited or any(marks_heavy(d) for d in statement.decorator_list))
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name.startswith("test"):
                found[f"{path}::{prefix}{statement.name}"] = inherited or any(
                    marks_heavy(d) for d in statement.decorator_list)

    visit(tree.body, "", module_marked)
    return found


def every_test() -> dict[str, bool]:
    found: dict[str, bool] = {}
    for file in sorted(TESTS.rglob("test_*.py")):
        found.update(functions(ast.parse(file.read_text(encoding="utf-8")), f"tests/{file.relative_to(TESTS).as_posix()}"))
    return found


def test_the_peaks_file_states_its_line_its_basis_and_its_date():
    assert PEAKS["measured"] and PEAKS["basis"] and isinstance(LINE_MIB, int) and LINE_MIB > 0
    assert PEAKS["peaks_mib"], "an empty list guards nothing"


def test_every_listed_test_exists():
    tests = every_test()
    gone = sorted(set(PEAKS["peaks_mib"]) - set(tests))
    assert not gone, f"tests/memory_peaks.json lists tests that do not exist (renamed or deleted?): {gone}"


def test_every_listed_test_over_the_line_is_marked_memory_heavy():
    tests = every_test()
    unmarked = sorted(name for name, peak in PEAKS["peaks_mib"].items() if peak > LINE_MIB and not tests.get(name))
    assert not unmarked, (f"these tests peak over {LINE_MIB} MiB (tests/memory_peaks.json) and are not marked "
                          f"@pytest.mark.memory_heavy, so xdist can run them beside each other: {unmarked}")


def test_every_marked_test_is_listed_over_the_line():
    tests = every_test()
    stray = sorted(name for name, marked in tests.items()
                   if marked and PEAKS["peaks_mib"].get(name, 0) <= LINE_MIB)
    assert not stray, (f"these tests are marked memory_heavy but tests/memory_peaks.json does not list them over "
                       f"{LINE_MIB} MiB (measure them, or take the mark off): {stray}")


def test_the_guard_sees_a_planted_unmarked_test_and_a_planted_marked_one():
    """The control: the reading of the marks must tell the three forms apart, or the guard above passes on anything."""
    source = textwrap.dedent('''
        import pytest
        def test_plain(): ...
        @pytest.mark.memory_heavy
        def test_decorated(): ...
        @pytest.mark.parametrize("x", [1])
        def test_other_mark(x): ...
        @pytest.mark.memory_heavy
        class TestGroup:
            def test_inside(self): ...
        ''')
    assert functions(ast.parse(source), "t.py") == {
        "t.py::test_plain": False, "t.py::test_decorated": True, "t.py::test_other_mark": False,
        "t.py::TestGroup::test_inside": True}
    assert functions(ast.parse("import pytest\npytestmark = [pytest.mark.memory_heavy]\ndef test_a(): ..."), "m.py") \
        == {"m.py::test_a": True}


SUITE = '''
import json, os, time
import pytest

@pytest.mark.parametrize("n", range(6))
{decorator}
def test_holds_the_floor(n):
    start = time.time()
    time.sleep(0.6)
    with open(os.environ["MIME_INTERVALS"], "a") as f:
        f.write(json.dumps([start, time.time()]) + "\\n")
'''

CONFTEST = '''
import importlib.util
spec = importlib.util.spec_from_file_location("mime_tests_conftest", {real!r})
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)
pytest_runtest_protocol = real.pytest_runtest_protocol
def pytest_configure(config):
    config.addinivalue_line("markers", "memory_heavy: the mark under test")
'''


def overlapping(tmp_path: Path, decorator: str) -> int:
    tmp_path.mkdir()
    """How many pairs of the six tests' intervals overlap, in a two-worker run of a suite whose tests carry ``decorator``."""
    (tmp_path / "conftest.py").write_text(CONFTEST.format(real=str(TESTS / "conftest.py")), encoding="utf-8")
    (tmp_path / "test_suite.py").write_text(SUITE.format(decorator=decorator), encoding="utf-8")
    intervals = tmp_path / "intervals.jsonl"
    environment = {"PATH": "", "MIME_INTERVALS": str(intervals), "HOME": str(tmp_path)}
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-n", "2", "--dist", "worksteal",
                           str(tmp_path / "test_suite.py")], cwd=tmp_path, capture_output=True, text=True,
                          env=environment, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    spans = [json.loads(line) for line in intervals.read_text(encoding="utf-8").splitlines()]
    assert len(spans) == 6
    return sum(1 for i, a in enumerate(spans) for b in spans[i + 1:] if a[0] < b[1] and b[0] < a[1])


def test_marked_tests_never_overlap_across_xdist_workers_and_unmarked_ones_do(tmp_path: Path):
    """The lock holds under ``--dist worksteal``; and the control, the same suite unmarked on the same two workers,
    does overlap, so that a run in which nothing overlaps anyway cannot pass this."""
    assert overlapping(tmp_path / "plain", "") > 0, "the control: unmarked tests did not overlap on two workers"
    assert overlapping(tmp_path / "marked", "@pytest.mark.memory_heavy") == 0
