"""The ``fixture_first`` order and scheduler (``tests/_fixture_first.py``): marked tests first, each group in its own order,
and on one worker, never stolen from it."""

import json
import os
import random
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import _fixture_first as order
import pytest
from xdist.scheduler import WorkStealingScheduling


def item(nodeid: str, marked: bool = False, fixtures: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(nodeid=nodeid, fixturenames=fixtures,
                           get_closest_marker=lambda name: object() if marked and name == order.MARKER else None)


def test_marked_tests_come_first_and_each_group_keeps_its_order():
    items = [item("a"), item("b", True), item("c"), item("d", True), item("e", True), item("f")]
    order.marked_first(items)  # type: ignore[arg-type]
    assert [i.nodeid for i in items] == ["b", "d", "e", "a", "c", "f"]


def test_control_a_run_with_nothing_marked_keeps_the_collection_as_it_is():
    items = [item(name) for name in "cabd"]
    order.marked_first(items)  # type: ignore[arg-type]
    assert [i.nodeid for i in items] == list("cabd")


def reader(nodeid: str, marked: bool = False, body: Any = None, fixture: Any = None) -> SimpleNamespace:
    """A collected test: the function it runs and, if it reads one, the fixture function."""
    info = SimpleNamespace(name2fixturedefs={"f": [SimpleNamespace(func=fixture)]} if fixture else {})
    return SimpleNamespace(nodeid=nodeid, function=body, _fixtureinfo=info,
                           get_closest_marker=lambda name: object() if marked and name == order.MARKER else None)


def test_a_test_calling_a_build_or_reading_a_fixture_that_does_is_named_unless_marked():
    def plain(): return 1
    def direct(): return tree_scan()  # noqa: F821
    def through_fixture(): return traced_steps()  # noqa: F821
    items = [reader("a", body=direct), reader("b", body=plain, fixture=through_fixture), reader("c", body=plain),
             reader("d", True, body=direct), reader("e", True, body=plain, fixture=through_fixture)]
    assert order.unmarked_readers(items) == ["a", "b"]  # type: ignore[arg-type]
    assert order.unmarked_readers(items, ["other"]) == []  # type: ignore[arg-type]


_READER = """
import pytest

def tree_scan():
    return 1

{mark}
def test_reads_the_build():
    assert tree_scan() == 1
"""


@pytest.mark.parametrize(("mark", "refused"), [("", True), ("@pytest.mark.fixture_first", False)])
def test_a_run_whose_test_calls_a_build_without_the_mark_is_refused_at_collection(tmp_path, mark, refused):
    """Control: the unmarked reader fails the run before any test runs (exit 4, a usage error naming the test), the marked
    one runs."""
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "_fixture_first.py").write_bytes((repo / "tests" / "_fixture_first.py").read_bytes())
    (tmp_path / "conftest.py").write_text(_CONFTEST)
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = fixture_first\n")
    (tmp_path / "test_reader.py").write_text(_READER.format(mark=mark))
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120)
    assert (proc.returncode == 4 and "test_reader.py::test_reads_the_build" in proc.stderr) if refused else proc.returncode == 0, \
        proc.stdout + proc.stderr


class Worker:
    """What the scheduler asks of a node: a queue it was sent, a steal request it answers, a shutdown."""

    def __init__(self, name: str, run: str) -> None:
        self.shutting_down = False
        self.workerinput = {"testrunuid": run, "workerid": name}
        self.gateway = SimpleNamespace(id=name)
        self.name = name
        self.asked: list[int] = []

    def send_runtest_some(self, indices: list[int]) -> None:
        pass

    def send_steal(self, indices: list[int]) -> None:
        self.asked = list(indices)

    def shutdown(self) -> None:
        self.shutting_down = True


#: What a worker pays, once, before its first marked test: the build, in the unit of a test's cost (a test about a second).
BUILD = 150.0


def deal(scheduler_class: type[WorkStealingScheduling], marked: set[int], costs: list[float], workers: int) -> dict[int, str]:
    """Which worker ran each test, when ``workers`` workers run ``costs`` (a test's seconds, by index) under
    ``scheduler_class``: the controller's calls, in the order a run would make them, with the workers' answers (a
    steal gives back what the worker has not started) and a clock for each worker. A worker's first marked test costs
    :data:`BUILD` more."""
    run = uuid.uuid4().hex
    names = [f"t{i}" for i in range(len(costs))]
    nodes = [Worker(f"gw{i}", run) for i in range(workers)]
    for node in nodes:
        order.marked_record(run, node.name).write_text(json.dumps([names[i] for i in sorted(marked)]), encoding="utf-8")
    scheduler = scheduler_class(SimpleNamespace(getvalue=lambda name: [f"{workers}*popen"]))  # type: ignore[arg-type]
    for node in nodes:
        scheduler.add_node(node)  # type: ignore[arg-type]
        scheduler.add_node_collection(node, names)  # type: ignore[arg-type]
    scheduler.schedule()
    clock = {node: 0.0 for node in nodes}
    ran: dict[int, str] = {}
    built: set[Worker] = set()
    working: set[Worker] = {node for node in nodes if scheduler.node2pending[node]}  # type: ignore[index]

    def wake(at: float) -> None:
        """A worker handed tests after it ran out starts them when they were handed, not when it stopped."""
        for node in nodes:
            queued = bool(scheduler.node2pending[node])  # type: ignore[index]
            if queued and node not in working:
                clock[node] = max(clock[node], at)
            working.add(node) if queued else working.discard(node)

    while working:
        asked = scheduler.steal_requested_from_node
        if asked is not None:
            held = scheduler.node2pending[asked]  # type: ignore[index]
            scheduler.remove_pending_tests_from_node(asked, [i for i in asked.asked if i in held[1:]])  # type: ignore[attr-defined]
            wake(clock[asked])  # type: ignore[index]
        node = min(working, key=lambda n: clock[n])
        index = scheduler.node2pending[node][0]  # type: ignore[index]
        paid = BUILD if index in marked and node not in built else 0.0
        built.add(node) if paid else None
        clock[node] += costs[index] + paid
        ran[index] = node.name
        scheduler.mark_test_complete(node, index)  # type: ignore[arg-type]
        wake(clock[node])
    return ran


def a_run(seed: int) -> tuple[set[int], list[float]]:
    """The marked tests at the head of the collection (``marked_first``), each a read of the build, and the rest of the shard 20 to 600 tests."""
    rng = random.Random(seed)
    marked_count = rng.randint(2, 11)
    others = rng.randint(20, 600)
    costs = [rng.uniform(0.5, 2.0) for _ in range(marked_count)] + [rng.uniform(0.05, 1.0) for _ in range(others)]
    return set(range(marked_count)), costs


DRAWS = 300


def test_the_marked_tests_all_run_on_one_worker_under_the_pinned_scheduler():
    """300 shards of 22 to 611 tests on 2 to 6 workers, the same marked block at the head of each."""
    scattered = []
    for seed in range(DRAWS):
        marked, costs = a_run(seed)
        ran = deal(order.PinnedWorkStealing, marked, costs, 2 + seed % 5)
        assert sorted(ran) == list(range(len(costs))), seed
        if len({ran[i] for i in marked}) != 1:
            scattered.append(seed)
    assert scattered == []


def test_control_the_stock_scheduler_scatters_the_marked_tests_over_workers_in_most_of_those_shards():
    """The same 300 shards under xdist's own work stealing, which is what CI ran: a steal reaches the block at the head."""
    scattered = 0
    for seed in range(DRAWS):
        marked, costs = a_run(seed)
        ran = deal(WorkStealingScheduling, marked, costs, 2 + seed % 5)
        scattered += len({ran[i] for i in marked}) > 1
    assert scattered >= DRAWS // 4, scattered


def test_every_other_test_still_runs_on_a_worker_that_is_free_when_the_pinned_worker_is_busy():
    """The pinned worker is not given the shard's other tests while it works through the marked ones, and joins in after."""
    marked, costs = a_run(7)
    ran = deal(order.PinnedWorkStealing, marked, costs, 4)
    home = {ran[i] for i in marked}
    assert home == {"gw0"}
    others = [ran[i] for i in range(len(costs)) if i not in marked]
    assert len(set(others) - home) == 3 and others.count("gw0") < len(others) / 4 + len(marked)


def test_a_run_with_nothing_marked_is_dealt_as_xdist_deals_it():
    for seed in range(20):
        _, costs = a_run(seed)
        assert deal(order.PinnedWorkStealing, set(), costs, 4) == deal(WorkStealingScheduling, set(), costs, 4)


_PROJECT = """
import os, time
import pytest

DONE = os.environ["DONE"]

@pytest.fixture(scope="module")
def build():
    # The build starts once 90 of the 120 other tests are done: the three workers that are not dealt the marked tests
    # have then finished their own quarters (90 tests) and are idle, so a scheduler that lets them steal does, during
    # the one second the build takes. Nothing here depends on how fast a worker starts.
    deadline = time.time() + 120
    while time.time() < deadline:
        if os.path.exists(DONE) and len(open(DONE).read().split()) >= 90:
            break
        time.sleep(0.02)
    else:
        raise TimeoutError("the other tests did not get to 90")
    time.sleep(1)
    with open(os.environ["BUILDS"], "a") as log:
        log.write(os.environ["PYTEST_XDIST_WORKER"] + "\\n")

@pytest.mark.fixture_first
@pytest.mark.parametrize("n", range(6))
def test_reader(build, n):
    pass

@pytest.mark.parametrize("n", range(120))
def test_other(n):
    with open(DONE, "a") as log:
        log.write("x\\n")
"""

_CONFTEST = """
import os
import _fixture_first
import pytest

def pytest_configure(config):
    config.pluginmanager.register(_fixture_first, "ff")
    if os.environ.get("PINNED") == "1":
        config.pluginmanager.register(_fixture_first.Scheduling(), "ffs")
"""


def builds_under(tmp_path: Path, pinned: bool, repo: Path) -> int:
    """The workers that built the project's fixture in a real ``-n 4 --dist worksteal`` run."""
    project = tmp_path / ("pinned" if pinned else "stock")
    project.mkdir()
    (project / "_fixture_first.py").write_bytes((repo / "tests" / "_fixture_first.py").read_bytes())
    (project / "conftest.py").write_text(_CONFTEST)
    (project / "pytest.ini").write_text("[pytest]\nmarkers = fixture_first\n")
    (project / "test_project.py").write_text(_PROJECT)
    builds = project / "builds.log"
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    env.update(BUILDS=str(builds), DONE=str(project / "done.log"), PINNED="1" if pinned else "0")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-n", "4", "--dist", "worksteal"],
                          cwd=project, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return len(set(builds.read_text().split()))


def test_a_real_xdist_run_builds_the_fixture_on_one_worker_where_the_stock_scheduler_builds_it_on_several(tmp_path):
    """Four workers, six tests behind a fixture at the head and 120 others after them; the fixture starts when 90 of the others
    are done (the three workers that have no marked test have finished their quarters). Run on this repository's pytest-xdist:
    the fixture's builds are counted in a file, one line a build. The stock scheduler's count is the control: the idle workers
    steal the marked tests while the first is in its build."""
    repo = Path(__file__).resolve().parents[1]
    assert builds_under(tmp_path, True, repo) == 1
    assert builds_under(tmp_path, False, repo) > 1
