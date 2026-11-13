"""The tests marked ``fixture_first`` run on one worker, first: a fixture that costs minutes is paid once per job.

CI deals the collection to the pytest-xdist workers with ``--dist worksteal``: each worker gets the next quarter, in
order, and a worker that runs short steals the tail of the longest queue. ``xdist_group`` is ignored in that mode, so a
fixture or cache that costs minutes is built by every worker that runs one of the tests reading it, and the steals reach
all of them: the switch scan's traced configurations took 262 to 452 s on each of four workers of one shard (run
43079637575), and a shard was cancelled at its 30 minutes. Each test that reads such a build carries
``@pytest.mark.fixture_first``.

Putting the marked tests at the head of the collection (:func:`marked_first`) was not enough: the head of the first worker's
queue is reached by a thief once the thieves have been idle through a few steals (a steal takes the tail half of the longest
queue, and each steal halves it), and at 30 shards the switch scan's shard is short beside the build (about 150 to 340 s):
5 of the file's 11 marked tests paid it in one run (43239668452), and a shard-16 test in two runs (21ec3f5, da8d167) paid
3.5 and 4.7 minutes against a 2.9 minute ceiling. So the scheduler is one's own: :class:`PinnedWorkStealing` is xdist's
work stealing with the marked tests dealt to one worker (the first) and never offered to a thief; every other test is
dealt and stolen as before, and the pinned worker steals like the rest once its marked tests are done.

The controller process does not collect, so it cannot read a marker: each worker writes the node ids it found marked to a
file named by the run's unique id and its own (:func:`marked_record`), and the scheduler reads it when the collection is complete.
Every worker sorts the one collection the one way, as xdist requires.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from xdist.scheduler import WorkStealingScheduling
from xdist.scheduler.worksteal import MIN_PENDING

MARKER = "fixture_first"


def marked_first(items: list[pytest.Item]) -> None:
    items.sort(key=lambda item: item.get_closest_marker(MARKER) is None)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    marked_first(items)


#: The cached builds that cost minutes (``tests/saved/switch_configurations.py``): every configuration's step traced, the
#: graphs it built, and the scan of the traces. A test that calls one, or reads a fixture that does, is marked
#: ``fixture_first``.
BUILDS = ("tree_scan", "traced_steps", "built_graphs")


def calls_a_build(function: Any, builds: Iterable[str]) -> bool:
    """Whether ``function``'s own code names one of ``builds`` (a call through another function is not seen)."""
    code = getattr(function, "__code__", None)
    return code is not None and bool(set(builds) & set(code.co_names))


def unmarked_readers(items: Iterable[pytest.Item], builds: Iterable[str] = BUILDS) -> list[str]:
    """The tests that call one of ``builds`` or read a fixture that does, and are not marked ``fixture_first``."""
    wanted = tuple(builds)
    found = []
    for item in items:
        if item.get_closest_marker(MARKER) is not None:
            continue
        fixtures = (definition for definitions in getattr(getattr(item, "_fixtureinfo", None), "name2fixturedefs", {}).values()
                    for definition in definitions)
        if calls_a_build(getattr(item, "function", None), wanted) or any(calls_a_build(d.func, wanted) for d in fixtures):
            found.append(item.nodeid)
    return found


@pytest.hookimpl
def pytest_collection_finish(session: pytest.Session) -> None:
    """Refuse a run with a test that reads a build that costs minutes without the mark that keeps it on one worker; and, on an
    xdist worker, record the node ids found marked for the controller (every worker writes the same list, atomically)."""
    unmarked = unmarked_readers(session.items)
    if unmarked:
        raise pytest.UsageError(f"these tests read a build that costs minutes without @pytest.mark.{MARKER} "
                                f"(tests/_fixture_first.py): {unmarked}")
    worker: dict[str, Any] | None = getattr(session.config, "workerinput", None)  # absent in a child pytest of a worker's test
    if worker is None:
        return
    marked = sorted(item.nodeid for item in session.items if item.get_closest_marker(MARKER) is not None)
    marked_record(worker["testrunuid"], worker["workerid"]).write_text(json.dumps(marked), encoding="utf-8")


def marked_record(run: str, worker: str) -> Path:
    """The file in which ``worker`` of the xdist run ``run`` records the marked node ids, for the controller (one file for each
    worker: none is ever replaced while another process reads it, which Windows refuses)."""
    return Path(tempfile.gettempdir()) / f"mime-fixture-first-{run}-{worker}.json"


class PinnedWorkStealing(WorkStealingScheduling):
    """xdist's work stealing with the marked tests dealt to the first worker and never stolen from it.

    ``pinned`` holds the collection indices of the marked tests. They go to the home worker in collection order,
    before anything is dealt; a thief is offered only the unpinned tail of a queue; and a worker with only pinned
    tests left counts as busy, as it is. The home worker is dealt, and steals, like the rest once it runs short.
    A worker that dies hands its tests back to the common pool, pinned ones included: those are then dealt
    to whoever is idle, and the fixture is paid again, which is the cost of a crash and nothing else.
    """

    def __init__(self, config: pytest.Config, log: Any = None) -> None:
        super().__init__(config, log)
        self.pinned: frozenset[int] = frozenset()
        self.placed = False

    def marked_names(self) -> set[str]:
        names: set[str] = set()
        for node in self.node2pending:
            record = marked_record(node.workerinput["testrunuid"], node.workerinput["workerid"])
            try:
                names |= set(json.loads(record.read_text(encoding="utf-8")))
            except FileNotFoundError:
                pass
            record.unlink(missing_ok=True)
        return names

    def place(self) -> None:
        """Deal the marked tests to the home worker, once, as the first act of the first scheduling."""
        self.placed = True
        assert self.collection is not None
        names = self.marked_names()
        self.pinned = frozenset(index for index, name in enumerate(self.collection) if name in names)
        mine = [index for index in self.pending if index in self.pinned]
        if not mine:
            return
        home = next(iter(self.node2pending))
        self.pending[:] = [index for index in self.pending if index not in self.pinned]
        self.node2pending[home].extend(mine)
        home.send_runtest_some(mine)

    def check_schedule(self) -> None:
        """xdist's ``WorkStealingScheduling.check_schedule``, with the pinned tests left out of what a thief is offered."""
        if not self.placed and self.collection is not None:
            self.place()
        queues = [(node, pending) for node, pending in self.node2pending.items() if not node.shutting_down]

        def idle_nodes() -> list[Any]:
            return [node for node, pending in queues if len(pending) < MIN_PENDING]

        idle = idle_nodes()
        if not idle:
            return
        if self.pending:
            for position, node in enumerate(idle):
                self._send_tests(node, len(self.pending) // (len(idle) - position))
            idle = idle_nodes()
            if not idle:
                return
        if self.steal_requested_from_node is not None:
            return
        stealable = [(node, pending, [index for index in pending if index not in self.pinned]) for node, pending in queues]
        steal_from = max(stealable, key=lambda queue: len(queue[2]), default=None)
        count = 0
        if steal_from is not None:
            node, pending, tail = steal_from
            count = min(len(tail) // 2, max(0, len(pending) - MIN_PENDING))
        if count == 0:
            for node in idle:
                node.shutdown()
            return
        assert steal_from is not None
        steal_from[0].send_steal(steal_from[2][-count:])
        self.steal_requested_from_node = steal_from[0]


class Scheduling:
    """Registered by the conftest when xdist is present: ``--dist worksteal`` gets :class:`PinnedWorkStealing`."""

    @pytest.hookimpl
    def pytest_xdist_make_scheduler(self, config: pytest.Config, log: Any) -> PinnedWorkStealing | None:
        return PinnedWorkStealing(config, log) if config.getvalue("dist") == "worksteal" else None
