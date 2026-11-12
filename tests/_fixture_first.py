"""The tests marked ``fixture_first`` run first: a module fixture that costs minutes is paid by one worker.

CI deals the collection to the pytest-xdist workers with ``--dist worksteal``: each worker gets the next quarter, in
order, and a worker that runs short steals the tail of the longest queue. ``xdist_group`` is ignored in that mode, so a
module fixture that costs minutes is built by every worker that runs one of the tests reading it, and the steals reach
all of them: the switch scan's traced configurations took 262 to 452 s on each of four workers of one shard (run
43079637575), and a shard was cancelled at its 30 minutes. Each such test carries ``@pytest.mark.fixture_first``; they
go to the head of the collection, in their order, the rest after them in theirs. The head of the first worker's queue is
not stolen while that worker is working through it, so the first marked test pays and the others find the fixture built.
Every worker sorts the one collection the one way, as xdist requires. The one test that still pays is the first marked
test of its file, so its row in ``tests/duration_allowlist.txt`` is the one that stays, not whichever test xdist's order
gave a worker first.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

MARKER = "fixture_first"


def marked_first(items: list[pytest.Item]) -> None:
    items.sort(key=lambda item: item.get_closest_marker(MARKER) is None)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    marked_first(items)


def unmarked_readers(items: Iterable[pytest.Item], module: str, fixtures: Iterable[str]) -> list[str]:
    """The tests of ``module`` that read one of ``fixtures`` and are not marked ``fixture_first``."""
    wanted = set(fixtures)
    return [item.nodeid for item in items if item.nodeid.split("::")[0] == module
            and wanted & set(getattr(item, "fixturenames", ())) and item.get_closest_marker(MARKER) is None]
