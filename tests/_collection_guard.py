"""Refuse a run whose tests have lost a directory's conftest fixtures.

pytest 9.1.1 (the locked version) hands a directory's ``conftest.py`` fixtures to the first ``Dir`` node it collects for the directory and
matches a fixture to a test by node identity. Name the files of one directory on the command line, then a file of a
directory above it, then another file of the first (``pytest tests/saved/a.py tests/test_b.py tests/saved/c.py``): the
second visit to ``tests/saved`` makes a new node that no fixture belongs to, and ``c.py`` runs without the conftest's
fixtures. An autouse fixture is dropped without a word. ``tests/saved/conftest.py``'s ``x64`` is one, so
``test_registry_walk.py`` ran in float32 beside the files that run before it and failed 89 of its tests on drawings 3e-10
m off the surface: no state leaked between the tests, the fixture that sets the precision never ran.

A run with no path (``testpaths``, as CI runs it) walks the tree once and meets no such revisit; ``scripts/
stress_select.py`` orders its targets so that it meets none either. This hook is for a run by hand, in any order:
after collection it checks that every test beneath a directory with an autouse conftest fixture has that fixture, and
refuses the run, naming the fixture and the order that works, when one has not. ``tests/test_collection_guard.py`` is
its test, and the control that pytest still does this.
"""

from __future__ import annotations

from typing import Any

import sys

import pytest


def lost_autouse_fixtures(session: pytest.Session) -> dict[str, int]:
    """``{fixture and its conftest: tests beneath it that lack it}`` over the collected tests; empty when none do."""
    manager: Any = session._fixturemanager
    lost: dict[str, int] = {}
    for node, names in manager._node_autousenames.items():
        if not isinstance(node, pytest.Directory):
            continue
        for name in names:
            owned = [held for held in manager._arg2fixturedefs.get(name, ()) if held.node is node]
            for item in session.items:
                if not item.path.is_relative_to(node.path):
                    continue
                if name in getattr(getattr(item, "callspec", None), "params", {}):
                    continue  # a parameter of that name stands in for the fixture, as test_gyroscopic_energy's x64 does
                known: Any = item._fixtureinfo.name2fixturedefs.get(name, ())
                if not any(held in known for held in owned):
                    key = f"{name!r} of {node.path.relative_to(session.config.rootpath)}/conftest.py"
                    lost[key] = lost.get(key, 0) + 1
    return lost


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session: pytest.Session) -> None:
    lost = lost_autouse_fixtures(session)
    if lost:
        held = "; ".join(f"the autouse fixture {key} is missing from {count} test(s)" for key, count in lost.items())
        message = (f"{held}. pytest 9.1.1 gives a directory's conftest fixtures to the first node it makes for the "
                   "directory, and this run met the directory again after a file of a directory above it. Name the paths "
                   "in directory order (`sorted`), or name none (tests/_collection_guard.py).")
        print(f"ERROR: {message}", file=sys.stderr)  # an xdist worker's exception reaches the terminal as a bare crash
        raise pytest.UsageError(message)
